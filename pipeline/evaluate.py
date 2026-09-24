"""
Evaluation harness for the 10-Q RAG pipeline.

Runs each question in a QnA csv through RAG_pipeline.ask(), judges the answer
against the ground-truth 'Answer' column with an LLM judge, and reports
accuracy by Question Type and Source Chunk Type, plus a RETRIEVAL HIT rate
(did the retrieved chunks include every filing listed in 'Source Docs'?).
That second number tells you whether a failure is retrieval or reading.

Usage (keys in .env: GROQ_API_KEY):
    python evaluate.py ingest /path/to/*.pdf        # one-time / after chunking changes
    python evaluate.py run qna_data_mini.csv        # quick smoke test
    python evaluate.py run qna_data.csv             # full run (resumable)
    python evaluate.py run qna_data.csv 30          # a fixed random sample of 30 questions
    python evaluate.py summary                      # re-print stats from eval_results.csv

Resumable: results are appended to eval_results.csv after every question and
already-answered questions are skipped, so you can Ctrl+C and rerun freely.
Delete eval_results.csv to start over.

Optional env vars:
    LLM_MODEL     Groq model for answering (set in .env), e.g. LLM_MODEL=llama-3.1-8b-instant
    JUDGE_PROVIDER provider for the judge (default: same as LLM_PROVIDER), e.g. groq
    JUDGE_MODEL   model used as judge (default: same as answering model)
    SLEEP_BETWEEN seconds between questions (default 45; 8K-TPM accounts need ~40s+)
"""

import os
import re
import sys
import time
import pandas as pd
from dotenv import load_dotenv

from RAG_pipeline import ingest, ask, retrieve, LLM_MODEL, PROVIDER, DEFAULT_MODELS, get_llm_client, CALL_COUNTS

load_dotenv()
RESULTS_PATH = "eval_results.csv"
JUDGE_PROVIDER = os.environ.get("JUDGE_PROVIDER", PROVIDER).lower()   # can differ from the answering provider
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", LLM_MODEL if JUDGE_PROVIDER == PROVIDER else DEFAULT_MODELS[JUDGE_PROVIDER])
SLEEP_BETWEEN = float(os.environ.get("SLEEP_BETWEEN", 45 if PROVIDER == "groq" else 13))
ALL_PERIODS = [("Q3", "2022"), ("Q1", "2023"), ("Q2", "2023"), ("Q3", "2023")]


# --------------------------------------------------------------------------
# JUDGING
# --------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are a strict grader of a RAG system's answer against a reference answer \
for a question about SEC 10-Q filings.

PASS only if the GENERATED answer contains the key facts of the REFERENCE answer:
- Every specific figure in the reference (dollar amounts, percentages, per-period values) must \
appear in the generated answer with the same value and the same period/company. A different or \
approximated number, a number from the wrong period, or a missing period is a FAIL.
- The direction of change and the main conclusion must match.
- Wording, formatting, and extra correct detail beyond the reference are fine; rounding \
($82.96 bn vs $82,959 million) is fine.
- If the reference has no figures, judge on whether the main claims agree.
If the generated answer says the data is unavailable but the reference gives figures, FAIL.

Respond with exactly one line in this format:
VERDICT: <PASS or FAIL> | REASON: <one short sentence naming the specific mismatch, if any>
"""


def with_retries(fn, tries=5):
    """Call fn(); on Groq rate-limit errors wait and retry."""
    for attempt in range(1, tries + 1):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "413" in msg or "too large" in msg.lower():
                raise   # waiting can't shrink an oversized request
            low = msg.lower()
            transient = any(k in low for k in ("503", "502", "504", "unavailable", "overloaded",
                                               "high demand", "timed out", "timeout", "connection"))
            rate_limited = "429" in msg or "rate" in low
            if not (rate_limited or transient) or attempt == tries:
                raise
            if transient and not rate_limited:
                wait = 15 * attempt        # server-side hiccup: back off and retry
                print(f"    temporary API error, waiting {wait}s (retry {attempt}/{tries}) ...")
                time.sleep(wait)
                continue
            m = re.search(r"(?:try again|retry) in (\d+(?:\.\d+)?)\s*(ms|s|m)\b", msg)
            wait = 30 * attempt
            if m:
                val, unit = float(m.group(1)), m.group(2)
                wait = val / 1000 if unit == "ms" else val * 60 if unit == "m" else val
                wait = min(wait + 2, 300)
            if "per day" in msg.lower() or "perday" in msg.lower().replace(" ", "") or "TPD" in msg:
                raise RuntimeError("Groq DAILY token limit reached -- rerun tomorrow (progress is saved).") from e
            print(f"    rate limited, waiting {wait:.0f}s (retry {attempt}/{tries}) ...")
            time.sleep(wait)


def judge_answer(client, question, reference_answer, generated_answer):
    user_prompt = (
        f"Question: {question}\n\n"
        f"Reference answer: {reference_answer}\n\n"
        f"Generated answer: {generated_answer}"
    )
    response = with_retries(lambda: client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    ))
    text = response.choices[0].message.content.strip()

    verdict, reason = "ERROR", (text or "judge returned empty output")
    if "VERDICT:" in text:
        try:
            verdict_part, reason_part = text.split("|", 1)
            verdict = verdict_part.split("VERDICT:")[1].strip().upper()
            reason = reason_part.split("REASON:")[1].strip() if "REASON:" in reason_part else reason_part.strip()
        except Exception:
            pass
    if verdict not in ("PASS", "FAIL"):
        verdict = "ERROR"
    return verdict, reason


# --------------------------------------------------------------------------
# RETRIEVAL CHECK
# --------------------------------------------------------------------------

_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+")


_FIG = re.compile(r"\$?\(?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+)\)?\s*(billion|million|bn|B\b|M\b)?", re.I)


def _figs(text):
    """Numbers in `text` as [(value_in_millions_or_raw, rounded_flag)]. '12.9 billion' -> (12900, True)."""
    out = []
    for m in _FIG.finditer(str(text)):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        try:
            v = float(raw.replace(",", ""))
        except ValueError:
            continue
        if unit in ("billion", "bn", "b"):
            out.append((v * 1000, True))
        else:
            out.append((v, False))
    return out


def _ref_figs(reference):
    """Reference figures worth checking: comma-grouped numbers, decimals, or numbers with a unit."""
    return [(v, r) for v, r in _figs(reference)
            if r or (v != int(v)) or (v >= 1000 and not 1900 <= v <= 2100)]


def _present(fig, pool):
    v, rounded = fig
    tol = 0.006 if rounded else 0.0005
    return any(abs(v - w) <= max(tol * v, 1e-9) or (rounded and abs(v - w) / v <= tol) for w, _ in pool)


def _share(reference, text):
    ref = _ref_figs(reference)
    if not ref:
        return None
    pool = _figs(text)
    # a raw '4.1' in text may also be 4,100 million -> add x1000 variants of billion-style decimals
    pool += [(w * 1000, True) for w, _ in pool if w < 1000 and w != int(w)]
    return round(sum(_present(f, pool) for f in ref) / len(ref), 2)


def answer_figure_recall(reference, generated):
    """Share of the reference answer's figures found in the generated answer (unit-aware,
    e.g. '$12.9 billion' matches 12,900). No LLM involved. None if no such figures."""
    return _share(reference, generated)


def expected_doc_ids(source_docs):
    """'*AAPL*' -> all 4 AAPL filings; '*2023 Q3 INTC*' -> {'INTC Q3 2023'}."""
    parts = str(source_docs).strip("*").split()
    if len(parts) == 1:
        return {f"{parts[0]} {q} {y}" for q, y in ALL_PERIODS}
    if len(parts) == 3:
        year, q, ticker = parts
        return {f"{ticker} {q} {year}"}
    return set()


def retrieval_metrics(question, source_docs, reference):
    """(filing_hit, figure_coverage). figure_coverage = share of the reference answer's
    numbers (e.g. 82,959) that appear in the retrieved chunks -- a direct check that the
    right table/text reached the LLM. None if the reference has no such numbers."""
    try:
        got = retrieve(question)
        exp = expected_doc_ids(source_docs)
        hit = int(bool(exp) and exp <= {c["doc_id"] for c in got})
        text = " ".join(c["text"] for c in got)
        return hit, _share(reference, text)
    except Exception:
        return 0, None


# --------------------------------------------------------------------------
# RUN
# --------------------------------------------------------------------------

def run_eval(csv_path, limit=None):
    df = pd.read_csv(csv_path)
    if limit:   # random sample (seeded) so a small run isn't just the first company
        df = df.sample(n=min(limit, len(df)), random_state=0).sort_index()

    done = pd.read_csv(RESULTS_PATH) if os.path.exists(RESULTS_PATH) else pd.DataFrame()
    if len(done):   # drop failed rows so they are retried, keep only real PASS/FAIL results
        ok = done["Verdict"].isin(["PASS", "FAIL"]) & ~done["Generated Answer"].astype(str).str.startswith("ERROR")
        done = done[ok]
    done_q = set(done["Question"]) if len(done) else set()
    results = done.to_dict("records") if len(done) else []

    groq_client = get_llm_client(JUDGE_PROVIDER)   # judge client (any provider)

    for n, (_, row) in enumerate(df.iterrows(), start=1):
        question = row["Question"]
        if question in done_q:
            continue

        print(f"[{n}/{len(df)}] {question[:80]}...")

        try:
            generated = with_retries(lambda: ask(question))
        except RuntimeError as e:
            print(f"STOPPING: {e}")
            break
        except Exception as e:
            generated = f"ERROR: {e}"

        if generated.startswith("ERROR"):
            verdict, reason = "ERROR", generated[:400]
        else:
            try:
                verdict, reason = judge_answer(groq_client, question, row["Answer"], generated)
            except RuntimeError as e:
                print(f"STOPPING: {e}")
                break
            except Exception as e:
                verdict, reason = "ERROR", str(e)

        hit, cov = retrieval_metrics(question, row["Source Docs"], row["Answer"])
        print(f"    -> {verdict} (filings={hit}, figures found={cov}): {reason[:300]}")

        results.append({
            "Question": question,
            "Question Type": row["Question Type"],
            "Source Chunk Type": row["Source Chunk Type"],
            "Source Docs": row["Source Docs"],
            "Reference Answer": row["Answer"],
            "Generated Answer": generated,
            "Verdict": verdict,
            "Judge Reason": reason,
            "Retrieval Hit": hit,
            "Figure Coverage": cov,
            "Answer Figure Recall": answer_figure_recall(row["Answer"], generated),
        })
        pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)   # save after every question
        time.sleep(SLEEP_BETWEEN)

    print(f"\nSaved results -> {RESULTS_PATH}")
    print("LLM calls made in this run:", {f"{p}:{m}": n for (p, m), n in CALL_COUNTS.items()})
    if results:
        print_summary(pd.DataFrame(results))


def print_summary(results_df):
    r = results_df.copy()
    r["Pass"] = (r["Verdict"] == "PASS").astype(int)
    n_err = int((r["Verdict"] == "ERROR").sum())
    r = r[r["Verdict"] != "ERROR"]
    if n_err:
        print(f"\n!! {n_err} question(s) ended in ERROR (API/rate-limit/parse problems) and are excluded below."
              f" Rerun the same command to retry them.")
    if r.empty:
        print("No graded results yet.")
        return

    print("\n" + "=" * 60)
    print("OVERALL ACCURACY:", f"{r['Pass'].mean():.1%}", f"({r['Pass'].sum()}/{len(r)})")
    if "Retrieval Hit" in r:
        print("FILING HIT RATE:", f"{r['Retrieval Hit'].mean():.1%}")
    if "Figure Coverage" in r and r["Figure Coverage"].notna().any():
        c = r[r["Figure Coverage"].notna()]
        print("AVG FIGURE COVERAGE (ref numbers present in retrieved text):", f"{c['Figure Coverage'].mean():.1%}")
        fails = c[c["Pass"] == 0]
        print(f"Failures where some reference figures never reached the model (coverage < 100%): {(fails['Figure Coverage'] < 0.95).sum()}")
        print(f"Failures where ALL figures were retrieved (LLM reading / bad reference): {(fails['Figure Coverage'] >= 0.95).sum()}")
    if "Answer Figure Recall" in r and r["Answer Figure Recall"].notna().any():
        rec = r["Answer Figure Recall"]
        sus = r[(r["Pass"] == 1) & (rec < 0.5)]
        strict = ((r["Pass"] == 1) & ((rec >= 0.5) | rec.isna())).mean()
        print(f"STRICT ACCURACY (PASS and >=50% of reference figures in answer): {strict:.1%}")
        print(f"Judge PASS but <50% of reference figures present (judge may be lenient): {len(sus)}")
    print("=" * 60)

    for col in ["Question Type", "Source Chunk Type"]:
        print(f"\nBy {col}:")
        print(r.groupby(col)["Pass"].agg(["mean", "count"])
              .rename(columns={"mean": "accuracy", "count": "n"}).sort_values("accuracy"))

    print("\nBy Question Type x Source Chunk Type (accuracy):")
    print(r.pivot_table(index="Question Type", columns="Source Chunk Type",
                        values="Pass", aggfunc="mean").round(2))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "ingest":
        ingest(sys.argv[2:])
    elif command == "run":
        csv_arg = sys.argv[2] if len(sys.argv) > 2 else "qna_data.csv"
        limit_arg = int(sys.argv[3]) if len(sys.argv) > 3 else None
        run_eval(csv_arg, limit=limit_arg)
    elif command == "summary":
        print_summary(pd.read_csv(RESULTS_PATH))
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
