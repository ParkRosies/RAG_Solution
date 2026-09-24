"""
RAG pipeline for 10-Q style PDFs (text + tables), using:
  - pdfplumber            for local, lightweight parsing (no GPU needed)
  - sentence-transformers for LOCAL embeddings (BAAI/bge-base-en-v1.5, no API limits)
  - an OpenAI-compatible LLM API (Groq / Gemini / Cerebras / OpenRouter) for answering
  - numpy                 for similarity search (no vector DB needed at this scale)

Usage (keys go in .env):
    GROQ_API_KEY=...

    python RAG_pipeline.py ingest /path/to/file1.pdf /path/to/file2.pdf
    python RAG_pipeline.py ask "What was Apple's Q3 2022 iPhone revenue?"

Everything is stored in ./rag_store/ (chunks.json + embeddings.npy).
`ingest` reuses vectors already in rag_store/ for any chunk whose text is
unchanged, so re-ingesting after a chunking tweak only embeds the new chunks.
"""

import os
import sys
import json
import re
import time
import math
import collections
import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

STORE_DIR = "rag_store"
CHUNKS_PATH = os.path.join(STORE_DIR, "chunks.json")
EMB_PATH = os.path.join(STORE_DIR, "embeddings.npy")

EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # local, no API limits
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# --- answering LLM: any OpenAI-compatible provider. Set in .env: LLM_PROVIDER, LLM_MODEL, <PROVIDER>_API_KEY ---
PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()      # groq | gemini | cerebras | openrouter
KEY_VARS = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY",
            "cerebras": "CEREBRAS_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
BASE_URLS = {"gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
             "cerebras": "https://api.cerebras.ai/v1", "openrouter": "https://openrouter.ai/api/v1"}
DEFAULT_MODELS = {"groq": "openai/gpt-oss-120b", "gemini": "gemini-2.5-flash",
                  "cerebras": "llama-3.3-70b", "openrouter": "meta-llama/llama-3.3-70b-instruct:free"}
LLM_MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODELS[PROVIDER])


# min seconds between calls to the same (provider, model); keeps us under free-tier RPM caps.
# 5s = at most 12 calls/min. Override with LLM_MIN_INTERVAL in .env.
DEFAULT_INTERVAL = {"groq": 0.0, "gemini": 5.0, "cerebras": 2.0, "openrouter": 3.0}
CALL_COUNTS = collections.Counter()      # (provider, model) -> LLM calls made this run
_last_call = {}


def _throttle(client, provider):
    interval = float(os.environ.get("LLM_MIN_INTERVAL", DEFAULT_INTERVAL[provider]))
    orig = client.chat.completions.create

    def create(**kw):
        key = (provider, kw.get("model"))
        wait = _last_call.get(key, 0) + interval - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            CALL_COUNTS[key] += 1
            return orig(**kw)
        finally:
            _last_call[key] = time.time()

    try:
        client.chat.completions.create = create
    except Exception:
        pass
    return client


def get_llm_client(provider=None):
    provider = provider or PROVIDER
    key = os.environ[KEY_VARS[provider]]
    # max_retries=0: the SDKs silently retry 429/5xx twice, and every retry counts against
    # your quota. evaluate.py's with_retries() handles waiting explicitly instead.
    if provider == "groq":
        from groq import Groq
        return _throttle(Groq(api_key=key, max_retries=0), provider)
    from openai import OpenAI          # pip install openai
    return _throttle(OpenAI(api_key=key, base_url=BASE_URLS[provider], max_retries=0), provider)


CHUNK_TOKEN_TARGET = 300   # bge max input is 512 tokens, so keep chunks smaller
CHUNK_OVERLAP = 50

TABLE_MAX_CHARS = 1300     # split bigger tables so bge (512 tok) sees every row
MAX_CONTEXT_CHARS = 1600   # per-chunk cap when building the LLM prompt
CONTEXT_CHAR_BUDGET = int(os.environ.get("CONTEXT_CHAR_BUDGET", 10000 if PROVIDER == "groq" else 24000))  # total context sent (~3-3.5K tokens)
MAX_ANSWER_TOKENS = int(os.environ.get("MAX_ANSWER_TOKENS", 1500))       # includes reasoning tokens
REASONING_EFFORT = os.environ.get("REASONING_EFFORT", "low")             # gpt-oss only: low/medium/high


# --------------------------------------------------------------------------
# 1. PARSING
# --------------------------------------------------------------------------

TICKER_NAMES = {
    "AAPL": "Apple Inc.",
    "AMZN": "Amazon.com Inc.",
    "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation",
    "INTC": "Intel Corporation",
}

# Matches filenames like "2023_Q1_AAPL.pdf", "2023 Q3 NVDA.pdf", "2022_Q3_AMZN.pdf"
FILENAME_PATTERN = re.compile(r"(\d{4})[_\s]Q(\d)[_\s]([A-Za-z]+)", re.IGNORECASE)


def parse_doc_id(filename: str):
    """
    Extract (ticker, year, quarter, doc_id) from a filing filename.
    doc_id is a normalized "TICKER QN YYYY" string used both in chunk text
    (so the LLM can distinguish quarters) and for matching against the
    'Source Docs' column in an eval CSV.
    """
    base = os.path.basename(filename)
    match = FILENAME_PATTERN.search(base)
    if match:
        year, quarter, ticker = match.group(1), match.group(2), match.group(3).upper()
        doc_id = f"{ticker} Q{quarter} {year}"
        return ticker, year, quarter, doc_id
    # fallback: no recognizable pattern, just use the filename stem
    stem = os.path.splitext(base)[0]
    return stem, None, None, stem


def guess_company(filename: str) -> str:
    """Human-readable company + period label, e.g. 'Apple Inc. (AAPL) — Q1 2023'."""
    ticker, year, quarter, doc_id = parse_doc_id(filename)
    full_name = TICKER_NAMES.get(ticker, ticker)
    if year and quarter:
        return f"{full_name} ({ticker}) — Q{quarter} {year}"
    return full_name



_NUMTOK = re.compile(r"\(?\$?\s?\d[\d,]*(?:\.\d+)?\)?%?|—")


def promote_numeric_runs(text, min_rows=4):
    """Borderless statements (Intel, Microsoft...) come out of pdfplumber as plain text lines.
    Find runs of lines that end in >=2 numbers, and turn them into table rows
    'label | n1 | n2' so they get table treatment (own chunk, header, table quota).
    Returns (text_without_those_lines, [(caption, rows), ...])."""
    lines = text.splitlines()
    def parse(ln):
        ln = ln.strip()
        if len(ln) > 160 or not re.search(r"[A-Za-z]", ln):
            return None
        toks = ln.split()
        nums = []
        while toks and re.fullmatch(r"\(?\$?\(?[\d—][\d,\.]*\)?%?\)?|\$|—", toks[-1]):
            t = toks.pop()
            if t != "$":
                nums.append(t.replace("$", ""))
        nums.reverse()
        if len(nums) < 2 or not toks:
            return None
        return [" ".join(toks).rstrip(" $")] + nums
    parsed = [parse(l) for l in lines]
    keep, tables, i, n = [True] * len(lines), [], 0, len(lines)
    while i < n:
        if parsed[i] is None:
            i += 1
            continue
        j, rows, last = i, [], i
        while j < n and j - last <= 1:          # allow one label-only line between rows
            if parsed[j] is not None:
                rows.append(parsed[j]); last = j
            j += 1
        if len(rows) >= min_rows:
            head = [l.strip() for l in lines[max(0, i - 4):i] if l.strip()]
            if re.search(r"Millions|Unaudited|Except", rows[0][0], re.I):   # unit/date header line
                head.append(rows[0][0] + " " + " ".join(rows[0][1:]))
                rows = rows[1:]
            tables.append((" / ".join(head)[-220:], rows))
            for k in range(i, last + 1):
                keep[k] = False
        i = last + 1
    return "\n".join(l for l, k in zip(lines, keep) if k), tables


def extract_pdf(path: str):
    """
    Returns a list of raw items, each either:
      {"type": "text", "page": n, "content": "..."}
      {"type": "table", "page": n, "content": [[...], [...]], "caption": "..."}
    """
    items = []
    company = guess_company(path)
    _, _, _, doc_id = parse_doc_id(path)

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # --- tables first, so we can exclude their text from the page text ---
            tables = page.find_tables()
            table_bboxes = []
            for t in tables:
                data = t.extract()
                if not data or len(data) < 2:
                    continue
                table_bboxes.append(t.bbox)
                # Column headers usually sit in the few lines just above the table
                # ("Three Months Ended / September 30, / 2022 2023"). Read that strip as
                # proper text lines -- sorting individual words by x-position scrambles
                # multi-line headers and makes the model mix up which column is which year.
                caption = ""
                try:
                    top = t.bbox[1]
                    strip = page.crop((0, max(0, top - 48), page.width, top))
                    lines = [ln.strip() for ln in (strip.extract_text() or "").splitlines() if ln.strip()]
                    caption = " / ".join(lines)[-220:]
                except Exception:
                    pass
                items.append({
                    "type": "table",
                    "page": page_num,
                    "company": company,
                    "doc_id": doc_id,
                    "caption": caption,
                    "content": data,
                })

            # --- page text, with table regions cropped out to avoid duplication ---
            cropped_page = page
            for bbox in table_bboxes:
                try:
                    cropped_page = cropped_page.outside_bbox(bbox)
                except Exception:
                    pass
            text = cropped_page.extract_text() or ""
            # only for pages where pdfplumber found no real (ruled) table
            promoted = []
            if not any(len(t.extract() or []) >= 4 for t in tables):
                text, promoted = promote_numeric_runs(text)
            for cap, rows in promoted:
                items.append({
                    "type": "table", "page": page_num, "company": company,
                    "doc_id": doc_id, "caption": cap, "content": rows,
                })
            if text.strip():
                items.append({
                    "type": "text",
                    "page": page_num,
                    "company": company,
                    "doc_id": doc_id,
                    "content": text,
                })

    return items


# --------------------------------------------------------------------------
# 2. CHUNKING
# --------------------------------------------------------------------------

def clean_rows(table_rows):
    """Drop empty cells, '$' cells; glue ')' and '%' back onto the number."""
    out = []
    for row in table_rows:
        merged = []
        for c in row:
            c = (c or "").replace("\n", " ").strip()
            if not c or c == "$":
                continue
            if c in (")", "%", ")%") and merged:
                merged[-1] += c
                continue
            merged.append(c)
        if merged:
            out.append(merged)
    return out


def table_to_pieces(table_rows, max_chars=TABLE_MAX_CHARS):
    """Cleaned table -> list of markdown-ish strings, each <= max_chars, header repeated."""
    rows = clean_rows(table_rows)
    if len(rows) < 2:
        return []
    lines = ["| " + " | ".join(r) + " |" for r in rows]
    if sum(len(l) for l in lines) < 80:
        return []
    header, body = lines[0], lines[1:]
    pieces, cur, size = [], [], len(header)
    for line in body:
        if cur and size + len(line) > max_chars:
            pieces.append(cur)
            cur, size = [], len(header)
        cur.append(line)
        size += len(line) + 1
    if cur:
        pieces.append(cur)
    return [header + "\n" + "\n".join(p) for p in pieces]


def chunk_text(text, target_words=CHUNK_TOKEN_TARGET, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + target_words]
        chunks.append(" ".join(chunk))
        i += target_words - overlap
    return chunks


def build_chunks(raw_items):
    """Turn raw parsed items into retrieval-ready chunks with metadata."""
    chunks = []
    for item in raw_items:
        meta = {
            "company": item["company"],
            "doc_id": item["doc_id"],
            "page": item["page"],
            "type": item["type"],
        }

        if item["type"] == "table":
            caption = (item.get("caption") or "").strip()
            header_line = f"[Table on page {item['page']}"
            if caption:
                header_line += f" — {caption}"
            header_line += f", from {item['company']}]\n"
            for piece in table_to_pieces(item["content"]):
                chunks.append({"text": header_line + piece, **meta})

        else:  # text
            for piece in chunk_text(item["content"]):
                prefix = f"[{item['company']}, page {item['page']}]\n"
                chunks.append({"text": prefix + piece, **meta})

    return chunks


# --------------------------------------------------------------------------
# 3. EMBEDDING + STORAGE
# --------------------------------------------------------------------------

_model = None


def embed_texts(texts, client=None, input_type="document"):
    """Embed locally with bge. `client` is unused; kept for compatibility."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, device="cpu")
    if input_type == "query":
        texts = [QUERY_PREFIX + t for t in texts]
    return _model.encode(
        texts, batch_size=32, normalize_embeddings=True,
        show_progress_bar=len(texts) > 1,
    ).astype(np.float32)


def ingest(pdf_paths):
    os.makedirs(STORE_DIR, exist_ok=True)

    # reuse vectors for chunks whose text hasn't changed (same embedding model!)
    reuse = {}
    if os.path.exists(CHUNKS_PATH) and os.path.exists(EMB_PATH):
        old_chunks = json.load(open(CHUNKS_PATH))
        old_vecs = np.load(EMB_PATH)
        if len(old_chunks) == len(old_vecs):
            reuse = {c["text"]: old_vecs[i] for i, c in enumerate(old_chunks)}

    all_chunks = []
    for path in pdf_paths:
        print(f"Parsing {path} ...")
        chunks = build_chunks(extract_pdf(path))
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    texts = [c["text"] for c in all_chunks]
    todo = [t for t in dict.fromkeys(texts) if t not in reuse]
    print(f"{len(texts) - len(todo)} chunks reused, embedding {len(todo)} new with {EMBED_MODEL} ...")
    if todo:
        new_vecs = embed_texts(todo, input_type="document")
        reuse.update({t: v for t, v in zip(todo, new_vecs)})
    embeddings = np.array([reuse[t] for t in texts], dtype=np.float32)

    with open(CHUNKS_PATH, "w") as f:
        json.dump(all_chunks, f)
    np.save(EMB_PATH, embeddings)
    print(f"Stored {len(all_chunks)} chunks -> {STORE_DIR}/")


# --------------------------------------------------------------------------
# 4. RETRIEVAL + ANSWERING
# --------------------------------------------------------------------------

def cosine_sim(query_vec, matrix):
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    matrix_norm = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8)
    return matrix_norm @ query_norm


COMPANY_PATTERNS = {
    "AAPL": r"\b(apple|aapl|iphone|ipad)\b",
    "AMZN": r"\b(amazon|amzn|aws)\b",
    "MSFT": r"\b(microsoft|msft|azure)\b",
    "NVDA": r"\b(nvidia|nvda)\b",
    "INTC": r"\b(intel|intc)\b",
}
LATEST_DOC = ("2023", "3")   # 'latest / most recent 10-Q' -> Q3 2023
_store = None
CHUNKS_PER_DOC = int(os.environ.get("CHUNKS_PER_DOC", 6))   # per filing, for multi-filing questions
STOPWORDS = set("the a an of in to and or how has have what is are was were for on by with as at "
                "from over any been its it this that which change changed changes reported across "
                "do does did there their between compare compared".split())


def _tok(text):
    return re.findall(r"[a-z0-9][a-z0-9,\.]*", text.lower())


def _load_store():
    global _store
    if _store is None:
        chunks = json.load(open(CHUNKS_PATH))
        _store = (chunks, np.load(EMB_PATH), [_tok(c["text"]) for c in chunks])
    return _store


def route_query(query, all_doc_ids):
    """Return the doc_ids this question should search (None = search everything)."""
    q = query.lower()
    tickers = [t for t, pat in COMPANY_PATTERNS.items() if re.search(pat, q)]
    if not tickers:
        return None

    words = {"first": "1", "second": "2", "third": "3"}
    wanted = {(m[1] or m[2], m[0] or m[3])
              for m in re.findall(r"q([1-3])\s*(?:of\s*)?(?:fy)?\s*(20\d\d)|(20\d\d)\s*q([1-3])", q)}
    for w, n in re.findall(r"\b(first|second|third)\s+(?:fiscal\s+)?quarter(?:\s+(?:of|in))?\s*(?:fiscal\s*)?(?:year\s*)?(20\d\d)", q):
        wanted.add((n, words[w]))
    compares = re.search(r"\b(previous|prior|earlier|compared|over time|trend|across|changed)", q)
    explicit_latest = re.search(r"\b(latest|most recent|newest)\s+(10-?q|filing|report)", q)
    vs_previous = re.search(r"(compar|relative|versus|\bvs\b|stack up).{0,40}\b(previous|prior|earlier|past)", q)
    if not wanted and not vs_previous and (explicit_latest or (not compares and re.search(r"\b(latest|most recent|current|newest)\b", q))):
        wanted = {LATEST_DOC}

    docs = []
    for d in all_doc_ids:                       # doc_id = "TICKER Qn YYYY"
        t, qn, yr = d.split()
        if t in tickers and (not wanted or (yr, qn[1]) in wanted):
            docs.append(d)
    return docs or None


def _bm25(idx, q_terms, toks, k1=1.5, b=0.75):
    """BM25 scores for chunks `idx` (a list of chunk indices), computed within that group."""
    N = len(idx)
    avg = sum(len(toks[i]) for i in idx) / max(N, 1)
    df = collections.Counter(w for i in idx for w in set(toks[i]))
    scores = {}
    for i in idx:
        tf = collections.Counter(toks[i])
        sc = 0.0
        for w in q_terms:
            if w in tf:
                sc += math.log(1 + (N - df[w] + .5) / (df[w] + .5)) * tf[w] * (k1 + 1) / (
                    tf[w] + k1 * (1 - b + b * len(toks[i]) / avg))
        scores[i] = sc
    return scores


def _hybrid_rank(idx, sims, q_terms, toks, rw_terms=None):
    """Reciprocal-rank fusion of dense similarity, BM25 (original question) and,
    optionally, BM25 on LLM-rewritten terms -- within one group of chunks."""
    rankers = [sorted(idx, key=lambda i: -sims[i])]
    for terms in (q_terms, rw_terms):
        if terms:
            bm = _bm25(idx, terms, toks)
            rankers.append(sorted(idx, key=lambda i: -bm[i]))
    fused = collections.defaultdict(float)
    for order in rankers:
        for r, i in enumerate(order):
            fused[i] += 1 / (60 + r)
    return sorted(idx, key=lambda i: -fused[i])


def select_with_tables(ranked, chunks, n, min_tables):
    """Take the top-n ranked chunks but guarantee at least `min_tables` table chunks:
    financial questions are mostly answered by tables, which often rank low against
    long MD&A prose. Result order: best overall chunk, the forced tables, then the rest
    (order matters: the prompt-size budget is filled from the front)."""
    picked = ranked[:n]
    have = [i for i in picked if chunks[i]["type"] == "table"]
    need = min_tables - len(have)
    if need > 0:
        extra = [i for i in ranked if chunks[i]["type"] == "table" and i not in picked][:need]
        drop = [i for i in reversed(picked) if chunks[i]["type"] != "table"][:len(extra)]
        keep = [i for i in picked if i not in drop]
        head = keep[:1]      # best overall chunk first, then the guaranteed tables
        return head + extra + [i for i in keep if i not in head]
    return picked


REWRITE = os.environ.get("REWRITE", "0") == "1"   # off by default: hurt retrieval in tests
REWRITE_CACHE_PATH = os.path.join(STORE_DIR, "rewrite_cache.json")
_rw_cache = None


def _llm_extra(max_tokens):
    """Provider-specific kwargs (Groq free tiers need tight token caps)."""
    extra = {}
    if PROVIDER == "groq":
        extra["max_tokens"] = max_tokens
        if "gpt-oss" in LLM_MODEL:
            extra["extra_body"] = {"reasoning_effort": REASONING_EFFORT}
    return extra


def rewrite_query(question):
    """Turn a natural-language question into the vocabulary the filings use
    (line items, table names, phrases), so keyword + vector search can find the
    right tables. Cached on disk, so each question costs one LLM call ever."""
    global _rw_cache
    if not REWRITE:
        return ""
    if _rw_cache is None:
        _rw_cache = json.load(open(REWRITE_CACHE_PATH)) if os.path.exists(REWRITE_CACHE_PATH) else {}
    if question in _rw_cache:
        return _rw_cache[question]
    try:
        resp = get_llm_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You write search queries for retrieving passages and tables from SEC 10-Q filings. "
                    "Given a question, output ONE line of at most 40 words: the exact financial-statement "
                    "line items, table titles and phrases that would appear in the filing (e.g. 'Total net "
                    "sales', 'Operating income', 'Segment information', 'Effective tax rate', 'Net cash "
                    "provided by operating activities', 'foreign currency', 'Research and development'), "
                    "plus synonyms. No explanation, no company names, no punctuation other than commas.")},
                {"role": "user", "content": question},
            ],
            temperature=0,
            **_llm_extra(400),
        )
        out = " ".join((resp.choices[0].message.content or "").split())[:400]
    except Exception:
        return ""          # never let rewriting break retrieval; fall back to the raw question
    _rw_cache[question] = out
    os.makedirs(STORE_DIR, exist_ok=True)
    json.dump(_rw_cache, open(REWRITE_CACHE_PATH, "w"))
    return out


def retrieve(query, top_k=8, total_chunks=None):
    chunks, embeddings, toks = _load_store()
    q_vec = embed_texts([query], input_type="query")[0]
    sims = cosine_sim(q_vec, embeddings)
    q_terms = list(dict.fromkeys(w for w in _tok(query) if w not in STOPWORDS))
    rw = rewrite_query(query)
    rw_terms = list(dict.fromkeys(w for w in _tok(rw) if w not in STOPWORDS)) if rw else None

    doc_ids = sorted({c["doc_id"] for c in chunks})
    docs = route_query(query, doc_ids)
    if docs is None:                             # no company recognised: global search
        ranked = _hybrid_rank(list(range(len(chunks))), sims, q_terms, toks, rw_terms)
        return [chunks[i] for i in ranked[:top_k]]

    per_doc = top_k if len(docs) == 1 else CHUNKS_PER_DOC
    lists = []
    for d in docs:
        idx = [i for i, c in enumerate(chunks) if c["doc_id"] == d]
        ranked = _hybrid_rank(idx, sims, q_terms, toks, rw_terms)
        lists.append(select_with_tables(ranked, chunks, per_doc, max(2, per_doc // 2)))

    # round-robin by rank across filings, so every quarter gets its best chunks
    # first, until the total context budget (Groq TPM limits) is used up
    picked, used = [], 0
    for r in range(per_doc):
        for lst in lists:
            if r < len(lst):
                cost = min(len(chunks[lst[r]]["text"]), MAX_CONTEXT_CHARS)
                if used + cost <= CONTEXT_CHAR_BUDGET:
                    picked.append(lst[r])
                    used += cost
    picked.sort(key=lambda i: chunks[i]["doc_id"])
    return [chunks[i] for i in picked]


def ask(query, top_k=8):
    top_chunks = retrieve(query, top_k=top_k)

    context = "\n\n---\n\n".join(
        f"(Source: {c['company']} [{c['doc_id']}], page {c['page']}, type={c['type']})\n{c['text'][:MAX_CONTEXT_CHARS]}"
        for c in top_chunks
    )

    system_prompt = (
        "You are a financial analyst assistant. Answer the question using ONLY "
        "the provided context from SEC 10-Q filings. If the context includes tables "
        "in Markdown, read them carefully by column and row before answering. "
        "Always cite the company and page number for any figure you use. "
        "Table columns are often ordered oldest to newest (for example 2022 then 2023), so always "
        "read the column-header line in the [Table ...] label to pick the correct period before "
        "quoting a figure; never assume the first column is the current period. "
        "The context is grouped by filing (e.g. 'AAPL Q3 2023'); when the question spans "
        "several periods, report each period separately with its figure. "
        "Quote figures exactly as they appear in the tables (e.g. $4,109 million, not \"$4.1 billion\"), and when a metric is available both as a dollar amount and a percentage, give both. Be direct: give the specific figures first, then briefly explain drivers. "
        "Only say information is missing if no chunk contains it; if it is partly available, answer with what is there. "
        "If the answer isn't in the context, say so explicitly."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    client = get_llm_client()
    extra = _llm_extra(MAX_ANSWER_TOKENS)
    if "gemma" in LLM_MODEL.lower():     # Gemma has no system role on the Gemini API
        messages = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]
        extra = {k: v for k, v in extra.items() if k in ("max_tokens",)}
    else:
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(
        model=LLM_MODEL, messages=messages, temperature=0.1, **extra,
    )
    return response.choices[0].message.content


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
    elif command == "ask":
        question = " ".join(sys.argv[2:])
        print(ask(question))
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
