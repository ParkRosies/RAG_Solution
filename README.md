# RAG over SEC 10-Q filings (AAPL, AMZN, MSFT, NVDA, INTC)

Retrieval-augmented QA over 20 10-Q PDFs (2022 Q3 - 2023 Q3), evaluated on a 195-question dataset
with an LLM judge plus LLM-free retrieval/answer metrics. Built to run entirely on free tiers.

## Pipeline
1. **Parse** (`pdfplumber`): ruled tables -> Markdown-style rows with column headers; borderless statements
   (Intel/Microsoft) recovered from numeric text lines; other text -> 300-word chunks, 50 overlap.
2. **Embed** locally with `BAAI/bge-base-en-v1.5` (no API limit); stored in `rag_store/`.
3. **Retrieve**: route the question to the right company/quarter filings, then per filing fuse dense
   cosine rank and BM25 with reciprocal-rank fusion; guarantee some table chunks; pack under a context budget.
4. **Answer** with a free-tier LLM (Gemini / Gemma / Groq / Cerebras / OpenRouter via OpenAI-compatible API).
5. **Evaluate**: strict LLM judge, filing hit, figure coverage, answer figure recall (unit-aware).

## Run
```
pip install -r requirements.txt
cp .env.example .env        # add your key(s)
python evaluate.py ingest pdfs/*.pdf
python evaluate.py run qna_data.csv [N]   # resumable; N = sample size
python evaluate.py summary
```
Questions/answers are not included; put `qna_data.csv` next to the scripts.

## Results
`results/eval_results_v1_full_run.csv`: full 195-question run, 43.6% (85/195) judged correct.
`results/audit_of_failures.csv` labels 49 failures against the filings (wrong references / over-strict judge vs real errors); estimated audited accuracy 50-57% vs raw 43.6%. See the PDF report. Later code changes (table recovery for borderless
statements, exact-figure prompt, unit-aware metrics, Gemma support) are included in this repo but
were not re-evaluated end to end because free-tier API quotas were exhausted.
