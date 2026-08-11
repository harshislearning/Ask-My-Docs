# Ask My Docs

A production-grade, domain-specific RAG system over a PDF knowledge base: hybrid
retrieval (dense + keyword) with reciprocal rank fusion, cross-encoder reranking,
cited generation, and post-hoc citation verification — served through FastAPI with
a Streamlit front end and gated by an automated evaluation suite in CI.

> **Complete.** Ingest, index, retrieve, rerank, generate, verify — behind an HTTP API
> and a Streamlit front end, measured by an evaluation harness and gated in CI.
>
> - **[What this is, in plain language](docs/PROJECT_EXPLAINED.md)** — start here
> - [How it was built, phase by phase](docs/BUILD_LOG.md) — the engineering log
---

## ScreenShots

<img width="1896" height="897" alt="Screenshot 2026-08-11 123718" src="https://github.com/user-attachments/assets/fe7f1d3c-c13b-486b-804f-bd2ede116b75" />
<img width="1891" height="872" alt="Screenshot 2026-08-11 124033" src="https://github.com/user-attachments/assets/dcff0a69-e9c4-45c4-b332-4176372fe0c9" />
<img width="1917" height="886" alt="Screenshot 2026-08-11 124354" src="https://github.com/user-attachments/assets/73aec530-56bc-4c31-a898-f7c007ba6906" />
<img width="1883" height="847" alt="Screenshot 2026-08-11 124557" src="https://github.com/user-attachments/assets/ef14a32a-d34b-47a6-832e-40505a36a3b3" />




```
---

## Architecture

```
PDFs ──> pdf_loader ──> structure ──> chunker ──> chunks.jsonl
          (PyMuPDF)     (headings)  (token-bounded)      │
                                                         ├──> FAISS  (bge-base-en-v1.5)
                                                         └──> BM25   (rank_bm25)
                                                                     │
query ──> both indexes ──> RRF fusion ──> cross-encoder rerank ──> top-k
                                                                     │
                                                    Groq llama-3.3-70b-versatile
                                                                     │
                                              citation verification ──> answer  [Phase 5]
```

`chunks.jsonl` is a deliberate hard boundary. Ingestion is pure — no ML libraries,
no network — so it is fast to test and can be re-run independently of indexing.
Re-indexing with different embedding parameters never requires re-parsing PDFs.

### Layout

| Path | Purpose |
|---|---|
| `src/askmydocs/config.py` | Every tunable, one place. YAML + env overrides. |
| `src/askmydocs/logging_setup.py` | structlog; JSON in CI, human-readable locally. |
| `src/askmydocs/models.py` | Domain types shared across phases. |
| `src/askmydocs/tokens.py` | Cheap token estimation for prompt budgeting. |
| `src/askmydocs/ingestion/` | PDF to chunks. See below. |
| `src/askmydocs/indexing/` | FAISS + BM25 index construction, embedding. |
| `src/askmydocs/retrieval/` | RRF fusion, hybrid retriever, cross-encoder reranker, pipeline. |
| `src/askmydocs/generation/` | Groq client, citation-tagged prompting, answerer. |
| `src/askmydocs/verification/` | Citation checking and entailment. |
| `src/askmydocs/api/` | FastAPI app, schemas, background jobs. |
| `src/askmydocs/ui/` | Streamlit front end and its API client. |
| `src/askmydocs/evaluation/` | Golden set, retrieval + generation metrics, harness. |
| `scripts/` | Batch entry points (ingestion is a job, not a request handler). |
| `config/` | `default.yaml` for real runs, `test.yaml` for the suite and CI. |

---

## Setup

Requires Python 3.11 (pinned in `.python-version`) and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"
```

`uv` fetches and pins CPython 3.11 itself, so you do not need it installed system-wide.
Heavy dependencies are split into extras and installed per phase:

```bash
uv pip install -e ".[ml]"      # embeddings, FAISS, BM25, reranker  (Phase 2+)
uv pip install -e ".[llm]"     # Groq client                        (Phase 4+)
uv pip install -e ".[serve]"   # FastAPI + Streamlit                (Phase 6+)
uv pip install -e ".[eval]"    # RAGAS + pandas                     (Phase 8+)
```

Copy `.env.example` to `.env` and set `GROQ_API_KEY`. It is only needed from Phase 4
onward — ingestion and retrieval run entirely offline.

<details>
<summary>Windows: <code>uv venv --python 3.11</code> fails with "Missing expected target directory for Python minor version link"</summary>

uv tries to create a directory junction for the `3.11` alias, which needs Developer Mode
or admin rights. The interpreter itself downloads fine — point uv straight at it:

```bash
uv python install 3.11
uv venv --python "$env:APPDATA/uv/python/cpython-3.11.15-windows-x86_64-none/python.exe"
```

Then use `uv pip install --python .venv/Scripts/python.exe -e ".[dev]"`, or activate the
venv first — without one of those, uv may install into your system Python instead.
</details>

### VS Code

Open the `RAG APP` folder (not its parent) and accept the recommended extensions.
`.vscode/` is committed, so run and debug work immediately.

**Pick the interpreter first:** `Ctrl+Shift+P` → *Python: Select Interpreter* →
`.venv\Scripts\python.exe`. Without it Pylance resolves imports against the wrong Python
and reports errors that do not exist.

`F5` offers a run configuration for every entry point — the API, the Streamlit UI,
ingestion, indexing, asking a question, evaluation, the regression check — plus an
**API + UI** compound that starts both at once. All of them read `.env`, so
`GROQ_API_KEY` is picked up without exporting anything.

Breakpoints work throughout, including inside request handlers: the uvicorn config sets
`subProcess: true`, without which the debugger attaches to the reloader's parent process
and never stops in your code.

Tests are discoverable in the Testing panel. `Ctrl+Shift+P` → *Run Task* has
**Full check (lint, types, tests)** — the same three commands CI runs.

---

## Running ingestion

Drop PDFs into `data/raw_pdfs/` (nested folders are searched too), then:

```bash
uv run python scripts/ingest.py
```

| Flag | Effect |
|---|---|
| `--input PATH` | Ingest a different folder |
| `--force` | Re-parse everything, ignoring the unchanged-file cache |
| `--config PATH` | Use a different config file |
| `--log-format json` | Machine-readable logs |

Outputs land in `data/processed/`:

- **`chunks.jsonl`** — one chunk per line, ready for indexing
- **`manifest.json`** — what happened to every document, including *why* anything was skipped

Re-running is cheap: documents are identified by content hash, so unchanged files keep
their existing chunks and chunk ids. Editing a PDF re-ingests only that file. Changing
any chunking parameter invalidates the cache automatically.

### How chunking works

Naive fixed-size chunking merges the end of one topic with the start of the next, which
is the most common cause of confidently wrong RAG answers. Instead:

1. **Parse** with PyMuPDF, keeping font size and weight per line — in a PDF there is no
   semantic markup, so typography is the only structural signal available. Repeating
   headers and footers are detected across pages and dropped.
2. **Detect headings** from font size relative to the document's modal body size,
   bold-and-short lines, and numbering patterns (`3.2.1 ...`). Numbered headings set
   their own nesting depth; the rest are ranked by font size.
3. **Build sections** with full breadcrumb trails (`3. Deployment > 3.2 Rollback`).
   Hyphenated line breaks are repaired; paragraphs are rebuilt from wrapped lines.
4. **Split within sections only**, never across them, sized in **tokens** using the
   embedding model's own tokenizer (bge-base truncates at 512 silently — character
   sizing overflows on dense technical prose and you lose the tail with no error).
   Undersized sections are merged forward so retrieval is not flooded with fragments.
5. **Tables** are detected, serialised to markdown, and emitted as atomic chunks.
   Technical manuals put the real answers in tables; as raw text flow they become
   unreadable column soup.

Every chunk stores two fields: `text` (shown to the user, sent to the LLM) and
`embed_text` (indexed) — the latter prefixed with its breadcrumb, so a chunk reading
"set this to 30 seconds" is still retrievable.

**Documents with no detectable headings** fall back to page-bounded splitting, tagged
`structure_source: page_fallback` so evaluation can compare retrieval quality across the
two modes. **Scanned PDFs** (no text layer) are skipped with a reason rather than indexed
as empty; there is no OCR step.

---

## Building the indexes

```bash
uv run python scripts/build_index.py
```

Reads `data/processed/chunks.jsonl` and writes to `data/indexes/`:

| File | Contents |
|---|---|
| `faiss.index` | Dense vectors, `IndexFlatIP` over L2-normalised bge embeddings |
| `faiss_ids.json` | Position to `chunk_id` mapping (FAISS addresses by position) |
| `bm25.pkl` | Keyword index |
| `index_manifest.json` | Model, dimensions, chunk count, fingerprint of the chunks file |

The fingerprint is how staleness is caught: re-run ingestion without re-indexing and the
next query logs `index_out_of_date` rather than silently answering from old chunks.

### Why these choices

**`IndexFlatIP`, not IVF/HNSW.** Exact cosine search. Approximate indexes only pay off in
the hundreds of thousands of vectors; below that they add a training step, tuning
parameters, and recall loss for no measurable latency win. Exact search also means the
Phase 8 retrieval metrics measure the *retriever*, not the index's approximation error.

**Identifier-aware BM25 tokenisation.** `request_timeout` is indexed whole *and* split
into `request` + `timeout`. This is the case dense retrieval reliably gets wrong —
embeddings put `request_timeout` and `connection_deadline` close together, which is
useful for paraphrase and actively harmful when someone asks about one exact parameter.

**Both indexes see `embed_text`**, not raw `text`, so the section breadcrumb is part of
what makes a chunk findable in either retriever.

### Retrieval and fusion

Each retriever produces its own ranking independently; neither sees the other's results.
Reciprocal Rank Fusion then combines them:

```
score(chunk) = sum over retrievers of  weight / (k + rank)
```

Ranks, not scores — cosine sits in [-1, 1] while BM25 is an unbounded sum of IDF terms
that shifts with corpus statistics, so any weighted average of the two requires a
normalisation that silently changes as documents are added. RRF needs no calibration.

`rrf_k` (default 60) controls damping: small k makes rank 1 dominate, large k flattens
the curve so *agreement between retrievers* outweighs any single retriever's confidence.
`vector_weight` / `bm25_weight` allow ablation — set one to 0 to measure the other alone.

Every candidate keeps its full provenance (which retrievers found it, at what rank, with
what raw score), because diagnosing bad retrieval means knowing *why* a chunk surfaced.

### Reranking

Retrieval and reranking answer different questions. The bi-encoder behind FAISS embeds
the query and each chunk *separately* — it never sees them together, so it compares two
independently-formed summaries. BM25 does not model meaning at all. Both are cheap enough
to run over the whole corpus, and both are approximations.

A cross-encoder reads query and passage *jointly* in one forward pass. Far more accurate,
far too slow for a corpus — so it runs last, over a few dozen fused candidates.

On the sample corpus, asking *"what triggers an automatic rollback?"*:

```
fused (RRF)                       reranked (cross-encoder)
1. field_notes.pdf                1. 2. Rollout Stages > 2.2 Rollback  (was #2, +5.20)
2. 2. Rollout Stages > 2.2 Roll   2. field_notes.pdf                   (was #1, -11.41)
```

`ms-marco-MiniLM-L-6-v2` emits **raw logits, not probabilities** — roughly -11 for
irrelevant and +4 to +6 for a genuine answer. Only ordering is meaningful, and any
`min_rerank_score` you set must be read on that scale (about 0 is the natural boundary).

The stage degrades rather than fails: if the model errors, returns the wrong number of
scores, or is disabled via `rerank_enabled`, the fused order is kept and truncated. An
optional refinement must never take down the answer path.

`RetrievalPipeline.search()` composes the whole thing and is the single entry point the
API and eval harness use.

---

## Asking a question

```bash
uv run python scripts/ask.py "what is the default request timeout?"
```

Runs the whole path — retrieval, fusion, reranking, generation — and prints the answer
with numbered sources. Needs `GROQ_API_KEY` in `.env`. Flags: `--show-sources` to print
each source's full text, `--no-rerank` to compare against unranked retrieval.

```
Q: What is the default request timeout?

The default request timeout is thirty seconds [1][2].

SOURCES
  [1] deployment_handbook.pdf - p. 3 - 3. Timeouts          rerank +7.93
  [2] config_reference.pdf - p. 1 - 4. Timeout Parameters   rerank +1.62
```

### How the prompt enforces citation

The prompt is treated as source code — one place, versioned, unit-tested — because it is
the only mechanism making the model cite evidence and refuse without it.

**Source numbering is the contract.** Chunks are numbered `[1]`, `[2]`, ... in rank order
and the model can only refer to them by number. That same mapping is stored on the
`Answer`, so Phase 5 verifies citations against exactly what the model was shown —
nothing is recomputed. Numbering stays contiguous even when context truncation drops
sources, since a gap would invite a citation to a number that is not there.

**The refusal string is exact and comes from config.** It is quoted verbatim in the
system prompt, and detection normalises case and punctuation — models reproduce the
sentence reliably but vary the trailing period, and a refusal misread as an answer gets
scored as a hallucination by eval.

**Partial answers are handled explicitly.** Asked something the sources only half cover,
the model answers the supported half with a citation and states plainly what is missing,
rather than filling the gap:

> Automatic rollback is triggered when the error rate exceeds the configured budget for
> two consecutive evaluation windows [1]. The sources do not cover how many retries
> happen before that [...] [2].

**Empty context never reaches the model.** If retrieval returns nothing, the answerer
refuses without an API call — spending a request to be told what we already know, and
inviting an answer from the model's own parametric memory.

### Talking to Groq

`GroqClient` wraps the SDK so the rest of the system depends on a protocol it can fake in
tests, every retry is logged as structured data, and provider failures become this
project's own exception types. Retries cover 429s, 5xx, timeouts and connection drops
with exponential backoff plus jitter, honouring `Retry-After` when sent. Auth failures
and malformed requests are never retried — repeating them just burns quota. SDK-level
retries are disabled so attempts are not silently multiplied.

---

## Running the API

```bash
uv run uvicorn askmydocs.api.main:app --reload
```

Interactive docs at `http://127.0.0.1:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Readiness, index stats, and which component is not ready |
| `POST /query` | Ask a question; returns answer, sources, verification |
| `POST /documents/upload` | Store PDFs in the ingestion folder |
| `POST /ingest` | Ingest and optionally rebuild the indexes (202 + job id) |
| `GET /documents` | What is ingested, including what was skipped and why |
| `GET /jobs/{id}` | Poll a background job |
| `POST /eval` | Trigger an evaluation run *(harness lands in Phase 8)* |

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What is the default request timeout?"}'
```

### Design decisions worth knowing

**Startup is tolerant, and warms the models.** A service that refuses to boot without
an index is unusable — you cannot call the ingest endpoint that would create one. So a
missing index means `/health` reports `degraded` and `/query` returns 503 with the exact
command to run, rather than a crash loop. When an index *is* present, startup forces bge
and the cross-encoder to load: measured cold, the first query took **13.2s** against
**0.4s** for every one after it, and that cost belongs at boot, not on a user.

**A refusal is a 200, not an error.** `refused: true` in the body. Refusing is correct
behaviour, and a client must be able to tell it apart from a failure it might retry.

**Every provider failure maps to a status that says what to do about it.** A rate limit
is 429 (retry), a timeout 504, a missing index 503, a generation failure 502. A bad API
key is **500 with no detail leaked** — the key is this deployment's problem, not the
caller's, so the detail goes to the logs and never to the response.

**Slow work runs as a job.** Ingesting a real corpus takes minutes, past any sensible
proxy timeout. `POST /ingest` returns 202 with a job id; poll `GET /jobs/{id}`. The
registry is in-process and non-durable — jobs are lost on restart and two workers would
each keep their own. That is a deliberate trade for an internal single-instance service,
and the 404 message says so rather than leaving you guessing.

**Ingestion is serialised against itself but not against queries.** It rewrites the
files queries read from, so a second ingest gets 409. When it finishes it swaps the
pipeline reference — atomic in Python — so in-flight queries complete against the old
indexes instead of a half-built one.

**Per-request overrides are applied to a copy** of the retrieval config. Mutating the
shared one would silently reconfigure every other caller.

**Every response carries `x-request-id`**, honoured from the request if supplied, and
bound to every log line emitted while handling it — so one wrong answer can be traced
across retrieval, generation and verification.

---

## Running the front end

Start the API first, then:

```bash
uv run streamlit run src/askmydocs/ui/streamlit_app.py
```

Two tabs. **Documents** uploads PDFs, runs ingestion, and shows what is indexed —
including anything that was skipped and why. **Ask** answers questions and shows the
answer with citation badges, a verification banner, and one expander per source.

Cited sources open by default with their page number, section breadcrumb and relevance
score; uncited ones stay collapsed but visible, so it is obvious what the model was
shown and chose not to use.

**The UI is a client of the API, not of the pipeline.** It costs a network hop and is
worth it: one code path into retrieval and generation, the UI cannot drift from what the
API actually does, and using the UI genuinely exercises the API contract. Everything
worth testing lives in `ui/api_client.py` and `ui/formatting.py` — a Streamlit script
re-runs top to bottom on every interaction, so logic defined inside it is hard to test
and easy to break unnoticed.

Answer text is HTML-escaped before citation badges are injected. An answer quoting XML
config would otherwise inject markup into the page.

> **Version pin:** `starlette` is held below 1.4.0 in the `serve` extra. Starlette 1.4
> made `GZipResponder.thread_minimum_size` a required argument and Streamlit's
> `_MediaAwareGZipResponder` subclasses it without passing one, so every page load
> returns 500. Both packages allow `<2`, so a resolver is free to pick the broken
> combination. Remove the pin once Streamlit catches up.

---

## Evaluation

```bash
uv run python scripts/run_eval.py                 # full: retrieval + generation
uv run python scripts/run_eval.py --no-generate   # retrieval only, no API calls
uv run python scripts/run_eval.py --limit 10      # quick smoke run
```

Reads `eval/golden/golden_set.jsonl` (see [its README](eval/golden/README.md) for the
format) and writes a timestamped artifact to `eval/runs/` containing both the aggregate
report and every per-item record — because an aggregate that moved is useless without
the item that moved it.

### What is measured

**Retrieval**, reported *twice* — once on the fused list and once after reranking. The
difference between those two blocks is the reranker's entire justification, and on a
given corpus it can be negative:

```
                    fused      reranked
  mrr               1.000      0.889
  recall@1          0.917      0.750
  ndcg@5            1.000      0.917
```

**Generation** — `citation_precision` (did it cite sources that exist?),
`citation_recall` (did it cite the *expected* ones?), `context_recall` (did retrieval
even put them in the prompt?), `groundedness`, and `must_contain_coverage` for exact
values. Citation recall and context recall together separate a retrieval problem from a
generation one: context recall is the ceiling, since the model cannot cite what it was
never shown.

**Refusal**, as a confusion matrix. The single most informative table in the report:

```
  correctly refused    3
  wrongly answered     0   <- misleading answers
  wrongly refused      1   <- lost usefulness
  correctly answered   5
```

Over-refusing looks like caution and destroys usefulness; under-refusing looks like
helpfulness and is how a RAG system misleads people. One number cannot show both.

### Design notes

**Each expected source can be found once.** Chunking regularly puts several chunks on
one page — a prose lead-in and the table it introduces — and all of them match a single
`(file, page)` label. Counting each as a separate hit lets the relevant count exceed the
number that exist, which pushed nDCG to 1.116 before this was fixed.

**Macro-averaged across queries.** Every question counts equally; micro-averaging would
let one question with many expected sources dominate the whole set.

**Generation is optional.** `--no-generate` gives deterministic retrieval metrics with
no API key and no cost — which is what CI gates on.

**RAGAS is opt-in** (`use_ragas: true`). It adds LLM-judged faithfulness and answer
relevance, wired to Groq and the local bge embeddings so it never reaches for OpenAI.
Off by default: it costs several calls per item and its scores are not reproducible,
which makes it a poor thing to gate a build on. Every failure inside it degrades to "no
RAGAS scores" rather than failing the run.

---

## Tests

```bash
uv run pytest
```

The suite builds its own PDFs at runtime (`tests/fixtures/pdf_factory.py`) rather than
checking in binary fixtures — the typography that drives each assertion is visible in the
test source. No test touches the network or loads an ML model: embeddings, reranking,
token counting and the LLM are all injected fakes.

```bash
uv run pytest tests/unit          # fast, no PDF rendering
uv run pytest -m "not slow"       # skip anything that would download a model
uv run ruff check . && uv run mypy
```

---

## The CI regression gate

`.github/workflows/ci.yml` runs on every PR. Two jobs: **quality** (ruff, mypy, the full
test suite) and **eval-gate**, which builds a synthetic corpus, indexes it, evaluates,
and fails the build if a gated metric regressed.

```bash
# what CI runs, locally
python scripts/check_regression.py --config config/ci.yaml --no-generate
```

```
  metric                              baseline   current     delta
  ------------------------------------------------------------------------
  reranked.recall@5                     1.0000    1.0000   +0.0000  ok
  reranked.mrr                          0.8889    0.8889   +0.0000  ok
  retrieval.mrr                         1.0000    0.9167   -0.0833  FAIL  regressed by 0.0833, tolerance is 0.05
  ------------------------------------------------------------------------
  FAIL - 1 gate(s) regressed
```

Exit codes: `0` passed, `1` a gate regressed, `2` could not run. The same table is
written to the PR's checks tab via `GITHUB_STEP_SUMMARY`, so a reviewer sees the numbers
without opening logs.

### How it stays trustworthy

**Retrieval-only in CI** (`--no-generate`). No API key, no cost, and deterministic — the
three properties a build gate has to have. Generation metrics exist and are gated in
`config/default.yaml` for local runs, but a gate that depends on a non-deterministic
model is a gate that gets disabled.

**Tolerances, not floors.** Each gate says how far a metric may move *in the bad
direction*. An absolute floor has to be rewritten every time the system improves; a
tolerance keeps meaning as the numbers move.

**Everything is reported, not just failures.** Improvements are labelled too. A gate you
only hear from when it is angry is one you learn to ignore.

**A vanished metric fails.** If `reranked.recall@5` is missing from a run, that is almost
always a broken run, and skipping it would hide the breakage behind a green build. A
metric missing from the *baseline* passes with a note — that is just a new gate.

**The CI corpus is generated, not committed.** `scripts/make_ci_corpus.py` writes the
same PDFs the test suite builds, so the corpus is reproducible and the typography driving
every assertion stays visible in Python. It deliberately includes a corrupt and a scanned
file, so CI also proves those are skipped rather than taking the run down.

**`config/ci.yaml` mirrors `config/default.yaml`** for everything that affects quality.
Config files are loaded whole rather than merged, so when you change a tunable in one,
change it in the other — otherwise the gate stops describing what you actually ship.

### Updating the baseline

Sometimes a regression is the right call: you traded recall@1 for precision, or changed
chunking knowing retrieval would shift. Record that deliberately:

```bash
python scripts/check_regression.py --config config/ci.yaml --no-generate \
  --update-baseline --reason "chunk_tokens 450 -> 320; recall@1 -0.06, precision@3 +0.11"
```

`--reason` is **required**. A baseline updated without one is indistinguishable from a
baseline updated to make a red build green, and six months later nobody can tell which
happened. The reason and a timestamp are stored in the file, along with the config
snapshot that produced the numbers.

Commit `eval/baselines/*.json` **in the same PR as the change that caused it**. That way
the diff shows the trade-off and the reasoning next to the code that made it, and a
reviewer can disagree.

**When updating is wrong:** the metric dropped and you do not know why. That is the case
the gate exists for. Find the cause first — the run artifact in `eval/runs/` (uploaded as
a CI artifact on failure too) has per-item records showing exactly which questions moved.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Scaffold, config, logging, ingestion + chunking | done |
| 2 | FAISS + BM25 indexes, RRF fusion | done |
| 3 | Cross-encoder reranking | done |
| 4 | Groq generation with citation-tagged prompting | done |
| 5 | Post-hoc citation verification | |
| 6 | FastAPI backend | done |
| 7 | Streamlit front end | done |
| 8 | Golden dataset + eval harness (RAGAS + custom metrics) | done |
| 9 | GitHub Actions regression gate | done |
