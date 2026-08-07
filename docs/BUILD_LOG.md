# How this was built

Nine phases, each reviewed before the next began. This document records what was
built at each stage, the decisions that shaped it, and — more usefully — the
things that turned out to be wrong.

The bugs are the interesting part. Most were found by *running* the system and
looking at real output, not by tests passing. A test suite tells you the code
does what you wrote; only output tells you whether what you wrote was right.

**Final state:** 567 tests, ruff and mypy clean, every phase verified against
real models and the real Groq API.

---

## Phase 1 — Scaffold, config, logging, ingestion and chunking

**Built:** project layout, `AppConfig` (YAML + env overrides), structlog setup,
PDF parsing with PyMuPDF, heading detection, section-aware chunking,
`chunks.jsonl`, the ingestion CLI, and synthetic PDF fixtures.

### Decisions

**`src/` layout with an editable install.** Imports behave identically in tests,
the API, Streamlit and CI. A flat layout works until it doesn't, and it fails in
CI where nobody is watching.

**`chunks.jsonl` as a hard boundary.** Ingestion imports no ML libraries and
touches no network, so chunking tests run in milliseconds. Re-indexing with new
embedding parameters never requires re-parsing PDFs.

**Chunks never cross a section boundary.** Splitting purely by size merges the
end of one topic with the start of the next — the most common cause of
confidently wrong RAG answers.

**Sizing in tokens, not characters.** bge-base truncates at 512 tokens
*silently*. Character-based sizing overflows on dense technical prose and you
lose the tail with no error at all.

**Tables extracted to markdown as atomic chunks.** Technical manuals put the real
answers in tables; as raw text flow they become unreadable column soup.

**Dual text fields.** `text` for display and the LLM, `embed_text` — prefixed
with the section breadcrumb — for indexing. A chunk reading "set this to 30
seconds" is unretrievable on its own.

### What went wrong

**The test fixtures were lying.** Text drawn past the page edge is clipped on
render, so extracted chunks came out truncated mid-word — and the tests asserted
against the truncated text, passing happily. Fixtures now wrap inside the
margins.

**Header/footer stripping ate real content.** The first rule dropped anything
repeating in a margin *band*. But a body paragraph wraps into several short
lines, so a page whose text started high lost its opening sentences. Narrowed to
the single topmost and bottommost line per page, short ones only.

**Small-section merging silently undid a decision the user had made.** Page-bounded
fallback exists so chunks never span pages; the merge step was merging pages 1
and 2 into one chunk. Merging is now disabled in fallback mode.

---

## Phase 2 — FAISS, BM25 and reciprocal rank fusion

**Built:** the embedding protocol and bge implementation, `FaissStore`,
`Bm25Store`, RRF, the hybrid retriever, index manifests, the index CLI.

### Decisions

**`IndexFlatIP`, not IVF or HNSW.** Exact cosine search. Approximate indexes pay
off in the hundreds of thousands of vectors; below that they add a training step,
tuning parameters and recall loss for no measurable latency win. Exact search
also means the Phase 8 metrics measure the retriever rather than the index's
approximation error.

**RRF fuses ranks, not scores.** Cosine is bounded; BM25 is an unbounded sum of
IDF terms that shifts with corpus statistics. Any weighted average needs a
normalisation that silently drifts as documents are added. Ranks need no
calibration.

**Identifier-aware BM25 tokenisation.** `request_timeout` is indexed whole *and*
split. This is precisely where dense retrieval fails: embeddings place
`request_timeout` and `connection_deadline` close together, which helps
paraphrase and hurts badly when someone asks about one exact parameter.

**Zero-score BM25 hits are dropped.** A chunk sharing no query term scores 0;
including it lets RRF reward it purely for existing.

**Full provenance on every candidate** — which retrievers found it, at what rank,
with what raw score. Diagnosing bad retrieval means knowing *why* a chunk
surfaced.

### Verification

Live with real bge (768 dims): `request_timeout` returned the parameter table at
rank 1 from **both** retrievers.

---

## Phase 3 — Cross-encoder reranking

**Built:** `CrossEncoderReranker` behind a protocol, `rerank_candidates()` as a
pure function, `RetrievalPipeline` composing the whole path.

### Decisions

**The model and the ordering logic are separate.** The part most likely to be
subtly wrong — what gets dropped, what order survives, what happens on failure —
is a pure function testable without loading a model.

**The whole fused pool is scored, not just the top-k.** Otherwise the reranker
could never promote a chunk that fusion buried.

**It degrades, never fails.** Model error, wrong number of scores, or disabled →
keep fused order and truncate. A reranker returning the wrong count is caught
explicitly: zipping mismatched lists would attach every score to the wrong
candidate, silently.

**Both `fused_rank` and `final_rank` survive.** The gap between them is exactly
how much the cross-encoder disagreed with the retrievers.

### Finding

The score distribution separates cleanly: relevant passages score **+4 to +6**,
irrelevant ones cluster at **≈ -11**. A ~15-point gap with nothing in between,
which makes `min_rerank_score` a real filter rather than a guess.

---

## Phase 4 — Groq generation with citation-tagged prompting

**Built:** the prompt module, `GroqClient` with retry and error translation, the
answerer, the `ask` CLI.

### Decisions

**Source numbering is the contract.** Chunks are numbered in rank order and the
model can only refer to them by number. The same mapping is stored on the
`Answer`, so verification checks against exactly what the model saw. Numbering
stays contiguous even when context truncation drops sources — a gap would invite
a citation to a number that isn't there.

**The refusal string is exact and comes from config**, quoted verbatim in the
prompt. Detection normalises case and punctuation: models reproduce the sentence
reliably but vary the trailing period, and a refusal misread as an answer is
scored as a hallucination by eval.

**Empty context never reaches the model.** No candidates → immediate refusal, no
API call. Spending a request to be told what we already know invites an answer
from the model's parametric memory.

**A bad API key is 500 with no detail leaked** — the key is the deployment's
problem, not the caller's.

### Verification

All three behaviours confirmed live. The partial-answer case is the one worth
quoting:

> Automatic rollback is triggered when the error rate exceeds the configured
> budget for two consecutive evaluation windows [1]. The sources do not cover how
> many retries happen before that […] [2].

It answered the supported half, cited it, and named the gap.

---

## Phase 5 — Post-hoc citation verification

**Built:** sentence segmentation, claim detection, citation validity and coverage
checks, and — after a proposal and explicit sign-off — layered entailment
checking.

### Decisions

**Code regions are masked before anything is parsed.** `items[0]` in a code span
would register as a citation to source 0, and a documentation assistant emits
code constantly. Periods inside `v2.1.0`, `30.5` and `config.yaml` defeat naive
sentence splitting too.

**"The sources do not cover X" is deliberately not a claim.** Flagging it as
uncited would penalise exactly the honest behaviour the prompt asks for.

**Entailment runs in two layers** (proposed, then chosen by the user):

1. *Lexical grounding*, always: every number and identifier in a claim must
   appear in the sources it cites. Free, offline, deterministic. Number words
   normalise to digits so "thirty seconds" matches `30`.
2. *An LLM judge*, only for what layer 1 cannot resolve. Each claim is shown
   **only the chunks it cites** — a judge given the whole context could mark a
   claim supported by evidence the answer never pointed at.

**Both layers fail open.** A rate limit or a malformed reply leaves a claim
inconclusive, never accused. A false accusation shown to a user is worse than a
missed one.

**Findings are reported, never repaired.** Rewriting the model's output to hide a
missing citation would conceal the failure from both the user and eval.

### Verification

Five deliberately defective answers; four correctly failed. The fifth —
*"Rollbacks are automatic."* uncited — **passed**, because `min_claim_words` was
4 and that sentence is 3 words. Lowered to 3.

The case only the judge can catch: an answer claiming a timeout is "fully
configurable" when the source says it is *not* configurable. No wrong number, no
missing identifier, just a contradiction.

---

## Phase 6 — FastAPI backend

**Built:** the app factory, typed schemas, error mapping, request-id middleware,
an in-process job registry, and routes for health, query, upload, ingest, jobs
and eval.

### Decisions

**Startup is tolerant.** A service that refuses to boot without an index is
unusable — you cannot call the ingest endpoint that would create one. Missing
index → `/health` reports `degraded`, `/query` returns 503 with the command that
fixes it.

**A refusal is a 200 with `refused: true`.** A client must distinguish "the docs
don't cover this" from "something broke, retry".

**Slow work returns 202 and a job id.** The registry is in-process and
non-durable; jobs are lost on restart and two workers would each keep their own.
A deliberate trade for a single-instance internal tool — and the 404 message says
so rather than leaving you guessing.

**Uploads are treated as hostile input.** `../../config/default.yaml` is reduced
to a bare filename before anything is written.

### What went wrong

**The docstring said models load at startup. They didn't.**
sentence-transformers loads weights on first *use*, so the first request after a
restart paid for both models:

| | First query | Subsequent |
|---|---|---|
| Before | **13,192 ms** | ~400 ms |
| After warmup | **416 ms** | ~350 ms |

Found by measuring rather than trusting the comment. Startup now forces an encode
and a rerank before accepting traffic.

---

## Phase 7 — Streamlit front end

**Built:** the API client, presentation helpers, and a two-tab app — Documents
for upload and ingestion, Ask for questions with expandable cited sources.

### Decisions

**The UI is a client of the API, not of the pipeline.** One code path into
retrieval and generation; the UI cannot drift from what the API does; using the
UI exercises the API contract. All testable logic lives outside the Streamlit
script, which re-runs top to bottom on every interaction.

**Cited sources open by default, uncited ones stay collapsed but visible.** It is
obvious what the model was shown and chose not to use.

**Answer text is HTML-escaped before citation badges are injected.** An answer
quoting XML config would otherwise inject markup into the page.

### What went wrong

**Streamlit and Starlette were incompatible** — every page load returned 500.
Starlette 1.4 made `GZipResponder.thread_minimum_size` required; Streamlit's
subclass doesn't pass it. Both allow `<2`, so the resolver is free to pick the
broken pair. Pinned `starlette<1.4.0` with the reason in the file.

**Relative imports broke `streamlit run`.** It executes the file as a top-level
script with no parent package. Switched to absolute imports.

**Adjacent citation badges read as one number.** `[1][2]` rendered as two flush
pills looking like "12".

Also a wrong turn in the test setup: `httpx.ASGITransport` is async-only, so a
sync client cannot use it. The client now takes an injected `httpx.Client`, which
lets tests pass FastAPI's `TestClient` directly — a better API arrived at by
being wrong first.

---

## Phase 8 — Golden dataset and evaluation harness

**Built:** the golden set format and loader, retrieval metrics (recall@k, MRR,
nDCG@k, precision@k), generation metrics (citation precision/recall, context
recall, groundedness, refusal matrix), the harness, optional RAGAS, and a
drafting script.

### Decisions

**Expected sources are pinned by file and page, not chunk id.** Chunk ids are
content-derived, so changing `chunk_tokens` would invalidate every label at once
and turn a tuning experiment into a measurement of nothing. File and page are
properties of the document.

**Retrieval is measured twice** — on the fused list and after reranking. The
difference between those two blocks is the reranker's entire justification.

**Refusal is reported as a confusion matrix.** Over-refusing looks like caution
and destroys usefulness; under-refusing looks like helpfulness and is how a RAG
system misleads people. One number cannot show both.

**Citation recall and context recall are separate.** Context recall is the
ceiling — the model cannot cite what it was never shown — so the gap between them
separates a retrieval problem from a generation one.

**RAGAS is opt-in.** It costs several calls per item and its scores are not
reproducible, which makes it a poor thing to gate a build on. Every failure
inside it degrades to "no RAGAS scores".

### What went wrong

**nDCG@5 = 1.116** — mathematically impossible. Several chunks land on one page (a
prose lead-in *and* the table it introduces) and both matched the same
`(file, page)` label, so the relevant count exceeded the number that exist. Each
expected source can now be found exactly once.

**`must_contain: ["30"]` was satisfied by an answer saying "300."** Substring
matching defeats the entire purpose of literal value checking. Now matched on
word boundaries, with digits excluding a leading `.` so `1.30` fails too.

**A byte-order mark broke the loader.** This file is hand-authored on Windows; an
invisible byte should not cost an afternoon.

**A schema gap found by running it:** the first labels demanded `"30"` where the
prose says "thirty seconds", so `must_contain_coverage` was 0.333 for a system
answering correctly. Entries can now be alternatives: `[["30", "thirty"]]`.

### Finding

**Reranking made retrieval worse** on the sample corpus: MRR 1.000 → 0.889,
recall@1 0.917 → 0.750. On nine questions over seven synthetic chunks this proves
nothing about a real corpus — but it is exactly what the harness exists to
surface.

---

## Phase 9 — CI regression gate

**Built:** baseline comparison with configurable gates, the `check_regression`
CLI, a generated CI corpus, `config/ci.yaml`, and the GitHub Actions workflow.

### Decisions

**Retrieval-only in CI.** No API key, no cost, deterministic — the three
properties a build gate has to have. A gate depending on a non-deterministic
model is a gate that gets disabled.

**Tolerances, not floors.** Each gate says how far a metric may move in the bad
direction. An absolute floor has to be rewritten every time the system improves.

**Everything is reported, including improvements.** A gate you only hear from when
it is angry is one you learn to ignore.

**A vanished metric fails; a metric missing from the baseline passes with a
note.** The first is almost always a broken run; the second is just a new gate.

**`--reason` is required to update a baseline.** A baseline updated without one is
indistinguishable from a baseline updated to make a red build green, and six
months later nobody can tell which happened.

**The CI corpus is generated, not committed** — the same PDFs the test suite
builds, including a corrupt and a scanned file so CI also proves those are
skipped rather than taking the run down.

### What went wrong

**A drop of exactly the stated tolerance failed the gate.** `0.90 - 0.85` is
`0.05000000000000004` in binary floating point, so a gate promising to allow a
0.05 drop rejected one. The "fires on noise" problem in miniature, in the very
code written to avoid it.

### Verification

Gate passes unchanged (exit 0). With dense retrieval disabled via an env
override, it fails (exit 1) naming the metric, the size of the drop, the
tolerance, and how to accept the change deliberately.

---

## Things left open

Three decisions were raised and deliberately left to the user:

1. **`min_rerank_score`** — the Phase 3 score distribution suggested about
   `-2.0`. Still `null`. Setting it would let obviously-irrelevant contexts
   short-circuit before the API call.
2. **Chunk-merge labelling** — when several undersized sections merge, the
   resulting chunk carries the last section's breadcrumb. On real documents with
   300–500 token sections this effectively never fires.
3. **Gate thresholds** — deliberately loose. Tighten once a few runs on a real
   corpus show how much the numbers move on their own.

And one piece of work that is genuinely the user's: **the golden set**.
`scripts/make_golden.py` drafts questions with sources pre-filled, but drafts are
not a golden set — model-written questions are answerable by the exact chunk they
came from in the exact wording, which flatters retrieval.

---

## What made the difference

**Running it beat testing it.** The 13-second cold start, nDCG above 1.0, "300"
matching "30", the truncated fixtures, the Streamlit 500 — every one was found by
looking at real output. Tests confirmed the code did what was written; only
output showed whether that was right.

**Fakes behind protocols.** Embedder, reranker, LLM and token counter are all
injected. The entire suite runs offline with no API key and no model download,
including full API and UI-client tests against the real ASGI app.

**Failures degrade rather than propagate.** One corrupt PDF doesn't fail
ingestion. A broken reranker falls back to fused order. A failed entailment judge
leaves claims inconclusive rather than accused. Each of those is a decision about
what the system owes the user when a part of it breaks.

**Comments explain why, not what.** The float epsilon, the `utf-8-sig` codec, the
starlette pin, the code masking — each carries the reason it exists, because the
next person to read it will otherwise "simplify" it back into a bug.
