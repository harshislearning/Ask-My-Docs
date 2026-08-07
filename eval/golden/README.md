# The golden set

`golden_set.jsonl` is the file the harness and the CI gate read. One JSON object
per line:

```json
{"id":"q001","question":"What is the default request timeout?","expected_answer":"30 seconds.","expected_sources":[{"source_file":"handbook.pdf","page":3}],"must_contain":["30"],"tags":["parameter"]}
{"id":"q002","question":"What is the parental leave policy?","answerable":false,"tags":["unanswerable"]}
```

| Field | Meaning |
|---|---|
| `id` | Stable, unique. Referenced in run artifacts and CI output. |
| `question` | Phrased the way a colleague would actually ask it. |
| `expected_answer` | What a correct answer says. Reporting and RAGAS only — never exact-matched. |
| `expected_sources` | `{source_file, page}`. Page is 1-based, as printed. |
| `answerable` | `false` means the documents do not answer it and the system must refuse. |
| `must_contain` | Values a correct answer must contain verbatim. See below. |
| `tags` | Free-form. Useful for slicing results (`table`, `multi-hop`, `unanswerable`). |

## Why sources are pinned by file and page

Not by `chunk_id`. Chunk ids are content-derived, so changing `chunk_tokens` or
fixing a heading rule would invalidate every label at once — and a tuning
experiment would silently become a measurement of nothing. File and page are
properties of the document, so a label written once stays true no matter how the
chunker changes.

A chunk that spans pages counts as a match if the expected page falls inside its
range.

## `must_contain`

The cheapest signal that survives rewording: the exact values a reader will act on.
Matched on word boundaries, so `"30"` is **not** satisfied by an answer saying `300`
or `1.30` — catching that transposition is the whole point.

An entry may be a list of alternatives, any one of which satisfies it:

```json
"must_contain": ["request_timeout", ["30", "thirty"]]
```

Use alternatives whenever a document writes the same value two ways — `30` in a table,
"thirty" in the prose beside it. Demanding a single spelling fails correct answers, and
that noise is how a metric stops being trusted.

## Composition

Aim for **50–100 items**, of which **20–30% unanswerable**. A set with no
unanswerable questions cannot tell a system that knows things from one that
answers everything, and the refusal path is the one most likely to regress
without anyone noticing.

Worth covering deliberately:

- **Exact values** — the answer is a number or an identifier (`must_contain`)
- **Tables** — where the answer lives in a grid, not prose
- **Multi-source** — needs two or more chunks, with both listed
- **Paraphrase** — asked in words the document never uses
- **Near-miss unanswerable** — plausible, adjacent to real content, but not covered.
  These are the hard ones and the most valuable.

## Drafting help

```bash
uv run python scripts/make_golden.py --per-doc 8 --unanswerable 10
```

Samples real chunks, asks the model for questions each one answers, and writes
drafts to `drafts.jsonl` with `expected_sources` pre-filled from the source chunk.

**Drafts are not a golden set.** Model-written questions tend to be answerable by
that exact chunk in that exact wording, which flatters retrieval. Read every line:
rewrite the phrasing, delete the ones that just read the chunk back, fill in
`must_contain`, and check each page against the real PDF. Then rename the file to
`golden_set.jsonl`.

## Sanity check

```bash
uv run python scripts/run_eval.py --no-generate
```

Retrieval metrics only, no API calls. The report flags any `expected_sources`
that are not in the index — a label pointing at a page that was never ingested
scores as a permanent retrieval miss and looks exactly like a retrieval bug.
