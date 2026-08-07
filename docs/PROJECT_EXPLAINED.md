# Ask My Docs — explained in plain language

A guide to talking about this project. No jargon without a definition.

> There is a companion document, [BUILD_LOG.md](BUILD_LOG.md), written for
> engineers reading the code. This one is for explaining the project to someone
> who has not seen it.

---

## The one-sentence version

> "I built a system that answers questions about a company's internal PDF
> documents, and — unlike a normal chatbot — every sentence it writes points at
> the exact page it came from, and it refuses to answer when the documents don't
> cover the question."

If they want one more sentence:

> "The hard part isn't getting an answer, it's proving the answer is real. So I
> also built the machinery that checks its own citations and a test suite that
> fails the build if answer quality drops."

---

## The problem it solves

Imagine a company with 400 pages of internal engineering documentation. An
engineer needs to know the default timeout for a service. Their options today:

- **Search by keyword** — only works if they guess the exact word the document used.
- **Ask ChatGPT** — it has never seen these private documents. It will guess, confidently.

Both fail. The second fails *dangerously*, because a confident wrong answer
looks exactly like a right one.

**This project fixes that** by only ever answering from the company's own
documents, and by making every claim traceable back to a page number.

---

## How it works, step by step

Think of it as a very careful research assistant.

### Step 1 — Reading the documents

The system reads every PDF and cuts it into small passages ("chunks"), because
you cannot fit 400 pages into an AI model at once.

**The naive way** is to cut every 500 words regardless of content. That
accidentally splits a sentence about *timeouts* onto the end of a chunk about
*deployments* — and then the AI reads both as one topic and gets confused.

**What I did instead:** detect the document's actual headings (by looking at
font size and boldness, since PDFs have no built-in structure), and never cut
across a section boundary. Tables get pulled out and kept whole, because
technical manuals put the real answers in tables and chopping one in half makes
it meaningless.

### Step 2 — Building two search systems, not one

When you ask a question, the system searches two different ways at once:

| Search type | Good at | Bad at |
|---|---|---|
| **Meaning search** (AI embeddings) | "How long before it gives up?" finding a page about *timeouts* | Exact names — it thinks `request_timeout` and `connection_deadline` are basically the same |
| **Keyword search** (BM25, classic search) | Finding the exact term `request_timeout` | Anything worded differently from the document |

They fail in *opposite* directions, which is exactly why using both is better
than either. The results get merged using a technique called **Reciprocal Rank
Fusion** — it combines two rankings without needing to compare their scores,
which matters because the two systems' scores aren't on the same scale.

### Step 3 — A second, smarter pass

The first two searches are fast but rough — they compare the question and each
passage *separately*. A slower, more accurate model called a **cross-encoder**
then re-reads the top ~30 results with the question and passage side by side,
and re-ranks them.

Too slow to run on 400 pages. Perfect for 30 candidates.

### Step 4 — Writing the answer

The top passages are handed to an AI model (Llama 3.3, running on Groq)
numbered `[1]`, `[2]`, `[3]`. The instructions are strict:

- Cite a source number after every factual claim
- Never invent a source number
- If the passages don't answer the question, say *"I don't have enough
  information"* — do not guess

### Step 5 — Checking its own work ← **the interesting part**

This is what separates the project from a tutorial. **Instructing an AI to cite
its sources is a request, not a guarantee.** So the system checks afterwards:

1. **Do the citations exist?** If the answer cites `[7]` but only 6 passages were
   provided, that's a fabrication.
2. **Is every claim cited at all?** Any sentence stating a fact with no citation
   is flagged.
3. **Does the cited passage actually say that?** Two layers:
   - *Free check:* every number and identifier in the answer must appear in the
     source it cites. Catches "the timeout is 300 seconds" when the document
     says 30.
   - *Smarter check:* for anything the first layer can't decide, a second, very
     narrow AI call judges whether the passage supports the claim.

I tested this with deliberately broken answers. It caught all of them:

| Fake defect | Caught? |
|---|---|
| Cited a source that didn't exist | Yes |
| Said "300 seconds" when the doc says thirty | Yes |
| Stated a fact with no citation | Yes |
| Claimed something the source contradicts | Yes |

---

## The nine phases

I built it in nine reviewed stages rather than all at once.

| # | Phase | In plain terms |
|---|---|---|
| 1 | **Foundation & reading PDFs** | Project structure, settings, logging, and turning PDFs into well-cut passages |
| 2 | **Two search systems** | Meaning search + keyword search, and merging their results |
| 3 | **Smarter re-ranking** | A slower, more accurate model re-orders the best candidates |
| 4 | **Writing answers** | Connecting to the AI model with strict citation rules |
| 5 | **Checking the answers** | Verifying citations exist, are used, and are actually supported |
| 6 | **The backend** | A web service so anything can use it — not just one script |
| 7 | **The interface** | A web app to upload documents and ask questions |
| 8 | **Measuring quality** | A test set of questions with known answers, and metrics scoring the system |
| 9 | **Automatic quality gate** | If a code change makes answers worse, the build fails |

**Phases 8 and 9 are the ones worth emphasising.** Most portfolio RAG projects
stop at phase 7 — a demo that works when you try it. Being able to *prove* it
works, and to catch it silently getting worse, is what "production-grade"
actually means.

---

## What makes this different from a tutorial project

Recruiters see a lot of RAG demos. These are the things that aren't in them:

**It knows when to say "I don't know."** Most demos always answer. This one
refuses when the documents don't support an answer, and I measure how often it
gets that decision right — both ways. Answering something it shouldn't is
misleading; refusing something it could have answered is useless. One number
can't show both, so I report a full breakdown.

**It checks its own citations.** Prompting a model to cite sources is a
suggestion. Verifying afterwards is engineering.

**Every part can be tested without the internet.** 561 automated tests run with
no API key and no AI model downloads, because the models sit behind swappable
interfaces. The whole suite runs in about 20 seconds.

**Quality is measured, not assumed.** There's a scored question set and metrics
for both search accuracy and answer accuracy. A change that makes the system
worse fails the build automatically.

**Failures degrade instead of crashing.** One corrupt PDF doesn't kill the
import. If the re-ranking model fails, results fall back to the earlier ranking.
If the verification AI call fails, claims are marked "unverified" rather than
falsely accused. Each of those was a decision about what the system owes the
user when a part of it breaks.

---

## Numbers you can quote

| | |
|---|---|
| Automated tests | 561, all passing |
| Test suite runtime | ~20 seconds, no network needed |
| Typical answer time | under 1 second (after a ~15s one-time startup) |
| Code quality | zero linting errors, fully type-checked |
| Phases | 9, each reviewed before the next |

**Be honest about scale if asked:** these were verified on a small corpus. The
architecture is what's being demonstrated; performance on a large document set
would need measuring, and the evaluation harness exists precisely to do that.

---

## Questions you might get, and how to answer

**"Why not just use ChatGPT / a vector database SaaS?"**
> ChatGPT has never seen the private documents. And an off-the-shelf tool gives
> you retrieval but not citation verification, refusal behaviour, or a quality
> gate — which is where the actual risk lives in this kind of system.

**"What was the hardest part?"**
> Cutting the documents up well. It sounds trivial and it isn't — PDFs have no
> structural markup, so you have to infer headings from font sizes. Get it wrong
> and the end of one topic gets glued to the start of the next, which is the
> single most common cause of confidently wrong answers in these systems.

**"How do you know it works?"**
> I built a scored question set with known correct answers and known source
> pages, including questions the documents deliberately *cannot* answer. The
> system is scored on how often it finds the right passage, cites the right
> source, and refuses when it should. Those numbers gate the build.

**"What would you do differently or next?"**
> Three things. Scanned PDFs are currently skipped — adding OCR would cover
> them. Everything is single-instance, so the background jobs would need a real
> queue to scale out. And a few thresholds are set conservatively because I
> haven't yet measured how much the numbers naturally vary run to run.

**"Did you use AI to build it?"**
> Yes, as a pair programmer — and the interesting part is what that *didn't*
> catch. Several real bugs only surfaced when I ran the system and looked at
> actual output: a ranking metric producing a mathematically impossible value,
> a first-request delay of 13 seconds caused by lazy model loading, a check for
> the number "30" that was happily satisfied by "300". Tests confirm the code
> does what you wrote. Only output tells you whether what you wrote was right.

---

## A 60-second spoken version

> "I built a question-answering system over private PDF documents — internal
> engineering docs, product manuals, that kind of thing.
>
> The core problem with these systems is that AI models sound equally confident
> whether they're right or making things up. So the design goal wasn't just
> getting answers, it was making them *verifiable*.
>
> Every answer cites the exact page it came from, and the system then checks
> its own citations — that the sources exist, that every claim has one, and that
> the cited page actually says what the answer claims. It refuses outright when
> the documents don't cover the question.
>
> Under the hood it searches two ways at once, keyword and semantic, because
> they fail in opposite directions, then re-ranks the results with a more
> accurate model.
>
> The part I'm most pleased with is the evaluation. There's a scored question
> set with known answers, and if a code change makes retrieval or citation
> accuracy worse, the build fails automatically. That's the difference between
> a demo and something you'd let people rely on."

---

## Glossary

| Term | Plain meaning |
|---|---|
| **RAG** | Retrieval-Augmented Generation — find relevant documents first, then let the AI answer using only those |
| **Chunk** | A small passage of a document, sized so the AI can read several at once |
| **Embedding** | A list of numbers representing a text's meaning; similar meanings get similar numbers |
| **Vector database** | Storage that finds text with similar meaning, rather than matching words |
| **BM25** | A classic keyword-search algorithm — the kind of thing search engines used before AI |
| **Re-ranking** | A second, slower, more accurate pass over the top results |
| **Hallucination** | An AI stating something confidently that isn't true |
| **Groundedness** | Whether an answer is actually supported by the documents it cites |
| **CI** | Continuous Integration — automated checks that run on every code change |
