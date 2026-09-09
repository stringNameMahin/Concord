# Concord

**A fact knowledge layer.** It reads PDFs, extracts facts that round-trip to
bytes in their source, and decides whether two facts corroborate each other,
genuinely contradict each other, or only *look* like they disagree because they
were measured under different conditions.

The graded object here is not the fact list; it is the **adjudicated pair**.

---

## Setup and Run Instructions

### Requirements

- **Python 3.12 or 3.13.** Pinned in `pyproject.toml`. `sentence-transformers`
  pulls `torch`, whose wheels are unreliable on 3.14.
- ~1.5 GB of disk for dependencies (`torch` is most of it) plus a ~130 MB
  sentence-transformer downloaded on first use.
- No GPU. No Docker. No database server.
- **No API key is needed to evaluate this project.** See *Offline evaluation*
  below.

### Clone and install

```bash
git clone https://github.com/stringNameMahin/Concord.git concord
cd concord
```

**With uv (recommended; this is what the project was built with):**

```bash
uv sync --extra dev          # creates .venv on Python 3.12 from uv.lock
```

**With pip:**

```bash
python3.12 -m venv .venv
# Windows:            .venv\Scripts\activate
# macOS / Linux:      source .venv/bin/activate
pip install -e ".[dev]"
```

Put the starter PDFs where the instructions below expect them (any path works;
these are just the paths used throughout the docs):

```
starter-datasets/delhivery/*.pdf
starter-datasets/india-macroeconomy/*.pdf
```

### Check the install

```bash
uv run pytest -q            # or: .venv/bin/pytest -q
```

Expected: **366 passed** in roughly 45 seconds. The suite needs no network and
no API key; the one test that loads the real embedding model is marked `slow`
and will download the sentence-transformer if it is not already cached.

You can run a keyless demo using the data cached from the 6 documents upload initially from the starter datasets. Simply start the application and you may browsed through the previously cached datasets.

> Do **not** set `CONCORD_OFFLINE=1` when running the tests. They stub their
> own LLM clients, and the flag only confuses the failure messages.

### Configure


**HOW TO GET API KEYs TO RUN AN EXAMPLE FOR NEW DOC**
```
1. Open Google AI Studio : https://aistudio.google.com
2. Create a Project (1 Key per project - for limits)
3. Create an API Key in that project
4. Paste it in .env for example GEMINI_API_KEY = <API KEY>

If you need to process a large document (e.g the eternal report from video (409 pages) you might need multiple free keys or 1 paid key)

Free way:
Create multiple projects in Google AI Studio which lets you create 1 key per project.
Then in .env do GEMINI_API_KEYS=<Key1>,<Key2>,<Key3>...

1 key is enough for smaller docs, for bigger docs I recommend using the multi key system so that it auto switches the key when one key is exhausted.
```

```bash
cp .env.example .env
```

Everything in `.env.example` is optional and commented. `.env` is gitignored;
**no credential of any kind is committed to this repository**. The only two
key names the code reads are `GEMINI_API_KEY` / `GEMINI_API_KEYS` and
`OPENROUTER_API_KEY` / `OPENROUTER_API_KEYS`, and both are blank in the
template. Verify for yourself:

```bash
git ls-files | grep -i env      # → .env.example, and nothing else
```

`data/db/concord.sqlite` and `data/cache/llm/**` are committed on purpose: they
are the evaluation path, and they contain document text and model responses,
not secrets.

### Run

```bash
uv run uvicorn concord.api.app:app --port 8000
# Windows without uv: .venv\Scripts\python.exe -m uvicorn concord.api.app:app --port 8000
```

Open <http://127.0.0.1:8000>. Four views, and a drop target at the top:

| View | What it shows |
| --- | --- |
| **Relations** | Every adjudicated pair except `unrelated`, with both facts side by side, the rule that fired, the qualifier that decided it, and click-through evidence highlighting into the source text |
| **Fact ledger** | All grounded facts, searchable, each with page, char span and verbatim quote |
| **Quarantine** | Facts whose quote could not be located in the passage that claimed it (kept, counted, never adjudicated) |
| **Schema timeline** | The predicate vocabulary the corpus taught the layer, and when it learned each part |

Configuration is read at import time, so **restart the server after changing an
environment variable.**

### Ingest a new PDF

**Through the UI:** drag a PDF onto the drop target at the top of the page, or
click *choose a file*. The response reports facts, quarantine rate, blocking
counts, verdicts and schema growth. The new document's facts are blocked
against everything already in the ledger, so relations are cross-document from
the second upload onward.

**Through the API:**

```bash
curl -F "file=@path/to/document.pdf" http://127.0.0.1:8000/ingest
```

| Endpoint | Purpose |
| --- | --- |
| `POST /ingest` | Upload a PDF: parse → extract → ground → block → decide → adjudicate → store |
| `GET /facts` | The ledger. `?doc_id=`, `?predicate=`, `?q=`, `?quarantined=true`, `?limit=`, `?offset=` |
| `GET /relations` | Adjudicated pairs. `?verdict=`, `?cross_document=true`. Hides `unrelated` unless asked for by name |
| `GET /evidence/{fact_id}` | The quote in its surrounding text, with offsets for highlighting |
| `GET /verify/{fact_id}` | Re-cuts `text[char_start:char_end]` from the source and compares it to the stored quote |
| `GET /schema` | Predicate registry and the schema-event timeline |
| `GET /stats` | Counts, quarantine rate, verdict mix, per-document breakdown |
| `GET /health` | Liveness plus whether offline mode is on |

Ingesting a PDF that is **not already in the response cache** needs an API key.
Set `GEMINI_API_KEY` in `.env` and restart; a Google AI Studio free-tier key is
enough; a 100-page PDF is about 7 requests at the shipped batch size. Without
one, `/ingest` answers **HTTP 503** naming the missing key rather than silently
returning zero facts, and says the committed ledger is still browsable.

Keep experiments off the shipped ledger because one careless upload overwrites it:

```bash
# Windows PowerShell
$env:CONCORD_DB   = "data/db/scratch.sqlite"
$env:CONCORD_WORK = "data/work-scratch"
```

### Offline evaluation: the whole system, with no API key

`data/cache/llm/` holds **299 content-addressed LLM responses** covering every
extraction, alias question and adjudication for the six starter documents. That
set is exactly sufficient and I verified it rather than assuming: replaying the
whole corpus against a copy of the cache containing only the git-tracked files
produced byte-identical results: 41 requests, **0 cache misses, 0 live
calls**.

`CONCORD_OFFLINE=1` refuses every network call and serves from that cache only;
a request with no cached response raises `MissingFromCache` instead of being
guessed at.

```bash
# Windows PowerShell
$env:CONCORD_OFFLINE = "1"
.venv\Scripts\python.exe -m uvicorn concord.api.app:app --port 8000

# macOS / Linux
CONCORD_OFFLINE=1 uv run uvicorn concord.api.app:app --port 8000
```

Two things a reviewer can do from here, both at zero cost:

1. **Browse and byte-verify the shipped ledger.** Every fact in the Fact
   ledger view links to its evidence; `GET /verify/{fact_id}` re-reads the
   source text and proves the offsets still cut the stored quote.

2. **Re-derive the entire corpus from scratch.** Drop all six starter PDFs onto
   the page in this order:

   ```
   delhivery/03-delhivery-q4-fy24-earnings-presentation.pdf
   delhivery/02-delhivery-annual-report-fy24-excerpt.pdf
   delhivery/01-delhivery-prospectus-2022-excerpt.pdf
   india-macroeconomy/01-india-economic-survey-2024-25-excerpt.pdf
   india-macroeconomy/02-rbi-annual-report-2024-25-excerpt.pdf
   india-macroeconomy/03-imf-india-2025-article-iv-excerpt.pdf
   ```

   Measured: **56.1 seconds, 41 extraction requests, 0 live API calls**, giving
   488 grounded facts, 5 quarantined and 121 relations.

   **The shipped `.sqlite` now contains this replay, including Case 1's
   corroboration.** You can evaluate the four cases directly.
   `docs/EVIDENCE.md` opens with the earlier batch-size mismatch and the
   arithmetic behind its repair.

---

## Video Demo

**Link:** : https://drive.google.com/file/d/10zHrwhABFf-fdw-qt_kofQyPxU6nmgJW/view?usp=sharing

---

## Approach

### The idea, in three sentences

A fact splits into exactly two keys: a **comparison key** (subject identity
plus canonical predicate), which answers *"are these two facts even about the
same thing?"*, and a **context key** (an open bag of qualifiers, each carrying
whether it was *stated*, *inherited* from the enclosing page or section, or
*absent*), which answers *"under what conditions?"*. Verdicts then follow from a
decision table over those two keys rather than from a judgement call. So
"apparent contradiction explained by context" stops being a reasoning problem
and becomes a **set difference over two qualifier bags**, rendered as a
sentence.

That is the whole design. Everything below is in service of it.

### Architecture

```
PDF
 │
 ├─ parse/pdf.py         PyMuPDF → ONE canonical string + page/line index.
 │                       Every offset in the system points into this string;
 │                       nothing downstream reopens the PDF.
 │
 ├─ parse/structure.py   Where context LIVES: running page furniture found by
 │                       repetition, headings found by typography. It never
 │                       decides what the context MEANS; that is the model's
 │                       job, and that split is the generalisation argument.
 │
 ├─ parse/chunk.py       Overlapping windows, each carrying its page frame and
 │                       heading path as a context block.
 │
 ├─ extract/             16 passages per request. The model returns a verbatim
 │   prompt · runner     `quote` and a `passage_id`, but NEVER an offset;
 │   align · models      `FactOut` has no field for one.
 │                       Aligner.locate(strict=True) finds the quote inside
 │                       that passage or the fact is QUARANTINED.
 │
 ├─ normalize/           numbers.py: Indian digit grouping, lakh/crore/mn/bn,
 │   numbers · periods   parenthesised negatives, bounds, currency, percent →
 │                       a base-unit value and a PRECISION INTERVAL.
 │                       periods.py: FY24 / FY 2023-24 / "nine months ended
 │                       December 31, 2021" → intervals, fiscal basis read
 │                       out of the document rather than assumed.
 │
 ├─ facts.py             The Fact record. comparison_key and the qualifier bag.
 │
 ├─ registry.py          Emergent schema: a new predicate is embedded and
 │                       cosine-searched; below 0.86 it is simply new and costs
 │                       nothing; above it, ONE yes/no question. Nothing merges
 │                       without positive confirmation.
 │
 ├─ compare/block.py     Three strategies, unioned, tuned for recall:
 │                       exact comparison key · top-10 cosine neighbours ·
 │                       log-magnitude bucket on the NORMALISED value.
 │
 ├─ compare/decide.py    THE DECISION TABLE. Zero LLM calls. Every verdict in
 │                       the system is produced here.
 │
 ├─ compare/adjudicate.py + judge.py
 │                       Only the residue. Three guards between the model's
 │                       answer and the ledger.
 │
 └─ store/ + api/        SQLite, five tables. FastAPI, one static page.
```

### The decision table

| Condition (all deterministic) | Verdict | `rule_fired` | LLM |
| --- | --- | --- | --- |
| Comparison keys differ | `unrelated` | `comparison_key_differs` | no |
| Values incomparable: unit clash, percent vs absolute, unparsed figure | `insufficient_context` | `incomparable_values` | no |
| Values agree, a **recognised** condition differs | `unrelated` | `context_differs_values_agree` | no |
| Values agree, bags compatible | `corroborates` | `keys_match_intervals_overlap` | no |
| Values disagree, a condition is stated on one side and **absent** on the other | `insufficient_context` | `missing_qualifier_guard` | no |
| Values disagree, a stated condition differs | `reconciled_by_context` | `discriminating_qualifier_differs` | prose only |
| Values disagree, both sides state the **same** conditions | `contradicts` | `same_context_disjoint_intervals` | yes, adjudicates |

Two guards are enforced in Python, not asked of a prompt:

- **Precision intervals, never point values.** `8,142 Cr` committed to whole
  crore, so it stands for `[8141.5, 8142.5]` crore. `81,415.38 mn` committed to
  two decimals of a million. Those intervals overlap, so the figures agree,
  with no tolerance constant anyone has to defend. `intervals_overlap` is
  **strict** on both sides, because a closed test made `8,142 Cr` and
  `8,143 Cr` share the endpoint `8142.5` and compare as agreeing, which silently
  disabled contradiction detection for every pair of consecutive figures written
  to the same precision.
- **The missing-qualifier guard**, binding on *disagreement only*. If a
  condition is stated on one side and absent on the other, the ceiling is
  `insufficient_context` and the response names the missing key. When the values
  *agree*, a qualifier the other document never stated cannot turn agreement
  into a conflict. Applied unconditionally, the guard would abstain on almost
  every cross-document pair, and Case 1 would be unreachable.

### The deterministic / LLM split

The model has exactly four jobs: extraction, confirming a predicate alias,
adjudicating the residue of genuine value conflicts, and writing explanations.

**It never decides whether two numbers are equal, never converts a unit, and
never compares two dates.** FinVerBench measures models falling from ~95% on
simple financial lookups to near zero on multivariate calculation and,
crucially for the architecture, being unable to reliably detect their own
arithmetic errors. A system that cannot detect its own arithmetic errors cannot
be audited. So every scale conversion, tolerance check, interval comparison and
date comparison is Python.

Measured on the six-document corpus:

```
118,828 theoretical pairs
  →  3,804 candidates after blocking          (96.80% reduction)
  →  3,725 finished by a deterministic rule   (97.9% of candidates)
  →     79 pairs reached a model              (0.066% of the theoretical space)
       of which 68 were prose-only, on verdicts already decided
```

When a model *is* asked, three guards stand between its answer and the ledger:
a verdict must **name** the qualifier that drove it and that key must exist on
one of the two facts, or the verdict is rejected; the pair is judged in **both
presentation orders** and disagreement downgrades to `insufficient_context`; and
a failed or uncached call is **not a verdict**; the deterministic verdict
stands and `unadjudicated` increments. Nothing is ever guessed, which is also
why the whole pipeline runs with no key at all.

### Storage

**SQLite, five tables** (`documents`, `facts`, `relations`,
`predicate_registry`, `schema_events`) with JSON columns. It ships as a 1.5 MB
file a reviewer opens with `sqlite3` and no Docker. The canonical document text
lives *beside* the database as `data/work/<doc_id>.txt` rather than inside it,
so an offset can be checked by hand and `/verify` can re-read the bytes fresh.

### Rejected alternatives

| Rejected | Why |
| --- | --- |
| **Neo4j / any graph DB** | The brief warns against it; it adds a Docker dependency; and the interesting object here is the adjudicated pair *together with its explanation*, which is a row, not an edge |
| **A vector database** | At 488 facts the full cosine matrix is a few megabytes and the search is milliseconds. `numpy` brute force is the entire retrieval layer |
| **LangChain / LangGraph** | Four LLM call sites needing strict JSON is `httpx` + `pydantic` + a retry decorator. A framework adds install weight and a layer of indirection between a reviewer and the logic, which the brief makes *actively* costly by preferring a smaller understandable prototype |
| **Multi-agent negotiation** (extractor, critic, reconciler, arbiter) | The core insight is that most adjudication is deterministic. A state machine around a straight line whose interesting logic is a decision table would obscure exactly the thing being graded |
| **A fixed typed schema** `(metric, value, unit, period, entity)` | Dies on contracts, rosters, addresses and policy statements. The starter corpus alone needed the qualifier keys `auditor`, `condition`, `category` and `service` to reach correct verdicts |
| **Free-text claim + embedding similarity** | "Revenue was ₹8,142 Cr" and "Revenue was ₹7,224 Cr" are near-identical in embedding space. Nothing distinguishes a different period from a disagreement |
| **Docling for parsing** | Bake-off on the A3 two-up annual report: PyMuPDF 0.38 s, Docling no output after 25 minutes. Latency that far apart is not a trade-off worth measuring further |
| **LlamaParse / Unstructured Serverless** | Cloud APIs, which fails "a reviewer must be able to evaluate without your account" |
| **Growing the discriminating-key allowlist** | Every unseen document family would need new entries, which is the hard-coding the brief prohibits |
| **Merging predicates on similarity alone** | At cosine 0.91 the encoder called `standalone_revenue_from_operations` and `revenue_from_operations` the same relation. They are not. A silent merge is unrecoverable downstream |

### AI tools used, and what was delegated

- **Claude Code**: Big help under a time constraint, helped generate codes quicker than manually typing through it all.
- **Extraction and adjudication model:** `gemini-3.1-flash-lite` over plain
  `httpx` with pydantic validation and a content-addressed disk cache. An
  OpenRouter provider sits behind the same seam as a per-token fallback. An open
  30B model was trialled for extraction and **reverted**; it produced more
  facts (625 against 488 across the corpus, 51 against 16 on the earnings deck)
  and folded conditions into predicate names (`consolidated_revenue`,
  `standalone_revenue`), which collapsed cross-document relations from 22 to 0
  and left just 2 predicates shared across 491. More facts is not better extraction; the metric that
  matters is whether two documents describing the same thing produce the same
  comparison key.
- **Local embeddings:** `BAAI/bge-small-en-v1.5` via `sentence-transformers`, on
  CPU. It is deliberately local, so the only credential in the system is the
  LLM key. The committed cache removes that from the evaluation path too. Embeddings
  never decide anything; they widen the candidate set that the decision table
  then judges.

---

## Limitations and Next Steps

### The numbers, from the offline replay of all six starter PDFs

```
Documents                     6            511 pages, 1,658,944 chars, 602 chunks
Parse + structure + chunk     2.45 s
Extraction requests           41           all served from data/cache/llm/
Live API requests             0
Wall clock, all six           56.1 s       includes one cold embedding-model load

Facts extracted               501
  grounded                    496          99.0%
  quarantined                 5            1.0%
  duplicates collapsed        8
  in the ledger               488          477 located exactly, 11 fuzzily

Theoretical pairs             118,828      = C(488, 2)
Candidates after blocking     3,804        96.80% reduction
  comparison-key block        79
  value-anchored block        207
  semantic block              3,741
Blocking + decision table     9.38 s
Finished deterministically    3,725
Reached the LLM               79           90 calls: 68 prose-only, 11 adjudicated

Relations stored              121          cross-document 34
  reconciled_by_context       70
  insufficient_context        35
  corroborates                11
  contradicts                 5
  unrelated                   3,683        counted, not stored

Hallucinated-qualifier rejection   4 of 90 verdicts   4.4%
Judge self-consistency             7 of 8 pairs       87.5%
Unadjudicated                      0

Predicates registered         297          89 questions asked, 32 confirmed, 57 refused
Tests                         366 passed in 43.67 s
```

**Cost.** The full corpus replays for **$0.00**; it is all cache. Extracting it
the first time was 41 requests on a Google AI Studio **free-tier** key, so the
Gemini spend to date is also $0.00. The only money spent on this project was
**$0.1933** on OpenRouter, during the open-model extraction trial that was
subsequently reverted.

### What does not work

1. **Three of the five `contradicts` verdicts are false positives**, and the
   system's own explanation text says so while its verdict field says the
   opposite. `subjects_match` accepts one surface token set as a subset of the
   other so that `Delhivery` matches `Delhivery Limited` without a
   corporate-suffix list, but `{delhivery, limited}` is also a subset of
   `{delhivery, freight, services, private, limited}`, so a parent company
   matches every subsidiary whose name contains its own. Every guard in the
   system operates on qualifiers; the subject has none.

2. **Qualifier recall is the real ceiling, and it is what breaks Case 2.**
   `active_customers` 7,900 against 23,113 is called a contradiction, and the
   reconciling phrase is sitting inside *both* quotes: "Together with Spoton"
   against "(excluding those serviced by Spoton)". The extractor never emitted
   it as a `scope` qualifier, so the diff had nothing to find and the judge,
   which is shown only the qualifier lists, had nothing to name.

3. **Extraction recall is thin, and batching is most of the reason.** The
   build's own log recorded 35 facts on the earnings deck under a stronger model
   at batch 8 against 16 under `flash-lite` at batch 16, and noted that model
   and batch had changed together so the causes were unseparated. Replaying the
   deck from cache at both batch sizes on the **same** model separates them:

   | Model | Batch | Extracted | Grounded | Quarantined |
   | --- | ---: | ---: | ---: | ---: |
   | `gemini-3.1-flash-lite` | 8 | 30 | 27 | 3 (10.0%) |
   | `gemini-3.1-flash-lite` | 16 | 16 | 16 | 0 |
   | `gemini-3.6-flash` (logged earlier) | 8 | 35 | N/A | N/A |

   Batch size is the dominant cause, not the model: 30 against 16 on one model,
   where the model change is worth 5. Doubling the passages per request roughly
   halves recall but buys a 0% quarantine rate against 10%, so the deeper
   batch also produces cleaner quotes. That is a live trade-off the shipped
   default sits on the wrong side of for recall, and the right side of for
   request count.

   Either way, recall is genuinely thin. The Gurugram PIN discrepancy the brief
   nominated as a contradiction case (`122002` in three places against `122001`
   in the BRSR section) is in the source, and sits in the same chunk as a fact
   that *was* extracted, yet neither PIN came back.

4. **Abstention is now the failure mode, by design.**
   `insufficient_context` is 35 of 121 relations. A pair whose one-sided
   qualifier is genuinely incidental abstains instead of resolving. Contradiction
   over-calling is the documented failure of this task, so this is the right
   direction to err, but it is a real cost.

5. **The semantic block carries the run.** All 121 stored relations were
   proposed by it; 37 by nothing else, including 4 of the 5 contradictions. It
   is also the least precise strategy by an order of magnitude. The honest
   description of the blocking layer is "one recall device and two cheap
   accelerators".

6. **A reviewer with no key cannot ingest a genuinely new PDF.** Everything
   already in the cache replays; anything else needs a key. This is the one
   honest gap in the keyless path.

7. **Embeddings are recomputed for every fact on every ingest.** The
   `facts.embedding` BLOB column exists and is not read back. Two seconds at 488
   facts, prohibitive at 50,000.

8. **Smaller, real, and unfixed:** no UI pagination (`limit` is fixed at
   200–300; fine at 488 facts, broken at 5,000); no true table cell grid, so
   row/column coordinates are not recorded; OCR and scanned PDFs are out of
   scope, though an unreadable or password-protected file is answered as a 400
   with a reason rather than a 500.

### Next steps, in the order the failures above dictate

1. **To fix subject identity (fixes failure 1).** Refuse a subset match when the
   larger surface contains a token that also heads another known subject in the
   same document, using structure rather than vocabulary. Then propagate a hard key found
   anywhere in a document to every fact whose surface matches it, so the CIN on
   page 0 of the prospectus arms `subjects_match` for all 98 of its facts.
2. **Attack qualifier recall (fixes 2, moves mass out of 4).** A second
   extraction pass over facts that reached a verdict, asking only "what
   conditions does this quote state?", would have found `scope` on both
   `active_customers` facts. This moves more mass out of `insufficient_context`
   than any new feature would.
3. **Follow the batch-size finding (acts on 3).** The A/B above says batch 8
   nearly doubles recall on the deck and triples the quarantine rate. Run the
   same comparison across all six documents, then pick the batch size on
   measured recall rather than on the request cap that set it, then re-extract
   the corpus once at whichever wins.
4. **A discontinuous-span aligner (recovers part of the quarantine).** Accept a
   quote as *k* ordered fragments inside one passage. The stitched table cell
   (`"FY23\n48 days"` from `"Particulars FY24 FY23 ... 45 days 48 days"`) is
   recoverable without weakening the grounding guarantee, because the bytes are
   still the document's.
5. **Ablate the semantic block (settles 5).** Rerun at `k` = 5, 10, 20 and plot
   relations recovered against candidates proposed; record the cosine on each
   candidate so a similarity floor can finally be argued from evidence.
6. **A local-LLM provider behind the existing seam (closes 6).** Roughly an
   hour: the provider interface is three methods.
7. **Cache the embeddings (closes 7)**, then paginate the UI (8).

---

## Additional Notes

- **[`docs/EVIDENCE.md`](docs/EVIDENCE.md)**: the graded artifact. A one-page
  pipeline walkthrough, then the four required cases as the system actually
  produced them: real `Fact` records with subject, predicate, normalised value,
  qualifier bag and evidence span; the verbatim source quote with document, page
  and char offsets; which blocking strategy surfaced each pair; which decision
  rule fired and the qualifier diff or interval comparison behind it; and the
  system's own explanation. Then the five evaluation criteria answered with code
  and measurements, and the four optional extensions answered only as far as the
  code supports them.

- **[`docs/decisions.md`](docs/decisions.md)**: 36 dated entries, each naming
  the options rejected and why. This is where the trade-offs actually live: the
  parser bake-off, why the discriminating-key allowlist was the wrong shape,
  why precision intervals are half-open, why an open model that extracted twice
  as much was reverted, why a spending ceiling is a reservation and not a count.

- **[`docs/devRead.md`](docs/devRead.md)**: the original phase plan, scope
  decisions and explicit non-goals. **[`docs/status.md`](docs/status.md)**: the
  live build log, including every bug found and fixed.

- **Cost of a full run: $0.00 from cache.** The corpus was extracted on a
  free-tier key across 41 requests; the only money spent on the project was
  $0.1933 on OpenRouter for an extraction experiment that was reverted.
  `CONCORD_MAX_CALLS` (default 200) is a hard ceiling on live requests, claimed
  under a lock *before* each request is sent so it holds across the thread pool.

- **Offline evaluation path:** `CONCORD_OFFLINE=1`, then either browse the
  shipped ledger or re-derive the whole corpus by dropping the six starter PDFs
  onto the page in 56 seconds with zero live requests. See *Setup and Run
  Instructions* above. The rebuild note at the top of `docs/EVIDENCE.md`
  records the earlier batch-size mismatch and its repair.
