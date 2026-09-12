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

Expected: **454 passed** in roughly 20 seconds. The suite needs no network and
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
git ls-files | grep -i env      # -> .env.example, and nothing else
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
| `POST /ingest` | Upload a PDF: parse -> extract -> ground -> block -> decide -> adjudicate -> store |
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

### Offline evaluation - the whole system, with no API key

`data/cache/llm/` holds **483 content-addressed LLM responses** covering every
extraction, alias question and adjudication for the six starter documents.
**The whole corpus replays with 0 cache misses and 0 live calls.**

It briefly did not. The correctness pass changed six judge prompts - the cache
is keyed on the whole request payload, and the prompt carries both qualifier
bags, so a pair that gained an inherited `segment` no longer matched its stored
response. Offline those six counted as `unadjudicated` and kept their
deterministic explanation, which is the documented behaviour: **failure is not
a verdict.** Six live calls regenerated them and wrote them back to the cache;
no verdict moved, because a `reconciled_by_context` call is prose-only and
cannot change one.

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

   Measured: **41 extraction requests, 0 live API calls, 0 cache misses**,
   giving 488 grounded facts, 5 quarantined and **118 relations**.

   **The shipped `.sqlite` contains this replay**, plus one further document -
   a 409-page annual report from a company outside the starter set, ingested to
   test that the fixes below generalise. The file therefore holds 7 documents,
   773 grounded facts, 12 quarantined and 174 relations; the six starters
   account for 488 / 5 / 118 of those, and every per-corpus number in this
   README is the six-starter figure unless it says otherwise.

---

## Video Demo

**Link:** : https://drive.google.com/file/d/10zHrwhABFf-fdw-qt_kofQyPxU6nmgJW/view?usp=sharing

---

## Approach

### The idea, in three sentences

A fact splits into exactly two keys: a **comparison key** - subject identity
plus canonical predicate - which answers *"are these two facts even about the
same thing?"*, and a **context key** - an open bag of qualifiers, each carrying
whether it was *stated*, *inherited* from the enclosing page or section, or
*absent* - which answers *"under what conditions?"*. Verdicts then follow from a
decision table over those two keys rather than from a judgement call. So
"apparent contradiction explained by context" stops being a reasoning problem
and becomes a **set difference over two qualifier bags**, rendered as a
sentence.

That is the whole design. Everything below is in service of it.

### Architecture

```
PDF
 |
 +- parse/pdf.py         PyMuPDF -> ONE canonical string + page/line index.
 |                       Every offset in the system points into this string;
 |                       nothing downstream reopens the PDF.
 |
 +- parse/structure.py   Where context LIVES: running page furniture found by
 |                       repetition, headings found by typography, and which
 |                       of those headings a BULLET scopes. It never decides
 |                       what a heading MEANS - that is the model's job, and
 |                       that split is the generalisation argument. The one
 |                       exception is `inherited_qualifiers`, which resolves
 |                       the bullet enclosing a fact's own offset into a
 |                       `segment` qualifier marked `inherited`.
 |
 +- parse/chunk.py       Overlapping windows, each carrying its page frame and
 |                       heading path as a context block.
 |
 +- extract/             16 passages per request. The model returns a verbatim
 |   prompt | runner     `quote` and a `passage_id` - and NEVER an offset;
 |   align | models      `FactOut` has no field for one.
 |                       Aligner.locate(strict=True) finds the quote inside
 |                       that passage or the fact is QUARANTINED.
 |
 +- normalize/           numbers.py: Indian digit grouping, lakh/crore/mn/bn,
 |   numbers | periods   parenthesised negatives, bounds, currency, percent ->
 |                       a base-unit value and a PRECISION INTERVAL.
 |                       periods.py: FY24 / FY 2023-24 / "nine months ended
 |                       December 31, 2021" / "first eight months of FY25" ->
 |                       intervals, fiscal basis read out of the document
 |                       rather than assumed. A label that names less than a
 |                       whole year never widens into one.
 |
 +- facts.py             The Fact record. comparison_key and the qualifier bag.
 |
 +- registry.py          Emergent schema: a new predicate is embedded and
 |                       cosine-searched; below 0.86 it is simply new and costs
 |                       nothing; above it, ONE yes/no question. Nothing merges
 |                       without positive confirmation.
 |
 +- compare/block.py     Three strategies, unioned, tuned for recall:
 |                       exact comparison key | top-10 cosine neighbours |
 |                       log-magnitude bucket on the NORMALISED value.
 |
 +- compare/partition.py Which facts are complementary categories of one
 |                       distribution rather than rival claims about it -
 |                       decided by whether their shares add up to a whole.
 |
 +- compare/decide.py    THE DECISION TABLE. Zero LLM calls. Every verdict in
 |                       the system is produced here.
 |
 +- compare/adjudicate.py + judge.py
 |                       Only the residue. Three guards between the model's
 |                       answer and the ledger.
 |
 +- store/ + api/        SQLite, five tables. FastAPI, one static page.
```

### The decision table

| Condition (all deterministic) | Verdict | `rule_fired` | LLM |
| --- | --- | --- | --- |
| Comparison keys differ | `unrelated` | `comparison_key_differs` | no |
| Keys match, the two facts are different categories of one distribution | `unrelated` | `complementary_categories` | no |
| Values incomparable - unit clash, percent vs absolute, unparsed figure | `insufficient_context` | `incomparable_values` | no |
| Values agree, a **recognised** condition differs | `unrelated` | `context_differs_values_agree` | no |
| Values agree, bags compatible | `corroborates` | `keys_match_intervals_overlap` | no |
| Values disagree, a condition is stated on one side and **absent** on the other | `insufficient_context` | `missing_qualifier_guard` | no |
| Values disagree, a stated condition differs | `reconciled_by_context` | `discriminating_qualifier_differs` | prose only |
| Values disagree, both sides state the **same** conditions | `contradicts` | `same_context_disjoint_intervals` | yes, adjudicates |

Three guards are enforced in Python, not asked of a prompt:

- **Subject identity before anything else.** Two hard keys (CIN, DIN, ISIN)
  must be *equal*; without one on both sides the normalised surfaces must be
  the same name, up to a single trailing corporate legal form, so `Delhivery`
  still matches `Delhivery Limited` while `Delhivery Limited` no longer matches
  `Delhivery Freight Services Private Limited`. Nothing about a pair's values or
  qualifiers is looked at until the subject test passes. This replaced a
  token-*subset* rule; see Limitations for what that rule cost.
- **Precision intervals, never point values.** `8,142 Cr` committed to whole
  crore, so it stands for `[8141.5, 8142.5]` crore. `81,415.38 mn` committed to
  two decimals of a million. Those intervals overlap, so the figures agree -
  with no tolerance constant anyone has to defend. `intervals_overlap` is
  **strict** on both sides, because a closed test made `8,142 Cr` and
  `8,143 Cr` share the endpoint `8142.5` and compare as agreeing, which silently
  disabled contradiction detection for every pair of consecutive figures written
  to the same precision.
- **The missing-qualifier guard**, binding on *disagreement only*. If a
  condition is stated on one side and absent on the other, the ceiling is
  `insufficient_context` and the response names the missing key. When the values
  *agree*, a qualifier the other document never stated cannot turn agreement
  into a conflict - applied unconditionally the guard would abstain on almost
  every cross-document pair, and Case 1 would be unreachable.
- **Complementary categories never reach the value comparison.** A breakdown
  table states one predicate over mutually exclusive categories, so every row
  disagrees with every other row by construction and every row is correct.
  `concord/compare/partition.py` groups facts that share a subject, a predicate
  and every qualifier but one, and asks whether the values under that qualifier
  add up to a whole - decided by the same precision intervals, so three integer
  percentages of 63, 15 and 22 span `[98.5, 101.5]` and qualify while four
  training-coverage percentages of 97.67, 100.00, 96.32 and 100.00 do not.
  Pairs inside such a group are `unrelated`: they divide a total rather than
  making rival claims about it, and running them through the reconciliation
  path produced text that read like an explanation of a conflict that never
  existed.

### The deterministic / LLM split

The model has exactly four jobs: extraction, confirming a predicate alias,
adjudicating the residue of genuine value conflicts, and writing explanations.

**It never decides whether two numbers are equal, never converts a unit, and
never compares two dates.** FinVerBench measures models falling from ~95% on
simple financial lookups to near zero on multivariate calculation, and - the
part that decides the architecture - unable to reliably detect their own
arithmetic errors. A system that cannot detect its own arithmetic errors cannot
be audited. So every scale conversion, tolerance check, interval comparison and
date comparison is Python.

Measured on the six-document corpus:

```
118,828 theoretical pairs
  ->  3,670 candidates after blocking          (96.91% reduction)
  ->  3,591 finished by a deterministic rule   (97.8% of candidates)
  ->     79 pairs reached a model              (0.066% of the theoretical space)
       of which 75 were prose-only, on verdicts already decided
  ->      1 verdict in the ledger is the model's          (was 7)
```

That last line is the one that moved, and it has moved twice. Before the
correctness pass, seven relations carried `decided_by: "llm"` - the model had
settled them, including five `contradicts`. Every one of those five turned out
to rest on a defect in the deterministic layer: three on a subject-identity rule
that matched a parent to its subsidiaries, one on a predicate alias that merged
two different FDI series, one on a qualifier the context stack never propagated.
Fixing those took it to zero.

It then went back up, and the reason is the useful part. Adding a 409-page
annual report put five verdicts back in the model's hands, and four of them were
the same defect wearing a new hat: the report states its figures twice, once
consolidated and once standalone, and nothing promoted that distinction to a
qualifier. Two identical bags, disjoint values, so the table proposed
`contradicts` and the model was left guessing - it replied that one side was
"possibly consolidated" and the distinction "not explicitly stated", about a
phrase 230 characters away in a running page header. Promoting it (see
`CONSOLIDATION` in `parse/structure.py`) returned those four to the
deterministic layer as `reconciled_by_context` on `consolidation`.

**One verdict in the shipped ledger is the model's**, and it is the one that
should be: two publishers giving India's FY2024-25 real GDP growth as 6.4 and
6.5 for the same period on the same basis. No qualifier reconciles that, and
calling it a contradiction is a judgment rather than a rule. Every other verdict
in the file is deterministic and the model only writes prose.

The pattern is worth naming because it has now happened three times: **a prompt
rule with no deterministic backstop.** The scale, the subject and the
consolidation basis were each left to the extractor to volunteer, and each
produced silently wrong output until a guard read it back out of the bytes.

When a model *is* asked, three guards stand between its answer and the ledger:
a verdict must **name** the qualifier that drove it and that key must exist on
one of the two facts, or the verdict is rejected; the pair is judged in **both
presentation orders** and disagreement downgrades to `insufficient_context`; and
a failed or uncached call is **not a verdict** - the deterministic verdict
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
| **LangChain / LangGraph** | Four LLM call sites needing strict JSON is `httpx` + `pydantic` + a retry decorator. A framework adds install weight and a layer of indirection between a reviewer and the logic - which the brief makes *actively* costly by preferring a smaller understandable prototype |
| **Multi-agent negotiation** (extractor, critic, reconciler, arbiter) | The core insight is that most adjudication is deterministic. A state machine around a straight line whose interesting logic is a decision table would obscure exactly the thing being graded |
| **A fixed typed schema** `(metric, value, unit, period, entity)` | Dies on contracts, rosters, addresses and policy statements. The starter corpus alone needed the qualifier keys `auditor`, `condition`, `category` and `service` to reach correct verdicts |
| **Free-text claim + embedding similarity** | "Revenue was ₹8,142 Cr" and "Revenue was ₹7,224 Cr" are near-identical in embedding space. Nothing distinguishes a different period from a disagreement |
| **Docling for parsing** | Bake-off on the A3 two-up annual report: PyMuPDF 0.38 s, Docling no output after 25 minutes. Latency that far apart is not a trade-off worth measuring further |
| **LlamaParse / Unstructured Serverless** | Cloud APIs, which fails "a reviewer must be able to evaluate without your account" |
| **Growing the discriminating-key allowlist** | Every unseen document family would need new entries - the hard-coding the brief prohibits |
| **Merging predicates on similarity alone** | At cosine 0.91 the encoder called `standalone_revenue_from_operations` and `revenue_from_operations` the same relation. They are not. A silent merge is unrecoverable downstream |
| **A reranking pass over the candidate pairs** | Measured against the false positives this build actually had, not in the abstract. Their claim-text cosines run 0.73 to 0.96, median 0.8532; the correct relations run 0.62 to 1.00, median 0.8533. The two distributions are the same distribution, and the single highest-scoring pair in the whole file is a false positive. A cut placed below the lowest false positive would discard 95% of the correct relations. Reranking scores textual relatedness, and an entity mismatch between `current other assets` and `current other financial assets` is *more* textually related, not less - the thing that separates them is referential and the subject key already decides it exactly. See Limitations |

### AI tools used, and what was delegated

- **Claude Code**: Big help under a time constraint, helped generate codes quicker than manually typing through it all. Testing throughout stages and final polishes to the documentation for a much smoother experience.
- **Extraction and adjudication model:** `gemini-3.1-flash-lite` over plain
  `httpx` with pydantic validation and a content-addressed disk cache. An
  OpenRouter provider sits behind the same seam as a per-token fallback. An open
  30B model was trialled for extraction and **reverted** - it produced more
  facts (625 against 488 across the corpus, 51 against 16 on the earnings deck)
  and folded conditions into predicate names (`consolidated_revenue`,
  `standalone_revenue`), which collapsed cross-document relations from 22 to 0
  and left just 2 predicates shared across 491. More facts is not better extraction; the metric that
  matters is whether two documents describing the same thing produce the same
  comparison key.
- **Local embeddings:** `BAAI/bge-small-en-v1.5` via `sentence-transformers`, on
  CPU. Deliberately local so the only credential in the system is the LLM key -
  and the committed cache removes that from the evaluation path too. Embeddings
  never decide anything; they widen the candidate set that the decision table
  then judges.

---

## Limitations and Next Steps

### The numbers, from the offline replay of all six starter PDFs

Both columns are the same replay of the same six PDFs from the same committed
cache. `before` is the code as it stood before this correctness pass; `after`
is the code as it stands now. Nothing in the fixes touches parsing, chunking,
extraction or blocking, so those rows are identical by construction and are
listed once.

```
Documents                     6          511 pages, 1,658,944 chars, 602 chunks
Parse + structure + chunk     2.9 s
Extraction requests           41         all served from data/cache/llm/
Live API requests             0
Wall clock, all six           16.3 s     warm embedding model

Facts extracted               501
  grounded                    496        99.0%
  quarantined                 5          1.0%
  duplicates collapsed        8
  in the ledger               488        477 located exactly, 11 fuzzily

Theoretical pairs             118,828    = C(488, 2)
Candidates after blocking     3,804      96.80% reduction
  comparison-key block        79
  value-anchored block        207
  semantic block              3,741
Blocking + decision table     1.4 s

                                             before      after
Facts carrying an inherited segment          0           16
Finished by a deterministic rule             3,725       3,732
Reached the LLM                              79          72
  prose-only, on a settled verdict           68          69
  adjudicated, the model may decide          11          3

Relations stored                             121         115
  reconciled_by_context                      70          69
  insufficient_context                       35          35
  corroborates                               11          11
  contradicts                                5           0
  unrelated (counted, not stored)            3,683       3,689
  cross-document                             34          33
Partition groups detected                    n/a         0

Verdicts in the ledger decided by the model  7           0
Adjudication calls                           90          75
Hallucinated-qualifier rejection             4 of 90     3 of 75
Judge self-consistency                       7 of 8      0 of 1
Unadjudicated                                0           0

Predicates registered                        297         298
  alias questions asked                      89          88
  confirmed                                  32          31
  refused                                    57          58
Tests                                        366         454   (19.3 s)
```

**Reading the verdict rows.** Seven relations left the ledger, one entered it,
and two changed verdict in place.

Leaving, all seven because the two facts were never about the same subject:
three parent-against-subsidiary incorporation dates (`Delhivery Limited`
against `Delhivery Freight Services Private Limited`, `Delhivery Cross Border
Services Private Limited` and `Delhivery Corp Limited`); one
`gross FDI inflows` 81.0 against `Total FDI` 50.0, which are two different RBI
series the predicate registry had merged; `Delhivery Limited` against
`Delhivery HK Pte Limited`; `Delhivery` against `Delhivery gateways`; and
`global economy` against `Global`.

Entering: `installed_solar_power_capacity` 4.6 MW against itself, stated on
one page `as_at "March 31, 2024"` and on another `as_at "end of FY24"`. Neither
label is a fiscal year, so neither resolves to an interval and the comparison
falls through to string equality; the two spellings differ, and a difference
in wording used to be enough to declare two identical figures separate states
of affairs. The years the two labels name are checked instead, and they agree.

Changing: `active_customers` 7,900 against 23,113 - the run's former headline
contradiction - is now `insufficient_context` naming `segment`, because the
bullet heading `PTL Freight` that the 7,900 sits under now reaches its
qualifier bag, and the 23,113 has no counterpart to compare it against. The
same inherited qualifier moves `449` against `23,113` from
`reconciled_by_context` to the same abstention.

**Every `contradicts` in the run was a false positive**, and the ledger is now
honest about the corpus containing none. Three pairs still reach the
contradiction branch deterministically - two CIN spellings of one company
(`U...` before listing, `L...` after), two different name-change dates, and the
Economic Survey's 6.4% against the RBI's 6.5% for the same year - and all three
are downgraded to `insufficient_context` by the adjudication guards. Those are
the honest residue: same subject, same predicate, disagreeing values, and no
qualifier on either side to explain it.

`Judge self-consistency 0 of 1` is one pair, not a rate. With the false
contradictions gone only three pairs still reach the adjudicating branch; two
are refused by the hallucinated-qualifier guard before the consistency check is
reached, and the third disagreed with itself and was correctly downgraded. A
denominator of one is worth printing as a denominator of one rather than as 0%.

**Cost.** The full corpus replays for **$0.00** - it is all cache. Extracting it
the first time was 41 requests on a Google AI Studio **free-tier** key, so the
Gemini spend to date is also $0.00. The only money spent on this project was
**$0.1933** on OpenRouter, during the open-model extraction trial that was
subsequently reverted.

### What was wrong and is now fixed

Six defects in the comparison path, all found by auditing every `contradicts`
and `reconciled_by_context` verdict in the ledger rather than by sampling. Each
is a root cause with a regression test, not a patched instance.

1. **Subject identity accepted a token subset.** `subjects_match` fell back,
   without a hard key on both sides, to accepting one surface token set as a
   *subset* of the other, so that `Delhivery` would match `Delhivery Limited`
   without a corporate-suffix list. But `{delhivery, limited}` is also a subset
   of `{delhivery, freight, services, private, limited}`, so a parent matched
   every subsidiary whose name contained its own - and `{cash, and,
   equivalents}` is a subset of `bank balances other than cash and cash
   equivalents`, so a balance sheet's line items matched each other. Across the
   whole ledger the rule produced **20 mismatched-subject pairs, including
   every one of the eight false contradictions**.

   It is now equality of the normalised surface, plus one narrow bridge: the
   longer name must be the shorter one *as an ordered prefix* followed by
   exactly one corporate legal form spelled as a whole phrase. `Delhivery` +
   `limited` passes; `Delhivery Limited` against `Delhivery Corp Limited` fails
   the prefix test even though the only extra word is one the list knows.
   Measured over the 223 distinct subject surfaces in the ledger, it joins two
   pairs, both correct. Subject mismatch now rejects before any value or
   qualifier is looked at, for any predicate.

2. **The context stack never propagated bullet-scoped headings.** A prospectus
   heads each business line with a bullet glyph and a short label -
   `PTL Freight`, `Express Parcel`, `Supply Chain Services` - and every
   unqualified figure below one belongs to it, exactly as a cell belongs to
   its table header. The chunk carried the heading path only as free text in
   the prompt, and only as computed at the chunk's *start*, so a
   4,000-character window crossing three bullets told the model about the
   first one.

   `Structure.inherited_qualifiers` now resolves the enclosing scope at the
   fact's own byte offset and attaches it as a `segment` qualifier with
   provenance `inherited`, which never overwrites one the extractor stated.
   Whether a bulleted line names a scope or is a sentence that happens to sit
   in a list is decided typographically: no digits, at most six words,
   title-cased. Across the seven documents in the ledger that accepts six
   labels and rejects nineteen bulleted sentences, and it puts a `segment` on
   16 facts - five of them `active_customers` figures for five different
   business lines.

3. **The predicate registry merged two different measures.** `gross FDI
   inflows` (US$ 81.0 bn, RBI narrative and Table II.7.2 line 1.1.1) and
   `Total FDI` (US$ 50.0 bn, Appendix Table 9's country- and industry-wise
   series) were merged at cosine 0.930 on the model's yes, manufacturing a
   contradiction out of two series the same report prints on two pages. The
   source settles it: Table II.7.2 lists `Gross Inflows 81.0` as a *component*
   of Net Inward FDI, so this is a genuine predicate distinction and not a
   missing scope qualifier - Appendix Table 9's own footnote is not even inside
   the 100-page excerpt.

   The registry now refuses, in code and without asking, any pair whose names
   reduce to the same head after one leading measure modifier is removed and
   whose modifiers differ: `gross` against `total`, `net` against `gross`,
   `basic` against `diluted`. It is checked against all 68 aliases the corpus
   had confirmed and refuses exactly one - this one - while independently
   reproducing three refusals the model had already made. The threshold was not
   touched: it gates *asking*, and the model's answers on this class were a
   coin flip (it refused `gross` against `cumulative` at 0.892 and accepted
   `gross` against `total` at 0.930).

4. **Complementary categories were run through the reconciliation path.** A
   breakdown by region, segment or class produces a pair between every two of
   its rows; each disagrees with every other by construction and each is
   correct. `concord/compare/partition.py` now groups facts sharing a subject,
   a predicate and every qualifier but one, and asks whether the values under
   that qualifier add up to a whole - using the same precision intervals as
   every other comparison, so there is no tolerance constant. Pairs inside such
   a group are `unrelated` before the value comparison runs. Verified against
   the 10-K in `starter-datasets/extras-tested/`: workforce by region, 63%
   Americas / 15% Europe / 22% Asia Pacific, produced three
   `reconciled_by_context` rows before and none after. It fires on nothing in
   the six starters, which is the correct answer for a corpus with no
   percentage breakdown in it.

5. **A partial-year label resolved to its whole year.** `parse_period` matched
   `FY24` anywhere in a string and discarded everything around it, so
   `first eight months of FY24`, `H1 of FY25`, `FY20 to FY24` and
   `FY25 (April-December)` all came back as one whole fiscal year. A
   partial-year figure and a full-year figure then carried *equal* intervals,
   read as the same stated condition, and their disagreement came out
   `contradicts` - which is precisely what `compare_qualifier`'s own docstring
   promises will not happen. Two such pairs were in the shipped ledger and both
   had to be rescued by the adjudicating model.

   A year label now means that year only when the rest of the string adds
   nothing to it. Where the words name a sub-span exactly it is resolved
   (`first eight months of FY24` -> 2023-04-01..2023-11-30, and
   `last three months of FY24` now equals `Q4 FY24`); where they name something
   else the label is refused and compared as text, which keeps two different
   labels different. This is the discipline `bare_date` already applied to
   dates, carried over to years.

6. **A period label nobody could parse was treated as proof of a difference.**
   Refusing to resolve `till FY26` is right - it is not a fiscal year - but the
   two facts then fall through to string comparison, and the agreement branch
   read any difference in a recognised key as "these are separate states of
   affairs". So `Feeding India ... 290 million meals since inception till FY26`
   and `... 290 million meals till FY26`, which are the same sentence written
   twice on two pages of one report, stopped corroborating.

   A recognised condition now breaks up agreement only when it was actually
   established to differ. If neither side's label resolved to an interval, the
   comparison was string equality, and two spellings are often one period - so
   the years each label names are checked instead, and only a genuine
   disagreement there counts. `till FY26` and `inception till FY26` both name
   2026; `April-December 2024` and `April-December 2023` do not, and stay a
   difference. Measured across the ledger this recovers three corroborations
   and changes nothing else: the two above, and `4.6 MW as_at "March 31, 2024"`
   against `4.6 MW as_at "end of FY24"`, which is one instant written two
   ways.

### What still does not work

1. **Extraction recall is thin, and batching is most of the reason.** The
   build's own log recorded 35 facts on the earnings deck under a stronger
   model at batch 8 against 16 under `flash-lite` at batch 16, and noted that
   model and batch had changed together so the causes were unseparated.
   Replaying the deck from cache at both batch sizes on the **same** model
   separates them:

   | Model | Batch | Extracted | Grounded | Quarantined |
   | --- | ---: | ---: | ---: | ---: |
   | `gemini-3.1-flash-lite` | 8 | 30 | 27 | 3 (10.0%) |
   | `gemini-3.1-flash-lite` | 16 | 16 | 16 | 0 |
   | `gemini-3.6-flash` (logged earlier) | 8 | 35 | - | - |

   Batch size is the dominant cause, not the model: 30 against 16 on one model,
   where the model change is worth 5. Doubling the passages per request roughly
   halves recall - and buys a 0% quarantine rate against 10%, so the deeper
   batch also produces cleaner quotes. That is a live trade-off the shipped
   default sits on the wrong side of for recall, and the right side of for
   request count.

   The concrete cost: the Gurugram PIN discrepancy the brief nominated as a
   contradiction case (`122002` in three places against `122001` in the BRSR
   section) is in the source, sits in the same chunk as a fact that *was*
   extracted, and neither PIN came back.

2. **Abstention is the failure mode, by design, and it grew.**
   `insufficient_context` is 32 of 118 relations, and two of those are new: the
   `active_customers` pairs where one side now carries an inherited `segment`
   the other cannot match. That is the missing-qualifier guard working as
   specified - it binds on *any* key present on one side and absent on the
   other, whether stated or inherited - but it means enriching one side's
   context can turn a confident reconciliation into an abstention. A pair whose
   one-sided qualifier is genuinely incidental abstains instead of resolving.
   Contradiction over-calling is the documented failure of this task, so this is
   the right direction to err, and it is still a real cost.

3. **The subject rule cannot bridge two spellings of one legal form.**
   `Delhivery` reaches `Delhivery Limited` because one name is the other plus
   a trailing legal form. `Delhivery Ltd` does not reach `Delhivery Limited`,
   because neither is a prefix of the other and the rule only ever strips a
   suffix from the longer side. No document in this corpus writes `Ltd` as a
   subject surface, so the exposure today is zero - all 24 surfaces that end in
   a legal form spell it out - but a filing that abbreviates would silently
   lose every cross-document pair for that entity. Stripping one form from
   *both* sides would fix it and would also let `Acme Corp` reach
   `Acme Limited`, which are different companies, so it is not a free change
   and it is not made on zero evidence.

4. **A figure's scale can be stated in prose the extractor does not read into
   the value.** The RBI narrative says "US$ 81.0 billion"; the fact's
   `value.scale` is `null` and it normalises to 81.0, while a table cell for a
   related series normalises to 5.0e10. Both facts are now `unrelated` for a
   different and correct reason, so no verdict currently depends on it, but the
   defect is real and would bite a corpus that stated the same series both ways.

5. **The semantic block carries the run.** All 115 stored relations were
   proposed by it; 31 by nothing else, including 16 of the 35 abstentions. It
   is also the least precise strategy by an order of magnitude - 3,741
   proposals for 115 kept relations. The honest description of the blocking
   layer is "one recall device and two cheap accelerators".

6. **A reviewer with no key cannot ingest a genuinely new PDF.** Everything
   already in the cache replays; anything else needs a key. This is the one
   honest gap in the keyless path. It is also a maintenance cost on every
   change to the comparison layer: the judge prompt carries both qualifier
   bags, so any fix that changes a bag invalidates that pair's cached
   explanation and needs a live call to refill. The correctness pass cost six.

7. **Embeddings are recomputed for every fact on every ingest.** The
   `facts.embedding` BLOB column exists and is not read back. Two seconds at
   488 facts, prohibitive at 50,000.

8. **Smaller, real, and unfixed:** no UI pagination (`limit` is fixed at
   200-300; fine at 773 facts, broken at 5,000); no true table cell grid, so
   row/column coordinates are not recorded; OCR and scanned PDFs are out of
   scope, though an unreadable or password-protected file is answered as a 400
   with a reason rather than a 500. `subject_key` is still only used when both
   sides carry one - a CIN on a document's cover page does not arm the rest of
   its facts.

### Next steps, in the order the failures above dictate

1. **Attack qualifier recall (acts on 2).** A second extraction pass over facts
   that reached a verdict, asking only "what conditions does this quote state?",
   would put `scope: excluding Spoton` on the `23,113` fact and turn its
   abstention into a reconciliation. This moves more mass out of
   `insufficient_context` than any new feature would, and it is now the largest
   single defect in the run.
2. **Follow the batch-size finding (acts on 1).** The A/B above says batch 8
   nearly doubles recall on the deck and triples the quarantine rate. Run the
   same comparison across all six documents, then pick the batch size on
   measured recall rather than on the request cap that set it - and re-extract
   the corpus once at whichever wins.
3. **Propagate a document's hard keys (acts on 3 and 8).**
   `subject_key` is populated only where the extractor read an identifier out of
   the sentence. Propagating a CIN found anywhere in a document to every fact
   whose surface matches it would arm the identity test for all 98 of the
   prospectus's facts rather than for one, and make the subject rule stronger
   than a name comparison can be. It closes the `Ltd` against `Limited` gap
   too, on evidence rather than on a guess about suffixes.
4. **A discontinuous-span aligner (recovers part of the quarantine).** Accept a
   quote as *k* ordered fragments inside one passage. The stitched table cell -
   `"FY23\n48 days"` from `"Particulars FY24 FY23 ... 45 days 48 days"` - is
   recoverable without weakening the grounding guarantee, because the bytes are
   still the document's.
5. **Extend partition detection past percentages (acts on 4 above).** A
   breakdown in absolute units is the same shape but needs the total to check
   against; where the total is itself an extracted fact, the parts could be
   summed against it instead of against 100.
6. **Ablate the semantic block (settles 5).** Rerun at `k` = 5, 10, 20 and plot
   relations recovered against candidates proposed; record the cosine on each
   candidate so a similarity floor can finally be argued from evidence.
7. **A local-LLM provider behind the existing seam (closes 6).** Roughly an
   hour: the provider interface is three methods.
8. **Cache the embeddings (closes 7)**, then paginate the UI (8).

---

## Additional Notes

- **Cost of a full run: $0.00 from cache.** The corpus was extracted on a
  free-tier key across 41 requests; the only money spent on the project was
  $0.1933 on OpenRouter for an extraction experiment that was reverted.
  `CONCORD_MAX_CALLS` (default 200) is a hard ceiling on live requests, claimed
  under a lock *before* each request is sent so it holds across the thread pool.

- **Offline evaluation path:** `CONCORD_OFFLINE=1`, then either browse the
  shipped ledger or re-derive the whole corpus by dropping the six starter PDFs
  onto the page - 56 seconds, zero live requests. See *Setup and Run
  Instructions* above.
