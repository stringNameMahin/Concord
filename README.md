# Concord

**A fact knowledge layer.** It reads PDFs, extracts facts that round-trip to
bytes in their source, and decides whether two facts corroborate each other,
genuinely contradict each other, or only *look* like they disagree because they
were measured under different conditions.

The graded object here is not the fact list; it is the **adjudicated pair**.

---

## Branch Navigation

- **'sub1'** - 1st submission. Working, but has some bugs and issues.
- **'dev'** - Active development branch with ongoing fixes and new features.

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
starter-datasets/starter-datasets/delhivery/*.pdf
starter-datasets/starter-datasets/india-macroeconomy/*.pdf
```

They are already committed at those paths, so a clone has them.

### Check the install

```bash
uv run pytest -q            # or: .venv/bin/pytest -q
```

Expected: **659 passed**, in about 40 seconds. The suite needs no network and
no API key; the one test that loads the real embedding model is marked `slow`
and will download the sentence-transformer if it is not already cached.

Twelve of those tests run against the committed ledger itself rather than
against fixtures, so they check the shipped artifacts and not only the code.

The whole demo runs with no API key at all: the ledger and the model responses
behind it are committed, so starting the server is enough to browse, search and
byte-verify everything already in it. See *Offline evaluation* below.

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

`data/cache/llm/` holds **515 content-addressed LLM responses** covering every
extraction and alias question for the corpus, and most of its adjudication.
**The whole corpus replays with 0 live calls and no failed extraction batch.**

Adjudication is the one leg with holes in it, and the reason is worth stating
plainly. The cache is keyed on the whole request payload and a judge prompt
carries both qualifier bags, so every fix that changes a qualifier invalidates
that pair's stored explanation. The correctness passes have changed a lot of
qualifier bags - a `segment` inherited from a bullet, a `consolidation` read
off a running header, a basis lifted out of a predicate - and the blocking pass
below adds pairs that had never been judged at all, so **62 of the six
starters' 134 relations now carry the deterministic explanation rather than
model prose.** That is the documented behaviour and not a failure: **failure is
not a verdict.** No verdict is affected, because those calls are prose-only on
verdicts the table has already settled. Refilling them costs one live call
each.

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

   Measured: **41 extraction requests, 0 live API calls, 0 extraction cache
   misses**, giving 488 grounded facts, 5 quarantined and **134 relations**, in
   60 seconds on a laptop.

   **The shipped `.sqlite` contains this replay**, plus one further document -
   a 409-page annual report from a company outside the starter set, ingested to
   test that the fixes below generalise. The file therefore holds 7 documents,
   773 grounded facts, 12 quarantined and 209 relations; the six starters
   account for 488 / 5 / 134 of those, and every per-corpus number in this
   README is the six-starter figure unless it says otherwise.

   The order above is the order the shipped file was built in, and it is now
   recorded for reproducibility rather than because the answer depends on it.
   It used to: a pair was only ever judged while one of its facts was new, so
   the same six documents uploaded in a different sequence produced a different
   ledger, and re-running the comparison over the stored facts produced a third
   - 186 relations against the 174 that shipped. Every ingest now re-blocks the
   whole corpus, which the persisted fact embeddings make affordable: a full
   comparison run over 773 facts is **0.09 s warm against 13.8 s cold**, and
   the three blocking strategies together are 3 ms of that.

   Re-blocking the whole corpus fixed the order dependence and left a subtler
   one behind, which is also now closed. The semantic block is top-k, and k is
   an absolute budget: as facts accumulate, a true pair can be pushed out of
   both facts' neighbour lists, and a whole-corpus run then deletes the
   relation it no longer proposes. Two went that way. The fix was not a larger
   k - see *What was wrong and is now fixed*, item 19.

---

## Video Demo

**Link:** : https://drive.google.com/file/d/10zHrwhABFf-fdw-qt_kofQyPxU6nmgJW/view?usp=sharing

The video was recorded against an earlier ledger, so the relation counts on
screen are lower than the ones in this README. Nothing about the behaviour it
shows has changed.

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
 |                       The exact one resolves the key the way the decision
 |                       table does - through the registry's aliases and the
 |                       subject identities `same_entity` accepts - so it
 |                       proposes every pair that could be stored. The other
 |                       two are recall insurance over a vocabulary the
 |                       registry has not linked yet.
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
| Values disagree, the only condition that differs is one dimension spelled two ways, neither wording resolving | `insufficient_context` | `unestablished_context_difference` | no |
| Values disagree, a stated condition differs | `reconciled_by_context` | `discriminating_qualifier_differs` | prose only |
| Values disagree, both sides state the **same** conditions | `contradicts` | `same_context_disjoint_intervals` | yes, adjudicates |

Bags are diffed by **dimension**, not by literal key name. An open vocabulary
spells one condition several ways - `period`, `as_at`, `as_of`, `time_period`
and `financial_year` all answer *when?* - and comparing the names meant two
facts that both stated the time each read as stating a condition the other
lacked. `concord/facts.py` declares the one dimension that is measurably needed
and, just as importantly, the moments deliberately kept out of it: a contract's
`effective_date` is not the period a figure covers, and merging those would be
the false merge the whole change exists to avoid.

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
  ->  3,679 candidates after blocking          (96.90% reduction)
  ->  3,588 finished by a deterministic rule   (97.5% of candidates)
  ->     91 pairs a deterministic rule referred
  ->     72 of those reached a model           (0.061% of the theoretical space)
       every one of them prose-only, on a verdict already decided
       (the other 19 have no cached response and keep the table's own sentence)
  ->      0 verdicts in the ledger are the model's        (was 7, then 1)
```

That last line is the one that moved, and it has moved three times. Before the
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

The fifth was the interesting one and it is now gone too, at a cost worth
stating rather than hiding. Two publishers give India's FY2024-25 real GDP
growth as 6.4 and 6.5, and the model called that a contradiction. It reached
the model at all because the predicate registry had merged
`real_gdp_growth_rate` into `gdp_growth_rate` on the model's yes - and real and
nominal growth are different measures, so that merge was wrong even though this
pair happened to be right. The registry now refuses a one-sided basis modifier
in code (`contrastive` in `registry.py`), which splits the GDP family into
three vocabulary entries and leaves 6.4 and 6.5 under different comparison
keys. **A real cross-publisher disagreement is no longer surfaced.** The trade
is a latent wrong merge closed against a right answer lost, and it is the one
place in this pass where the fix took something with it.

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
`predicate_registry`, `schema_events`) with JSON columns. It ships as a 4.4 MB
file a reviewer opens with `sqlite3` and no Docker. The canonical document text
lives *beside* the database as `data/work/<doc_id>.txt` rather than inside it,
so an offset can be checked by hand and `/verify` can re-read the bytes fresh.
The row stores that file's *name* and the reader resolves it against
`CONCORD_WORK`, so a ledger is readable from whatever directory it was cloned
into. It used to store the absolute path of the checkout that wrote it, which is
a path no reviewer has - see *What was wrong and is now fixed*.

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

One replay of the six starter PDFs from the committed cache, re-derived after
the correctness passes below. There is no before/after column any more: the
ledger has been rebuilt three times now and a stale comparison column is worse
than none. What moved in the latest pass is named underneath.

```
Documents                     6          511 pages, 1,658,944 chars, 602 chunks
Extraction requests           41         all served from data/cache/llm/
Live API requests             0          and 0 failed extraction batches
Wall clock, all six           59.8 s     warm embedding model, six full ingests

Facts extracted               501
  grounded                    496        99.0%
  quarantined                 5          1.0%
  duplicates collapsed        8
  in the ledger               488        477 located exactly, 11 fuzzily

Facts carrying a qualifier the layer supplied, not the sentence
  consolidation (page frame / statement title)   33
  segment (bullet heading)                       16
Facts whose magnitude was read back out of the bytes    66
Facts flagged for a subject that names the measurement   5

Theoretical pairs             118,828    = C(488, 2)
Candidates after blocking     3,679      96.90% reduction
  comparison-key block        141
  value-anchored block        228
  semantic block              3,596
Full comparison run           0.06 s     warm vectors; 13.8 s if re-embedded

Finished by a deterministic rule             3,588     97.5% of candidates
Referred by a deterministic rule                91
  ...of which reached a model                   72     all prose-only
  ...no cached response, kept the table's own    19
Verdicts in the ledger decided by the model      0

Relations stored                               134
  reconciled_by_context                         89
  insufficient_context                          29
     ...missing qualifier                       24
     ...values incomparable                      2
     ...the judge named a qualifier neither      2
        fact carries
     ...a difference that could not be shown     1
  corroborates                                  16
  contradicts                                    0
  unrelated (counted, not stored)            3,545
  cross-document                                48
Partition groups detected                        0

Which strategy proposed each stored relation
  comparison-key block                         134     all of them
  semantic block                               124
  value-anchored block                          17
  proposed by the comparison-key block alone     8
  proposed by any other strategy alone           0

Relations carrying model prose                  72
Relations carrying the table's own sentence     62

Predicates registered                          298
  alias questions asked                         85
  confirmed                                     29
  refused by the model                          56
  refused in code, without asking                3
Tests                                          659   (about 40 s)
```

**What moved in this pass, and why.** Relations went from 129 to 134 and
`insufficient_context` fell from 38 to 29, with `corroborates` unchanged at 16.
Two changes account for all of it, and both are about pairs the layer was
failing to *compare* rather than failing to judge.

The qualifier bag is now diffed by dimension rather than by key name, which
moved 11 relations out of `insufficient_context` and into
`reconciled_by_context`: both documents had stated when the figure held, one
writing `as_at` and the other `as_of`, and the guard had been abstaining with
an explanation that said one of them had not.

The exact comparison-key block now resolves predicates through the registry's
aliases and subjects through the identities `same_entity` accepts, which added
11 relations that only the semantic block could previously have found - and, on
three of them, had stopped finding.

Three rows left the ledger in the same pass, and they are the one place to look
twice. `1 billion` cumulative shipments against `740 million` in a year used to
be stored as `insufficient_context`, the single-digit guard refusing to read a
coarse figure's overlap as agreement. Those pairs now differ on a condition the
table can prove differs - a fiscal year against an as-at date - so they are
`unrelated` before the precision question is reached, and `unrelated` is not
stored. Both answers refuse the corroboration, which is the point; the cost is
that the single-digit guard no longer has a live example in this corpus, though
it is still in the table and still pinned by its own tests.

**Reading the residue.** Two pairs still reach the contradiction branch
deterministically, and both are refused before they reach the ledger as one:
the two CIN spellings of one company (`U...` before listing, `L...` after) and
two different name-change dates. In each the model answered
`reconciled_by_context` naming a condition that appears on neither record, the
named-qualifier guard threw the answer out, and the pair is stored as
`insufficient_context` saying so. That is the designed behaviour: the system
declines to explain a real conflict with a qualifier nobody wrote down.

Self-consistency therefore has no denominator in this run. Both adjudicated
pairs were refused by the first guard, so neither reached the second. A rate
over zero pairs is worth printing as zero pairs rather than as 0%.

**Every `contradicts` in the six starters was a false positive**, and the
ledger is now honest about the corpus containing none. The one genuine
cross-publisher disagreement it used to surface - 6.4 against 6.5 for India's
FY2024-25 real GDP growth - is no longer surfaced either, for a reason recorded
in full under *The deterministic / LLM split* above: the registry merge that
brought those two facts onto one comparison key was itself unsound, and closing
it took the right answer with the wrong one.

**What the inherited qualifiers changed.** `active_customers` 7,900 against
23,113 was the run's original headline contradiction. The bullet heading
`PTL Freight` that the 7,900 sits under now reaches its qualifier bag and the
23,113 has no counterpart to compare against, so it abstains naming `segment`.
The extraction failure underneath it is unchanged and is case 4 below: the
distinguishing phrase is in both quotes and was never emitted as a qualifier.

**And one pair that agreement recovered.** `installed_solar_power_capacity`
4.6 MW is stated on one page `as_at "March 31, 2024"` and on another
`as_at "end of FY24"`. Both now resolve to the same instant - the closing edge
of a fiscal year is a moment, not a year - so two spellings of one date are one
condition by interval rather than by a fallback that checks the years they
name.

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

### What a third audit pass found, and what it cost

A full re-audit of the codebase against the shipped ledger raised twelve more
findings. These are the ones acted on, each with a regression test and a
measurement rather than an assertion.

9. **The ledger was a function of upload order.** A pair was only ever judged
   while one of its facts was new, so the same documents uploaded in a
   different sequence produced a different ledger - and re-running the
   comparison over the stored facts produced a third answer again, 186
   relations against the 174 that shipped. Every ingest now re-blocks the whole
   corpus and replaces what it no longer keeps. What stays incremental is the
   part that costs money: a pair a model has already answered carries that
   answer forward and never re-enters the queue, which is the same guarantee
   the incremental path was written for and is now recorded on the row
   (`relations.judged`) instead of inferred from which document was new.

10. **Every ingest re-embedded the entire ledger.** `facts.embedding` was
    declared and never written, so a comparison run spent 13.8 of its 13.9
    seconds encoding facts it had encoded before, growing with the ledger
    rather than with the document being added. Vectors are now persisted
    against the model that produced them and a different model's are ignored
    rather than trusted. **13.8 s to 0.09 s** over 773 facts, which is what
    makes re-blocking the whole corpus affordable on every upload.

11. **A figure written to one digit corroborated most of its decade.**
    `1 billion` spans `[5e8, 1.5e9]` - half its own value either way - so it
    overlapped `740 million` and the ledger stored cumulative lifetime
    shipments as confirmation of one year's volume. An overlap that exists only
    because one side committed to a single digit is now `insufficient_context`
    naming the figure that carried it. The test is not "is it round": it is
    whether the *other* figure's own precision could account for the gap, which
    is what keeps `8,142 Cr` against `81,415.38 mn` corroborating.

12. **A basis folded into a predicate name fragmented the comparison key.**
    The model sometimes coins `consolidated_revenue_from_operations` instead of
    qualifying `revenue_from_operations`, which puts the basis where it splits
    the key instead of conditioning it - three spellings of one line item,
    every pair `unrelated`, no trace. The basis is now lifted into the
    `consolidation` qualifier, where the decision table already knows what to
    do with it. The model's own spelling is kept for display beside the quote.

13. **`the Company` named no entity.** 50 facts in the ledger had a subject
    that is document-relative, so two annual reports both saying "the Company"
    would share a comparison key while meaning two different organisations.
    Anaphors are now resolved to the document's own dominant subject, and only
    where that is unambiguous - a tie leaves them alone, because attaching
    facts to the wrong entity is worse than leaving them orphaned. The rewrite
    is flagged on the fact, because a subject the layer supplied and one the
    sentence stated are different evidence.

14. **A fifth of period labels never resolved to an interval.** `Fiscal 2021`
    is one document's house style with 212 occurrences of it, `Mar-26` is a
    column header, `April-December 2024` is a partial year stated by its bounds
    and `end of FY24` is an instant rather than a year. None of them parsed, so
    they fell through to string equality and two spellings of one period read
    as two conditions. **46 of the 103 unresolved period qualifiers now
    resolve.** A bare `2024` deliberately still does not - see 8 below.

15. **A quarantined row's id moved between processes.** It was built from
    Python's `hash()` of the quote, which is randomised per interpreter, so the
    same unlocated quote got a different id in every run and its
    `/evidence/{id}` link broke on the next ingest. Content-addressed now, the
    way `make_fact_id` already was.

16. **Adjudication failure was silent.** `/ingest` caught every exception from
    the judge and omitted the block, so a run whose deterministic `contradicts`
    rows were never judged looked exactly like a clean one. The reason now
    travels in the payload and the upload line renders it, along with failed
    extraction batches and the fiscal basis the document was read on.

17. **The two quarantine rates could not be reconciled.** `/ingest` measured it
    before `dedupe` and `/stats` after, and the counters that would explain the
    difference - extracted, requests, failed batches, duplicates collapsed -
    lived only in one HTTP response. They are persisted on the document row
    now, so the extracted-to-grounded leg is checkable from the shipped
    artifacts by someone who did not run the ingest.

18. **The partition gate could have suppressed a time series.** It looks for
    facts that share everything but one qualifier and whose values sum to a
    whole; a two-point series whose values happen to sum near 100 has that
    shape exactly, and the nearest miss in the ledger was 5.5% away. Time is
    now excluded as a split axis under every name the extractor coins for it.
    The gate still fires on nothing in this corpus, which is the correct answer
    for a corpus with no distribution in it.

### What a fourth audit pass found, and what it cost

An audit of the shipped repository against a fresh clone rather than against
the working tree. The first finding is the one that mattered, and it is the
kind only a clone can show you.

19. **A clone could not read its own evidence.** `documents.text_path` stored
    the absolute path of the checkout that wrote the ledger -
    `D:\...\data\work\<doc>.txt` - and nothing in the read path ever consulted
    `CONCORD_WORK`. `data/work/` ships with the repository, so a clone *has*
    every document's canonical text; it simply had no way to find it. On a
    reviewer's machine that path does not exist, so `GET /evidence/{id}` and
    `GET /verify/{id}` answered **500 on every fact** - the two endpoints the
    grounding guarantee rests on, and every *show evidence & verify* button in
    the UI - and the ledger invariant that re-cuts each quote failed with
    `FileNotFoundError`, so `pytest` reported a failure on a clean clone.

    The miss was not the worst of it. On a machine that *did* have the original
    checkout, the stored path resolved - to the other repository's bytes. A
    clone would have been verifying its facts against a directory nobody had
    cloned, and saying `verified: true` about it.

    **Fixed.** The row stores the file's name; `resolve_text_path` reads it
    against this installation's `CONCORD_WORK` first and falls back to the
    stored value, so an old ledger still opens and a relocated one prefers the
    text sitting next to it. Verified by cloning the repository, making the
    original checkout unreachable, and running the suite and both endpoints
    from the clone: 659 passed, both endpoints 200, `verified: true`.

20. **One condition spelled two ways read as two conditions, each missing.**
    The qualifier bag is an open vocabulary, so `period`, `as_at`, `as_of`,
    `time_period` and `financial_year` all arrive meaning *when this figure
    holds*. The decision table diffed the bags by literal key name, so a pair
    where both documents stated the time - one writing `as_at`, the other
    `as_of` - saw two keys each absent from the other side, hit the
    missing-qualifier guard, and abstained with an explanation that said one
    fact did not state a period two words from where it did. **18 of the 46
    abstentions in the seven-document ledger were this.**

    **Fixed.** Bags are diffed by dimension. The dimension is declared narrowly
    and the exclusions are the load-bearing half: `effective_date`,
    `acquisition_date`, `start_date` and `maturity_period` name moments
    belonging to an event in the claim, not the reporting time of the
    measurement, and folding those in would have been a worse bug than the one
    being fixed. Where the two sides spell one dimension differently and
    *neither* wording resolves to an interval, the pair is not reconciled
    either: nothing was shown to differ, and a difference that has not been
    shown cannot account for the figures. That is a new row in the table
    (`unestablished_context_difference`) and it fires 4 times.

21. **The one strategy that costs nothing could not see what the system had
    already decided.** The exact block grouped on `Fact.comparison_key`
    verbatim - the key as *written*. The decision table resolves it twice over
    before deciding anything: the predicate registry says whether two names are
    one relation, and `subjects_match` says whether two surfaces are one entity
    up to a trailing legal form. Neither question reached blocking, so a
    confirmed alias, a fact carrying a CIN against one carrying only the name,
    and `Delhivery` against `Delhivery Limited` all sat in separate buckets and
    could only be found by the semantic block. **12 of the 215 pairs that share
    a resolved comparison key were never proposed at all.**

    **Fixed.** The block resolves predicates through the alias map and files
    each fact under every identity it answers to. Candidates rose by 12 across
    the whole corpus - 5,973 to 5,985 - and recall over storable pairs went
    from 203/215 to **215/215**.

22. **A relation could disappear because an unrelated document was uploaded.**
    The semantic block is top-k and k is an absolute budget, so as facts
    accumulate a true pair can be pushed out of both facts' neighbour lists.
    A whole-corpus run replaces what it no longer proposes, so the relation is
    then deleted. Two were, between one rebuild and the next: a `>2.8Bn`
    cumulative shipment figure against two statements of `740 million`, which
    had fallen to neighbour ranks 16 and 22 at 773 facts.

    **The fix was not a larger k.** A pair is only stored if its comparison
    keys match, so once item 21 made the exact block resolve those keys it
    enumerates every storable pair by construction - and it has no rank to be
    crowded out of. Both lost relations are back, proposed by the exact block.
    A ledger invariant now asserts that every stored relation is proposed by
    that strategy, so if anything ever starts resting on top-k again the suite
    says so rather than a future rebuild quietly deleting it.

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

2. **Abstention is the failure mode, by design.** `insufficient_context` is 29
   of 134 relations, and 24 of those are the missing-qualifier guard: a
   discriminating key present on one side and absent on the other. The guard
   binds on *any* such key, stated or inherited, so enriching one side's
   context turns a confident reconciliation into an abstention - the
   `active_customers` pairs abstain on `segment` precisely because the bullet
   heading now reaches one of them. Contradiction over-calling is the
   documented failure of this task, so this is the right direction to err, and
   it is still a real cost: a pair whose one-sided qualifier is genuinely
   incidental abstains instead of resolving.

   This used to be worse and by a measurable amount. Eighteen of the 46
   abstentions in the seven-document ledger were pairs where *both* documents
   stated the condition and had only spelled it differently, which the
   dimension diff now handles. What is left is the real shape of the problem:
   one document says something the other does not.

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

4. **A footnote marker's legend is usually in a different chunk, and a
   multi-marker reference keeps only the first.** Across the eight PDFs there
   are 600 `(n)` references attached to a figure, and only 26 of them sit in a
   chunk that also holds the matching legend line; 204 have a legend elsewhere
   in the document that no chunk holding the reference can see. Worse, where a
   figure carries two markers the second is lost every time - a headcount of
   `98,135 (1,5)` kept the `as_of` from the first marker and dropped the
   second, which is the one that says whether contractors are counted. Those
   are exactly the discriminating qualifiers the missing-qualifier guard exists
   to protect. The fix is structural rather than semantic - collect the legend
   lines per page during structure analysis and append the ones a chunk's
   figures actually reference to its context block - and nothing is wrong
   today only because no counterpart fact exists for any of the three measured
   cases.

5. **The semantic block no longer carries the run, and that is the fix rather
   than a loss.** It used to propose 127 of 129 stored relations with 40 of
   them found by nothing else, which sounds like strength and was a weakness:
   top-k is an absolute budget, so a relation whose only proposer was the
   semantic block could be crowded out by an unrelated upload and then deleted
   by the whole-corpus replace. Two were.

   The exact block now resolves the comparison key the way the decision table
   does. Because a pair is only ever stored if its comparison keys match, that
   makes it complete: it proposes all 134 stored relations, 8 of them alone,
   and no other strategy is the sole proposer of any. The semantic block still
   runs - it is 3 ms and it is the only recall a vocabulary has before the
   registry has linked it - but nothing in the ledger now depends on it.

   It remains by far the least precise strategy: 3,596 proposals for 124 of the
   kept relations, against the comparison-key block's 141 proposals for all
   134. That is the honest description of the blocking layer now: one complete
   deterministic strategy, and two cheap devices that widen the net for
   vocabulary the system has not learned yet.

6. **A reviewer with no key cannot ingest a genuinely new PDF, and the judge
   cache is expensive to keep warm.** Everything already in the cache replays;
   anything else needs a key. That is the one honest gap in the keyless path.
   The second half is a standing maintenance cost: the judge prompt carries
   both qualifier bags, so every fix that changes a bag invalidates that pair's
   cached explanation. **62 of the 134 six-starter relations currently carry
   the deterministic explanation rather than model prose** for that reason, and
   the blocking fix added pairs that had never been judged at all. No verdict
   is affected - those calls are prose-only on verdicts the table has already
   settled - and refilling them is one live call each, but the number only goes
   up as the context stack improves.

7. **Closing an unsound predicate merge cost a right answer.** The registry now
   refuses, in code, a merge where one name carries a basis modifier the other
   does not: `real_gdp_growth_rate` and `gdp_growth_rate` really are different
   measures. Splitting that family put the Economic Survey's 6.4% and the RBI's
   6.5% for the same year under different comparison keys, so the one genuine
   cross-publisher disagreement in the corpus is no longer surfaced at all. The
   merge that used to surface it was unsound; the relation it produced was
   right. Closing this properly needs the registry to recognise
   `real_gdp_growth` and `real_gdp_growth_rate` as one measure - which is
   precisely the question the model got wrong when it was asked, refusing at
   cosine 0.979.

8. **Smaller, real, and unfixed:** no UI pagination - the fact ledger view
   asks for 300 rows and the ledger holds 773, so it is already truncating and
   the header count beside it is the full total; search reaches the rest. No
   true table cell grid, so
   row/column coordinates are not recorded; OCR and scanned PDFs are out of
   scope, though an unreadable or password-protected file is answered as a 400
   with a reason rather than a 500. `subject_key` is still only used when both
   sides carry one - a CIN on a document's cover page does not arm the rest of
   its facts. A bare `2024` as a period label is deliberately left unresolved,
   because it is either the calendar year or the fiscal year ending in it and
   the label does not say which; 9 qualifiers in the ledger sit in that state.

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
5. **Carry footnote legends into the chunk that references them (closes 4).**
   Collect the `(n) text` lines per page alongside the page frame and append
   the ones a chunk's figures reference to its context block, keeping every
   marker rather than the first. It reuses the offset machinery already there
   and adds no vocabulary.
6. **Refill the judge cache, then keep it warm (acts on 6).** 62 relations
   carry a deterministic explanation because a qualifier fix invalidated their
   cached prose or because the blocking fix proposed the pair for the first
   time. One live call each, prose-only, no verdict at risk. Worth doing as the
   last step before a demo rather than after every fix.
7. **Ablate the semantic block (settles 5).** Now that the exact block proposes
   every stored relation, the open question is whether the semantic block earns
   its 3,596 proposals at all, or only pays for itself on vocabulary the
   registry has not yet linked. Rerun at `k` = 0, 5, 10, 20 and plot relations
   recovered against candidates proposed. A measured answer of "nothing" would
   be a real simplification rather than a loss.
8. **A local-LLM provider behind the existing seam (closes the keyless gap in
   6).** Roughly an hour: the provider interface is three methods. Then
   paginate the UI (8).

---

## Additional Notes

- **Cost of a full run: $0.00 from cache.** The corpus was extracted on a
  free-tier key across 41 requests; the only money spent on the project was
  $0.1933 on OpenRouter for an extraction experiment that was reverted. The
  correctness passes since, including the one that produced items 19-22 above
  and the ledger rebuild behind every number here, made **zero live requests**.
  `CONCORD_MAX_CALLS` (default 200) is a hard ceiling on live requests, claimed
  under a lock *before* each request is sent so it holds across the thread pool.

- **Offline evaluation path:** `CONCORD_OFFLINE=1`, then either browse the
  shipped ledger or re-derive the whole corpus by dropping the six starter PDFs
  onto the page - about a minute, zero live requests. See *Setup and Run
  Instructions* above.
