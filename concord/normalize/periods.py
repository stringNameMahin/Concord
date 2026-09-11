import calendar
import re
from dataclasses import dataclass
from datetime import date

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

COUNT_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}

DEFAULT_FY_END_MONTH = 3

# Words that may surround a date without making the value something other than
# a date. Anything else means the string is a statement that cites a date.
DATE_FILLER = {"as", "at", "of", "on", "the", "dated", "date", "ended", "ending"}

# The same discipline for year labels. A label resolves to a whole fiscal or
# calendar year only when the rest of the string adds nothing to it. These are
# the words that add nothing: naming the basis, or nothing at all. Provisional,
# revised and budget markers are here because they qualify the vintage of the
# figure rather than the span it covers.
YEAR_FILLER = {
    "fy", "f", "y", "fiscal", "financial", "year", "years", "yr",
    "cy", "calendar", "the", "of", "for", "in", "during", "full", "entire",
    "whole", "period", "p", "be", "re", "prov", "provisional",
}

MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
COUNT_RE = "|".join(COUNT_WORDS)

ENDED = re.compile(
    rf"\b(?:(?P<count>{COUNT_RE}|\d{{1,2}})[\s-]+(?P<span>month|months|quarter|year)s?"
    rf"[\s\w]{{0,12}}?)?\bended?\s+(?:on\s+)?(?P<month>{MONTH_RE})\.?\s+(?P<day>\d{{1,2}}),?\s+"
    rf"(?P<year>\d{{4}})",
    re.I,
)
# A *year* that ended on a date, which is a document stating its reporting
# basis. `ENDED` above is deliberately looser because it serves `parse_period`,
# where any "ended <date>" is a period boundary worth resolving - but a basis
# is a claim about the whole document and needs the stronger evidence.
#
# The difference is one word and it decides everything. "for the year ended
# March 31, 2024" names a reporting year; "discussions that ended on September
# 18, 2025" names a meeting. `ENDED` matches both, and the second put the IMF
# Article IV on a September fiscal year off one sentence in ninety-five pages -
# see F2 in docs/devRead.md. Partial periods are excluded by construction: a
# "nine months ended December 31" carries no "year" before "ended".
YEAR_ENDED = re.compile(
    rf"\byears?\s+end(?:ed|ing)\s+(?:on\s+)?(?P<month>{MONTH_RE})\.?\s+"
    rf"(?P<day>\d{{1,2}}),?\s+(?P<year>\d{{4}})",
    re.I,
)
AS_OF = re.compile(
    rf"\bas\s+(?:at|of|on)\s+(?P<month>{MONTH_RE})\.?\s+(?P<day>\d{{1,2}}),?\s+(?P<year>\d{{4}})",
    re.I,
)
AS_OF_NUMERIC = re.compile(r"\bas\s+(?:at|of|on)\s+(\d{1,2})[/-](\d{1,2})[/-](\d{4})", re.I)
BARE_DATE = re.compile(rf"\b(?P<month>{MONTH_RE})\.?\s+(?P<day>\d{{1,2}}),?\s+(?P<year>\d{{4}})", re.I)
DAY_FIRST_DATE = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<month>{MONTH_RE})\.?,?\s+(?P<year>\d{{4}})",
    re.I,
)
NUMERIC_DATE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b")

FY_SPLIT = re.compile(r"\b(?:FY|F\.Y\.?|fiscal(?:\s+year)?)?\s*(\d{4})\s*[-/]\s*(\d{2,4})\b", re.I)
FY_SHORT = re.compile(r"\bFY\s*[-']?\s*(\d{2}|\d{4})\b", re.I)
CY = re.compile(r"\b(?:CY|calendar\s+year)\s*(\d{4})\b", re.I)
QUARTER = re.compile(
    r"\bQ([1-4])\s*[-,]?\s*(?:of\s+)?(?:FY|F\.Y\.?|fiscal(?:\s+year)?)?\s*"
    r"(\d{2,4})(?:\s*[-/]\s*(\d{2,4}))?\b",
    re.I,
)

# A run of months or quarters cut out of a year: "first eight months of FY25",
# "first quarter of FY2025/26", "last three months of 2024".
SUBSPAN = re.compile(
    rf"\b(?P<edge>first|initial|last|final)\s+"
    rf"(?:(?P<count>{COUNT_RE}|\d{{1,2}})\s+)?"
    rf"(?P<unit>month|months|quarter|quarters)\b",
    re.I,
)
HALF = re.compile(r"\b(?:H\s*(?P<index>[12])|(?P<edge>first|second)\s+half)\b", re.I)


@dataclass(frozen=True)
class Period:
    """A closed interval plus how its fiscal basis was decided.

    Two facts are period-comparable only when their intervals are equal.
    Overlapping but unequal intervals are a reconciliation, not a conflict, so
    the basis has to be recorded rather than silently assumed.
    """

    start: date
    end: date
    label: str
    granularity: str
    fiscal_basis: str

    @property
    def assumed(self) -> bool:
        return self.fiscal_basis.startswith("assumed")

    def __str__(self) -> str:
        return f"{self.start.isoformat()}..{self.end.isoformat()}"


def _end_of(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def _add_months(anchor: date, months: int) -> date:
    total = anchor.month - 1 + months
    year = anchor.year + total // 12
    month = total % 12 + 1
    return date(year, month, min(anchor.day, calendar.monthrange(year, month)[1]))


def _widen_year(value: str, near: int) -> int:
    number = int(value)
    if len(value) == 4:
        return number
    century = near - near % 100
    candidate = century + number
    if candidate < near - 50:
        candidate += 100
    return candidate


def _fy_from_end_year(end_year: int, fy_end_month: int, label: str, basis: str) -> Period:
    end = _end_of(end_year, fy_end_month)
    start = _add_months(date(end_year, fy_end_month, 1), -11)
    return Period(start, end, label, "year", basis)


def fiscal_year_end_evidence(text: str) -> dict[int, int]:
    """Every month this document says one of its years ended in, counted.

    Returned rather than kept private because the basis is an inference the
    layer makes about a whole document off a handful of sentences, and an
    inference nobody can see the evidence for is the kind that goes wrong
    quietly. `POST /ingest` reports this alongside the month it chose.
    """
    counts: dict[int, int] = {}
    for match in YEAR_ENDED.finditer(text):
        month = MONTHS[match.group("month").lower()]
        counts[month] = counts.get(month, 0) + 1
    return counts


def infer_fiscal_year_end(text: str) -> int | None:
    """Read the document's own fiscal year end rather than assuming one.

    A filing that says "for the year ended March 31, 2024" has told us its
    basis. Only when nothing in the document says so does the configured
    default apply, and the period is then marked assumed - which is the right
    answer far more often than a guess is, because the label a document writes
    means what its publisher's calendar says and getting that wrong moves
    every period in the document by months.

    Two things make the reading conservative.

    **Only a year ending counts.** `YEAR_ENDED` requires the word before
    `ended`, so a sentence about anything else that ended on a date cannot vote.
    The IMF Article IV was put on a September basis by exactly one match in
    ninety-five pages - "discussions that ended on September 18, 2025", the
    mission's own meeting schedule - and every period label in it then resolved
    six months away from every other publisher's reading of the same label.

    **A tie is not an answer.** Two months with equal support means the
    document has not told us, and inventing a winner out of dict ordering would
    be a coin flip dressed as a reading. Returning None costs a label the
    `assumed` tier it should have had anyway.
    """
    counts = fiscal_year_end_evidence(text)
    if not counts:
        return None

    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def find_date(text: str) -> tuple[date, int, int] | None:
    """Locate a written date and report where it sits in the string.

    The span matters to the caller: a value that merely *mentions* a date is
    not a date, and comparing it as one is how two different conditions
    anchored to the same day come to look identical.
    """
    raw = text or ""
    for pattern in (BARE_DATE, DAY_FIRST_DATE):
        match = pattern.search(raw)
        if match:
            try:
                found = date(
                    int(match.group("year")),
                    MONTHS[match.group("month").lower()],
                    int(match.group("day")),
                )
            except ValueError:
                return None
            return found, match.start(), match.end()

    numeric = NUMERIC_DATE.search(raw)
    if numeric:
        day, month, year = (int(group) for group in numeric.groups())
        if month <= 12:
            try:
                return date(year, month, day), numeric.start(), numeric.end()
            except ValueError:
                return None
    return None


def parse_date(text: str) -> date | None:
    """Resolve a written date in any of the orders these documents use.

    Dates are compared as dates and never as strings, so "March 31, 2024" and
    "31 March 2024" have to reach the same value. Numeric dates are read
    day-first, matching `parse_as_of`. A string that does not resolve returns
    None and is then compared as text rather than guessed at.
    """
    found = find_date(text)
    return found[0] if found else None


def bare_date(text: str) -> date | None:
    """The date this value *is*, as opposed to one it happens to mention.

    "As at March 31, 2024" is a date wearing a preposition. "Cessation of
    employment prior to one year from August 24, 2021" is a condition that
    cites one, and two such conditions citing the same day are not the same
    condition.
    """
    found = find_date(text)
    if not found:
        return None

    moment, start, end = found
    remainder = f"{text[:start]} {text[end:]}"
    words = re.findall(r"[a-zA-Z0-9]+", remainder)
    return moment if all(word.lower() in DATE_FILLER for word in words) else None


def parse_as_of(text: str) -> date | None:
    match = AS_OF.search(text or "")
    if match:
        return date(
            int(match.group("year")), MONTHS[match.group("month").lower()], int(match.group("day"))
        )
    numeric = AS_OF_NUMERIC.search(text or "")
    if numeric:
        day, month, year = (int(g) for g in numeric.groups())
        if month <= 12:
            return date(year, month, day)
    return None


def _run_of_months(start: date, months: int) -> date:
    """The last day of a run of `months` months beginning at `start`."""
    return _end_of(*_add_months(start, months - 1).timetuple()[:2])


def _subspan(remainder: str, months: int = 12) -> tuple[int, int] | None:
    """A run cut out of a year, as (offset, length) in months, or None."""
    half = HALF.search(remainder)
    if half:
        index = (
            int(half.group("index"))
            if half.group("index")
            else (1 if half.group("edge").lower() == "first" else 2)
        )
        return (0, months // 2) if index == 1 else (months // 2, months - months // 2)

    match = SUBSPAN.search(remainder)
    if not match:
        return None

    unit = match.group("unit").lower()
    raw_count = match.group("count")
    if raw_count:
        count = COUNT_WORDS.get(raw_count.lower()) or int(raw_count)
    elif unit.startswith("quarter"):
        count = 1
    else:
        return None  # "months of FY24" names no length we can use

    length = count * (3 if unit.startswith("quarter") else 1)
    if not 0 < length < months:
        return None
    leading = match.group("edge").lower() in ("first", "initial")
    return (0, length) if leading else (months - length, length)


def restrict(base: Period, remainder: str) -> Period | None:
    """Narrow a whole-year interval by whatever else the label says, or refuse.

    A year label means that year only when the rest of the string adds nothing
    to it. `first eight months of FY25` is eight months, `FY20 to FY24` is five
    years and `FY25 (April-December)` is nine months; all three used to resolve
    to one whole fiscal year, because the year was matched anywhere in the
    string and everything around it discarded. A partial-year figure and a
    full-year figure then carried equal intervals, read as the same stated
    condition, and their disagreement came out `contradicts`. Two such pairs
    are in the shipped ledger, and both had to be rescued by the adjudicating
    model - which is the layer that is not supposed to be settling arithmetic.

    So: resolve the sub-span where the words name one exactly, and return None
    where they name something else. None is not a failure. The label survives
    as a string and is compared as one, which keeps two different labels
    different; silently widening a part into its whole is the only outcome
    that loses information.

    This is the discipline `bare_date` already applies to dates - a value that
    *mentions* a date is not a date - carried over to years.
    """
    months = 12
    cut = _subspan(remainder, months)
    consumed = HALF.sub(" ", SUBSPAN.sub(" ", remainder)) if cut is not None else remainder

    if any(word.lower() not in YEAR_FILLER for word in re.findall(r"[a-zA-Z0-9]+", consumed)):
        return None
    if cut is None:
        return base

    offset, length = cut
    start = _add_months(base.start, offset)
    return Period(
        start,
        _run_of_months(start, length),
        base.label,
        "quarter" if length == 3 else "months",
        base.fiscal_basis,
    )


def _outside(raw: str, match: re.Match) -> str:
    return f"{raw[: match.start()]} {raw[match.end() :]}"


def parse_period(text: str, fy_end_month: int | None = None) -> Period | None:
    """Resolve a written period label to an interval.

    Handles the shapes this document population actually uses: explicit "year
    ended" statements, partial periods such as "nine months ended", FY labels
    in several spellings, calendar years, and fiscal quarters.

    A label built around a year label means that whole year only when nothing
    else in the string narrows or extends it - see `restrict`.
    """
    if not text:
        return None
    raw = " ".join(text.split())
    basis_month = fy_end_month or DEFAULT_FY_END_MONTH
    known = fy_end_month is not None
    basis = f"{'stated' if known else 'assumed'}:fy_ends_{basis_month:02d}"

    ended = ENDED.search(raw)
    if ended:
        end = date(
            int(ended.group("year")),
            MONTHS[ended.group("month").lower()],
            int(ended.group("day")),
        )
        count, span = ended.group("count"), (ended.group("span") or "year").lower()
        months = 12
        granularity = "year"
        if count:
            number = COUNT_WORDS.get(count.lower(), None)
            if number is None:
                number = int(count)
            if span.startswith("month"):
                months, granularity = number, "months" if number != 12 else "year"
            elif span.startswith("quarter"):
                months, granularity = number * 3, "quarter" if number == 1 else "months"
            else:
                months, granularity = number * 12, "year"
        start = _add_months(_add_months(end, -months), 1).replace(day=1)
        stated = f"stated:ends_{end.month:02d}{end.day:02d}"
        return Period(start, end, raw.strip(), granularity, stated)

    quarter = QUARTER.search(raw)
    if quarter:
        index = int(quarter.group(1))
        tail = quarter.group(3) or quarter.group(2)
        end_year = _widen_year(tail, date.today().year)
        fy_end = _end_of(end_year, basis_month)
        fy_start = _add_months(date(end_year, basis_month, 1), -11)
        q_start = _add_months(fy_start, 3 * (index - 1))
        q_end = _end_of(*_add_months(q_start, 2).timetuple()[:2])
        return Period(q_start, q_end, raw.strip(), "quarter", basis)

    cy = CY.search(raw)
    if cy:
        year = int(cy.group(1))
        whole = Period(
            date(year, 1, 1), date(year, 12, 31), raw.strip(), "year", "stated:calendar"
        )
        return restrict(whole, _outside(raw, cy))

    span = FY_SPLIT.search(raw)
    if span:
        first, second = span.group(1), span.group(2)
        end_year = _widen_year(second, int(first))
        if end_year <= int(first):
            return None
        whole = _fy_from_end_year(end_year, basis_month, raw.strip(), basis)
        return restrict(whole, _outside(raw, span))

    short = FY_SHORT.search(raw)
    if short:
        end_year = _widen_year(short.group(1), date.today().year)
        whole = _fy_from_end_year(end_year, basis_month, raw.strip(), basis)
        return restrict(whole, _outside(raw, short))

    return None


YEAR_4 = re.compile(r"\b(?:19|20)\d{2}\b")
YEAR_FY = re.compile(r"\bFY\s*[-']?\s*(\d{2,4})\b", re.I)


def mentioned_years(text: str) -> frozenset[int]:
    """Every calendar year a label names, however it spells them.

    Used to tell two labels that could not be parsed apart. "till FY26" and
    "inception till FY26" name the same year; "April-December 2024" and
    "April-December 2023" do not.
    """
    raw = text or ""
    years = {int(m.group()) for m in YEAR_4.finditer(raw)}
    years |= {_widen_year(m.group(1), date.today().year) for m in YEAR_FY.finditer(raw)}
    return frozenset(years)


def years_compatible(left: str, right: str) -> bool:
    """True when neither label names a year the other rules out."""
    a, b = mentioned_years(left), mentioned_years(right)
    return a <= b or b <= a


def same_interval(a: Period | None, b: Period | None) -> bool:
    return bool(a and b and a.start == b.start and a.end == b.end)


def overlaps(a: Period | None, b: Period | None) -> bool:
    return bool(a and b and a.start <= b.end and b.start <= a.end)
