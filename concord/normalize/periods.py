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

MONTH_RE = "|".join(sorted(MONTHS, key=len, reverse=True))
COUNT_RE = "|".join(COUNT_WORDS)

ENDED = re.compile(
    rf"\b(?:(?P<count>{COUNT_RE}|\d{{1,2}})[\s-]+(?P<span>month|months|quarter|year)s?"
    rf"[\s\w]{{0,12}}?)?\bended?\s+(?:on\s+)?(?P<month>{MONTH_RE})\.?\s+(?P<day>\d{{1,2}}),?\s+"
    rf"(?P<year>\d{{4}})",
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
QUARTER = re.compile(r"\bQ([1-4])\s*[-,]?\s*(?:FY|F\.Y\.?)?\s*(\d{2,4})(?:\s*[-/]\s*(\d{2,4}))?\b", re.I)


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


def infer_fiscal_year_end(text: str) -> int | None:
    """Read the document's own fiscal year end rather than assuming one.

    A filing that says "for the year ended March 31, 2024" has told us its
    basis. Only when nothing in the document says so does the configured
    default apply, and the period is then marked assumed.
    """
    counts: dict[int, int] = {}
    for match in ENDED.finditer(text):
        if match.group("span") and match.group("span").lower().startswith(("month", "quarter")):
            continue
        month = MONTHS[match.group("month").lower()]
        counts[month] = counts.get(month, 0) + 1
    return max(counts, key=counts.get) if counts else None


def parse_date(text: str) -> date | None:
    """Resolve a written date in any of the orders these documents use.

    Dates are compared as dates and never as strings, so "March 31, 2024" and
    "31 March 2024" have to reach the same value. Numeric dates are read
    day-first, matching `parse_as_of`. A string that does not resolve returns
    None and is then compared as text rather than guessed at.
    """
    raw = text or ""
    for pattern in (BARE_DATE, DAY_FIRST_DATE):
        match = pattern.search(raw)
        if match:
            try:
                return date(
                    int(match.group("year")),
                    MONTHS[match.group("month").lower()],
                    int(match.group("day")),
                )
            except ValueError:
                return None

    numeric = NUMERIC_DATE.search(raw)
    if numeric:
        day, month, year = (int(group) for group in numeric.groups())
        if month <= 12:
            try:
                return date(year, month, day)
            except ValueError:
                return None
    return None


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


def parse_period(text: str, fy_end_month: int | None = None) -> Period | None:
    """Resolve a written period label to an interval.

    Handles the shapes this document population actually uses: explicit "year
    ended" statements, partial periods such as "nine months ended", FY labels
    in several spellings, calendar years, and fiscal quarters.
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
        return Period(date(year, 1, 1), date(year, 12, 31), raw.strip(), "year", "stated:calendar")

    span = FY_SPLIT.search(raw)
    if span:
        first, second = span.group(1), span.group(2)
        end_year = _widen_year(second, int(first))
        if end_year <= int(first):
            return None
        return _fy_from_end_year(end_year, basis_month, raw.strip(), basis)

    short = FY_SHORT.search(raw)
    if short:
        end_year = _widen_year(short.group(1), date.today().year)
        return _fy_from_end_year(end_year, basis_month, raw.strip(), basis)

    return None


def same_interval(a: Period | None, b: Period | None) -> bool:
    return bool(a and b and a.start == b.start and a.end == b.end)


def overlaps(a: Period | None, b: Period | None) -> bool:
    return bool(a and b and a.start <= b.end and b.start <= a.end)
