import math
import re
from dataclasses import dataclass

SCALES = {
    "k": 1e3,
    "thousand": 1e3,
    "thousands": 1e3,
    "lakh": 1e5,
    "lakhs": 1e5,
    "lac": 1e5,
    "lacs": 1e5,
    "mn": 1e6,
    "m": 1e6,
    "million": 1e6,
    "millions": 1e6,
    "crore": 1e7,
    "crores": 1e7,
    "cr": 1e7,
    "bn": 1e9,
    "b": 1e9,
    "billion": 1e9,
    "billions": 1e9,
    "trillion": 1e12,
    "tn": 1e12,
    # Indian compound scales. `lakh crore` is how the RBI and the Union Budget
    # write a trillion, and it has to be matched as one phrase: read as `lakh`
    # alone it is out by a factor of ten million, and as `crore` alone by a
    # hundred thousand. Both misreadings are silent - the figure still parses
    # and still looks ordinary. Five facts in the shipped RBI report were
    # stored unscaled because the model returned this phrase correctly and this
    # table did not hold it.
    "lakh crore": 1e12,
    "lakh crores": 1e12,
    "lakhs crore": 1e12,
    "lakhs crores": 1e12,
    "thousand crore": 1e10,
    "thousand crores": 1e10,
}

CURRENCIES = {
    "\u20b9": "INR",             # rupee sign
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "\u0930\u0942": "INR",         # rupee, Devanagari
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "\u20ac": "EUR",             # euro sign
    "eur": "EUR",
    "\u00a3": "GBP",             # pound sign
    "gbp": "GBP",
    "\u00a5": "JPY",             # yen sign
}

# Written currency names, matched as substrings so "Indian Rupees" and
# "US dollars" both resolve. Symbols live in CURRENCIES above.
CURRENCY_WORDS = {
    "rupee": "INR",
    "dollar": "USD",
    "euro": "EUR",
    "pound sterling": "GBP",
    "yen": "JPY",
}

# A percentage is a kind of number, not a unit of measure. When the figure
# itself is bare and the context supplies "per cent", it must reach the same
# state as if the cell had read "6.5%" - otherwise two spellings of the same
# word compare as two different units.
PERCENT_WORDS = {"%", "percent", "per cent", "pct", "percentage", "percentage points", "pp"}

NIL = {"nil", "-", "\u2013", "\u2014", "na", "n/a", "none", ""}

BOUNDS = [
    (">=", "at_least"),
    ("<=", "at_most"),
    ("\u2265", "at_least"),
    ("\u2264", "at_most"),
    (">", "greater_than"),
    ("<", "less_than"),
    ("~", "about"),
    ("\u2248", "about"),
    ("over", "greater_than"),
    ("above", "greater_than"),
    ("more than", "greater_than"),
    ("at least", "at_least"),
    ("under", "less_than"),
    ("below", "less_than"),
    ("less than", "less_than"),
    ("upto", "at_most"),
    ("up to", "at_most"),
    ("approximately", "about"),
    ("about", "about"),
    ("around", "about"),
    ("circa", "about"),
]

NUMBER = re.compile(r"\d[\d,\u00a0\s]*(?:\.\d+)?")
SCALE_WORD = re.compile(r"[A-Za-z]+\.?")


def _scale_key(text: str | None) -> str:
    """Fold a written scale to its `SCALES` key.

    Case, a trailing full stop, and the line break a PDF drops into the middle
    of `lakh\\ncrore` are all noise; the phrase names the same scale either way.
    """
    return " ".join((text or "").lower().rstrip(".").split())


# Every scale this module knows, longest first so `lakh crore` wins over
# `lakh`, `millions` over `million` and `crores` over `cr`. Built from SCALES
# rather than restated, so the two cannot drift, and with the space inside a
# compound relaxed to any whitespace because a PDF breaks the line between its
# two halves as readily as not.
_SCALE_ALT = "|".join(
    re.escape(word).replace(r"\ ", r"\s+")
    for word in sorted(SCALES, key=len, reverse=True)
)

# The same alternation anchored at the start of a string, for reading the scale
# off the tail of the figure itself.
LEADING_SCALE = re.compile(rf"({_SCALE_ALT})\b\.?", re.I)

# Any scale word standing as a whole word, for lifting a magnitude out of a unit
# the model wrote as a column header (`INR crores`). Word-bounded on both sides
# so `k` does not bite `kg` or `kWh`.
WHOLE_SCALE_WORD = re.compile(rf"\b(?:{_SCALE_ALT})\b\.?", re.I)

_CURRENCY_ALT = (
    r"[\u20b9$\u20ac\u00a3\u00a5]|rs\.?|inr|usd|eur|gbp|"
    r"indian\s+rupees?|us\s+dollars?|rupees?|dollars?|euros?|pounds?"
)

# A scale word sitting immediately after a figure, with nothing between them
# but a closing parenthesis, a currency token and a little whitespace.
# Deliberately tight: a table row holds several figures, and a scale word three
# words away belongs to a different one.
#
# The two things allowed in between are the two that cannot themselves be
# another figure. The `)` is the accounting negative, which wraps the figure
# alone, so `(217) Cr` is minus two hundred and seventeen crore. The currency
# token is how a written-out claim spells one quantity - `622 INR crores` -
# and a currency word cannot smuggle in a second figure for the scale to
# belong to instead. Without it the two halves of a restated pair recovered
# asymmetrically: `622 INR crores` kept its scale, `Trade payables 622` did
# not, and two statements of one figure came out as a contradiction.
TRAILING_SCALE = re.compile(
    rf"\)?[\s\u00a0]{{0,3}}(?:(?:{_CURRENCY_ALT})[\s\u00a0]{{0,3}})?({_SCALE_ALT})\b\.?",
    re.I,
)


def is_scale_word(text: str | None) -> bool:
    """Does this name a magnitude rather than a thing being measured?

    Used to tell the model's two units fields apart when it fills them the
    wrong way round, which it does: on the RBI report it returned `scale="₹"`
    and `unit="lakh crore"` for five figures, exactly reversed. Both fields are
    free text, so neither can be trusted by position alone.
    """
    return _scale_key(text) in SCALES


def truncates(stated: str | None, recovered: str | None) -> bool:
    """Is `stated` only part of the compound scale `recovered` names?

    `lakh` against `lakh crore` is out by a factor of ten million, and `crore`
    against it by a hundred thousand - both silent, because either half parses
    as a magnitude on its own. A model that answers with one word of a two-word
    phrase the page wrote has not contradicted the document, it has read part of
    it, so the document wins. `million` against `cr` is a real disagreement -
    neither contains the other - and there the model still wins.
    """
    if not stated or not recovered:
        return False
    left, right = _scale_key(stated).split(), _scale_key(recovered).split()
    if not left or len(left) >= len(right):
        return False
    return any(right[i : i + len(left)] == left for i in range(len(right) - len(left) + 1))


def _whole_number_at(text: str, start: int, end: int) -> bool:
    """Is `text[start:end]` a figure in its own right, not part of a longer one?

    The needle is a digit run, so it matches inside any longer number that
    contains it - and the match that wins is whichever one a scale word happens
    to follow. Searching for `6` in "reaching USD 602.6 billion, witnessing a
    YoY growth of 6 per cent" found the final digit of `602.6`, read `billion`
    off it, and stored six per cent as six billion. A digit, a thousands comma
    or a decimal point on either side means the run belongs to a bigger figure
    and says nothing about this one.
    """
    boundary = "0123456789.,"
    if start > 0 and text[start - 1] in boundary:
        return False
    return end >= len(text) or text[end] not in boundary


def scale_after_figure(figure: str, text: str) -> str | None:
    """The scale the document wrote immediately after this figure.

    A model handed a sentence reading "81,415.38 million" routinely returns
    `raw="81,415.38"` with `scale` left null, and the record is then wrong by a
    factor of a million while looking entirely ordinary - the quote is right,
    the displayed value is right, and only `normalized` is wrong. The dropped
    magnitude is sitting in the bytes the aligner already located, so reading
    it back is the same deterministic guard the quote gets from the aligner and
    the subject gets from `subject_names_the_measurement`. The magnitude was
    the last load-bearing field still taken on the model's word.

    Returns None unless the figure occurs in `text` as a whole number with a
    scale directly behind it. Every other case - the figure absent, a bare
    figure, a scale word further away - is silence, because a wrong magnitude is
    worse than a missing one.
    """
    digits = NUMBER.search(figure or "")
    if digits is None or not text:
        return None

    # Match on the digits alone. Everything ahead of them in `raw` is a
    # currency symbol, an opening parenthesis or a comparison operator, and the
    # document spaces those differently from the model - `Rs8,142` in the
    # record against `Rs 8,142` on the page.
    needle = digits.group().strip()
    if not needle:
        return None

    for match in re.finditer(re.escape(needle), text):
        if not _whole_number_at(text, match.start(), match.end()):
            continue
        trailing = TRAILING_SCALE.match(text, match.end())
        if trailing:
            return _scale_key(trailing.group(1))
    return None


@dataclass
class Quantity:
    """A number parsed into base units, with the precision it was written to.

    `interval` is derived from how many digits the source actually committed
    to, so 8,142 Cr and 81,415.38 million can be compared without inventing a
    tolerance. Comparison is always interval against interval, never point
    against point.
    """

    raw: str
    number: float
    unit: str | None
    scale: str | None
    scale_factor: float
    normalized: float
    interval: tuple[float, float]
    bound: str | None
    is_percent: bool

    @property
    def is_bounded(self) -> bool:
        return self.bound in ("greater_than", "less_than", "at_least", "at_most")


class NotANumber(ValueError):
    pass


def _decimals(digits: str) -> int:
    return len(digits.split(".")[1]) if "." in digits else 0


def _strip_grouping(text: str) -> str:
    return re.sub(r"[,\u00a0\s]", "", text)


def is_percent_unit(text: str | None) -> bool:
    """Does this unit name a percentage rather than a thing being measured?"""
    if not text:
        return False
    return " ".join(text.strip().lower().replace("-", " ").split()) in PERCENT_WORDS


def _currency_code(text: str) -> str | None:
    """The ISO code this text names, or None if it names no currency."""
    key = text.lower().strip().rstrip(".")
    if key in CURRENCIES:
        return CURRENCIES[key]
    for word, code in CURRENCY_WORDS.items():
        if word in key:
            return code
    for symbol, code in CURRENCIES.items():
        if not symbol.isalpha() and symbol in text:
            return code
    return None


def normalize_currency(text: str | None) -> str | None:
    """Resolve a written currency to its ISO code, leaving other units alone.

    A currency reaches a figure three ways - as a symbol in the cell, as a code
    in a header, as words in a footnote - and the same money must compare equal
    however it was written. Without this, a figure carrying the inherited unit
    written as the rupee sign and one carrying `INR` are refused as incomparable,
    which looks like caution and is actually a bug. A unit that names no
    currency (`Tons`, `days`) is returned unchanged.

    **A magnitude in the unit field is not part of the unit.** The model writes
    the column header verbatim, so `INR crores`, `INR crore` and `USD billion`
    all arrive as units while the magnitude is *also* sitting in
    `Quantity.scale`, already applied. Left alone they are four different
    identities for two currencies, and `compare_values` refuses every pair that
    spans them: on the shipped ledger 140 facts, and 12 of the 14
    `incomparable_values` relations - including `revenue_from_operations 54,364`
    against `revenue_from_operations 54,364`, the same figure declared
    incomparable with itself because one side said `INR` and the other
    `INR crore`.

    The strip is deliberately conditional: it only stands when what is left
    names a currency. `per one million-person hours worked` and `lakh metric
    tonnes` are units whose magnitude is part of their meaning, and rewriting
    those would be a new bug in place of this one.
    """
    if not text:
        return None

    raw = text.strip()
    code = _currency_code(raw)
    if code:
        return code

    stripped = WHOLE_SCALE_WORD.sub(" ", raw).strip()
    if stripped != raw:
        code = _currency_code(stripped)
        if code:
            return code
    return raw


def parse_quantity(raw: str, default_scale: str | None = None,
                   default_unit: str | None = None) -> Quantity:
    """Parse a written figure into base units without ever inferring magnitude.

    Scale and currency fall back to the enclosing context only when the figure
    itself is bare, which is the normal case inside a table whose header
    carries the units.
    """
    text = (raw or "").strip()
    lowered = text.lower().strip()

    if lowered in NIL:
        raise NotANumber(f"not a quantity: {raw!r}")

    body = text
    bound = None
    scan = body.lower()
    for token, name in BOUNDS:
        if scan.lstrip().startswith(token):
            bound = name
            body = body.lstrip()[len(token):]
            break

    unit = None
    for symbol, code in CURRENCIES.items():
        idx = body.lower().find(symbol)
        if idx != -1 and idx <= 3:
            unit = code
            body = body[:idx] + body[idx + len(symbol):]
            break

    match = NUMBER.search(body)
    if not match:
        raise NotANumber(f"no digits in {raw!r}")

    digits = _strip_grouping(match.group())
    try:
        value = float(digits)
    except ValueError as exc:
        raise NotANumber(f"cannot parse {raw!r}") from exc

    # Accounting negatives wrap the figure alone, not the whole cell, so
    # "(452) Cr" has to keep its scale word while flipping sign.
    before = body[: match.start()].rstrip()
    after = body[match.end():].lstrip()
    negative = before.endswith("(") and ")" in after
    if negative:
        after = after.replace(")", "", 1)

    tail = after.strip()
    # Percentness is settled before any scale is applied, and it has two
    # sources: the figure's own tail, and the unit. Deciding it from the tail
    # alone and consulting the unit afterwards let a percentage take a scale it
    # should have refused - "6" with unit "per cent" was handed `billion` from
    # the surrounding bytes and stored as six billion, because `not is_percent`
    # was still true when the scale was applied and only became false three
    # lines later. Both sources have to be read first for the guard to bind.
    is_percent = (
        tail.startswith("%")
        or "percent" in tail.lower()
        or "bps" in tail.lower()
        or is_percent_unit(default_unit)
    )

    scale = None
    word = LEADING_SCALE.match(tail)
    if word:
        candidate = _scale_key(word.group(1))
        if candidate in SCALES:
            scale = candidate

    # The figure's own tail wins, except where the context names a compound the
    # tail is only the first half of. A model handed "an outlay of Rs 1.5 lakh
    # crore" returned `raw="Rs1.5 lakh"`, dropping the second word *inside* the
    # figure - so the scale read off the tail is `lakh` and the document says
    # `lakh crore`, a factor of ten million apart. Same rule as `materialize`
    # applies to the model's `scale` field: a truncation is not a disagreement.
    if not is_percent and truncates(scale, default_scale):
        scale = _scale_key(default_scale)

    if scale is None and not is_percent and default_scale:
        candidate = _scale_key(default_scale)
        if candidate in SCALES:
            scale = candidate

    factor = SCALES.get(scale, 1.0) if scale else 1.0

    if unit is None and not is_percent:
        unit = normalize_currency(default_unit)

    if tail.lower().startswith("bps"):
        value, is_percent = value / 100.0, True

    if negative:
        value = -value

    normalized = value * factor
    interval = precision_interval(normalized, _decimals(digits), factor, bound)

    return Quantity(
        raw=text,
        number=value,
        unit=unit if not is_percent else None,
        scale=scale,
        scale_factor=factor,
        normalized=normalized,
        interval=interval,
        bound=bound,
        is_percent=is_percent,
    )


def precision_interval(
    normalized: float, decimals: int, factor: float, bound: str | None
) -> tuple[float, float]:
    """Half a unit of the last written digit, carried into base units.

    A figure written 8,142 Cr committed to whole crore, so it stands for
    anything in [8141.5, 8142.5] crore. Written 81,415.38 million it committed
    two decimals of a million. Those two intervals overlap, which is why the
    same underlying fact reported at different scales corroborates instead of
    contradicting.
    """
    half = 0.5 * (10.0**-decimals) * factor

    if bound in ("greater_than", "at_least"):
        return (normalized, math.inf)
    if bound in ("less_than", "at_most"):
        return (-math.inf, normalized)
    if bound == "about":
        widened = max(half, abs(normalized) * 0.05)
        return (normalized - widened, normalized + widened)

    return (normalized - half, normalized + half)


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Strict overlap, because a precision interval is half-open.

    8,142 Cr and 8,143 Cr meet at exactly 8,142.5 Cr. Counting that shared
    endpoint as agreement would mean two consecutive figures written to the
    same precision could never disagree, which would quietly disable
    contradiction detection for every rounded integer in the corpus.
    """
    return a[0] < b[1] and b[0] < a[1]
