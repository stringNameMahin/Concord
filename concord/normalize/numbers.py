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
}

CURRENCIES = {
    "₹": "INR",
    "rs": "INR",
    "rs.": "INR",
    "inr": "INR",
    "रू": "INR",
    "$": "USD",
    "us$": "USD",
    "usd": "USD",
    "€": "EUR",
    "eur": "EUR",
    "£": "GBP",
    "gbp": "GBP",
    "¥": "JPY",
}

NIL = {"nil", "-", "–", "—", "na", "n/a", "none", ""}

BOUNDS = [
    (">=", "at_least"),
    ("<=", "at_most"),
    ("≥", "at_least"),
    ("≤", "at_most"),
    (">", "greater_than"),
    ("<", "less_than"),
    ("~", "about"),
    ("≈", "about"),
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

NUMBER = re.compile(r"\d[\d, \s]*(?:\.\d+)?")
SCALE_WORD = re.compile(r"[A-Za-z]+\.?")


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
    return re.sub(r"[, \s]", "", text)


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
    is_percent = tail.startswith("%") or "percent" in tail.lower() or "bps" in tail.lower()

    scale = None
    factor = 1.0
    word = SCALE_WORD.match(tail)
    if word:
        candidate = word.group().lower().rstrip(".")
        if candidate in SCALES:
            scale = candidate
            factor = SCALES[candidate]

    if scale is None and not is_percent and default_scale:
        candidate = default_scale.lower().rstrip(".")
        if candidate in SCALES:
            scale = candidate
            factor = SCALES[candidate]

    if unit is None and not is_percent:
        unit = default_unit

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
