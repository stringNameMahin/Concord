from dataclasses import dataclass

from rapidfuzz import fuzz

FUZZY_THRESHOLD = 88.0
WINDOW_MARGIN = 2000


@dataclass
class Alignment:
    """Where a quoted string really lives in the source text.

    `text` is what the document actually says, not what the model claimed it
    said. Storing the located substring rather than the model's version is what
    makes the round-trip guarantee falsifiable.
    """

    status: str
    start: int
    end: int
    text: str
    score: float

    @property
    def located(self) -> bool:
        return self.status != "unlocated"


def normalize_ws(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space, keeping a map back to offsets.

    A model quoting a table row writes "Revenue from services* 81,415.38" while
    the parsed text has a newline between the label and the figure. Matching on
    a normalized copy and mapping the hit back is what bridges that gap without
    ever guessing an offset.
    """
    out: list[str] = []
    index: list[int] = []
    in_space = False

    for offset, char in enumerate(text):
        if char.isspace():
            if not in_space:
                out.append(" ")
                index.append(offset)
                in_space = True
            continue
        out.append(char)
        index.append(offset)
        in_space = False

    return "".join(out), index


class Aligner:
    def __init__(self, text: str, threshold: float = FUZZY_THRESHOLD):
        self.text = text
        self.threshold = threshold
        self.norm, self.index = normalize_ws(text)
        self._norm_of = self._build_reverse()

    def _build_reverse(self) -> list[int]:
        reverse = [0] * (len(self.text) + 1)
        for norm_pos, orig_pos in enumerate(self.index):
            reverse[orig_pos] = norm_pos
        last = 0
        for offset in range(len(reverse)):
            if reverse[offset] == 0 and offset > 0:
                reverse[offset] = last
            else:
                last = reverse[offset]
        return reverse

    def _to_original(self, norm_start: int, norm_end: int) -> tuple[int, int]:
        start = self.index[norm_start]
        end = self.index[norm_end - 1] + 1
        return start, end

    def locate(self, quote: str, near: tuple[int, int] | None = None) -> Alignment:
        quote = quote.strip()
        if not quote:
            return Alignment("unlocated", -1, -1, "", 0.0)

        lo, hi = 0, len(self.text)
        if near:
            lo = max(0, near[0] - WINDOW_MARGIN)
            hi = min(len(self.text), near[1] + WINDOW_MARGIN)

        hit = self.text.find(quote, lo, hi)
        if hit == -1:
            hit = self.text.find(quote)
        if hit != -1:
            return Alignment("exact", hit, hit + len(quote), quote, 100.0)

        needle, _ = normalize_ws(quote)
        needle = needle.strip()
        n_lo, n_hi = self._norm_of[lo], self._norm_of[min(hi, len(self.text) - 1)]

        window = self.norm[n_lo:n_hi]
        found = window.find(needle)
        if found == -1:
            found = self.norm.find(needle)
            if found != -1:
                n_lo = 0
        if found != -1:
            start, end = self._to_original(n_lo + found, n_lo + found + len(needle))
            return Alignment("exact", start, end, self.text[start:end], 100.0)

        return self._fuzzy(needle, n_lo, n_hi)

    def _fuzzy(self, needle: str, n_lo: int, n_hi: int) -> Alignment:
        window = self.norm[n_lo:n_hi]
        if not window or not needle:
            return Alignment("unlocated", -1, -1, "", 0.0)

        match = fuzz.partial_ratio_alignment(needle, window, score_cutoff=self.threshold)
        if match is None:
            return Alignment("unlocated", -1, -1, "", 0.0)

        start, end = self._to_original(n_lo + match.dest_start, n_lo + match.dest_end)
        return Alignment("fuzzy", start, end, self.text[start:end], match.score)
