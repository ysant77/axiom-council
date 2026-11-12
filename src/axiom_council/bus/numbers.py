"""Number grounding: find the numbers in a message and check them against cited evidence.

This module only reads text. It has no bus types, so any caller can reuse it.

What counts as a number (a "quantity"). The text is first normalised with Unicode NFKC. Circled
digits, fullwidth digits and superscripts become ASCII digits (`x` plus superscript two is
`x2`). A vulgar fraction becomes two numbers (one half is `1` and `2`). Other decimal digits
(Arabic-Indic and so on) are read as ASCII digits. Zero-width characters and soft hyphens are
removed.

* Integers and decimals: `12`, `0.67`, `.5`. A decimal comma is not read: `12,5` is `12` and `5`.
* Thousands separators: a comma or a space-like character (no-break, thin, hair) between
  groups of exactly three digits. `1,234` and `1234` are the same number. `1 234` (plain space)
  and `12,34` are two numbers, so formatting cannot hide a number.
* A sign counts when it touches the digits and does not follow a letter, digit or dot.
  `-5` is negative; `10-20` is the numbers 10 and 20. Unicode minus signs are signs.
  `+5` is 5. A negative number is only grounded by the same negative number.
* A percent is a number followed by `%` or `percent`. It matches a percent in the evidence.
  With the setting `percent_matches_fraction` it may also match the equal evidence fraction
  (`67%` and `0.67`). A plain number never matches a percent, and the reverse.
* A unit after the number is ignored: `3.5 GB` and `12ms` match on `3.5` and `12`. The unit
  is not checked, and scale suffixes are not expanded (`1.2k` is `1.2`).
* A number joined to a word is still a number: `accuracy-95%`, `score.95`, `x95`, `tasks18`,
  `pass_rate_95`. When in doubt, the digits are read as a number.
* Dictionary keys in a JSON payload are read as text. Numeric JSON values are read as numbers.

How a quantity is grounded: it must equal a value in the evidence text, in this order:

1. Exact value: `0.50` equals `0.5`.
2. Rounding. If the message shows a decimal point and at least `rounding_min_digits` digits
   after it, an evidence value matches when it rounds (half up) to the message value at the
   digits shown. So `0.67` is grounded by `0.6712`, and `0.68` is not. `0.7` shows one digit,
   below the default of two, so `0.6712` does not ground it. Integers are never rounded.
3. Relative tolerance. If `relative_tolerance` is above zero, a value matches when it is
   within that share of an evidence value of the same sign. The default is zero.

Matching is whole-number matching. `12` is not grounded by `112`, and an evidence id is not
evidence. Only the evidence text is read.

What is not a measurement, and is skipped in the message and in the evidence. A number
followed by `%` or `percent` is never skipped.

* List markers at the start of a line (`1.`, `2)`, optionally after a bullet).
* Ordinals (`3rd`) and `#47` style references.
* Code-like identifiers: an uppercase-led word with a digit (`R1.2`, `PM2.5`, `K100`,
  `Node3-8`), an uppercase-led word joined to a number (`F-12`, `COVID-19`) and a
  mixed-case word with a digit (`pH7`, `eBox15`).
* Names in the setting `identifier_allowlist` (`p99`) and names the caller passes, such as the
  model ids of the registry (`name4:26b`, with the `:tag`). The whole token must match, in any
  case. Every other lowercase name with digits (`node-5`, `x95`, `sha256`) is read as a
  number. Add such names to the setting or to the caller list.
* Dates: `2024-05-12`, `12/05/2024`, `March 5, 2024`, `5 March 2024`, `March 2024`.
* Versions: three or more dotted groups (`1.2.3`; `1.234.567` is read as a version too),
  `v2.0` (a lowercase or uppercase `v` and a dotted number), and a number after the word
  `version` or `ver`.

Not handled: Roman numerals and spelled-out numbers (`twelve`) are out of scope and are not
read. Scientific notation (`1e5`), fractions such as `3/4` and times such as `12:30` are read
as plain digit groups. A list marker in the middle of a line (`1) one 2) two`) is read as a
number. Hashes and UUIDs (`7cd6ad7`) are read as digit groups, so a message that quotes them
needs those digits in its evidence.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Iterable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Context, Decimal
from functools import lru_cache
from typing import Any

from axiom_council.settings import GroundingSettings

_INVISIBLE = {ord(c): None for c in "\u200b\u200c\u200d\u2060\ufeff\u00ad"}
_PUNCTUATION = {
    0x2212: "-",  # minus sign
    0x066B: ".",  # Arabic decimal separator
    0x066C: ",",  # Arabic thousands separator
}
# A space-like character between a digit and exactly three digits is a thousands separator.
# It must become a comma before NFKC, which would turn it into a plain space.
_GROUP_GAP = re.compile(r"(?<=\d)[\u00a0\u2008\u2009\u200a\u202f](?=\d{3}(?!\d))")

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|"
    r"December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept?|Oct|Nov|Dec)"
)
# A code-like token must end at a digit boundary and must not be followed by a percent sign:
# a number followed by a percent sign is always a measurement.
_NOT_PERCENT = r"(?!\d)(?![ \t]?(?:%|(?i:percent|per[ \t]cent)\b))"
# Things that look like numbers but are not measurements. They are consumed so that the
# number pattern never sees them. Order matters: the first alternative that matches wins.
_SKIP = (
    r"^[ \t]*(?:[-*\u2022][ \t]+)?\d{1,3}[.)](?=[ \t]|$)",  # list marker
    r"(?i:\b(?:version|ver)\b\.?[ \t]*)\d+(?:\.\d+)*",  # "version 3.12"
    rf"\d{{1,2}}(?:st|nd|rd|th)?[ \t]+(?:of[ \t]+)?{_MONTH}\b\.?(?:,?[ \t]+\d{{4}}(?!\d))?",
    rf"{_MONTH}\b\.?[ \t]+\d{{1,2}}(?:st|nd|rd|th)?(?!\d)(?:,?[ \t]+\d{{4}}(?!\d))?",
    rf"{_MONTH}\b\.?,?[ \t]+\d{{4}}(?!\d)",  # "March 2024"
    r"\d{4}-\d{2}-\d{2}(?!\d)",  # ISO date
    r"\d{1,4}/\d{1,2}/\d{1,4}(?!\d)",  # numeric date
    r"\d+(?:\.\d+){2,}",  # version, dotted date
    r"\d+(?:st|nd|rd|th)\b",  # ordinal
    r"#\d+",  # reference
    rf"(?<!\w)[vV]\d+(?:\.\d+)+(?!\d){_NOT_PERCENT}",  # "v2.0"
    rf"(?<!\w)[A-Z][A-Za-z_]*\d\w*(?:[.\-]\d+)*{_NOT_PERCENT}",  # R1.2, PM2.5, K100
    rf"(?<!\w)[A-Z][A-Za-z_]*(?:[.\-]\d+)+{_NOT_PERCENT}",  # F-12, COVID-19
    rf"(?<!\w)[a-z]+[A-Z][A-Za-z_]*\d\w*(?:[.\-]\d+)*{_NOT_PERCENT}",  # pH7, eBox15
)
_NUMBER = (
    r"(?P<sign>(?<![\w.])[-+])?"
    r"(?P<num>(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d+)?|(?<![\d.])\.\d+)"
    r"(?P<pct>[ \t]?(?:%|(?i:percent|per[ \t]cent)\b))?"
)


@lru_cache(maxsize=16)
def _tokenizer(allowlist: frozenset[str]) -> re.Pattern[str]:
    names = sorted((n for n in allowlist if n), key=len, reverse=True)
    allowed = (
        [rf"(?<!\w)(?i:{'|'.join(map(re.escape, names))})(?!\w){_NOT_PERCENT}"] if names else []
    )
    return re.compile("|".join((*allowed, *_SKIP, _NUMBER)), re.MULTILINE)


@dataclass(frozen=True)
class Quantity:
    """One number as written: its value, the decimal digits it shows and whether it is a percent."""

    value: Decimal
    digits: int
    percent: bool
    shown: str


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFKC", _GROUP_GAP.sub(",", text.translate(_INVISIBLE)))
    chars = (str(unicodedata.decimal(c)) if not c.isascii() and c.isdecimal() else c for c in text)
    return "".join(chars).translate(_PUNCTUATION)


def _quantity(match: re.Match[str]) -> Quantity:
    raw = match["num"]
    plain = raw.replace(",", "")
    negative = match["sign"] == "-"
    value = Decimal(plain)
    percent = match["pct"] is not None
    return Quantity(
        value=-value if negative else value,
        digits=len(plain.partition(".")[2]),
        percent=percent,
        shown=f"{'-' if negative else ''}{raw}{'%' if percent else ''}",
    )


def extract_quantities(text: str, allowlist: Collection[str] = ()) -> list[Quantity]:
    """Return every measurement-like number in `text`, in order of appearance."""
    pattern = _tokenizer(frozenset(allowlist))
    found = (m for m in pattern.finditer(_normalise(text)) if m["num"] is not None)
    return [_quantity(m) for m in found]


def _round(value: Decimal, digits: int) -> Decimal:
    ctx = Context(prec=len(value.as_tuple().digits) + digits + 2, rounding=ROUND_HALF_UP)
    return value.quantize(Decimal(1).scaleb(-digits), context=ctx)


class _Pool:
    """Evidence values of one kind. Exact and rounded lookups use sets, not scans."""

    def __init__(self, values: Iterable[Decimal]):
        self._exact = set(values)
        self._rounded: dict[int, set[Decimal]] = {}

    def has_exact(self, value: Decimal) -> bool:
        return value in self._exact

    def has_rounded(self, value: Decimal, digits: int) -> bool:
        if digits not in self._rounded:
            self._rounded[digits] = {_round(v, digits) for v in self._exact}
        return value in self._rounded[digits]

    def has_within(self, value: Decimal, tol: Decimal) -> bool:
        """True if some evidence value `e` has |value - e| <= tol * |e|. A linear scan."""
        return any(abs(value - e) <= tol * abs(e) for e in self._exact)


def _matches(
    value: Decimal, digits: int, pool: _Pool, rules: GroundingSettings, *, rounding: bool
) -> bool:
    if pool.has_exact(value):
        return True
    if rounding and pool.has_rounded(value, digits):
        return True
    tol = Decimal(str(rules.relative_tolerance))
    return tol > 0 and pool.has_within(value, tol)


def _is_grounded(q: Quantity, plain: _Pool, percent: _Pool, rules: GroundingSettings) -> bool:
    rounding = q.digits >= rules.rounding_min_digits
    if _matches(q.value, q.digits, percent if q.percent else plain, rules, rounding=rounding):
        return True
    if q.percent and rules.percent_matches_fraction:
        fraction = q.value.scaleb(-2)
        return _matches(fraction, q.digits + 2, plain, rules, rounding=rounding)
    return False


def ungrounded_numbers(
    body: str,
    evidence: Iterable[str],
    rules: GroundingSettings,
    identifiers: Collection[str] = (),
) -> list[str]:
    """Return the numbers in `body` (as written, once each) that no evidence text grounds.

    `identifiers` are extra names that are not measurements, added to the settings allowlist.
    """
    names = rules.identifier_allowlist | frozenset(identifiers)
    quantities = extract_quantities(body, names)
    if not quantities:
        return []
    cited = [q for t in evidence for q in extract_quantities(t, names)]
    plain = _Pool(q.value for q in cited if not q.percent)
    percent = _Pool(q.value for q in cited if q.percent)
    missing: dict[str, None] = {}
    for q in quantities:
        if q.shown not in missing and not _is_grounded(q, plain, percent, rules):
            missing[q.shown] = None
    return list(missing)


def payload_text(value: Any) -> str:
    """All text and numbers in a payload, keys included, one per line. Booleans are skipped."""
    return "\n".join(_leaves(value))


def _leaves(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, float):
        return [format(Decimal(repr(value)), "f")] if value == value else []
    if isinstance(value, dict):
        return [s for k, v in value.items() for s in (*_leaves(k), *_leaves(v))]
    if isinstance(value, list | tuple):
        return [s for v in value for s in _leaves(v)]
    return []
