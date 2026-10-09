"""Model-name matching between free text and the hardware CSVs.

Plain substring matching picks the wrong part whenever one model name is a
prefix of another ("Ryzen 9 9950X" also matches "Ryzen 9 9950X3D") and
misses names written differently ("GeForce RTX 5080", "RTX 5070Ti").
This module matches on tokens instead:

* Text is split into letter runs and digit runs, so "5070Ti", "5070 Ti"
  and "5070-Ti" all become ``["5070", "ti"]``.
* Each model is known by its full name and by its name from the model
  number onward ("Ryzen 7 9800X3D" is also just "9800X3D").
* A mention must include the model number, so "Ryzen 7" alone matches
  nothing rather than every Ryzen 7. It must also start and end on word
  boundaries, so "7600X3D" never matches as "7600X".
* At each position the longest match wins, and an exact name beats a
  longer model that merely starts the same way. "RTX 4060" is the RTX
  4060, while "RTX 4060 Ti" is ambiguous between the 8GB and 16GB cards.

Pure Python with no project imports, so it can be tested on its own.
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z]+|\d+")


def tokenize(text: str) -> tuple[str, ...]:
    """Split text into lowercase letter runs and digit runs.

    Args:
        text: Any string, e.g. ``"Core i7-14700K"``.

    Returns:
        Token tuple, e.g. ``("core", "i", "7", "14700", "k")``.
    """
    return tuple(_TOKEN_RE.findall(str(text).lower()))


def _tokenize_with_bounds(text: str) -> tuple[tuple[str, ...], set[int], set[int]]:
    """Tokenize, also noting which tokens start and end a word.

    "7600X3D" is one word made of four tokens, so a match may not stop
    after its "7600X" part.

    Returns:
        ``(tokens, starts, ends)`` where ``starts``/``ends`` hold the
        indices of tokens that begin/finish a run of letters and digits.
    """
    text = str(text).lower()
    spans = [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    tokens = tuple(tok for tok, _, _ in spans)
    starts = {
        i for i, (_, s, _) in enumerate(spans)
        if s == 0 or not text[s - 1].isalnum()
    }
    ends = {
        i for i, (_, _, e) in enumerate(spans)
        if e == len(text) or not text[e].isalnum()
    }
    return tokens, starts, ends


def same_model(a: str, b: str) -> bool:
    """Return True if two names refer to the same model, ignoring case,
    spacing, and punctuation."""
    return tokenize(a) == tokenize(b)


def _number_index(tokens: tuple[str, ...]) -> int | None:
    """Index of the model-number token (first run of 3+ digits), if any."""
    for i, tok in enumerate(tokens):
        if tok.isdigit() and len(tok) >= 3:
            return i
    return None


@dataclass(frozen=True)
class Mention:
    """One place in the text that refers to hardware.

    Attributes:
        models: Every model the mention could mean, in CSV order. More
            than one means the text was ambiguous (e.g. "RTX 4060 Ti").
        exact: True if the text named a model exactly rather than only
            the start of one or more longer names.
    """

    models: tuple[str, ...]
    exact: bool

    @property
    def best(self) -> str:
        """The single most likely model: the shortest candidate name."""
        return min(self.models, key=lambda m: len(tokenize(m)))


class ModelIndex:
    """Finds mentions of known model names in free text."""

    def __init__(self, models: Iterable[str]):
        """Index the given model names.

        Args:
            models: Model names, e.g. a DataFrame's ``Model`` column.
                Duplicates and non-strings are ignored.
        """
        self.models: list[str] = []
        # Each alias is (tokens, number_index, model).
        self._aliases: list[tuple[tuple[str, ...], int, str]] = []
        for model in models:
            if not isinstance(model, str) or model in self.models:
                continue
            tokens = tokenize(model)
            num = _number_index(tokens)
            if num is None:
                continue
            self.models.append(model)
            self._aliases.append((tokens, num, model))
            if num > 0:
                self._aliases.append((tokens[num:], 0, model))
        self._max_len = max((len(t) for t, _, _ in self._aliases), default=0)

    def _match_at(
        self, tokens: tuple[str, ...], ends: set[int], i: int,
    ) -> tuple[int, Mention] | None:
        """Longest valid match starting at ``tokens[i]``, as (length, mention).

        The match must end on a word boundary (``ends``); the caller
        checks that it starts on one.
        """
        for k in range(min(self._max_len, len(tokens) - i), 0, -1):
            if i + k - 1 not in ends:
                continue
            window = tokens[i:i + k]
            candidates: list[str] = []
            exact: list[str] = []
            for alias, num, model in self._aliases:
                # The window must reach the model number.
                if k > num and alias[:k] == window:
                    if model not in candidates:
                        candidates.append(model)
                    if len(alias) == k and model not in exact:
                        exact.append(model)
            if candidates:
                if exact:
                    return k, Mention(tuple(exact), exact=True)
                return k, Mention(tuple(candidates), exact=False)
        return None

    def find_all(self, text: str) -> list[Mention]:
        """Find every hardware mention in the text, left to right.

        Args:
            text: Free text, e.g. ``"RTX 5080 vs 9070 XT at 4K"``.

        Returns:
            One ``Mention`` per referenced model or ambiguous group.
        """
        tokens, starts, ends = _tokenize_with_bounds(text)
        mentions: list[Mention] = []
        i = 0
        while i < len(tokens):
            hit = self._match_at(tokens, ends, i) if i in starts else None
            if hit is None:
                i += 1
                continue
            k, mention = hit
            mentions.append(mention)
            i += k
        return mentions

    def resolve(self, name: str) -> Mention | None:
        """Resolve a single component name to its best mention.

        Exact mentions are preferred over partial ones; otherwise the
        first mention in the text is used.

        Args:
            name: A component name such as ``"Ryzen 9 9950X"``.

        Returns:
            The best ``Mention``, or None if no known model is named.
        """
        mentions = self.find_all(name)
        if not mentions:
            return None
        for mention in mentions:
            if mention.exact:
                return mention
        return mentions[0]


def ambiguity_note(query: str, mention: Mention) -> str:
    """Line telling the model which variant was assumed, or ``""``.

    Args:
        query: What the caller asked for.
        mention: The resolved mention.

    Returns:
        A one-line note when the query matched several models, else an
        empty string.
    """
    if len(mention.models) < 2:
        return ""
    others = [m for m in mention.models if m != mention.best]
    return (
        f"Note: '{query}' matches several models — assumed {mention.best}. "
        f"Other matches: {', '.join(others)}.\n"
    )
