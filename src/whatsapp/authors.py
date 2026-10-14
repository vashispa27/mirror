"""Safe target-author resolution for WhatsApp exports."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Sequence

from .models import WhatsAppMessage


class AuthorResolutionError(ValueError):
    """Raised when a target-author selector is not exactly one export author."""


_WHITESPACE = re.compile(r"\s+")


def normalize_author(value: str) -> str:
    """Normalize an author name for literal, Unicode-aware comparisons."""

    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).strip()).casefold()


def extract_authors(messages: Iterable[WhatsAppMessage]) -> tuple[str, ...]:
    """Extract unique, non-system authors in their first-seen export order."""

    authors: list[str] = []
    for message in messages:
        if message.is_system or not message.author:
            continue
        if message.author not in authors:
            authors.append(message.author)
    return tuple(authors)


def _format_authors(authors: Sequence[str]) -> str:
    return ", ".join(repr(author) for author in authors) or "(none)"


def _raise_match_error(selector: str, matches: Sequence[str], authors: Sequence[str]) -> None:
    if matches:
        raise AuthorResolutionError(
            "Target author {!r} is ambiguous; matches: {}. "
            "Use a more specific --target-author value.".format(
                selector, _format_authors(matches)
            )
        )
    raise AuthorResolutionError(
        "No export author matches {!r}. Available authors: {}.".format(
            selector, _format_authors(authors)
        )
    )


def _is_token_prefix_match(selector: str, candidate: str) -> bool:
    """Match only whole name-token prefixes, never an arbitrary substring."""

    selector_tokens = selector.split()
    candidate_tokens = candidate.split()
    if not selector_tokens or len(selector_tokens) > len(candidate_tokens):
        return False
    width = len(selector_tokens)
    for start in range(len(candidate_tokens) - width + 1):
        if all(
            candidate_tokens[start + offset].startswith(token)
            for offset, token in enumerate(selector_tokens)
        ):
            return True
    return False


def resolve_target_author(
    messages: Iterable[WhatsAppMessage],
    target_author: str,
) -> str:
    """Resolve a literal target name to one canonical name in an export.

    Exact normalized matches win.  Otherwise a selector may be an ordered run
    of whole-token prefixes (for example ``Rohan`` for ``Rohan Verma``).
    Both stages reject ambiguity rather than guessing an identity.
    """

    authors = extract_authors(messages)
    normalized_selector = normalize_author(target_author)
    if not normalized_selector:
        raise AuthorResolutionError("Target author must not be empty.")

    exact = [author for author in authors if normalize_author(author) == normalized_selector]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        _raise_match_error(target_author, exact, authors)

    partial = [
        author
        for author in authors
        if _is_token_prefix_match(normalized_selector, normalize_author(author))
    ]
    if len(partial) == 1:
        return partial[0]
    _raise_match_error(target_author, partial, authors)
    raise AssertionError("unreachable")


def resolve_target_author_regex(
    messages: Iterable[WhatsAppMessage],
    pattern: str,
) -> str:
    """Resolve an explicitly requested regular expression to exactly one author."""

    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise AuthorResolutionError(f"Invalid target-author regex: {exc}") from exc
    authors = extract_authors(messages)
    matches = [author for author in authors if compiled.search(author)]
    if len(matches) == 1:
        return matches[0]
    _raise_match_error(pattern, matches, authors)
    raise AssertionError("unreachable")
