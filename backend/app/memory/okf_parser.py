"""Simple regex-based parser for Open Knowledge Format (OKF) facts."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

NON_NAME_WORDS = {
    "a", "an", "the", "not", "just", "so", "very", "also", "currently",
    "learning", "studying", "working", "living", "trying", "doing", "going",
    "wondering", "thinking", "looking", "writing", "reading", "using",
    "happy", "sad", "tired", "fine", "good", "okay", "ready", "sure", "here",
    "from", "in", "at", "based", "located", "interested", "excited", "sorry",
    "student", "developer", "engineer", "programmer", "teacher", "doctor",
    "freelancer", "designer", "researcher", "manager", "user", "human",
}


@dataclass
class ParsedFact:
    """Structured representation of an extracted knowledge item."""

    category: str
    text: str
    attribute: str
    value: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OKFParser:
    """Extracts structured knowledge categories and key-value facts using regex."""

    def __init__(self):
        # Rules: (pattern, category, default_attr_or_None, optional_validator_fn)
        self.rules: list[tuple[re.Pattern, str, str | None, Callable[[str], bool] | None]] = [
            # Explicit Name: "My name is Rahul." or "Call me Rahul."
            (
                re.compile(r"^\s*(?:my\s+name\s+is|call\s+me)\s+([A-Za-z0-9_ -]+)", re.IGNORECASE),
                "identity",
                "name",
                None,
            ),
            # Conversational Name: "I am Rahul." or "I'm Rahul." (with validator to prevent "I am learning Python")
            (
                re.compile(r"^\s*(?:i\s+am|i'm)\s+([A-Za-z0-9_ -]+)", re.IGNORECASE),
                "identity",
                "name",
                self._is_valid_name,
            ),
            # Location: "I live in Hyderabad."
            (
                re.compile(
                    r"^\s*(?:i\s+live\s+(?:in|at)|i\s+am\s+(?:based|located)\s+in)\s+([A-Za-z0-9_ ,.-]+)",
                    re.IGNORECASE,
                ),
                "location",
                "location",
                None,
            ),
            # Specific Favorite: "My favorite language is Python."
            (
                re.compile(
                    r"^\s*my\s+favorite\s+([A-Za-z0-9_ -]+?)\s+(?:is|are)\s+([A-Za-z0-9_ ,.-]+)",
                    re.IGNORECASE,
                ),
                "preference",
                None,  # dynamic attribute from group 1
                None,
            ),
            # Preference / Likes: "I prefer Python." / "I like AI." / "I love coding."
            (
                re.compile(
                    r"^\s*i\s+(?:prefer|like|love)\s+([A-Za-z0-9_ ,.-]+)",
                    re.IGNORECASE,
                ),
                "preference",
                "preference",
                None,
            ),
            # Education / Study: "I study AI." or "I am studying Computer Science."
            (
                re.compile(
                    r"^\s*i\s+(?:study|am\s+studying)\s+([A-Za-z0-9_ ,.-]+)",
                    re.IGNORECASE,
                ),
                "education",
                "field_of_study",
                None,
            ),
            # Education / Major: "My major is Computer Science."
            (
                re.compile(
                    r"^\s*my\s+major\s+is\s+([A-Za-z0-9_ ,.-]+)",
                    re.IGNORECASE,
                ),
                "education",
                "major",
                None,
            ),
            # Work / Profession: "I work at Google." or "I work as a software engineer."
            (
                re.compile(
                    r"^\s*i\s+work\s+(?:as\s+(?:a|an)\s+|at\s+|in\s+)?([A-Za-z0-9_ ,.-]+)",
                    re.IGNORECASE,
                ),
                "profession",
                "work",
                None,
            ),
            # Generic personal statements: "My hobby is hiking."
            (
                re.compile(
                    r"^\s*my\s+([A-Za-z0-9_ -]+?)\s+(?:is|are)\s+([A-Za-z0-9_ ,.-]+)",
                    re.IGNORECASE,
                ),
                "personal_info",
                None,  # dynamic attribute from group 1
                None,
            ),
        ]

    @staticmethod
    def _clean(val: str) -> str:
        """Strip trailing punctuation and extra whitespace."""
        return val.strip().strip(".!?,;\"'").strip()

    @classmethod
    def _is_valid_name(cls, value: str) -> bool:
        """Verify whether an extracted value is a legitimate person's name."""
        cleaned = cls._clean(value)
        if not cleaned:
            return False

        tokens = cleaned.split()
        # Names are typically 1 to 4 words
        if len(tokens) > 4:
            return False

        first = tokens[0].lower()
        if first in NON_NAME_WORDS or first.endswith("ing"):
            return False

        # Check if any token is a non-name verb/adjective/preposition
        if any(t.lower() in NON_NAME_WORDS or t.lower().endswith("ing") for t in tokens):
            return False

        # Only allow alphabetic tokens, hyphens, and apostrophes
        if not all(part.replace("-", "").replace("'", "").isalpha() for part in tokens):
            return False

        return True

    def parse_fact(self, sentence: str) -> ParsedFact | None:
        """Parse a single sentence into a structured ParsedFact."""
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence or len(cleaned_sentence) > 500:
            return None

        for pattern, category, default_attr, validator in self.rules:
            match = pattern.search(cleaned_sentence)
            if not match:
                continue

            groups = match.groups()
            if default_attr is None:
                # Dynamic attribute (e.g. "favorite language" -> attribute="favorite_language", value="Python")
                if len(groups) >= 2:
                    attr_name = self._clean(groups[0]).lower()
                    attr_name = re.sub(r"\s+", "_", attr_name)
                    if category == "preference" and not attr_name.startswith("favorite_"):
                        attr_name = f"favorite_{attr_name}"
                    value = self._clean(groups[1])
                else:
                    attr_name = "attribute"
                    value = self._clean(groups[0])
            else:
                attr_name = default_attr
                value = self._clean(groups[0])

            if not value:
                continue

            # Run validator if present (e.g. to ensure "I am learning Python" is rejected as a name)
            if validator is not None and not validator(value):
                continue

            metadata = {
                "category": category,
                "attribute": attr_name,
                "value": value,
                "raw_text": cleaned_sentence,
            }

            return ParsedFact(
                category=category,
                text=cleaned_sentence,
                attribute=attr_name,
                value=value,
                metadata=metadata,
            )

        return None

    def parse_text(self, text: str) -> list[ParsedFact]:
        """Split a block of text into sentences and extract all matching facts."""
        sentences = re.split(r"[.\n;]+", text)
        facts: list[ParsedFact] = []

        for sentence in sentences:
            fact = self.parse_fact(sentence)
            if fact:
                facts.append(fact)

        return facts


# Module-level instance for convenience
default_parser = OKFParser()


def parse_fact(sentence: str) -> ParsedFact | None:
    """Convenience helper to parse a fact using the default parser."""
    return default_parser.parse_fact(sentence)
