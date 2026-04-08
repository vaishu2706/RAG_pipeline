"""
Guardrails: input validation and output safety for the RAG pipeline.

Covers:
- Prompt injection attempts
- Jailbreak patterns
- Overly vague / one-word queries
- Query length (embedding model + LLM context window limits)
- Harmful / off-topic output detection
"""

import re

# Max chars sent to embedding model (text-embedding-3-small supports ~8191 tokens ≈ 32K chars)
MAX_QUERY_CHARS = 2000

# Patterns that signal prompt injection or jailbreak attempts
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"you\s+are\s+now\s+(?:a\s+)?(?:an?\s+)?\w+\s*(?:without|with no)\s+restrictions?",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"act\s+as\s+(if\s+you\s+(are|were)\s+)?(?:a\s+)?(?:an?\s+)?\w+\s*(?:without|with no)\s+",
    r"do\s+anything\s+now",
    r"dan\s+mode",
    r"developer\s+mode",
    r"show\s+(me\s+)?(your\s+)?(system\s+prompt|source\s+code|instructions?|prompt)",
    r"reveal\s+(your\s+)?(system\s+prompt|source\s+code|instructions?|prompt)",
    r"print\s+(your\s+)?(system\s+prompt|instructions?)",
    r"bypass\s+(your\s+)?(safety|filter|restriction|guideline)",
    r"override\s+(your\s+)?(safety|filter|restriction|guideline)",
    r"translate\s+the\s+above",
    r"repeat\s+(everything|all)\s+(above|before|prior)",
]

_COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

# Output phrases that suggest the LLM hallucinated despite having context
_HALLUCINATION_SIGNALS = [
    "as an ai language model",
    "i was trained on",
    "my training data",
    "i cannot access the internet",
    "i don't have access to real-time",
]


class GuardrailViolation(Exception):
    """Raised when a guardrail check fails."""
    pass


def check_input(query: str) -> str:
    """
    Validate and sanitize user query.
    Returns the (stripped) query if safe, raises GuardrailViolation otherwise.
    """
    query = query.strip()

    if not query:
        raise GuardrailViolation("Query cannot be empty.")

    # Length guard — prevents context window overflow and embedding truncation
    if len(query) > MAX_QUERY_CHARS:
        raise GuardrailViolation(
            f"Query is too long ({len(query)} chars). Please keep it under {MAX_QUERY_CHARS} characters."
        )

    # Vague / one-word query — warn but still allow (return with flag via warning key)
    words = query.split()
    if len(words) == 1:
        raise GuardrailViolation(
            "Query is too vague. Please provide more context for a useful answer."
        )

    # Prompt injection / jailbreak detection
    for pattern in _COMPILED_INJECTION:
        if pattern.search(query):
            raise GuardrailViolation(
                "Your query contains instructions that cannot be processed. Please ask a genuine question."
            )

    return query


def check_output(answer: str, sources: list) -> dict:
    """
    Inspect the generated answer for hallucination signals and no-context situations.
    Returns a dict with 'warning' key if issues detected, otherwise empty.
    """
    warnings = []

    # Detect LLM leaking training-data reasoning instead of using context
    lower = answer.lower()
    for signal in _HALLUCINATION_SIGNALS:
        if signal in lower:
            warnings.append(
                "The answer may not be grounded in your documents — the model referenced its training data."
            )
            break

    # Detect when no sources were returned (retrieval failed)
    if not sources:
        warnings.append(
            "No relevant document chunks were retrieved. The answer may be unreliable."
        )

    # Detect conflicting sources — same question, different answers hinted by contradictory keywords
    if len(sources) >= 2:
        contents = [s["content"].lower() for s in sources]
        contradiction_pairs = [
            ("yes", "no"), ("true", "false"), ("allowed", "prohibited"),
            ("approved", "rejected"), ("increase", "decrease"),
        ]
        for pos, neg in contradiction_pairs:
            has_pos = any(pos in c for c in contents)
            has_neg = any(neg in c for c in contents)
            if has_pos and has_neg:
                warnings.append(
                    "Retrieved sources may contain conflicting information. Verify the answer manually."
                )
                break

    return {"warnings": warnings} if warnings else {}
