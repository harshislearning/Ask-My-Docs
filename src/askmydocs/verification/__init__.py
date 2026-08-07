"""Post-hoc verification of a generated answer against its sources.

Prompting the model to cite is a request, not a guarantee. These checks are what
turn "we asked it to cite" into "we know whether it did".
"""

from .citations import check_citations, claims_in
from .entailment import ClaimCheck, Verdict, check_entailment, check_lexically
from .sentences import Sentence, is_claim, parse_citations, split_sentences
from .verifier import Verifier

__all__ = [
    "ClaimCheck",
    "Sentence",
    "Verdict",
    "Verifier",
    "check_citations",
    "check_entailment",
    "check_lexically",
    "claims_in",
    "is_claim",
    "parse_citations",
    "split_sentences",
]
