"""Typed exceptions.

Ingestion must never abort a whole run because one PDF is broken, so parse
failures are all subclasses of :class:`DocumentError` and the pipeline catches
that one type and records it in the manifest.
"""

from __future__ import annotations


class AskMyDocsError(Exception):
    """Base class for every error this system raises deliberately."""


class ConfigError(AskMyDocsError):
    """Configuration is missing, malformed, or internally inconsistent."""


class GenerationError(AskMyDocsError):
    """The language model could not produce an answer."""


class LlmRateLimitError(GenerationError):
    """The provider rate-limited us and retries were exhausted."""


class LlmAuthError(GenerationError):
    """The API key is missing, invalid, or lacks access to the model."""


class LlmTimeoutError(GenerationError):
    """The request exceeded the configured timeout."""


class IndexingError(AskMyDocsError):
    """An index could not be built, saved, or loaded."""


class IndexNotFoundError(IndexingError):
    """No index exists on disk yet."""


class StaleIndexError(IndexingError):
    """The index was built from a different set of chunks than the ones on disk."""


class DocumentError(AskMyDocsError):
    """A single document could not be ingested. Recoverable: skip and continue."""

    #: Short machine-readable reason recorded in the manifest.
    code = "document_error"

    def __init__(self, message: str, *, path: str | None = None) -> None:
        super().__init__(message)
        self.path = path


class PdfParseError(DocumentError):
    """PyMuPDF could not open or read the file."""

    code = "parse_failed"


class EncryptedPdfError(DocumentError):
    """The PDF is password protected and cannot be read."""

    code = "encrypted"


class ScannedPdfError(DocumentError):
    """The PDF has no extractable text layer (image-only / needs OCR)."""

    code = "scanned_no_text_layer"


class EmptyDocumentError(DocumentError):
    """The document parsed cleanly but produced no usable text."""

    code = "empty"


class FileTooLargeError(DocumentError):
    """The file exceeds the configured size ceiling."""

    code = "too_large"
