from .base import ParserProvider
from .pdf_parser import PdfParser


def create_default_parser_provider() -> ParserProvider:
    """
    Create a default parser provider with commonly used parsers.

    Returns:
        ParserProvider configured with default parsers
    """
    parsers = [
        PdfParser(),
        # DocxParser(),
        # TxtParser(),
    ]
    return ParserProvider(parsers)
