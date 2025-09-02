from typing import List, Dict, Type
from .base import Parser, ParserProvider
from .pdf_parser import PdfParser

# Global parser registry - maps parser names to parser classes
PARSER_MAP: Dict[str, Type[Parser]] = {
    "pdf": PdfParser,
    # Future parsers can be added here:
    # "docx": DocxParser,
    # "txt": TxtParser,
    # "html": HtmlParser,
}


def create_custom_parser_provider(parsers: List[Parser]) -> ParserProvider:
    """
    Create a parser provider with custom list of parsers.

    Args:
        parsers: List of parser instances to use

    Returns:
        ParserProvider configured with the provided parsers
    """
    return ParserProvider(parsers)


def create_parser_provider_from_settings(settings) -> ParserProvider:
    """
    Create a parser provider based on application settings.

    Args:
        settings: Application settings object

    Returns:
        ParserProvider configured according to settings
    """
    parsers = []
    enabled_parsers = getattr(settings, "ENABLED_PARSERS", ["pdf"])

    for parser_name in enabled_parsers:
        parser_class = PARSER_MAP.get(parser_name.lower())
        if parser_class:
            parsers.append(parser_class())
        else:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning(
                f"Unknown parser '{parser_name}' in ENABLED_PARSERS, skipping"
            )

    if not parsers:
        import logging

        logger = logging.getLogger(__name__)
        logger.warning("No valid parsers found in settings, falling back to PDF parser")
        parsers.append(PdfParser())

    return ParserProvider(parsers)
