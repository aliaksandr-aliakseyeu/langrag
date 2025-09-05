import logging
import sys
from dotenv import load_dotenv

from rag_chat_app.config import INGESTION_LOGGING_CONFIG, settings
from rag_chat_app.document_sources.source_factory import create_localfile_source
from rag_chat_app.ingestion.ingestion_service import IngestionService
from rag_chat_app.parsers.parser_factory import create_parser_provider_from_settings
from rag_chat_app.storage.store_factory import (
    # create_sqlite_metadata_store,
    create_json_metadata_store,
)
from rag_chat_app.vector.embedding_factory import (
    create_huggingface_embeddings,
    create_openai_embeddings,
)
from rag_chat_app.vector.vector_store_factory import create_chroma_vector_store


def main():
    """Run the document ingestion pipeline."""

    logging.basicConfig(**INGESTION_LOGGING_CONFIG)
    print("🚀 Starting Document Ingestion Pipeline...")
    load_dotenv()
    try:
        parser_provider = create_parser_provider_from_settings(settings)

        # Metadata store can be either SQLite or JSON
        # metadata_store = create_sqlite_metadata_store()
        metadata_store = create_json_metadata_store()

        document_source = create_localfile_source(
            settings.DOCUMENT_FOLDER, parser_provider.get_suported_extentions()
        )
        try:
            embedding_function = create_openai_embeddings(
                settings.OPENAI_EMBEDDING_MODEL
            )
            print("✅ Using OpenAI embeddings")
        except (ValueError, ImportError) as e:
            print(
                f"⚠️ OpenAI embeddings not available ({e}), falling back to HuggingFace..."
            )
            embedding_function = create_huggingface_embeddings(
                settings.HUGGINGFACE_EMBEDDING_MODEL
            )
            print("✅ Using HuggingFace embeddings")

        vector_store = create_chroma_vector_store(
            embedding_function=embedding_function,
            collection_name=settings.VECTOR_COLLECTION_NAME,
        )
        print("✅ Using ChromaDB vector store")

        ingestion_service = IngestionService(
            parser_provider=parser_provider,
            metadata_store=metadata_store,
            vector_store=vector_store,
            document_source=document_source,
        )

    except Exception as e:
        print(f"❌ Failed to initialize ingestion service: {e}")
        return 1

    try:
        ingestion_service.run_full_ingestion(verbose=True)

        stats = ingestion_service.get_ingestion_stats()
        print("\n📊 Final Ingestion Statistics:")
        for status, count in stats.items():
            print(f"  {status.title()}: {count}")

        print("\n🎉 Ingestion completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
