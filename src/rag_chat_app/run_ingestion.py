import sys
from dotenv import load_dotenv

from rag_chat_app.ingestion import IngestionService
from rag_chat_app.parsers import create_default_parser_provider
from rag_chat_app.storage import create_sqlite_metadata_store
from rag_chat_app.vector import (
    create_huggingface_embeddings,
    create_openai_embeddings,
    create_chroma_vector_store,
)


def main():
    """Run the document ingestion pipeline."""
    print("🚀 Starting Document Ingestion Pipeline...")
    load_dotenv()
    try:
        parser_provider = create_default_parser_provider()
        metadata_store = create_sqlite_metadata_store()
        try:
            embedding_function = create_openai_embeddings("text-embedding-3-small")
            print("✅ Using OpenAI embeddings")
        except (ValueError, ImportError) as e:
            print(
                f"⚠️ OpenAI embeddings not available ({e}), falling back to HuggingFace..."
            )
            embedding_function = create_huggingface_embeddings("all-MiniLM-L6-v2")
            print("✅ Using HuggingFace embeddings")

        vector_store = create_chroma_vector_store(
            embedding_function=embedding_function, collection_name="rag_documents"
        )
        print("✅ Using ChromaDB vector store")

        ingestion_service = IngestionService(
            parser_provider=parser_provider,
            metadata_store=metadata_store,
            vector_store=vector_store,
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
