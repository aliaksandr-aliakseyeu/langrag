import logging
from pathlib import Path
from typing import Optional
from pprint import pprint

from rag_chat_app.config import ChunkingConfig
from rag_chat_app.document_sources.base import DocumentSource
from rag_chat_app.storage.metadata_store import MetadataStore
from rag_chat_app.enums import VectorStatus
from rag_chat_app.parsers.base import ParserProvider
from rag_chat_app.vector.chunker import LangChainChunker
from rag_chat_app.vector.stores.base import VectorStore
from rag_chat_app.utils.files_classifier import classifier

logger = logging.getLogger(__name__)


class IngestionService:
    """
    Handles the complete document ingestion pipeline for RAG applications.

    This service orchestrates:
    - Document discovery and metadata tracking
    - Document parsing and content extraction
    - Text chunking for optimal retrieval
    - Vector embedding and storage
    """

    def __init__(
        self,
        parser_provider: ParserProvider,
        metadata_store: MetadataStore,
        vector_store: VectorStore,
        document_source: DocumentSource,
        chunker_config: Optional[ChunkingConfig] = None,
    ):
        self.parser_provider = parser_provider
        self.supported_extensions = self.parser_provider.get_suported_extentions()
        self.meta_store = metadata_store
        self.vector_store = vector_store
        self.document_source = document_source
        self.chunker = LangChainChunker(chunker_config)

    def discover_documents(self) -> dict:
        """
        Discover documents in the configured folder and classify their status.

        Returns:
            Dictionary with 'new', 'updated', and 'deleted' document lists
        """

        logger.info(
            "Discovering documents using: %s", self.document_source.__class__.__name__
        )

        documents = self.document_source.list_documents()

        documents_status_map = classifier(
            metadata_db=self.meta_store,
            documents=documents,
            supported_extensions=self.supported_extensions,
        )

        logger.info("Document classification complete:")
        logger.info("  New: %d", len(documents_status_map["new"]))
        logger.info("  Updated: %d", len(documents_status_map["updated"]))
        logger.info("  Deleted: %d", len(documents_status_map["deleted"]))
        logger.info("  unchanged: %d", len(documents_status_map["unchanged"]))

        return documents_status_map

    def update_metadata(self, documents_status_map: dict) -> None:
        """
        Update metadata store with document changes.

        Args:
            documents_status_map: Dictionary from discover_documents()
        """

        documents_to_save = (
            documents_status_map["new"] + documents_status_map["updated"]
        )
        if documents_to_save:
            self.meta_store.save_documents_metadata(documents_to_save)
            logger.info("Saved metadata for %d documents", len(documents_to_save))

        documents_to_delete = documents_status_map["deleted"]
        if documents_to_delete:
            self.meta_store.delete_documents_metadata(documents_to_delete)
            logger.info("Deleted metadata for %d documents", len(documents_to_delete))

    def process_pending_documents(self) -> None:
        """
        Process all documents with PENDING status through the full pipeline.

        This includes:
        1. Parsing document content
        2. Chunking text for optimal retrieval
        3. Creating vector embeddings
        4. Storing in vector database
        """
        # Get pending documents
        pending_documents = self.meta_store.load_documents_metadata(
            vector_status=VectorStatus.PENDING
        )

        if not pending_documents:
            logger.info("No pending documents to process")
            return

        logger.info("Processing %d pending documents", len(pending_documents))

        for doc_metadata in pending_documents:
            self._process_single_document(doc_metadata)

    def _process_single_document(self, doc_metadata) -> None:
        """
        Process a single document through the complete pipeline.

        Args:
            doc_metadata: Document metadata object
        """
        try:
            logger.info("Processing document: %s", doc_metadata.file_name)

            self.meta_store.update_document_processing_status(
                document=doc_metadata, vector_status=VectorStatus.PROCESSING
            )

            if not Path(doc_metadata.source_path).exists():
                error_msg = f"File does not exist: {doc_metadata.source_path}"
                logger.error(error_msg)
                self.meta_store.update_document_processing_status(
                    document=doc_metadata,
                    vector_status=VectorStatus.FAILED,
                    vector_error=error_msg,
                )
                return

            parser = self.parser_provider.get_parser(doc_metadata)
            if not parser:
                error_msg = f"No parser available for {doc_metadata.file_name}"
                logger.error(error_msg)
                self.meta_store.update_document_processing_status(
                    document=doc_metadata,
                    vector_status=VectorStatus.FAILED,
                    vector_error=error_msg,
                )
                return

            parsed_content = parser.parse(doc_metadata)
            chunk_documents = self.chunker.chunk_documents(parsed_content)

            if not chunk_documents:
                error_msg = (
                    f"Document {doc_metadata.source_path} contains no extractable text"
                )
                logger.warning(error_msg)
                self.meta_store.update_document_processing_status(
                    document=doc_metadata,
                    vector_status=VectorStatus.FAILED,
                    vector_error=error_msg,
                )
                return

            logger.info("Checking if document exists in vector store...")
            try:
                if self.vector_store.document_exists(
                    source_path=doc_metadata.source_path
                ):
                    logger.info("Document exists, deleting old vectors...")
                    self.vector_store.delete_vectors_by_source(
                        source_path=doc_metadata.source_path
                    )
                    logger.info(
                        "Deleted existing vectors for: %s", doc_metadata.source_path
                    )
                else:
                    logger.info("Document doesn't exist in vector store yet")
            except Exception as e:
                logger.error(f"Error checking/deleting existing document: {e}")
                raise

            self.vector_store.add_documents(chunk_documents)

            self.meta_store.update_document_processing_status(
                document=doc_metadata,
                vector_status=VectorStatus.COMPLETED,
                chunk_count=len(chunk_documents),
            )

            logger.info(
                "Successfully processed %s: %d chunks created",
                doc_metadata.file_name,
                len(chunk_documents),
            )

        except Exception as e:
            error_msg = f"Failed to process {doc_metadata.file_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.meta_store.update_document_processing_status(
                document=doc_metadata,
                vector_status=VectorStatus.FAILED,
                vector_error=error_msg,
            )

    def run_full_ingestion(self, verbose: bool = False) -> None:
        """
        Run the complete ingestion pipeline.

        Args:
            verbose: Whether to print detailed progress information
        """
        if verbose:
            print("=" * 60)
            print("STARTING DOCUMENT INGESTION PIPELINE")
            print("=" * 60)

        if verbose:
            print("\n📁 DISCOVERING DOCUMENTS...")
        documents_status_map = self.discover_documents()

        if verbose:
            print("\n📊 DOCUMENT CLASSIFICATION RESULTS:")
            pprint(documents_status_map)

        if verbose:
            print("\n💾 UPDATING METADATA...")
        self.update_metadata(documents_status_map)

        if verbose:
            print("\n🔄 PROCESSING DOCUMENTS...")
        self.process_pending_documents()

        if verbose:
            print("\n✅ INGESTION PIPELINE COMPLETE!")
            print("=" * 60)

    def get_ingestion_stats(self) -> dict:
        """
        Get statistics about the current ingestion state.

        Returns:
            Dictionary with counts of documents in each status
        """

        stats = {}
        for status in VectorStatus:
            count = len(self.meta_store.load_documents_metadata(vector_status=status))
            stats[status.value] = count

        return stats

    def get_vector_store(self) -> VectorStore:
        """
        Get the configured vector store for retrieval operations.

        Returns:
            Configured VectorStore instance
        """
        if not self.vector_store:
            raise RuntimeError("Ingestion service not set up. Call setup() first.")

        return self.vector_store
