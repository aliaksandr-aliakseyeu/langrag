import logging
from dotenv import load_dotenv
from typing import List, Tuple

from rag_chat_app.chat.chat_service import ChatService
from rag_chat_app.llm.llm_service import LLMService
from rag_chat_app.vector.vector_store_factory import create_chroma_vector_store
from rag_chat_app.vector.embedding_factory import (
    create_huggingface_embeddings,
    create_openai_embeddings,
)
from rag_chat_app.config import settings, CHAT_LOGGING_CONFIG


load_dotenv()
logging.basicConfig(**CHAT_LOGGING_CONFIG)


def main():
    print("💬 Starting RAG Chat Application...")
    print("🔍 Initializing vector store...")
    try:
        try:
            embedding_function = create_openai_embeddings("text-embedding-3-small")
            print("✅ Using OpenAI embeddings")
        except (ValueError, ImportError):
            embedding_function = create_huggingface_embeddings("all-MiniLM-L6-v2")
            print("✅ Using HuggingFace embeddings")

        vector_store = create_chroma_vector_store(
            embedding_function=embedding_function,
            collection_name=settings.VECTOR_COLLECTION_NAME,
        )
        print("✅ Vector store initialized!")

    except Exception as e:
        print(f"❌ Failed to initialize vector store: {e}")
        return

    print("🤖 Initializing chat service...")
    try:
        chat_service = ChatService(
            vector_store=vector_store,
            llm_service=LLMService(),
            intent_confidence_threshold=0.7,
        )
        print("✅ Chat service initialized!")

    except Exception as e:
        print(f"❌ Failed to initialize chat service: {e}")
        print("💡 Make sure you have set your OPENAI_API_KEY environment variable")
        return

    chat_history: List[Tuple[str, str]] = []

    print("\n" + "=" * 60)
    print("🤖 RAG CHAT INTERFACE")
    print("=" * 60)
    print("💡 Type your questions below. Use 'exit' or 'quit' to stop.")
    print("🔍 I can help you with document search, summaries, and general chat!")
    print("=" * 60)

    while True:
        try:
            question = input("\n💭 You: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye! Thanks for chatting!")
                break

            print("🤔 Processing your request...")
            response = chat_service.chat(message=question, chat_history=chat_history)
            print(
                f"🎯 Intent: {response.intent} (confidence: {response.confidence:.2f})"
            )
            print(f"🤖 Assistant: {response.answer}")

            chat_history.append((question, response.answer))

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    main()
