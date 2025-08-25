from typing import List, Tuple
from dotenv import load_dotenv

from rag_chat_app.llm.llm_service import LLMService
from rag_chat_app.prompts.intention_prompt import IntentPromtManager
from rag_chat_app.utils.utils import format_chat_history


load_dotenv()


def main():
    """Start the RAG chat application."""
    print("💬 Starting RAG Chat Application...")

    # Initialize LLM services for intent classification
    print("🧠 Initializing LLM services...")
    try:
        llm_service = LLMService()
        intention_llm = llm_service.get_intent_llm()

        # Initialize intent classification chain
        intention_prompt_manager = IntentPromtManager()
        intention_promt = intention_prompt_manager.create_intent_prompt()
        intention_output_parser = intention_prompt_manager.get_output_parser()
        intention_chaine = intention_promt | intention_llm | intention_output_parser

        print("✅ LLM services initialized!")

    except Exception as e:
        print(f"❌ Failed to initialize LLM services: {e}")
        print("💡 Make sure you have set your OPENAI_API_KEY environment variable")
        return

    # Initialize chat history
    chat_history: List[Tuple[str, str]] = []

    print("\n" + "=" * 60)
    print("🤖 RAG CHAT INTERFACE")
    print("=" * 60)
    print("💡 Type your questions below. Use 'exit' or 'quit' to stop.")
    print("🔍 I can help you with document search, summaries, and general chat!")
    print("=" * 60)

    # Start chat loop
    while True:
        try:
            question = input("\n💭 You: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit", "bye"]:
                print("👋 Goodbye! Thanks for chatting!")
                break

            # Get recent chat history for context (last message only)
            intention_chat_history = chat_history[-1:] if chat_history else []

            inputs = {
                "query": question,
                "chat_history": format_chat_history(intention_chat_history),
            }

            # Classify user intent
            print("🤔 Analyzing your request...")
            result = intention_chaine.invoke(input=inputs)

            print(f"🎯 Intent: {result.intent}")
            print(f"🤖 Assistant: {result.reasoning}")

            # Store conversation
            chat_history.append((question, result.reasoning))

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"❌ An error occurred: {e}")
            print("Please try again or type 'exit' to quit.")


if __name__ == "__main__":
    main()
