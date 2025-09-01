# RAG Chat Application - Architecture Documentation

## Overview

This is a Retrieval Augmented Generation (RAG) chat application built with Python, LangChain, and ChromaDB. The application follows SOLID principles and uses a modular architecture for easy extension and maintenance.

## Core Architecture

The application uses a **two-stage RAG pipeline**:
1. **Intent Classification** - Determine what the user wants to do
2. **Intent-specific Retrieval** - Execute appropriate retrieval chain based on detected intent

---

## Module: LLM (`src/rag_chat_app/llm/`)

### Purpose
Manages Language Model creation and configuration with support for multiple providers (OpenAI, Ollama, HuggingFace).

### Components

#### 1. **LLM Enums** (`llm/enums.py`)
```python
LLMProvider:        # Available providers: OPENAI, OLLAMA, HUGGINGFACE
OpenAIModel:        # OpenAI models: GPT_4O, GPT_4O_MINI, GPT_3_5
OllamaModel:        # Ollama models: MISTRAL, LLAMA3, GEMMA
HuggingFaceModel:   # HuggingFace models: ALL_MINI_LM
```

#### 2. **LLM Registry** (`llm/llm_registry.py`)
Maps providers to their model enums for validation:
```python
MODEL_ENUM_MAP = {
    LLMProvider.OPENAI: OpenAIModel,
    LLMProvider.OLLAMA: OllamaModel,
    LLMProvider.HUGGINGFACE: HuggingFaceModel,
}
```

#### 3. **LLM Configuration** (`llm/llm_config.py`)
```python
@dataclass
class LLMConfig:
    # Intent classification settings
    intent_provider: LLMProvider = LLMProvider.OPENAI
    intent_model: Union[OpenAIModel, OllamaModel, HuggingFaceModel] = OpenAIModel.GPT_4O_MINI
    intent_temperature: float = 0.0
    
    # Chat response settings
    chat_provider: LLMProvider = LLMProvider.OPENAI
    chat_model: Union[OpenAIModel, OllamaModel, HuggingFaceModel] = OpenAIModel.GPT_4O_MINI
    chat_temperature: float = 0.1
```

**Features:**
- **Automatic validation** using `MODEL_ENUM_MAP` in `__post_init__`
- **Default from settings**: `LLMConfig.from_settings()`
- **Custom configuration**: Create manually with desired models

#### 4. **LLM Service** (`llm/llm_service.py`)
Factory service for creating LLM instances.

```python
class LLMService:
    def __init__(self, config: LLMConfig = None):
        self.config = config or LLMConfig.from_settings()
    
    def create_llm(self, provider, model, temperature) -> Runnable:
        # Universal LLM creation method
        
    def create_intent_llm(self) -> Runnable:
        # Creates LLM for intent classification using config
        
    def create_chat_llm(self) -> Runnable:
        # Creates LLM for chat responses using config
```

### How It Works

#### Default Usage (from settings):
```python
# Uses default configuration from settings.py
llm_service = LLMService()
chat_service = ChatService(vector_store, llm_service)
```

#### Custom Configuration:
```python
# Create custom LLM configuration
custom_config = LLMConfig(
    intent_model=OpenAIModel.GPT_4O,        # Better intent classification
    chat_model=OpenAIModel.GPT_3_5,         # Cost-effective chat
    chat_temperature=0.2                     # More creative responses
)

# Create LLM service with custom config
llm_service = LLMService(custom_config)

# Create chat service with configured LLM service
chat_service = ChatService(vector_store, llm_service)
```

#### For Web Service (user preferences):
```python
# User sends preferences from frontend
user_preferences = {
    "intent_model": "gpt-4o",
    "chat_model": "gpt-3.5-turbo",
    "chat_temperature": 0.3
}

# Create config for this user
user_config = LLMConfig(
    intent_model=OpenAIModel.GPT_4O,
    chat_model=OpenAIModel.GPT_3_5,
    chat_temperature=0.3
)

# Each user gets their own configured service
llm_service = LLMService(user_config)
chat_service = ChatService(vector_store, llm_service)
```

### Adding New LLM Provider

To add a new LLM provider (e.g., Anthropic):

1. **Add provider enum** (`llm/enums.py`):
```python
class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"  # Add new provider
```

2. **Add model enums** (`llm/enums.py`):
```python
class AnthropicModel(str, Enum):
    CLAUDE_3 = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
```

3. **Update registry** (`llm/llm_registry.py`):
```python
MODEL_ENUM_MAP = {
    LLMProvider.ANTHROPIC: AnthropicModel,  # Add mapping
}
```

4. **Update factory** (`llm/llm_service.py`):
```python
def create_llm(self, provider, model, temperature):
    if provider == LLMProvider.ANTHROPIC:
        return AnthropicLLM(model=model, temperature=temperature)
```

5. **Export new enums** (`llm/__init__.py`):
```python
from .enums import AnthropicModel
__all__ = [..., "AnthropicModel"]
```

### Key Benefits

✅ **Type Safety**: Enums prevent invalid model/provider combinations  
✅ **Easy Extension**: Add new providers in 4 simple steps  
✅ **Configuration Flexibility**: Default settings or custom per-user configs  
✅ **Validation**: Automatic validation ensures provider-model compatibility  
✅ **Clean Separation**: LLM logic isolated in dedicated module  

---

## Usage Summary

### Default Configuration:
```python
# Uses settings.py configuration
llm_service = LLMService()
```

### Custom Configuration:
```python
# ⚠️ IMPORTANT: To use custom LLM models, you MUST pass LLMConfig
custom_config = LLMConfig(
    intent_model=OpenAIModel.GPT_4O,
    chat_model=OpenAIModel.GPT_3_5
)
llm_service = LLMService(custom_config)

# Then pass configured service to ChatService
chat_service = ChatService(vector_store, llm_service)
```

**Remember**: If you want custom LLM settings, create `LLMConfig` first, then pass it to `LLMService`, then pass that service to `ChatService`.

---

## Module: Intent (`src/rag_chat_app/intent/`)

### Purpose
Handles user intent classification - the first stage of the two-stage RAG pipeline. Determines what the user wants to do with their message.

### Components

#### 1. **Intent Enums** (`intent/enums.py`)
```python
class UserIntent(str, Enum):
    SEARCH_DOCUMENTS = "search_documents"       # Find specific info in documents
    GET_DOCUMENT_NAMES = "get_document_names"   # List relevant documents  
    SUMMARIZE_DOCUMENT = "summarize_document"   # Summarize content
    CHAT_GENERAL = "chat_general"               # General conversation
    UNKNOWN = "unknown"                         # Unrecognized intent
```

**Features:**
- **Descriptions**: Each intent has human-readable description via `.description()`
- **All intents listing**: Use `UserIntent.all_with_description()` for documentation

#### 2. **Intent Manager** (`intent/intent_manager.py`)
Core class that handles intent classification using LLM.

```python
class IntentManager:
    def __init__(self, llm_service: LLMService, confidence_threshold: float = 0.7):
        # Creates intent classification chain using LLM service
        
    def classify_intent(message: str, chat_history: List) -> IntentClassificationResult:
        # Main method: classifies user message intent
        
    def is_high_confidence(intent_result: IntentClassificationResult) -> bool:
        # Checks if classification confidence is above threshold
        
    def get_intent_enum(intent_result: IntentClassificationResult) -> UserIntent:
        # Converts string result to UserIntent enum with fallback
```

### How It Works

#### Intent Classification Flow:
1. **User message** → `IntentManager.classify_intent()`
2. **LLM processes** message + chat history context  
3. **Returns** `IntentClassificationResult` with:
   - `intent`: Classified intent as string
   - `confidence`: How confident the model is (0.0-1.0)
   - `reasoning`: Why this intent was chosen
   - `parameters`: Extracted parameters (e.g., document names)

#### Usage in ChatService:
```python
# Step 1: Classify intent
intent_result = self.intent_manager.classify_intent(message, chat_history)

# Step 2: Check confidence
if self.intent_manager.is_high_confidence(intent_result):
    # Use detected intent for retrieval
    intent_enum = self.intent_manager.get_intent_enum(intent_result)
    answer = self.retrieval_manager.run(intent_enum, chat_history, message)
else:
    # Low confidence - fallback to general chat
    answer = self.retrieval_manager.run(UserIntent.CHAT_GENERAL, chat_history, message)
```

### Intent Classification Examples:

#### Search Documents:
```python
User: "What documents do I need for a visa?"
Result: IntentClassificationResult(
    intent="search_documents",
    confidence=0.89,
    reasoning="User asking for specific document information",
    parameters={"search_term": "visa documents"}
)
```

#### Get Document Names:
```python
User: "Which files contain information about taxes?"
Result: IntentClassificationResult(
    intent="get_document_names", 
    confidence=0.94,
    reasoning="User wants to know which documents contain tax information",
    parameters={"topic": "taxes"}
)
```

#### Summarize Document:
```python
User: "Give me a summary of the requirements.pdf file"
Result: IntentClassificationResult(
    intent="summarize_document",
    confidence=0.96, 
    reasoning="User explicitly requesting document summary",
    parameters={"document_name": "requirements.pdf"}
)
```

#### General Chat:
```python
User: "How are you today?"
Result: IntentClassificationResult(
    intent="chat_general",
    confidence=0.99,
    reasoning="General greeting, not document-related",
    parameters={}
)
```

### Configuration

#### Default Confidence Threshold:
```python
intent_manager = IntentManager(llm_service, confidence_threshold=0.7)
```

#### Low Confidence Behavior:
If confidence < threshold:
- **Fallback** to `CHAT_GENERAL` intent
- **Logs warning** about low confidence
- **Continues processing** with general chat flow

### Integration with Other Modules

#### With LLM Module:
- Uses `LLMService.create_intent_llm()` for classification model
- Configured via `LLMConfig.intent_model` and `LLMConfig.intent_temperature`

#### With Prompts Module:
- Uses `IntentPromtManager` to build classification prompts
- Integrates with `IntentClassificationResult` for structured output

#### With Retrieval Module:
- Passes classified intent to `RetrievalManager.run()`
- Intent determines which retrieval strategy to use

### Key Benefits

✅ **Intelligent Routing**: Automatically determines appropriate response strategy  
✅ **Confidence-based Fallback**: Handles uncertain classifications gracefully  
✅ **Context Aware**: Uses chat history for better classification  
✅ **Extensible**: Easy to add new intent types  
✅ **Clean Architecture**: Separated from general enums, focused responsibility

### Adding New Intent Type

To add a new intent (e.g., `DELETE_DOCUMENT`):

1. **Add to enum** (`intent/enums.py`):
```python
class UserIntent(str, Enum):
    DELETE_DOCUMENT = "delete_document"
```

2. **Update prompt examples** (`prompts/intention_prompt.py`):
```python
# Add training examples for the new intent
```

3. **Handle in retrieval** (`retrieval/retrieval_manager.py`):
```python
# Add logic for delete_document intent
```

The intent classification will automatically work with the new intent type!
