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

---

## Module: Parsers (`src/rag_chat_app/parsers/`)

### Purpose
Handles document parsing and text extraction from various file formats for the RAG pipeline. Supports multiple document types with unified interface and automatic format detection.

### Core Architecture

#### Factory Pattern Implementation
```python
# Parser Factory manages all document parsers
PARSER_MAP = {
    "pdf": PdfParser,
    "docx": DocxParser,
    "doc": DocxParser,
    "rtf": RtfParser,
    "txt": TxtParser,
    "md": MarkdownParser,
    "xlsx": XlsxParser,
    "xls": XlsxParser,
}

# Usage
parser_provider = create_parser_provider_from_settings(settings)
parser = parser_provider.get_parser(document_metadata)
documents = parser.parse_safe(document_metadata)
```

### Components

#### 1. **Base Parser** (`parsers/base.py`)
Abstract base class defining the parser interface:

```python
class Parser(ABC):
    """Abstract base class for document parsers."""

    supported_extensions: List[str] = []

    @abstractmethod
    def parse(self, metadata: DocumentMetadata) -> List[Document]:
        """Parse document and return LangChain Document objects."""

    def parse_safe(self, metadata: DocumentMetadata) -> List[Document]:
        """Safe parsing that returns empty list on error."""

    def is_applicable(self, metadata: DocumentMetadata) -> bool:
        """Check if parser supports the file type."""

class ParserProvider:
    """Manages multiple parsers and selects appropriate one."""
```

#### 2. **Individual Parsers**

##### PDF Parser (`parsers/pdf_parser.py`)
```python
class PdfParser(Parser):
    supported_extensions = [".pdf"]

    def parse(self, metadata: DocumentMetadata) -> List[Document]:
        # Uses PDFPlumberLoader from LangChain
        documents = PDFPlumberLoader(str(metadata.source_path)).load()
        # Adds metadata to each document page
```

##### Word Parser (`parsers/docx_parser.py`)
```python
class DocxParser(Parser):
    supported_extensions = [".docx", ".doc"]

    def parse(self, metadata: DocumentMetadata) -> List[Document]:
        # Uses UnstructuredWordDocumentLoader
        # Supports both .docx (modern) and .doc (legacy) formats
```

##### Excel Parser (`parsers/xlsx_parser.py`)
```python
class XlsxParser(Parser):
    supported_extensions = [".xlsx", ".xls"]

    def parse(self, metadata: DocumentMetadata) -> List[Document]:
        # Uses UnstructuredExcelLoader
        # Extracts tabular data and preserves structure
```

##### Text & Markdown Parsers
```python
class TxtParser(Parser):
    supported_extensions = [".txt", ".log"]
    # Uses TextLoader with UTF-8 encoding

class MarkdownParser(Parser):
    supported_extensions = [".md"]
    # Uses UnstructuredMarkdownLoader
    # Preserves headers, lists, and code blocks
```

##### RTF Parser (`parsers/rtf_parser.py`)
```python
class RtfParser(Parser):
    supported_extensions = [".rtf"]
    # Uses UnstructuredRTFLoader
```

#### 3. **Parser Factory** (`parsers/parser_factory.py`)
```python
def create_parser_provider_from_settings(settings) -> ParserProvider:
    """Create parsers based on ENABLED_PARSERS setting."""

def create_custom_parser_provider(parsers: List[Parser]) -> ParserProvider:
    """Create provider with custom parser list."""
```


#### System Package Requirements

##### Pandoc (Required for RTF documents)
```bash
# Ubuntu/Debian
sudo apt-get install pandoc

# macOS
brew install pandoc

# Windows
choco install pandoc
# OR download from: https://pandoc.org/installing.html
```

##### LibreOffice (Recommended for legacy .doc files)
```bash
# Ubuntu/Debian
sudo apt-get install libreoffice

# macOS
brew install --cask libreoffice

# Windows
# Download from: https://www.libreoffice.org/download/download/
```

### Configuration

#### Settings Configuration (`config.py`)
```python
ENABLED_PARSERS: List[str] = [
    "pdf",      # PDF documents
    "docx",     # Modern Word documents
    "doc",      # Legacy Word documents
    "rtf",      # Rich Text Format
    "txt",      # Plain text files
    "md",       # Markdown files
    "xlsx",     # Modern Excel files
    "xls",      # Legacy Excel files
]
```

#### Environment Setup
```bash
# Ensure pandoc is available in PATH
which pandoc  # Should return path to pandoc executable

# Check LibreOffice (optional)
which libreoffice  # Should return path to LibreOffice
```

### How It Works

#### Document Processing Flow
1. **File Discovery** → `DocumentSource` finds files
2. **Metadata Creation** → `DocumentMetadata` with file info
3. **Parser Selection** → `ParserProvider.get_parser()` selects appropriate parser
4. **Document Parsing** → Parser extracts text and creates LangChain Documents
5. **Chunking** → Text split into chunks for embeddings
6. **Vector Storage** → Documents stored in ChromaDB

#### Error Handling
```python
# Safe parsing with automatic retry support
documents = parser.parse_safe(metadata)
if not documents:
    # Parser failed, document marked as FAILED
    # Will be retried on next ingestion run
```

### Supported File Formats

| Format | Extension | Parser | Dependencies |
|--------|-----------|--------|--------------|
| PDF | `.pdf` | PdfParser | PDFPlumberLoader |
| Word (Modern) | `.docx` | DocxParser | Unstructured |
| Word (Legacy) | `.doc` | DocxParser | Unstructured + LibreOffice |
| RTF | `.rtf` | RtfParser | Unstructured + Pandoc |
| Excel (Modern) | `.xlsx` | XlsxParser | Unstructured + Pandas |
| Excel (Legacy) | `.xls` | XlsxParser | Unstructured + Pandas |
| Text | `.txt`, `.log` | TxtParser | TextLoader |
| Markdown | `.md` | MarkdownParser | Unstructured |

### Key Features

✅ **Unified Interface**: All parsers implement same `Parser` interface  
✅ **Automatic Detection**: File extension-based parser selection  
✅ **Safe Parsing**: `parse_safe()` prevents batch processing failures  
✅ **Extensible**: Easy to add new parsers following the pattern  
✅ **LangChain Integration**: Uses LangChain loaders for consistency  
✅ **Metadata Preservation**: Original file metadata preserved in documents  

### Adding New Parser

To add support for new format (e.g., JSON):

1. **Create parser** (`parsers/json_parser.py`):
```python
class JsonParser(Parser):
    supported_extensions = [".json"]

    def parse(self, metadata: DocumentMetadata) -> List[Document]:
        # Implementation using appropriate loader
```

2. **Register in factory** (`parsers/parser_factory.py`):
```python
PARSER_MAP = {
    "json": JsonParser,  # Add new parser
}
```

3. **Add to settings** (`config.py`):
```python
ENABLED_PARSERS = [..., "json"]
```

### Integration with Ingestion

#### Automatic Retry System
- Failed documents marked as `FAILED` status
- Next ingestion run automatically retries failed documents
- `parse_safe()` prevents one bad file from breaking entire batch

#### Batch Processing
- Vector store uses batching to avoid OpenAI token limits
- Documents processed in batches of 100 to prevent timeouts
- Progress logged at debug level

### Troubleshooting

#### Common Issues

**Memory issues with large documents:**
- Reduce `max_chunk_size` in settings
- Increase batch size in vector store
- Use more powerful hardware

### Performance Considerations

- **Small batches**: Better error recovery, lower memory usage
- **Chunk size**: Balance between context preservation and embedding limits
- **Parser selection**: Fast extension-based routing
- **Async processing**: Could be added for better performance

### Future Enhancements

- **Async parsing**: Parallel document processing
- **Custom loaders**: Specialized parsers for domain-specific formats
- **OCR support**: Image text extraction for scanned documents
- **Format detection**: Content-based format detection (not just extension)
- **Progress callbacks**: Real-time parsing progress for large files