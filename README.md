# RAG Chat App 🌟

A flexible Retrieval-Augmented Generation (RAG) chat application. Supports various document sources, metadata databases, vector stores, embedding models, and LLMs. Implements a complete document processing and intelligent search pipeline.

## 🌿 Project Branches

### `main` (current)
- 📝 Project documentation

### `langchain-version` ⭐
**Status: ✅ Implemented**

**Current implementation:**
- **Documents**: PDF, DOCX, DOC, RTF, XLSX, XLS, TXT, Markdown files (local file system)
- **Metadata**: SQLite database, JSON
- **Vector Store**: ChromaDB
- **Embeddings**: OpenAI text-embedding-3-small
- **LLM**: OpenAI GPT models
- **Framework**: LangChain

**Features:**
- 🔄 Complete document processing pipeline
- 🏗️ Modular architecture with dependency injection
- ⚙️ Configuration via environment variables
- 🛡️ Comprehensive error handling
- 📊 Detailed logging
- 🎯 SOLID principles and factory patterns
- 📋 User intent classification
- 💬 Contextual responses with source citations
- 🔍 Document change detection
- 💾 Conversation history

## 🚀 Quick Start

### Requirements
- Python 3.11+
- Poetry
- OpenAI API key

### Installation and Launch

```bash
# Switch to implementation branch
git checkout langchain-version

# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Add OPENAI_API_KEY

# Run document ingestion
poetry run python src/rag_chat_app/run_ingestion.py

# Start chat
poetry run python src/rag_chat_app/main.py
```

## 🛠️ Architecture

The project is built on a modular architecture that **allows easy** integration of various component implementations:

### Supported Components (architectural capabilities):
- **Document Sources**: Local file system, OneDrive, Google Drive, SharePoint
- **Parsers**: PDF, DOCX, TXT, HTML, Markdown
- **Metadata Storage**: SQLite, PostgreSQL, MongoDB
- **Vector Stores**: ChromaDB, Pinecone, FAISS, Weaviate
- **Embedding Models**: OpenAI, HuggingFace, Cohere
- **LLM**: OpenAI GPT, Anthropic Claude, Google PaLM

### Technologies:
- **Python 3.11+** - main language
- **Pydantic** - validation and settings
- **Poetry** - dependency management
- **SOLID principles** - architectural patterns
- **Factory patterns** - configurable components

## 📞 Contacts

**Aliaksandr Aliakseyeu** - Python Backend Developer

📍 Kraków, Poland | 📞 +48451011168 | ✉️ aliaksandr.aliakseyeu@gmail.com

---

*Project demonstrates flexible RAG system architecture with support for various technologies and components.*
