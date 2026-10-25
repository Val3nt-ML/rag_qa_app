# Rag - Document Q&A Application

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A Retrieval-Augmented Generation (RAG) application for document-based question answering with a user-friendly Gradio interface. This system allows users to upload documents, build searchable indexes, and ask questions about the content using large language models.

## Features

- 📄 **Document Upload**: Support for multiple formats (.txt, .pdf, .docx, .md)
- 🔍 **Intelligent Search**: Vector-based document retrieval using embeddings
- 🤖 **AI-Powered Q&A**: Question answering using HuggingFace OpenSource models
- 📊 **Q&A Generation**: Automatic generation of question-answer pairs from documents
- 🐳 **Docker Support**: Containerized deployment with Docker Compose
- 🌐 **Web Interface**: Intuitive Gradio-based web interface

## Project Architecture

```
├── .dockerignore          <- Docker ignore patterns
├── .env                   <- Environment variables (not tracked in git)
├── .gitignore            <- Git ignore patterns
├── docker-compose.yaml   <- Docker Compose configuration for the RAG application
├── Dockerfile            <- Docker image definition
├── pyproject.toml        <- Project configuration and dependencies
├── README.md             <- This file
├── uv.lock               <- Locked Python dependencies
│
├── data/                 <- Document storage directory
│   └── *.pdf            <- PDF documents for processing
│
├── notebooks/           <- Jupyter notebooks for experimentation
│   └── *.ipynb          <- LLM & RAG testing notebooks
│
├── src/                 <- Source code for the application
│   ├── __init__.py      <- Makes src a Python module
│   ├── app.py           <- Main Gradio application entry point
│   ├── config.py        <- Configuration management
│   ├── config.yaml      <- Application configuration file
│   │
│   ├── logic/           <- Core App logic modules
│   │   ├── __init__.py
│   │   ├── button.py    <- UI button handlers and state management
│   │   ├── qa.py        <- Question-answer generation logic
│   │   └── rag.py       <- RAG implementation and document querying
│   │
│   └── retrieval/       <- Document retrieval and processing
│       ├── __init__.py
│       └── rag_helper.py <- RAG utility functions and helpers
│
└── tests/               <- Test files
    └── test_data.py     <- Not used/defined
```

## Technical Stack

- **Frontend**: Gradio web interface
- **Backend**: Python with async support
- **AI/ML**: OpenAI GPT models (gpt-4o-mini) with text-embedding-3-small
- **Vector Database**: Qdrant for document embeddings
- **Document Processing**: Support for PDF, TXT, DOCX, MD formats
- **Containerization**: Docker & Docker Compose
- **Configuration**: YAML-based configuration management

## Quick Start

### Using Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Val3nt-ML/rag_qa_app.git
   cd rag_qa_app
   ```

2. **Set up environment variables**:

   Copy `.env.example` to `.env` and fill in your Hugging Face token. Edit `src/config.yaml` for the other settings.

3. **Launch the application**:
   ```bash
   docker compose up rag_app --build
   ```

4. **Access the interface**:
   Open your browser and navigate to `http://localhost:7860`

### Local Development

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Configure the application**:

   Copy `.env.example` to `.env` and fill in your Hugging Face token. Edit `src/config.yaml` for the other settings.

3. **Run the application**:
   ```bash
   uv run python src/app.py
   ```

## Usage

1. **Upload a Document**: Use the file upload interface to add your document
2. **Build Index**: Click "Build Index from document" to create searchable embeddings
3. **Ask Questions**: Type your questions about the document in the text area
4. **Generate Q&A Pairs**: Use the automated Q&A generation feature to create training data

## Configuration

The application is configured through `src/config.yaml`:

- **RAG Settings**: Embedding models, similarity thresholds, LLM parameters
- **Hugging Face authentication**: `HUGGINGFACE_TOKEN` in the root `.env` file (ignored by Git), or in the environment. Environment variables take precedence. Docker Compose loads `.env` automatically.
- **Application**: Server settings, debug mode, sharing options
- **Q&A Generation**: Automated question generation parameters

## Development

Install the development environment and activate the Git hooks after cloning:

```bash
uv sync --locked
make install-hooks
```

Before each commit, prek runs `ruff check .`, `ruff format --check .`, and
`ty check` through uv using the versions pinned in `uv.lock`. A failed check
blocks the commit. Run all checks manually with `make check`.

To fix lint and formatting issues:

```bash
uv run ruff check . --fix
uv run ruff format .
```

Git hooks are local and must be installed in each clone. They can be bypassed
with `git commit --no-verify`; enforcing these checks for a shared branch also
requires CI and branch protection.

- **Code Style and Import Sorting**: Ruff, with 88-character line length
- **Type Checking**: ty
- **Testing**: Pytest framework
- **Python Version**: 3.13.x

--------

