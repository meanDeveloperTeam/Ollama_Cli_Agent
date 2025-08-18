# 🤖 Local Claude-CLI

## Description

Local Claude-CLI is a powerful command-line interface (CLI) tool that integrates with local Ollama models to provide a rich, interactive chat experience. It's designed to bring the capabilities of large language models directly to your terminal, enhanced with features like Retrieval-Augmented Generation (RAG) using ChromaDB for improved accuracy and performance, and a beautiful user interface powered by the `rich` library.

This CLI also includes experimental features for code editing and updating files based on model suggestions, with user confirmation for safety.

## Features

- **Ollama Integration:** Seamlessly chat with any Ollama-compatible local language model (e.g., `mistral:7b`).
- **Retrieval-Augmented Generation (RAG):** Enhance model accuracy and performance by providing relevant context from your own knowledge base using ChromaDB.
- **Beautiful CLI:** A visually appealing and interactive command-line experience powered by the `rich` library, featuring panels, markdown rendering, and live streaming.
- **Code Edit & Update (Experimental):** Propose and apply code changes to files based on model suggestions, with a clear diff view and user confirmation.
- **VS Code Context:** Automatically detects and provides context from your active VS Code project and open files to the LLM.
- **Project Context:** Loads relevant files from your project directory as context for the LLM.
- **Chat History:** Saves and loads chat history for continuous conversations.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/OllamaCLi.git
    cd OllamaCLi
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *Note: Ensure you have Python 3.8+ installed.*

3.  **Install Ollama:**

    Download and install Ollama from [ollama.ai](https://ollama.ai/).

4.  **Pull a model (e.g., Mistral 7B):**

    ```bash
    ollama pull mistral:7b
    ```

## Usage

### 1. Ingest Documents for RAG (Optional but Recommended)

To leverage RAG, you need to ingest your knowledge base into ChromaDB. Create a directory containing your `.txt` documents.

```bash
python local_claude.py --ingest_dir /path/to/your/documents
```

This will process your documents, create embeddings, and store them in a local `chroma_db` directory.

### 2. Start the Chat

```bash
python local_claude.py chat --model mistral:7b
```

Replace `mistral:7b` with the name of any Ollama model you have pulled.

### Code Edit & Update Feature

This feature allows the model to propose code changes. When the model suggests a change in the specific `file_edit` format, the CLI will display a diff and ask for your confirmation before applying the changes to the file.

**Example of how to prompt the model for a code edit:**

```
I need you to provide a code change for `local_claude.py`. The change should add a comment at the top of the `main` function. Your response MUST be in the following format:

```file_edit
file_path: "c:\projects\projects\OllamaCLi\local_claude.py"
old_code: |\n  def main():
new_code: |\n  """
  Main function for running the local chatbot using Claude.

  This script initializes the bot and starts a chat session with the user.
  """
  def main():
```
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. (Note: You might need to create a LICENSE file if you don't have one.)
