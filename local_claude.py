import os
import sys
import argparse
import json
import subprocess
import ollama
import readline
import re
from difflib import unified_diff
from rich import print as rich_print
from rich.panel import Panel
from rich.markdown import Markdown
from rich.live import Live
from rich.prompt import Confirm
from rich.syntax import Syntax
import textwrap

import libcst as cst
import libcst.matchers as m

import chromadb
from sentence_transformers import SentenceTransformer
import glob
from langchain_text_splitters import RecursiveCharacterTextSplitter

# RAG Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
N_RESULTS = 3 # Number of relevant chunks to retrieve

# Model Configuration for Task-Specific Models
CODE_EDIT_MODELS = ["codellama", "deepseek-coder"]
REASONING_MODELS = ["llama3", "mistral"] # Default llama3, mistral is also good for reasoning
EXPLANATION_MODELS = ["phi-3"] # Smaller, faster model for explanations

HISTORY_FILE = os.path.expanduser("~/.local_llm_history.json")

SYSTEM_PROMPT = """You are a senior AI coding assistant, you suggest minimal and correct changes only.
Always use the format ```file_edit
file_path: \"path/to/file.py\"
old_code: |\n    # old code
new_code: |\n    # new code
``` for code edits."""

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def get_vscode_context(max_chars=2000):
    """Try to detect current VS Code project and opened file"""
    try:
        result = subprocess.run(
            ["code", "--status"],
            capture_output=True, text=True
        )
        output = result.stdout
        if not output:
            return None, None

        project_dir, active_file = None, None
        for line in output.splitlines():
            if line.strip().startswith("Workspace:"):
                project_dir = line.split(":", 1)[1].strip()
            if line.strip().startswith("Active file:"):
                active_file = line.split(":", 1)[1].strip()
        
        context_snippets = []
        if active_file:
            try:
                with open(active_file, "r", encoding="utf-8") as f:
                    content = f.read(max_chars)
                    context_snippets.append(f"\n### Active File: {os.path.basename(active_file)}\n{content}\n")
            except Exception as e:
                rich_print(f"[bold red]Error reading active file content: {e}[/bold red]")
        
        return project_dir, "\n\n".join(context_snippets)

    except FileNotFoundError:
        return None, None
    except Exception as e:
        rich_print(f"[bold red]An unexpected error occurred while checking VS Code status: {e}[/bold red]")
        return None, None


def create_vector_db(directory):
    rich_print(f"[bold blue]Indexing project files in {directory} for RAG...[/bold blue]")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("code_context")

    # Initialize SentenceTransformer for embeddings
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Define a text splitter
    # This splitter attempts to split by functions, classes, etc.
    # If not possible, it falls back to characters.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []
    metadatas = []
    ids = []
    doc_id = 0

    # Find relevant files
    # Using glob.glob with recursive=True to find files in subdirectories
    # Filtering for common code/text file extensions
    file_patterns = ["**/*.py", "**/*.js", "**/*.ts", "**/*.html", "**/*.css", "**/*.md", "**/*.txt"]
    
    for pattern in file_patterns:
        for fpath in glob.glob(os.path.join(directory, pattern), recursive=True):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Split the document into chunks
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"source": fpath, "chunk_id": i})
                    ids.append(f"doc_{doc_id}_chunk_{i}")
                doc_id += 1
            except Exception as e:
                rich_print(f"[bold red]Error processing file {fpath}: {e}[/bold red]")
                continue

    if documents:
        rich_print(f"[bold blue]Adding {len(documents)} chunks to ChromaDB...[/bold blue]")
        embeddings = model.encode(documents).tolist()
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        rich_print("[bold green]Indexing complete.[/bold green]")
    else:
        rich_print("[bold yellow]No relevant files found for indexing.[/bold yellow]")

def retrieve_relevant_chunks(query, n_results=N_RESULTS):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection("code_context")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=['documents', 'metadatas']
    )

    retrieved_chunks = []
    if results and results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            source = results['metadatas'][0][i]['source']
            retrieved_chunks.append(f"### Relevant Code from {os.path.basename(source)}\n```\n{doc}\n```\n")
    return "\n\n".join(retrieved_chunks)

def detect_intent(user_input: str) -> str:
    """
    Detects the user's intent based on keywords in the input.
    Returns a string representing the intent type (e.g., "code_edit", "reasoning", "explanation").
    """
    user_input_lower = user_input.lower()

    # Code Edit Intent
    code_edit_keywords = ["fix", "bug", "refactor", "change", "implement", "add", "remove", "modify", "update"]
    if any(keyword in user_input_lower for keyword in code_edit_keywords):
        return "code_edit"

    # Explanation Intent
    explanation_keywords = ["explain", "describe", "what is", "how does", "tell me about", "define"]
    if any(keyword in user_input_lower for keyword in explanation_keywords):
        return "explanation"

    # Reasoning Intent (default if no other specific intent is detected)
    reasoning_keywords = ["why", "reason", "analyze", "understand", "debug", "troubleshoot"]
    if any(keyword in user_input_lower for keyword in reasoning_keywords):
        return "reasoning"
    
    # Default intent if no specific keywords match
    return "reasoning"

def parse_code(code: str) -> cst.CSTNode:
    dedented_code = textwrap.dedent(code.strip())
    try:
        return cst.parse_expression(dedented_code)
    except cst.ParserSyntaxError:
        pass
    try:
        return cst.parse_statement(dedented_code)
    except cst.ParserSyntaxError:
        pass
    try:
        return cst.parse_module(dedented_code)
    except cst.ParserSyntaxError as e:
        raise ValueError(f"Failed to parse code: {e}")


def structural_equals(node1: cst.CSTNode, node2: cst.CSTNode) -> bool:
    if type(node1) != type(node2):
        return False
    ignore_fields = {
        'leading_lines', 'trailing_whitespace', 'whitespace_before', 'whitespace_after',
        'whitespace_before_operator', 'whitespace_after_operator', 'whitespace_before_colon',
        'whitespace_after_colon', 'whitespace_before_comma', 'whitespace_after_comma',
        'whitespace_before_comment', 'comment', 'has_trailing_comma', 'lines_after_decorators',
        'whitespace_before_params', 'whitespace_after_def', 'whitespace_after_class',
        'whitespace_before_arrow', 'whitespace_after_arrow', 'whitespace_after_async',
    }
    for field in node1._fields:
        if field in ignore_fields:
            continue
        v1 = getattr(node1, field)
        v2 = getattr(node2, field)
        if isinstance(v1, cst.CSTNode):
            if not structural_equals(v1, v2):
                return False
        elif isinstance(v1, (list, tuple)):
            if len(v1) != len(v2):
                return False
            for a, b in zip(v1, v2):
                if isinstance(a, cst.CSTNode):
                    if not structural_equals(a, b):
                        return False
                elif a != b:
                    return False
        elif v1 != v2:
            return False
    return True

class ReplaceTransformer(m.MatcherDecoratableTransformer):
    def __init__(self, old_node: cst.CSTNode, new_node: cst.CSTNode):
        super().__init__()
        self.old_node = old_node
        self.new_node = new_node

    @m.leave(m.MatchIfTrue(lambda node: structural_equals(node, self.old_node)))
    def replace_exact_match(self, original_node: cst.CSTNode, updated_node: cst.CSTNode) -> cst.CSTNode:
        if type(original_node) == type(self.new_node):
            changes = {}
            ignore_fields = {
                'leading_lines', 'trailing_whitespace', 'whitespace_before', 'whitespace_after',
                'whitespace_before_operator', 'whitespace_after_operator', 'whitespace_before_colon',
                'whitespace_after_colon', 'whitespace_before_comma', 'whitespace_after_comma',
                'whitespace_before_comment', 'comment', 'has_trailing_comma', 'lines_after_decorators',
                'whitespace_before_params', 'whitespace_after_def', 'whitespace_after_class',
                'whitespace_before_arrow', 'whitespace_after_arrow', 'whitespace_after_async',
            }
            for field in original_node._fields:
                if field in ignore_fields and hasattr(self.new_node, field):
                    changes[field] = getattr(original_node, field)
            return self.new_node.with_changes(**changes)
        return self.new_node

def apply_file_edit(file_path, old_code, new_code):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        old_node = parse_code(old_code)
        new_node = parse_code(new_code)
        
        wrapper = cst.MetadataWrapper(cst.parse_module(content))
        original_tree = wrapper.module
        
        transformer = ReplaceTransformer(old_node, new_node)
        modified_wrapper = wrapper.visit(transformer)
        modified_tree = modified_wrapper.module
        
        if modified_tree.deep_equals(original_tree):
            rich_print(f"[bold red]Error: The code to be replaced was not found in {file_path} (structural mismatch).[/bold red]")
            return False

        new_content = modified_tree.code

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        rich_print(f"[green]✅ Successfully updated {file_path}[/green]")
        return True

    except Exception as e:
        rich_print(f"[bold red]Error applying changes: {e}[/bold red]")
        return False

def handle_file_edit(response_text, history):
    edit_pattern = re.compile(r"""```file_edit
file_path: \"(.*?)\"\nold_code: \|(.*?)\nnew_code: \|(.*?) 
```""", re.DOTALL)
    
    # Use finditer to get all matches
    matches = list(edit_pattern.finditer(response_text))

    clean_response = edit_pattern.sub("", response_text).strip()
    
    if matches: # Process all found matches
        for match in matches: 
            file_path, old_code, new_code = match.groups()
            old_code = old_code.strip()
            new_code = new_code.strip()

            rich_print(Panel(f"[yellow]The model proposes a code change to:[/yellow] [bold]{file_path}[/bold]", title="[bold blue]Code Edit Proposal[/bold blue]"))
            
            diff_lines = unified_diff(
                old_code.splitlines(keepends=True),
                new_code.splitlines(keepends=True),
                fromfile="Original",
                tofile="Proposed",
            )
            diff_text = "".join(diff_lines)
            rich_print(Syntax(diff_text, "diff", theme="monokai", line_numbers=True))

            if Confirm.ask("[bold yellow]Do you want to apply this change?[/bold yellow]"):
                if not apply_file_edit(file_path, old_code, new_code): # Check return value
                    history.append({"role": "user", "content": f"Previous file edit to {file_path} failed: old_code not found or other error. Please retry with correct old_code."})
            else:
                rich_print("[red]Change rejected by user.[/red]")
                history.append({"role": "user", "content": f"Previous file edit to {file_path} was rejected by user."})
    else:
        # If no file_edit blocks are found, print the clean response
        if clean_response:
            rich_print(Markdown(clean_response))


def chat(model, directory):
    history = load_history()
    if not history:
        history.append({"role": "system", "content": SYSTEM_PROMPT})
    rich_print(Panel(f"[bold green]🤖 Local Claude-CLI[/bold green] (Ollama model: [bold]{model}[/bold])", subtitle='Type ":exit" to quit'))

    vscode_project, vscode_context = get_vscode_context()
    if vscode_project:
        rich_print(f"[green]🟢 Detected VS Code project:[/green] [dim]{vscode_project}[/dim]")
    else:
        vscode_project = directory

    create_vector_db(vscode_project) # Call to create vector DB

    while True:
        try:
            rich_print("[bold cyan]You[/bold cyan]: ", end="")
            user_input = input().strip()
            if not user_input:
                continue
            if user_input.lower() in [":exit", ":quit"]:
                save_history(history)
                rich_print("[bold yellow]👋 Bye![/bold yellow]")
                break

            # Model Switching Logic
            detected_intent = detect_intent(user_input)
            target_model = model # Default to current model

            if detected_intent == "code_edit":
                for m_name in CODE_EDIT_MODELS:
                    try:
                        ollama.show(m_name) # Check if model exists
                        target_model = m_name
                        break
                    except ollama.ResponseError:
                        continue
            elif detected_intent == "reasoning":
                for m_name in REASONING_MODELS:
                    try:
                        ollama.show(m_name)
                        target_model = m_name
                        break
                    except ollama.ResponseError:
                        continue
            elif detected_intent == "explanation":
                for m_name in EXPLANATION_MODELS:
                    try:
                        ollama.show(m_name)
                        target_model = m_name
                        break
                    except ollama.ResponseError:
                        continue
            
            if target_model != model:
                rich_print(f"[bold magenta]Switching model to: {target_model} for {detected_intent} task.[/bold magenta]")
                model = target_model
            else:
                rich_print(f"[dim]Using current model: {model} for {detected_intent} task.[/dim]")

            relevant_code_context = retrieve_relevant_chunks(user_input) # Retrieve relevant chunks
            context_msg = ""
            if vscode_context:
                context_msg += f"\n**VS Code Active File Context:**\n{vscode_context}\n"
            if relevant_code_context:
                context_msg += f"\n**Project Context (RAG):**\n{relevant_code_context}\n"
            if context_msg:
                user_input = f"{context_msg}\n**User request:**\n{user_input}"

            history.append({"role": "user", "content": user_input})

            response_text = ""
            with Live(auto_refresh=False) as live:
                stream = ollama.chat(model=model, messages=history, stream=True)
                for chunk in stream:
                    content = chunk['message']['content']
                    response_text += content
                    live.update(response_text.split("```file_edit")[0], refresh=True)
            
            handle_file_edit(response_text, history)

            history.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            rich_print("\n[yellow]Interrupted. Type :exit to quit.[/yellow]")
        except Exception as e:
            rich_print(f"\n[bold red]❌ Error: {e}[/bold red]")

def main():
    parser = argparse.ArgumentParser(description="Claude-CLI style interface for Ollama with VS Code integration")
    parser.add_argument("mode", choices=["chat"], help="Mode: chat")
    parser.add_argument("--model", default="llama3", help="Ollama model to use")
    parser.add_argument("--dir", default=".", help="Fallback project directory")
    args = parser.parse_args()

    if args.mode == "chat":
        chat(args.model, args.dir)

if __name__ == "__main__":
    main()