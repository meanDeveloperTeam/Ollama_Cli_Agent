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

import chromadb
from sentence_transformers import SentenceTransformer
import glob

# RAG Configuration
CHROMA_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
N_RESULTS = 3 # Number of relevant chunks to retrieve

HISTORY_FILE = os.path.expanduser("~/.local_llm_history.json")

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


def load_project_context(directory, max_files=3, max_chars=1500):
    context = []
    try:
        files = []
        for root, _, filenames in os.walk(directory):
            for name in filenames:
                if name.endswith((".js", ".ts", ".py", ".html", ".md", ".txt")):
                    files.append(os.path.join(root, name))
        files = sorted(files, key=os.path.getmtime, reverse=True)[:max_files]
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read(max_chars)
                    context.append(f"\n### File: {os.path.basename(fpath)}\n{content}\n")
            except Exception:
                continue
    except Exception:
        pass
    return "\n".join(context)

def apply_file_edit(file_path, old_code, new_code):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        if old_code not in content:
            rich_print(f"[bold red]Error: The code to be replaced was not found in {file_path}.[/bold red]")
            return False

        new_content = content.replace(old_code, new_code, 1)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        rich_print(f"[green]✅ Successfully updated {file_path}[/green]")
        return True

    except Exception as e:
        rich_print(f"[bold red]Error applying changes: {e}[/bold red]")
        return False

def handle_file_edit(response_text):
    edit_pattern = re.compile(r"""```file_edit
file_path: \"(.*?)\"\nold_code: \|(.*?)\nnew_code: \|(.*?) 
```""", re.DOTALL)
    match = edit_pattern.search(response_text)

    clean_response = edit_pattern.sub("", response_text).strip()
    
    # Removed: if clean_response: rich_print(Markdown(clean_response))

    if match:
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
            apply_file_edit(file_path, old_code, new_code)
        else:
            rich_print("[red]Change rejected by user.[/red]")

def chat(model, directory):
    history = load_history()
    rich_print(Panel(f"[bold green]🤖 Local Claude-CLI[/bold green] (Ollama model: [bold]{model}[/bold])", subtitle='Type ":exit" to quit'))

    vscode_project, vscode_context = get_vscode_context()
    if vscode_project:
        rich_print(f"[green]🟢 Detected VS Code project:[/green] [dim]{vscode_project}[/dim]")
    else:
        vscode_project = directory

    project_context = load_project_context(vscode_project)

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

            if len(history) == 0:
                context_msg = ""
                if vscode_context:
                    context_msg += f"\n**VS Code Active File Context:**\n{vscode_context}\n"
                if project_context:
                    context_msg += f"\n**Project Context:**\n{project_context}\n"
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
            
            handle_file_edit(response_text)

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
