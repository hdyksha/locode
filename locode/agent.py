import os
import json
from openai import OpenAI
from typing import List, Dict, Any
from .tools import TOOLS
from .utils import is_safe_path, show_diff
from rich.console import Console
from rich.prompt import Confirm
import os

class LocodeAgent:
    def __init__(self, model: str = "llama3.1"):
        self.console = Console()
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )
        self.model = model
        self.history = [
            {"role": "system", "content": "You are a helpful AI coding assistant. You can read files, write files, and run commands. Always use the provided tools to interact with the system."}
        ]

    def run(self, instruction: str):
        self.console.print(f"[bold blue][Agent] Received instruction:[/bold blue] {instruction}")
        self.history.append({"role": "user", "content": instruction})

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    tools=TOOLS,
                    tool_choice="auto",
                )
            except Exception as e:
                self.console.print(f"[bold red]Error connecting to LLM:[/bold red] {e}")
                self.console.print("[yellow]Ensure Ollama is running (e.g., `ollama serve`) and the model is pulled.[/yellow]")
                break

            message = response.choices[0].message
            self.console.print(f"[dim][Agent] Thinking... {message.content[:50] if message.content else ''}...[/dim]") 

            if message.tool_calls:
                self.history.append(message)
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    self.console.print(f"[bold magenta][Agent] Calling tool:[/bold magenta] {function_name} with args: {function_args}")
                    
                    # Execute tool
                    tool_result = self._execute_tool(function_name, function_args)
                    
                    self.history.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": str(tool_result),
                    })
            else:
                self.history.append(message)
                self.console.print(f"[bold green][Agent] Final Answer:[/bold green] {message.content}")
                return message.content

    def _execute_tool(self, name: str, args: Dict[str, Any]) -> str:
        current_dir = os.getcwd()

        if name == "read_file":
            path = args["path"]
            if not is_safe_path(current_dir, path):
                return f"Error: Access denied. Cannot read file outside of {current_dir}"
            
            try:
                with open(path, "r") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"

        elif name == "write_file":
            path = args["path"]
            content = args["content"]
            
            if not is_safe_path(current_dir, path):
                return f"Error: Access denied. Cannot write file outside of {current_dir}"

            # If file exists, show diff and ask for confirmation
            if os.path.exists(path):
                self.console.print(f"[bold yellow]File already exists:[/bold yellow] {path}")
                self.console.print("Showing diff:")
                show_diff(path, content)
                if not Confirm.ask("Do you want to overwrite this file?"):
                    return "Error: User cancelled file write."
            else:
                self.console.print(f"[bold green]Creating new file:[/bold green] {path}")
                self.console.print("Content preview:")
                self.console.print(content[:500] + ("..." if len(content) > 500 else ""))
                if not Confirm.ask("Do you want to create this file?"):
                    return "Error: User cancelled file creation."
            
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
                with open(path, "w") as f:
                    f.write(content)
                return f"Successfully wrote to {path}"
            except Exception as e:
                return f"Error writing file: {e}"

        elif name == "run_command":
            import subprocess
            try:
                result = subprocess.run(args["command"], shell=True, capture_output=True, text=True)
                return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            except Exception as e:
                return f"Error running command: {e}"
        else:
            return f"Unknown tool: {name}"
