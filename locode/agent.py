import os
import json
from openai import OpenAI
from typing import List, Dict, Any
from .tools import TOOLS
from rich.console import Console

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
        if name == "read_file":
            try:
                with open(args["path"], "r") as f:
                    return f.read()
            except Exception as e:
                return f"Error reading file: {e}"
        elif name == "write_file":
            try:
                with open(args["path"], "w") as f:
                    f.write(args["content"])
                return f"Successfully wrote to {args['path']}"
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
