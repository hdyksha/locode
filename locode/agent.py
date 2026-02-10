import os
import json
import re
import subprocess
from typing import List, Dict, Any
from openai import OpenAI
from rich.console import Console
from rich.prompt import Confirm
from rich.live import Live
from rich.text import Text

from .tools import TOOLS
from .utils import is_safe_path, show_diff
from .schema import AgentAction

class LocodeAgent:
    def __init__(self, model: str = "llama3.1", verbose: bool = False):
        self.console = Console()
        self.verbose = verbose
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # required, but unused
        )
        self.model = model
        
        # Construct system prompt with schema instruction
        schema_json = AgentAction.model_json_schema()
        self.system_prompt = f"""You are a helpful AI coding assistant.
You can read files, write files, and run commands.

You must output your response as a valid JSON object matching the following schema:
{json.dumps(schema_json, indent=2)}

Explanations:
- `thought`: Explain your reasoning and plan.
- `action`: "tool" to use a tool, "finish" to end the task.
- `tool_name`: Name of the tool (read_file, write_file, run_command). Required if action is "tool".
- `tool_args`: Arguments for the tool. Required if action is "tool".
  - write_file: {{"path": "filename", "content": "file content"}}
  - read_file: {{"path": "filename"}}
  - run_command: {{"command": "shell command"}}
- `final_answer`: Final response to the user. Required if action is "finish".

IMPORTANT: output ONLY the JSON object. Do not wrap it in markdown code blocks.
- Do NOT use triple quotes (''' or \"\"\") for strings.
- Escape all newlines within strings as \\n.
- Ensure the JSON is valid and parsable.

Example of correct JSON output:
{{
  "thought": "I will create a file with multi-line content.",
  "action": "tool",
  "tool_name": "write_file",
  "tool_args": {{
    "path": "example.txt",
    "content": "Line 1\\nLine 2\\nLine 3"
  }}
}}
"""
        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def run(self, instruction: str):
        self.console.print(f"[bold blue][Agent] Received instruction:[/bold blue] {instruction}")
        self.history.append({"role": "user", "content": instruction})

        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.history,
                    stream=True,
                )

                full_response = ""
                with Live(Text(""), refresh_per_second=4, console=self.console) as live:
                    for chunk in response:
                        if chunk.choices[0].delta.content:
                            content = chunk.choices[0].delta.content
                            full_response += content
                            live.update(Text(full_response))
                
                self.console.print() # Newline

                if self.verbose:
                    self.console.print(f"[dim]Raw Response:\n{full_response}[/dim]")

                # Parse JSON
                try:
                    # simplistic cleanup for code blocks if model ignores instruction
                    cleaned_response = full_response.strip()
                    if cleaned_response.startswith("```json"):
                        cleaned_response = cleaned_response[7:]
                    if cleaned_response.endswith("```"):
                        cleaned_response = cleaned_response[:-3]
                    cleaned_response = cleaned_response.strip()

                    action_data = AgentAction.model_validate_json(cleaned_response)
                except Exception as e:
                    self.console.print(f"[red]Failed to parse JSON:[/red] {e}")
                    # Give feedback to model
                    self.history.append({"role": "assistant", "content": full_response})
                    self.history.append({"role": "user", "content": f"Error: Your response was not valid JSON matching the schema. Error: {e}. Please try again."})
                    continue

                self.console.print(f"[bold cyan]Thought:[/bold cyan] {action_data.thought}")

                if action_data.action == "finish":
                    self.console.print(f"[bold green]Final Answer:[/bold green] {action_data.final_answer}")
                    self.history.append({"role": "assistant", "content": full_response})
                    return action_data.final_answer

                elif action_data.action == "tool":
                    tool_name = action_data.tool_name
                    tool_args = action_data.tool_args
                    
                    self.history.append({"role": "assistant", "content": full_response})
                    
                    self.console.print(f"[bold magenta]Action:[/bold magenta] Calling {tool_name} with {tool_args}")
                    tool_result = self._execute_tool(tool_name, tool_args)
                    self.console.print(f"[dim]Tool Output: {tool_result}[/dim]")
                    self.history.append({"role": "user", "content": f"Tool Output: {tool_result}"})
                    # Loop continues, model sees tool output and decides next step

            except Exception as e:
                self.console.print(f"[bold red]Error in agent loop:[/bold red] {e}")
                import traceback
                traceback.print_exc()
                break

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
            command = args["command"]
            self.console.print(f"[bold yellow]Executing command:[/bold yellow] {command}")
            if not Confirm.ask("Do you want to run this command?"):
                return "Error: User cancelled command execution."

            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True)
                return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            except Exception as e:
                return f"Error running command: {e}"
        else:
            return f"Unknown tool: {name}"
