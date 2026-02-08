import typer
from locode.agent import LocodeAgent
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def main(
    instruction: str = typer.Argument(None, help="Initial instruction"),
    model: str = "llama3.1",
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Run in interactive mode")
):
    """
    Locode: A local coding agent.
    """
    console.print(f"[bold green]Starting Locode with model: {model}[/bold green]")
    agent = LocodeAgent(model=model)

    if instruction:
        agent.run(instruction)

    if interactive:
        from rich.prompt import Prompt
        console.print("[bold yellow]Entering Interactive Mode. Type 'exit' or 'quit' to leave.[/bold yellow]")
        while True:
            try:
                user_input = Prompt.ask("[bold blue]You[/bold blue]")
                if user_input.lower() in ["exit", "quit"]:
                    break
                agent.run(user_input)
            except KeyboardInterrupt:
                console.print("\n[bold yellow]Exiting...[/bold yellow]")
                break

if __name__ == "__main__":
    app()
