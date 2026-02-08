import typer
from locode.agent import LocodeAgent
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def main(instruction: str, model: str = "llama3.1"):
    """
    Locode: A local coding agent.
    """
    console.print(f"[bold green]Starting Locode with model: {model}[/bold green]")
    agent = LocodeAgent(model=model)
    agent.run(instruction)

if __name__ == "__main__":
    app()
