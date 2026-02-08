# Locode

Locode involves the development of a local coding agent that runs on your machine and uses local LLMs (via Ollama) to assist with coding tasks.

## Prerequisites

- [Ollama](https://ollama.com/) installed and running.
- Python 3.10+
- `ollama pull llama3.1` (Required for tool support)

## Setup

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd locode
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

```bash
python main.py "Your instruction here"
```
