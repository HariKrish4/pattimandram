# Pattimandram

A notebook demonstrating a five-LLM Pattimandram debate: two teams of two speakers and a judge.

## Requirements

- Python 3.10 or newer
- [uv](https://docs.astral.sh/uv/)
- An OpenRouter API key

## Setup

Install the project dependencies:

```sh
uv sync
```

Create a `.env` file in the project directory and set your OpenRouter API key:

```dotenv
OPENROUTER_API_KEY=your-api-key
```

Open `pattimandram.ipynb` in Jupyter and run the cells. The notebook sends requests to OpenRouter and may incur API usage charges.
