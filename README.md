# Pattimandram

A five-LLM Pattimandram debate app and notebook: two teams of two speakers and a judge.

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

## Run the Gradio app

```sh
uv run python app.py
```

Open the local URL printed by Gradio. Set the topic, each team's position, debate
length, and the model used by each speaker and the judge. The app streams the
transcript as speakers respond and displays the judge's verdict at the end.
Requests go to OpenRouter and may incur API usage charges.

## Run the notebook

Open `pattimandram.ipynb` in Jupyter and run the cells. The notebook also sends
requests to OpenRouter and may incur API usage charges.
