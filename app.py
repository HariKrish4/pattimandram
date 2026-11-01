import os
from collections.abc import Iterator

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL_OPTIONS = [
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3.5-super-vl:free",
    "nvidia/nemotron-3.5-lightning:free",
    "dots-studio/dots-3-note-preview:free",
    "deepseek/deepseek-v4.1-flash-1017",
]

SPEAKERS = {
    "Raja": {
        "team": 1,
        "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
        "persona": (
            "You are Raja, an expert debate speaker with everyday family humor, "
            "sharp wit, and relatable literary insights. Be witty, concise, and "
            "focused on the topic."
        ),
    },
    "Gnanasambhantham": {
        "team": 1,
        "model": "nvidia/nemotron-3.5-super-vl:free",
        "persona": (
            "You are Gnanasambhantham, a scholar and researcher. Use balanced but "
            "strong reasoning, real-world examples, cultural references, and "
            "gentle sarcasm. Stay calm and concise."
        ),
    },
    "Manikandan": {
        "team": 2,
        "model": "nvidia/nemotron-3.5-lightning:free",
        "persona": (
            "You are Manikandan, a rooted scholar and orator. Use storytelling, "
            "situational irony, and cultural anecdotes with thoughtful humor. "
            "Stay on topic."
        ),
    },
    "Barathi": {
        "team": 2,
        "model": "dots-studio/dots-3-note-preview:free",
        "persona": (
            "You are Barathi, a poised speaker with scholarly depth, structured "
            "logic, and emotional conviction. Give fact-based, insightful "
            "perspectives without bullet points."
        ),
    },
}


def _speaker_system_prompt(
    name: str,
    team1_view: str,
    team2_view: str,
) -> str:
    speaker = SPEAKERS[name]
    team = speaker["team"]
    position = team1_view if team == 1 else team2_view
    teammates = (
        "Raja and Gnanasambhantham" if team == 1 else "Manikandan and Barathi"
    )
    opposing_team = (
        "Team 2 (Manikandan and Barathi)"
        if team == 1
        else "Team 1 (Raja and Gnanasambhantham)"
    )
    return (
        f"{speaker['persona']}\n\n"
        f"You are on Team {team} with {teammates}. Defend this position: "
        f"{position}\n"
        f"Respond to the conversation and engage {opposing_team} respectfully. "
        "Address the audience when appropriate. Keep your response between "
        "50 and 100 words."
    )


def _stream_completion(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    conversation: str,
    speaker: str,
) -> Iterator[str]:
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{conversation}\n\n"
                    f"Now respond as {speaker}."
                ),
            },
        ],
        stream=True,
    )
    received_content = False
    for chunk in stream:
        if not chunk.choices:
            continue
        content = chunk.choices[0].delta.content
        if content:
            received_content = True
            yield content
    if not received_content:
        raise gr.Error(f"{speaker}'s model returned an empty response.")


def run_debate(
    topic: str,
    team1_view: str,
    team2_view: str,
    rounds: float,
    raja_model: str,
    gnanasambhantham_model: str,
    manikandan_model: str,
    barathi_model: str,
    judge_model: str,
) -> Iterator[tuple[str, str]]:
    topic = topic.strip()
    team1_view = team1_view.strip()
    team2_view = team2_view.strip()
    if not topic or not team1_view or not team2_view:
        raise gr.Error("Enter a topic and both team positions to start the debate.")

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise gr.Error(
            "OPENROUTER_API_KEY is missing. Add it to the project .env file."
        )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )
    models = {
        "Raja": raja_model,
        "Gnanasambhantham": gnanasambhantham_model,
        "Manikandan": manikandan_model,
        "Barathi": barathi_model,
    }

    conversation = (
        f"Rangaraj: Welcome, everyone! Today's topic is “{topic}”. "
        f"Team 1 (Raja and Gnanasambhantham) will argue: {team1_view}. "
        f"Team 2 (Manikandan and Barathi) will argue: {team2_view}. "
        "Let’s begin with Team 1.\n\n"
        f"Team 1: Present a persuasive opening argument for {team1_view}. "
        "Address the audience and be ready to respond to Team 2.\n\n"
        f"Team 2: Present a persuasive opening argument for {team2_view}. "
        "Address the audience and be ready to challenge Team 1."
    )
    transcript = f"## {topic}\n\n{conversation}\n"
    yield transcript, ""

    for _ in range(int(rounds)):
        for name in ("Raja", "Manikandan", "Gnanasambhantham", "Barathi"):
            team = SPEAKERS[name]["team"]
            header = f"\n### {name} · Team {team} · `{models[name]}`\n\n"
            transcript += header
            reply = ""
            for content in _stream_completion(
                client,
                model=models[name],
                system_prompt=_speaker_system_prompt(name, team1_view, team2_view),
                conversation=conversation,
                speaker=name,
            ):
                reply += content
                yield transcript + reply, ""
            conversation += f"\n\n{name}: {reply}"
            transcript += f"{reply}\n"
            yield transcript, ""

    judge_prompt = (
        "You are Rangaraj, a highly respected and impartial Pattimandram judge. "
        "Evaluate truthfulness, evidence, clarity, logic, responsiveness, and "
        "persuasiveness. Give each speaker constructive feedback, compare both "
        "teams fairly, and announce one clear winner with justification. "
        "Flag unsupported factual claims rather than treating them as established."
    )
    verdict = ""
    for content in _stream_completion(
        client,
        model=judge_model,
        system_prompt=judge_prompt,
        conversation=conversation,
        speaker="Rangaraj",
    ):
        verdict += content
        yield transcript, f"## Rangaraj’s verdict · `{judge_model}`\n\n{verdict}"


with gr.Blocks(
    title="Pattimandram",
) as demo:
    gr.Markdown(
        "# Pattimandram\n"
        "### Five voices. Two perspectives. One impartial verdict.\n"
        "Set the motion and positions, choose a model for each role, and start "
        "the debate.",
        elem_classes=["hero"],
    )
    with gr.Row(elem_classes=["start-action"]):
        start = gr.Button("Start debate", variant="primary", size="lg")

    with gr.Row(elem_classes=["main-layout"]):
        with gr.Column(scale=1, elem_classes=["config-panel"]):
            gr.Markdown("## Debate setup")
            topic = gr.Textbox(
                label="Topic",
                placeholder="e.g. Joint family vs. nuclear family",
                value="Joint family vs Nuclear family",
            )
            team1_view = gr.Textbox(
                label="Team 1 position · Raja & Gnanasambhantham",
                placeholder="What should Team 1 argue?",
                value="Joint families give people more happiness",
                lines=2,
            )
            team2_view = gr.Textbox(
                label="Team 2 position · Manikandan & Barathi",
                placeholder="What should Team 2 argue?",
                value="Nuclear families give people more happiness",
                lines=2,
            )
            rounds = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=2,
                label="Debate rounds",
                info="Each round gives the floor to all four speakers.",
            )
            gr.Markdown("## Cast & models")
            raja_model = gr.Dropdown(
                MODEL_OPTIONS,
                value=SPEAKERS["Raja"]["model"],
                label="Raja · Team 1",
            )
            gnanasambhantham_model = gr.Dropdown(
                MODEL_OPTIONS,
                value=SPEAKERS["Gnanasambhantham"]["model"],
                label="Gnanasambhantham · Team 1",
            )
            manikandan_model = gr.Dropdown(
                MODEL_OPTIONS,
                value=SPEAKERS["Manikandan"]["model"],
                label="Manikandan · Team 2",
            )
            barathi_model = gr.Dropdown(
                MODEL_OPTIONS,
                value=SPEAKERS["Barathi"]["model"],
                label="Barathi · Team 2",
            )
            judge_model = gr.Dropdown(
                MODEL_OPTIONS,
                value="deepseek/deepseek-v4.1-flash-1017",
                label="Rangaraj · Judge",
            )
        with gr.Column(scale=2, elem_classes=["debate-panel"]):
            transcript = gr.Markdown(label="Debate")
            verdict = gr.Markdown(label="Judge's verdict")
    start.click(
        fn=run_debate,
        inputs=[
            topic,
            team1_view,
            team2_view,
            rounds,
            raja_model,
            gnanasambhantham_model,
            manikandan_model,
            barathi_model,
            judge_model,
        ],
        outputs=[transcript, verdict],
    )

if __name__ == "__main__":
    demo.queue().launch(
        show_error=True,
        theme=gr.themes.Soft(
            primary_hue="orange",
            secondary_hue="red",
            neutral_hue="stone",
        ),
        css="""
        .gradio-container { max-width: 1500px !important; }
        .hero { text-align: center; padding: 1.5rem 1rem 1rem; }
        .hero h1 { margin-bottom: 0.25rem; }
        .hero p { color: #777; font-size: 1.05rem; }
        .start-action { justify-content: flex-end; margin-bottom: 0.75rem; }
        .start-action button { min-width: 220px; font-weight: 700; }
        .main-layout { align-items: flex-start !important; gap: 1.25rem; }
        .config-panel, .debate-panel { min-width: 0 !important; }
        .debate-panel { position: sticky; top: 1rem; }
        @media (max-width: 850px) {
            .start-action button { width: 100%; }
            .main-layout { flex-direction: column !important; }
            .main-layout > .column { width: 100% !important; }
            .debate-panel { position: static; }
        }
        """,
    )
