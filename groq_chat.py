"""
Interface de chat pour Groq avec du tracing MLflow.

1) mlflow ui --port 5000
2) python groq_chat.py
3) Chat sur http://localhost:7860, historique dans http://localhost:5000 (Experiments -> "Groq Chat" -> Traces)
"""
import os

import gradio as gr
import mlflow
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Groq Chat")
mlflow.groq.autolog()  # traces every client.chat.completions.create(...) call

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-120b"


@mlflow.trace(name="chat_turn")
def ask_groq(messages: list[dict]) -> str:
    # Note: MLflow's Groq autolog does not trace streaming calls,
    # so we use a normal (non-streaming) request here.
    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        reasoning_effort="medium",
        stream=False,
    )
    return completion.choices[0].message.content or ""


def respond(message: str, history: list[dict]) -> str:
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})
    return ask_groq(messages)


demo = gr.ChatInterface(
    fn=respond,
    title="Interface chat (Groq historisé avec MLflow)",
    description=f"Model: {MODEL} · Traces: http://localhost:5000",
)

if __name__ == "__main__":
    demo.launch()