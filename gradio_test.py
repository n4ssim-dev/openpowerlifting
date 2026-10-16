"""
Minimal Groq chat UI with Gradio.

Setup:
    pip install groq gradio python-dotenv
    echo "GROQ_API_KEY=your_key_here" > .env

Run:
    python groq_chat.py
    -> open http://localhost:7860
"""
import os

import gradio as gr
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise SystemExit("GROQ_API_KEY is missing: add it to a .env file next to this script.")

client = Groq(api_key=api_key)
MODEL = "openai/gpt-oss-120b"
SYSTEM_PROMPT = "You are a helpful, concise assistant. Answer in the language of the user's last message."


def _text(content) -> str:
    # Recent Gradio versions may return history content as blocks [{"type": "text", "text": ...}].
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text")
    return ""


def respond(message: str, history: list[dict]):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [
        {"role": m["role"], "content": _text(m["content"])}
        for m in history
        if m["role"] in ("user", "assistant") and _text(m["content"])
    ]
    messages.append({"role": "user", "content": message})

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=1500,
        stream=True,
    )
    reply = ""
    for chunk in stream:
        reply += chunk.choices[0].delta.content or ""
        yield reply


demo = gr.ChatInterface(
    fn=respond,
    title="Groq Chat",
    description=f"Model: {MODEL}",
)

if __name__ == "__main__":
    demo.launch()