"""
Gemma Chatbot — A simple terminal chatbot powered by a local Gemma model via Ollama.

Usage:
    python chatbot.py

Prerequisites:
    - Ollama must be running locally (default: http://localhost:11434)
    - The 'gemma:latest' model must be available in Ollama
"""

import json
import sys

import requests

# ── Configuration ────────────────────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "gemma4:12b"
SYSTEM_PROMPT = (
    "You are a helpful, friendly AI assistant powered by Gemma 4. "
    "Answer questions clearly and concisely."
)


def check_ollama_running() -> bool:
    """Verify that the Ollama server is reachable."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False


def check_model_available() -> bool:
    """Verify that the target model is pulled and available."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if resp.status_code != 200:
            return False
        models = resp.json().get("models", [])
        return any(m.get("name", "").startswith("gemma4") or m.get("name", "").startswith("gemma") for m in models)
    except (requests.ConnectionError, json.JSONDecodeError):
        return False


def chat(messages: list[dict]) -> str:
    """
    Send the conversation history to Ollama and return the assistant's reply.
    Uses streaming so the response prints token-by-token for a nicer UX.
    """
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": True,
    }

    full_response = ""
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                token = chunk.get("message", {}).get("content", "")
                full_response += token
                # Print each token as it arrives (streaming effect)
                print(token, end="", flush=True)
            print()  # newline after the full response
    except requests.exceptions.Timeout:
        print("\n[Error] Request timed out. The model may be loading — try again.")
    except requests.exceptions.ConnectionError:
        print("\n[Error] Lost connection to Ollama. Is the server still running?")
    except requests.exceptions.HTTPError as exc:
        print(f"\n[Error] HTTP error: {exc}")

    return full_response


def main() -> None:
    """Run the interactive chatbot loop."""
    # ── Pre-flight checks ────────────────────────────────────────────────
    print("=" * 60)
    print("  🤖  Gemma Chatbot  (local via Ollama)")
    print("=" * 60)

    if not check_ollama_running():
        print(
            "\n❌  Cannot reach Ollama at {url}.\n"
            "    Make sure Ollama is running:  ollama serve\n".format(url=OLLAMA_BASE_URL)
        )
        sys.exit(1)

    if not check_model_available():
        print(
            "\n⚠️  Model '{model}' not found.\n"
            "    Pull it first:  ollama pull gemma\n".format(model=MODEL_NAME)
        )
        sys.exit(1)

    print(f"\n✅  Connected to Ollama  •  Model: {MODEL_NAME}")
    print("    Type your message and press Enter. Commands:")
    print("      /clear  — reset conversation history")
    print("      /quit   — exit the chatbot\n")

    # ── Conversation state ───────────────────────────────────────────────
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            continue

        # ── Slash commands ───────────────────────────────────────────────
        if user_input.lower() == "/quit":
            print("\nGoodbye! 👋")
            break

        if user_input.lower() == "/clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[Conversation cleared]\n")
            continue

        # ── Send to model ────────────────────────────────────────────────
        messages.append({"role": "user", "content": user_input})

        print("\nGemma: ", end="", flush=True)
        assistant_reply = chat(messages)

        if assistant_reply:
            messages.append({"role": "assistant", "content": assistant_reply})
        else:
            # Remove the user message if the model didn't respond
            messages.pop()

        print()  # extra blank line for readability


if __name__ == "__main__":
    main()
