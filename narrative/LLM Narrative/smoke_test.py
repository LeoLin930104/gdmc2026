import time
from lm_client import API_BASE_URL, API_MODEL, chat

PROMPTS = [
    (
        "Settlement name",
        "Invent a name for a small Minecraft village built on the edge of a dark forest. "
        "Reply with ONLY the name, nothing else.",
    ),
    (
        "Zone subtitle",
        "Write a single poetic subtitle (10-20 words) for the town center of a coastal village.",
    ),
    (
        "Relic flavor line",
        "Write one italic-style flavor sentence for a nautilus-shell relic called "
        "'Abyssal Conch of Whispers'.",
    ),
]


def main() -> None:
    print(f"Connecting to LLM API at {API_BASE_URL} (model: {API_MODEL}) ...\n")
    for label, prompt in PROMPTS:
        start = time.perf_counter()
        reply = chat(prompt)
        elapsed = time.perf_counter() - start
        print(f"--- {label} ({elapsed:.1f}s) ---")
        print(reply)
        print()


if __name__ == "__main__":
    main()
