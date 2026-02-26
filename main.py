import argparse

from src.app.runners import build_runner
from src.app.runtime import build_runtime


def parse_args():
    parser = argparse.ArgumentParser(description="Run the local RAG assistant.")
    parser.add_argument(
        "--mode",
        choices=("vanilla", "agentic"),
        default="vanilla",
        help="Choose the execution strategy.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading components...")

    runtime = build_runtime()
    runner = build_runner(args.mode, runtime)

    print(f"Ready in {runner.mode} mode.\nType quit() in order to exit")
    while True:
        question = input("> ").strip()
        if not question:
            continue
        if question == "quit()":
            break

        result = runner.run(question)
        print("\n", result.answer, "\n")


if __name__ == "__main__":
    main()
