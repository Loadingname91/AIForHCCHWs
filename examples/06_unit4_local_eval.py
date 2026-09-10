"""
Example 6 — Unit 4: score your agent against GAIA on your own machine.

    python examples/06_unit4_local_eval.py --list            # see the questions
    python examples/06_unit4_local_eval.py --run --limit 3   # answer the first 3
    python examples/06_unit4_local_eval.py --run             # answer all 20

This is a local practice harness, not a submission tool. It fetches the 20
public GAIA questions the course uses, runs your agent over them, and writes a
report you can read. Nothing is uploaded anywhere.

Because it runs on your own model, you can do this as many times as you like
for free — which is the whole point. Reading twenty traces and noticing *why*
each one failed teaches more than any single score does.

Note that no answer key is published, so this cannot tell you if you were
right. What it does tell you is where your agent falls apart: a step limit hit,
a tool that errored, an answer wrapped in prose that exact-match scoring would
reject. Those are the fixable things.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from importlib import import_module
from pathlib import Path

import requests

from cs6961_agents import describe_backend

sys.path.insert(0, str(Path(__file__).parent))
gaia = import_module("05_unit4_gaia_agent")

API = "https://agents-course-unit4-scoring.hf.space"
TIMEOUT = 60


def fetch_questions() -> list[dict]:
    """The question set is public; no token needed."""
    response = requests.get(f"{API}/questions", timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="print the questions and exit")
    group.add_argument("--run", action="store_true", help="run your agent over them")
    parser.add_argument("--limit", type=int, default=0, help="only do the first N")
    parser.add_argument(
        "--out", default="gaia_report.json", help="where to write the report"
    )
    args = parser.parse_args()

    print(f"Backend: {describe_backend()}\n")

    questions = fetch_questions()
    if args.limit:
        questions = questions[: args.limit]
    print(f"Fetched {len(questions)} question(s) from {API}\n")

    if args.list:
        for i, item in enumerate(questions, 1):
            attachment = f"  [attachment: {item['file_name']}]" if item.get("file_name") else ""
            print(f"{i:2d}. {item['question'][:150]}{attachment}")
        return 0

    agent = gaia.build_agent()
    report = []

    for i, item in enumerate(questions, 1):
        print(f"\n{'=' * 68}\n[{i}/{len(questions)}] {item['question'][:200]}\n{'=' * 68}")
        started = time.time()
        try:
            answer = gaia.finalize(agent.run(item["question"]))
            error = None
        except Exception as exc:  # noqa: BLE001 — one bad question must not end the run
            print(f"  !! agent error: {type(exc).__name__}: {exc}")
            answer, error = "", f"{type(exc).__name__}: {exc}"

        elapsed = time.time() - started
        print(f"  -> {answer!r}   ({elapsed:.1f}s)")

        report.append({
            "question": item["question"],
            "has_attachment": bool(item.get("file_name")),
            "answer": answer,
            "error": error,
            "seconds": round(elapsed, 1),
        })

    # -- summary. No answer key exists, so report what CAN be measured. --
    errors = [r for r in report if r["error"]]
    empty = [r for r in report if not r["error"] and not r["answer"]]
    wordy = [r for r in report if len(r["answer"].split()) > 6]
    attached = [r for r in report if r["has_attachment"]]

    print(f"\n{'=' * 68}\nSUMMARY\n{'=' * 68}")
    print(f"  answered             {len(report) - len(errors) - len(empty)}/{len(report)}")
    print(f"  crashed              {len(errors)}")
    print(f"  came back empty      {len(empty)}")
    print(f"  suspiciously wordy   {len(wordy)}   <- exact-match scoring would reject these")
    print(f"  needed an attachment {len(attached)}   <- see docs/05, the file endpoint is down")
    print(f"  total time           {sum(r['seconds'] for r in report):.0f}s")

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Full report: {args.out}")
    print("  Read the wordy and crashed ones first — those are the fixable failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
