# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare the two native paced paths: the inline driver and the session.

The session extraction (``moss_vl_realtime.session``) is only safe if it drives
the real checkpoint to the same decisions the previously verified driver made.
Both write the same event schema, so this compares the traces field by field and
records the decisions that produced the text.

Example:
    python tests/assets/moss_vl_realtime/compare_paced_paths.py \
        --driver artifacts/moss/native-paced --session artifacts/moss/native-paced-session \
        --out artifacts/moss/session-contract.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

#: Fields the session adds on purpose; their absence in the driver is expected.
SESSION_ONLY_FIELDS = {
    "session_open": {"path"},
    "prefill": {"decision"},
    "session_closed": {"frames_accepted", "silences", "events", "vision_length_at_close", "text_tokens_at_close"},
}


def latest_run(root: Path) -> Path:
    runs = sorted(path for path in root.iterdir() if path.is_dir())
    if not runs:
        raise SystemExit(f"no run directory under {root}")
    return runs[-1]


def load_events(root: Path) -> list[dict[str, Any]]:
    run = latest_run(root)
    lines = (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def load_report(root: Path) -> dict[str, Any]:
    return json.loads((latest_run(root) / "paced-report.json").read_text(encoding="utf-8"))


def strip(event: dict[str, Any], allowed_extra: set[str]) -> dict[str, Any]:
    return {key: value for key, value in event.items() if key != "wall" and key not in allowed_extra}


def compare(driver: list[dict[str, Any]], session: list[dict[str, Any]]) -> dict[str, Any]:
    divergences: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(driver, session)):
        allowed = SESSION_ONLY_FIELDS.get(left["kind"], set())
        if strip(left, allowed) != strip(right, allowed):
            divergences.append({"index": index, "driver": strip(left, allowed), "session": strip(right, allowed)})
    if len(driver) != len(session):
        divergences.append({"index": None, "driver": len(driver), "session": len(session), "reason": "event count"})

    decisions = [
        {
            "event_id": event.get("event_id"),
            "segment_argmax": event.get("segment_argmax"),
            "segment_max": event.get("segment_max"),
            "segment_top5": event.get("segment_top5"),
            "vision_length": event.get("vision_length"),
            "next_position": event.get("next_position"),
        }
        for event in driver
        if event["kind"] == "frame_pushed"
    ]
    return {
        "identical": not divergences,
        "driver_events": len(driver),
        "session_events": len(session),
        "decisions": decisions,
        "divergences": divergences,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", required=True, help="output root of replay_paced.py without --via-session")
    parser.add_argument("--session", required=True, help="output root of replay_paced.py --via-session")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    driver_root, session_root = Path(args.driver).resolve(), Path(args.session).resolve()
    report = compare(load_events(driver_root), load_events(session_root))
    driver_report, session_report = load_report(driver_root), load_report(session_root)
    report["driver_output"] = driver_report["combined_output"]
    report["session_output"] = session_report["combined_output"]
    report["outputs_match"] = driver_report["combined_output"] == session_report["combined_output"]

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["identical"] and report["outputs_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
