from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from integrations.llm_services.claude_code_trace_replay.service import parse_replay_trace


def main() -> None:
    dataset_path = Path("results/datasets/claude_code_replay_benchmark.jsonl")
    if not dataset_path.exists():
        raise SystemExit(f"Dataset file missing: {dataset_path}")

    dataset_dir = dataset_path.parent
    delays: list[float] = []
    trace_count = 0
    skipped = 0

    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        entry = json.loads(raw_line)
        trace_rel = (
            entry
            .get("llm_service_config", {})
            .get("trace_path")
        )
        if not trace_rel:
            skipped += 1
            continue
        trace_path = (dataset_dir / trace_rel).resolve()
        if not trace_path.exists():
            print(f"missing trace: {trace_path}")
            skipped += 1
            continue
        try:
            parsed = parse_replay_trace(trace_path)
        except Exception as exc:
            print(f"failed to parse {trace_path}: {exc}")
            skipped += 1
            continue
        trace_count += 1
        for delay in parsed.agent_step_delays_ms:
            if delay is not None:
                delays.append(delay)

    if not delays:
        raise SystemExit("no replayed response delays were extracted")

    delays.sort()
    total = len(delays)

    def percentile(p: float) -> float:
        idx = min(total - 1, max(0, int(p / 100.0 * total)))
        return delays[idx]

    stats = {
        "count": total,
        "mean_ms": mean(delays),
        "min_ms": delays[0],
        "max_ms": delays[-1],
        "p50_ms": percentile(50),
        "p95_ms": percentile(95),
        "p99_ms": percentile(99),
    }

    print("response delay stats (ms)")
    for key, value in stats.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"parsed_traces: {trace_count}")
    print(f"skipped_traces: {skipped}")

    cdf_x = delays
    cdf_y = [(i + 1) / total for i in range(total)]
    plt.figure(figsize=(6, 4.5))
    plt.plot(cdf_x, cdf_y, drawstyle="steps-post")
    plt.xlabel("Response delay (ms)")
    plt.ylabel("CDF")
    plt.title("Claude code trace replay response delay CDF")
    plt.grid(True)
    out_path = Path("results") / "claude_llm_response_cdf.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"cdf_plot: {out_path}")


if __name__ == "__main__":
    main()
