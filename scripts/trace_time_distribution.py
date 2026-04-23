"""Compute LLM vs. tool-action time distributions from agent trace directories.

Supports three trace formats, auto-detected per trace directory:

  1. claude-code (session JSONL)
     Path pattern: <root>/<task>-results/<slug>/agent/sessions/projects/<cwd>/<session>.jsonl
     Each LLM call emits a sequence of assistant events sharing a requestId.
     tool_use and tool_result events are separate records with their own
     timestamps, so per-call LLM and action durations are exact.

  2. terminus-2 (ATIF trajectory.json)
     Path pattern: <root>/<run>/<task>/agent/trajectory.json
     Each agent step bundles an LLM call plus execution of its `tool_calls`.
     The harness sends keystrokes, sleeps exactly `duration` seconds, then
     records step timestamp. So for step N (agent):
         turn_ms    = step_N.ts - step_{N-1}.ts
         action_ms  = sum(cmd.duration) * 1000
         llm_ms     = turn_ms - action_ms
     Empirically delta >= sum(duration) on every step in these traces.

  3. mini-swe-agent (<instance>.traj.json)
     Path pattern: <root>/<instance>/<instance>.traj.json
     Each assistant message carries `extra.timestamp` (when harness received
     the reply) and `extra.response.created` (provider-side). User (tool-
     result) messages also carry `extra.timestamp`. So per turn:
         llm_ms    = assistant.ts - previous_msg_with_ts.ts
         action_ms = next_user.ts - assistant.ts
     For the first assistant (no prior timestamped message), LLM time is
     anchored at provider's response.created.

Outputs per root: a directory under --out-root (default
results/trace_time_distribution/<label>) containing a summary, a per-trace
CSV, and CDF + breakdown plots.

Usage:
  # single root
  python3 scripts/trace_time_distribution.py --root <dir> --label <name>

  # multiple roots
  python3 scripts/trace_time_distribution.py \
      --root dirA --label nameA --root dirB --label nameB ...

  # the --auto-tbench-terminus flag expands each
  # <root>/<Terminus2__X>/<date>/ as a separate root.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover
    plt = None


@dataclass
class Interval:
    start_ms: float
    end_ms: float

    @property
    def dur_ms(self) -> float:
        return self.end_ms - self.start_ms


@dataclass
class TraceStats:
    task_id: str
    session_id: str
    source_path: str
    n_requests: int
    n_tools: int
    llm_ms: list[float] = field(default_factory=list)
    action_ms: list[float] = field(default_factory=list)
    wall_ms: float = 0.0
    # Union of LLM / action intervals (claude-code only; equals sum for terminus).
    active_llm_ms: float = 0.0
    active_action_ms: float = 0.0
    # Per-turn pairing: llm + its following action. Each entry is for a turn
    # that has both a non-zero LLM part and a non-zero action part.
    turn_llm_ms: list[float] = field(default_factory=list)
    turn_action_ms: list[float] = field(default_factory=list)
    turn_action_frac: list[float] = field(default_factory=list)


def parse_ts(s: str) -> float:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp() * 1000.0


def union_duration(intervals: Iterable[Interval]) -> float:
    xs = sorted(((i.start_ms, i.end_ms) for i in intervals), key=lambda p: p[0])
    total = 0.0
    cur_s: float | None = None
    cur_e: float = 0.0
    for s, e in xs:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total


# ---------------- claude-code (session JSONL) ----------------

def analyze_claude_code_session(path: Path, task_id: str) -> TraceStats | None:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    events = []
    for r in rows:
        if r.get("type") == "queue-operation":
            continue
        msg = r.get("message")
        if not isinstance(msg, dict):
            continue
        ts = parse_ts(r["timestamp"])
        role = msg.get("role") or r.get("type")
        req_id = r.get("requestId")
        content = msg.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        elif not isinstance(content, list):
            content = []
        events.append((ts, role, req_id, content))

    if not events:
        return None

    events.sort(key=lambda e: e[0])

    req_events: dict[str, list[tuple[float, list[dict]]]] = defaultdict(list)
    non_assistant: list[tuple[float, str, list[dict]]] = []
    for ts, role, req_id, content in events:
        if role == "assistant" and req_id:
            req_events[req_id].append((ts, content))
        else:
            non_assistant.append((ts, role, content))

    llm_intervals: list[Interval] = []
    llm_ms: list[float] = []
    req_info: list[tuple[str, float]] = []  # (req_id, per-request llm_ms)
    req_order = sorted(req_events.items(), key=lambda kv: kv[1][0][0])
    for req_id, asst_events in req_order:
        first_ts = asst_events[0][0]
        last_ts = asst_events[-1][0]
        prior_inputs = [t for (t, _role, _c) in non_assistant if t < first_ts]
        if not prior_inputs:
            continue
        start_ts = max(prior_inputs)
        if last_ts <= start_ts:
            continue
        llm_ms.append(last_ts - start_ts)
        llm_intervals.append(Interval(start_ts, last_ts))
        req_info.append((req_id, last_ts - start_ts))

    # Collect tool_uses per request so we can build turn pairs.
    tool_use_ts: dict[str, float] = {}
    tool_use_req: dict[str, str] = {}
    tool_use_order: list[str] = []
    tool_result_ts: dict[str, float] = {}
    for ts, role, req_id, content in events:
        for c in content:
            if not isinstance(c, dict):
                continue
            ctype = c.get("type")
            if ctype == "tool_use":
                tid = c.get("id")
                if tid and tid not in tool_use_ts:
                    tool_use_ts[tid] = ts
                    if role == "assistant" and req_id:
                        tool_use_req[tid] = req_id
                    tool_use_order.append(tid)
            elif ctype == "tool_result":
                tid = c.get("tool_use_id")
                if tid and tid not in tool_result_ts:
                    tool_result_ts[tid] = ts

    action_intervals: list[Interval] = []
    action_ms: list[float] = []
    req_actions: dict[str, list[Interval]] = defaultdict(list)
    for tid in tool_use_order:
        start = tool_use_ts[tid]
        end = tool_result_ts.get(tid)
        if end is None or end <= start:
            continue
        iv = Interval(start, end)
        action_ms.append(iv.dur_ms)
        action_intervals.append(iv)
        rid = tool_use_req.get(tid)
        if rid:
            req_actions[rid].append(iv)

    turn_llm_ms: list[float] = []
    turn_action_ms: list[float] = []
    turn_action_frac: list[float] = []
    for rid, rllm in req_info:
        ivs = req_actions.get(rid) or []
        if not ivs:
            continue
        act = union_duration(ivs)
        if rllm <= 0 or act <= 0:
            continue
        turn_llm_ms.append(rllm)
        turn_action_ms.append(act)
        turn_action_frac.append(act / (rllm + act))

    wall_ms = events[-1][0] - events[0][0]
    return TraceStats(
        task_id=task_id,
        session_id=path.stem,
        source_path=str(path),
        n_requests=len(llm_ms),
        n_tools=len(action_ms),
        llm_ms=llm_ms,
        action_ms=action_ms,
        wall_ms=wall_ms,
        active_llm_ms=union_duration(llm_intervals),
        active_action_ms=union_duration(action_intervals),
        turn_llm_ms=turn_llm_ms,
        turn_action_ms=turn_action_ms,
        turn_action_frac=turn_action_frac,
    )


# ---------------- terminus-2 (ATIF trajectory.json) ----------------

def analyze_terminus_trajectory(path: Path, task_id: str) -> TraceStats | None:
    try:
        traj = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    steps = traj.get("steps") or []
    if not steps:
        return None

    llm_ms: list[float] = []
    action_ms: list[float] = []
    turn_llm_ms: list[float] = []
    turn_action_ms: list[float] = []
    turn_action_frac: list[float] = []
    first_ts = None
    last_ts = None
    n_tools = 0
    prev_ts: float | None = None
    for s in steps:
        try:
            ts = parse_ts(s["timestamp"])
        except (KeyError, ValueError):
            continue
        if first_ts is None:
            first_ts = ts
        last_ts = ts
        if s.get("source") == "agent" and prev_ts is not None:
            tc = s.get("tool_calls") or []
            sum_dur_s = 0.0
            for c in tc:
                args = c.get("arguments") or {}
                d = args.get("duration")
                if isinstance(d, (int, float)):
                    sum_dur_s += float(d)
            act_ms = sum_dur_s * 1000.0
            turn_ms = ts - prev_ts
            llm_part = max(0.0, turn_ms - act_ms)
            if act_ms > 0:
                action_ms.append(act_ms)
            if llm_part > 0:
                llm_ms.append(llm_part)
            if llm_part > 0 and act_ms > 0:
                turn_llm_ms.append(llm_part)
                turn_action_ms.append(act_ms)
                turn_action_frac.append(act_ms / (llm_part + act_ms))
            n_tools += len(tc)
        prev_ts = ts

    if first_ts is None or last_ts is None:
        return None
    wall_ms = last_ts - first_ts
    return TraceStats(
        task_id=task_id,
        session_id=traj.get("session_id", path.stem),
        source_path=str(path),
        n_requests=len(llm_ms),
        n_tools=n_tools,
        llm_ms=llm_ms,
        action_ms=action_ms,
        wall_ms=wall_ms,
        active_llm_ms=sum(llm_ms),
        active_action_ms=sum(action_ms),
        turn_llm_ms=turn_llm_ms,
        turn_action_ms=turn_action_ms,
        turn_action_frac=turn_action_frac,
    )


# ---------------- mini-swe-agent (traj.json) ----------------

def analyze_mini_swe_trajectory(path: Path, task_id: str) -> TraceStats | None:
    try:
        traj = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    msgs = traj.get("messages") or []
    if not msgs:
        return None

    # Collect timestamped views in message order.
    events: list[tuple[int, str, float | None, float | None]] = []
    for i, m in enumerate(msgs):
        role = m.get("role")
        ex = m.get("extra") or {}
        ts = ex.get("timestamp")
        created = None
        if role == "assistant":
            resp = ex.get("response") or {}
            c = resp.get("created")
            if isinstance(c, (int, float)):
                created = float(c)
        events.append((i, role or "", float(ts) if isinstance(ts, (int, float)) else None, created))

    llm_ms: list[float] = []
    action_ms: list[float] = []
    turn_llm_ms: list[float] = []
    turn_action_ms: list[float] = []
    turn_action_frac: list[float] = []
    n_tools = 0
    first_ts: float | None = None
    last_ts: float | None = None

    for i, role, ts, created in events:
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts

    if first_ts is None:
        return None

    for idx, (i, role, ts, created) in enumerate(events):
        if role != "assistant" or ts is None:
            continue

        # LLM time: prior timestamped message → this assistant.ts
        prior_ts = None
        for j in range(idx - 1, -1, -1):
            _, _r, _ts, _ = events[j]
            if _ts is not None:
                prior_ts = _ts
                break
        if prior_ts is None and created is not None:
            prior_ts = created
        if prior_ts is not None and ts > prior_ts:
            llm_ms.append((ts - prior_ts) * 1000.0)

        # Action time: next user (or any timestamped) message
        next_ts = None
        for j in range(idx + 1, len(events)):
            _, r2, ts2, _ = events[j]
            if ts2 is not None and r2 == "user":
                next_ts = ts2
                break
        # Count action only if there were commands
        ex = msgs[i].get("extra") or {}
        acts = ex.get("actions") or []
        if acts and next_ts is not None and next_ts > ts:
            act_ms = (next_ts - ts) * 1000.0
            action_ms.append(act_ms)
            n_tools += len(acts)
            llm_part = llm_ms[-1] if llm_ms and prior_ts is not None and ts > prior_ts else None
            if llm_part is not None and llm_part > 0 and act_ms > 0:
                turn_llm_ms.append(llm_part)
                turn_action_ms.append(act_ms)
                turn_action_frac.append(act_ms / (llm_part + act_ms))

    wall_ms = (last_ts - first_ts) * 1000.0 if last_ts is not None and first_ts is not None else 0.0
    return TraceStats(
        task_id=task_id,
        session_id=traj.get("instance_id", path.stem),
        source_path=str(path),
        n_requests=len(llm_ms),
        n_tools=n_tools,
        llm_ms=llm_ms,
        action_ms=action_ms,
        wall_ms=wall_ms,
        active_llm_ms=sum(llm_ms),
        active_action_ms=sum(action_ms),
        turn_llm_ms=turn_llm_ms,
        turn_action_ms=turn_action_ms,
        turn_action_frac=turn_action_frac,
    )


# ---------------- format dispatch & discovery ----------------

def discover_traces(root: Path) -> tuple[str, list[tuple[Path, str]]]:
    """Return (format, [(source_path, task_id), ...])."""
    cc = sorted(root.glob("*-results/*/agent/sessions/projects/*/*.jsonl"))
    if cc:
        out = []
        for p in cc:
            # .../<task_id>-results/<slug>/agent/sessions/projects/<cwd>/<session>.jsonl
            try:
                task = p.parts[-6].removesuffix("-results")
            except IndexError:
                task = p.stem
            out.append((p, task))
        return "claude-code", out

    tr = sorted(root.glob("*/agent/trajectory.json"))
    if tr:
        out = [(p, p.parts[-3]) for p in tr]
        return "terminus-2", out

    ms = sorted(root.glob("*/*.traj.json"))
    if ms:
        out = [(p, p.parts[-2]) for p in ms]
        return "mini-swe-agent", out

    # fallback: recurse a bit
    tr = sorted(root.glob("**/agent/trajectory.json"))
    if tr:
        out = [(p, p.parts[-3]) for p in tr]
        return "terminus-2", out

    return "unknown", []


def analyze_one(fmt: str, src: Path, task_id: str) -> TraceStats | None:
    if fmt == "claude-code":
        return analyze_claude_code_session(src, task_id)
    if fmt == "terminus-2":
        return analyze_terminus_trajectory(src, task_id)
    if fmt == "mini-swe-agent":
        return analyze_mini_swe_trajectory(src, task_id)
    return None


# ---------------- reporting ----------------

def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    frac = k - lo
    return xs[lo] + (xs[hi] - xs[lo]) * frac


def summary_line(label: str, xs: list[float]) -> str:
    if not xs:
        return f"  {label}: (empty)"
    return (
        f"  {label}: n={len(xs)} "
        f"mean={statistics.mean(xs):.1f}ms "
        f"median={percentile(xs,50):.1f}ms "
        f"p90={percentile(xs,90):.1f}ms "
        f"p99={percentile(xs,99):.1f}ms "
        f"max={max(xs):.1f}ms "
        f"sum={sum(xs)/1000:.1f}s"
    )


def plot_cdf(ax, xs: list[float], label: str) -> None:
    if not xs:
        return
    xs_sorted = sorted(xs)
    ys = [(i + 1) / len(xs_sorted) for i in range(len(xs_sorted))]
    ax.plot(xs_sorted, ys, label=f"{label} (n={len(xs_sorted)})")


def report(label: str, fmt: str, traces: list[TraceStats], out_dir: Path,
           skipped: list[tuple[Path, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_llm = [x for t in traces for x in t.llm_ms]
    all_act = [x for t in traces for x in t.action_ms]
    sum_llm_per_trace = [sum(t.llm_ms) for t in traces]
    sum_act_per_trace = [sum(t.action_ms) for t in traces]
    wall_per_trace = [t.wall_ms for t in traces]
    active_llm_per_trace = [t.active_llm_ms for t in traces]
    active_act_per_trace = [t.active_action_ms for t in traces]
    all_turn_frac = [f for t in traces for f in t.turn_action_frac]
    all_turn_llm = [x for t in traces for x in t.turn_llm_ms]
    all_turn_act = [x for t in traces for x in t.turn_action_ms]
    per_trace_mean_frac = [
        statistics.mean(t.turn_action_frac) for t in traces if t.turn_action_frac
    ]

    total_llm = sum(sum_llm_per_trace)
    total_act = sum(sum_act_per_trace)
    total_wall = sum(wall_per_trace)
    total_active_llm = sum(active_llm_per_trace)
    total_active_act = sum(active_act_per_trace)

    lines: list[str] = []
    lines.append(f"label:            {label}")
    lines.append(f"format:           {fmt}")
    lines.append(f"analysed traces:  {len(traces)}")
    lines.append(f"skipped:          {len(skipped)}")
    lines.append("")
    lines.append("=== per-call distribution ===")
    lines.append(summary_line("llm_ms (per request)", all_llm))
    lines.append(summary_line("action_ms (per tool/step)", all_act))
    lines.append("")
    lines.append("=== per-trace totals ===")
    lines.append(summary_line("llm_ms sum", sum_llm_per_trace))
    lines.append(summary_line("action_ms sum", sum_act_per_trace))
    lines.append(summary_line("wall_ms", wall_per_trace))
    if fmt == "claude-code":
        lines.append(summary_line("active_llm_ms (union)", active_llm_per_trace))
        lines.append(summary_line("active_action_ms (union)", active_act_per_trace))
    lines.append("")
    lines.append("=== aggregate share ===")
    if total_llm + total_act > 0:
        share_llm = 100.0 * total_llm / (total_llm + total_act)
        lines.append(f"  sum-based    : LLM {share_llm:5.1f}% / action {100-share_llm:5.1f}%")
    if fmt == "claude-code" and total_active_llm + total_active_act > 0:
        share_llm_a = 100.0 * total_active_llm / (total_active_llm + total_active_act)
        lines.append(f"  union-based  : LLM {share_llm_a:5.1f}% / action {100-share_llm_a:5.1f}%")
    if total_wall > 0:
        lines.append(
            f"  of wall-clock: LLM {100*total_active_llm/total_wall:5.1f}% "
            f"action {100*total_active_act/total_wall:5.1f}%  "
            f"(wall total {total_wall/1000:.0f}s)"
        )

    lines.append("")
    lines.append("=== per-turn action fraction (turn = LLM call + its next action) ===")
    if not all_turn_frac:
        lines.append("  (no paired turns found)")
    else:
        # distribution stats on the fraction itself
        n = len(all_turn_frac)
        m = statistics.mean(all_turn_frac)
        med = percentile(all_turn_frac, 50)
        p10 = percentile(all_turn_frac, 10)
        p25 = percentile(all_turn_frac, 25)
        p75 = percentile(all_turn_frac, 75)
        p90 = percentile(all_turn_frac, 90)
        p99 = percentile(all_turn_frac, 99)
        lines.append(
            f"  per-turn frac : n={n} mean={m:.3f} "
            f"p10={p10:.3f} p25={p25:.3f} p50={med:.3f} p75={p75:.3f} p90={p90:.3f} p99={p99:.3f}"
        )
        # weighted (sum-based) — treats each ms equally
        sum_turn_act = sum(all_turn_act)
        sum_turn_llm = sum(all_turn_llm)
        denom = sum_turn_act + sum_turn_llm
        if denom > 0:
            lines.append(
                f"  weighted frac : sum_action / (sum_llm + sum_action) = "
                f"{sum_turn_act/denom:.3f}  "
                f"(action {sum_turn_act/1000:.0f}s, llm {sum_turn_llm/1000:.0f}s, "
                f"{n} turns)"
            )
        if per_trace_mean_frac:
            mt = statistics.mean(per_trace_mean_frac)
            mtm = percentile(per_trace_mean_frac, 50)
            lines.append(
                f"  per-trace mean: traces={len(per_trace_mean_frac)} "
                f"mean-of-means={mt:.3f} median-of-means={mtm:.3f}"
            )

    text = "\n".join(lines)
    print(text)
    (out_dir / "summary.txt").write_text(text + "\n")

    with (out_dir / "per_trace.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "task_id", "session_id", "n_requests", "n_tools",
            "sum_llm_ms", "sum_action_ms", "wall_ms",
            "active_llm_ms", "active_action_ms",
            "n_turns", "mean_turn_action_frac",
            "weighted_turn_action_frac",
            "source_path",
        ])
        for t in traces:
            n_turns = len(t.turn_action_frac)
            mean_frac = (statistics.mean(t.turn_action_frac) if n_turns else float("nan"))
            ta = sum(t.turn_action_ms)
            tl = sum(t.turn_llm_ms)
            wfrac = (ta / (ta + tl)) if (ta + tl) > 0 else float("nan")
            w.writerow([
                t.task_id, t.session_id, t.n_requests, t.n_tools,
                f"{sum(t.llm_ms):.0f}", f"{sum(t.action_ms):.0f}",
                f"{t.wall_ms:.0f}",
                f"{t.active_llm_ms:.0f}", f"{t.active_action_ms:.0f}",
                n_turns,
                "" if n_turns == 0 else f"{mean_frac:.4f}",
                "" if (ta + tl) == 0 else f"{wfrac:.4f}",
                t.source_path,
            ])

    if skipped:
        with (out_dir / "skipped.txt").open("w") as f:
            for p, reason in skipped:
                f.write(f"{p}\t{reason}\n")

    if plt is None:
        print("matplotlib not available; skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    plot_cdf(ax, all_llm, "LLM per request")
    plot_cdf(ax, all_act, "Action per step")
    ax.set_xscale("log")
    ax.set_xlabel("Duration (ms, log)")
    ax.set_ylabel("CDF")
    ax.set_title(f"{label} — per-call CDF")
    ax.grid(True, which="both", ls=":")
    ax.legend()

    ax = axes[1]
    plot_cdf(ax, sum_llm_per_trace, "sum LLM")
    plot_cdf(ax, sum_act_per_trace, "sum action")
    plot_cdf(ax, wall_per_trace, "wall")
    ax.set_xscale("log")
    ax.set_xlabel("Duration (ms, log)")
    ax.set_ylabel("CDF")
    ax.set_title(f"{label} — per-trace totals CDF")
    ax.grid(True, which="both", ls=":")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "time_distribution_cdf.png", dpi=120)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 4))
    if active_llm_per_trace and active_act_per_trace:
        means = [
            statistics.mean(active_llm_per_trace) / 1000,
            statistics.mean(active_act_per_trace) / 1000,
            max(
                0.0,
                (statistics.mean(wall_per_trace)
                 - statistics.mean(active_llm_per_trace)
                 - statistics.mean(active_act_per_trace)) / 1000,
            ),
        ]
        ax2.bar(["active LLM", "active action", "other/overlap"], means)
        ax2.set_ylabel("Seconds (mean per trace)")
        ax2.set_title(f"{label} — mean time breakdown")
        ax2.grid(True, axis="y", ls=":")
        fig2.tight_layout()
        fig2.savefig(out_dir / "time_breakdown_bar.png", dpi=120)
    plt.close(fig2)

    if all_turn_frac:
        fig3, ax3 = plt.subplots(figsize=(6, 4.5))
        plot_cdf(ax3, all_turn_frac, "per-turn action fraction")
        if per_trace_mean_frac:
            plot_cdf(ax3, per_trace_mean_frac, "per-trace mean of fractions")
        ax3.set_xlim(0, 1)
        ax3.set_xlabel("action_ms / (llm_ms + action_ms)  per turn")
        ax3.set_ylabel("CDF")
        ax3.set_title(f"{label} — turn action fraction CDF")
        ax3.grid(True, ls=":")
        ax3.legend()
        fig3.tight_layout()
        fig3.savefig(out_dir / "turn_action_fraction_cdf.png", dpi=120)
        plt.close(fig3)


# ---------------- CLI ----------------

def iter_root_label_pairs(args) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    if args.auto_tbench_terminus:
        base = Path(args.auto_tbench_terminus)
        for model_dir in sorted(base.iterdir()):
            if not model_dir.is_dir() or not model_dir.name.startswith("Terminus"):
                continue
            for run_dir in sorted(model_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                label = f"{model_dir.name}__{run_dir.name}"
                pairs.append((run_dir, label))
    if args.root:
        labels = args.label or []
        if labels and len(labels) != len(args.root):
            raise SystemExit("--label count must match --root count")
        for i, r in enumerate(args.root):
            lbl = labels[i] if labels else Path(r).name
            pairs.append((Path(r), lbl))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", default=[],
                    help="trace root (can be given multiple times)")
    ap.add_argument("--label", action="append", default=[],
                    help="label for --root (same count as --root). "
                         "Defaults to basename.")
    ap.add_argument("--auto-tbench-terminus", type=str, default=None,
                    help="shortcut: expand <base>/Terminus2__<model>/<run> "
                         "into one root per run.")
    ap.add_argument("--out-root", type=Path,
                    default=Path("results/trace_time_distribution"),
                    help="output directory; each label gets a subdir.")
    args = ap.parse_args()

    pairs = iter_root_label_pairs(args)
    if not pairs:
        raise SystemExit("no --root or --auto-tbench-terminus supplied")

    for root, label in pairs:
        print("\n" + "=" * 72)
        print(f">>> {label}  ({root})")
        if not root.is_dir():
            print(f"!! not a directory; skipping")
            continue
        fmt, srcs = discover_traces(root)
        if not srcs:
            print(f"!! no traces discovered (format={fmt}); skipping")
            continue
        traces: list[TraceStats] = []
        skipped: list[tuple[Path, str]] = []
        for src, task in srcs:
            try:
                st = analyze_one(fmt, src, task)
            except Exception as exc:  # noqa: BLE001
                skipped.append((src, f"exception: {exc}"))
                continue
            if st is None:
                skipped.append((src, "empty/parse-fail"))
                continue
            traces.append(st)
        out_dir = args.out_root / label
        report(label, fmt, traces, out_dir, skipped)
        print(f"written: {out_dir}")


if __name__ == "__main__":
    main()
