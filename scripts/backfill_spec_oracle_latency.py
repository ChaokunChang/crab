"""Backfill oracle latency into existing speculation sidecars and re-emit CSVs.

This is a no-API-cost companion to ``collect_spec_decisions.py``. It walks
existing JSON sidecars under ``--sidecar-root``, derives the oracle's per-turn
wall-clock latency from the *original recorded trajectory file* (cited in
``sidecar.trace_path``), updates the sidecar JSON in place, and re-writes the
side-by-side CSV with the three new columns:

* ``oracle_latency_ms`` — recorded oracle latency from the trace timestamps
* ``draft_latency_ms`` — already captured during collection
* ``speedup`` — ``oracle / draft`` (>1 means draft is faster)

Usage:
    python3 scripts/backfill_spec_oracle_latency.py \\
        --sidecar-root results/spec_decisions \\
        --draft-tag deepseek-chat \\
        --csv-root results/spec_decisions_csv \\
        --run-id 2026-04-30-spec-friendly

If ``--csv-root`` is omitted only the JSON sidecars are updated. If
``--draft-tag`` is omitted every draft tag under the sidecar root is
processed.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from integrations.llm_services.speculation.reconstruct import (  # noqa: E402
    reconstruct_claude_code_turns,
    reconstruct_mini_swe_turns,
    reconstruct_terminus_turns,
)
from integrations.llm_services.speculation.schema import (  # noqa: E402
    AGENT_CLAUDE_CODE,
    AGENT_MINI_SWE,
    AGENT_TERMINUS,
    SpeculationSidecar,
    load_sidecar,
    resolve_side_by_side_csv_path,
    write_side_by_side_csv,
    write_sidecar,
)


logger = logging.getLogger("backfill_spec_oracle_latency")


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    sidecar_root = Path(args.sidecar_root).expanduser().resolve()
    if not sidecar_root.is_dir():
        parser.error(f"--sidecar-root not found: {sidecar_root}")
    csv_root = (
        Path(args.csv_root).expanduser().resolve()
        if args.csv_root and args.csv_root.strip()
        else None
    )

    sidecars = list(_iter_sidecars(sidecar_root, draft_tag=args.draft_tag))
    if not sidecars:
        logger.warning("no sidecars found under %s", sidecar_root)
        return 0
    logger.info("Found %d sidecar(s) to backfill", len(sidecars))

    summary = _BackfillSummary()
    for index, (sidecar_path, draft_tag) in enumerate(sidecars, start=1):
        logger.info("[%d/%d] %s", index, len(sidecars), sidecar_path)
        try:
            stats = _process_sidecar(
                sidecar_path=sidecar_path,
                draft_tag=draft_tag,
                csv_root=csv_root,
                run_id=args.run_id,
                dry_run=args.dry_run,
            )
        except Exception:
            logger.exception("backfill failed for %s", sidecar_path)
            summary.failed += 1
            continue
        summary.merge(stats)

    summary.log()
    return 0 if summary.failed == 0 else 1


def _iter_sidecars(
    sidecar_root: Path, *, draft_tag: str | None
) -> Iterable[tuple[Path, str]]:
    if draft_tag:
        sub = sidecar_root / draft_tag
        if not sub.is_dir():
            logger.warning("draft tag dir not found: %s", sub)
            return
        for path in sorted(sub.rglob("*.spec.json")):
            yield path, draft_tag
        return
    for tag_dir in sorted(p for p in sidecar_root.iterdir() if p.is_dir()):
        for path in sorted(tag_dir.rglob("*.spec.json")):
            yield path, tag_dir.name


def _process_sidecar(
    *,
    sidecar_path: Path,
    draft_tag: str,
    csv_root: Path | None,
    run_id: str | None,
    dry_run: bool,
) -> dict[str, int]:
    sidecar = load_sidecar(sidecar_path)
    trace_path = Path(sidecar.trace_path).expanduser().resolve()
    if not trace_path.is_file():
        raise FileNotFoundError(f"trace not found: {trace_path}")

    if sidecar.agent_kind == AGENT_TERMINUS:
        recon_turns = reconstruct_terminus_turns(trace_path)
    elif sidecar.agent_kind == AGENT_MINI_SWE:
        recon_turns = reconstruct_mini_swe_turns(trace_path)
    elif sidecar.agent_kind == AGENT_CLAUDE_CODE:
        recon_turns = reconstruct_claude_code_turns(trace_path)
    else:
        raise ValueError(f"unsupported agent_kind {sidecar.agent_kind!r}")

    by_index = {t.turn_index: t.oracle_latency_ms for t in recon_turns}
    updated = 0
    skipped = 0
    for turn in sidecar.turns:
        latency = by_index.get(turn.turn_index)
        if latency is None:
            skipped += 1
            continue
        if turn.oracle_latency_ms == latency:
            skipped += 1
            continue
        turn.oracle_latency_ms = latency
        updated += 1

    if dry_run:
        logger.info("Dry-run: would update %d turn(s), skip %d", updated, skipped)
    else:
        write_sidecar(sidecar_path, sidecar)
        if csv_root is not None and run_id:
            csv_path = resolve_side_by_side_csv_path(
                csv_root=csv_root,
                draft_tag=draft_tag,
                run_id=run_id,
                trace_path=trace_path,
            )
            write_side_by_side_csv(csv_path, sidecar)
            logger.info("Wrote CSV %s", csv_path)
        elif csv_root is not None and not run_id:
            logger.warning(
                "skipping CSV write for %s: --csv-root requires --run-id",
                sidecar_path,
            )
    return {"updated": updated, "skipped": skipped}


class _BackfillSummary:
    def __init__(self) -> None:
        self.sidecars = 0
        self.failed = 0
        self.updated_turns = 0
        self.skipped_turns = 0

    def merge(self, stats: dict[str, int]) -> None:
        self.sidecars += 1
        self.updated_turns += int(stats.get("updated", 0))
        self.skipped_turns += int(stats.get("skipped", 0))

    def log(self) -> None:
        logger.info(
            "Done: sidecars=%d failed=%d turns_updated=%d turns_skipped=%d",
            self.sidecars,
            self.failed,
            self.updated_turns,
            self.skipped_turns,
        )


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--sidecar-root",
        default=str(_REPO_ROOT / "results" / "spec_decisions"),
        help="root directory containing JSON sidecars to backfill",
    )
    parser.add_argument(
        "--draft-tag",
        default=None,
        help="restrict to a specific draft tag (default: process every tag dir)",
    )
    parser.add_argument(
        "--csv-root",
        default=None,
        help="re-emit side-by-side CSVs at this root after updating sidecars",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="run id sub-directory for CSV output (required with --csv-root)",
    )
    parser.add_argument("--dry-run", action="store_true", help="don't write any files")
    parser.add_argument("--verbose", action="store_true")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
