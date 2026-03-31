#!/usr/bin/env python3
import argparse
import dataclasses
import html as html_lib
import json
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import requests


UA = "tbench-trace-downloader/1.0"
DOWNLOAD_ACTION_ID = "60d4f52179607920f9068c57b1a3b92175733665e6"


@dataclasses.dataclass
class TrajectoryRecord:
    trial_id: str
    path: str
    results_path: str | None
    session_id: str | None
    schema_version: str | None
    agent_name: str | None
    agent_version: str | None
    model_name: str | None
    total_steps: int | None
    result: str | None
    source_url: str


def to_relpath(path: Path, base: Path) -> str:
    return str(path.relative_to(base))


def fetch_text(session: requests.Session, url: str) -> str:
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.text


def extract_trial_ids(submission_html: str) -> list[str]:
    ids = re.findall(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', submission_html)
    # Keep only the small set of trial IDs present on the submission page.
    # This avoids accidentally collecting unrelated UUIDs from the page shell.
    ids = [trial_id for trial_id in ids if trial_id in submission_html]
    # Preserve order but dedupe.
    seen = set()
    out = []
    for trial_id in ids:
        if trial_id not in seen:
            seen.add(trial_id)
            out.append(trial_id)
    return out


def extract_task_checksums(submission_html: str) -> list[tuple[str, str]]:
    # The parent submission page embeds a task table with taskName/taskChecksum pairs.
    pairs = re.findall(
        r'taskName\\":\\"([^"]+)\\",\\"taskChecksum\\":\\"([a-f0-9]{64})\\"',
        submission_html,
    )
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for task_name, checksum in pairs:
        if checksum not in seen:
            seen.add(checksum)
            out.append((task_name, checksum))
    return out


def extract_trajectory_object(page_html: str) -> dict[str, Any]:
    marker = r'\"trajectory\":{'
    idx = page_html.find(marker)
    if idx == -1:
        raise ValueError("trajectory payload not found")

    # Extract the script chunk containing the payload, then decode the escaped JSON text.
    chunk = page_html[idx : page_html.find("</script>", idx)]
    chunk = chunk.encode("utf-8").decode("unicode_escape")

    start = chunk.find("{")
    if start == -1:
        raise ValueError("trajectory JSON start not found")

    level = 0
    in_str = False
    esc = False
    end = None
    for i in range(start, len(chunk)):
        ch = chunk[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                level += 1
            elif ch == "}":
                level -= 1
                if level == 0:
                    end = i + 1
                    break

    if end is None:
        raise ValueError("trajectory JSON end not found")

    return json.loads(chunk[start:end])


def strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = html_lib.unescape(text)
    text = text.replace("\u2713", "").replace("\u2717", "")
    text = " ".join(text.split())
    return text.strip()


def extract_result_value(page_html: str) -> str | None:
    match = re.search(
        r'>Result</p><p[^>]*>(.*?)</p>',
        page_html,
        flags=re.DOTALL,
    )
    if not match:
        return None
    value = strip_tags(match.group(1))
    if value.lower() == "pass":
        return "Passed"
    if value.lower() == "fail":
        return "Failed"
    return value or None


def call_trial_file_download_url(
    session: requests.Session,
    page_url: str,
    trial_id: str,
    file_type: str,
) -> str | None:
    headers = {
        "Next-Action": DOWNLOAD_ACTION_ID,
        "Accept": "text/x-component",
        "Content-Type": "text/plain;charset=UTF-8",
        "Origin": "https://www.tbench.ai",
        "Referer": page_url,
    }
    body = json.dumps([trial_id, file_type])
    resp = session.post(page_url, headers=headers, data=body.encode("utf-8"), timeout=60)
    resp.raise_for_status()
    for line in resp.text.splitlines():
        if line.startswith("1:"):
            return json.loads(line[2:])
    return None


def download_json(session: requests.Session, url: str) -> dict[str, Any]:
    resp = session.get(url, timeout=60)
    resp.raise_for_status()
    return resp.json()


def safe_extract_tar(archive_path: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (dest_dir / member.name).resolve()
            if not str(member_path).startswith(str(dest_dir.resolve())):
                raise ValueError(f"unsafe tar entry: {member.name}")
        tar.extractall(dest_dir)


def download_and_extract_results(
    session: requests.Session,
    signed_url: str,
    results_dir: Path,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with session.get(signed_url, timeout=120, stream=True) as resp:
            resp.raise_for_status()
            with tmp_path.open("wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
        safe_extract_tar(tmp_path, results_dir)
    finally:
        tmp_path.unlink(missing_ok=True)


def summarize_trajectory(traj: dict[str, Any]) -> TrajectoryRecord:
    agent = traj.get("agent") or {}
    return TrajectoryRecord(
        trial_id=traj.get("trial_id") or traj.get("session_id") or "",
        path="",
        results_path=None,
        session_id=traj.get("session_id"),
        schema_version=traj.get("schema_version"),
        agent_name=agent.get("name"),
        agent_version=agent.get("version"),
        model_name=agent.get("model_name"),
        total_steps=len(traj.get("steps", [])) if isinstance(traj.get("steps"), list) else None,
        result=None,
        source_url="",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download all TBench trajectories for a submission page."
    )
    parser.add_argument("url", help="Submission page URL, e.g. https://www.tbench.ai/leaderboard/.../claude-opus-4-6%40anthropic")
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Directory to save trajectory JSON files and manifest.",
    )
    parser.add_argument(
        "--manifest",
        default="manifest.json",
        help="Manifest filename inside output-dir.",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    manifest_path = outdir / args.manifest
    records: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    def write_manifest() -> None:
        manifest = {
            "source_url": args.url,
            "output_dir": ".",
            "trajectory_count": len(records),
            "skipped_count": len(skipped),
            "skipped": skipped,
            "trajectories": records,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    submission_html = fetch_text(session, args.url)
    task_checksums = extract_task_checksums(submission_html)
    if not task_checksums:
        raise SystemExit("No task checksums found on the submission page.")
    for task_name, checksum in task_checksums:
        task_url = args.url.rstrip("/") + f"/{checksum}"
        task_html = fetch_text(session, task_url)
        trial_ids = extract_trial_ids(task_html)
        if not trial_ids:
            raise SystemExit(f"No trial IDs found on task page: {task_url}")

        for trial_id in trial_ids:
            trial_url = task_url + f"/{trial_id}"
            try:
                html = fetch_text(session, trial_url)
                result_value = extract_result_value(html)
                trajectory_download_url = call_trial_file_download_url(session, trial_url, trial_id, "trajectory")
                if not trajectory_download_url:
                    skipped.append(
                        {
                            "task_name": task_name,
                            "task_checksum": checksum,
                            "trial_id": trial_id,
                            "source_url": trial_url,
                            "reason": "no trajectory download URL",
                        }
                    )
                    print(f"skipped {task_name} {trial_id} (no trajectory download URL)")
                    write_manifest()
                    continue

                traj = download_json(session, trajectory_download_url)
                results_download_url = call_trial_file_download_url(session, trial_url, trial_id, "tarGz")
                record = summarize_trajectory(traj)
                record.trial_id = trial_id
                traj_path = outdir / f"{trial_id}-traj.json"
                record.path = to_relpath(traj_path, outdir)
                record.result = result_value
                record.source_url = trial_url
                if results_download_url:
                    results_dir = outdir / f"{trial_id}-results"
                    if results_dir.exists():
                        shutil.rmtree(results_dir)
                    download_and_extract_results(session, results_download_url, results_dir)
                    record.results_path = to_relpath(results_dir, outdir)

                traj["_local"] = {
                    "result": record.result,
                    "results_path": record.results_path,
                    "source_url": record.source_url,
                    "task_name": task_name,
                    "task_checksum": checksum,
                }
                traj_path.write_text(json.dumps(traj, indent=2), encoding="utf-8")
                rec = dataclasses.asdict(record)
                rec["task_name"] = task_name
                rec["task_checksum"] = checksum
                records.append(rec)
                write_manifest()
                print(f"saved {task_name} {trial_id} -> {traj_path}")
            except Exception as exc:
                skipped.append(
                    {
                        "task_name": task_name,
                        "task_checksum": checksum,
                        "trial_id": trial_id,
                        "source_url": trial_url,
                        "reason": str(exc),
                    }
                )
                write_manifest()
                print(f"skipped {task_name} {trial_id} ({exc})")

    write_manifest()
    print(f"wrote manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
