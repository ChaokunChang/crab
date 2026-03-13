from __future__ import annotations

import os
import struct
from pathlib import Path


PAGEMAP_ENTRY_SIZE = 8
SOFT_DIRTY_BIT = 55
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def list_cgroup_pids(cgroup_path: str | None) -> set[int]:
    if not cgroup_path:
        return set()
    path = Path("/sys/fs/cgroup") / cgroup_path.lstrip("/") / "cgroup.procs"
    if not path.exists():
        return set()
    return {int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def clear_soft_dirty(pid: int) -> None:
    Path(f"/proc/{pid}/clear_refs").write_text("4\n", encoding="utf-8")


def reset_soft_dirty_for_pids(pids: set[int]) -> set[int]:
    cleared: set[int] = set()
    for pid in sorted(pids):
        try:
            clear_soft_dirty(pid)
            cleared.add(pid)
        except FileNotFoundError:
            continue
    return cleared


def parse_writable_ranges(pid: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    with open(f"/proc/{pid}/maps", "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 2:
                continue
            if "w" not in parts[1]:
                continue
            start_hex, end_hex = parts[0].split("-", 1)
            start = int(start_hex, 16)
            end = int(end_hex, 16)
            if end > start:
                ranges.append((start, end))
    return ranges


def range_has_soft_dirty(pid: int, start: int, end: int) -> bool:
    start_page = start // PAGE_SIZE
    end_page = (end + PAGE_SIZE - 1) // PAGE_SIZE
    remaining = end_page - start_page
    if remaining <= 0:
        return False

    chunk_pages = 4096
    with open(f"/proc/{pid}/pagemap", "rb", buffering=0) as handle:
        current_page = start_page
        while remaining > 0:
            pages_now = min(remaining, chunk_pages)
            handle.seek(current_page * PAGEMAP_ENTRY_SIZE)
            data = handle.read(pages_now * PAGEMAP_ENTRY_SIZE)
            if len(data) != pages_now * PAGEMAP_ENTRY_SIZE:
                return False
            for offset in range(0, len(data), PAGEMAP_ENTRY_SIZE):
                entry = struct.unpack_from("Q", data, offset)[0]
                if (entry >> SOFT_DIRTY_BIT) & 1:
                    return True
            current_page += pages_now
            remaining -= pages_now
    return False


def pid_has_soft_dirty(pid: int) -> bool:
    try:
        for start, end in parse_writable_ranges(pid):
            if range_has_soft_dirty(pid, start, end):
                return True
    except FileNotFoundError:
        return False
    return False


def dirty_pids(pids: set[int]) -> set[int]:
    dirty: set[int] = set()
    for pid in sorted(pids):
        if pid_has_soft_dirty(pid):
            dirty.add(pid)
    return dirty
