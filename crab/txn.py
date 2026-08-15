"""Transaction API (roadmap B2 + B3).

A transaction wraps a span of sandbox work between a base checkpoint
and an explicit resolution. Two isolation modes:

- ``snapshot`` (B2, default): actions run *in place* on the sandbox;
  commit delivers staged observations and drops a freshly-taken base;
  abort drops observations and restores the base (concurrent readers
  see the dirt while the txn is open).
- ``fork`` (B3): begin forks the sandbox and actions run in the fork;
  the source stays clean and serving. Commit promotes the fork's whole
  state (filesystem + processes) back onto the source's identity;
  abort just destroys the fork — the source is never restored.

Either way the airtight part is observation staging: nothing gated
escapes an uncommitted transaction.

All transaction state lives in ``CrabSystem`` (one active txn per
sandbox); the :class:`Transaction` object here is a thin SDK handle so
daemon-mode clients can reattach via ``Sandbox.current_txn()``.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import SandboxExecResult
    from .sandbox import Sandbox


class TxnError(RuntimeError):
    """Base class for transaction failures."""


class TxnActiveError(TxnError):
    """A transaction is already active for the sandbox (one per sandbox)."""


class TxnMismatchError(TxnError):
    """The given txn_id does not match the sandbox's active transaction."""


class TxnResolvedError(TxnError):
    """The SDK handle was already committed or aborted."""


class TxnAbortError(TxnError):
    """Abort could not restore the base checkpoint; the transaction stays
    open (observations are already dropped — dropping is idempotent, so
    retrying abort is safe)."""

    def __init__(self, message: str, *, restore_result: object | None = None) -> None:
        super().__init__(message)
        self.restore_result = restore_result


class TxnCommitConflict(TxnError):
    """Fork-backed commit refused: the source changed since the fork
    point, so promoting the fork would silently discard those writes.
    Commit with ``force=True`` to discard them, or abort the txn."""


def new_txn_id() -> str:
    return f"txn-{uuid.uuid4().hex[:12]}"


@dataclass(frozen=True)
class TxnDescription:
    """System-side record of an active transaction."""

    txn_id: str
    sandbox_id: str
    base_checkpoint_id: str | None
    base_was_fresh: bool
    started_at: str
    label: str | None = None
    isolation: str = "snapshot"
    fork_sandbox_id: str | None = None


@dataclass(frozen=True)
class TxnCommitResult:
    txn_id: str
    released_observations: int
    base_dropped: bool
    promoted_checkpoint_id: str | None = None
    observations_consolidated: int | None = None


@dataclass(frozen=True)
class TxnAbortResult:
    txn_id: str
    discarded_observations: int
    restored_checkpoint_id: str | None


class Transaction:
    """SDK handle for an active transaction.

    Context-manager semantics: commit on clean exit, abort when the block
    raises (the exception still propagates). Resolving twice raises
    :class:`TxnResolvedError`; the system-side registry stays the source
    of truth for what is actually active.
    """

    def __init__(self, sandbox: "Sandbox", description: TxnDescription) -> None:
        self._sandbox = sandbox
        self._description = description
        self._resolved: str | None = None
        self._fork_sandbox: "Sandbox | None" = None

    @property
    def txn_id(self) -> str:
        return self._description.txn_id

    @property
    def base_checkpoint_id(self) -> str | None:
        return self._description.base_checkpoint_id

    @property
    def label(self) -> str | None:
        return self._description.label

    @property
    def isolation(self) -> str:
        return self._description.isolation

    @property
    def fork_sandbox_id(self) -> str | None:
        """The fork actions run in (fork-backed txns only)."""
        return self._description.fork_sandbox_id

    @property
    def resolved(self) -> str | None:
        """None while open; "committed" or "aborted" afterwards."""
        return self._resolved

    def _exec_target(self) -> "Sandbox":
        fork_id = self._description.fork_sandbox_id
        if fork_id is None:
            return self._sandbox
        if self._fork_sandbox is None:
            from .sandbox import Sandbox

            self._fork_sandbox = Sandbox.connect(fork_id, engine=self._sandbox._engine)
        return self._fork_sandbox

    def exec(self, cmd=None, *, argv=None, **kwargs) -> "SandboxExecResult":
        """Sugar over ``sandbox.commands.run`` so txn code reads
        ``txn.exec(...)``. Snapshot txns run in place; fork-backed txns
        route to the fork. Every exec during the txn is journal-tagged
        with the txn_id regardless of which API issued it."""
        self._ensure_open()
        return self._exec_target().commands.run(cmd, argv=argv, **kwargs)

    def commit(self, *, force: bool = False) -> TxnCommitResult:
        """``force`` only affects fork-backed txns: promote even when the
        source changed since the fork point (its writes are discarded)."""
        self._ensure_open()
        kwargs = {"force": True} if force else {}
        result = self._sandbox._engine.system.commit_txn(
            self._sandbox.sandbox_id, self.txn_id, **kwargs
        )
        self._resolved = "committed"
        return result

    def abort(self) -> TxnAbortResult:
        self._ensure_open()
        result = self._sandbox._engine.system.abort_txn(
            self._sandbox.sandbox_id, self.txn_id
        )
        self._resolved = "aborted"
        return result

    def _ensure_open(self) -> None:
        if self._resolved is not None:
            raise TxnResolvedError(
                f"transaction {self.txn_id} already {self._resolved}"
            )

    def __enter__(self) -> "Transaction":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._resolved is not None:
            return None
        if exc_type is None:
            self.commit()
        else:
            self.abort()
        return None


__all__ = [
    "Transaction",
    "TxnAbortError",
    "TxnAbortResult",
    "TxnActiveError",
    "TxnCommitConflict",
    "TxnCommitResult",
    "TxnDescription",
    "TxnError",
    "TxnMismatchError",
    "TxnResolvedError",
    "new_txn_id",
]
