"""Transaction API v1 (roadmap B2) — snapshot-based, weak isolation.

A transaction wraps a span of sandbox work between an adaptive base
checkpoint and an explicit resolution:

- ``commit`` delivers the staged observations and drops a freshly-taken
  base checkpoint;
- ``abort`` drops the staged observations (gated LLM callers get a 409)
  and restores the sandbox to the base.

The sandbox executes txn actions *in place* (concurrent readers see the
dirt — strong isolation is B3's fork-backed mode); the airtight part is
observation staging: nothing gated escapes an uncommitted transaction.

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


@dataclass(frozen=True)
class TxnCommitResult:
    txn_id: str
    released_observations: int
    base_dropped: bool


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
    def resolved(self) -> str | None:
        """None while open; "committed" or "aborted" afterwards."""
        return self._resolved

    def exec(self, cmd=None, *, argv=None, **kwargs) -> "SandboxExecResult":
        """Sugar over ``sandbox.commands.run`` so txn code reads
        ``txn.exec(...)``. Every exec during the txn is journal-tagged
        with the txn_id regardless of which API issued it."""
        self._ensure_open()
        return self._sandbox.commands.run(cmd, argv=argv, **kwargs)

    def commit(self) -> TxnCommitResult:
        self._ensure_open()
        result = self._sandbox._engine.system.commit_txn(
            self._sandbox.sandbox_id, self.txn_id
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
    "TxnCommitResult",
    "TxnDescription",
    "TxnError",
    "TxnMismatchError",
    "TxnResolvedError",
    "new_txn_id",
]
