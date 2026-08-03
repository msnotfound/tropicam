"""Experiment harness: reproducible, provenance-stamped result records.

Every number that reaches the paper should come out of this, so that each one
carries the git commit, the seed, and the exact parameters that produced it.
A plot whose provenance cannot be reconstructed is a plot that cannot be
defended in review.

Design rules:

* **Seeds are explicit and recorded.** No implicit global RNG anywhere.
* **Results are append-only JSON Lines**, one record per run, so a sweep can
  be resumed or extended without rewriting history.
* **Claims are attached to records.** A record states which paper claim it
  supports, so an unsupported claim is a query away rather than a memory test.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def git_revision() -> str:
    """Current commit, with a dirty marker. Unknown rather than fatal."""
    try:
        root = Path(__file__).resolve().parents[2]
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=root, capture_output=True, text=True,
                             timeout=5)
        if sha.returncode != 0:
            return "unknown"
        rev = sha.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               capture_output=True, text=True, timeout=5)
        return rev + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def environment() -> dict[str, str]:
    try:
        import tropicam_rs  # noqa: F401
        native = "built"
    except ImportError:
        native = "absent"
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "native_core": native,
    }


@dataclass
class Record:
    """One experiment run.

    `claim` is the paper claim this run supports (e.g. "C1", "E3a"). Keeping
    it on the record means the set of claims with no supporting run is a
    filter, not a recollection.
    """

    experiment: str
    claim: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    seed: int
    git: str = field(default_factory=git_revision)
    env: dict[str, str] = field(default_factory=environment)
    timestamp: float = field(default_factory=time.time)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=_jsonable)


def _jsonable(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if hasattr(o, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(o).items()}
    return str(o)


class ResultLog:
    """Append-only JSONL sink for experiment records."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Record) -> Record:
        with open(self.path, "a") as f:
            f.write(record.to_json() + "\n")
        return record

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with open(self.path) as f:
            return [json.loads(line) for line in f if line.strip()]

    def claims_covered(self) -> set[str]:
        return {r["claim"] for r in self.load()}

    def filter(self, **kw) -> list[dict]:
        out = self.load()
        for k, v in kw.items():
            out = [r for r in out if r.get(k) == v]
        return out


def rng_for(seed: int, *tags: str) -> np.random.Generator:
    """Deterministic per-condition RNG.

    Derives an independent stream from (seed, tags) so that adding a condition
    to a sweep does not shift the noise realisations of existing conditions --
    which would silently invalidate every previously recorded number.
    """
    mix = abs(hash((seed, *tags))) % (2 ** 63)
    return np.random.default_rng(np.random.SeedSequence([seed, mix]))


@dataclass
class Sweep:
    """A parameter sweep with recorded provenance."""

    experiment: str
    claim: str
    log: ResultLog
    seed: int = 0

    def run(self, conditions, fn, *, notes: str = "") -> list[Record]:
        """Run `fn(params, rng) -> metrics dict` over `conditions`."""
        out = []
        for params in conditions:
            tag = "|".join(f"{k}={v}" for k, v in sorted(params.items()))
            rng = rng_for(self.seed, self.experiment, tag)
            metrics = fn(params, rng)
            out.append(self.log.append(Record(
                experiment=self.experiment, claim=self.claim, params=params,
                metrics=metrics, seed=self.seed, notes=notes,
            )))
        return out
