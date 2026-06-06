"""Small helper for writing viewer manifests from experiment scripts."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Manifest:
    run_id: str
    command: str | None = None
    seed: int | None = None
    notes: str | None = None
    config_path: str | None = None
    commit: str | None = None
    created: str = field(default_factory=utc_now_iso)
    examples: list[dict[str, Any]] = field(default_factory=list)
    planning: list[dict[str, Any]] = field(default_factory=list)
    latent_space: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None

    def add_example(
        self,
        example_id: str,
        video: str,
        *,
        label: str | None = None,
        metrics: dict[str, Any] | None = None,
        frames: list[str] | None = None,
        context_range: list[int] | tuple[int, int] | None = None,
        target_range: list[int] | tuple[int, int] | None = None,
        embedding: list[float] | tuple[float, float] | None = None,
        embedding_hi: str | None = None,
        prediction: dict[str, Any] | None = None,
        perturbations: list[dict[str, Any]] | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        item: dict[str, Any] = {"id": example_id, "video": video}
        optional = {
            "label": label,
            "metrics": metrics,
            "frames": frames,
            "context_range": list(context_range) if context_range is not None else None,
            "target_range": list(target_range) if target_range is not None else None,
            "embedding": list(embedding) if embedding is not None else None,
            "embedding_hi": embedding_hi,
            "prediction": prediction,
            "perturbations": perturbations,
        }
        item.update({key: value for key, value in optional.items() if value is not None})
        item.update({key: value for key, value in extra.items() if value is not None})
        self.examples.append(item)
        return item

    def add_planning(self, plan: dict[str, Any]) -> dict[str, Any]:
        self.planning.append(plan)
        return plan

    def to_dict(self) -> dict[str, Any]:
        run = {
            "id": self.run_id,
            "created": self.created,
            "command": self.command or " ".join(sys.argv),
        }
        for key in ("seed", "notes", "config_path", "commit"):
            value = getattr(self, key)
            if value is not None:
                run[key] = value

        data: dict[str, Any] = {"run": run, "examples": self.examples}
        if self.planning:
            data["planning"] = self.planning
        if self.latent_space:
            data["latent_space"] = self.latent_space
        if self.analysis:
            data["analysis"] = self.analysis
        return data

    def write(self, run_dir: str | Path) -> Path:
        path = Path(run_dir)
        path.mkdir(parents=True, exist_ok=True)
        manifest_path = path / "manifest.json"
        manifest_path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return manifest_path


def load_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    return json.loads(manifest_path.read_text())

