"""Local HuggingFace model management.

Snapshots required models to `models/hf/<flat_repo_id>/` so loading is
fully offline after the first download. Resolve helpers fall back to the
HF Hub id if the local snapshot is missing — so first run isn't blocked.
"""

from __future__ import annotations
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.utils import disable_progress_bars

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models" / "hf"

disable_progress_bars()


def _flat(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def local_path(repo_id: str) -> Path:
    return MODELS_DIR / _flat(repo_id)


def is_downloaded(repo_id: str) -> bool:
    p = local_path(repo_id)
    return p.is_dir() and any(p.iterdir())


def resolve(repo_id: str) -> str:
    """Return the local snapshot path if present, else the HF repo id."""
    return str(local_path(repo_id)) if is_downloaded(repo_id) else repo_id


def download(repo_id: str, force: bool = False) -> Path:
    target = local_path(repo_id)
    if is_downloaded(repo_id) and not force:
        print(f"[skip]  {repo_id} already at {target}")
        return target
    target.mkdir(parents=True, exist_ok=True)
    print(f"[fetch] {repo_id} → {target}")
    snapshot_download(repo_id=repo_id, local_dir=str(target))
    print(f"[ok]    {repo_id}")
    return target
