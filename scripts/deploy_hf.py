"""Deploy Desk Review to a Hugging Face Gradio Space."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

SPACE_ID = "HaseebArif11/desk-review"
ROOT = Path(__file__).resolve().parents[1]
COPY_ITEMS = (
    "app.py",
    "core.py",
    "keywords.py",
    "test_app.py",
    "requirements.txt",
    ".env.example",
    "pyproject.toml",
    ".gitignore",
    "samples",
    "ARCHITECTURE.md",
)

README_FRONTMATTER = """---
title: Desk Review
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
---

"""

README_BODY = (ROOT / "README.md").read_text(encoding="utf-8")


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print(
            "Set HF_TOKEN (write access) from https://huggingface.co/settings/tokens",
            file=sys.stderr,
        )
        return 1

    api = HfApi(token=token)
    user = api.whoami()["name"]
    if user != "HaseebArif11":
        print(f"Logged in as {user}; expected HaseebArif11.", file=sys.stderr)
        return 1

    try:
        api.space_info(SPACE_ID)
        print(f"Space {SPACE_ID} already exists — updating files.")
    except Exception:
        print(f"Creating Space {SPACE_ID}...")
        api.create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            space_sdk="gradio",
            private=False,
        )

    with tempfile.TemporaryDirectory() as tmp:
        upload_dir = Path(tmp) / "upload"
        upload_dir.mkdir()

        for item in COPY_ITEMS:
            src = ROOT / item
            if src.exists():
                dest = upload_dir / item
                if src.is_dir():
                    shutil.copytree(src, dest)
                else:
                    shutil.copy2(src, dest)

        (upload_dir / "README.md").write_text(
            README_FRONTMATTER + README_BODY,
            encoding="utf-8",
            newline="\n",
        )

        api.upload_folder(
            repo_id=SPACE_ID,
            folder_path=upload_dir,
            token=token,
            repo_type="space",
            delete_patterns=["*"],
            commit_message="Deploy Desk Review to Hugging Face Spaces",
        )

        gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if gemini_key:
            api.add_space_secret(
                repo_id=SPACE_ID,
                key="GEMINI_API_KEY",
                value=gemini_key,
                token=token,
            )
            api.add_space_variable(
                repo_id=SPACE_ID,
                key="RATE_LIMIT_DEFAULT",
                value="3 per minute",
                token=token,
            )
            api.add_space_variable(
                repo_id=SPACE_ID,
                key="GEMINI_MODEL",
                value=os.environ.get("GEMINI_MODEL", "gemini-3.6-flash"),
                token=token,
            )
            print(
                "Configured GEMINI_API_KEY secret, GEMINI_MODEL, "
                "and RATE_LIMIT_DEFAULT variables."
            )

    print(f"Deployed: https://huggingface.co/spaces/{SPACE_ID}")
    print("Next: Space Settings -> Hardware -> CPU basic (API-only; no GPU needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
