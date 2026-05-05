#!/usr/bin/env python3
"""Fetch lightweight Hugging Face metadata for candidate public sources.

This script is intentionally dependency-free so the project can bootstrap from a
nearly empty repository. It collects high-level repository metadata and tries to
query refs when available.
"""

from __future__ import annotations

import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


REPOS = [
    {
        "source_name": "OLMo",
        "repo_id": "allenai/OLMo-7B-0424",
        "purpose": "公开训练曲线主数据源",
    },
    {
        "source_name": "Pythia",
        "repo_id": "EleutherAI/pythia-1b-deduped",
        "purpose": "跨尺度受控实验主数据源",
    },
    {
        "source_name": "LLM360",
        "repo_id": "LLM360/Amber",
        "purpose": "公开全过程日志补充源",
    },
    {
        "source_name": "TinyLlama",
        "repo_id": "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
        "purpose": "小规模长训练补充",
    },
]


def build_ssl_context() -> ssl.SSLContext:
    """Prefer certifi when available to avoid local certificate issues."""
    try:
        import certifi  # type: ignore

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        return ssl.create_default_context()


SSL_CONTEXT = build_ssl_context()


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "DL-final-bootstrap/0.1",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=20, context=SSL_CONTEXT) as response:
        return json.loads(response.read().decode("utf-8"))


def safe_fetch(url: str) -> dict:
    try:
        return {"ok": True, "data": fetch_json(url)}
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "error": f"HTTP {exc.code}",
            "url": url,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "url": url,
        }


def summarize_model_info(model_info: dict) -> dict:
    siblings = model_info.get("siblings") or []
    tags = model_info.get("tags") or []
    return {
        "id": model_info.get("id"),
        "private": model_info.get("private"),
        "pipeline_tag": model_info.get("pipeline_tag"),
        "downloads": model_info.get("downloads"),
        "likes": model_info.get("likes"),
        "last_modified": model_info.get("lastModified"),
        "tags_sample": tags[:10],
        "sibling_count": len(siblings),
        "checkpoint_like_files": [
            item.get("rfilename")
            for item in siblings
            if "step" in item.get("rfilename", "").lower()
            or "checkpoint" in item.get("rfilename", "").lower()
        ][:20],
    }


def collect_repo(repo: dict) -> dict:
    repo_id = repo["repo_id"]
    encoded = urllib.parse.quote(repo_id, safe="")
    model_url = f"https://huggingface.co/api/models/{encoded}"
    refs_url = f"https://huggingface.co/api/models/{encoded}/refs"

    model_result = safe_fetch(model_url)
    refs_result = safe_fetch(refs_url)

    payload = {
        "source_name": repo["source_name"],
        "repo_id": repo_id,
        "purpose": repo["purpose"],
        "model_api_ok": model_result["ok"],
        "refs_api_ok": refs_result["ok"],
        "fetched_at": int(time.time()),
    }

    if model_result["ok"]:
        payload["model_info"] = summarize_model_info(model_result["data"])
    else:
        payload["model_api_error"] = model_result["error"]

    if refs_result["ok"]:
        data = refs_result["data"]
        payload["refs_summary"] = {
            "branch_count": len(data.get("branches") or []),
            "convert_count": len(data.get("converts") or []),
            "tag_count": len(data.get("tags") or []),
            "branch_sample": [
                branch.get("name") for branch in (data.get("branches") or [])[:20]
            ],
        }
    else:
        payload["refs_api_error"] = refs_result["error"]

    return payload


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "hf_source_metadata.json"

    results = [collect_repo(repo) for repo in REPOS]
    output_path.write_text(
        json.dumps(
            {
                "generated_by": "scripts/fetch_hf_source_metadata.py",
                "repo_count": len(results),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Wrote metadata to {output_path}")
    for item in results:
        print(
            f"- {item['source_name']}: "
            f"model_api_ok={item['model_api_ok']}, refs_api_ok={item['refs_api_ok']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
