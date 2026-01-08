from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os, json, hashlib, random, base64
import requests
import mimetypes
import sys
import tempfile
from urllib.parse import urlparse
from typing import Optional, List
from openai import OpenAI

# ==============================
# Basic Config
# ==============================
OWNER = "marjjo"
REPO = "bamboo_images"
BRANCH = "main"
BASE_PATH = "images"

BASE_DIR = os.path.dirname(__file__)
GENERATED_DIR = os.path.join(BASE_DIR, "generated")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH", "annotations.json")

PUBLIC_BASE_URL = os.getenv(
    "PUBLIC_BASE_URL",
    "https://bamboo-images.onrender.com"
).rstrip("/")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_IMAGE_MODEL = "gpt-image-1"
MAX_REFERENCE_IMAGES = 4

ALLOWED_REF_HOSTS = set(
    h.strip().lower()
    for h in (
        os.getenv(
            "ALLOWED_REF_HOSTS",
            "raw.githubusercontent.com,githubusercontent.com,bamboo-images.onrender.com",
        ).split(",")
    )
    if h.strip()
)

MAX_IMAGE_BYTES = 50 * 1024 * 1024

app = Flask(__name__)
CORS(app)

_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ==============================
# Load annotations.json
# ==============================
def load_annotations(path: str = ANNOTATIONS_PATH):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        norm = {}
        for fname, meta in data.items():
            tags_raw = meta.get("tags", []) if isinstance(meta, dict) else meta
            if not isinstance(tags_raw, list):
                tags_raw = [tags_raw]
            cleaned = [str(t).strip() for t in tags_raw if str(t).strip()]
            if cleaned:
                norm[fname] = cleaned
        return norm
    except Exception:
        return {}

ANNOTATIONS = load_annotations()


# ==============================
# GitHub helpers
# ==============================
def gh_list(path: str):
    base = f"https://api.github.com/repos/{OWNER}/{REPO}"
    suffix = f"/contents/{path}" if path else "/contents"
    url = f"{base}{suffix}?ref={BRANCH}"

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


# ==============================
# Tag helpers
# ==============================
def canonicalize_parts(parts):
    tags = []
    if not parts:
        return tags
    category = parts[0].strip()
    for p in parts[1:]:
        if p:
            tags.append(f"{category}.{p}")
    return tags


def walk_images(path: str):
    stack = [path]
    while stack:
        cur = stack.pop()
        items = gh_list(cur)

        if isinstance(items, list):
            for it in items:
                if it["type"] == "dir":
                    stack.append(it["path"])
                elif it["type"] == "file" and it["name"].lower().endswith(
                    (".jpg", ".jpeg", ".png", ".gif", ".webp")
                ):
                    rel = it["path"][len(BASE_PATH):].strip("/")
                    parts = rel.split("/")[:-1]
                    folder_tags = canonicalize_parts(parts)

                    fname = it["name"]
                    extra_tags = ANNOTATIONS.get(fname, [])
                    all_tags = sorted(set(folder_tags) | set(extra_tags))

                    short_id = it["sha"][:8]

                    yield {
                        "id": short_id,
                        "source_url": it["download_url"],
                        "url": f"{PUBLIC_BASE_URL}/img/{short_id}.jpg",  # 🔑 EXTENSION
                        "title": fname,
                        "tags": all_tags,
                    }


def list_all_images(limit: int = 1000):
    out = []
    for img in walk_images(BASE_PATH):
        out.append(img)
        if len(out) >= limit:
            break
    return out


# ==============================
# Image proxy (FIXED)
# ==============================
@app.get("/img/<image_id>")
@app.get("/img/<image_id>.jpg")  # 🔑 extension support
def proxy_image(image_id):
    try:
        images = list_all_images(limit=10_000)
        img = next((i for i in images if i["id"] == image_id), None)
        if not img:
            return jsonify({"error": "image not found"}), 404

        r = requests.get(img["source_url"], stream=True, timeout=15)
        r.raise_for_status()

        resp = Response(
            r.iter_content(chunk_size=8192),
            content_type=r.headers.get("content-type", "image/jpeg"),
        )

        # 🔑 INLINE RENDERING HEADERS
        resp.headers["Content-Disposition"] = "inline"
        resp.headers["Cache-Control"] = "public, max-age=86400"

        return resp

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
