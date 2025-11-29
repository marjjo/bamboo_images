from __future__ import annotations

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import os
import json
import time
import math
import base64
import uuid
import hashlib
import random
from io import BytesIO
from typing import Any, Dict, List, Set, Tuple
from urllib.parse import urlparse

import requests

# ==============================
# Optional Pillow for moodboards
# ==============================
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

# ==============================
# OpenAI SDK
# ==============================
try:
    from openai import OpenAI
    OPENAI_SDK_OK = True
except Exception:
    OPENAI_SDK_OK = False


# ==============================
# Basic Config (your existing settings)
# ==============================
OWNER = "marjjo"
REPO = "bamboo_images"
BRANCH = "main"
BASE_PATH = "images"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH", "annotations.json")

# ==============================
# Caching (important for multi-step workflows)
# ==============================
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "180"))  # 3 minutes default

_CACHE: Dict[str, Any] = {
    "ts": 0.0,
    "images": None,       # List[dict]
    "tags": None,         # List[str]
    "tag_index": None,    # Dict[str, List[dict]]
}

# ==============================
# Image generation config
# ==============================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DEFAULT_IMAGE_MODEL = os.getenv("DEFAULT_IMAGE_MODEL", "gpt-image-1-mini")
DEFAULT_EDIT_MODEL = os.getenv("DEFAULT_EDIT_MODEL", "gpt-image-1")

GENERATED_DIR = os.getenv("GENERATED_DIR", "/tmp/generated")
os.makedirs(GENERATED_DIR, exist_ok=True)

# Allow fetching only from GitHub raw URLs to avoid SSRF
ALLOWED_REF_HOSTS = {
    "raw.githubusercontent.com",
    "github.com",
}

app = Flask(__name__)
CORS(app)


# ==============================
# Load annotations.json
# ==============================
def load_annotations(path: str = ANNOTATIONS_PATH) -> Dict[str, List[str]]:
    """
    Load extra tags for images from annotations.json.

    Accepted formats:
    1) "file.jpg": ["tag.one", "tag.two"]
    2) "file.jpg": {"tags": ["tag.one", "tag.two"]}
    """
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        norm: Dict[str, List[str]] = {}
        for fname, meta in data.items():
            if isinstance(meta, dict):
                tags_raw = meta.get("tags", [])
            else:
                tags_raw = meta

            if not isinstance(tags_raw, list):
                tags_raw = [tags_raw]

            cleaned = [str(t).strip() for t in tags_raw if str(t).strip()]
            if cleaned:
                norm[fname] = cleaned

        return norm
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


ANNOTATIONS = load_annotations()


# ==============================
# GitHub helpers
# ==============================
def gh_list(path: str):
    """
    List files/folders at a GitHub path using the Contents API.
    """
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
# Tag helpers (folder to "family.value")
# ==============================
def canonicalize_parts(parts: List[str]) -> List[str]:
    """
    Convert folder parts into canonical tag strings.
    Example:
      parts ["program","pavilion"] -> ["program.pavilion"]
    Deeper nests under one family generate multiple tags:
      ["environment","resort","beach"] -> ["environment.resort","environment.beach"]
    """
    tags: List[str] = []
    if not parts:
        return tags

    category = parts[0].strip() if parts[0] else None
    if not category:
        return tags

    for p in parts[1:]:
        p = (p or "").strip()
        if not p:
            continue
        tags.append(f"{category}.{p}")
    return tags


def walk_images(path: str):
    """
    Recursively walk the BASE_PATH in GitHub and yield image items
    with merged tags (folder-derived + annotations.json).
    """
    stack = [path]
    while stack:
        cur = stack.pop()
        items = gh_list(cur)

        if isinstance(items, list):
            for it in items:
                if it["type"] == "dir":
                    stack.append(it["path"])
                elif (
                    it["type"] == "file"
                    and it["name"].lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))
                ):
                    rel = it["path"][len(BASE_PATH):].strip("/")
                    parts = rel.split("/")[:-1]  # folder parts only
                    folder_tags = canonicalize_parts(parts)

                    fname = it["name"]
                    extra_tags = ANNOTATIONS.get(fname, [])

                    all_tags = sorted(set(folder_tags) | set(extra_tags))

                    yield {
                        "id": it["sha"][:8],
                        "url": it["download_url"],
                        "title": fname,
                        "tags": all_tags,
                    }


def list_all_images(limit: int = 10_000) -> List[dict]:
    results: List[dict] = []
    for item in walk_images(BASE_PATH):
        results.append(item)
        if len(results) >= limit:
            break
    return results


def index_tag_images(images: List[dict]) -> Dict[str, List[dict]]:
    tag_index: Dict[str, List[dict]] = {}
    for img in images:
        for t in img.get("tags", []):
            tag_index.setdefault(t, []).append(img)
    return tag_index


# ==============================
# Cache refresh
# ==============================
def get_cached_library() -> Tuple[List[dict], List[str], Dict[str, List[dict]]]:
    now = time.time()
    fresh = (
        _CACHE["images"] is not None and
        _CACHE["tags"] is not None and
        _CACHE["tag_index"] is not None and
        (now - _CACHE["ts"] < CACHE_TTL_SECONDS)
    )
    if fresh:
        return _CACHE["images"], _CACHE["tags"], _CACHE["tag_index"]

    images = list_all_images(limit=10_000)
    tags_set: Set[str] = set()
    for img in images:
        for t in img.get("tags", []):
            tags_set.add(t)

    tags_list = sorted(tags_set)
    tag_index = index_tag_images(images)

    _CACHE["images"] = images
    _CACHE["tags"] = tags_list
    _CACHE["tag_index"] = tag_index
    _CACHE["ts"] = now

    return images, tags_list, tag_index


# ==============================
# Randomized options (family suggestions)
# ==============================
def get_rng_for_user(user_key: str, family: str) -> random.Random:
    base = f"{user_key}:{family}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()
    seed = int(h, 16) % (2**32)
    return random.Random(seed)


def pick_options_for_family(
    family: str,
    tag_index: Dict[str, List[dict]],
    limit: int = 3,
    user_key: str = "",
) -> List[dict]:
    prefix = f"{family}."
    family_tags = [t for t in tag_index.keys() if t.startswith(prefix)]
    if not family_tags:
        return []

    rng = get_rng_for_user(user_key or "default", family)
    rng.shuffle(family_tags)

    used_titles: Set[str] = set()
    options: List[dict] = []

    for tag in family_tags:
        imgs = tag_index[tag][:]
        rng.shuffle(imgs)

        chosen = None
        for img in imgs:
            title = img.get("title") or ""
            if title and title not in used_titles:
                chosen = img
                used_titles.add(title)
                break

        if not chosen:
            continue

        options.append({
            "tag": tag,
            "image": {
                "id": chosen["id"],
                "url": chosen["url"],
                "title": chosen["title"],
                "tags": chosen.get("tags", []),
            }
        })

        if len(options) >= limit:
            break

    return options


# ==============================
# Image helpers: serving + safe fetch + moodboard
# ==============================
def public_url_for(path: str) -> str:
    return request.host_url.rstrip("/") + path


def save_b64_image(b64_json: str, output_format: str = "png") -> str:
    if not PIL_OK:
        raise RuntimeError("Pillow not installed (pip install pillow).")

    fmt = (output_format or "png").lower().strip()
    if fmt not in {"png", "jpg", "jpeg", "webp"}:
        fmt = "png"

    raw = base64.b64decode(b64_json)
    img = Image.open(BytesIO(raw))

    ext = "jpg" if fmt == "jpeg" else fmt
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(GENERATED_DIR, filename)

    if ext in {"jpg", "jpeg"}:
        img = img.convert("RGB")
        img.save(filepath, format="JPEG", quality=92, optimize=True)
    elif ext == "webp":
        img.save(filepath, format="WEBP", quality=92, method=6)
    else:
        img.save(filepath, format="PNG")

    return filename


def validate_ref_url(url: str) -> None:
    u = urlparse(url)
    if u.scheme not in ("https",):
        raise ValueError("Only https URLs are allowed for references.")
    host = (u.hostname or "").lower()
    if host not in ALLOWED_REF_HOSTS:
        raise ValueError(f"Host not allowed for reference fetch: {host}")


def fetch_ref_bytes(url: str) -> bytes:
    validate_ref_url(url)
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return r.content


def make_moodboard(ref_urls: List[str], tile: int = 512, cols: int = 2) -> bytes:
    if not PIL_OK:
        raise RuntimeError("Pillow not installed (pip install pillow).")

    cols = max(1, int(cols))
    tile = max(64, int(tile))

    tiles: List[Image.Image] = []
    for url in ref_urls:
        b = fetch_ref_bytes(url)
        im = Image.open(BytesIO(b)).convert("RGB")
        im.thumbnail((tile, tile))

        canvas = Image.new("RGB", (tile, tile), (255, 255, 255))
        x = (tile - im.size[0]) // 2
        y = (tile - im.size[1]) // 2
        canvas.paste(im, (x, y))
        tiles.append(canvas)

    if not tiles:
        return b""

    rows = math.ceil(len(tiles) / cols)
    board = Image.new("RGB", (cols * tile, rows * tile), (255, 255, 255))

    for i, t in enumerate(tiles):
        r = i // cols
        c = i % cols
        board.paste(t, (c * tile, r * tile))

    out = BytesIO()
    board.save(out, format="PNG")
    return out.getvalue()


# ==============================
# OpenAI client
# ==============================
openai_client = OpenAI(api_key=OPENAI_API_KEY) if (OPENAI_SDK_OK and OPENAI_API_KEY) else None


# ==============================
# Routes: Library
# ==============================
@app.get("/")
def home():
    return "Hi marjo! Bamboo Image API (library + generation)"


@app.get("/images")
def list_images():
    tags_param = request.args.get("tags", "").strip()
    any_param = request.args.get("any", "").strip()
    q = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 8))

    required_tags = {t.strip() for t in tags_param.split(",") if t.strip()}
    any_tags = {t.strip() for t in any_param.split(",") if t.strip()}

    images, _, _ = get_cached_library()
    q_tokens = [tok for tok in q.replace(",", " ").split() if tok]

    results: List[dict] = []
    for img in images:
        tagset = set(img.get("tags", []))

        if required_tags and not required_tags.issubset(tagset):
            continue
        if any_tags and not (tagset & any_tags):
            continue

        if q_tokens:
            hay = (img.get("title", "") + " " + " ".join(img.get("tags", []))).lower()
            if any(tok not in hay for tok in q_tokens):
                continue

        results.append(img)
        if len(results) >= limit:
            break

    return jsonify({"count": len(results), "items": results})


@app.get("/tags")
def list_tags():
    _, tags, _ = get_cached_library()
    return jsonify({"count": len(tags), "tags": tags})


@app.get("/options")
def list_family_options():
    family = request.args.get("family", "").strip()
    limit = int(request.args.get("limit", 3))
    user_key = request.args.get("user", "").strip()

    if not family:
        return jsonify({"error": "family query param is required"}), 400

    _, _, tag_index = get_cached_library()
    options = pick_options_for_family(family=family, tag_index=tag_index, limit=limit, user_key=user_key)

    return jsonify({
        "family": family,
        "user": user_key or None,
        "count": len(options),
        "options": options,
    })


@app.get("/health")
def health():
    gh_ok, gh_msg = True, "ok"
    try:
        _ = gh_list(BASE_PATH)
    except Exception as e:
        gh_ok, gh_msg = False, str(e)

    ann_ok = bool(ANNOTATIONS) or os.path.exists(ANNOTATIONS_PATH)
    cache_age = (time.time() - _CACHE["ts"]) if _CACHE["ts"] else None
    openai_ok = bool(openai_client)

    return jsonify({
        "status": "ok" if gh_ok else "degraded",
        "github": {"ok": gh_ok, "detail": gh_msg},
        "annotations": {"ok": ann_ok, "path": ANNOTATIONS_PATH},
        "cache": {"ttl_seconds": CACHE_TTL_SECONDS, "age_seconds": cache_age},
        "openai": {"ok": openai_ok, "model_default": DEFAULT_IMAGE_MODEL},
        "pillow": {"ok": PIL_OK},
    })


# ==============================
# Routes: Moodboard (library → tiled PNG)
# ==============================
@app.post("/moodboard")
def moodboard():
    if not PIL_OK:
        return jsonify({"error": "Pillow not installed. pip install pillow"}), 503

    data = request.get_json(silent=True) or {}
    tile = int(data.get("tile", 512))
    cols = int(data.get("cols", 2))

    ref_urls = data.get("ref_urls")
    selected_refs: List[dict] = []

    if not ref_urls:
        tags_param = (data.get("tags") or "").strip()
        any_param = (data.get("any") or "").strip()
        q = (data.get("q") or "").strip().lower()
        limit = int(data.get("limit", 4))

        required_tags = {t.strip() for t in tags_param.split(",") if t.strip()}
        any_tags = {t.strip() for t in any_param.split(",") if t.strip()}
        q_tokens = [tok for tok in q.replace(",", " ").split() if tok]

        images, _, _ = get_cached_library()
        used_titles: Set[str] = set()
        for img in images:
            tagset = set(img.get("tags", []))
            if required_tags and not required_tags.issubset(tagset):
                continue
            if any_tags and not (tagset & any_tags):
                continue
            if q_tokens:
                hay = (img.get("title", "") + " " + " ".join(img.get("tags", []))).lower()
                if any(tok not in hay for tok in q_tokens):
                    continue

            title = img.get("title") or ""
            if title in used_titles:
                continue
            used_titles.add(title)

            selected_refs.append(img)
            if len(selected_refs) >= limit:
                break

        ref_urls = [r["url"] for r in selected_refs]

    if not isinstance(ref_urls, list) or not ref_urls:
        return jsonify({"error": "Provide ref_urls or (tags/any/q) that yields at least 1 image."}), 400

    ref_urls = ref_urls[:6]

    png_bytes = make_moodboard(ref_urls, tile=tile, cols=cols)
    b64 = base64.b64encode(png_bytes).decode("utf-8")

    filename = f"{uuid.uuid4().hex}.png"
    path = os.path.join(GENERATED_DIR, filename)
    with open(path, "wb") as f:
        f.write(png_bytes)

    return jsonify({
        "ref_count": len(ref_urls),
        "selected_refs": selected_refs,
        "moodboard_png_base64": b64,
        "moodboard_url": public_url_for(f"/generated/{filename}"),
    })


@app.get("/generated/<path:filename>")
def serve_generated(filename: str):
    return send_from_directory(GENERATED_DIR, filename, as_attachment=False)


# ==============================
# Routes: Image generation
# ==============================
@app.post("/generate")
def generate():
    if not OPENAI_SDK_OK:
        return jsonify({"error": "openai package not installed. pip install openai -U"}), 503
    if openai_client is None:
        return jsonify({"error": "OPENAI_API_KEY not set on server"}), 500
    if not PIL_OK:
        return jsonify({"error": "Pillow not installed. pip install pillow"}), 503

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    model = (data.get("model") or DEFAULT_IMAGE_MODEL).strip()
    size = (data.get("size") or "1024x1024").strip()
    quality = (data.get("quality") or "auto").strip()
    output_format = (data.get("output_format") or "png").strip().lower()
    background = (data.get("background") or "").strip() or None
    return_base64 = bool(data.get("return_base64", False))

    result = openai_client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
        output_format=output_format,
        background=background,
    )

    b64 = result.data[0].b64_json
    filename = save_b64_image(b64, output_format=output_format)
    url = public_url_for(f"/generated/{filename}")

    payload = {
        "model": model,
        "size": size,
        "quality": quality,
        "output_format": output_format,
        "background": background,
        "image_url": url,
    }
    if return_base64:
        payload["image_base64"] = b64
    return jsonify(payload)


@app.post("/generate_with_refs")
def generate_with_refs():
    """
    IMPORTANT: images.edits supports gpt-image-1 and dall-e-2 (NOT mini).
    Use this only if you want true reference-based conditioning via edits.
    """
    if not OPENAI_SDK_OK:
        return jsonify({"error": "openai package not installed. pip install openai -U"}), 503
    if openai_client is None:
        return jsonify({"error": "OPENAI_API_KEY not set on server"}), 500
    if not PIL_OK:
        return jsonify({"error": "Pillow not installed. pip install pillow"}), 503

    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    ref_urls = data.get("ref_urls") or []
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    if not isinstance(ref_urls, list) or len(ref_urls) == 0:
        return jsonify({"error": "ref_urls must be a non-empty list"}), 400

    model = (data.get("model") or DEFAULT_EDIT_MODEL).strip()
    if model == "gpt-image-1-mini":
        return jsonify({
            "error": "images.edits does not support gpt-image-1-mini. Use model=gpt-image-1 for refs, or use /generate for text-only."
        }), 400

    size = (data.get("size") or "1024x1024").strip()
    output_format = (data.get("output_format") or "png").strip().lower()
    background = (data.get("background") or "").strip() or None

    ref_urls = ref_urls[:6]
    files: List[tuple] = []
    try:
        for u in ref_urls:
            b = fetch_ref_bytes(u)
            files.append(("ref.png", b, "image/png"))
    except Exception as e:
        return jsonify({"error": f"failed to fetch reference images: {e}"}), 400

    result = openai_client.images.edit(
        model=model,
        prompt=prompt,
        image=files,
        size=size,
        background=background,
        output_format=output_format,
    )

    b64 = result.data[0].b64_json
    filename = save_b64_image(b64, output_format=output_format)
    url = public_url_for(f"/generated/{filename}")

    return jsonify({
        "model": model,
        "size": size,
        "output_format": output_format,
        "background": background,
        "ref_count": len(ref_urls),
        "image_url": url,
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
