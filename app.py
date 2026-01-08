wait read the app.py first this is the current version

from flask import Flask, request, jsonify, send_file, Response  # CHANGED: add Response
from flask_cors import CORS
import os, json, hashlib, random, base64
import requests
import mimetypes
import sys

# NEW: for safe temp files and URL parsing
import tempfile
from urllib.parse import urlparse
from typing import Optional, List

# OpenAI client
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

# NEW: public base URL so returned image URLs are on your API domain (for inline rendering)
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://bamboo-images.onrender.com").rstrip("/")  # NEW

# Model config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_IMAGE_MODEL = "gpt-image-1"
MAX_REFERENCE_IMAGES = 4

# NEW: allowlist to prevent SSRF; include your API host so /img/<id> can be used as reference URLs too
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

MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 50MB per image

app = Flask(__name__)
CORS(app)

_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# ==============================
# Load annotations.json
# ==============================
def load_annotations(path: str = ANNOTATIONS_PATH):
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

        norm = {}
        for fname, meta in data.items():
            tags_raw = meta.get("tags", []) if isinstance(meta, dict) else meta
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
    """List files/folders at a GitHub path using the Contents API."""
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
    """
    Convert folder parts into canonical tag strings.
    For deeper nests: ["environment", "resort", "beach"]
    -> ["environment.resort", "environment.beach"]
    """
    tags = []
    if not parts:
        return tags

    category = parts[0].strip() if parts[0] else None
    if not category:
        return tags

    for p in parts[1:]:
        p = (p or "").strip()
        if p:
            tags.append(f"{category}.{p}")
    return tags


def walk_images(path: str):
    """
    Recursively walk the BASE_PATH in GitHub and yield image items
    with merged tags (folder-derived + annotations.json).

    IMPORTANT:
    - Keep GitHub URL as source_url (for proxy fetch + generation fallback).
    - Return public url as /img/<id> so ChatGPT can render inline reliably.
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
                    rel = it["path"][len(BASE_PATH) :].strip("/")
                    parts = rel.split("/")[:-1]
                    folder_tags = canonicalize_parts(parts)

                    fname = it["name"]
                    extra_tags = ANNOTATIONS.get(fname, [])
                    all_tags = sorted(set(folder_tags) | set(extra_tags))

                    short_id = it["sha"][:8]
                    yield {
                        "id": short_id,
                        "source_url": it["download_url"],  # NEW: keep GitHub as true source
                        "url": f"{PUBLIC_BASE_URL}/img/{short_id}",  # NEW: proxy URL for clients/ChatGPT
                        "title": fname,
                        "tags": all_tags,
                    }


def list_all_images(limit: int = 1000):
    results = []
    for item in walk_images(BASE_PATH):
        results.append(item)
        if len(results) >= limit:
            break
    return results


def index_tag_images(images):
    tag_index = {}
    for img in images:
        for t in img.get("tags", []):
            tag_index.setdefault(t, []).append(img)
    return tag_index


def get_rng_for_user(user_key: str, family: str) -> random.Random:
    base = f"{user_key}:{family}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()
    seed = int(h, 16) % (2**32)
    return random.Random(seed)


def pick_options_for_family(family: str, tag_index: dict, limit: int = 3, user_key: str = ""):
    prefix = f"{family}."
    family_tags = [t for t in tag_index.keys() if t.startswith(prefix)]
    if not family_tags:
        return []

    rng = get_rng_for_user(user_key or "default", family)
    rng.shuffle(family_tags)

    used_filenames = set()
    options = []

    for tag in family_tags:
        imgs = tag_index[tag][:]
        rng.shuffle(imgs)

        chosen_img = None
        for img in imgs:
            fname = img.get("title") or ""
            if fname and fname not in used_filenames:
                chosen_img = img
                used_filenames.add(fname)
                break

        if not chosen_img:
            continue

        options.append(
            {
                "tag": tag,
                "image": {
                    "id": chosen_img["id"],
                    "url": chosen_img["url"],  # will be proxied /img/<id>
                    "title": chosen_img["title"],
                    "tags": chosen_img.get("tags", []),
                },
            }
        )

        if len(options) >= limit:
            break

    return options


# ==============================
# Image generation (gpt-image-1 only)
# ==============================
def _validate_ref_url(url: str) -> None:
    u = urlparse(url)
    if u.scheme != "https":
        raise ValueError("Reference URL must be https.")
    host = (u.hostname or "").lower()
    if not host:
        raise ValueError("Reference URL has no hostname.")
    if host not in ALLOWED_REF_HOSTS:
        raise ValueError(f"Reference host not allowed: {host}")


def _download_to_temp_image(url: str):
    """
    Download an image URL to a temp file and return (file_obj, path, content_type).
    Enforces: https + host allowlist + <50MB + image/*.
    """
    _validate_ref_url(url)

    r = requests.get(url, stream=True, timeout=20)
    r.raise_for_status()

    ctype = (r.headers.get("content-type") or "").lower()
    if not ctype.startswith("image/"):
        raise ValueError(f"Reference is not an image (content-type={ctype}).")

    if "png" in ctype:
        suffix = ".png"
    elif "jpeg" in ctype or "jpg" in ctype:
        suffix = ".jpg"
    elif "webp" in ctype:
        suffix = ".webp"
    else:
        suffix = ".img"

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    bytes_written = 0
    try:
        for chunk in r.iter_content(chunk_size=1024 * 64):
            if not chunk:
                continue
            bytes_written += len(chunk)
            if bytes_written > MAX_IMAGE_BYTES:
                raise ValueError("Reference image too large (>50MB).")
            tf.write(chunk)
        tf.flush()
        tf.close()

        f = open(tf.name, "rb")
        return f, tf.name, ctype
    except Exception:
        try:
            tf.close()
        except Exception:
            pass
        try:
            os.unlink(tf.name)
        except Exception:
            pass
        raise


def generate_with_visual_references(
    prompt: str,
    reference_urls: Optional[List[str]] = None,
    size: Optional[str] = None,
    quality: Optional[str] = None,
    output_format: Optional[str] = None,
):
    if not _openai_client:
        raise RuntimeError("OPENAI_API_KEY is missing on the server environment.")

    size = (size or "1024x1024").strip()
    if size not in ("1024x1024", "1024x1536", "1536x1024", "auto"):
        size = "1024x1024"

    # HARD OVERRIDE: always low + jpeg
    quality = "low"
    fmt = "jpeg"

    files = []
    refs = (reference_urls or [])[:MAX_REFERENCE_IMAGES]

    try:
        for url in refs:
            try:
                f, _tmp_path, _ctype = _download_to_temp_image(url)
                files.append(f)
            except Exception as e:
                print(f"[generate_with_visual_references] skip ref {url}: {e}", flush=True)

        if files:
            result = _openai_client.images.edit(
                model=OPENAI_IMAGE_MODEL,
                image=files if len(files) > 1 else files[0],
                prompt=prompt,
                size=size,
                quality=quality,
                output_format=fmt,
                input_fidelity="high",
            )
        else:
            result = _openai_client.images.generate(
                model=OPENAI_IMAGE_MODEL,
                prompt=prompt,
                size=size,
                quality=quality,
                output_format=fmt,
            )

        b64 = result.data[0].b64_json
        return b64, fmt

    finally:
        for f in files:
            try:
                path = f.name
                f.close()
                os.unlink(path)
            except Exception:
                pass


# ==============================
# Routes
# ==============================
@app.get("/")
def home():
    return "Hi marjo! Welcome back to Bamboo Image API (images + tags + gpt-image-1 generation)"


@app.get("/images")
def list_images():
    tags_param = request.args.get("tags", "").strip()
    any_param = request.args.get("any", "").strip()
    q = request.args.get("q", "").strip().lower()

    limit = int(request.args.get("limit", 8))
    offset = int(request.args.get("offset", 0))

    if limit < 1:
        limit = 1
    if limit > 200:
        limit = 200
    if offset < 0:
        offset = 0

    required_tags = {t.strip() for t in tags_param.split(",") if t.strip()}
    any_tags = {t.strip() for t in any_param.split(",") if t.strip()}

    images = list_all_images(limit=10_000)

    # Deterministic ordering so offset works correctly (use source_url/id to avoid URL changes)
    images.sort(
        key=lambda x: (
            x.get("source_url", ""),
            x.get("title", ""),
            x.get("id", ""),
        )
    )

    results = []
    total_matches = 0

    for img in images:
        tagset = set(img.get("tags", []))

        if required_tags and not required_tags.issubset(tagset):
            continue
        if any_tags and not (tagset & any_tags):
            continue
        if q:
            hay = (img.get("title", "") + " " + " ".join(img.get("tags", []))).lower()
            if q not in hay:
                continue

        total_matches += 1
        if total_matches <= offset:
            continue

        results.append(img)
        if len(results) >= limit:
            break

    return jsonify(
        {
            "count": len(results),
            "total": total_matches,
            "offset": offset,
            "limit": limit,
            "items": results,
        }
    )


@app.get("/tags")
def list_tags():
    images = list_all_images(limit=10_000)
    all_tags = set()
    for img in images:
        for t in img.get("tags", []):
            all_tags.add(t)
    return jsonify({"count": len(all_tags), "tags": sorted(all_tags)})


@app.get("/options")
def list_family_options():
    family = request.args.get("family", "").strip()
    limit = int(request.args.get("limit", 3))
    user_key = request.args.get("user", "").strip()

    if not family:
        return jsonify({"error": "family query param is required"}), 400

    images = list_all_images(limit=10_000)
    tag_index = index_tag_images(images)

    options = pick_options_for_family(
        family=family,
        tag_index=tag_index,
        limit=limit,
        user_key=user_key,
    )

    return jsonify({"family": family, "user": user_key or None, "count": len(options), "options": options})


@app.get("/health")
def health():
    gh_ok, gh_msg = True, "ok"
    try:
        _ = gh_list(BASE_PATH)
    except Exception as e:
        gh_ok, gh_msg = False, str(e)

    ann_ok = bool(ANNOTATIONS) or os.path.exists(ANNOTATIONS_PATH)
    openai_ok = bool(OPENAI_API_KEY)

    status = "ok" if (gh_ok and openai_ok) else "degraded"

    return jsonify(
        {
            "status": status,
            "github": {"ok": gh_ok, "detail": gh_msg},
            "annotations": {"ok": ann_ok, "path": ANNOTATIONS_PATH},
            "openai": {"ok": openai_ok, "model": OPENAI_IMAGE_MODEL},
            "refs": {
                "max_reference_images": MAX_REFERENCE_IMAGES,
                "allowed_ref_hosts": sorted(ALLOWED_REF_HOSTS),
            },
        }
    )


@app.get("/generated/<path:filename>")
def serve_generated(filename):
    """Serve generated images from the 'generated' folder."""
    full_path = os.path.join(GENERATED_DIR, filename)

    if not os.path.exists(full_path):
        print(f"[serve_generated] file not found: {full_path}", file=sys.stderr, flush=True)
        return jsonify({"error": "file not found"}), 404

    mime_type, _ = mimetypes.guess_type(full_path)
    if mime_type is None:
        mime_type = "image/jpeg"

    return send_file(full_path, mimetype=mime_type)


@app.post("/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    reference_urls = payload.get("reference_urls") or payload.get("ref_urls") or []
    size = (payload.get("size") or "").strip() or None
    quality = (payload.get("quality") or "").strip() or None
    output_format = (payload.get("output_format") or "").strip() or None

    try:
        b64, fmt = generate_with_visual_references(
            prompt=prompt,
            reference_urls=reference_urls,
            size=size,
            quality=quality,
            output_format=output_format,
        )

        try:
            img_bytes = base64.b64decode(b64)
        except Exception as e:
            return jsonify({"error": f"failed to decode image base64: {e}"}), 500

        os.makedirs(GENERATED_DIR, exist_ok=True)

        fmt_lower = (fmt or "jpeg").lower()
        if fmt_lower not in ("png", "jpeg", "jpg", "webp"):
            fmt_lower = "jpeg"

        digest = hashlib.md5((prompt + str(len(img_bytes))).encode("utf-8")).hexdigest()[:8]
        filename = f"img_{digest}.{fmt_lower}"
        filepath = os.path.join(GENERATED_DIR, filename)

        with open(filepath, "wb") as f:
            f.write(img_bytes)

        # Use PUBLIC_BASE_URL instead of request.url_root (more reliable behind proxies)
        image_url = f"{PUBLIC_BASE_URL}/generated/{filename}"

        return (
            jsonify(
                {
                    "model": OPENAI_IMAGE_MODEL,
                    "image_url": image_url,
                    "output_format": fmt_lower,
                    "reference_count": min(len(reference_urls or []), MAX_REFERENCE_IMAGES),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/img/<image_id>")
def proxy_image(image_id):
    """
    Proxy GitHub-hosted images through this API domain
    so ChatGPT can render them inline.
    """
    try:
        images = list_all_images(limit=10_000)
        img = next((i for i in images if i["id"] == image_id), None)

        if not img:
            return jsonify({"error": "image not found"}), 404

        src = img.get("source_url")
        if not src:
            return jsonify({"error": "source_url missing"}), 500

        r = requests.get(src, stream=True, timeout=15)
        r.raise_for_status()

        content_type = r.headers.get("content-type", "image/jpeg")
        return Response(r.iter_content(chunk_size=8192), content_type=content_type)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# Entry Point (local dev)
# ==============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
