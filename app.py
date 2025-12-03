from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, hashlib, random, base64
import requests

# NEW: for safe temp files and URL parsing
import tempfile  # NEW
from urllib.parse import urlparse  # NEW
from typing import Optional, List  # NEW

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

# CHANGED: model config -> gpt-image-1 only
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_IMAGE_MODEL = "gpt-image-1"  # CHANGED (was gpt-image-1-mini)
MAX_REFERENCE_IMAGES = 4            # unchanged

# NEW: allowlist to prevent SSRF; default allows GitHub raw + user-images
ALLOWED_REF_HOSTS = set(
    h.strip().lower()
    for h in (os.getenv("ALLOWED_REF_HOSTS", "raw.githubusercontent.com,githubusercontent.com").split(","))
    if h.strip()
)  # NEW

MAX_IMAGE_BYTES = 50 * 1024 * 1024  # NEW: 50MB per image (matches gpt-image-1 limits)

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
                    parts = rel.split("/")[:-1]
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

        options.append({
            "tag": tag,
            "image": {
                "id": chosen_img["id"],
                "url": chosen_img["url"],
                "title": chosen_img["title"],
                "tags": chosen_img.get("tags", []),
            }
        })

        if len(options) >= limit:
            break

    return options


# ==============================
# CHANGED: Image generation (gpt-image-1 only)
# ==============================
# REMOVED: _extract_first_image_b64 + Responses API tool parsing

def _validate_ref_url(url: str) -> None:  # NEW
    u = urlparse(url)
    if u.scheme != "https":
        raise ValueError("Reference URL must be https.")
    host = (u.hostname or "").lower()
    if not host:
        raise ValueError("Reference URL has no hostname.")
    if host not in ALLOWED_REF_HOSTS:
        raise ValueError(f"Reference host not allowed: {host}")


def _download_to_temp_image(url: str):  # NEW
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

    # Pick a safe suffix
    if "png" in ctype:
        suffix = ".png"
    elif "jpeg" in ctype or "jpg" in ctype:
        suffix = ".jpg"
    elif "webp" in ctype:
        suffix = ".webp"
    else:
        suffix = ".img"

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)  # we delete manually
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

    # Normalise size (still allow auto)
    size = (size or "1024x1024").strip()
    if size not in ("1024x1024", "1024x1536", "1536x1024", "auto"):
        size = "1024x1024"

    # HARD OVERRIDE: always low + jpeg
    quality = "low"
    fmt = "jpeg"

    print("[generate_with_visual_references] start", size, quality, fmt, flush=True)
    result = _openai_client.images.generate(
        model=OPENAI_IMAGE_MODEL,
        prompt=prompt,
        size=size,
        quality=quality,
        output_format=fmt,
    )
    print("[generate_with_visual_references] done", flush=True)

    b64 = result.data[0].b64_json
    return b64, fmt



# ==============================
# Routes
# ==============================
@app.get("/")
def home():
    return "Hi marjo! Welcome back to Bamboo Image API (images + tags + gpt-image-1 generation)"  # CHANGED


@app.get("/images")
def list_images():
    tags_param = request.args.get("tags", "").strip()
    any_param = request.args.get("any", "").strip()
    q = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 8))

    required_tags = {t.strip() for t in tags_param.split(",") if t.strip()}
    any_tags = {t.strip() for t in any_param.split(",") if t.strip()}

    images = list_all_images(limit=10_000)
    results = []

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

        results.append(img)
        if len(results) >= limit:
            break

    return jsonify({"count": len(results), "items": results})


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

    return jsonify({
        "status": status,
        "github": {"ok": gh_ok, "detail": gh_msg},
        "annotations": {"ok": ann_ok, "path": ANNOTATIONS_PATH},
        "openai": {"ok": openai_ok, "model": OPENAI_IMAGE_MODEL},  # CHANGED
        "refs": {
            "max_reference_images": MAX_REFERENCE_IMAGES,
            "allowed_ref_hosts": sorted(ALLOWED_REF_HOSTS),
        }
    })


@app.get("/generated/<path:filename>")
def serve_generated(filename):
    """
    Serve generated images from the 'generated' folder.
    """
    full_path = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(full_path):
        return jsonify({"error": "file not found"}), 404

    return send_from_directory(GENERATED_DIR, filename)


@app.post("/generate")
def generate():
    """
    JSON body:
      {
        "prompt": "Render a conceptual bamboo hypar pavilion ...",
        "reference_urls": ["https://raw.githubusercontent.com/.../a.jpg", "..."],
        "size": "1024x1024",
        "quality": "medium",
        "output_format": "png"
      }

    Returns a small JSON with a public URL to the generated image:
      {
        "model": "gpt-image-1",
        "image_url": "https://bamboo-images.onrender.com/generated/....jpeg",
        "output_format": "jpeg",
        "reference_count": 0
      }
    """
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    reference_urls = payload.get("reference_urls") or payload.get("ref_urls") or []
    size = (payload.get("size") or "").strip() or None
    quality = (payload.get("quality") or "").strip() or None
    output_format = (payload.get("output_format") or "").strip() or None

    try:
        # 1) Call OpenAI (returns base64 + format)
        b64, fmt = generate_with_visual_references(
            prompt=prompt,
            reference_urls=reference_urls,
            size=size,
            quality=quality,
            output_format=output_format,
        )

        # 2) Decode base64 to bytes
        try:
            img_bytes = base64.b64decode(b64)
        except Exception as e:
            return jsonify({"error": f"failed to decode image base64: {e}"}), 500

        # 3) Ensure output folder exists
        os.makedirs(GENERATED_DIR, exist_ok=True)

        # 4) Build a semi-unique filename
        fmt_lower = (fmt or "jpeg").lower()
        if fmt_lower not in ("png", "jpeg", "jpg", "webp"):
            fmt_lower = "jpeg"

        digest = hashlib.md5(
            (prompt + str(len(img_bytes))).encode("utf-8")
        ).hexdigest()[:8]
        filename = f"img_{digest}.{fmt_lower}"
        filepath = os.path.join(GENERATED_DIR, filename)

        # 5) Save file
        with open(filepath, "wb") as f:
            f.write(img_bytes)

        # 6) Build public URL
        base_url = request.url_root.rstrip("/")  # e.g. https://bamboo-images.onrender.com
        image_url = f"{base_url}/generated/{filename}"

        return jsonify({
            "model": OPENAI_IMAGE_MODEL,
            "image_url": image_url,
            "output_format": fmt_lower,
            "reference_count": min(len(reference_urls or []), MAX_REFERENCE_IMAGES),
        }), 200

    except Exception as e:
        # IMPORTANT: do NOT reference `filename` here
        return jsonify({"error": str(e)}), 500


# ==============================
# Entry Point (local dev)
# ==============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
