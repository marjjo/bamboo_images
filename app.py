from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, hashlib, random, base64
import requests

# NEW: OpenAI client (only used for image generation via gpt-image-1-mini)
from openai import OpenAI  # NEW

# ==============================
# Basic Config
# ==============================
OWNER = "marjjo"
REPO = "bamboo_images"
BRANCH = "main"
BASE_PATH = "images"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH", "annotations.json")

# NEW: OpenAI config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # NEW
OPENAI_IMAGE_MODEL = "gpt-image-1-mini"       # NEW: force mini, no gpt-image-1 anywhere
MAX_REFERENCE_IMAGES = 4                      # NEW: keep references bounded for stability

app = Flask(__name__)
CORS(app)

# NEW: initialize OpenAI client once
_openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None  # NEW


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
# NEW: Image generation (ONLY gpt-image-1-mini)
# ==============================
def _extract_first_image_b64(response_obj) -> str | None:
    """
    Tries to pull base64 image output from the Responses API result.
    The docs indicate image_generation_call results include a base64-encoded image. :contentReference[oaicite:3]{index=3}
    """
    if not response_obj:
        return None

    output = getattr(response_obj, "output", None) or response_obj.get("output", [])
    if not output:
        return None

    # Normalize output items into dicts
    norm_items = []
    for item in output:
        if isinstance(item, dict):
            norm_items.append(item)
        else:
            # OpenAI SDK objects often have model_dump()
            md = getattr(item, "model_dump", None)
            norm_items.append(md() if callable(md) else {})

    # Look for image generation call
    for item in norm_items:
        t = item.get("type")
        if t == "image_generation_call":
            # most common patterns observed in docs/snippets:
            # - item["result"] is base64 string
            # - item["result"]["b64_json"]
            res = item.get("result")
            if isinstance(res, str) and res.strip():
                return res.strip()
            if isinstance(res, dict):
                b64 = res.get("b64_json") or res.get("image_base64") or res.get("b64")
                if isinstance(b64, str) and b64.strip():
                    return b64.strip()

    # Fallback: sometimes output contains direct image items
    for item in norm_items:
        if item.get("type") in ("output_image", "image"):
            b64 = item.get("b64_json") or item.get("image_base64") or item.get("b64")
            if isinstance(b64, str) and b64.strip():
                return b64.strip()

    return None


def generate_with_visual_references(prompt: str, reference_urls: list[str] | None = None, size: str | None = None):
    """
    Generates an image using ONLY `gpt-image-1-mini`, optionally with image URLs as references.
    Uses Responses API image_generation tool which supports optional image inputs. :contentReference[oaicite:4]{index=4}
    """
    if not _openai_client:
        raise RuntimeError("OPENAI_API_KEY is missing on the server environment.")

    reference_urls = reference_urls or []
    reference_urls = [u for u in reference_urls if isinstance(u, str) and u.strip()][:MAX_REFERENCE_IMAGES]

    # We put size into text to avoid relying on undocumented tool args shape.
    # (You can still pass "size" in JSON to influence the instruction.)
    size_line = f"Output size preference: {size}." if size else "Output size preference: 1024x1024."

    content = [
        {"type": "input_text", "text": f"{prompt}\n{size_line}\nUse the provided reference images as visual inspiration and stay structurally plausible."}
    ]

    # Add each reference image as an input_image
    for url in reference_urls:
        content.append({"type": "input_image", "image_url": url})

    resp = _openai_client.responses.create(
        model=OPENAI_IMAGE_MODEL,                # NEW: always gpt-image-1-mini
        input=[{"role": "user", "content": content}],
        tools=[{"type": "image_generation"}],    # NEW: tool-based image generation (supports image inputs)
    )

    b64 = _extract_first_image_b64(resp)
    if not b64:
        raise RuntimeError("No image returned from OpenAI response.")

    return b64


# ==============================
# Routes
# ==============================
@app.get("/")
def home():
    return "Hi marjo! Welcome back to Bamboo Image API (images + tags + gpt-image-1-mini generation)"


@app.get("/images")
def list_images():
    tags_param = request.args.get("tags", "").strip()
    any_param  = request.args.get("any", "").strip()
    q          = request.args.get("q", "").strip().lower()
    limit      = int(request.args.get("limit", 8))

    required_tags = {t.strip() for t in tags_param.split(",") if t.strip()}
    any_tags      = {t.strip() for t in any_param.split(",") if t.strip()}

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

    # NEW: OpenAI health (key presence only; don't make paid calls here)
    openai_ok = bool(OPENAI_API_KEY)

    status = "ok" if (gh_ok and openai_ok) else "degraded"

    return jsonify({
        "status": status,
        "github": {"ok": gh_ok, "detail": gh_msg},
        "annotations": {"ok": ann_ok, "path": ANNOTATIONS_PATH},
        "openai": {"ok": openai_ok, "model": OPENAI_IMAGE_MODEL},
    })


# NEW: main generation endpoint for your Custom GPT Action
@app.post("/generate")
def generate():
    """
    JSON body:
      {
        "prompt": "Render a bamboo hypar pavilion...",
        "reference_urls": ["https://.../img1.jpg", "https://.../img2.jpg"],
        "size": "1024x1024"
      }

    Returns:
      {
        "model": "gpt-image-1-mini",
        "b64_png": "<base64...>",
        "data_url": "data:image/png;base64,..."
      }
    """
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "prompt is required"}), 400

    reference_urls = payload.get("reference_urls") or payload.get("ref_urls") or []
    size = (payload.get("size") or "").strip() or None

    try:
        b64 = generate_with_visual_references(prompt=prompt, reference_urls=reference_urls, size=size)
        return jsonify({
            "model": OPENAI_IMAGE_MODEL,
            "b64_png": b64,
            "data_url": f"data:image/png;base64,{b64}",
            "reference_count": min(len(reference_urls or []), MAX_REFERENCE_IMAGES),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# Entry Point (local dev)
# ==============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
