from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os, json
import hashlib, random

# ==============================
# Basic Config
# ==============================
OWNER = "marjjo"
REPO = "bamboo_images"       # your GitHub repo name
BRANCH = "main"
BASE_PATH = "images"         # top-level folder for images in the repo

# Optional: GitHub token for higher rate limits (set as env var on Render)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Path to annotations file (extra tags per image)
ANNOTATIONS_PATH = os.getenv("ANNOTATIONS_PATH", "annotations.json")

app = Flask(__name__)
CORS(app)

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
            # 1) If it's a dict, expect {"tags": [...]}
            if isinstance(meta, dict):
                tags_raw = meta.get("tags", [])
            else:
                tags_raw = meta

            # 2) Normalize to list
            if not isinstance(tags_raw, list):
                tags_raw = [tags_raw]

            # 3) Clean strings and drop empty values
            cleaned = [
                str(t).strip()
                for t in tags_raw
                if str(t).strip()
            ]

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

    Example:
      rel path: geometry/hypar/pavilion_01.jpg
      parts   : ["geometry", "hypar"]
      tags    : ["geometry.hypar"]

    For deeper nests like environment/resort/beach:
      parts   : ["environment", "resort", "beach"]
      tags    : ["environment.resort", "environment.beach"]
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
                    # Relative path under BASE_PATH (e.g. "geometry/hypar/file.jpg")
                    rel = it["path"][len(BASE_PATH):].strip("/")
                    parts = rel.split("/")[:-1]  # folder parts only
                    folder_tags = canonicalize_parts(parts)

                    fname = it["name"]
                    extra_tags = ANNOTATIONS.get(fname, [])

                    # Merge and deduplicate
                    all_tags = sorted(set(folder_tags) | set(extra_tags))

                    yield {
                        "id": it["sha"][:8],
                        "url": it["download_url"],
                        "title": fname,
                        "tags": all_tags,
                    }


def list_all_images(limit: int = 1000):
    """
    Collect up to `limit` images from the repo with their tags.
    """
    results = []
    for item in walk_images(BASE_PATH):
        results.append(item)
        if len(results) >= limit:
            break
    return results


def build_tag_families(images):
    """
    Group tags by their family prefix before the first dot.
    e.g. "program.pavilion" -> family "program"
    Returns: {"program": {"program.pavilion", "program.bridge", ...}, ...}
    """
    families = {}
    for img in images:
        for t in img.get("tags", []):
            if "." in t:
                family, _ = t.split(".", 1)
            else:
                family = "misc"
            families.setdefault(family, set()).add(t)
    return families


def index_tag_images(images):
    """
    Build an index: tag -> list of images that have this tag.
    """
    tag_index = {}
    for img in images:
        for t in img.get("tags", []):
            tag_index.setdefault(t, []).append(img)
    return tag_index


def get_rng_for_user(user_key: str, family: str) -> random.Random:
    """
    Build a deterministic RNG based on user_key + family.
    Different users -> different sequences.
    Same user + same family -> same sequence.
    """
    base = f"{user_key}:{family}"
    h = hashlib.md5(base.encode("utf-8")).hexdigest()
    seed = int(h, 16) % (2**32)
    return random.Random(seed)


def pick_options_for_family(family: str, tag_index: dict, limit: int = 3, user_key: str = ""):
    """
    For a given family (e.g. 'program'), pick up to `limit` options.
    Each option = (tag + one representative image).
    Filenames will NOT repeat.

    If user_key is provided, the selection is randomized per user.
    """
    prefix = f"{family}."
    family_tags = [t for t in tag_index.keys() if t.startswith(prefix)]

    if not family_tags:
        return []

    rng = get_rng_for_user(user_key or "default", family)

    # Randomize order of tags for this user
    rng.shuffle(family_tags)

    used_filenames = set()
    options = []

    for tag in family_tags:
        imgs = tag_index[tag][:]  # copy list
        # Randomize images for this tag too
        rng.shuffle(imgs)

        chosen_img = None
        for img in imgs:
            fname = img.get("title") or img.get("filename") or ""
            if fname and fname not in used_filenames:
                chosen_img = img
                used_filenames.add(fname)
                break

        if chosen_img is None:
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
# Routes
# ==============================
@app.get("/")
def home():
    return "Hi marjo! Welcome back to Bamboo Image API (images + tags)"


@app.get("/images")
def list_images():
    """
    Query params:
      - tags: comma-separated, all MUST be present in image.tags
      - any : comma-separated, at least one MUST be present (optional)
      - q   : free text search in title or tags (optional)
      - limit: max number of images (default 8)
    """
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

        # Require all required_tags
        if required_tags and not required_tags.issubset(tagset):
            continue

        # Require at least one of any_tags (if provided)
        if any_tags and not (tagset & any_tags):
            continue

        # Text search in title or tags
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
    """
    Return all unique tags (folder-based + annotations) used in the library.
    """
    images = list_all_images(limit=10_000)
    all_tags = set()
    for img in images:
        for t in img.get("tags", []):
            all_tags.add(t)
    return jsonify({"count": len(all_tags), "tags": sorted(all_tags)})


@app.get("/options")
def list_family_options():
    """
    Suggest options for a tag family.

    Query params:
      - family: e.g. "program", "bamboo", "system", "environment"
      - limit : max number of options (default 3)
      - user  : arbitrary user identifier for per-user randomness
    """
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

    return jsonify({
        "family": family,
        "user": user_key or None,
        "count": len(options),
        "options": options
    })


@app.get("/health")
def health():
    """
    Simple health check: confirms GitHub access and annotations load.
    """
    gh_ok, gh_msg = True, "ok"
    try:
        _ = gh_list(BASE_PATH)
    except Exception as e:
        gh_ok, gh_msg = False, str(e)

    ann_ok = bool(ANNOTATIONS) or os.path.exists(ANNOTATIONS_PATH)

    status = "ok" if gh_ok else "degraded"

    return jsonify({
        "status": status,
        "github": {"ok": gh_ok, "detail": gh_msg},
        "annotations": {"ok": ann_ok, "path": ANNOTATIONS_PATH}
    })


# ==============================
# Entry Point (local dev)
# ==============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
