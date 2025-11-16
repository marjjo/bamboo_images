from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os

# ==============================
# Config
# ==============================
OWNER = "marjjo"
REPO = "bamboo_images"
BRANCH = "main"
BASE_PATH = "images"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

app = Flask(__name__)
CORS(app)

# ==============================
# Folder → Canonical Tag Mapping
# ==============================
# Top-level folders that act as "groups"
TAG_GROUPS = {
    "geometry",
    "species",
    "joinery",
    "assembly",
    "bending",
    "form",
    "environment",
    "lighting",
    "style",
    "collection",   # optional legacy
    "material"      # optional legacy
}

# Optional mapping for some legacy/simple names to canonical tags
FOLDER_CANON = {
    # legacy / simple folders → canonical tag
    "bamboo": "material.bamboo",
    "precedents": "collection.precedent",

    # group folders map to themselves (group name)
    "geometry": "geometry",
    "species": "species",
    "joinery": "joinery",
    "assembly": "assembly",
    "bending": "bending",
    "form": "form",
    "environment": "environment",
    "lighting": "lighting",
    "style": "style",
    "collection": "collection",
    "material": "material",
}


def canonicalize_parts(parts):
    """
    Turn folder path segments into canonical tags.

    Example:
      images/geometry/hypar/file.jpg
      → parts = ["geometry", "hypar"]
      → tags = ["geometry", "geometry.hypar"]

      images/species/petung/...
      → ["species", "species.petung"]
    """
    canon = []
    for i, p in enumerate(parts):
        p = (p or "").strip()
        if not p:
            continue

        # If it's already canonical like "geometry.hypar", keep as is.
        if "." in p:
            canon.append(p)
            continue

        parent = parts[i - 1] if i > 0 else None

        # If this folder has a parent that is a tag group, combine as "group.value"
        if parent and parent in TAG_GROUPS:
            canon.append(f"{parent}.{p}")
            continue

        # If this folder name itself has a mapping (e.g. "bamboo" -> "material.bamboo")
        if p in FOLDER_CANON:
            canon.append(FOLDER_CANON[p])
            continue

        # Fallback: just use the raw folder name as a tag
        canon.append(p)

    return canon


# ==============================
# GitHub helpers
# ==============================
def gh_list(path: str):
    """List files/folders at a GitHub path using the Contents API."""
    base = f"https://api.github.com/repos/{OWNER}/{REPO}"
    suffix = f"/contents/{path}" if path else "/contents"
    url = f"{base}{suffix}?ref={BRANCH}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


def walk_images(path: str):
    """
    Recursively walk BASE_PATH in the GitHub repo and yield image items with tags.

    Each item:
      {
        "id": "...",      # short SHA
        "url": "https://raw.githubusercontent.com/...",
        "title": "filename.jpg",
        "tags": ["geometry.hypar", "environment.resort", ...]
      }
    """
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
                    rel = it["path"][len(BASE_PATH):].strip("/")  # e.g. "geometry/hypar/file.jpg"
                    raw_parts = rel.split("/")[:-1]               # folder names only
                    tags = canonicalize_parts(raw_parts)
                    yield {
                        "id": it["sha"][:8],
                        "url": it["download_url"],
                        "title": it["name"],
                        "tags": [t for t in tags if t],
                    }


def list_all_images(limit: int = 8, any_tags=None, q: str = ""):
    """
    Return up to `limit` images.

    - any_tags: list of canonical tag strings; image must match AT LEAST ONE of them (OR logic).
    - q: optional text filter that searches in title and tag strings.
    """
    any_set = set(any_tags or [])
    q = (q or "").lower().strip()

    results = []
    for item in walk_images(BASE_PATH):
        tagset = set(item.get("tags", []))

        # OR tag filter: if tags requested, must share at least one
        if any_set and not (tagset & any_set):
            continue

        # q filter: substring in title or tags
        if q:
            hay = (item.get("title", "") + " " + " ".join(tagset)).lower()
            if q not in hay:
                continue

        results.append(item)
        if len(results) >= limit:
            break

    return results


# ==============================
# Routes
# ==============================
@app.get("/")
def home():
    return "Bamboo Image API – returns images + canonical tags for your GPT."


@app.get("/images")
def list_images():
    """
    GET /images?tags=geometry.hypar,environment.forest&q=pavilion&limit=6

    - tags: comma-separated list of canonical tags; image must match ANY of them (OR).
    - q: optional keyword filter (matches in filename or tag strings).
    - limit: max number of images to return (default 8).
    """
    tag_str = request.args.get("tags", "").strip()
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 8))

    any_tags = [t.strip() for t in tag_str.split(",") if t.strip()]
    items = list_all_images(limit=limit, any_tags=any_tags, q=q)
    return jsonify({"count": len(items), "items": items})


@app.get("/tags")
def list_tags():
    """
    GET /tags

    Returns all discovered canonical tags from the image library.
    Useful for your GPT to learn what tags exist.
    """
    images = list_all_images(limit=10_000)
    tagset = set()
    for img in images:
        for t in img.get("tags", []):
            tagset.add(t)

    return jsonify({"count": len(tagset), "tags": sorted(tagset)})


@app.get("/health")
def health():
    """
    Simple health check:
    - tests access to the GitHub repo path
    """
    gh_ok, gh_msg = True, "ok"
    try:
        _ = gh_list(BASE_PATH)
    except Exception as e:
        gh_ok, gh_msg = False, str(e)

    status = "ok" if gh_ok else "degraded"
    return jsonify({
        "status": status,
        "github": {"ok": gh_ok, "detail": gh_msg},
    })


@app.get("/openapi.yaml")
def openapi_spec():
    """
    Serve the OpenAPI spec so your Custom GPT Actions can discover this API.
    """
    path = os.path.join(os.path.dirname(__file__), "openapi.yaml")
    if not os.path.exists(path):
        return jsonify({"error": "openapi.yaml not found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        return app.response_class(f.read(), mimetype="text/yaml")


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
