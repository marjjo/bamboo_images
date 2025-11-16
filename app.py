from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import json

# ==============================
# Config
# ==============================
OWNER = "marjjo"
REPO = "bamboo_images"
BRANCH = "main"
BASE_PATH = "images"  # top-level folder in the repo where images live

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")          # optional, for higher rate limits
TAGS_PATH = os.getenv("TAGS_PATH", "tags.json")   # local tags definition file
ANNO_PATH = os.getenv("ANNO_PATH", "annotations.json")  # local annotations file

app = Flask(__name__)
CORS(app)


# ==============================
# Helpers: load JSON configs
# ==============================
def load_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return {}


def load_tags_schema():
    """
    tags.json format (example):

    {
      "geometry.hypar": {
        "group": "geometry",
        "label": "Hypar shell",
        "syn": ["hypar", "hyperbolic paraboloid"]
      },
      "species.petung": {
        "group": "species",
        "label": "Petung bamboo",
        "syn": ["petung"]
      }
    }
    """
    return load_json(TAGS_PATH)


def load_annotations():
    """
    annotations.json format (example):

    {
      "hypar_ibuku_01.jpg": {
        "tags": ["geometry.hypar", "environment.resort", "species.petung"],
        "title": "Hypar Bamboo Pavilion",
        "source": "Ibuku, Bali",
        "notes": "Hypar shell using Petung bamboo for resort context."
      }
    }
    """
    return load_json(ANNO_PATH)


TAGS_SCHEMA = load_tags_schema()
ANNOTATIONS = load_annotations()


# ==============================
# GitHub helpers
# ==============================
def gh_list(path):
    """List files/folders at a GitHub path using the Contents API."""
    base = f"https://api.github.com/repos/{OWNER}/{REPO}"
    suffix = f"/contents/{path}" if path else "/contents"
    url = f"{base}{suffix}?ref={BRANCH}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()


def folder_tags_from_relpath(rel_path: str):
    """
    Convert a path like "geometry/hypar/pavilion01.jpg"
    into a list of canonical folder-based tags, e.g. ["geometry.hypar"].

    Assumes top-level folder names are groups:
      geometry/, species/, joinery/, assembly/, bending/, form/,
      environment/, lighting/, style/
    """
    rel_path = rel_path.strip("/")

    if not rel_path:
        return []

    parts = rel_path.split("/")  # e.g. ["geometry", "hypar", "pavilion01.jpg"]
    if len(parts) < 2:
        return []

    group = parts[0]       # "geometry"
    value = parts[1]       # "hypar"
    return [f"{group}.{value}"]


def parse_filename_meta(filename: str):
    """
    Parse filenames of the form 'title_source_number.ext'
    into (title, source, index).

    Example:
      'hypar_ibuku_01.jpg' =>
         title  = 'Hypar'
         source = 'Ibuku'
         index  = '01'
    """
    name, _dot, ext = filename.partition(".")
    parts = name.split("_")

    title = None
    source = None
    index = None

    if len(parts) >= 3:
        title = parts[0].replace("-", " ").strip() or None
        source = parts[1].replace("-", " ").strip() or None
        index = parts[2].strip() or None
    elif len(parts) == 2:
        title = parts[0].replace("-", " ").strip() or None
        source = parts[1].replace("-", " ").strip() or None
    elif len(parts) == 1:
        title = parts[0].replace("-", " ").strip() or None

    # Capitalize nicely if present
    if title:
        title = title.title()
    if source:
        source = source.title()

    return title, source, index


def walk_images(path):
    """
    Walk all image files under BASE_PATH using GitHub Contents API.

    For each image:
      - derive folder-based tags (e.g. geometry.hypar)
      - merge annotation-based tags from annotations.json
      - parse filename into title/source if needed
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
                    # Relative path inside BASE_PATH
                    rel = it["path"][len(BASE_PATH):].strip("/")  # e.g. "geometry/hypar/img.jpg"
                    folder_tags = folder_tags_from_relpath(rel)

                    # Load annotation for this filename if present
                    anno = ANNOTATIONS.get(it["name"], {}) or ANNOTATIONS.get(it["path"], {})

                    extra_tags = anno.get("tags", [])
                    combined_tags = sorted(set(folder_tags) | set(extra_tags))

                    # Title/source from annotations or from filename
                    parsed_title, parsed_source, _idx = parse_filename_meta(it["name"])
                    title = anno.get("title") or parsed_title or it["name"]
                    source = anno.get("source") or parsed_source

                    # Optional notes
                    notes = anno.get("notes")

                    yield {
                        "id": it["sha"][:8],
                        "filename": it["name"],
                        "path": it["path"],
                        "url": it["download_url"],
                        "title": title,
                        "source": source,
                        "tags": combined_tags,
                        "notes": notes,
                    }


def list_all_images(limit: int = None, required_tags=None, q: str = ""):
    """
    Return up to `limit` images, optionally filtered by:
      - required_tags: list of canonical tag keys that ALL must be present
      - q: free-text search over filename, title, source, and tags
    """
    required = set(required_tags or [])
    q = (q or "").lower().strip()

    results = []
    for item in walk_images(BASE_PATH):
        # tag filter
        if required and not required.issubset(set(item["tags"])):
            continue

        # free-text search
        if q:
            haystack = " ".join([
                item.get("filename", ""),
                item.get("title", "") or "",
                item.get("source", "") or "",
                " ".join(item.get("tags", [])),
            ]).lower()
            if q not in haystack:
                continue

        results.append(item)
        if limit and len(results) >= limit:
            break

    return results


# ==============================
# Routes
# ==============================
@app.get("/")
def home():
    return "Hi marjo! You've reached Bamboo Image API (images + tags + annotations)"


@app.get("/images")
def list_images():
    """
    Query parameters:
      - tags: comma-separated canonical tags (e.g. geometry.hypar,species.petung)
      - q: free-text search over filename, title, source, tags
      - limit: max number of results (default 8)
    """
    tag_str = request.args.get("tags", "").strip()
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 8))

    want_tags = [t.strip() for t in tag_str.split(",") if t.strip()]
    items = list_all_images(limit=limit, required_tags=want_tags, q=q)
    return jsonify({"count": len(items), "items": items})


@app.get("/tags")
def list_tags():
    """
    Return tag definitions from tags.json + any discovered folder-based tags.
    """
    # Static schema from tags.json
    schema = load_tags_schema()

    # Discover tags from folder structure
    discovered = set()
    for item in list_all_images(limit=None):
        for t in item.get("tags", []):
            discovered.add(t)

    # Ensure all discovered tags exist in the schema, at least minimally
    for t in sorted(discovered):
        if t not in schema:
            group = t.split(".", 1)[0] if "." in t else "misc"
            schema[t] = {
                "group": group,
                "label": t,
                "syn": [],
            }

    # Return as a list
    out = [
        {"key": k, **v}
        for k, v in sorted(schema.items(), key=lambda kv: kv[0])
    ]
    return jsonify({"tags": out})


@app.get("/health")
def health():
    """
    Basic health check:
      - can we reach the GitHub repo and list BASE_PATH?
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


# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
