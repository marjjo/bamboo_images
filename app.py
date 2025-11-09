from flask import Flask, request, jsonify
import requests
import os

# --- CONFIG: set these to YOUR repo ---
OWNER = "marjjo"
REPO = "bamboo-images"          # your repo name
BRANCH = "main"                 # or "master"
BASE_PATH = "images"            # top folder that holds all images
# --------------------------------------

# Optional: use a GitHub token to avoid low rate limits (not required)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

app = Flask(__name__)

def gh_list(path):
    """List files/folders at a GitHub path using the Contents API."""
    url = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{path}?ref={BRANCH}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def walk_images(path):
    """Recursively walk the BASE_PATH and yield image items with tags."""
    stack = [path]
    while stack:
        cur = stack.pop()
        items = gh_list(cur)
        # GitHub returns either a dict (file) or list (directory)
        if isinstance(items, dict) and items.get("type") == "file":
            if items["name"].lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")):
                rel = items["path"][len(BASE_PATH):].strip("/")  # e.g. "hypar/hypar01.jpg"
                parts = rel.split("/")[:-1]  # folder parts as tags
                yield {
                    "id": items["sha"][:8],
                    "url": items["download_url"],   # direct raw file URL from GitHub
                    "title": items["name"],
                    "tags": [p for p in parts if p] # tags from folder names
                }
        elif isinstance(items, list):
            for it in items:
                if it["type"] == "dir":
                    stack.append(it["path"])
                elif it["type"] == "file" and it["name"].lower().endswith((".jpg",".jpeg",".png",".gif",".webp")):
                    rel = it["path"][len(BASE_PATH):].strip("/")
                    parts = rel.split("/")[:-1]
                    yield {
                        "id": it["sha"][:8],
                        "url": it["download_url"],
                        "title": it["name"],
                        "tags": [p for p in parts if p]
                    }

@app.get("/")
def home():
    return "Bamboo Image API (auto-indexes your GitHub folders)"

@app.get("/images")
def list_images():
    # Optional filters: ?tags=hypar,petung&limit=6&q=pavilion
    tag_str = request.args.get("tags", "").strip()
    q = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 8))

    want_tags = set([t.strip().lower() for t in tag_str.split(",") if t.strip()])
    results = []

    for item in walk_images(BASE_PATH):
        # tag filter
        if want_tags and not want_tags.issubset({t.lower() for t in item["tags"]}):
            continue
        # text filter (filename or tags)
        if q and (q not in item["title"].lower() and all(q not in t.lower() for t in item["tags"])):
            continue

        results.append(item)
        if len(results) >= limit:
            break

    return jsonify({"items": results, "count": len(results)})

if __name__ == "__main__":
    app.run(debug=True)

