from flask import Flask, request, jsonify
import requests, os

OWNER = "marjjo"
REPO = "bamboo_images"   # exact repo name
BRANCH = "main"
BASE_PATH = "images"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # only needed if repo is private

app = Flask(__name__)

def gh_list(path):
    base = f"https://api.github.com/repos/{OWNER}/{REPO}"
    suffix = f"/contents/{path}" if path else "/contents"
    url = f"{base}{suffix}?ref={BRANCH}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def walk_images(path):
    stack = [path]
    while stack:
        cur = stack.pop()
        items = gh_list(cur)
        if isinstance(items, list):
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
    return "Bamboo Image API (Railway)"

@app.get("/images")
def list_images():
    tag_str = request.args.get("tags", "").strip()
    q = request.args.get("q", "").strip().lower()
    limit = int(request.args.get("limit", 8))
    want_tags = set(t.strip().lower() for t in tag_str.split(",") if t.strip())
    results = []
    for item in walk_images(BASE_PATH):
        if want_tags and not want_tags.issubset({t.lower() for t in item["tags"]}):
            continue
        if q and (q not in item["title"].lower() and all(q not in t.lower() for t in item["tags"])):
            continue
        results.append(item)
        if len(results) >= limit:
            break
    return jsonify({"count": len(results), "items": results})

from openai import OpenAI
import base64

# create a single client (reads OPENAI_API_KEY from env)
_oai = OpenAI()

@app.post("/generate")
def generate_image():
    """
    JSON body:
      {
        "prompt": "hypar petung bamboo pavilion at night",
        "size": "1024x1024"     # optional: 256x256, 512x512, 1024x1024
      }
    Returns:
      { "b64": "<base64_png>", "size": "1024x1024" }
    """
    data = request.get_json(force=True) or {}
    prompt = (data.get("prompt") or "").strip()
    size = (data.get("size") or "1024x1024").strip()

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    # call OpenAI Images
    try:
        resp = _oai.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            size=size,
            n=1
        )
        # OpenAI returns base64
        b64 = resp.data[0].b64_json
        return jsonify({"b64": b64, "size": size})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # for local testing only; Railway will use gunicorn
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
