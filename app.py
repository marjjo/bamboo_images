from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os, json

# --- GitHub Image Library Config ---
OWNER = "marjjo"
REPO = "bamboo_images"   # exact repo name
BRANCH = "main"
BASE_PATH = "images"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # only needed if repo is private
# -----------------------------------

# --- OpenAI ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
CORS(app)  # allow browser clients

# ===============================
# Tag utilities (schema + search)
# ===============================

def load_tags_schema(path: str = "tags.json"):
    """
    Load optional tag schema that explains meanings, groups, and synonyms.
    Expected structure:
    {
      "shape.hypar": { "group": "shape", "prompt": "hyperbolic paraboloid roof", "syn": ["hypar","hyperbolic_paraboloid"] },
      "material.petung": { "group": "material", "prompt": "Petung (Dendrocalamus asper) bamboo", "syn": ["petung","asper"] },
      ...
    }
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # normalize
    norm = {}
    for k, v in data.items():
        norm[k] = {
            "group": v.get("group", "misc"),
            "prompt": v.get("prompt", k.replace("_", " ")),
            "syn": [s.lower() for s in v.get("syn", [])]
        }
    return norm

TAGS_SCHEMA = load_tags_schema(os.getenv("TAGS_PATH", "tags.json"))

def ensure_schema_keys_for_discovered_tags(discovered: set):
    """
    If we found folder tags that aren't in tags.json, add lightweight entries
    so GPTs can still interpret them. Group guesses use the prefix before '.'
    (e.g., 'shape.hypar' -> group 'shape'); otherwise 'misc'.
    """
    for key in sorted(discovered):
        if key in TAGS_SCHEMA:
            continue
        group = "misc"
        if "." in key:
            group = key.split(".", 1)[0] or "misc"
        TAGS_SCHEMA[key] = {
            "group": group,
            "prompt": key.replace("_", " "),
            "syn": []
        }

def synonym_to_keys(token: str, schema: dict):
    """Map a token (exact key or synonym) -> set of known tag keys; fallback to substring match on keys."""
    t = token.lower().strip()
    matches = set()
    for key, meta in schema.items():
        if t == key.lower() or t in meta.get("syn", []):
            matches.add(key)
    if not matches:
        for key in schema.keys():
            if t in key.lower():
                matches.add(key)
    return matches

def expand_tokens(tokens, schema):
    expanded = set()
    for t in tokens or []:
        if not t:
            continue
        expanded |= synonym_to_keys(t, schema)
    return expanded

def search_images(images, schema, req_tags=None, any_tags=None, q: str = ""):
    """
    images: list of {id, url, title, tags: [tagKeys or folder parts]}
    schema: normalized tag dict
    req_tags: ALL must match (keys or synonyms)
    any_tags: ANY may match (keys or synonyms)
    q: free text over id, title, tag keys
    """
    req = expand_tokens(req_tags, schema)
    any_ = expand_tokens(any_tags, schema)
    q = (q or "").lower().strip()

    def match(img):
        tagset = set(img.get("tags", []))
        if req and not req.issubset(tagset):
            return False
        if any_ and not (tagset & any_):
            return False
        if q:
            hay = (img.get("id", "") + " " + img.get("title", "") + " " + " ".join(img.get("tags", []))).lower()
            if q not in hay:
                return False
        return True

    return [img for img in images if match(img)]

def build_prompt(tag_keys, schema):
    """
    Turn a list of tag keys into a single DALL·E prompt using schema prompts.
    Use at most one prompt phrase per group, in priority order.
    """
    order = ["shape", "system", "material", "scale", "context", "style", "lighting", "camera"]
    first_by_group = {}
    for key in tag_keys or []:
        meta = schema.get(key)
        if not meta:
            continue
        g = meta.get("group", "misc")
        if g not in first_by_group:
            first_by_group[g] = meta.get("prompt", key.replace("_", " "))

    phrases = []
    for g in order:
        if g in first_by_group:
            phrases.append(first_by_group[g])

    if "scale" not in first_by_group:
        phrases.append("open-air pavilion")  # safe default

    # Generic rendering hints
    phrases.append("high-detail, photorealistic, architectural visualization")
    return ", ".join(phrases)

# ======================
# GitHub list/walk code
# ======================

def gh_list(path):
    base = f"https://api.github.com/repos/{OWNER}/{REPO}"
    suffix = f"/contents/{path}" if path else "/contents"
    url = f"{base}{suffix}?ref={BRANCH}"
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
    r = requests.get(url, headers=headers, timeout=15)
    r.raise_for_status()
    return r.json()

def walk_images(path):
    """
    Traverse the GitHub repo folder tree under BASE_PATH.
    Produce items with tags equal to folder parts between BASE_PATH and the file.
    """
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
                        "tags": [p for p in parts if p]  # raw folder tags (e.g., ['precedents','pavilion'])
                    }

def list_all_images(limit: int = 8, required_tags=None, q: str = ""):
    required_tags = set(t.lower() for t in (required_tags or []))
    results = []
    for item in walk_images(BASE_PATH):
        if required_tags and not required_tags.issubset({t.lower() for t in item["tags"]}):
            continue
        if q:
            ql = q.lower()
            if (ql not in item["title"].lower() and all(ql not in t.lower() for t in item["tags"])):
                continue
        results.append(item)
        if len(results) >= limit:
            break
    return results

# ============
# Basic Routes
# ============

@app.get("/")
def home():
    return "Bamboo Image API (Render/Railway)"

@app.get("/images")
def list_images():
    """
    Query params:
      - tags: comma-separated folder tags (exact, from path)
      - q: free text over title and raw tags
      - limit: default 8
    """
    tag_str = request.args.get("tags", "").strip()
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 8))
    want_tags = [t.strip() for t in tag_str.split(",") if t.strip()]
    items = list_all_images(limit=limit, required_tags=want_tags, q=q)
    return jsonify({"count": len(items), "items": items})

# =========================
# New: Tags & Search Routes
# =========================

@app.get("/tags")
def list_tags():
    """
    Returns the **effective** tag schema (tags.json + auto-discovered folder tags).
    This is what your GPT should read to understand tag meanings & synonyms.
    """
    # Discover all folder tags that exist in the repo
    discovered = set()
    for item in walk_images(BASE_PATH):
        for t in item["tags"]:
            discovered.add(t)
    # Enrich schema with any missing discovered keys
    ensure_schema_keys_for_discovered_tags(discovered)

    # Present schema as array for easier client consumption
    out = [{"key": k, **v} for k, v in sorted(TAGS_SCHEMA.items(), key=lambda kv: kv[0])]
    return jsonify({"tags": out})

@app.get("/search")
def search():
    """
    Flexible search:
      - tags: comma-separated REQUIRED tags (keys or synonyms)
      - any: comma-separated OPTIONAL tags (keys or synonyms; match ANY)
      - q: free text over id, title, and tag keys
      - limit: result cap (default 12)
    """
    tags_param = request.args.get("tags","")
    any_param  = request.args.get("any","")
    q          = request.args.get("q","")
    limit      = int(request.args.get("limit", 12))

    req_tags = [t.strip() for t in tags_param.split(",") if t.strip()]
    any_tags = [t.strip() for t in any_param.split(",") if t.strip()]

    # Build a full list of images once
    images = list_all_images(limit=10_000)  # large cap for filtering

    # Ensure schema knows about all discovered folder tags
    discovered = set()
    for img in images:
        for t in img["tags"]:
            discovered.add(t)
    ensure_schema_keys_for_discovered_tags(discovered)

    results = search_images(images, TAGS_SCHEMA, req_tags=req_tags, any_tags=any_tags, q=q)
    results = results[:limit]
    return jsonify({"count": len(results), "results": results})

@app.get("/prompt")
def prompt_from_tags():
    """
    Build a DALL·E prompt from:
      - ?tags=shape.hypar,material.petung
    Or from an image id:
      - ?id=<image_id>
    """
    tags_param = request.args.get("tags", "").strip()
    image_id   = request.args.get("id", "").strip()

    # Resolve tags either from query or by looking up an image
    tag_keys = []
    if image_id:
        imgs = list_all_images(limit=10_000)
        for img in imgs:
            if img["id"] == image_id:
                tag_keys = img.get("tags", [])
                break
        if not tag_keys:
            return jsonify({"error": f"Image id '{image_id}' not found"}), 404
    else:
        tag_keys = [t.strip() for t in tags_param.split(",") if t.strip()]
        if not tag_keys:
            return jsonify({"error": "Provide ?tags=... or ?id=<image_id>"}), 400

    # Ensure schema includes discovered tags
    discovered = set(tag_keys)
    ensure_schema_keys_for_discovered_tags(discovered)

    prompt = build_prompt(tag_keys, TAGS_SCHEMA)
    return jsonify({"tags": tag_keys, "prompt": prompt})

# =======================================
# Image Generation via OpenAI REST API
# =======================================

@app.post("/generate")
def generate_image():
    """
    POST JSON:
    {
      "prompt": "hypar petung bamboo pavilion at night",
      "size": "1024x1024"
    }
    """
    data = request.get_json(force=True) or {}
    prompt = (data.get("prompt") or "").strip()
    size = (data.get("size") or "1024x1024").strip()

    if not prompt:
        return jsonify({"error": "Missing 'prompt'"}), 400

    try:
        r = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-image-1",
                "prompt": prompt,
                "size": size,
                "n": 1
            },
            timeout=90
        )
        r.raise_for_status()
        b64 = r.json()["data"][0]["b64_json"]
        return jsonify({"b64": b64, "size": size})
    except requests.HTTPError as e:
        return jsonify({"error": f"OpenAI HTTP {e.response.status_code}: {e.response.text}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Entry Point ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
