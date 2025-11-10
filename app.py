from flask import Flask, request, jsonify
from flask_cors import CORS
import requests, os, json, time

# ==============================
# Config
# ==============================
OWNER = "marjjo"
REPO = "bamboo_images"
BRANCH = "main"
BASE_PATH = "images"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAGS_PATH = os.getenv("TAGS_PATH", "tags.json")

app = Flask(__name__)
CORS(app)

# ==============================
# Folder → Canonical Tag Mapping
# ==============================
FOLDER_CANON = {
    "bamboo": "material.bamboo",
    "precedents": "collection.precedent",
    "pavilion": "scale.pavilion",
    "resort": "program.resort",
    "shape": "shape",
    "system": "system",
    "material": "material",
    "scale": "scale",
    "program": "program",
    "context": "context",
    "style": "style",
    "lighting": "lighting",
    "camera": "camera",
    "collection": "collection"
}

def canonicalize_parts(parts):
    canon = []
    for i, p in enumerate(parts):
        p = (p or "").strip()
        if not p:
            continue
        if "." in p:
            canon.append(p)
            continue
        parent = parts[i-1] if i > 0 else None
        if parent and parent in FOLDER_CANON and parent in {
            "shape","system","material","scale","program","context",
            "style","lighting","camera","collection"
        }:
            canon.append(f"{parent}.{p}")
            continue
        canon.append(FOLDER_CANON.get(p, p))
    return canon

# ==============================
# Tag Schema (groups/prompts/synonyms)
# ==============================
def load_tags_schema(path: str = TAGS_PATH):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    norm = {}
    for k, v in data.items():
        norm[k] = {
            "group": v.get("group", "misc"),
            "prompt": v.get("prompt", k.replace("_", " ")),
            "syn": [s.lower() for s in v.get("syn", [])],
        }
    return norm

# Initial schema load
TAGS_SCHEMA = load_tags_schema()

# ==============================
# 🔁 Auto-reload for tags.json
# ==============================
_last_reload = 0
_reload_interval = 60  # seconds
_last_mtime = None

def maybe_reload_schema(path=TAGS_PATH):
    """Reload tags.json automatically every _reload_interval seconds if changed."""
    global TAGS_SCHEMA, _last_reload, _last_mtime
    now = time.time()
    if now - _last_reload < _reload_interval:
        return
    try:
        mtime = os.path.getmtime(path)
    except FileNotFoundError:
        return
    if _last_mtime is None or mtime != _last_mtime:
        try:
            TAGS_SCHEMA = load_tags_schema(path)
            _last_mtime = mtime
            print(f"[Auto-reload] tags.json reloaded at {time.ctime(mtime)}")
        except Exception as e:
            print(f"[Auto-reload] Failed to reload schema: {e}")
    _last_reload = now

# ==============================
# Tag utilities
# ==============================
def ensure_schema_keys_for_discovered_tags(discovered: set):
    for key in sorted(discovered):
        if key in TAGS_SCHEMA:
            continue
        group = key.split(".", 1)[0] if "." in key else "misc"
        TAGS_SCHEMA[key] = {
            "group": group or "misc",
            "prompt": key.replace("_", " "),
            "syn": [],
        }

def synonym_to_keys(token: str, schema: dict):
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
    for t in (tokens or []):
        if not t:
            continue
        expanded |= synonym_to_keys(t, schema)
    return expanded

def build_prompt(tag_keys, schema):
    order = ["shape", "system", "material", "scale", "context", "style", "lighting", "camera"]
    first_by_group = {}
    for key in (tag_keys or []):
        meta = schema.get(key)
        if not meta:
            continue
        g = meta.get("group", "misc")
        if g not in first_by_group:
            first_by_group[g] = meta.get("prompt", key.replace("_", " "))
    phrases = [first_by_group[g] for g in order if g in first_by_group]
    if "scale" not in first_by_group:
        phrases.append("open-air pavilion")
    phrases.append("high-detail, photorealistic, architectural visualization")
    return ", ".join(phrases)

def search_images(images, schema, req_tags=None, any_tags=None, q: str = ""):
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
            hay = (img.get("id","") + " " + img.get("title","") + " " + " ".join(img.get("tags", []))).lower()
            if q not in hay:
                return False
        return True
    return [img for img in images if match(img)]

# ==============================
# GitHub list/walk
# ==============================
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
                    raw_parts = rel.split("/")[:-1]
                    tags = canonicalize_parts(raw_parts)
                    yield {
                        "id": it["sha"][:8],
                        "url": it["download_url"],
                        "title": it["name"],
                        "tags": [t for t in tags if t],
                    }

def list_all_images(limit: int = 8, required_tags=None, q: str = ""):
    required = set((required_tags or []))
    results = []
    for item in walk_images(BASE_PATH):
        if required and not required.issubset(set(item["tags"])):
            continue
        if q:
            ql = q.lower()
            if (ql not in item["title"].lower() and all(ql not in t.lower() for t in item["tags"])):
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
    return "Bamboo Image API (Render/Railway)"

@app.get("/images")
def list_images():
    tag_str = request.args.get("tags", "").strip()
    q = request.args.get("q", "").strip()
    limit = int(request.args.get("limit", 8))
    want_tags = [t.strip() for t in tag_str.split(",") if t.strip()]
    items = list_all_images(limit=limit, required_tags=want_tags, q=q)
    return jsonify({"count": len(items), "items": items})

@app.get("/tags")
def list_tags():
    maybe_reload_schema()
    discovered = set()
    for item in list_all_images(limit=10_000):
        for t in item["tags"]:
            discovered.add(t)
    ensure_schema_keys_for_discovered_tags(discovered)
    out = [{"key": k, **v} for k, v in sorted(TAGS_SCHEMA.items(), key=lambda kv: kv[0])]
    return jsonify({"tags": out})

@app.get("/search")
def search():
    maybe_reload_schema()
    tags_param = request.args.get("tags", "")
    any_param  = request.args.get("any", "")
    q          = request.args.get("q", "")
    limit      = int(request.args.get("limit", 12))
    req_tags = [t.strip() for t in tags_param.split(",") if t.strip()]
    any_tags = [t.strip() for t in any_param.split(",") if t.strip()]
    images = list_all_images(limit=10_000)
    discovered = {t for img in images for t in img["tags"]}
    ensure_schema_keys_for_discovered_tags(discovered)
    results = search_images(images, TAGS_SCHEMA, req_tags=req_tags, any_tags=any_tags, q=q)
    return jsonify({"count": len(results[:limit]), "results": results[:limit]})

@app.get("/prompt")
def prompt_from_tags():
    maybe_reload_schema()
    tags_param = request.args.get("tags", "").strip()
    image_id   = request.args.get("id", "").strip()
    if image_id:
        imgs = list_all_images(limit=10_000)
        tag_keys = []
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
    ensure_schema_keys_for_discovered_tags(set(tag_keys))
    prompt = build_prompt(tag_keys, TAGS_SCHEMA)
    return jsonify({"tags": tag_keys, "prompt": prompt})

@app.post("/generate")
def generate_image():
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
                "Content-Type": "application/json",
            },
            json={"model": "gpt-image-1", "prompt": prompt, "size": size, "n": 1},
            timeout=90,
        )
        r.raise_for_status()
        b64 = r.json()["data"][0]["b64_json"]
        return jsonify({"b64": b64, "size": size})
    except requests.HTTPError as e:
        return jsonify({"error": f"OpenAI HTTP {e.response.status_code}: {e.response.text}"}), 502
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
