from flask import Flask, render_template, request, jsonify
from pathlib import Path
import json, os
from engine import City

app = Flask(__name__)
SAVE_DIR = Path("saves")
SAVE_DIR.mkdir(exist_ok=True)

city = City()
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/grid", methods=["GET"])
def api_grid():
    return jsonify({"grid": city.render_grid()})

@app.route("/api/command", methods=["POST"])
def api_command():
    data = request.get_json(force=True)
    cmd = (data.get("cmd") or "").strip()
    try:
        if not cmd:
            city.tick()
            return jsonify({"ok": True, "grid": city.render_grid(), "msg": "tick"})
        msg = city.handle_command(cmd)
        return jsonify({"ok": True, "grid": city.render_grid(), "msg": msg})
    except Exception as e:
        return jsonify({"ok": False, "grid": city.render_grid(), "msg": str(e)}), 400

@app.route("/api/save", methods=["POST"])
def api_save():
    data = request.get_json(force=True)
    name = (data.get("name") or "city").strip()
    path = SAVE_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(city.to_dict(), f, ensure_ascii=False, indent=2)
    return jsonify({"ok": True, "msg": f"Save in {path.name}"})

@app.route("/api/load", methods=["POST"])
def api_load():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    path = SAVE_DIR / f"{name}.json"
    if not path.exists():
        return jsonify({"ok": False, "msg": "File Not Found"}), 404
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    global city
    city = City.from_dict(payload)
    return jsonify({"ok": True, "grid": city.render_grid(), "msg": f"Charged {path.name}"})


if __name__ == "__main__":
    app.run(debug=True)