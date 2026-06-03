"""
Flask mobile web demo — Gait Analysis
Routes:
  GET  /           업로드 폼
  POST /diagnose   영상 업로드 → 진단 결과 JSON
  GET  /health     서버 상태 확인
"""

import uuid
import importlib.util
from pathlib import Path

from flask import Flask, request, jsonify, render_template

# 04_diagnosis.py는 숫자로 시작해 일반 import 불가 → importlib으로 로드
_spec = importlib.util.spec_from_file_location(
    "diagnosis",
    Path(__file__).resolve().parent / "04_diagnosis.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
diagnose = _mod.diagnose

_BASE        = Path(__file__).resolve().parent.parent
UPLOAD_DIR   = _BASE / "output" / "uploads"
TEMPLATE_DIR = _BASE / "templates"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}

app = Flask(__name__, template_folder=str(TEMPLATE_DIR))
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB


# ── 업로드 폼 ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


# ── 진단 ───────────────────────────────────────────────────────────────────────

@app.route("/diagnose", methods=["POST"])
def diagnose_route():
    if "video" not in request.files:
        return jsonify({"error": "video 파일이 없습니다."}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "파일명이 비어 있습니다."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"지원하지 않는 형식입니다: {ext}"}), 415

    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    try:
        file.save(tmp_path)
        result = diagnose(tmp_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return jsonify(result)


# ── 헬스체크 ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ── 진입점 ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
