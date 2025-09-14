import os
import json
import logging
import sqlite3
from collections import OrderedDict
from datetime import timedelta

from flask import Flask, request, jsonify, render_template, session, g
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# --------------------------
# Flask setup
# --------------------------
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

CORS(app, supports_credentials=True)
bcrypt = Bcrypt(app)

DB_PATH = os.environ.get("DB_PATH", "database.db")
SETTINGS_FILE = "config.json"

# --------------------------
# Database helpers
# --------------------------
def get_db():
    if "db" not in g:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        g.db = conn
    return g.db

@app.teardown_appcontext
def close_db(exception):
    db = g.pop("db", None)
    if db is not None:
        db.close()

def query_one(query, args=()):
    cur = get_db().execute(query, args)
    row = cur.fetchone()
    cur.close()
    return row

def execute(query, args=()):
    db = get_db()
    cur = db.execute(query, args)
    db.commit()
    return cur.lastrowid

# --------------------------
# Logging setup
# --------------------------
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def log_action(user_id: int, action: str, details: dict | None = None):
    logging.info("user_id=%s action=%s details=%s", user_id, action, json.dumps(details or {}))
    ua = request.headers.get("User-Agent")
    ip = request.headers.get("X-Forwarded-For", request.remote_addr)
    execute(
        "INSERT INTO logs (user_id, action, details, ip, user_agent) VALUES (?, ?, ?, ?, ?)",
        (user_id, action, json.dumps(details or {}), ip, ua),
    )

# --------------------------
# Auth helpers
# --------------------------
def current_user():
    uid = session.get("user_id")
    if not uid:
        return None
    row = query_one("SELECT id, name, email, role FROM users WHERE id=?", (uid,))
    return dict(row) if row else None

def login_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not session.get("user_id"):
            return jsonify({"error": "Bạn cần đăng nhập"}), 401
        return fn(*args, **kwargs)
    return wrapper

def admin_required(fn):
    from functools import wraps
    @wraps(fn)
    def wrapper(*args, **kwargs):
        user = current_user()
        if not user or user.get("role") != "admin":
            return jsonify({"error": "Chỉ admin được phép"}), 403
        return fn(*args, **kwargs)
    return wrapper

# --------------------------
# Model utilities
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
transform = None

class_names = [
    "butterfly","cat","chicken","cow","dog",
    "elephant","horse","sheep","spider","squirrel",
]

def get_current_model():
    if not os.path.exists(SETTINGS_FILE):
        return "AnimalNet18"
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f).get("current_model", "AnimalNet18")

def set_current_model(model_name):
    with open(SETTINGS_FILE, "w") as f:
        json.dump({"current_model": model_name}, f)

def load_model(model_name):
    global model, transform
    model_file = f"{model_name}.pth"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Không tìm thấy file {model_file}")

    state_dict = torch.load(model_file, map_location=device)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:]] = v if k.startswith("module.") else v

    backbone = models.resnet18(pretrained=False)
    num_ftrs = backbone.fc.in_features
    backbone.fc = nn.Linear(num_ftrs, len(class_names))
    backbone.load_state_dict(new_state_dict)
    model = backbone.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

# --------------------------
# Routes
# --------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not name or not email or not password:
        return jsonify({"error": "Thiếu thông tin"}), 400
    if query_one("SELECT id FROM users WHERE email=?", (email,)):
        return jsonify({"error": "Email đã tồn tại"}), 409

    pw_hash = bcrypt.generate_password_hash(password).decode("utf-8")
    uid = execute(
        "INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)",
        (name, email, pw_hash, "user"),
    )
    session["user_id"] = uid
    log_action(uid, "register", {"email": email})
    return jsonify({"message": "Đăng ký thành công", "user": {"id": uid, "name": name, "email": email, "role": "user"}})

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    user = query_one("SELECT * FROM users WHERE email=?", (email,))
    if not user or not bcrypt.check_password_hash(user["password"], password):
        return jsonify({"error": "Sai email hoặc mật khẩu"}), 401
    session["user_id"] = user["id"]
    log_action(user["id"], "login", {"email": email})
    return jsonify({"message": "Đăng nhập thành công","user": {"id": user["id"], "name": user["name"], "email": user["email"], "role": user["role"]}})

@app.route("/logout", methods=["POST"])
@login_required
def logout():
    uid = session.get("user_id")
    session.pop("user_id", None)
    log_action(uid, "logout", {})
    return jsonify({"message": "Đăng xuất thành công"})

@app.route("/me", methods=["GET"])
def me():
    user = current_user()
    if not user:
        return jsonify({"user": None})
    return jsonify({"user": user})

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Thiếu file ảnh"}), 400
    file = request.files["file"]
    img = Image.open(file).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    label = class_names[pred.item()]
    confidence = float(conf.item())
    user = current_user()
    log_action(
        user_id=user["id"],
        action="predict",
        details={"filename": getattr(file, "filename", None),
                 "label": label, "confidence": confidence}
    )

    # 👇 thêm dòng này để đọc model từ config.json
    current_model = get_current_model()

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "model": current_model
    })


@app.route("/history", methods=["GET"])
@login_required
def history():
    user = current_user()
    rows = get_db().execute(
        "SELECT id, action, details, created_at FROM logs WHERE user_id=? AND action='predict' ORDER BY id DESC LIMIT 50",
        (user["id"],),
    ).fetchall()
    items = []
    for r in rows:
        d = json.loads(r["details"]) if r["details"] else {}
        items.append({
            "id": r["id"],"action": r["action"],"created_at": r["created_at"],
            "filename": d.get("filename"),"label": d.get("label"),"confidence": d.get("confidence"),
        })
    return jsonify({"items": items})

@app.route("/logs", methods=["GET"])
@admin_required
def logs_all():
    rows = get_db().execute(
        """
        SELECT logs.id, users.email, users.name, users.role, logs.action, logs.details, logs.ip, logs.user_agent, logs.created_at
        FROM logs JOIN users ON logs.user_id = users.id
        ORDER BY logs.id DESC LIMIT 200
        """
    ).fetchall()
    data = []
    for r in rows:
        d = json.loads(r["details"]) if r["details"] else {}
        data.append({
            "id": r["id"],"email": r["email"],"name": r["name"],"role": r["role"],
            "action": r["action"],"details": d,"ip": r["ip"],"user_agent": r["user_agent"],"created_at": r["created_at"],
        })
    return jsonify({"logs": data})

# --------------------------
# Update model API
# --------------------------
@app.route("/update-model", methods=["POST"])
@admin_required
def update_model():
    data = request.get_json(force=True)
    model_name = data.get("model")
    if model_name not in ["AnimalVie", "AnimalNet18"]:
        return jsonify({"error": "Model không hợp lệ"}), 400
    try:
        load_model(model_name)
        set_current_model(model_name)
        user = current_user()
        log_action(user["id"], "update_model", {"model": model_name})
        return jsonify({"message": f"Model {model_name} đã được đặt làm mặc định"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    # load default model lúc khởi động
    current = get_current_model()
    load_model(current)
    if not os.path.exists(DB_PATH):
        print("Chưa có database.db. Chạy: python database_init.py")
    app.run(host="0.0.0.0", port=5000, debug=True)
