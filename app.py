from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, send_from_directory
import os
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
 # heavy imports (will be imported lazily where needed)
fitz = None
docx = None
TfidfVectorizer = None
cosine_similarity = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "knowledge.db")

ALLOWED_EXT = {'.pdf', '.docx', '.txt'}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "change_this_secret_in_production"

# --- Database helpers ---
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    );""")
    conn.commit()
    conn.close()

init_db()

# --- utilities ---
def extract_text_from_pdf(filepath):
    text = []
    try:
        global fitz
        if fitz is None:
            import fitz as _fitz
            fitz = _fitz
        with fitz.open(filepath) as doc:
            for page in doc:
                text.append(page.get_text())
    except Exception as e:
        print("PDF read error:", e)
    return "\\n".join(text)

def extract_text_from_docx(filepath):
    try:
        global docx
        if docx is None:
            import docx as _docx
            docx = _docx
        _doc = docx.Document(filepath)
        return "\\n".join([para.text for para in _doc.paragraphs])
    except Exception as e:
        print("DOCX read error:", e)
        return ""

def save_extracted_text_for_user(username, text):
    user_txt = os.path.join(app.config["UPLOAD_FOLDER"], f"{username}.txt")
    with open(user_txt, "w", encoding="utf-8") as f:
        f.write(text)

def load_extracted_text_for_user(username):
    user_txt = os.path.join(app.config["UPLOAD_FOLDER"], f"{username}.txt")
    if os.path.exists(user_txt):
        with open(user_txt, "r", encoding="utf-8") as f:
            return f.read()
    return ""

def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("signin"))
        return f(*args, **kwargs)
    return decorated

# --- Routes ---
@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(os.path.join(BASE_DIR, "static"), filename)

@app.route("/")
def home():
    if "user_id" in session:
        return redirect(url_for("index"))
    return redirect(url_for("signin"))

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if not username or not password:
            flash("Provide username and password", "danger")
            return redirect(url_for("signup"))
        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                        (username, generate_password_hash(password)))
            conn.commit()
            flash("Account created. Please sign in.", "success")
            return redirect(url_for("signin"))
        except sqlite3.IntegrityError:
            flash("Username already exists.", "danger")
            return redirect(url_for("signup"))
        finally:
            conn.close()
    return render_template("signup.html")

@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        conn.close()
        if row and check_password_hash(row["password"], password):
            session["user_id"] = row["id"]
            session["username"] = username
            flash("Signed in successfully", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for("signin"))
    return render_template("signin.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for("signin"))

@app.route("/app")
@login_required
def index():
    return render_template("index.html", username=session.get("username"))

@app.route("/upload", methods=["POST"])
@login_required
def upload_file():
    if "file" not in request.files:
        return jsonify({"message":"No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"message":"No selected file"}), 400
    filename = secure_filename(file.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"message":"Unsupported file type"}), 400
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{session['username']}_{filename}")
    file.save(save_path)
    # extract text based on extension
    if ext == ".pdf":
        text = extract_text_from_pdf(save_path)
    elif ext == ".docx":
        text = extract_text_from_docx(save_path)
    else:
        with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    if not text.strip():
        return jsonify({"message":"Uploaded but no text extracted"}), 200
    save_extracted_text_for_user(session["username"], text)
    return jsonify({"message":"File uploaded and processed successfully"}), 200

@app.route("/ask", methods=["POST"])
@login_required
def ask_question():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"answer":"Ask a non-empty question"}), 200
    doc_text = load_extracted_text_for_user(session["username"])
    if not doc_text.strip():
        return jsonify({"answer":"No document uploaded yet. Please upload first."}), 200
    # better sentence splitting (split on sentence boundaries and newlines)
    import re
    raw = doc_text.replace("\r\n", "\n").replace("\r", "\n")
    # split on common sentence boundaries (.!?\n) while keeping reasonable chunk sizes
    parts = re.split(r'(?<=[\.!?])\s+|\n+', raw)
    sentences = [re.sub('\s+', ' ', p).strip() for p in parts if p and p.strip()]
    # fallback to the whole document if no sentences found
    if not sentences:
        sentences = [raw[:2000]]

    try:
        # lazy import sklearn pieces
        global TfidfVectorizer, cosine_similarity
        if TfidfVectorizer is None or cosine_similarity is None:
            from sklearn.feature_extraction.text import TfidfVectorizer as _T
            from sklearn.metrics.pairwise import cosine_similarity as _C
            TfidfVectorizer = _T
            cosine_similarity = _C

        # preprocess: lowercase (TF-IDF will handle tokenization)
        processed_sentences = [s.lower() for s in sentences]
        proc_question = question.lower()

        # use ngrams and english stop words to improve matching
        vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
        vectors = vectorizer.fit_transform(processed_sentences + [proc_question])
        q_vec = vectors[-1]
        s_vecs = vectors[:-1]
        similarities = cosine_similarity(q_vec, s_vecs).flatten()

        # choose top-k matches
        top_k = 3
        ranked_idx = similarities.argsort()[::-1][:top_k]
        top_scores = similarities[ranked_idx]

        # threshold to avoid returning unrelated sentences
        score_threshold = 0.20
        if top_scores.size == 0 or top_scores[0] < score_threshold:
            return jsonify({"answer":"I couldn't find a confident answer in the uploaded document."}), 200

        # build a concise answer from the top non-empty matches
        matched = []
        used = set()
        for idx, score in zip(ranked_idx, top_scores):
            if score < score_threshold:
                continue
            text = sentences[int(idx)].strip()
            if text and text not in used:
                matched.append(text)
                used.add(text)
        # join top matches (limit length)
        if not matched:
            return jsonify({"answer":"I couldn't find a confident answer in the uploaded document."}), 200
        answer = ' '.join(matched)[:1200]
    except Exception as e:
        answer = "Could not compute answer: " + str(e)
    return jsonify({"answer": answer}), 200


@app.route('/health')
def health():
    return jsonify({'status':'ok'})

if __name__ == "__main__":
    app.run(debug=True)
