import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import time
import json
import re
import os
import io
from typing import List, Dict, Any

# -----------------------------
# Optional libraries
# -----------------------------

# PDF text extraction
try:
    from pypdf import PdfReader
    PDF_OK = True
except Exception:
    PDF_OK = False

# PDF rendering
PDF_RENDER_OK = False
try:
    import fitz  # PyMuPDF
    PDF_RENDER_OK = True
except Exception:
    PDF_RENDER_OK = False

# PPTX parsing (text + embedded images)
PPTX_OK = False
try:
    from pptx import Presentation
    PPTX_OK = True
except Exception:
    PPTX_OK = False

# Groq
GROQ_OK = False
try:
    from groq import Groq
    GROQ_OK = True
except Exception:
    GROQ_OK = False


# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Smart Study Planner", page_icon="📚", layout="wide")
st.title("📚 Smart Study Planner")
st.caption("From the works of STEM 12 A")


# -----------------------------
# Profile
# -----------------------------
with st.expander("👤 Profile", expanded=False):
    user_code = st.text_input("User code", value="demo").strip().lower()

safe_code = "".join(ch for ch in user_code if ch.isalnum() or ch in ["_", "-"]) or "demo"

DATA_PATH = Path(f"data_{safe_code}.csv")
FILES_DIR = Path(f"files_{safe_code}")
FILES_DIR.mkdir(exist_ok=True)
META_PATH = FILES_DIR / "files_meta.json"


# -----------------------------
# Helpers
# -----------------------------
def default_df() -> pd.DataFrame:
    t = date.today()
    return pd.DataFrame(
        {
            "Subject": ["Physics", "Gen Math", "Biology"],
            "Difficulty (1-5)": [5, 4, 3],
            "Exam Date": [t + timedelta(days=10), t + timedelta(days=7), t + timedelta(days=14)],
            "Minutes Done (this week)": [0, 0, 0],
        }
    )

def load_df() -> pd.DataFrame:
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            df["Exam Date"] = pd.to_datetime(df["Exam Date"], errors="coerce").dt.date
            return df
        except Exception:
            return default_df()
    return default_df()

def save_df(df: pd.DataFrame) -> None:
    out = df.copy()
    out["Exam Date"] = pd.to_datetime(out["Exam Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out.to_csv(DATA_PATH, index=False)

def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")

def now_str() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

def load_meta() -> Dict[str, Any]:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"files": []}
    return {"files": []}

def save_meta(meta: Dict[str, Any]) -> None:
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def parse_json_loose(raw: str):
    raw = (raw or "").strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(raw[start:end + 1])
    raise ValueError("No JSON object found")

def extract_text_from_pptx(p: Path, max_slides: int = 50) -> str:
    if not (PPTX_OK and p.suffix.lower() == ".pptx"):
        return ""
    try:
        prs = Presentation(str(p))
        out = []
        for i, slide in enumerate(prs.slides):
            if i >= max_slides:
                break
            chunks = []
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text:
                    t = (shape.text or "").strip()
                    if t:
                        chunks.append(t)
            if chunks:
                out.append(f"[Slide {i+1}]\n" + "\n".join(chunks))
        return "\n\n".join(out).strip()
    except Exception:
        return ""

def extract_images_from_pptx(p: Path, max_slides: int = 30, max_images: int = 50) -> List[Dict[str, Any]]:
    if not (PPTX_OK and p.suffix.lower() == ".pptx"):
        return []
    images: List[Dict[str, Any]] = []
    try:
        prs = Presentation(str(p))
        for si, slide in enumerate(prs.slides):
            if si >= max_slides:
                break
            for shape in slide.shapes:
                if len(images) >= max_images:
                    break
                img_obj = getattr(shape, "image", None)
                if img_obj is None:
                    continue
                images.append(
                    {
                        "bytes": img_obj.blob,
                        "slide": si + 1,
                    }
                )
            if len(images) >= max_images:
                break
        return images
    except Exception:
        return []


# -----------------------------
# Session init
# -----------------------------
if "active_user_code" not in st.session_state or st.session_state.active_user_code != safe_code:
    st.session_state.active_user_code = safe_code
    st.session_state.subjects = load_df()

if "plan_ready" not in st.session_state:
    st.session_state.plan_ready = False

# timer
if "timer_running" not in st.session_state:
    st.session_state.timer_running = False
if "timer_subject" not in st.session_state:
    st.session_state.timer_subject = ""
if "timer_minutes" not in st.session_state:
    st.session_state.timer_minutes = 25
if "timer_start" not in st.session_state:
    st.session_state.timer_start = 0.0
if "timer_end" not in st.session_state:
    st.session_state.timer_end = 0.0

# preview state
if "preview_path" not in st.session_state:
    st.session_state.preview_path = ""
if "preview_name" not in st.session_state:
    st.session_state.preview_name = ""
if "preview_subject" not in st.session_state:
    st.session_state.preview_subject = ""

# assessment state
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "quiz_subject" not in st.session_state:
    st.session_state.quiz_subject = ""
if "quiz_file" not in st.session_state:
    st.session_state.quiz_file = ""
if "user_answers" not in st.session_state:
    st.session_state.user_answers = {}


# -----------------------------
# Settings
# -----------------------------
with st.expander("⚙️ Settings", expanded=False):
    minutes_per_day = st.number_input("Minutes available per day", 30, 600, 180, 10)
    days_to_plan = st.slider("Plan length (days)", 3, 14, 7)


# -----------------------------
# 1) Subjects
# -----------------------------
st.subheader("1) Subjects")

with st.expander("➕ Add a subject", expanded=False):
    new_subject = st.text_input("Subject name", placeholder="e.g., Chemistry")
    new_diff = st.slider("Difficulty (1–5)", 1, 5, 3)
    new_exam = st.date_input("Exam date", value=date.today() + timedelta(days=7))
    if st.button("Add subject"):
        s = (new_subject or "").strip()
        if not s:
            st.warning("Please enter a subject name.")
        else:
            df_add = st.session_state.subjects.copy()
            df_add.loc[len(df_add)] = [s, int(new_diff), new_exam, 0]
            st.session_state.subjects = df_add
            save_df(df_add)
            st.success(f"Added: {s}")
            st.rerun()

edited_df = st.data_editor(
    st.session_state.subjects,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Exam Date": st.column_config.DateColumn("Exam Date", format="MMM DD, YYYY"),
        "Difficulty (1-5)": st.column_config.NumberColumn("Difficulty (1-5)", min_value=1, max_value=5, step=1),
        "Minutes Done (this week)": st.column_config.NumberColumn("Minutes Done (this week)", min_value=0, step=10),
    },
)

cA, cB = st.columns([1, 1])
with cA:
    if st.button("💾 Save changes"):
        st.session_state.subjects = edited_df.copy()
        save_df(st.session_state.subjects)
        st.success("Saved!")
with cB:
    if st.button("🗑 Reset to default"):
        st.session_state.subjects = default_df()
        save_df(st.session_state.subjects)
        st.success("Reset done!")
        st.rerun()

subjects_clean = (
    pd.Series(st.session_state.subjects["Subject"])
    .dropna()
    .astype(str)
    .str.strip()
)
subjects_clean = subjects_clean[subjects_clean != ""].tolist()

# -----------------------------
# (Rest of your app stays the same)
# -----------------------------
# NOTE: You asked ONLY:
# 1) Caption changed to "From the works of STEM 12 A"
# 2) Remove the date caption line
#
# So I removed the "Dates display like: Feb 21, 2026" caption block.
#
# Paste this file OVER your current app.py (or copy the changes into your full file).
