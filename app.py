import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import time
import json
import re
import os
import base64
import streamlit.components.v1 as components

# Optional PDF text extraction
try:
    from pypdf import PdfReader
    PDF_OK = True
except Exception:
    PDF_OK = False

# Optional Groq AI
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
st.caption("Planner + Timer + Study Files + AI Assessments (Groq)")

# -----------------------------
# Profile (Per-user)
# -----------------------------
with st.expander("👤 Profile", expanded=False):
    user_code = st.text_input("User code", value="demo").strip().lower()

safe_code = "".join(ch for ch in user_code if ch.isalnum() or ch in ["_", "-"]) or "demo"

DATA_PATH = Path(f"data_{safe_code}.csv")
FILES_DIR = Path(f"files_{safe_code}")
FILES_DIR.mkdir(exist_ok=True)
META_PATH = FILES_DIR / "files_meta.json"


# -----------------------------
# Helpers: data
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

def nice_date(d) -> str:
    try:
        return pd.to_datetime(d).strftime("%b %d, %Y")
    except Exception:
        return ""

def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")

# -----------------------------
# Helpers: files meta
# -----------------------------
def load_meta() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"files": []}
    return {"files": []}

def save_meta(meta: dict) -> None:
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def now_str() -> str:
    return pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")

# -----------------------------
# Helpers: extract text
# -----------------------------
def extract_text_from_path(p: Path) -> str:
    suf = p.suffix.lower()
    if suf == ".txt":
        return p.read_text(errors="ignore")
    if suf == ".pdf" and PDF_OK:
        try:
            reader = PdfReader(str(p))
            parts = []
            for page in reader.pages[:12]:
                parts.append(page.extract_text() or "")
            return "\n".join(parts)
        except Exception:
            return ""
    return ""

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

# -----------------------------
# Session init
# -----------------------------
if "active_user_code" not in st.session_state or st.session_state.active_user_code != safe_code:
    st.session_state.active_user_code = safe_code
    st.session_state.subjects = load_df()

if "plan_ready" not in st.session_state:
    st.session_state.plan_ready = False

# timer state
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

# file preview state (for better UX)
if "preview_path" not in st.session_state:
    st.session_state.preview_path = ""
if "preview_name" not in st.session_state:
    st.session_state.preview_name = ""
if "preview_subject" not in st.session_state:
    st.session_state.preview_subject = ""


# -----------------------------
# Settings (mobile-friendly)
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

cA, cB, cC = st.columns([1, 1, 2])
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
with cC:
    st.caption("Dates display like: Feb 21, 2026")

subjects_clean = (
    pd.Series(st.session_state.subjects["Subject"])
    .dropna()
    .astype(str)
    .str.strip()
)
subjects_clean = subjects_clean[subjects_clean != ""].tolist()


# -----------------------------
# Study Files (tagged by subject) + Better Preview UX
# -----------------------------
st.subheader("📁 Study Files (tagged by subject)")

meta = load_meta()

with st.expander("Upload study files", expanded=False):
    if not subjects_clean:
        st.warning("Add at least 1 subject first so files can be tagged.")
    else:
        tag_subject = st.selectbox("Which subject is this file for?", subjects_clean)
        uploads = st.file_uploader(
            "Upload notes (TXT / PDF / images)",
            type=["txt", "pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        if st.button("⬆️ Save uploaded files"):
            if not uploads:
                st.warning("No files selected.")
            else:
                subj_dir = FILES_DIR / tag_subject
                subj_dir.mkdir(exist_ok=True)

                added = 0
                for f in uploads:
                    fname = sanitize_name(f.name)
                    path = subj_dir / fname
                    path.write_bytes(f.getbuffer())
                    meta["files"].append(
                        {
                            "subject": tag_subject,
                            "name": fname,
                            "path": str(path),
                            "uploaded_at": now_str(),
                        }
                    )
                    added += 1
                save_meta(meta)
                st.success(f"Saved {added} file(s) under {tag_subject}.")
                st.rerun()

# File browser grouped by subject
if not meta.get("files"):
    st.info("No study files yet. Upload notes above.")
else:
    files_by_subject = {}
    for item in meta["files"]:
        files_by_subject.setdefault(item["subject"], []).append(item)

    for subj in sorted(files_by_subject.keys(), key=lambda x: x.lower()):
        with st.expander(f"📚 {subj}", expanded=False):
            items = sorted(files_by_subject[subj], key=lambda x: x["name"].lower())
            for item in items:
                p = Path(item["path"])
                if not p.exists():
                    st.warning(f"Missing file: {item['name']}")
                    continue

                col1, col2, col3, col4 = st.columns([7, 1.5, 1.5, 1.5])
                with col1:
                    st.write(f"**{item['name']}**  \nUploaded: {item.get('uploaded_at','')}")
                with col2:
                    st.download_button(
                        "Download",
                        data=p.read_bytes(),
                        file_name=item["name"],
                        mime="application/octet-stream",
                        key=f"dl_{subj}_{item['name']}",
                    )
                with col3:
                    if st.button("Delete", key=f"del_{subj}_{item['name']}"):
                        p.unlink(missing_ok=True)
                        meta["files"] = [
                            x for x in meta["files"]
                            if not (x["subject"] == subj and x["name"] == item["name"])
                        ]
                        save_meta(meta)
                        # close preview if it was this file
                        if st.session_state.preview_path == str(p):
                            st.session_state.preview_path = ""
                            st.session_state.preview_name = ""
                            st.session_state.preview_subject = ""
                        st.rerun()
                with col4:
                    if st.button("Preview", key=f"prev_{subj}_{item['name']}"):
                        st.session_state.preview_path = str(p)
                        st.session_state.preview_name = item["name"]
                        st.session_state.preview_subject = subj
                        st.rerun()

# Big preview area (more accessible)
if st.session_state.preview_path:
    p = Path(st.session_state.preview_path)
    if p.exists():
        st.divider()
        st.subheader(f"🔎 Preview: {st.session_state.preview_subject} — {st.session_state.preview_name}")

        close_col, _ = st.columns([1, 4])
        with close_col:
            if st.button("❌ Close preview"):
                st.session_state.preview_path = ""
                st.session_state.preview_name = ""
                st.session_state.preview_subject = ""
                st.rerun()

        tab1, tab2 = st.tabs(["📄 View file", "📝 Extracted text"])

        with tab1:
            if p.suffix.lower() == ".pdf":
                # Embed PDF viewer (mobile-friendly)
                pdf_bytes = p.read_bytes()
                # Avoid embedding huge PDFs (can be slow)
                if len(pdf_bytes) > 8_000_000:
                    st.warning("PDF is large. Use Download for smoother viewing.")
                else:
                    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
                    pdf_display = f"""
                    <iframe
                        src="data:application/pdf;base64,{b64}"
                        width="100%"
                        height="600"
                        style="border: none;">
                    </iframe>
                    """
                    components.html(pdf_display, height=620)

            elif p.suffix.lower() == ".txt":
                st.text_area("Text file", p.read_text(errors="ignore"), height=380)
            else:
                # images
                try:
                    st.image(str(p), use_container_width=True)
                except Exception:
                    st.info("Preview not available. Use Download.")

        with tab2:
            if p.suffix.lower() in [".txt", ".pdf"]:
                txt = extract_text_from_path(p)
                if not txt.strip():
                    if p.suffix.lower() == ".pdf" and not PDF_OK:
                        st.warning("PDF text extraction needs pypdf in requirements.txt.")
                    else:
                        st.warning("No text extracted from this file.")
                else:
                    st.text_area("Extracted text", txt, height=380)
            else:
                st.info("Text extraction is available for TXT/PDF only.")


# -----------------------------
# Update Plan (gate)
# -----------------------------
st.divider()
if st.button("✅ Update Plan"):
    st.session_state.plan_ready = True

if not st.session_state.plan_ready:
    st.info("Tap **Update Plan** to compute schedule, timer logging, and AI assessments.")
    st.stop()

# Persist the latest edits (so timer/plan use updated table)
df = edited_df.copy()
st.session_state.subjects = df
save_df(df)

# -----------------------------
# Compute (priority)
# -----------------------------
df = st.session_state.subjects.copy()
df = df.dropna(subset=["Subject"])
df["Subject"] = df["Subject"].astype(str).str.strip()
df = df[df["Subject"] != ""]

if df.empty:
    st.warning("Add at least one subject.")
    st.stop()

today = date.today()
weekly_minutes = int(minutes_per_day * 7)

df["Difficulty (1-5)"] = pd.to_numeric(df["Difficulty (1-5)"], errors="coerce").fillna(0)
df["Minutes Done (this week)"] = pd.to_numeric(df["Minutes Done (this week)"], errors="coerce").fillna(0)

exam_ts = pd.to_datetime(df["Exam Date"], errors="coerce").fillna(pd.Timestamp(today))
df["Exam Date"] = exam_ts.dt.date

df["Days Left"] = exam_ts.dt.date.apply(lambda d: max(1, (d - today).days)).astype(int)
df["Urgency"] = 10 / df["Days Left"]
df["Priority"] = (df["Difficulty (1-5)"] * df["Urgency"]).fillna(0)

total_priority = float(df["Priority"].sum())
if total_priority <= 0:
    df["Minutes/Week (Suggested)"] = 0
else:
    df["Minutes/Week (Suggested)"] = ((df["Priority"] / total_priority) * weekly_minutes).fillna(0).round().astype(int)

df["Progress %"] = (df["Minutes Done (this week)"] / df["Minutes/Week (Suggested)"]).fillna(0).clip(0, 1)


# -----------------------------
# Focus Timer (auto-log)
# -----------------------------
st.subheader("⏱ Focus Timer (auto-logs minutes)")

subjects_list = df["Subject"].astype(str).tolist()

if not st.session_state.timer_running:
    if subjects_list:
        chosen = st.selectbox("Choose subject", subjects_list, key="timer_subject_select")
        mins = st.number_input(
            "Minutes (use 1–3 minutes for demo)",
            min_value=1, max_value=120, value=int(st.session_state.timer_minutes),
        )

        t1, t2 = st.columns(2)
        with t1:
            if st.button("▶ Start Timer"):
                st.session_state.timer_running = True
                st.session_state.timer_subject = str(chosen)
                st.session_state.timer_minutes = int(mins)
                st.session_state.timer_start = time.time()
                st.session_state.timer_end = st.session_state.timer_start + (int(mins) * 60)
                st.rerun()
        with t2:
            if st.button("🛑 Cancel"):
                st.session_state.timer_running = False
                st.rerun()
    else:
        st.info("Add at least 1 subject first.")
else:
    remaining = int(st.session_state.timer_end - time.time())
    planned = int(st.session_state.timer_minutes)
    subj = st.session_state.timer_subject

    elapsed_minutes = int((time.time() - st.session_state.timer_start) // 60)
    elapsed_minutes = max(0, min(planned, elapsed_minutes))

    b1, b2 = st.columns(2)
    with b1:
        if st.button("⏹ Stop & Log elapsed"):
            if elapsed_minutes > 0:
                base = load_df()
                base["Minutes Done (this week)"] = pd.to_numeric(base["Minutes Done (this week)"], errors="coerce").fillna(0)
                mask = base["Subject"].astype(str) == str(subj)
                if mask.any():
                    base.loc[mask, "Minutes Done (this week)"] += elapsed_minutes
                    save_df(base)
                    st.session_state.subjects = base
            st.session_state.timer_running = False
            st.rerun()

    with b2:
        if st.button("🛑 Cancel (no log)"):
            st.session_state.timer_running = False
            st.rerun()

    if remaining <= 0:
        base = load_df()
        base["Minutes Done (this week)"] = pd.to_numeric(base["Minutes Done (this week)"], errors="coerce").fillna(0)
        mask = base["Subject"].astype(str) == str(subj)
        if mask.any():
            base.loc[mask, "Minutes Done (this week)"] += planned
            save_df(base)
            st.session_state.subjects = base
        st.session_state.timer_running = False
        st.success(f"✅ Logged {planned} minutes to {subj}")
        st.rerun()
    else:
        st.info(f"Studying **{subj}** — {remaining//60:02d}:{remaining%60:02d} (elapsed: {elapsed_minutes} min)")
        time.sleep(1)
        st.rerun()

# Recompute df after timer updates
df = load_df()
df["Exam Date"] = pd.to_datetime(df["Exam Date"], errors="coerce").dt.date
df = df.dropna(subset=["Subject"])
df["Subject"] = df["Subject"].astype(str).str.strip()
df = df[df["Subject"] != ""]
df["Difficulty (1-5)"] = pd.to_numeric(df["Difficulty (1-5)"], errors="coerce").fillna(0)
df["Minutes Done (this week)"] = pd.to_numeric(df["Minutes Done (this week)"], errors="coerce").fillna(0)

exam_ts = pd.to_datetime(df["Exam Date"], errors="coerce").fillna(pd.Timestamp(today))
df["Exam Date"] = exam_ts.dt.date
df["Days Left"] = exam_ts.dt.date.apply(lambda d: max(1, (d - today).days)).astype(int)
df["Urgency"] = 10 / df["Days Left"]
df["Priority"] = (df["Difficulty (1-5)"] * df["Urgency"]).fillna(0)
total_priority = float(df["Priority"].sum())
if total_priority <= 0:
    df["Minutes/Week (Suggested)"] = 0
else:
    df["Minutes/Week (Suggested)"] = ((df["Priority"] / total_priority) * weekly_minutes).fillna(0).round().astype(int)
df["Progress %"] = (df["Minutes Done (this week)"] / df["Minutes/Week (Suggested)"]).fillna(0).clip(0, 1)


# -----------------------------
# AI Assessment (Groq)
# -----------------------------
st.subheader("🧠 Assessment Generator (AI from your notes)")

groq_key = None
if "GROQ_API_KEY" in st.secrets:
    groq_key = st.secrets["GROQ_API_KEY"]
else:
    groq_key = os.environ.get("GROQ_API_KEY")

ai_available = bool(groq_key) and GROQ_OK

if not ai_available:
    st.warning("AI assessment is OFF. Add GROQ_API_KEY in Streamlit Secrets and include 'groq' in requirements.txt.")
else:
    client = Groq(api_key=groq_key)

    if not subjects_clean:
        st.info("Add subjects first and upload notes to generate assessments.")
    else:
        assess_subject = st.selectbox("Subject", subjects_clean, key="assess_subject_groq")

        # gather notes files for this subject
        meta = load_meta()
        subject_paths = []
        for item in meta.get("files", []):
            if item.get("subject") == assess_subject:
                p = Path(item.get("path", ""))
                if p.exists() and p.suffix.lower() in [".txt", ".pdf"]:
                    subject_paths.append(p)

        if not subject_paths:
            st.warning("Upload at least one TXT or PDF notes file for this subject.")
        else:
            chosen_file = st.selectbox("Notes file", [p.name for p in subject_paths], key="assess_file_groq")
            p = next(x for x in subject_paths if x.name == chosen_file)

            notes_text = normalize_spaces(extract_text_from_path(p))
            if not notes_text:
                st.warning("No text extracted from this file. Try TXT or a different PDF.")
            else:
                n_items = st.slider("Questions per type", 3, 10, 5)
                difficulty = st.select_slider("Difficulty", options=["easy", "medium", "hard"], value="medium")

                with st.expander("Preview notes used for AI (first 1200 chars)", expanded=False):
                    st.text_area("Notes", notes_text[:1200], height=200)

                if st.button("✅ Generate ALL (MCQ + Identification + Fill + Short Answer)"):
                    notes_limited = notes_text[:9000]

                    prompt = f"""
Return ONLY valid JSON with keys: mcq, identification, fill_blanks, short_answer.
Each item must include: question, answer_key.
MCQ must include choices A,B,C,D and answer_key must be one of A/B/C/D.
Fill blanks must contain exactly ONE blank '_____' in the question.

Subject: {assess_subject}
Difficulty: {difficulty}
Items per type: {n_items}

NOTES:
{notes_limited}
"""

                    try:
                        chat = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": "You are a strict Grade 12 STEM teacher. Use ONLY the notes given."},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=0.2,
                        )
                        raw = (chat.choices[0].message.content or "").strip()
                    except Exception as e:
                        st.error("Groq request failed. Check GROQ_API_KEY in Secrets.")
                        st.stop()

                    try:
                        data = json.loads(raw)
                    except Exception:
                        st.error("AI returned non-JSON. Try again or use a shorter/cleaner notes file.")
                        st.text_area("Raw output", raw, height=250)
                        st.stop()

                    def show_section(title: str, items: list):
                        st.markdown(f"### {title}")
                        if not items:
                            st.info("No items generated.")
                            return
                        out_df = pd.DataFrame(items)
                        st.dataframe(out_df, use_container_width=True, hide_index=True)
                        st.download_button(
                            f"⬇️ Download {title} (CSV)",
                            data=out_df.to_csv(index=False).encode("utf-8"),
                            file_name=f"{safe_code}_{assess_subject}_{title.replace(' ','_').lower()}.csv",
                            mime="text/csv",
                        )

                    show_section("Multiple Choice", data.get("mcq", []))
                    show_section("Identification", data.get("identification", []))
                    show_section("Fill in the Blanks", data.get("fill_blanks", []))
                    show_section("Short Answer", data.get("short_answer", []))


# -----------------------------
# Dashboard + Results + Plan
# -----------------------------
st.subheader("2) Dashboard")
d1, d2 = st.columns(2)
d3, d4 = st.columns(2)

most_urgent = df.sort_values("Days Left").iloc[0]["Subject"]
next_exam = df["Exam Date"].min()

d1.metric("Minutes/day", f"{minutes_per_day}")
d2.metric("Weekly minutes", f"{weekly_minutes}")
d3.metric("Most urgent", str(most_urgent))
d4.metric("Next exam", nice_date(next_exam))

st.subheader("3) Results")
show_cols = ["Subject", "Difficulty (1-5)", "Exam Date", "Days Left", "Minutes/Week (Suggested)", "Minutes Done (this week)"]
st.dataframe(
    df[show_cols].sort_values("Minutes/Week (Suggested)", ascending=False),
    use_container_width=True,
    hide_index=True,
    column_config={"Exam Date": st.column_config.DateColumn("Exam Date", format="MMM DD, YYYY")},
)

st.subheader("4) Progress")
for _, row in df.sort_values("Minutes/Week (Suggested)", ascending=False).iterrows():
    done = int(row["Minutes Done (this week)"])
    target = int(row["Minutes/Week (Suggested)"])
    st.write(f"**{row['Subject']}** — {done}/{target} min")
    st.progress(float(row["Progress %"]))

st.subheader("5) Study Plan")

ranked = df.sort_values("Minutes/Week (Suggested)", ascending=False).reset_index(drop=True)
top = ranked.head(min(3, len(ranked))).copy()

if len(top) == 1:
    split = [1.0]
elif len(top) == 2:
    split = [0.6, 0.4]
else:
    split = [0.5, 0.3, 0.2]

plan_rows = []
for i in range(days_to_plan):
    day_label = (pd.Timestamp(today) + pd.Timedelta(days=i)).strftime("%b %d, %Y")
    for idx in range(len(top)):
        plan_rows.append(
            {"Day": day_label, "Subject": top.loc[idx, "Subject"], "Minutes": int(round(minutes_per_day * split[idx]))}
        )

st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)

st.caption("✅ Preview is now a big section below the file list (easier on mobile).")
