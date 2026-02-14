import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import time
import json
import re

# ---------- Optional PDF extraction ----------
try:
    from pypdf import PdfReader
    PDF_OK = True
except Exception:
    PDF_OK = False

# ---------- Optional AI ----------
AI_OK = False
try:
    from openai import OpenAI
    AI_OK = True
except Exception:
    AI_OK = False

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Smart Study Planner", page_icon="📚", layout="wide")
st.title("📚 Smart Study Planner")
st.caption("Planner + Timer + Study Files + AI Assessments")

# -------------------------------------------------
# PROFILE (per-user)
# -------------------------------------------------
with st.expander("👤 Profile", expanded=False):
    user_code = st.text_input("User code", value="demo").strip().lower()

safe_code = "".join(ch for ch in user_code if ch.isalnum() or ch in ["_", "-"]) or "demo"

DATA_PATH = Path(f"data_{safe_code}.csv")
FILES_DIR = Path(f"files_{safe_code}")
FILES_DIR.mkdir(exist_ok=True)
META_PATH = FILES_DIR / "files_meta.json"

# -------------------------------------------------
# HELPERS: Data save/load
# -------------------------------------------------
def default_df() -> pd.DataFrame:
    t = date.today()
    return pd.DataFrame({
        "Subject": ["Physics", "Gen Math", "Biology"],
        "Difficulty (1-5)": [5, 4, 3],
        "Exam Date": [t + timedelta(days=10), t + timedelta(days=7), t + timedelta(days=14)],
        "Minutes Done (this week)": [0, 0, 0],
    })

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

# -------------------------------------------------
# HELPERS: Files metadata
# -------------------------------------------------
def load_meta() -> dict:
    if META_PATH.exists():
        try:
            return json.loads(META_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {"files": []}
    return {"files": []}

def save_meta(meta: dict) -> None:
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

def sanitize_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_")

# -------------------------------------------------
# HELPERS: Text extraction for assessments
# -------------------------------------------------
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
    return ""  # images not supported without OCR (kept simple)

def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -------------------------------------------------
# SESSION INIT
# -------------------------------------------------
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

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
with st.expander("⚙️ Settings", expanded=False):
    minutes_per_day = st.number_input("Minutes available per day", 30, 600, 180, 10)
    days_to_plan = st.slider("Plan length (days)", 3, 14, 7)

# -------------------------------------------------
# SUBJECTS
# -------------------------------------------------
st.subheader("1) Subjects")

with st.expander("➕ Add subject", expanded=False):
    new_subject = st.text_input("Subject name", placeholder="e.g., Statistics")
    new_diff = st.slider("Difficulty (1–5)", 1, 5, 3)
    new_exam = st.date_input("Exam date", value=date.today() + timedelta(days=7))
    if st.button("Add subject"):
        s = (new_subject or "").strip()
        if not s:
            st.warning("Enter a subject name.")
        else:
            df_add = st.session_state.subjects.copy()
            df_add.loc[len(df_add)] = [s, int(new_diff), new_exam, 0]
            st.session_state.subjects = df_add
            save_df(df_add)
            st.success(f"Added {s}")
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

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    if st.button("💾 Save changes"):
        st.session_state.subjects = edited_df.copy()
        save_df(st.session_state.subjects)
        st.success("Saved!")
with c2:
    if st.button("🗑 Reset"):
        st.session_state.subjects = default_df()
        save_df(st.session_state.subjects)
        st.success("Reset done!")
        st.rerun()
with c3:
    st.caption("Dates are displayed like Mar 23, 2026.")

subjects_clean = (
    pd.Series(st.session_state.subjects["Subject"])
    .dropna().astype(str).str.strip()
)
subjects_clean = subjects_clean[subjects_clean != ""].tolist()

# -------------------------------------------------
# STUDY FILES (tagged by subject)
# -------------------------------------------------
st.subheader("📁 Study Files (tagged by subject)")

meta = load_meta()

with st.expander("Upload study files", expanded=False):
    if not subjects_clean:
        st.warning("Add at least 1 subject first.")
    else:
        tag_subject = st.selectbox("Which subject is this file for?", subjects_clean, key="tag_subject")
        uploads = st.file_uploader(
            "Upload notes (TXT / PDF / images)",
            type=["txt", "pdf", "png", "jpg", "jpeg"],
            accept_multiple_files=True
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
                    meta["files"].append({
                        "subject": tag_subject,
                        "name": fname,
                        "path": str(path),
                        "uploaded_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    })
                    added += 1
                save_meta(meta)
                st.success(f"Saved {added} file(s) under {tag_subject}.")
                st.rerun()

# File browser
if not meta.get("files"):
    st.info("No study files yet.")
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
                    st.warning(f"Missing: {item['name']}")
                    continue

                a, b, c, d = st.columns([6, 2, 2, 2])
                with a:
                    st.write(f"**{item['name']}**  \nUploaded: {item.get('uploaded_at','')}")
                with b:
                    st.download_button(
                        "Download",
                        data=p.read_bytes(),
                        file_name=item["name"],
                        mime="application/octet-stream",
                        key=f"dl_{subj}_{item['name']}"
                    )
                with c:
                    if st.button("Delete", key=f"del_{subj}_{item['name']}"):
                        p.unlink(missing_ok=True)
                        meta["files"] = [x for x in meta["files"] if not (x["subject"] == subj and x["name"] == item["name"])]
                        save_meta(meta)
                        st.rerun()
                with d:
                    if p.suffix.lower() in [".txt", ".pdf"]:
                        if st.button("Preview", key=f"prev_{subj}_{item['name']}"):
                            txt = extract_text_from_path(p)
                            if not txt.strip():
                                st.warning("No text extracted (try TXT or a different PDF).")
                            else:
                                st.text_area("Preview", txt[:2500], height=220)

# -------------------------------------------------
# UPDATE PLAN GATE
# -------------------------------------------------
st.divider()
if st.button("✅ Update Plan"):
    st.session_state.plan_ready = True

if not st.session_state.plan_ready:
    st.info("Tap **Update Plan** to compute schedule, timer, and assessments.")
    st.stop()

# Persist latest edits first
df = edited_df.copy()
st.session_state.subjects = df
save_df(df)

# Always compute from saved state
df = st.session_state.subjects.copy()

# -------------------------------------------------
# COMPUTE (priority model)
# -------------------------------------------------
df = df.dropna(subset=["Subject"])
df["Subject"] = df["Subject"].astype(str).str.strip()
df = df[df["Subject"] != ""]

if df.empty:
    st.warning("Add at least one subject.")
    st.stop()

df["Difficulty (1-5)"] = pd.to_numeric(df["Difficulty (1-5)"], errors="coerce").fillna(0)
df["Minutes Done (this week)"] = pd.to_numeric(df["Minutes Done (this week)"], errors="coerce").fillna(0)

today = date.today()
weekly_minutes = int(minutes_per_day * 7)

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

# -------------------------------------------------
# FOCUS TIMER (logs minutes)
# -------------------------------------------------
st.subheader("⏱ Focus Timer (auto-logs minutes)")

subjects_list = df["Subject"].astype(str).tolist()

if not st.session_state.timer_running:
    if subjects_list:
        chosen = st.selectbox("Choose subject", subjects_list, key="timer_subject_select")
        mins = st.number_input("Minutes", 1, 120, int(st.session_state.timer_minutes), help="Use 1–3 minutes for demo.")

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
            if st.button("🧹 Reset Timer"):
                st.session_state.timer_running = False
                st.session_state.timer_subject = ""
                st.session_state.timer_minutes = 25
                st.session_state.timer_start = 0.0
                st.session_state.timer_end = 0.0
                st.rerun()
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
                base = st.session_state.subjects.copy()
                base["Minutes Done (this week)"] = pd.to_numeric(base["Minutes Done (this week)"], errors="coerce").fillna(0)
                mask = base["Subject"].astype(str) == str(subj)
                if mask.any():
                    base.loc[mask, "Minutes Done (this week)"] += elapsed_minutes
                    st.session_state.subjects = base
                    save_df(base)
            st.session_state.timer_running = False
            st.rerun()
    with b2:
        if st.button("🛑 Cancel (no log)"):
            st.session_state.timer_running = False
            st.rerun()

    if remaining <= 0:
        base = st.session_state.subjects.copy()
        base["Minutes Done (this week)"] = pd.to_numeric(base["Minutes Done (this week)"], errors="coerce").fillna(0)
        mask = base["Subject"].astype(str) == str(subj)
        if mask.any():
            base.loc[mask, "Minutes Done (this week)"] += planned
            st.session_state.subjects = base
            save_df(base)
        st.session_state.timer_running = False
        st.success(f"✅ Logged {planned} minutes to {subj}")
        st.rerun()
    else:
        st.info(f"Studying **{subj}** — {remaining//60:02d}:{remaining%60:02d} (elapsed: {elapsed_minutes} min)")
        time.sleep(1)
        st.rerun()

# Refresh computed df after timer may change minutes
df = load_df()
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

# -------------------------------------------------
# AI ASSESSMENT GENERATOR (better UX)
# -------------------------------------------------
st.subheader("🧠 Assessment Generator (AI from your notes)")

api_key = None
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    # also support env var
    import os
    api_key = os.environ.get("OPENAI_API_KEY")

ai_available = bool(api_key) and AI_OK

if not ai_available:
    st.warning("AI assessment is OFF (missing OPENAI_API_KEY or openai library). You can still upload & preview notes.")
    st.caption("If you want AI assessments: add OPENAI_API_KEY in Streamlit Secrets and add 'openai' to requirements.txt.")
else:
    client = OpenAI(api_key=api_key)

    if not subjects_clean:
        st.info("Add subjects first.")
    else:
        assess_subject = st.selectbox("Subject", subjects_clean, key="assess_subject_ai")

        # collect notes for this subject
        meta = load_meta()
        subject_paths = []
        for item in meta.get("files", []):
            if item.get("subject") == assess_subject:
                p = Path(item.get("path", ""))
                if p.exists() and p.suffix.lower() in [".txt", ".pdf"]:
                    subject_paths.append(p)

        if not subject_paths:
            st.warning("Upload at least one TXT/PDF notes file for this subject.")
            if not PDF_OK:
                st.caption("PDF extraction needs pypdf.")
        else:
            chosen_file = st.selectbox("Notes file", [p.name for p in subject_paths], key="assess_file_ai")
            p = next(x for x in subject_paths if x.name == chosen_file)
            notes_text = clean_text(extract_text_from_path(p))

            if not notes_text:
                st.warning("No text extracted. Try TXT or a different PDF.")
            else:
                n_items = st.slider("How many questions per type?", 3, 10, 5)
                difficulty = st.select_slider("Difficulty", options=["easy", "medium", "hard"], value="medium")

                if st.button("✅ Generate (MCQ + Identification + Fill + Short Answer)"):
                    # keep input size reasonable
                    notes_text_limited = notes_text[:9000]

                    prompt = f"""
You are a strict high-school STEM teacher.
Create assessments ONLY from the notes below.
Return in clean JSON with keys: mcq, identification, fill_blanks, short_answer.
Each item must include: question, and answer_key.
MCQ must include choices A,B,C,D and answer_key = one of A/B/C/D.
Fill blanks must include the full sentence with ONE blank '_____'.

Subject: {assess_subject}
Difficulty: {difficulty}
Items per type: {n_items}

NOTES:
{notes_text_limited}
"""
                    resp = client.responses.create(
                        model="gpt-5.2",
                        input=prompt
                    )
                    raw = resp.output_text.strip()

                    # Try to parse JSON
                    try:
                        data = json.loads(raw)
                    except Exception:
                        st.error("AI returned non-JSON. Try again (or shorten notes).")
                        st.text_area("Raw output", raw, height=240)
                        st.stop()

                    # Display nicely
                    def show_section(title, items):
                        st.markdown(f"### {title}")
                        if not items:
                            st.info("No items generated.")
                            return
                        df_out = pd.DataFrame(items)
                        st.dataframe(df_out, use_container_width=True, hide_index=True)
                        st.download_button(
                            f"Download {title} (CSV)",
                            data=df_out.to_csv(index=False).encode("utf-8"),
                            file_name=f"{safe_code}_{assess_subject}_{title.replace(' ','_').lower()}.csv",
                            mime="text/csv"
                        )

                    show_section("Multiple Choice", data.get("mcq", []))
                    show_section("Identification", data.get("identification", []))
                    show_section("Fill in the Blanks", data.get("fill_blanks", []))
                    show_section("Short Answer", data.get("short_answer", []))

# -------------------------------------------------
# DASHBOARD + RESULTS
# -------------------------------------------------
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
        plan_rows.append({
            "Day": day_label,
            "Subject": top.loc[idx, "Subject"],
            "Minutes": int(round(minutes_per_day * split[idx])),
        })

st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)
st.caption("✅ Dates formatted, notes accessible per subject, AI assessment enabled when OPENAI_API_KEY is set.")
