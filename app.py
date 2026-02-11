import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import time

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Smart Study Planner", page_icon="📚", layout="wide")

st.title("📚 Smart Study Planner")
st.caption("Portfolio Project • Grade 12 STEM • Priority = Difficulty × Urgency")

# -----------------------------
# Profile (Per-user storage)
# -----------------------------
with st.expander("👤 Profile", expanded=False):
    user_code = st.text_input(
        "User code (choose anything like jhune, demo1)",
        value="demo"
    ).strip().lower()

safe_code = "".join(ch for ch in user_code if ch.isalnum() or ch in ["_", "-"])
if not safe_code:
    safe_code = "demo"

DATA_PATH = Path(f"data_{safe_code}.csv")

# -----------------------------
# Helpers
# -----------------------------
def default_df():
    today = date.today()
    return pd.DataFrame({
        "Subject": ["Physics", "Gen Math", "Biology"],
        "Difficulty (1-5)": [5, 4, 3],
        "Exam Date": [
            today + timedelta(days=10),
            today + timedelta(days=7),
            today + timedelta(days=14),
        ],
        "Minutes Done (this week)": [0, 0, 0],
    })

def load_df():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        df["Exam Date"] = pd.to_datetime(df["Exam Date"], errors="coerce").dt.date
        return df
    return default_df()

def save_df(df):
    out = df.copy()
    out["Exam Date"] = pd.to_datetime(out["Exam Date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(DATA_PATH, index=False)

def nice_date(d):
    return pd.to_datetime(d).strftime("%b %d, %Y")

# -----------------------------
# Session init
# -----------------------------
if "active_user_code" not in st.session_state or st.session_state.active_user_code != safe_code:
    st.session_state.active_user_code = safe_code
    st.session_state.subjects = load_df()

# -----------------------------
# Settings
# -----------------------------
with st.expander("⚙️ Settings", expanded=False):
    minutes_per_day = st.number_input("Minutes available per day", 30, 600, 180, 10)
    days_to_plan = st.slider("Plan length (days)", 3, 14, 7)

# -----------------------------
# Add Subject
# -----------------------------
st.subheader("1) Add / Edit Subjects")

with st.expander("➕ Add Subject", expanded=False):
    new_sub = st.text_input("Subject name")
    new_diff = st.slider("Difficulty (1–5)", 1, 5, 3)
    new_exam = st.date_input("Exam date", value=date.today() + timedelta(days=7))
    if st.button("Add"):
        if new_sub.strip():
            df_add = st.session_state.subjects.copy()
            df_add.loc[len(df_add)] = [new_sub.strip(), new_diff, new_exam, 0]
            st.session_state.subjects = df_add
            save_df(df_add)
            st.success("Subject added!")

edited_df = st.data_editor(
    st.session_state.subjects,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Exam Date": st.column_config.DateColumn("Exam Date", format="MMM DD, YYYY"),
    },
)

if st.button("💾 Save Changes"):
    st.session_state.subjects = edited_df.copy()
    save_df(edited_df)
    st.success("Saved!")

# -----------------------------
# Update Plan
# -----------------------------
st.divider()
if not st.button("✅ Update Plan"):
    st.stop()

df = edited_df.copy()
st.session_state.subjects = df
save_df(df)

df = df.dropna(subset=["Subject"])
df["Difficulty (1-5)"] = pd.to_numeric(df["Difficulty (1-5)"], errors="coerce").fillna(0)
df["Minutes Done (this week)"] = pd.to_numeric(df["Minutes Done (this week)"], errors="coerce").fillna(0)

today = date.today()
weekly_minutes = minutes_per_day * 7

exam_ts = pd.to_datetime(df["Exam Date"], errors="coerce").fillna(pd.Timestamp(today))
df["Exam Date"] = exam_ts.dt.date

df["Days Left"] = exam_ts.dt.date.apply(lambda d: max(1, (d - today).days))
df["Urgency"] = 10 / df["Days Left"]
df["Priority"] = df["Difficulty (1-5)"] * df["Urgency"]

total_priority = df["Priority"].sum()
if total_priority <= 0:
    df["Minutes/Week (Suggested)"] = 0
else:
    df["Minutes/Week (Suggested)"] = (df["Priority"] / total_priority * weekly_minutes).round()

df["Progress %"] = (df["Minutes Done (this week)"] / df["Minutes/Week (Suggested)"]).fillna(0).clip(0, 1)

# -----------------------------
# Focus Timer
# -----------------------------
st.subheader("⏱ Focus Timer")

if "timer_running" not in st.session_state:
    st.session_state.timer_running = False

if st.session_state.timer_running:
    remaining = int(st.session_state.timer_end - time.time())
    if remaining <= 0:
        subj = st.session_state.timer_subject
        mins = st.session_state.timer_minutes

        mask = st.session_state.subjects["Subject"] == subj
        st.session_state.subjects.loc[mask, "Minutes Done (this week)"] += mins
        save_df(st.session_state.subjects)

        st.session_state.timer_running = False
        st.success(f"Logged {mins} minutes to {subj}")
        st.experimental_rerun()
    else:
        mm = remaining // 60
        ss = remaining % 60
        st.info(f"Time left: {mm:02d}:{ss:02d}")
        time.sleep(1)
        st.experimental_rerun()

else:
    subject_list = df["Subject"].tolist()
    if subject_list:
        chosen = st.selectbox("Choose subject", subject_list)
        mins = st.number_input("Minutes", 1, 120, 25)
        if st.button("Start Timer"):
            st.session_state.timer_running = True
            st.session_state.timer_subject = chosen
            st.session_state.timer_minutes = mins
            st.session_state.timer_end = time.time() + mins * 60
            st.experimental_rerun()

# -----------------------------
# Dashboard
# -----------------------------
st.subheader("2) Dashboard")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("Minutes/day", minutes_per_day)
col2.metric("Weekly minutes", weekly_minutes)
col3.metric("Most urgent", df.sort_values("Days Left").iloc[0]["Subject"])
col4.metric("Next exam", nice_date(df["Exam Date"].min()))

# -----------------------------
# Results
# -----------------------------
st.subheader("3) Results")

st.dataframe(
    df.sort_values("Minutes/Week (Suggested)", ascending=False),
    use_container_width=True,
    column_config={
        "Exam Date": st.column_config.DateColumn("Exam Date", format="MMM DD, YYYY")
    }
)

# -----------------------------
# Progress
# -----------------------------
st.subheader("4) Progress")

for _, row in df.iterrows():
    st.write(f"{row['Subject']} — {int(row['Minutes Done (this week)'])}/{int(row['Minutes/Week (Suggested)'])}")
    st.progress(float(row["Progress %"]))

# -----------------------------
# Plan
# -----------------------------
st.subheader("5) Study Plan")

plan_rows = []
top = df.sort_values("Minutes/Week (Suggested)", ascending=False).head(3)

for i in range(days_to_plan):
    day_label = (pd.Timestamp(today) + pd.Timedelta(days=i)).strftime("%b %d, %Y")
    for _, row in top.iterrows():
        plan_rows.append({
            "Day": day_label,
            "Subject": row["Subject"],
            "Minutes": round(minutes_per_day / len(top))
        })

st.dataframe(pd.DataFrame(plan_rows), use_container_width=True)

st.caption("✅ Data saved per user code.")
