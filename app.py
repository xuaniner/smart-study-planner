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
    user_code = st.text_input("User code", value="demo").strip().lower()

safe_code = "".join(ch for ch in user_code if ch.isalnum() or ch in ["_", "-"])
if not safe_code:
    safe_code = "demo"

DATA_PATH = Path(f"data_{safe_code}.csv")

# -----------------------------
# Helpers
# -----------------------------
def default_df():
    today_ = date.today()
    return pd.DataFrame({
        "Subject": ["Physics", "Gen Math", "Biology"],
        "Difficulty (1-5)": [5, 4, 3],
        "Exam Date": [
            today_ + timedelta(days=10),
            today_ + timedelta(days=7),
            today_ + timedelta(days=14),
        ],
        "Minutes Done (this week)": [0, 0, 0],
    })

def load_df():
    if DATA_PATH.exists():
        try:
            df_ = pd.read_csv(DATA_PATH)
            df_["Exam Date"] = pd.to_datetime(df_["Exam Date"], errors="coerce").dt.date
            return df_
        except Exception:
            return default_df()
    return default_df()

def save_df(df_):
    out = df_.copy()
    out["Exam Date"] = pd.to_datetime(out["Exam Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out.to_csv(DATA_PATH, index=False)

def nice_date(d):
    try:
        return pd.to_datetime(d).strftime("%b %d, %Y")
    except Exception:
        return ""

# -----------------------------
# Session Init
# -----------------------------
if "active_user_code" not in st.session_state or st.session_state.active_user_code != safe_code:
    st.session_state.active_user_code = safe_code
    st.session_state.subjects = load_df()

if "plan_ready" not in st.session_state:
    st.session_state.plan_ready = False

# Timer state init
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

# -----------------------------
# Settings
# -----------------------------
with st.expander("⚙️ Settings", expanded=False):
    minutes_per_day = st.number_input("Minutes available per day", 30, 600, 180, 10)
    days_to_plan = st.slider("Plan length (days)", 3, 14, 7)

# -----------------------------
# Add / Edit Subjects
# -----------------------------
st.subheader("1) Subjects")

with st.expander("➕ Add Subject", expanded=False):
    new_sub = st.text_input("Subject name")
    new_diff = st.slider("Difficulty (1–5)", 1, 5, 3)
    new_exam = st.date_input("Exam date", value=date.today() + timedelta(days=7))

    if st.button("Add"):
        if new_sub.strip():
            df_add = st.session_state.subjects.copy()
            df_add.loc[len(df_add)] = [new_sub.strip(), int(new_diff), new_exam, 0]
            st.session_state.subjects = df_add
            save_df(df_add)
            st.success("Subject added!")

edited_df = st.data_editor(
    st.session_state.subjects,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Exam Date": st.column_config.DateColumn("Exam Date", format="MMM DD, YYYY"),
        "Difficulty (1-5)": st.column_config.NumberColumn("Difficulty (1-5)", min_value=1, max_value=5, step=1),
        "Minutes Done (this week)": st.column_config.NumberColumn("Minutes Done (this week)", min_value=0, step=10),
    },
)

colA, colB = st.columns(2)
with colA:
    if st.button("💾 Save Changes"):
        st.session_state.subjects = edited_df.copy()
        save_df(edited_df)
        st.success("Saved!")
with colB:
    if st.button("🗑 Reset to default"):
        st.session_state.subjects = default_df()
        save_df(st.session_state.subjects)
        st.success("Reset done!")

# -----------------------------
# Update Plan gate (doesn't block timer reruns)
# -----------------------------
st.divider()
if st.button("✅ Update Plan"):
    st.session_state.plan_ready = True

if not st.session_state.plan_ready:
    st.info("Tap **Update Plan** to compute the schedule.")
    st.stop()

# -----------------------------
# Use latest on-screen edits AND persist them
# -----------------------------
df = edited_df.copy()
st.session_state.subjects = df
save_df(df)

# IMPORTANT: Always compute from saved state (so timer-logged minutes are included)
df = st.session_state.subjects.copy()

# -----------------------------
# Compute Logic (robust)
# -----------------------------
df = df.dropna(subset=["Subject"])
df["Subject"] = df["Subject"].astype(str).str.strip()
df = df[df["Subject"] != ""]

if df.empty:
    st.warning("Add at least one subject.")
    st.stop()

# Ensure numeric + handle None
df["Difficulty (1-5)"] = pd.to_numeric(df["Difficulty (1-5)"], errors="coerce").fillna(0)
df["Minutes Done (this week)"] = pd.to_numeric(df["Minutes Done (this week)"], errors="coerce").fillna(0)

today = date.today()
weekly_minutes = int(minutes_per_day * 7)

# Dates
exam_ts = pd.to_datetime(df["Exam Date"], errors="coerce").fillna(pd.Timestamp(today))
df["Exam Date"] = exam_ts.dt.date

# Priority model
df["Days Left"] = exam_ts.dt.date.apply(lambda d: max(1, (d - today).days)).astype(int)
df["Urgency"] = 10 / df["Days Left"]
df["Priority"] = (df["Difficulty (1-5)"] * df["Urgency"]).fillna(0)

total_priority = float(df["Priority"].sum())

if total_priority <= 0:
    df["Minutes/Week (Suggested)"] = 0
else:
    df["Minutes/Week (Suggested)"] = (
        (df["Priority"] / total_priority) * weekly_minutes
    ).fillna(0).round().astype(int)

df["Progress %"] = (
    (df["Minutes Done (this week)"] / df["Minutes/Week (Suggested)"])
    .fillna(0).clip(0, 1)
)

# -----------------------------
# Focus Timer (auto-log minutes; stop doesn't break update)
# -----------------------------
st.subheader("⏱ Focus Timer")

subjects_list = df["Subject"].astype(str).tolist()

if not st.session_state.timer_running:
    if subjects_list:
        chosen = st.selectbox("Choose subject", subjects_list)
        mins = st.number_input(
            "Minutes (Tip: set 1–3 minutes for demo)",
            min_value=1, max_value=120, value=int(st.session_state.timer_minutes)
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
            if st.button("🧹 Clear timer"):
                st.session_state.timer_subject = ""
                st.session_state.timer_minutes = 25
                st.session_state.timer_start = 0.0
                st.session_state.timer_end = 0.0
                st.rerun()
    else:
        st.info("Add at least 1 subject first.")
else:
    remaining = int(st.session_state.timer_end - time.time())
    planned = int(st.session_state.timer_minutes)
    subj = st.session_state.timer_subject

    elapsed_minutes = int((time.time() - st.session_state.timer_start) // 60)
    elapsed_minutes = max(0, min(planned, elapsed_minutes))

    b1, b2, b3 = st.columns(3)

    with b1:
        if st.button("⏹ Stop & Log elapsed"):
            if elapsed_minutes > 0:
                base = st.session_state.subjects.copy()
                base["Minutes Done (this week)"] = pd.to_numeric(
                    base["Minutes Done (this week)"], errors="coerce"
                ).fillna(0)

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

    with b3:
        if st.button("🔄 Refresh"):
            st.rerun()

    if remaining <= 0:
        base = st.session_state.subjects.copy()
        base["Minutes Done (this week)"] = pd.to_numeric(
            base["Minutes Done (this week)"], errors="coerce"
        ).fillna(0)

        mask = base["Subject"].astype(str) == str(subj)
        if mask.any():
            base.loc[mask, "Minutes Done (this week)"] += planned
            st.session_state.subjects = base
            save_df(base)

        st.session_state.timer_running = False
        st.success(f"✅ Logged {planned} minutes to {subj}")
        st.rerun()
    else:
        mm = remaining // 60
        ss = remaining % 60
        st.info(f"Studying **{subj}** — Time left: **{mm:02d}:{ss:02d}** (elapsed: {elapsed_minutes} min)")
        time.sleep(1)
        st.rerun()

# -----------------------------
# Dashboard
# -----------------------------
st.subheader("2) Dashboard")

c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

most_urgent_subject = df.sort_values("Days Left").iloc[0]["Subject"]
next_exam_date = df["Exam Date"].min()

c1.metric("Minutes/day", minutes_per_day)
c2.metric("Weekly minutes", weekly_minutes)
c3.metric("Most urgent subject", str(most_urgent_subject))
c4.metric("Next exam date", nice_date(next_exam_date))

# -----------------------------
# Results
# -----------------------------
st.subheader("3) Results (auto-calculated)")

show_cols = [
    "Subject",
    "Difficulty (1-5)",
    "Exam Date",
    "Days Left",
    "Minutes/Week (Suggested)",
    "Minutes Done (this week)",
]

st.dataframe(
    df[show_cols].sort_values("Minutes/Week (Suggested)", ascending=False),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Exam Date": st.column_config.DateColumn("Exam Date", format="MMM DD, YYYY"),
    },
)

# -----------------------------
# Progress
# -----------------------------
st.subheader("4) Progress")

for _, row in df.sort_values("Minutes/Week (Suggested)", ascending=False).iterrows():
    done = int(row["Minutes Done (this week)"])
    target = int(row["Minutes/Week (Suggested)"])
    st.write(f"**{row['Subject']}** — {done}/{target} min")
    st.progress(float(row["Progress %"]))

# -----------------------------
# Study Plan
# -----------------------------
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

plan = pd.DataFrame(plan_rows)
st.dataframe(plan, use_container_width=True, hide_index=True)

st.caption("✅ Data saved per user code. Timer logs minutes automatically.")
