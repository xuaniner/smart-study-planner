import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pathlib import Path

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Smart Study Planner", page_icon="📚", layout="wide")

st.title("📚 Smart Study Planner")
st.caption("Portfolio Project • Grade 12 STEM • Priority = Difficulty × Urgency")

# -----------------------------
# Profile (Per-user storage)
# -----------------------------
with st.expander("👤 Profile", expanded=False):
    user_code = st.text_input(
        "User code (anything you choose, e.g. jhune, demo1)",
        value="demo"
    ).strip().lower()

# Make it filename-safe
safe_code = "".join(ch for ch in user_code if ch.isalnum() or ch in ["_", "-"])
if not safe_code:
    safe_code = "demo"

DATA_PATH = Path(f"data_{safe_code}.csv")

# -----------------------------
# Helpers: Load / Save (Persistent memory)
# -----------------------------
def default_df() -> pd.DataFrame:
    today = date.today()
    return pd.DataFrame(
        {
            "Subject": ["Physics", "Gen Math", "Biology"],
            "Difficulty (1-5)": [5, 4, 3],
            "Exam Date": [
                today + timedelta(days=10),
                today + timedelta(days=7),
                today + timedelta(days=14),
            ],
            "Minutes Done (this week)": [0, 0, 0],
        }
    )

def load_df() -> pd.DataFrame:
    if DATA_PATH.exists():
        try:
            df = pd.read_csv(DATA_PATH)
            needed = ["Subject", "Difficulty (1-5)", "Exam Date", "Minutes Done (this week)"]
            for col in needed:
                if col not in df.columns:
                    raise ValueError("Missing columns in saved file.")
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

# -----------------------------
# Session state init (reload when user_code changes)
# -----------------------------
if "active_user_code" not in st.session_state or st.session_state.active_user_code != safe_code:
    st.session_state.active_user_code = safe_code
    st.session_state.subjects = load_df()

# -----------------------------
# Mobile-friendly Settings (no sidebar)
# -----------------------------
with st.expander("⚙️ Settings", expanded=False):
    minutes_per_day = st.number_input("Minutes available per day", 30, 600, 180, 10)
    days_to_plan = st.slider("Plan length (days)", 3, 14, 7)

# -----------------------------
# Add subject (mobile-friendly)
# -----------------------------
st.subheader("1) Add / Edit Subjects")

with st.expander("➕ Add a subject", expanded=False):
    new_subject = st.text_input("Subject name", placeholder="e.g., Statistics")
    new_diff = st.slider("Difficulty (1–5)", 1, 5, 3)
    new_exam = st.date_input("Exam date", value=date.today() + timedelta(days=7))
    add_btn = st.button("Add subject")

    if add_btn:
        s = (new_subject or "").strip()
        if not s:
            st.warning("Please enter a subject name.")
        else:
            df_add = st.session_state.subjects.copy()
            df_add.loc[len(df_add)] = [s, int(new_diff), new_exam, 0]
            st.session_state.subjects = df_add
            save_df(df_add)
            st.success(f"Added: {s}")

# -----------------------------
# Editable table
# -----------------------------
st.write("You can also edit directly below (works best on laptop):")

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

# Buttons: Save / Reset
colA, colB, colC = st.columns([1, 1, 3])
with colA:
    save_now = st.button("💾 Save changes")
with colB:
    reset = st.button("🗑 Reset to default")

if reset:
    st.session_state.subjects = default_df()
    save_df(st.session_state.subjects)
    st.success("Reset done. Reloading is safe now.")

if save_now:
    st.session_state.subjects = edited_df.copy()
    save_df(st.session_state.subjects)
    st.success("Saved! Your data will stay even if you refresh.")

# -----------------------------
# Update button (prevents constant reruns feeling)
# -----------------------------
st.divider()
with st.form("update_form"):
    st.write("When ready, tap **Update plan** to compute the schedule.")
    update = st.form_submit_button("✅ Update plan")

if not update:
    st.stop()

# -----------------------------
# Compute (safe, robust)
# -----------------------------
# Always compute from what’s currently on the screen
df = edited_df.copy()

# Also update memory so it matches your latest edits
st.session_state.subjects = df
save_df(df)

df = df.dropna(subset=["Subject"])
df["Subject"] = df["Subject"].astype(str).str.strip()
df = df[df["Subject"] != ""]

if df.empty:
    st.warning("Add at least one subject.")
    st.stop()

today = date.today()
weekly_minutes = int(minutes_per_day * 7)

# Numeric conversions
df["Difficulty (1-5)"] = pd.to_numeric(df["Difficulty (1-5)"], errors="coerce").fillna(0)
df["Minutes Done (this week)"] = pd.to_numeric(df["Minutes Done (this week)"], errors="coerce").fillna(0)

# Date conversion (handle blanks safely)
exam_ts = pd.to_datetime(df["Exam Date"], errors="coerce").fillna(pd.Timestamp(today))
df["Exam Date"] = exam_ts.dt.date

# Days left / urgency / priority
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
# Dashboard (compact for mobile)
# -----------------------------
st.subheader("2) Dashboard")

c1, c2 = st.columns(2)
c3, c4 = st.columns(2)

most_urgent_subject = df.sort_values("Days Left").iloc[0]["Subject"]
next_exam_date = df["Exam Date"].min()

c1.metric("Minutes/day", f"{minutes_per_day}")
c2.metric("Weekly minutes", f"{weekly_minutes}")
c3.metric("Most urgent subject", str(most_urgent_subject))
c4.metric("Next exam date", nice_date(next_exam_date))

# -----------------------------
# Results table (formatted dates)
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
# Progress bars
# -----------------------------
st.subheader("4) Progress")
for _, row in df.sort_values("Minutes/Week (Suggested)", ascending=False).iterrows():
    done = int(row["Minutes Done (this week)"])
    target = int(row["Minutes/Week (Suggested)"])
    st.write(f"**{row['Subject']}** — {done}/{target} min")
    st.progress(float(row["Progress %"]))

# -----------------------------
# Plan (7–14 days)
# -----------------------------
st.subheader(f"5) {days_to_plan}-Day Plan")

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
            {
                "Day": day_label,
                "Subject": top.loc[idx, "Subject"],
                "Minutes": int(round(minutes_per_day * split[idx])),
            }
        )

plan = pd.DataFrame(plan_rows)
st.dataframe(plan, use_container_width=True, hide_index=True)

st.caption("✅ Data is saved per User Code. Different codes have different saved subjects.")
