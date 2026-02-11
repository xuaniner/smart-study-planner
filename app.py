import streamlit as st
import pandas as pd
from datetime import date

# ---------- Page config ----------
st.set_page_config(page_title="Smart Study Planner", page_icon="📚", layout="wide")

st.title("📚 Smart Study Planner Dashboard")
st.caption("Portfolio Project • Grade 12 STEM • Priority = Difficulty × Urgency")

# ---------- Sidebar ----------
st.sidebar.header("Settings")
minutes_per_day = st.sidebar.number_input(
    "Minutes available per day", min_value=30, max_value=600, value=180, step=10
)
days_to_plan = st.sidebar.slider("Plan length (days)", min_value=3, max_value=14, value=7)

# ---------- Default data ----------
today = date.today()
default = pd.DataFrame(
    {
        "Subject": ["Physics", "Gen Math", "Biology"],
        "Difficulty (1-5)": [5, 4, 3],
        "Exam Date": [
            today.replace(day=min(today.day + 10, 28)),
            today.replace(day=min(today.day + 7, 28)),
            today.replace(day=min(today.day + 14, 28)),
        ],
        "Minutes Done (this week)": [0, 0, 0],
    }
)

st.subheader("1) Enter your subjects")
df = st.data_editor(
    default,
    num_rows="dynamic",
    use_container_width=True,
)

# ---------- Clean / validate ----------
df = df.dropna(subset=["Subject"])
df["Subject"] = df["Subject"].astype(str).str.strip()
df = df[df["Subject"] != ""]

if df.empty:
    st.warning("Add at least one subject.")
    st.stop()

# ---------- Type fixes (data_editor can output text/object types) ----------
df["Difficulty (1-5)"] = pd.to_numeric(df["Difficulty (1-5)"], errors="coerce").fillna(0)
df["Minutes Done (this week)"] = pd.to_numeric(df["Minutes Done (this week)"], errors="coerce").fillna(0)

# Convert Exam Date safely; blanks/invalid become NaT then set to today
exam_ts = pd.to_datetime(df["Exam Date"], errors="coerce").fillna(pd.Timestamp(today))
df["Exam Date"] = exam_ts.dt.date

# ---------- Compute scores ----------
# Days left (min 1)
df["Days Left"] = exam_ts.dt.date.apply(lambda d: max(1, (d - today).days)).astype(int)

# Urgency and Priority
df["Urgency"] = 10 / df["Days Left"]
df["Priority"] = (df["Difficulty (1-5)"] * df["Urgency"]).fillna(0)

# ---------- Allocate minutes/week ----------
weekly_minutes = minutes_per_day * 7
total_priority = float(df["Priority"].sum())

if total_priority <= 0:
    df["Minutes/Week (Suggested)"] = 0
else:
    df["Minutes/Week (Suggested)"] = (
        (df["Priority"] / total_priority) * weekly_minutes
    ).fillna(0).round().astype(int)

# Progress %
df["Progress %"] = (
    (df["Minutes Done (this week)"] / df["Minutes/Week (Suggested)"])
    .replace([pd.NA, pd.NaT], 0)
    .fillna(0)
    .clip(0, 1)
)

# ---------- Dashboard cards ----------
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Minutes/day", f"{minutes_per_day}")
c2.metric("Weekly minutes", f"{weekly_minutes}")

most_urgent_subject = df.sort_values("Days Left").iloc[0]["Subject"]
next_exam_date = df["Exam Date"].min()

c3.metric("Most urgent subject", str(most_urgent_subject))
c4.metric("Next exam date", str(next_exam_date))

# ---------- Results table ----------
st.subheader("2) Results (auto-calculated)")
show_cols = [
    "Subject",
    "Difficulty (1-5)",
    "Exam Date",
    "Days Left",
    "Urgency",
    "Priority",
    "Minutes/Week (Suggested)",
    "Minutes Done (this week)",
]
st.dataframe(
    df[show_cols].sort_values("Minutes/Week (Suggested)", ascending=False),
    use_container_width=True,
    hide_index=True,
)

# ---------- Progress (wow factor) ----------
st.subheader("3) Progress")
for _, row in df.sort_values("Minutes/Week (Suggested)", ascending=False).iterrows():
    done = int(row["Minutes Done (this week)"])
    target = int(row["Minutes/Week (Suggested)"])
    st.write(f"**{row['Subject']}** — {done}/{target} min")
    st.progress(float(row["Progress %"]))

# ---------- Plan ----------
st.divider()
st.subheader(f"4) {days_to_plan}-Day Plan (Top subjects each day)")

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
    day_label = (pd.Timestamp(today) + pd.Timedelta(days=i)).strftime("%a, %b %d")
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

st.caption("Tip: Try changing exam dates or difficulty—everything updates automatically.")
