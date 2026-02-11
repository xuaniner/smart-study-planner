import streamlit as st
import pandas as pd
from datetime import date

st.set_page_config(page_title="Smart Study Planner", page_icon="📚", layout="wide")

st.title("📚 Smart Study Planner Dashboard")
st.caption("Portfolio Project • Grade 12 STEM • Priority = Difficulty × Urgency")

# Sidebar inputs
st.sidebar.header("Settings")
minutes_per_day = st.sidebar.number_input("Minutes available per day", min_value=30, max_value=600, value=180, step=10)
days_to_plan = st.sidebar.slider("Plan length (days)", min_value=3, max_value=14, value=7)

st.subheader("1) Enter your subjects")

default = pd.DataFrame(
    {
        "Subject": ["Physics", "Gen Math", "Biology"],
        "Difficulty (1-5)": [5, 4, 3],
        "Exam Date": [date.today().replace(day=min(date.today().day + 10, 28)),
                      date.today().replace(day=min(date.today().day + 7, 28)),
                      date.today().replace(day=min(date.today().day + 14, 28))],
        "Minutes Done (this week)": [0, 0, 0],
    }
)

df = st.data_editor(
    default,
    num_rows="dynamic",
    use_container_width=True
)

# Clean / validate
df = df.dropna(subset=["Subject"])
if df.empty:
    st.warning("Add at least one subject.")
    st.stop()

today = date.today()

# Compute Days Left (min 1), Urgency, Priority
df["Days Left"] = df["Exam Date"].apply(lambda d: max(1, (d - today).days))
df["Urgency"] = 10 / df["Days Left"]
df["Priority"] = df["Difficulty (1-5)"] * df["Urgency"]

# Weekly allocation
weekly_minutes = minutes_per_day * 7
total_priority = df["Priority"].sum()
df["Minutes/Week (Suggested)"] = (df["Priority"] / total_priority * weekly_minutes).round().astype(int)

# Progress %
df["Progress %"] = (df["Minutes Done (this week)"] / df["Minutes/Week (Suggested)"]).fillna(0).clip(0, 1)

# Dashboard cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Minutes/day", f"{minutes_per_day}")
c2.metric("Weekly minutes", f"{weekly_minutes}")
most_urgent = df.sort_values("Days Left").iloc[0]["Subject"]
next_exam = df["Exam Date"].min()
c3.metric("Most urgent subject", f"{most_urgent}")
c4.metric("Next exam date", f"{next_exam}")

st.divider()

st.subheader("2) Results (auto-calculated)")
st.dataframe(
    df.sort_values("Minutes/Week (Suggested)", ascending=False),
    use_container_width=True
)

st.subheader("3) Progress (wow factor)")
for _, row in df.sort_values("Minutes/Week (Suggested)", ascending=False).iterrows():
    st.write(f"**{row['Subject']}** — {row['Minutes Done (this week)']}/{row['Minutes/Week (Suggested)']} min")
    st.progress(float(row["Progress %"]))

st.divider()

st.subheader(f"4) {days_to_plan}-Day Plan (Top subjects each day)")

# Simple daily schedule: split daily minutes across top 3 subjects by suggested minutes/week
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
    day_label = (today + pd.Timedelta(days=i)).strftime("%a, %b %d")
    for idx in range(len(top)):
        plan_rows.append({
            "Day": day_label,
            "Subject": top.loc[idx, "Subject"],
            "Minutes": int(round(minutes_per_day * split[idx]))
        })

plan = pd.DataFrame(plan_rows)
st.dataframe(plan, use_container_width=True, hide_index=True)

st.caption("Tip: For higher marks, explain the algorithm and show how changing exam date or difficulty updates the schedule instantly.")
