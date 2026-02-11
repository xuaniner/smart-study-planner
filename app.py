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

# Convert to numeric safely
df["Priority"] = pd.to_numeric(df["Priority"], errors="coerce")
df["Minutes Done (this week)"] = pd.to_numeric(df["Minutes Done (this week)"], errors="coerce")

total_priority = df["Priority"].sum()

if total_priority == 0:
    df["Minutes/Week (Suggested)"] = 0
else:
    df["Minutes/Week (Suggested)"] = (
        df["Priority"] / total_priority * weekly_minutes
    ).fillna(0).round().astype(int)
