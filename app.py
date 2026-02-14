import streamlit as st
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import time
import json
import re
from collections import Counter

# Optional PDF reader
try:
    from pypdf import PdfReader
    PDF_OK = True
except:
    PDF_OK = False

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Smart Study Planner", page_icon="📚", layout="wide")

st.title("📚 Smart Study Planner")
st.caption("Planner + Timer + Files + Assessments")

# -------------------------------------------------
# USER PROFILE
# -------------------------------------------------
with st.expander("👤 Profile", expanded=False):
    user_code = st.text_input("User code", value="demo").lower().strip()

safe_code = "".join(c for c in user_code if c.isalnum() or c in ["_", "-"])
if not safe_code:
    safe_code = "demo"

DATA_PATH = Path(f"data_{safe_code}.csv")
FILES_DIR = Path(f"files_{safe_code}")
FILES_DIR.mkdir(exist_ok=True)
META_PATH = FILES_DIR / "meta.json"

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def default_df():
    t = date.today()
    return pd.DataFrame({
        "Subject":["Physics","Math","Biology"],
        "Difficulty (1-5)":[5,4,3],
        "Exam Date":[t+timedelta(days=10),t+timedelta(days=7),t+timedelta(days=14)],
        "Minutes Done (this week)":[0,0,0]
    })

def load_df():
    if DATA_PATH.exists():
        df=pd.read_csv(DATA_PATH)
        df["Exam Date"]=pd.to_datetime(df["Exam Date"]).dt.date
        return df
    return default_df()

def save_df(df):
    out=df.copy()
    out["Exam Date"]=pd.to_datetime(out["Exam Date"]).dt.strftime("%Y-%m-%d")
    out.to_csv(DATA_PATH,index=False)

def load_meta():
    if META_PATH.exists():
        return json.loads(META_PATH.read_text())
    return {"files":[]}

def save_meta(m):
    META_PATH.write_text(json.dumps(m,indent=2))

# -------------------------------------------------
# SESSION INIT
# -------------------------------------------------
if "subjects" not in st.session_state:
    st.session_state.subjects=load_df()

if "timer_running" not in st.session_state:
    st.session_state.timer_running=False

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
with st.expander("⚙ Settings"):
    minutes_per_day=st.number_input("Minutes per day",30,600,180,10)
    days_to_plan=st.slider("Days",3,14,7)

# -------------------------------------------------
# SUBJECTS
# -------------------------------------------------
st.subheader("Subjects")

with st.expander("Add Subject"):
    s=st.text_input("Subject")
    d=st.slider("Difficulty",1,5,3)
    e=st.date_input("Exam",value=date.today()+timedelta(days=7))
    if st.button("Add Subject"):
        if s.strip():
            df=st.session_state.subjects
            df.loc[len(df)]=[s,d,e,0]
            st.session_state.subjects=df
            save_df(df)
            st.rerun()

edited_df=st.data_editor(st.session_state.subjects,use_container_width=True,num_rows="dynamic")

if st.button("Save Changes"):
    st.session_state.subjects=edited_df.copy()
    save_df(edited_df)
    st.success("Saved")

subjects=st.session_state.subjects["Subject"].dropna().astype(str).tolist()

# -------------------------------------------------
# FILES
# -------------------------------------------------
st.subheader("Study Files")

meta=load_meta()

if subjects:
    subj=st.selectbox("Subject for upload",subjects)
    files=st.file_uploader("Upload notes",accept_multiple_files=True)
    if st.button("Save Files"):
        for f in files:
            p=FILES_DIR/f.name
            p.write_bytes(f.getbuffer())
            meta["files"].append({"subject":subj,"name":f.name,"path":str(p)})
        save_meta(meta)
        st.success("Saved")
        st.rerun()

for f in meta["files"]:
    p=Path(f["path"])
    if p.exists():
        c1,c2=st.columns([4,1])
        with c1:
            st.write(f["subject"],"—",f["name"])
        with c2:
            if st.button("Delete",key=f["name"]):
                p.unlink(missing_ok=True)
                meta["files"].remove(f)
                save_meta(meta)
                st.rerun()

# -------------------------------------------------
# COMPUTE PLAN
# -------------------------------------------------
df=st.session_state.subjects.copy()
df["Difficulty (1-5)"]=pd.to_numeric(df["Difficulty (1-5)"],errors="coerce").fillna(0)
df["Minutes Done (this week)"]=pd.to_numeric(df["Minutes Done (this week)"],errors="coerce").fillna(0)

today=date.today()
weekly=minutes_per_day*7

dt=pd.to_datetime(df["Exam Date"],errors="coerce").fillna(pd.Timestamp(today))
df["Days Left"]=dt.dt.date.apply(lambda d:max(1,(d-today).days))
df["Priority"]=df["Difficulty (1-5)"]*(10/df["Days Left"])

tot=df["Priority"].sum()
df["Minutes/Week (Suggested)"]=0 if tot<=0 else (df["Priority"]/tot*weekly).round()
df["Progress %"]=(df["Minutes Done (this week)"]/df["Minutes/Week (Suggested)"]).fillna(0).clip(0,1)

# -------------------------------------------------
# TIMER
# -------------------------------------------------
st.subheader("Focus Timer")

if not st.session_state.timer_running and subjects:
    sub=st.selectbox("Study subject",subjects)
    mins=st.number_input("Minutes",1,120,25)
    if st.button("Start"):
        st.session_state.timer_running=True
        st.session_state.end=time.time()+mins*60
        st.session_state.sub=sub
        st.session_state.mins=mins
        st.rerun()

elif st.session_state.timer_running:
    remain=int(st.session_state.end-time.time())
    if remain<=0:
        mask=st.session_state.subjects["Subject"]==st.session_state.sub
        st.session_state.subjects.loc[mask,"Minutes Done (this week)"]+=st.session_state.mins
        save_df(st.session_state.subjects)
        st.session_state.timer_running=False
        st.success("Session logged")
        st.rerun()
    else:
        st.info(f"{remain//60:02d}:{remain%60:02d}")
        if st.button("Stop"):
            st.session_state.timer_running=False
            st.rerun()
        time.sleep(1)
        st.rerun()

# -------------------------------------------------
# DASHBOARD
# -------------------------------------------------
st.subheader("Dashboard")

c1,c2=st.columns(2)
c1.metric("Daily",minutes_per_day)
c2.metric("Weekly",weekly)

# -------------------------------------------------
# RESULTS
# -------------------------------------------------
st.subheader("Results")
st.dataframe(df,use_container_width=True)

# -------------------------------------------------
# PLAN
# -------------------------------------------------
st.subheader("Study Plan")

top=df.sort_values("Minutes/Week (Suggested)",ascending=False).head(3)
rows=[]
for i in range(days_to_plan):
    day=(pd.Timestamp(today)+pd.Timedelta(days=i)).strftime("%b %d")
    for _,r in top.iterrows():
        rows.append({"Day":day,"Subject":r["Subject"],"Minutes":round(minutes_per_day/len(top))})
st.dataframe(pd.DataFrame(rows),use_container_width=True)

# -------------------------------------------------
# ASSESSMENT GENERATOR
# -------------------------------------------------
st.subheader("Assessment Generator")

def text_from_file(p):
    if p.suffix==".txt":
        return p.read_text(errors="ignore")
    if p.suffix==".pdf" and PDF_OK:
        r=PdfReader(str(p))
        return "\n".join(page.extract_text() or "" for page in r.pages[:10])
    return ""

STOP=set("the and of to in a is that for on with as by at from".split())

def terms(txt,n=15):
    w=re.findall(r"[A-Za-z]{4,}",txt.lower())
    w=[x for x in w if x not in STOP]
    return [t for t,_ in Counter(w).most_common(n)]

if subjects:
    s=st.selectbox("Subject",subjects,key="assess")
    files=[Path(f["path"]) for f in meta["files"] if f["subject"]==s]
    files=[f for f in files if f.exists() and f.suffix in [".txt",".pdf"]]

    if files:
        f=st.selectbox("File",[x.name for x in files])
        path=[x for x in files if x.name==f][0]
        txt=text_from_file(path)

        if txt:
            t=terms(txt,15)
            if st.button("Generate Questions"):
                rows=[]
                for term in t:
                    rows.append({"Type":"Identification","Question":f"Define {term}"})
                    rows.append({"Type":"Short Answer","Question":f"Explain {term}"})
                    rows.append({"Type":"Fill Blank","Question":txt.replace(term,"_____ ",1)})
                st.dataframe(pd.DataFrame(rows),use_container_width=True)

st.caption("All data saved per user profile.")
