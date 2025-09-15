import streamlit as st
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Disable sidebar
st.set_page_config(page_title="Progress Dashboard", layout="wide", initial_sidebar_state="collapsed")

def reset_database():
    try:
        conn = sqlite3.connect('workout_progress.db')
        c = conn.cursor()
        c.execute('DROP TABLE IF EXISTS sessions')
        conn.commit()
        conn.close()
        conn = sqlite3.connect('workout_progress.db')
        c = conn.cursor()
        c.execute('''
            CREATE TABLE sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                exercise TEXT,
                correct_reps INTEGER,
                incorrect_reps INTEGER,
                total_reps INTEGER,
                correct_percentage REAL,
                session_duration REAL
            )
        ''')
        conn.commit()
        conn.close()
        print("Database reset successfully")
        st.session_state.reset_trigger = True
        st.success("Progress reset successfully!")
    except sqlite3.Error as e:
        print(f"Database reset error: {e}")
        st.error(f"Failed to reset progress: {e}")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'progress'
if 'reset_trigger' not in st.session_state:
    st.session_state.reset_trigger = False

# Redirect to main page
if st.session_state.page == 'main':
    print("Redirecting from progress.py to main.py")
    st.switch_page("main.py")

# Check if reset was triggered
if st.session_state.reset_trigger:
    st.session_state.reset_trigger = False
    st.rerun()

st.title("Progress Dashboard")

# Reset Progress button
if st.button("Reset Progress"):
    reset_database()

# Back to Home button
st.button("Back to Home", on_click=lambda: st.session_state.update({'page': 'main'}))

try:
    conn = sqlite3.connect('workout_progress.db')
    df = pd.read_sql_query("SELECT * FROM sessions", conn)
    conn.close()

    if df.empty:
        st.info("No workout sessions recorded yet.")
    else:
        st.subheader("Overall Progress")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sessions", len(df))
        col2.metric("Average Correct %", f"{df['correct_percentage'].mean():.1f}%")
        col3.metric("Total Reps", df['total_reps'].sum())

        st.subheader("Session History")
        st.dataframe(df[['date', 'exercise', 'correct_reps', 'incorrect_reps', 'correct_percentage', 'session_duration']])

        st.subheader("Progress Graphs")
        fig, ax = plt.subplots()
        df.plot(x='date', y=['correct_reps', 'incorrect_reps'], kind='bar', ax=ax, color=['green', 'red'])
        ax.set_title("Reps per Session")
        ax.set_xlabel("Date")
        ax.set_ylabel("Reps")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
except sqlite3.Error as e:
    st.error(f"Database error: {e}")