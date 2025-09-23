import os
os.environ['MEDIAPIPE_HOME'] = '/tmp'

import streamlit as st
import cv2
import numpy as np
from pose_detector import PoseDetector
from rep_counter import SquatCounter, BicepCurlCounter, OverheadPressCounter
from utils import draw_rep_count
import tempfile
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
import json
import re

def craft_rule_based_feedback(snapshot):
    try:
        exercise = snapshot.get("exercise")
        errors = snapshot.get("errors") or []
        targets = snapshot.get("targets") or {}
        def clamp_num(x):
            try:
                return round(float(x), 1)
            except Exception:
                return x

        if exercise == "Squat":
            knee_min = clamp_num(snapshot.get("knee_min"))
            knee_target = clamp_num(targets.get("knee_angle_down_correct"))
            back_avg = clamp_num(snapshot.get("back_avg"))
            back_max = clamp_num(targets.get("back_angle_max"))
            if isinstance(knee_min, (int, float)) and isinstance(knee_target, (int, float)) and knee_min > knee_target:
                return f"You are doing it wrong: squat deeper. Reach ~{knee_target}°; you hit {knee_min}°."
            if isinstance(back_avg, (int, float)) and isinstance(back_max, (int, float)) and back_avg >= back_max:
                return f"You are doing it wrong: keep back upright. Keep <{back_max}°; you averaged {back_avg}°."
            if errors:
                return f"You are doing it wrong: {errors[0]}. Fix and try again."

        if exercise == "Bicep Curl":
            elbow_min = clamp_num(snapshot.get("elbow_min"))
            elbow_target = clamp_num(targets.get("elbow_angle_down_correct"))
            shoulder_avg = clamp_num(snapshot.get("shoulder_avg"))
            shoulder_max = clamp_num(targets.get("shoulder_angle_max"))
            if isinstance(elbow_min, (int, float)) and isinstance(elbow_target, (int, float)) and elbow_min > elbow_target:
                return f"You are doing it wrong: curl higher. Reach ≤{elbow_target}°; you hit {elbow_min}°."
            if isinstance(shoulder_avg, (int, float)) and isinstance(shoulder_max, (int, float)) and shoulder_avg >= shoulder_max:
                return f"You are doing it wrong: keep shoulders still. Keep <{shoulder_max}°; you averaged {shoulder_avg}°."
            if errors:
                return f"You are doing it wrong: {errors[0]}. Fix and try again."

        if exercise == "Overhead Press":
            elbow_max = clamp_num(snapshot.get("elbow_max"))
            elbow_target = clamp_num(targets.get("elbow_angle_up_correct"))
            arm_avg = clamp_num(snapshot.get("arm_avg"))
            arm_max = clamp_num(targets.get("arm_angle_max"))
            torso_avg = clamp_num(snapshot.get("torso_avg"))
            torso_max = clamp_num(targets.get("torso_angle_max"))
            hip_change = clamp_num(snapshot.get("hip_y_change"))
            hip_max = clamp_num(targets.get("hip_y_max_change"))
            if isinstance(elbow_max, (int, float)) and isinstance(elbow_target, (int, float)) and elbow_max < elbow_target:
                return f"You are doing it wrong: press higher. Reach ≥{elbow_target}°; you hit {elbow_max}°."
            if isinstance(arm_avg, (int, float)) and isinstance(arm_max, (int, float)) and arm_avg >= arm_max:
                return f"You are doing it wrong: align arms vertical. Keep <{arm_max}°; you averaged {arm_avg}°."
            if isinstance(torso_avg, (int, float)) and isinstance(torso_max, (int, float)) and torso_avg >= torso_max:
                return f"You are doing it wrong: keep torso upright. Keep <{torso_max}°; you averaged {torso_avg}°."
            if isinstance(hip_change, (int, float)) and isinstance(hip_max, (int, float)) and hip_change > hip_max:
                return f"You are doing it wrong: keep hips stable. Move ≤{hip_max}px; you moved {hip_change}px."
            if errors:
                return f"You are doing it wrong: {errors[0]}. Fix and try again."
    except Exception:
        pass
    return "You are doing it wrong: adjust your form to meet targets."

# Load environment variables from .env file
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API if key is available
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    st.warning("Gemini API key not found in .env file. Gemini feedback will be disabled.")

# Disable sidebar
st.set_page_config(page_title="AI Gym Instructor", layout="wide", initial_sidebar_state="collapsed")

def init_database():
    conn = sqlite3.connect('workout_progress.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
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

def save_session(exercise, correct_reps, incorrect_reps, session_duration):
    total_reps = correct_reps + incorrect_reps
    correct_percentage = (correct_reps / total_reps * 100) if total_reps > 0 else 0
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        conn = sqlite3.connect('workout_progress.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO sessions (date, exercise, correct_reps, incorrect_reps, total_reps, correct_percentage, session_duration)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (date, exercise, correct_reps, incorrect_reps, total_reps, correct_percentage, session_duration))
        conn.commit()
        conn.close()
        print(f"Saved session: {exercise}, Correct: {correct_reps}, Incorrect: {incorrect_reps}, Duration: {session_duration:.2f} min")
        return True
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        st.error(f"Failed to save session: {e}")
        return False

def get_gemini_feedback(exercise, correct_reps, incorrect_reps, errors, rep_correct, model_name=None, temperature=0.4, snapshot=None):
    """Generate concise, responsive feedback using Gemini API with structured output."""
    if not GEMINI_API_KEY:
        return "Gemini API key not configured."
    
    total_reps = correct_reps + incorrect_reps
    primary_error = errors[0] if errors else ""
    # If snapshot indicates the rep was incorrect, generate deterministic coaching
    if snapshot and isinstance(snapshot, dict) and snapshot.get("status") == "incorrect":
        msg = craft_rule_based_feedback(snapshot)
        if msg:
            return msg
    snapshot_text = json.dumps(snapshot, ensure_ascii=False) if snapshot else "{}"
    prompt = f"""
You are a concise AI gym coach.

Context:
- Exercise: {exercise}
- Correct reps: {correct_reps}
- Incorrect reps: {incorrect_reps}
- Total reps: {total_reps}
- Last rep status: {"correct" if rep_correct else "incorrect"}
- Primary form issue: {primary_error if primary_error else "none"}
 - Rep metrics snapshot: {snapshot_text}

Task:
Return STRICT JSON only with keys:
- type: "correct" | "incorrect"
- message: <=25 words, specific, encouraging
- tip: <=10 words, imperative, optional
No extra commentary, no markdown.
"""
    try:
        mdl = model_name or 'gemini-2.0-flash'
        model = genai.GenerativeModel(mdl)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": float(temperature) if temperature is not None else 0.4,
                "response_mime_type": "application/json"
            }
        )
        text = (response.text or "").strip()

        # Strip code fences if present and extract JSON object
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            text = match.group(0)

        try:
            data = json.loads(text)
            msg = (data.get("message") or "").strip()
            if msg:
                return msg
            return ""
        except Exception:
            # If still not JSON, avoid showing stray tokens like 'json'
            return ""
    except Exception as e:
        print(f"Gemini API error: {str(e)}")
        return ""

def wrap_text(text, max_width, font, font_scale, thickness):
    """Wrap text to fit within max_width pixels, returning lines of text."""
    words = text.split()
    lines = []
    current_line = []
    current_width = 0
    for word in words:
        text_size = cv2.getTextSize(word + " ", font, font_scale, thickness)[0][0]
        if current_width + text_size <= max_width:
            current_line.append(word)
            current_width += text_size
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_width = text_size
    if current_line:
        lines.append(" ".join(current_line))
    return lines

def process_video(input_path, detector, counter, use_gemini):
    """Process the uploaded video and return the output path."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, "Error playing video. Try uploading an MP4 file.", 0.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    output_path = tempfile.mktemp(suffix='.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        os.unlink(output_path) if os.path.exists(output_path) else None
        return None, "Error: Failed to initialize video writer. Check codec support.", 0.0

    start_time = datetime.now()
    current_feedback = ""
    feedback_frames = 0
    feedback_duration = int(fps * 5)  # Display feedback for 5 seconds

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = detector.detect_pose(frame, draw=True)
        frame, default_feedback = counter.update(landmarks, frame)
        if use_gemini:
            if default_feedback and feedback_frames <= 0:
                current_feedback = get_gemini_feedback(
                    getattr(counter, 'exercise_name', counter.__class__.__name__.replace("Counter", "")),
                    counter.correct_reps,
                    counter.incorrect_reps,
                    counter.errors,
                    counter.rep_correct,
                    st.session_state.gemini_model if 'gemini_model' in st.session_state else 'gemini-2.0-flash',
                    st.session_state.gemini_temperature if 'gemini_temperature' in st.session_state else 0.4,
                    getattr(counter, 'last_rep_snapshot', None)
                ) or default_feedback
                feedback_frames = feedback_duration if (current_feedback or default_feedback) else 0
        else:
            current_feedback = default_feedback
            feedback_frames = feedback_duration if default_feedback else 0

        frame = draw_rep_count(frame, counter.correct_reps, counter.incorrect_reps)
        if current_feedback and feedback_frames > 0:
            print(f"Feedback: {current_feedback}")
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            max_width = width - 40
            lines = wrap_text(current_feedback, max_width, font, font_scale, thickness)
            text_y = height - 50 - (len(lines) - 1) * 30
            for i, line in enumerate(lines):
                text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                text_x = (width - text_size[0]) // 2
                cv2.putText(
                    frame,
                    line,
                    (text_x, text_y + i * 30),
                    font,
                    font_scale,
                    (0, 0, 0),
                    thickness + 2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    line,
                    (text_x, text_y + i * 30),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )
            feedback_frames -= 1
        else:
            current_feedback = ""
            feedback_frames = 0

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    cap.release()
    out.release()
    session_duration = (datetime.now() - start_time).total_seconds() / 60.0
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return output_path, None, session_duration
    else:
        os.unlink(output_path) if os.path.exists(output_path) else None
        return None, "Error: Processed video could not be generated or is empty.", session_duration

# Initialize session state
if 'counter' not in st.session_state:
    st.session_state.counter = None
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'session_data' not in st.session_state:
    st.session_state.session_data = None
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'use_gemini' not in st.session_state:
    st.session_state.use_gemini = False

# Main page content
st.title("AI Gym Instructor")

# Initialize database
init_database()

# Navigation buttons
col1, col2 = st.columns(2)
col1.button("View Progress", on_click=lambda: st.session_state.update({'page': 'progress'}))
col2.button("View Tutorials", on_click=lambda: st.session_state.update({'page': 'tutorials'}))

# Redirect based on session state
if st.session_state.page == 'progress':
    print("Redirecting from main.py to progress.py")
    st.switch_page("pages/progress.py")
elif st.session_state.page == 'tutorials':
    print("Redirecting from main.py to tutorials.py")
    st.switch_page("pages/tutorials.py")

# Exercise selection
exercise = st.selectbox("Select Exercise", ["Squat", "Bicep Curl", "Overhead Press"])

# Gemini feedback toggle
st.session_state.use_gemini = st.checkbox("Enable Gemini Feedback", value=st.session_state.use_gemini)

# Optional Gemini controls
if 'gemini_model' not in st.session_state:
    st.session_state.gemini_model = 'gemini-2.0-flash'
if 'gemini_temperature' not in st.session_state:
    st.session_state.gemini_temperature = 0.4

if st.session_state.use_gemini and GEMINI_API_KEY:
    model_col, temp_col = st.columns(2)
    st.session_state.gemini_model = model_col.selectbox(
        "Gemini Model",
        ["gemini-2.0-flash", "gemini-2.0-pro"],
        index=["gemini-2.0-flash", "gemini-2.0-pro"].index(st.session_state.gemini_model)
    )
    st.session_state.gemini_temperature = temp_col.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=float(st.session_state.gemini_temperature),
        step=0.05
    )

# Initialize counter based on exercise
if exercise == "Squat":
    st.session_state.counter = SquatCounter()
elif exercise == "Bicep Curl":
    st.session_state.counter = BicepCurlCounter()
elif exercise == "Overhead Press":
    st.session_state.counter = OverheadPressCounter()

# Input selection
input_type = st.radio("Select Input", ["Webcam", "Upload Video"])

# Initialize pose detector
detector = PoseDetector()

# Save session if previous webcam session exists
if st.session_state.session_data and not st.session_state.webcam_active:
    exercise, correct_reps, incorrect_reps, session_duration = st.session_state.session_data
    save_session(exercise, correct_reps, incorrect_reps, session_duration)
    st.session_state.session_data = None

if input_type == "Webcam":
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot open webcam")
    else:
        stframe = st.empty()
        start_time = datetime.now()
        st.session_state.webcam_active = True
        stop_button = st.button("Stop Webcam")
        current_feedback = ""
        feedback_frames = 0
        feedback_duration = int(30 * 5)  # 5 seconds at 30 FPS
        try:
            while st.session_state.webcam_active and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                landmarks = detector.detect_pose(frame, draw=True)
                frame, default_feedback = st.session_state.counter.update(landmarks, frame)
                if st.session_state.use_gemini:
                    if default_feedback and feedback_frames <= 0:
                        current_feedback = get_gemini_feedback(
                            exercise,
                            st.session_state.counter.correct_reps,
                            st.session_state.counter.incorrect_reps,
                            st.session_state.counter.errors,
                            st.session_state.counter.rep_correct,
                            st.session_state.gemini_model,
                            st.session_state.gemini_temperature,
                            getattr(st.session_state.counter, 'last_rep_snapshot', None)
                        ) or default_feedback
                        feedback_frames = feedback_duration if (current_feedback or default_feedback) else 0
                else:
                    current_feedback = default_feedback
                    feedback_frames = feedback_duration if default_feedback else 0

                frame = draw_rep_count(frame, st.session_state.counter.correct_reps, st.session_state.counter.incorrect_reps)
                if current_feedback and feedback_frames > 0:
                    print(f"Feedback: {current_feedback}")
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    max_width = frame.shape[1] - 40
                    lines = wrap_text(current_feedback, max_width, font, font_scale, thickness)
                    text_y = frame.shape[0] - 50 - (len(lines) - 1) * 30
                    for i, line in enumerate(lines):
                        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                        text_x = (frame.shape[1] - text_size[0]) // 2
                        cv2.putText(
                            frame,
                            line,
                            (text_x, text_y + i * 30),
                            font,
                            font_scale,
                            (0, 0, 0),
                            thickness + 2,
                            cv2.LINE_AA
                        )
                        cv2.putText(
                            frame,
                            line,
                            (text_x, text_y + i * 30),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                            cv2.LINE_AA
                        )
                    feedback_frames -= 1
                else:
                    current_feedback = ""
                    feedback_frames = 0

                stframe.image(frame, channels="RGB")
                st.session_state.session_data = (
                    exercise,
                    st.session_state.counter.correct_reps,
                    st.session_state.counter.incorrect_reps,
                    (datetime.now() - start_time).total_seconds() / 60.0
                )
        except Exception as e:
            st.error(f"Webcam error: {e}")
        finally:
            cap.release()
            st.session_state.webcam_active = False
            stframe.empty()
            if st.session_state.session_data:
                exercise, correct_reps, incorrect_reps, session_duration = st.session_state.session_data
                save_session(exercise, correct_reps, incorrect_reps, session_duration)
                st.session_state.session_data = None
else:
    uploaded_file = st.file_uploader("Upload a video (MP4)", type=["mp4"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_file.read())
            input_path = tfile.name

        st.info("Processing video... This may take a moment.")
        output_path, error, session_duration = process_video(input_path, detector, st.session_state.counter, st.session_state.use_gemini)
        
        if error:
            st.error(error)
        elif output_path and os.path.exists(output_path):
            try:
                with open(output_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes, format="video/mp4")
                st.write(f"Correct Reps: {st.session_state.counter.correct_reps}, Incorrect Reps: {st.session_state.counter.incorrect_reps}")
                save_session(exercise, st.session_state.counter.correct_reps, st.session_state.counter.incorrect_reps, session_duration)
            except Exception as e:
                st.error(f"Failed to display video: {e}")
        else:
            st.error("Error: Processed video could not be generated or is empty.")

        try:
            os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)
        except PermissionError as e:

            st.warning(f"Could not delete temporary files: {e}. They will be removed on system cleanup.")
