# AI Gym Instructor

## Overview
The **AI Gym Instructor** is a web-based application designed to revolutionize home workouts by providing real-time exercise form correction and progress tracking using artificial intelligence. Developed as a final project during an internship at **Abacus Consulting, Lahore**, this tool addresses the common issue of improper exercise form, reducing injury risks for fitness enthusiasts. It supports three exercises—**Squat**, **Bicep Curl**, and **Overhead Press**—offering rep counting, actionable feedback, and a progress dashboard.

## Features
- **Real-Time Form Analysis**: Uses MediaPipe for body landmark detection and OpenCV for video processing.
- **Rep Counting**: Tracks correct and incorrect repetitions with form evaluation (e.g., knee angle ≤90° for Squats).
- **Feedback System**: Provides immediate feedback via rule-based logic, with optional AI-enhanced tips using the Gemini API.
- **Progress Tracking**: Stores session data in a SQLite database, visualized with charts and metrics.
- **User-Friendly UI**: Built with Streamlit, supporting webcam and video upload inputs.
- **Tutorials**: Offers static guides for proper exercise techniques.

## Tech Stack
- **Python 3.8+**: Core language.
- **Streamlit**: Web framework for the UI.
- **MediaPipe**: Pose estimation and landmark detection.
- **OpenCV**: Video frame processing and text overlays.
- **SQLite**: Lightweight database for progress tracking.
- **NumPy & Matplotlib**: Numerical operations and visualizations.
- **Google Gemini API** (optional): Advanced feedback (requires API key).

- ## How To Run
- use these files in a project folder
- make a folder within project folder named pages
- within pages folder, put progrees,py and tutorials.py
- then run 'pip install -r requirements.txt' in your cmd terminal
- after that , write 'streamlit run main.py' in your terminal
