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

Usage

Select an exercise from the dropdown.
Choose "Webcam" to start real-time analysis or "Upload Video" to process an MP4 file.
Toggle "Use Gemini AI Feedback" for enhanced tips (if configured).
View progress or tutorials via the respective buttons.
Stop the webcam or review the annotated video output.

Project Structure

main.py: Orchestrates the app, handles UI, and manages sessions.
pose_detector.py: Detects body landmarks using MediaPipe.
rep_counter.py: Tracks reps and evaluates form for each exercise.
utils.py: Provides utility functions (e.g., angle calculation, drawing).
progress.py: Displays workout progress with charts.
tutorials.py: Offers exercise guidance.
requirements.txt: Lists dependencies.
workout_progress.db: SQLite database file (generated on first run).
.env (optional): Stores Gemini API key.

Development Process

Phase 1: Integrated MediaPipe for pose detection with sample images.
Phase 2: Developed rep counters with form logic and angle thresholds.
Phase 3: Built the Streamlit UI and added video processing.
Phase 4: Implemented optional Gemini API feedback with fallbacks.
Phase 5: Added progress tracking, testing, and polishing.

Future Enhancements

Support for additional exercises (e.g., Push-Up, Deadlift).
Mobile app integration (e.g., using Kivy or Flutter).
Cloud-based progress syncing (e.g., Firebase).
Voice feedback and multi-language support.

Acknowledgments

Abacus Consulting, Lahore: For providing the internship opportunity and guidance.
Open-Source Community: For tools like MediaPipe and Streamlit.

Contact

Author: Muhammad Ahmad
Email: [ahmadghafoor555786@gmail.com]
GitHub: ahmad123576

- ## How To Run
- use these files in a project folder
- make a folder within project folder named pages
- within pages folder, put progrees,py and tutorials.py
- then run 'pip install -r requirements.txt' in your cmd terminal
- after that , write 'streamlit run main.py' in your terminal
