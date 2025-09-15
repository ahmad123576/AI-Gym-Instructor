import streamlit as st

# Disable sidebar
st.set_page_config(page_title="Exercise Tutorials", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'tutorials'

# Redirect to main page
if st.session_state.page == 'main':
    print("Redirecting from tutorials.py to main.py")
    st.switch_page("main.py")

st.title("Exercise Tutorials")

st.markdown("""
Learn the proper form for each exercise to maximize your workout and minimize errors. Select an exercise below to view its tutorial.
""")

exercise = st.selectbox("Select Exercise", ["Squat", "Bicep Curl", "Overhead Press"])

if exercise == "Squat":
    st.subheader("Squat Tutorial")
    st.markdown("""
    **How to Perform a Squat:**
    - **Starting Position**: Stand with feet shoulder-width apart, toes slightly turned out.
    - **Movement**: Push your hips back and bend your knees, lowering your body until your thighs are at least parallel to the ground (knees at ~90°).
    - **Key Points**:
      - Keep your back straight and chest up to avoid leaning forward.
      - Ensure knees track over toes, not collapsing inward.
      - Engage your core for stability.
    - **Common Mistakes**:
      - Shallow squats (knees not bending enough).
      - Leaning forward, causing back strain.
    - **Tip**: Practice in front of a mirror or use the AI Gym Instructor to check your form!
    """)
elif exercise == "Bicep Curl":
    st.subheader("Bicep Curl Tutorial")
    st.markdown("""
    **How to Perform a Bicep Curl:**
    - **Starting Position**: Stand with feet hip-width apart, holding weights (dumbbells or barbell) with palms facing up.
    - **Movement**: Bend your elbows to lift the weights toward your shoulders, keeping elbows close to your body.
    - **Key Points**:
      - Avoid swinging your body or using momentum.
      - Keep wrists straight and shoulders relaxed.
      - Fully extend arms at the bottom of each rep.
    - **Common Mistakes**:
      - Using back or shoulders to lift the weight.
      - Not fully extending arms at the bottom.
    - **Tip**: Start with light weights to master form before increasing resistance.
    """)
elif exercise == "Overhead Press":
    st.subheader("Overhead Press Tutorial")
    st.markdown("""
    **How to Perform an Overhead Press:**
    - **Starting Position**: Stand with feet shoulder-width apart, holding weights at shoulder height, palms facing forward.
    - **Movement**: Press the weights overhead until arms are fully extended, keeping your core tight.
    - **Key Points**:
      - Keep your back straight, avoiding excessive arching.
      - Engage your core to prevent leaning back.
      - Lower weights slowly to shoulder level.
    - **Common Mistakes**:
      - Arching the lower back excessively.
      - Not locking out arms at the top.
    - **Tip**: Use a mirror or the AI Gym Instructor to ensure proper alignment.
    """)

st.button("Back to Home", on_click=lambda: st.session_state.update({'page': 'main'}))