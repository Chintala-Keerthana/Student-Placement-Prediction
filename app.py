import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #ede9fe, #faf5ff);
}

/* Header Card */
.header-card {
    background: linear-gradient(135deg,#4c1d95,#6d28d9);
    color: white;
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 25px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.15);
}

/* Form Card */
.form-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border: 2px solid #d8b4fe;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(124,58,237,0.08);
}

.result-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border: 2px solid #c4b5fd;
    margin-top: 20px;
    margin-bottom: 20px;
}

.graph-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border: 2px solid #a78bfa;
    margin-top: 20px;
    margin-bottom: 20px;
}

.message-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    border: 2px solid #8b5cf6;
    margin-top: 20px;
    margin-bottom: 20px;
}

.result-card,
.graph-card,
.message-card {
    background: white;
    padding: 20px;
    border-radius: 18px;
    margin-top: 20px;
    margin-bottom: 20px;
    box-shadow: 0 4px 12px rgba(124,58,237,0.08);
}

.result-card {
    border: 2px solid #c4b5fd;
}

.graph-card {
    border: 2px solid #a78bfa;
}

.message-card {
    border: 2px solid #8b5cf6;
}

/* Labels */
label {
    font-weight: 600 !important;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(90deg,#7c3aed,#a855f7);
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px;
    font-weight: 700;
}

/* Metric Cards */
[data-testid="stMetric"] {
    background: white;
    border: 2px solid #d8b4fe;
    padding: 15px;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(124,58,237,0.1);
}

[data-testid="stMetricValue"] {
    color: #7c3aed;
    font-weight: bold;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("placement_model.pkl")

try:
    model = load_model()
except Exception:
    st.error("❌ placement_model.pkl file not found.")
    st.stop()

# ---------------- HEADER ----------------
st.markdown("""
<div class="header-card">
    <h2>🎓 Student Placement Predictor</h2>
    <p>Fill the details below to predict the placement chances of a student.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- STUDENT DETAILS ----------------

with st.container(border=True):

    st.subheader("📋 Student Details")
    st.caption("Provide accurate information")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 30, 21)

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        internships = st.number_input(
            "Internships",
            min_value=0,
            max_value=10,
            value=0
        )

    with col2:
        stream = st.selectbox(
            "Stream",
            ["CSE", "IT", "ECE", "EEE", "Mechanical", "Civil"]
        )

        cgpa = st.slider(
            "CGPA",
            0.0,
            10.0,
            7.0,
            0.1
        )

        backlogs = st.selectbox(
            "Backlogs",
            ["0","1","2","3","4","5","6","7","8","9","10"]
        )

    hostel = st.selectbox(
        "Hostel",
        ["No", "Yes"]
    )# ---------------- BUTTON ----------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    predict = st.button(
        "🚀 Predict Placement",
        use_container_width=True
    )

# ---------------- ENCODING ----------------
gender_encoded = 1 if gender == "Male" else 0

hostel_encoded = 1 if hostel == "Yes" else 0

backlogs_encoded = int(backlogs)

stream_dict = {
    "CSE": 0,
    "IT": 1,
    "ECE": 2,
    "EEE": 3,
    "Mechanical": 4,
    "Civil": 5
}

stream_encoded = stream_dict[stream]

# ---------------- PREDICTION ----------------
if predict:

    input_data = np.array([[
    age,
    gender_encoded,
    stream_encoded,
    internships,
    cgpa,
    hostel_encoded,
    backlogs_encoded
]], dtype=float)

    prediction = model.predict(input_data)

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1] * 100
    else:
        probability = 75 if prediction[0] == 1 else 25

    result = "PLACED 🎉" if prediction[0] == 1 else "NOT PLACED ❌"

    # ---------------- RESULT SECTION ----------------
    with st.container(border=True):

        st.subheader("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Placement Probability",
                f"{probability:.2f}%"
            )

        with col2:
            st.metric(
                "Result",
                result
            )

        st.write("### Probability Score")
        st.progress(int(probability))

    # ---------------- GRAPH SECTION ----------------
    with st.container(border=True):

        st.subheader("📈 Placement Probability Graph")

        graph_data = pd.DataFrame(
            {
                "Probability": [
                    probability,
                    100 - probability
                ]
            },
            index=["Placed", "Not Placed"]
        )

        st.bar_chart(graph_data)

    # ---------------- MESSAGE SECTION ----------------
    with st.container(border=True):

        if prediction[0] == 1:
            st.success(
                f"🎉 Student is likely to be placed with {probability:.2f}% confidence."
            )
            st.balloons()
        else:
            st.warning(
                f"⚠️ Student may not get placed. Confidence: {100 - probability:.2f}%"
            )
# ---------------- FOOTER ----------------
st.write("---")
st.caption("💜 Built with Python • Streamlit • Scikit-Learn • Machine Learning")