import streamlit as st
import sys
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
from src.logger import logging
from src.exception import CustomException

st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="centered"
    )


st.markdown("""
<style>

/* ===============================
   1. Main App Background
================================= */

.stApp {
    background-color: #f2f2f2;
    color: #000000;
}

/* Header and Toolbar Background */

header[data-testid="stHeader"] {
    background-color: #f2f2f2 !important;
}

[data-testid="stToolbar"] {
    background-color: #f2f2f2 !important;
}

[data-testid="stDecoration"] {
    background-color: #f2f2f2 !important;
}

/* Main Container */

.block-container {
    max-width: 1050px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* ===============================
   2. General Text Styling
================================= */

h1, h2, h3, h4, h5, h6 {
    color: #111111 !important;
    font-weight: 700 !important;
}

p, label, span {
    color: #000000;
}

/* Main Title */

h1 {
    font-size: 2.5rem !important;
    margin-bottom: 2rem !important;
}

/* Input Labels */

.stSelectbox label,
.stSlider label,
.stNumberInput label,
.stTextInput label {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    color: #000000 !important;
}

/* ===============================
   3. Dropdown Styling
================================= */

/* Dropdown Box */

div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #cccccc !important;
    border-radius: 10px !important;
    min-height: 60px !important;
}

/* Selected Dropdown Text */

div[data-baseweb="select"] span {
    color: #000000 !important;
    font-size: 1.1rem !important;
}

/* Dropdown Arrow */

div[data-baseweb="select"] svg {
    fill: #000000 !important;
}

/* Dropdown Menu */

ul[data-baseweb="menu"] {
    background-color: #ffffff !important;
}

/* Dropdown Options */

li[role="option"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-size: 1rem !important;
}

/* Dropdown Hover */

li[role="option"]:hover {
    background-color: #e8f5e9 !important;
}

/* ===============================
   4. Number Input Styling
================================= */

div[data-baseweb="input"] {
    background-color: #ffffff !important;
    border-radius: 8px !important;
}

div[data-baseweb="input"] input {
    color: #000000 !important;
    background-color: #ffffff !important;
    font-size: 1.1rem !important;
}

/* ===============================
   5. Slider Styling
================================= */

div[data-testid="stSlider"] {
    color: #000000 !important;
}

/* Slider Value */

div[data-testid="stSlider"] [data-testid="stMarkdownContainer"] {
    color: #000000 !important;
}

/* ===============================
   6. Submit Button Styling
================================= */

div.stButton > button {
    background-color: #198754 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 30px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    transition: 0.3s ease;
}

/* Button Hover */

div.stButton > button:hover {
    background-color: #146c43 !important;
    color: #ffffff !important;
    border: none !important;
}

/* Button Text */

div.stButton > button p {
    color: #ffffff !important;
}

/* ===============================
   7. Prediction Result Styling
================================= */

div[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dddddd;
}

div[data-testid="stMetricLabel"] {
    color: #333333 !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
}

div[data-testid="stMetricValue"] {
    color: #198754 !important;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
}

/* ===============================
   8. Horizontal Divider
================================= */

hr {
    border-color: #cccccc !important;
}

/* ===============================
   9. Responsive Design
================================= */

@media screen and (max-width: 768px) {

    h1 {
        font-size: 1.8rem !important;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

}

</style>
""", unsafe_allow_html=True)


with st.form("student_prediction_form"):

    st.subheader("Student Exam Performance Indicator")

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["male", "female"]
        )

        race_ethnicity = st.selectbox(
            "Race/Ethnicity",
            ["group A", "group B", "group C",
             "group D", "group E"]
        )

        parental_level_of_education = st.selectbox(
            "Parental Level of Education",
            [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree"
            ]
        )

    with col2:

        lunch = st.selectbox(
            "Lunch",
            ["standard", "free/reduced"]
        )

        test_preparation_course = st.selectbox(
            "Test Preparation Course",
            ["none", "completed"]
        )

        reading_score = st.slider(
            "Reading Score",
            min_value=0,
            max_value=100,
            value=50
        )

        writing_score = st.slider(
            "Writing Score",
            min_value=0,
            max_value=100,
            value=50
        )

    submitted = st.form_submit_button(
        "Predict Math Score",
        use_container_width=True
    )

if submitted:
    try:
        input_data=CustomData(gender,race_ethnicity,
                              parental_level_of_education,
                              lunch,test_preparation_course,
                              reading_score,writing_score)
        pred_df=input_data.get_input_data_as_dataFrame()

        predict_pipeline=PredictPipeline()
        math_score=predict_pipeline.predict(pred_df)[0]
        st.metric(
            label="Predicted Mathematics Score",
            value=f"{math_score:.2f} / 100"
        )
        logging.info("predicted successfull")
    except Exception as e:
        logging.error("an error occured")
        raise CustomException(e,sys)
