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

/* App background */
.stApp {
    background-color: #f2f2f2 !important;
    color: #111111 !important;
}


/* Main page background */
.stApp {
    background-color: #f2f2f2 !important;
}
/* Main content width */
.block-container {
    max-width: 760px;
    padding-top: 2rem;
}

/* Default text color */
.stApp, .stApp p, .stApp label,
.stApp h1, .stApp h2, .stApp h3,
.stMarkdown {
    color: #111111 !important;
}

/* Input labels */
[data-testid="stWidgetLabel"] p {
    color: #111111 !important;
    font-weight: 600 !important;
}

/* Dropdown */
div[data-baseweb="select"] > div {
    background-color: white !important;
    color: #111111 !important;
    border: 1px solid #d5d5d5 !important;
}

/* Dropdown selected text */
div[data-baseweb="select"] span {
    color: #111111 !important;
}

/* Number input */
div[data-testid="stNumberInput"] input {
    background-color: white !important;
    color: #111111 !important;
    border: 1px solid #d5d5d5 !important;
}

/* Green button */
div[data-testid="stFormSubmitButton"] button {
    background-color: #65ad59 !important;
    color: white !important;
    border: none !important;
    border-radius: 5px;
    height: 42px;
    font-weight: 600;
}

/* Button hover */
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: #4c9842 !important;
    color: white !important;
}

/* Prediction result */
.prediction {
    text-align: center;
    color: #111111 !important;
    font-size: 18px;
    font-weight: bold;
    margin-top: 15px;
}
/* Main title */
h2 {
    font-size: 28px !important;
}

/* Input labels */
[data-testid="stWidgetLabel"] p {
    font-size: 20px !important;
    font-weight: 600 !important;
}

/* Dropdown text */
div[data-baseweb="select"] span {
    font-size: 18px !important;
}

/* Number input text */
div[data-testid="stNumberInput"] input {
    font-size: 18px !important;
}

/* Predict button - larger text */
div[data-testid="stFormSubmitButton"] button,
div[data-testid="stFormSubmitButton"] button p,
div[data-testid="stFormSubmitButton"] button span {
    font-size: 24px !important;
    font-weight: 400 !important;
    color: white !important;
}

/* Predicted Mathematics Score label */
[data-testid="stMetricLabel"] p {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #111111 !important;
}

/* Predicted score value */
[data-testid="stMetricValue"] {
    font-size: 36px !important;
    font-weight: 700 !important;
}
/* Fix top white background */
html,
body,
#root,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
header[data-testid="stHeader"],
[data-testid="stMain"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background-color: #f2f2f2 !important;
    background: #f2f2f2 !important;
}

/* Remove header shadow */
header[data-testid="stHeader"] {
    box-shadow: none !important;
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
