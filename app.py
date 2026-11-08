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
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background-color: #f0f2f6 !important; 
}

[data-testid="stForm"] {
    background-color: #ffffff !important;
    padding: 40px !important; 
    border-radius: 12px !important;
    border: 1px solid #e5e5e5 !important;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08) !important;
    margin-top: 10px;
}

h3, [data-testid="stMarkdownContainer"] h3 {
    text-align: center !important;
    font-weight: 800 !important;
    color: #111111 !important;
    font-size: 28px !important; 
    padding-bottom: 20px !important;
}

[data-testid="stWidgetLabel"] p {
    font-weight: 700 !important;
    color: #222222 !important;
    font-size: 18px !important; 
    margin-bottom: 5px !important;
}

div[data-baseweb="select"] > div, 
div[data-testid="stNumberInput"] input {
    background-color: #fcfcfc !important;
    border: 1px solid #cccccc !important;
    border-radius: 6px !important;
    color: #000000 !important;
    font-size: 18px !important; 
}

div[data-baseweb="select"] span {
    font-size: 18px !important;
}

[data-testid="stFormSubmitButton"] button {
    background-color: #5d9b5d !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    border: none !important;
    height: 55px !important; 
    width: 100% !important;
    margin-top: 10px !important;
}

[data-testid="stFormSubmitButton"] button p {
    font-size: 22px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
}

[data-testid="stFormSubmitButton"] button:hover {
    background-color: #4a7d4a !important;
    box-shadow: 0 4px 10px rgba(74, 125, 74, 0.4) !important;
}

[data-testid="stMetric"] {
    text-align: center !important;
    margin-top: 20px !important;
    padding: 15px !important;
    background-color: #f8f9fa !important;
    border-radius: 8px !important;
}

[data-testid="stMetricLabel"] p {
    color: #000000 !important;
    font-size: 22px !important;
    font-weight: 700 !important;
    justify-content: center !important;
}

[data-testid="stMetricValue"], 
[data-testid="stMetricValue"] > div {
    color: #000000 !important;
    font-size: 36px !important;
    font-weight: 900 !important;
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
