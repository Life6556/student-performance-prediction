# Student Exam Performance Prediction 🎓

An End-to-End Machine Learning project built to predict the mathematics score of a student based on various demographic and academic features. 

This project demonstrates a complete modular machine learning pipeline, from data ingestion to model deployment, following industry-standard software engineering practices.

## 🔗 Live Demo
Check out the live web application here: [https://student-performance-predicition.streamlit.app/](https://student-performance-predicition.streamlit.app/)

---

## 🚀 Key Features

- **End-to-End ML Pipeline:** Fully automated pipeline covering Data Ingestion, Data Transformation, and Model Training.
- **Modular Codebase:** Structured using Object-Oriented Programming (OOP) principles for easy maintenance and scalability.
- **Custom Exception & Logging:** Integrated system-wide logging and custom exception handling for robust debugging.
- **Interactive Web App:** A clean, responsive, and user-friendly web interface built with Streamlit and Custom CSS.
- **Cloud Deployment:** Hosted seamlessly on Streamlit Community Cloud.

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Machine Learning:** Scikit-Learn, Pandas, NumPy
- **Web Framework:** Streamlit
- **Serialization:** Pickle / Dill
- **Version Control:** Git & GitHub

---

## 📂 Project Structure

```text
├── artifacts/               # Contains saved models and preprocessor (model.pkl, preprocessor.pkl)
├── src/                     # Main source code folder
│   ├── components/          # Data Ingestion, Transformation, and Model Trainer logic
│   ├── pipeline/            # Training and Prediction pipeline scripts
│   ├── exception.py         # Custom exception handling module
│   ├── logger.py            # Custom logging module
│   └── utils.py             # Helper functions (e.g., model saving/loading)
├── app.py                   # Streamlit web application script
├── requirements.txt         # Required Python packages
├── setup.py                 # Script to build the project as a package
└── README.md                # Project documentation
💻 How to Run Locally
Follow these steps to run the project on your local machine:

1. Clone the repository:

Bash
git clone https://github.com/Life6556/student-performance-prediction.git
cd student-performance-prediction
2. Create and activate a virtual environment (Optional but recommended):

Bash
conda create -p venv python=3.10 -y
conda activate ./venv
3. Install the required dependencies:

Bash
pip install -r requirements.txt
4. Run the Streamlit Application:

Bash
streamlit run app.py
5. Open your browser:
Navigate to http://localhost:8501 to view the app.

👨‍💻 Author
Al Hafij

CSE Undergraduate Student

GitHub: Life6556

Email: lifealhafij@gmail.com

If you like this project, please consider giving it a ⭐ on GitHub!