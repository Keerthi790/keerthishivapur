
import streamlit as st
import multi_disease_model as mdm

st.set_page_config(page_title="Multi Disease Prediction", layout="centered")
st.title("🩺 Multiple Disease Prediction System")

st.sidebar.title("Select Disease")
choice = st.sidebar.radio("", [
    "Diabetes Prediction",
    "Heart Disease Prediction",
    "Kidney Disease Prediction"
])

if 'loaded' not in st.session_state:
    try:
        mdm.load_and_train_all()
        st.session_state.loaded = True
    except FileNotFoundError as e:
        st.sidebar.error(str(e))
        st.stop()

REAL_LABELS = {
    "diabetes": {
        "Pregnancies": "Number of Pregnancies",
        "Glucose": "Glucose Level (mg/dL)",
        "BloodPressure": "Blood Pressure (mmHg)",
        "SkinThickness": "Skin Thickness (mm)",
        "Insulin": "Insulin Level (uU/mL)",
        "BMI": "Body Mass Index (BMI)",
        "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
        "Age": "Age (years)"
    },
    "heart": {
        "age": "Age (years)",
        "sex": "Sex (0 = Female, 1 = Male)",
        "cp": "Chest Pain Type (0–3)",
        "trestbps": "Resting Blood Pressure (mmHg)",
        "chol": "Serum Cholesterol (mg/dL)",
        "fbs": "Fasting Blood Sugar > 126 mg/dL (1 = True, 0 = False)",
        "restecg": "Resting ECG Results (0–2)",
        "thalach": "Max Heart Rate Achieved (bpm)",
        "exang": "Exercise Induced Angina (0 = No, 1 = Yes)",
        "oldpeak": "ST Depression Induced by Exercise",
        "slope": "Slope of Peak Exercise ST Segment",
        "ca": "Number of Major Vessels Colored (0–4)",
        "thal": "Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible)"
    },
    "kidney": {
        "age": "Age (years)",
        "bp": "Blood Pressure (mmHg)",
        "sg": "Specific Gravity",
        "al": "Albumin",
        "su": "Sugar",
        "bgr": "Blood Glucose Random (mg/dL)",
        "bu": "Blood Urea (mg/dL)",
        "sc": "Serum Creatinine (mg/dL)",
        "sod": "Sodium (mEq/L)",
        "pot": "Potassium (mEq/L)",
        "hemo": "Hemoglobin (g/dL)",
        "pcv": "Packed Cell Volume",
        "wc": "White Blood Cell Count (cells/cumm)",
        "rc": "Red Blood Cell Count (millions/cumm)"
    }
}

def render_form(form_key, disease_key):
    st.header(f"{disease_key.capitalize()} Prediction")
    features = mdm.get_feature_info(disease_key)
    inputs = []

    with st.form(form_key):
        for feat in features:
            label = REAL_LABELS[disease_key].get(feat, feat)
            val = st.number_input(label=label, value=0.0, key=f"{disease_key}_{feat}")
            inputs.append(val)
        submit = st.form_submit_button("Predict")

    if submit:
        prediction = mdm.predict_disease(disease_key, inputs)
        if prediction == 1:
            st.error(f"⚠️ Prediction: Positive for {disease_key.capitalize()} Disease")
        else:
            st.success(f"✅ Prediction: Negative for {disease_key.capitalize()} Disease")

if choice == "Diabetes Prediction":
    render_form("form_diabetes", "diabetes")
elif choice == "Heart Disease Prediction":
    render_form("form_heart", "heart")
elif choice == "Kidney Disease Prediction":
    render_form("form_kidney", "kidney")
