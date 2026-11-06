from pathlib import Path
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title='Heart Disease Prediction | Research Demo', page_icon='🫀', layout='centered')
st.title('Heart Disease Prediction')
st.caption('Logistic regression • Research and educational demonstration')
st.warning('Not a medical device. Predictions are not diagnoses and must not be used to make clinical decisions.')

model_path = Path(__file__).with_name('heart_disease_logistic_regression.pkl')
if not model_path.exists():
    st.error('Model file is missing. Include heart_disease_logistic_regression.pkl in the app directory.')
    st.stop()

@st.cache_resource
def load_model():
    try:
        return joblib.load(model_path)
    except Exception as exc:
        st.error('Unable to load the trained model. Check that scikit-learn 1.6.1 is installed.')
        st.exception(exc)
        st.stop()

model = load_model()
with st.form('patient'):
    st.subheader('Patient information')
    age = st.number_input('Age (years)', min_value=1, max_value=120, value=55)
    sex = st.selectbox('Sex as recorded in dataset', ['Male', 'Female'])
    cp = st.selectbox('Chest pain type', ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
    trestbps = st.number_input('Resting blood pressure (mm Hg)', min_value=0, max_value=300, value=130, help='Enter 0 if unknown; it will be imputed.')
    chol = st.number_input('Serum cholesterol (mg/dL)', min_value=0, max_value=800, value=240, help='Enter 0 if unknown; it will be imputed.')
    fbs = st.selectbox('Fasting blood sugar > 120 mg/dL', ['No', 'Yes', 'Unknown'])
    restecg = st.selectbox('Resting ECG', ['normal', 'st-t abnormality', 'lv hypertrophy', 'Unknown'])
    thalch = st.number_input('Maximum heart rate achieved (beats/min)', min_value=0, max_value=260, value=150, help='Enter 0 if unknown; it will be imputed.')
    exang = st.selectbox('Exercise-induced angina', ['No', 'Yes', 'Unknown'])
    oldpeak = st.number_input('ST depression (oldpeak)', min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
    submitted = st.form_submit_button('Generate prediction', type='primary', use_container_width=True)

if submitted:
    data = pd.DataFrame([{
        'age': age,
        'trestbps': None if trestbps == 0 else trestbps,
        'chol': None if chol == 0 else chol,
        'thalch': None if thalch == 0 else thalch,
        'oldpeak': oldpeak,
        'sex': sex,
        'cp': cp,
        'fbs': {'Yes': True, 'No': False, 'Unknown': None}[fbs],
        'restecg': None if restecg == 'Unknown' else restecg,
        'exang': {'Yes': True, 'No': False, 'Unknown': None}[exang],
    }])
    prob = float(model.predict_proba(data)[0, 1])
    prediction = int(prob >= 0.5)
    st.divider()
    st.subheader('Model result')
    st.metric('Estimated probability of positive class', f'{prob:.1%}')
    st.progress(prob)
    if prediction:
        st.info('Model classification: Positive (probability ≥ 50%).')
    else:
        st.info('Model classification: Negative (probability < 50%).')
    st.caption('The 50% classification threshold is a demonstration setting, not a clinically validated threshold. These probabilities have not been externally calibrated.')

with st.expander('Model background and limitations'):
    st.write('Trained using the UCI heart disease dataset (920 records, four contributing hospital datasets), with logistic regression and C=0.1. Hospital identity was excluded as a predictor.')
    st.write('Nested leave-one-hospital-out validation: mean accuracy 74.3%, mean recall 83.2%, mean ROC-AUC 0.804. Results vary by hospital and do not establish clinical suitability.')
    st.write('Data entered in this demo is used for an immediate prediction and is not intentionally stored by this application. Do not enter identifiable patient information.')
