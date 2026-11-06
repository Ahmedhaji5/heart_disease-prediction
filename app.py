from pathlib import Path
import html
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CardioInsight | Heart Disease Research", page_icon="🫀", layout="wide", initial_sidebar_state="expanded")

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], [data-testid="stApp"] {font-family:'DM Sans',sans-serif;}
[data-testid="stAppViewContainer"] {background: #f5f8fc; color:#14233d;}
[data-testid="stSidebar"] {background:#0d203b;}
[data-testid="stSidebar"] * {color:#e8f0fb !important;}
.block-container {max-width:1240px; padding-top:2rem; padding-bottom:3rem;}
#MainMenu, footer {visibility:hidden;}
.hero {background:linear-gradient(112deg,#102a4b 0%,#16416a 58%,#087f8c 100%);color:#fff;border-radius:22px;padding:34px 38px;box-shadow:0 16px 40px rgba(17,45,77,.13);margin-bottom:24px;}
.hero .eyebrow {font-size:.78rem;letter-spacing:.17em;text-transform:uppercase;color:#8ce0e1;font-weight:800;margin-bottom:10px;}
.hero h1 {font-size:clamp(1.8rem,3vw,2.75rem);line-height:1.12;color:#fff;margin:0 0 12px;font-weight:800;}
.hero p {font-size:1rem;color:#d9e8f5;max-width:730px;margin:0;}
.hero-pill {display:inline-block;border:1px solid #75b7c6;border-radius:30px;padding:7px 12px;color:#d9fbf7;font-size:.76rem;margin-top:19px;}
.section-head {font-size:1.23rem;font-weight:800;color:#15324e;margin:12px 0 6px;}
.section-sub {color:#687b90;font-size:.91rem;margin-bottom:18px;}
.kpi {background:#fff;border:1px solid #e2eaf3;border-radius:16px;padding:19px 20px;min-height:116px;box-shadow:0 4px 14px rgba(28,57,91,.04);}
.kpi-label {font-size:.79rem;color:#6a7d92;font-weight:700;text-transform:uppercase;letter-spacing:.06em;}
.kpi-value {font-size:1.75rem;color:#163a5a;font-weight:800;margin-top:7px;}
.kpi-note {font-size:.76rem;color:#8291a4;margin-top:3px;}
.result-card {border-radius:20px;padding:27px 30px;background:#fff;border:1px solid #e1eaf3;box-shadow:0 10px 28px rgba(24,54,83,.07);margin:14px 0;}
.result-label {font-size:.84rem;text-transform:uppercase;letter-spacing:.1em;font-weight:800;color:#657c91;}
.result-number {font-size:3.65rem;line-height:1.1;font-weight:800;color:#123855;margin:12px 0 5px;}
.result-status {font-size:1.12rem;font-weight:800;margin-bottom:9px;}
.result-note {color:#66798d;font-size:.88rem;}
.progress-track {height:15px;background:#eaf0f6;border-radius:50px;overflow:hidden;margin:17px 0 10px;}
.progress-fill {height:100%;border-radius:50px;}
.info-box {border-left:4px solid #13a3a6;background:#e9f6f8;padding:14px 18px;border-radius:0 12px 12px 0;color:#24465b;font-size:.9rem;margin:12px 0 24px;}
.sidebar-brand {font-size:1.55rem;font-weight:800;letter-spacing:-.03em;color:white;margin:8px 0 8px;}
.sidebar-caption {font-size:.86rem;line-height:1.65;color:#c5d8ea;}
[data-testid="stForm"] {background:white;border:1px solid #e0e9f2;border-radius:18px;padding:22px 24px;box-shadow:0 5px 22px rgba(26,59,93,.04);}
[data-testid="stFormSubmitButton"] button {background:linear-gradient(90deg,#087f8c,#0b9a9c);color:white;border:0;border-radius:10px;font-weight:800;min-height:3rem;}
[data-testid="stFormSubmitButton"] button:hover {background:#066b79;color:white;border:0;}
[data-testid="stTabs"] button {font-weight:700;}
@media (max-width:640px) {.hero {padding:24px 20px;}.block-container {padding-left:1rem;padding-right:1rem;}.result-number{font-size:2.8rem;}}
</style>
''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-brand">🫀 CardioInsight</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-caption">Heart disease prediction<br>Logistic regression research dashboard</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown('**ABOUT THIS PROJECT**')
    st.caption('Developed using the UCI Heart Disease dataset, combining 920 records from four contributing hospital datasets.')
    st.markdown('**MODEL**')
    st.caption('Logistic regression · C = 0.1 · 10 input features · Hospital identity excluded')
    st.markdown('**IMPORTANT**')
    st.warning('Educational research demo only. Not a diagnostic tool or medical device.')
    st.caption('Do not enter names or other identifiable patient information.')

st.markdown('''<div class="hero">
<div class="eyebrow">Predictive analytics · Research demonstration</div>
<h1>Heart Disease Prediction</h1>
<p>Explore a logistic regression model through a structured cardiovascular assessment and an easy-to-read probability visualization.</p>
<span class="hero-pill">● UCI Heart Disease Dataset &nbsp; • &nbsp; 920 records &nbsp; • &nbsp; 4 hospital sources</span>
</div>''', unsafe_allow_html=True)

st.markdown('<div class="section-head">Model performance at a glance</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Mean nested leave-one-hospital-out validation scores — performance varied between hospitals.</div>', unsafe_allow_html=True)
for col, label, value, note in zip(st.columns(3), ['Mean accuracy','Mean recall','Mean ROC-AUC'], ['74.3%','83.2%','0.804'], ['Cross-hospital validation','Positive-class sensitivity','Ranking discrimination']):
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>', unsafe_allow_html=True)

model_path = Path(__file__).with_name('heart_disease_logistic_regression.pkl')
if not model_path.exists():
    st.error('The trained model file is missing. Add heart_disease_logistic_regression.pkl to the same folder as app.py.')
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(model_path)

try:
    model = load_model()
except Exception as exc:
    st.error('Unable to load the trained model. Confirm the scikit-learn version matches the saved model (1.6.1).')
    st.exception(exc)
    st.stop()

st.markdown('<div class="section-head" style="margin-top:28px">Patient assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Enter anonymous sample measurements below. Use “Unknown” where available, or 0 for specified numeric fields when missing.</div>', unsafe_allow_html=True)

with st.form('patient'):
    st.markdown('#### 01 · Demographics and symptoms')
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input('Age (years)', min_value=1, max_value=120, value=55)
    with c2:
        sex = st.selectbox('Sex as recorded in dataset', ['Male', 'Female'])
    with c3:
        cp = st.selectbox('Chest pain type', ['typical angina', 'atypical angina', 'non-anginal', 'asymptomatic'])
    st.divider()
    st.markdown('#### 02 · Cardiovascular measurements')
    c1, c2, c3 = st.columns(3)
    with c1:
        trestbps = st.number_input('Resting blood pressure (mm Hg)', min_value=0, max_value=300, value=130, help='Enter 0 if unknown; the model preprocessing will impute it.')
    with c2:
        chol = st.number_input('Serum cholesterol (mg/dL)', min_value=0, max_value=800, value=240, help='Enter 0 if unknown; the model preprocessing will impute it.')
    with c3:
        thalch = st.number_input('Maximum heart rate (beats/min)', min_value=0, max_value=260, value=150, help='Enter 0 if unknown; the model preprocessing will impute it.')
    st.divider()
    st.markdown('#### 03 · ECG and exercise indicators')
    c1, c2 = st.columns(2)
    with c1:
        fbs = st.selectbox('Fasting blood sugar > 120 mg/dL', ['No', 'Yes', 'Unknown'])
        restecg = st.selectbox('Resting ECG', ['normal', 'st-t abnormality', 'lv hypertrophy', 'Unknown'])
    with c2:
        exang = st.selectbox('Exercise-induced angina', ['No', 'Yes', 'Unknown'])
        oldpeak = st.number_input('ST depression (oldpeak)', min_value=-5.0, max_value=10.0, value=1.0, step=0.1)
    st.write('')
    submitted = st.form_submit_button('🫀  Generate prediction', type='primary', use_container_width=True)

if submitted:
    # Preserve the exact feature names, values, missingness mapping, and threshold from the original app.
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
    try:
        prob = float(model.predict_proba(data)[0, 1])
    except Exception as exc:
        st.error('The model could not process these inputs. Check the saved pipeline and input feature types.')
        st.exception(exc)
        st.stop()
    prediction = int(prob >= 0.5)
    accent = '#d97744' if prediction else '#0d9890'
    status = 'Positive classification' if prediction else 'Negative classification'
    st.markdown('<div class="section-head" style="margin-top:30px">Prediction results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Research-model output for the entered sample.</div>', unsafe_allow_html=True)
    st.markdown(f'''<div class="result-card">
        <div class="result-label">Estimated probability of positive class</div>
        <div class="result-number">{prob:.1%}</div>
        <div class="result-status" style="color:{accent}">● {html.escape(status)}</div>
        <div class="progress-track"><div class="progress-fill" style="width:{prob*100:.3f}%;background:{accent}"></div></div>
        <div class="result-note">Classification threshold: 50% · Probability of negative class: {1-prob:.1%}</div>
    </div>''', unsafe_allow_html=True)
    st.markdown('<div class="info-box"><b>How to interpret:</b> This is a model-estimated probability, not a diagnosis or an individual’s clinically validated risk. The 50% threshold is for demonstration only; probabilities have not been externally calibrated.</div>', unsafe_allow_html=True)

st.markdown('<div class="section-head" style="margin-top:28px">Understanding the model</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(['Methodology', 'Validation', 'Limitations & privacy'])
with tab1:
    st.write('The application uses a trained logistic regression pipeline (regularization C=0.1) on 10 features from the UCI Heart Disease dataset. The target is binary: absence versus presence of heart disease in the dataset. The hospital-source variable is not used as a predictor.')
    st.write('Missing values are handled by the saved preprocessing pipeline. Inputs are passed directly to the original model; no retraining takes place in the app.')
with tab2:
    st.write('Nested leave-one-hospital-out evaluation produced mean accuracy of **74.3%**, mean recall of **83.2%**, and mean ROC-AUC of **0.804**. These are averages across held-out hospital datasets, not guarantees of individual prediction accuracy.')
with tab3:
    st.warning('Not a medical device. Predictions must not be used to make clinical decisions or replace professional assessment.')
    st.write('Model performance varies by hospital, and probabilities have not been externally calibrated. The application does not intentionally persist submitted data; avoid entering identifiable patient information.')

st.caption('CardioInsight · Logistic regression research demo · For educational use only')
