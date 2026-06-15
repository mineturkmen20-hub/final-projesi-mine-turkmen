import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="RiskRadar - Musteri Kaybi Risk Asistani", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_columns = joblib.load("models/feature_columns.pkl")
    num_cols = joblib.load("models/num_cols.pkl")
    explainer = shap.TreeExplainer(model)
    return model, scaler, feature_columns, num_cols, explainer

model, scaler, feature_columns, num_cols, explainer = load_artifacts()

binary_map = {
    'gender': {'Female': 0, 'Male': 1},
    'Partner': {'No': 0, 'Yes': 1},
    'Dependents': {'No': 0, 'Yes': 1},
    'PhoneService': {'No': 0, 'Yes': 1},
    'PaperlessBilling': {'No': 0, 'Yes': 1},
}
multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
               'Contract', 'PaymentMethod']

st.title("RiskRadar - Musteri Kaybi Risk Karar Destek Sistemi")
st.write("Musteri bilgilerini girin, kayip (churn) riskini ve nedenlerini gorun.")

with st.form("musteri_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demografik Bilgiler")
        gender = st.selectbox("Cinsiyet", ["Female", "Male"])
        senior = st.selectbox("65 yas ustu mu?", ["No", "Yes"])
        partner = st.selectbox("Partneri var mi?", ["No", "Yes"])
        dependents = st.selectbox("Bakmakla yukumlu kisi var mi?", ["No", "Yes"])
        tenure = st.number_input("Hizmet suresi (ay)", min_value=0, max_value=100, value=12)

    with col2:
        st.subheader("Hizmetler")
        phone_service = st.selectbox("Telefon hizmeti", ["No", "Yes"])
        multiple_lines = st.selectbox("Coklu hat", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet hizmeti", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online guvenlik", ["No internet service", "No", "Yes"])
        online_backup = st.selectbox("Online yedekleme", ["No internet service", "No", "Yes"])
        device_protection = st.selectbox("Cihaz koruma", ["No internet service", "No", "Yes"])
        tech_support = st.selectbox("Teknik destek", ["No internet service", "No", "Yes"])
        streaming_tv = st.selectbox("TV yayini", ["No internet service", "No", "Yes"])
        streaming_movies = st.selectbox("Film yayini", ["No internet service", "No", "Yes"])

    with col3:
        st.subheader("Sozlesme ve Odeme")
        contract = st.selectbox("Sozlesme tipi", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Kagitsiz fatura", ["No", "Yes"])
        payment = st.selectbox("Odeme yontemi", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        monthly_charges = st.number_input("Aylik ucret ($)", min_value=0.0, max_value=200.0, value=70.0)
        total_charges = st.number_input("Toplam ucret ($)", min_value=0.0, max_value=10000.0, value=840.0)

    submitted = st.form_submit_button("Risk Hesapla")

if submitted:
    raw = {
        'gender': gender, 'SeniorCitizen': 1 if senior == "Yes" else 0,
        'Partner': partner, 'Dependents': dependents, 'tenure': tenure,
        'PhoneService': phone_service, 'MultipleLines': multiple_lines,
        'InternetService': internet_service, 'OnlineSecurity': online_security,
        'OnlineBackup': online_backup, 'DeviceProtection': device_protection,
        'TechSupport': tech_support, 'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies, 'Contract': contract,
        'PaperlessBilling': paperless, 'PaymentMethod': payment,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([raw])

    for col, mapping in binary_map.items():
        input_df[col] = input_df[col].map(mapping)

    input_df = pd.get_dummies(input_df, columns=multi_cols)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    proba = model.predict_proba(input_df)[0, 1]

    if proba < 0.3:
        risk_level = "Dusuk Risk"
    elif proba < 0.6:
        risk_level = "Orta Risk"
    else:
        risk_level = "Yuksek Risk"

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Kayip (Churn) Riski", f"%{proba*100:.1f}")
    with col_b:
        st.metric("Risk Seviyesi", risk_level)

    st.subheader("Bu Tahmini Etkileyen Faktorler")
    shap_values = explainer(input_df)
    shap.plots.waterfall(shap_values[0, :, 1], show=False)
    fig = plt.gcf()
    st.pyplot(fig)
    plt.clf()