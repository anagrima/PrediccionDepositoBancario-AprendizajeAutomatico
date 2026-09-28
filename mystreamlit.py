import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --------------------------------------------------
# Configuración general
# --------------------------------------------------
st.set_page_config(page_title="Predicción de depósito", layout="centered")
st.title("Predicción de suscripción a depósito bancario")
st.write("Aplicación para predecir si un cliente bancario suscribirá un depósito a plazo.")

# --------------------------------------------------
# Cargar modelo
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("modelo_final.joblib")

try:
    modelo_final = load_model()
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# --------------------------------------------------
# Formulario
# --------------------------------------------------
st.subheader("Introducir datos del cliente")

with st.form("prediction_form"):
    age = st.number_input("Edad", min_value=18, max_value=100, value=40, step=1)

    job = st.selectbox(
        "Tipo de trabajo",
        ["admin.", "technician", "services", "management", "retired",
         "blue-collar", "unemployed", "entrepreneur", "housemaid",
         "self-employed", "student", "unknown"]
    )

    marital = st.selectbox(
        "Estado civil",
        ["married", "single", "divorced", "None"]
    )

    education = st.selectbox(
        "Nivel de educación",
        ["primary", "secondary", "tertiary", "unknown"]
    )

    default = st.selectbox("¿Tiene crédito en impago?", ["yes", "no"])
    balance = st.number_input("Balance medio anual", value=1000.0, step=100.0)
    housing = st.selectbox("¿Tiene hipoteca?", ["yes", "no"])
    loan = st.selectbox("¿Tiene préstamo personal?", ["yes", "no"])

    contact = st.selectbox(
        "Tipo de contacto",
        ["cellular", "telephone", "unknown"]
    )

    day = st.number_input("Último día de contacto", min_value=1, max_value=31, value=15, step=1)

    month = st.selectbox(
        "Último mes de contacto",
        ["jan", "feb", "mar", "apr", "may", "jun",
         "jul", "aug", "sep", "oct", "nov", "dec"]
    )

    duration = st.number_input("Duración del último contacto (segundos)", min_value=0, value=200, step=10)
    campaign = st.number_input("Número de contactos en esta campaña", min_value=1, value=1, step=1)

    pdays_input = st.number_input(
        "Días desde el contacto anterior (-1 si no hubo contacto previo)",
        value=-1,
        step=1
    )

    previous = st.number_input("Número de contactos previos", min_value=0, value=0, step=1)

    poutcome = st.selectbox(
        "Resultado de campañas anteriores",
        ["success", "failure", "other", "unknown"]
    )

    submitted = st.form_submit_button("Predecir", use_container_width=True)

# --------------------------------------------------
# Preprocesado 
# --------------------------------------------------
contacted_before = 0 if pdays_input == -1 else 1
pdays = np.nan if pdays_input == -1 else float(pdays_input)

# --------------------------------------------------
# Construcción del dataframe
# --------------------------------------------------
input_df = pd.DataFrame([{
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
    "contacted_before": contacted_before
}])

st.write("### Datos introducidos")
st.dataframe(input_df)

# --------------------------------------------------
# Predicción
# --------------------------------------------------
if submitted:
    try:
        pred = modelo_final.predict(input_df)[0]

        st.write("### Resultado de la predicción")

        if pred in [1, "yes"]:
            st.success("Predicción: el cliente probablemente SÍ suscribirá el depósito.")
        else:
            st.error("Predicción: el cliente probablemente NO suscribirá el depósito.")

        if hasattr(modelo_final, "predict_proba"):
            proba = modelo_final.predict_proba(input_df)[0]
            st.write("### Probabilidades")
            st.write(f"Clase 0: {proba[0]*100:.1f}%")
            st.write(f"Clase 1: {proba[1]*100:.1f}%")

    except Exception as e:
        st.error(f"Error durante la predicción: {e}")