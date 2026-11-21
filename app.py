
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="CreditRisk AI", page_icon="🏦", layout="wide")

st.title("🏦 CreditRisk AI")
st.subheader("Application de prédiction du risque de crédit")

if not os.path.exists("credit_model.pkl"):
    st.warning("Le modèle n'est pas encore entraîné. Exécutez model.py.")
    st.stop()

model = joblib.load("credit_model.pkl")

st.write("Renseignez les informations du demandeur.")

# Formulaire
with st.form("credit_form"):
    duration = st.number_input("Durée du crédit (mois)", 1, 100, 12)
    amount = st.number_input("Montant du crédit", 100, 100000, 5000)
    age = st.number_input("Âge", 18, 100, 30)
    rate = st.slider("Taux d'échelonnement", 1, 4, 2)
    residence = st.slider("Ancienneté du logement", 1, 4, 2)
    credits = st.slider("Crédits existants", 1, 4, 1)
    dependents = st.slider("Personnes à charge", 1, 2, 1)

    checking = st.selectbox("Compte courant", ["A11", "A12", "A13", "A14"])
    history = st.selectbox("Historique de crédit", ["A30", "A31", "A32", "A33", "A34"])
    purpose = st.selectbox("Objet du crédit", ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"])
    savings = st.selectbox("Épargne", ["A61", "A62", "A63", "A64", "A65"])
    employment = st.selectbox("Emploi", ["A71", "A72", "A73", "A74", "A75"])
    status = st.selectbox("Situation personnelle", ["A91", "A92", "A93", "A94", "A95"])
    debtors = st.selectbox("Autres débiteurs", ["A101", "A102", "A103"])
    property_type = st.selectbox("Patrimoine", ["A121", "A122", "A123", "A124"])
    plans = st.selectbox("Autres plans de paiement", ["A141", "A142", "A143"])
    housing = st.selectbox("Logement", ["A151", "A152", "A153"])
    job = st.selectbox("Profession", ["A171", "A172", "A173", "A174"])
    telephone = st.selectbox("Téléphone", ["A191", "A192"])
    foreign = st.selectbox("Travailleur étranger", ["A201", "A202"])

    submitted = st.form_submit_button("Évaluer le risque")

if submitted:
    sample = pd.DataFrame([{
        "checking_account": checking,
        "duration": duration,
        "credit_history": history,
        "purpose": purpose,
        "credit_amount": amount,
        "savings": savings,
        "employment": employment,
        "installment_rate": rate,
        "personal_status": status,
        "other_debtors": debtors,
        "residence_since": residence,
        "property": property_type,
        "age": age,
        "other_installment_plans": plans,
        "housing": housing,
        "existing_credits": credits,
        "job": job,
        "dependents": dependents,
        "telephone": telephone,
        "foreign_worker": foreign
    }])

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    st.subheader("Résultat de l'analyse")
    st.metric("Probabilité de bon crédit estimée", f"{probability:.1%}")

    if prediction == 1:
        st.success("Profil classé dans la catégorie bon crédit.")
    else:
        st.warning("Profil classé dans la catégorie risque de crédit.")
    st.caption("Résultat expérimental : ne constitue pas une décision financière.")

    