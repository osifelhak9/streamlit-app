# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("loan_data.csv")

@st.cache_resource
def load_model(model_name):
    return model_name

df = load_data()

st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("## Loan Approval App")

model_choice = st.sidebar.selectbox(
    "Choisir le modèle",
    ["Logistic Regression", "Random Forest"]
)

min_income = int(df["ApplicantIncome"].min())
max_income = int(df["ApplicantIncome"].max())

income_range = st.sidebar.slider(
    "Filtrer par revenu",
    min_income,
    max_income,
    (min_income, max_income)
)

education_filter = st.sidebar.selectbox(
    "Filtrer par éducation",
    ["Tous"] + list(df["Education"].dropna().unique())
)

filtered_df = df[
    (df["ApplicantIncome"] >= income_range[0]) &
    (df["ApplicantIncome"] <= income_range[1])
]

if education_filter != "Tous":
    filtered_df = filtered_df[filtered_df["Education"] == education_filter]

model = load_model(model_choice)

st.title("🏦 Prédiction d'Approbation de Prêt")
st.write("Application de Machine Learning pour évaluer les demandes de prêt")

tab1, tab2, tab3 = st.tabs([
    "📊 Exploration des données",
    "🤖 Prédiction",
    "📈 Performance du modèle"
])

with tab1:
    st.subheader("📊 Exploration des données")

    total_demandes = len(filtered_df)
    approved_mask = filtered_df["Loan_Status"].astype(str).isin(["Y", "Yes", "Approved", "1"])
    taux_approbation = approved_mask.mean() * 100
    montant_moyen = filtered_df["LoanAmount"].mean()
    revenu_moyen = filtered_df["ApplicantIncome"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nombre total de demandes", total_demandes)
    c2.metric("Taux d'approbation global", f"{taux_approbation:.1f}%")
    c3.metric("Montant moyen des prêts", f"{montant_moyen:.1f}")
    c4.metric("Revenu moyen", f"{revenu_moyen:,.0f}")

    st.markdown("### 📈 Distributions")
    col1, col2 = st.columns(2)

    with col1:
        fig_hist = px.histogram(
            filtered_df,
            x="ApplicantIncome",
            nbins=30,
            title="Histogramme des revenus"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        fig_box = px.box(
            filtered_df,
            y="LoanAmount",
            title="Box plot des montants de prêt"
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown("### 📊 Analyses")
    col3, col4 = st.columns(2)

    with col3:
        temp_df = filtered_df.copy()
        temp_df["Approved"] = temp_df["Loan_Status"].astype(str).isin(["Y", "Yes", "Approved", "1"]).astype(int)
        approval_by_edu = temp_df.groupby("Education")["Approved"].mean().reset_index()
        approval_by_edu["Approved"] = approval_by_edu["Approved"] * 100

        fig_bar = px.bar(
            approval_by_edu,
            x="Education",
            y="Approved",
            title="Taux d'approbation par éducation",
            labels={"Approved": "Taux (%)"}
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col4:
        pie_df = filtered_df.copy()
        pie_df["Decision"] = pie_df["Loan_Status"].astype(str).apply(
            lambda x: "Approved" if x in ["Y", "Yes", "Approved", "1"] else "Rejected"
        )

        fig_pie = px.pie(
            pie_df,
            names="Decision",
            title="Répartition Approved / Rejected"
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("### 🔥 Corrélations")
    numeric_df = filtered_df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig_heatmap = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale="Blues",
            text=corr.round(2).values,
            texttemplate="%{text}"
        )
    )
    fig_heatmap.update_layout(title="Heatmap de corrélation")
    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.markdown("### 🗂 Dataset filtré")
    st.dataframe(filtered_df)

with tab2:
    st.subheader("🤖 Prédiction")
    st.success("Partie prédiction à compléter après.")

with tab3:
    st.subheader("📈 Performance du modèle")
    st.info(f"Modèle sélectionné : {model}")