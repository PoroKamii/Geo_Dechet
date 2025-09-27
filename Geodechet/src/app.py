# =========================
#  App Streamlit - GeoDéchet
# =========================

# -- Evite les erreurs de permissions sur Hugging Face Spaces --
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

import re
import unicodedata
from io import StringIO
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import shap
import streamlit as st
from botocore.config import Config
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from sklearn.linear_model import LinearRegression


# ========== Helpers robustes ==========
def _norm(s: str) -> str:
    """Normalise une chaîne (BOM/accents/espaces/casse)."""
    if s is None:
        return ""
    s = str(s).replace("\ufeff", "")  # BOM éventuel
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")  # enlève accents
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def find_col(df: pd.DataFrame, *aliases: str) -> str:
    """Trouve le nom réel d'une colonne à partir d'alias ('Département'/'Departement', 'année'/'annee')."""
    targets = {_norm(a) for a in aliases}
    for c in df.columns:
        if _norm(c) in targets:
            return c
    return ""


def get_numeric_value(series: pd.Series) -> float:
    """Convertit proprement une série (potentiellement str avec espaces, virgule décimale) en float."""
    s = series.astype(str)
    s = (
        s.str.replace("\u00a0", " ", regex=False)  # nbsp
        .str.replace("\u202f", " ", regex=False)  # thin space
        .str.replace(" ", "", regex=False)  # supprime espaces milliers
        .str.replace(",", ".", regex=False)  # virgule -> point
    )
    val = pd.to_numeric(s, errors="coerce")
    return float(val.iloc[0]) if len(val) and pd.notna(val.iloc[0]) else 0.0


def read_csv_robust(
    body_bytes: bytes, default_sep: str = ",", try_utf8sig_first: bool = True
) -> pd.DataFrame:
    """Lecture robuste : essaie utf-8-sig puis latin-1, set sep, et répare la mojibake si besoin."""
    df = None
    if try_utf8sig_first:
        try:
            df = pd.read_csv(StringIO(body_bytes.decode("utf-8-sig")), sep=default_sep)
        except Exception:
            pass
    if df is None:
        try:
            df = pd.read_csv(StringIO(body_bytes.decode("latin-1")), sep=default_sep)
        except Exception:
            # dernier fallback: utf-8 simple
            df = pd.read_csv(StringIO(body_bytes.decode("utf-8")), sep=default_sep)

    # Nettoyage BOM dans colonnes
    cols = [c.replace("\ufeff", "") for c in df.columns]
    # Répare si mojibake du style DÃ©partement, annÃ©e, ï»¿Code, etc.
    if any(("Ã" in c) or ("ï»¿" in c) or ("Â" in c) for c in cols):
        cols = [
            c.encode("latin-1", "ignore")
            .decode("utf-8", "ignore")
            .replace("\ufeff", "")
            for c in cols
        ]
    df.columns = cols
    return df


# # ========== Connexion S3 ==========
# def clean(s):
#     return (s or "").strip().replace("\r", "").replace("\n", "")


# session = boto3.Session(
#     aws_access_key_id=clean(os.getenv("AWS_ACCESS_KEY_ID")),
#     aws_secret_access_key=clean(os.getenv("AWS_SECRET_ACCESS_KEY")),
#     region_name="eu-west-3",
# )
# s3 = session.client("s3", config=Config(signature_version="s3v4"))
# BUCKET = "mygeodechetbuckets3"

# ========== Lecture des données ==========
# df_dummies_2019.csv : UTF-8-SIG, séparateur virgule
obj = "data/df_dummies_2019.csv"
df = read_csv_robust(obj["Body"].read(), default_sep=",", try_utf8sig_first=True)
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# data_wip_v5.csv : UTF-8-SIG (actuel), séparateur point-virgule (si changé, le reader s'adapte)
obj2 = "data/data_wip_v5.csv"
observed_df = read_csv_robust(
    obj2["Body"].read(), default_sep=";", try_utf8sig_first=True
)

# Debug léger (commenter quand tout est ok)
# st.write("Colonnes observed_df:", list(observed_df.columns))
# st.write("Colonnes df:", list(df.columns)[:15])

# Colonnes robustes pour département/année
DEPT_COL = find_col(observed_df, "Département", "Departement")
YEAR_COL = find_col(observed_df, "année", "annee")

# ========== Chargement centralisé des modèles ==========
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


def load_model(filename: str):
    path = MODELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Introuvable: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


MODELS = {
    "Déblais et Gravats": load_model("deblais_gravats.pkl"),
    "Déchets verts": load_model("dechets_verts.pkl"),
    "Encombrants": load_model("encombrants.pkl"),
    "Matériaux recyclables": load_model("materiaux_recyclables.pkl"),
    "Total autres déchets": load_model("total_autres_dechets.pkl"),
}

col_mapping = {
    "Déblais et Gravats": "Déblais_gravats",
    "Déchets verts": "Déchets_verts",
    "Encombrants": "Encombrants",
    "Matériaux recyclables": "Matériaux_recyclables",
    "Total autres déchets": "Total_autres_dechets",
}

# ========== UI ==========
st.set_page_config(layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>♻️ Simulateur de production de déchets par département</h1>",
    unsafe_allow_html=True,
)

top_col1, top_col2 = st.columns([1, 2])
with top_col1:
    st.markdown(
        "<h3 style='text-align: center;'>📍 Choix du département</h3>",
        unsafe_allow_html=True,
    )
with top_col2:
    st.markdown(
        "<h3 style='text-align: center;'>📈 Comparaison entre valeurs observées et prédites</h3>",
        unsafe_allow_html=True,
    )

# Liste des départements depuis df_dummies
departements = [
    c.replace("Département_", "") for c in df.columns if c.startswith("Département_")
]

top_input_col, chart_col = st.columns([1, 2])

with top_input_col:
    selected_dept = st.selectbox(
        "Sélectionner un département",
        sorted(departements),
        index=sorted(departements).index("Ain") if "Ain" in departements else 0,
    )
    row_default = df[df[f"Département_{selected_dept}"] == 1].iloc[0]
    default_dict = row_default.to_dict()

    st.subheader("⚙️ Paramètres modifiables")
    form_input = {}

    categories = {
        "📊 Population": [
            "pop_globale_n-2",
            "pop_globale",
            "tranche_age_0-24",
            "tranche_age_25-59",
            "tranche_age_60+",
            "csp1_agriculteurs",
            "csp2_artisans_commerçant_chef_entreprises",
            "csp3_cadres_professions_intellectuelles",
            "csp4_professions_intermédiaires",
            "csp5_employés",
            "csp6_ouvriers",
            "csp7_retraités",
            "csp8_sans_activité",
            "densité_n-2",
            "densité",
        ],
        "🏭 Activité économique": [
            "nbre_entreprises",
            "nbre_entreprises_agricole",
            "nb_salaries_secteur_agricole",
            "nbre_entreprises_industrie",
            "nb_salaries_secteur_industrie",
            "nb_salaries_secteur_service",
            "nbre_entreprises_service",
        ],
        "🗑️ Déchets": [
            "tonnage_dechet_produit",
            "Matériaux_recyclables",
            "Déblais_gravats",
            "Déchets_verts",
            "Encombrants",
            "Total_autres_dechets",
            "tonnage_dechet_produit_n-2",
            "Matériaux_recyclables_n-2",
            "Déblais_gravats_n-2",
            "Déchets_verts_n-2",
            "Encombrants_n-2",
            "Total_autres_dechets_n-2",
        ],
    }

    for category_name, variables in categories.items():
        with st.expander(category_name, expanded=True):
            for var in variables:
                if var in default_dict:
                    default_value = int(round(float(default_dict[var])))
                    max_value = int(max(1, default_value * 1.5))
                    val = st.slider(
                        f"🔧 {var}",
                        min_value=0,
                        max_value=max_value,
                        value=default_value,
                        step=1,
                        key=f"slider_{var}",
                    )
                    form_input[var] = int(val)

    # Build input row pour la prédiction
    input_df = pd.DataFrame([form_input])
    input_df_complete = row_default.to_frame().T.copy()
    for col in input_df.columns:
        if col in input_df_complete.columns:
            input_df_complete.at[input_df_complete.index[0], col] = input_df.at[0, col]

with chart_col:
    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    btn_col = st.columns([3, 2, 3])[1]
    with btn_col:
        run_eval = st.button("🔍 Lancer l'évaluation")

    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)

    valeurs_observees, valeurs_predites, labels = [], [], []

    if run_eval:
        run_eval = False
        for typologie, model in MODELS.items():
            try:
                expected_cols = model.model.exog_names
                if (
                    "const" in expected_cols
                    and "const" not in input_df_complete.columns
                ):
                    input_df_complete["const"] = 1.0

                # Prédiction
                pred = max(0, model.predict(input_df_complete[expected_cols]).iloc[0])
                valeurs_predites.append(float(pred))
                labels.append(typologie)

                # Observé (2019) — robustesse sur département/année + nom de typologie
                if not DEPT_COL or not YEAR_COL:
                    st.error(
                        f"Colonnes 'Département/année' introuvables. Colonnes: {list(observed_df.columns)}"
                    )
                    valeurs_observees.append(0.0)
                else:
                    sel_norm = _norm(selected_dept)
                    mask = (
                        observed_df[DEPT_COL].astype(str).map(_norm) == sel_norm
                    ) & (pd.to_numeric(observed_df[YEAR_COL], errors="coerce") == 2019)
                    filtered = observed_df[mask]

                    excel_col_raw = col_mapping.get(typologie)  # ex: "Déblais_gravats"
                    excel_col_norm = _norm(excel_col_raw)
                    cand_cols = [
                        c for c in observed_df.columns if _norm(c) == excel_col_norm
                    ]
                    target_col = cand_cols[0] if cand_cols else ""

                    if not filtered.empty and target_col:
                        valeurs_observees.append(
                            get_numeric_value(filtered[target_col].iloc[:1])
                        )
                    else:
                        valeurs_observees.append(0.0)

            except Exception as e:
                st.error(f"Erreur avec le modèle {typologie}")
                st.exception(e)

    if valeurs_observees and valeurs_predites:
        # Force float
        valeurs_observees = [float(v) for v in valeurs_observees]
        valeurs_predites = [float(v) for v in valeurs_predites]

        x = np.arange(len(labels))
        width = 0.4
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(
            x - width / 2,
            valeurs_observees,
            width,
            label="Observé (2019)",
            color="steelblue",
        )
        bar_colors = [
            (1, 0, 0, 0.6) if pred > obs else (0, 0.6, 0, 0.6)
            for pred, obs in zip(valeurs_predites, valeurs_observees)
        ]
        ax.bar(
            x + width / 2, valeurs_predites, width, label="Prévision", color=bar_colors
        )

        for i in range(len(labels)):
            ax.text(
                x[i] - width / 2,
                valeurs_observees[i] + max(valeurs_observees) * 0.01,
                f"{valeurs_observees[i]:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
            ax.text(
                x[i] + width / 2,
                valeurs_predites[i] + max(valeurs_predites) * 0.01,
                f"{valeurs_predites[i]:,.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_ylabel("Tonnes")
        ax.set_title("Comparaison Observé vs Prédit")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")


# ========== SHAP ==========
st.subheader(
    f"📉 SHAP - Analyse des contributions pour le département : {selected_dept}"
)

selected_typologie = st.selectbox(
    "Choisissez une typologie de déchets à analyser avec SHAP :", list(MODELS.keys())
)
typologie = selected_typologie
model_sm = MODELS[typologie]

try:
    used_features = model_sm.model.exog_names
    used_features_no_const = [f for f in used_features if f != "const"]
    X_used = df[used_features_no_const].copy()

    if "const" in used_features:
        X_used["const"] = 1.0

    intercept = model_sm.params["const"] if "const" in model_sm.params else 0
    coefs = model_sm.params[used_features_no_const].values

    lr = LinearRegression()
    lr.intercept_ = float(intercept)
    lr.coef_ = np.array(coefs, dtype=float)
    lr.feature_names_in_ = np.array(used_features_no_const)

    X_used_corrected = X_used.reindex(columns=lr.feature_names_in_, fill_value=0)

    explainer = shap.Explainer(lr, X_used_corrected)
    shap_values = explainer(X_used_corrected)

    selected_index = df[df[f"Département_{selected_dept}"] == 1].index[0]

    # On exclut les cibles (observées et n-2) des graphes SHAP (mais pas de la prédiction)
    exclude_vars = [
        "tonnage_dechet_produit_n-2",
        "tonnage_dechet_produit",
        "Total_autres_dechets_n-2",
        "Total_autres_dechets",
        "Déblais_gravats_n-2",
        "Déblais_gravats",
        "Déchets_verts_n-2",
        "Déchets_verts",
        "Encombrants_n-2",
        "Encombrants",
        "Matériaux_recyclables_n-2",
        "Matériaux_recyclables",
    ]
    mask = np.array([name not in exclude_vars for name in shap_values.feature_names])
    filtered_shap = shap.Explanation(
        values=shap_values.values[:, mask],
        base_values=shap_values.base_values,
        data=shap_values.data[:, mask],
        feature_names=[
            name for name in shap_values.feature_names if name not in exclude_vars
        ],
    )

    col1 = st.columns(1)[0]
    with col1:
        st.markdown(
            "<h6 style='text-align: center;'>🩜 Waterfall</h6>", unsafe_allow_html=True
        )
        fig = plt.figure(figsize=(3, 2))
        shap.plots.waterfall(filtered_shap[selected_index], max_display=10, show=False)
        st.pyplot(fig, bbox_inches="tight", dpi=200, clear_figure=True)

except Exception as e:
    st.error(f"Erreur dans le SHAP pour {typologie}")
    st.exception(e)


# ========== Interprétation LLM (Mistral) ==========
try:
    mean_shap_values = shap_values.abs.mean(0).values
    feature_names = lr.feature_names_in_
    sorted_indices = mean_shap_values.argsort()[::-1]
    top_n = 10
    list_coef = "\n".join(
        [
            f"{feature_names[i]}: {float(mean_shap_values[i]):.4f}"
            for i in sorted_indices[:top_n]
        ]
    )

    prompt_template = f"""
Tu es un expert en data science et en statistique, spécialisé dans l'interprétation de modèles explicatifs avec des coefficients de Shapley.

Je te fournis les contributions SHAP moyennes des variables d'un modèle linéaire.

Ta mission :
- Identifie les variables qui ont le plus d’impact (positif/négatif) sur la variable cible.
- Mets en évidence les tendances démographiques/économiques.
- Reste clair pour un public non expert, tout en étant rigoureux.

Coefficients (SHAP mean abs) :
{list_coef}

Contexte :
- Objectif : Comprendre comment les caractéristiques démographiques et économiques influencent la production des déchets.
- Variable cible : {typologie}
- Modèle : OLS (Statsmodels) interprété via SHAP (approx. linéaire).
- Variables explicatives (exemples) :
  secteurs (salariés agricole/industrie/service), CSP (1..8), tranches d’âge (0-24/25-59/60+),
  population/densité, historiques à n-2.
- Ne commente pas les variables cibles/exclues suivantes : {exclude_vars}.
"""

    load_dotenv()
    api_key = os.getenv("MISTRAL_API_KEY")

    if api_key:
        with st.spinner("🧠 Génération de l'interprétation avec Mistral..."):
            model_llm = ChatMistralAI(
                model="mistral-small-latest", mistral_api_key=api_key
            )
            parser = StrOutputParser()
            response = model_llm.invoke(prompt_template)
            explanation_text = parser.invoke(response)

        st.markdown("#### 🤖 Interprétation automatique (LLM)")
        st.success(explanation_text)
    else:
        st.info(
            "ℹ️ MISTRAL_API_KEY manquant dans l'environnement. Ajoute ta clé pour activer l'interprétation automatique."
        )

except Exception as e:
    st.error("Erreur lors de l'analyse SHAP / appel au LLM.")
    st.exception(e)
