# ♻️ Geodechet – Anticiper la production de déchets par région

## 📌 Contexte
Chaque année, les collectivités doivent gérer plusieurs millions de tonnes de déchets.  
Une mauvaise anticipation entraîne des **surcoûts logistiques**, des **surcharges d’infrastructures** et un **impact environnemental accru**.  

👉 Notre objectif : **utiliser la data science pour prédire les volumes de déchets produits par région et par typologie**, afin d’aider les collectivités à mieux planifier leurs ressources.

---

## 🎯 Objectifs
- Construire un **modèle prédictif** fiable de la production de déchets.  
- Fournir un **outil interactif** permettant de visualiser et tester les prédictions.  
- Explorer les **variables socio-économiques et démographiques** ayant le plus d’influence.  
- Démontrer la valeur business et environnementale d’une approche data-driven.  

---

## 📂 Structure du projet
06_final_project/
├── README.md # Documentation du projet
├── requirements.txt # Dépendances principales
├── docs/
│ └── data_dictionary.csv # Dictionnaire des données
├── notebooks/ # Notebooks d'analyse
│ ├── 0_data_overview.ipynb
│ ├── 1a_eda.ipynb
│ ├── 1b_corrélation.ipynb
│ ├── 2_feature-engineering.ipynb
│ ├── 3_modeling.ipynb
│ └── 4_results_viz.ipynb
├── visuals/ # Graphiques et résultats
├── Geodechet/ # Application déployée
│ ├── Dockerfile
│ ├── requirements.txt
│ ├── README.md
│ └── src/
│ ├── app.py # Application principale
│ └── models/ # Modèles entraînés (.pkl)
└── .env / .gitignore


---

## 📊 Données
- **Source** : données publiques (agrégées par région et par année).  
- **Variables cibles** : tonnages de déchets (verts, encombrants, recyclables, déblais, etc.).  
- **Variables explicatives** :  
  - Démographie (population, densité).  
  - Économie (nombre d’entreprises, secteurs).  
  - Contexte régional.  
- **Préparation** : nettoyage, normalisation, dictionnaire (`docs/data_dictionary.csv`).  

---

## 🛠️ Méthodologie
1. **Exploration (EDA)** → corrélations, heatmaps, tendances.  
2. **Feature engineering** → création et transformation de variables explicatives.  
3. **Modélisation** :    
   - Lasso  
   - OLS statsmodels.  
4. **Évaluation** → métriques (RMSE, R²).  
5. **Déploiement** → application Python conteneurisée avec Docker.  

---

## 📈 Résultats & Insights
- **Performances** : modèles capables de prédire les volumes par typologie avec une précision satisfaisante (cf. notebooks `3_modeling.ipynb` & `4_results_viz.ipynb`).  
- **Variables les plus explicatives** : densité de population, nombre d’entreprises de services, poids des activités industrielles.  
- **Visualisations** : disponibles dans `/visuals`.  

---

## 💻 Démo – Application Geodechet
- **Description** : application interactive permettant de sélectionner une région et d’obtenir les prédictions de tonnage par type de déchet.  
- **Technologies** : Python, Docker, modèles `.pkl`.  
- **Usage** :  
  ```bash
  cd Geodechet
  docker build -t geodechet .
  docker run -p 8501:8501 geodechet
Puis ouvrir http://localhost:8501

## 🌐 Démo en ligne
L’application est disponible directement sur Hugging Face Spaces :  
👉 [Geodechet – Hugging Face Spaces](https://huggingface.co/spaces/PoroKami/Geodechet) 


## 📅 Gestion de projet

Durée : 2 semaines.

Équipe : 4 personnes (collecte/préparation, modélisation, déploiement).

Planning :

Semaine 1 → collecte & nettoyage des données → EDA & visualisations.

Semaine 2 → feature engineering & modélisation → déploiement & livrables.

Pour les collectivités : meilleure planification budgétaire et logistique.

Pour l’économie : réduction des coûts opérationnels.

Pour l’environnement : optimisation de la gestion des déchets et contribution à l’économie circulaire.

## 🚀 Améliorations & Next Steps

Intégrer des données plus fines (département/ville).

Ajouter des données temps réel (capteurs IoT, climat, etc.).

Automatiser les pipelines (ETL, CI/CD).

Proposer la solution en SaaS pour collectivités et entreprises.

Équipe

## 👩‍💻 Projet réalisé dans le cadre du bootcamp Jedha – Fullstack Data Science & Engineering par par ​Nathalie DEVOGELAERE,David JAOUI,François MINARET et moi même Benoit PARADIS-CAMI ​.