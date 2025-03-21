import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Charger les modèles
reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")

# Définition des colonnes utilisées lors de l'entraînement
categorical_cols = ["make", "fuel_type", "num_doors", "body_style"]
numerical_cols = ["horsepower", "city_mpg"]

# Liste des catégories possibles (basée sur l'entraînement)
categories = {
    "make": ["audi", "bmw", "toyota", "honda", "mercedes"],
    "fuel_type": ["diesel", "essence"],
    "num_doors": ["two", "four"],
    "body_style": ["sedan", "hatchback", "wagon"]
}

def preprocess_input(input_data):
    """Transforme l'entrée utilisateur en un dataframe aligné avec les colonnes du modèle."""
    input_df = pd.DataFrame([input_data])
    input_df = pd.get_dummies(input_df)
    
    # Ajouter les colonnes manquantes avec des valeurs 0
    for col in categories:
        for category in categories[col]:
            col_name = f"{col}_{category}"
            if col_name not in input_df.columns:
                input_df[col_name] = 0
    
    # Assurer l'ordre des colonnes
    expected_columns = [col for col in reg_model.feature_names_in_]
    input_df = input_df.reindex(columns=expected_columns, fill_value=0)
    
    return input_df

# Titre de l'application
st.title("Prédiction du prix des voitures")

# Description de l'application
st.markdown(
    """
    Bienvenue dans cette application de prédiction du prix des voitures ! 🚗💰
    
    Cette application vous permet d'estimer le prix d'une voiture en fonction de ses caractéristiques. 
    Il vous suffit de remplir les informations ci-dessous et d'appuyer sur le bouton **Prédire**.
    
    L'algorithme utilisé repose sur un modèle de régression et un modèle de classification 
    qui permettent d'estimer à la fois le prix de la voiture et sa catégorie de prix (abordable ou chère).
    """
)

# Interface utilisateur pour entrer les données
st.header("Caractéristiques du véhicule")
make = st.selectbox("Marque", categories["make"])
fuel_type = st.selectbox("Type de carburant", categories["fuel_type"])
num_doors = st.selectbox("Nombre de portes", categories["num_doors"])
body_style = st.selectbox("Style de carrosserie", categories["body_style"])
horsepower = st.number_input("Puissance (ch)", min_value=50, max_value=500, value=100)
city_mpg = st.number_input("Consommation urbaine (mpg)", min_value=10, max_value=50, value=25)

# Bouton de prédiction
if st.button("Prédire"):
    input_data = {
        "make": make,
        "fuel_type": fuel_type,
        "num_doors": num_doors,
        "body_style": body_style,
        "horsepower": horsepower,
        "city_mpg": city_mpg
    }
    input_df = preprocess_input(input_data)
    
    # Prédiction
    price_pred = reg_model.predict(input_df)[0]
    class_pred = clf_model.predict(input_df)[0]
    class_label = "Chère" if class_pred == 1 else "Abordable"
    
    # Affichage des résultats
    st.header("Résultats de la prédiction")
    st.success(f"**Prix prédit :** ${price_pred:.2f}")
    st.info(f"**Catégorie de prix :** {class_label}")
