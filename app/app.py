import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime as dt
import os

# Configuración de la página de Streamlit
st.set_page_config(page_title="Prode App", layout="centered")
st.title("Predictor de Resultados de Partidos Internacionales de Fútbol")
st.write("Interfaz gráfica interactiva para estimar marcadores utilizando modelos de Machine Learning.")

# 1. Cargar modelos, encoders y datos históricos

# 1. Carga optimizada de artefactos en caché
@st.cache_resource
def load_assets():
    # Obtener la ruta del directorio donde está app.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construir rutas absolutas correctas a los archivos
    model_home_path = os.path.join(current_dir, "model_home.pkl")
    model_away_path = os.path.join(current_dir, "model_away.pkl")
    team_encoder_path = os.path.join(current_dir, "team_encoder.pkl")
    tournament_encoder_path = os.path.join(current_dir, "tournament_encoder.pkl")
    csv_path = os.path.join(current_dir, "results_proccesed.csv")
    
    # Cargar los recursos
    model_h = joblib.load(model_home_path)
    model_a = joblib.load(model_away_path)
    t_enc = joblib.load(team_encoder_path)
    tourn_enc = joblib.load(tournament_encoder_path)
    
    df_proccesed = pd.read_csv(csv_path, dtype={
        'home_goals_for_5': float,
        'home_goals_against_5': float,
        'away_goals_for_5': float,
        'away_goals_against_5': float
    })
    return model_h, model_a, t_enc, tourn_enc, df_proccesed

try:
    model_home, model_away, team_encoder, tourn_encoder, results_df = load_resources()
except Exception as e:
    st.error(f"Error al cargar recursos. Verifique la existencia de los archivos en sus directorios correspondientes: {e}")
    st.stop()

# Obtener listas únicas ordenadas de países y campeonatos desde los codificadores
listado_equipos = sorted(team_encoder.classes_)
listado_torneos = sorted(tourn_encoder.classes_)

# 2. Componentes de la Interfaz Gráfica (Inputs)
col1, col2 = st.columns(2)

with col1:
    team_A = st.selectbox("Seleccioná el Equipo Local (A):", listado_equipos, index=listado_equipos.index("Netherlands") if "Netherlands" in listado_equipos else 0)

with col2:
    default_away = "Tunisia" if "Tunisia" in listado_equipos else listado_equipos[1]
    team_B = st.selectbox("Seleccioná el Equipo Visitante (B):", listado_equipos, index=listado_equipos.index(default_away))

tournament_name = st.selectbox("Seleccioná el Campeonato / Torneo:", listado_torneos, index=listado_torneos.index("FIFA World Cup") if "FIFA World Cup" in listado_torneos else 0)

is_neutral = st.checkbox("Se juega en terreno neutral?", value=True)


# 3. Lógica interna de extracción de características históricas corregida (operador <=)
# 3. Lógica interna de extracción de características históricas (CORRECCIÓN DE INDEXACIÓN SIMÉTRICA)
# 3. Lógica interna de extracción de características históricas corregida por asignación directa de equipo
def get_match_features_live(home_team, away_team, tournament_name, is_neutral, results_df, team_enc, tourn_enc, model_reference):
    today = pd.to_datetime(dt.today().date())
    
    # 1. Buscar el último partido del equipo LOCAL (independientemente de si fue local o visitante ese día)
    home_last_match = results_df[
        ((results_df['home_team'] == home_team) | (results_df['away_team'] == home_team)) & 
        (pd.to_datetime(results_df['date']) <= today)
    ].sort_values('date').iloc[-1]
    
    # Extraer métricas correspondientes al equipo LOCAL
    if home_last_match['home_team'] == home_team:
        home_goals_for = home_last_match['home_goals_for_5']
        home_goals_against = home_last_match['home_goals_against_5']
    else:
        home_goals_for = home_last_match['away_goals_for_5']
        home_goals_against = home_last_match['away_goals_against_5']
        
    # 2. Buscar el último partido del equipo VISITANTE (independientemente de si fue local o visitante ese día)
    away_last_match = results_df[
        ((results_df['home_team'] == away_team) | (results_df['away_team'] == away_team)) & 
        (pd.to_datetime(results_df['date']) <= today)
    ].sort_values('date').iloc[-1]
    
    # Extraer métricas correspondientes al equipo VISITANTE
    if away_last_match['away_team'] == away_team:
        away_goals_for = away_last_match['away_goals_for_5']
        away_goals_against = away_last_match['away_goals_against_5']
    else:
        away_goals_for = away_last_match['home_goals_for_5']
        away_goals_against = away_last_match['home_goals_against_5']
    
    # Mapeo base de características
    features = {
        'home_team_encoded': team_enc.transform([home_team])[0],
        'away_team_encoded': team_enc.transform([away_team])[0],
        'home_goals_for_5': home_goals_for,
        'home_goals_against_5': home_goals_against,
        'away_goals_for_5': away_goals_for,
        'away_goals_against_5': away_goals_against,
        'tournament_encoded': tourn_enc.transform([tournament_name])[0],
        'neutral': int(is_neutral)
    }
    
    df_features = pd.DataFrame([features])
    
    # Reordenar y renombrar dinámicamente según el modelo cargado
    if hasattr(model_reference, "feature_names_in_"):
        expected_features = model_reference.feature_names_in_
        df_features = df_features.reindex(columns=expected_features)
        for col in expected_features:
            if col not in features and col.lower() in [k.lower() for k in features.keys()]:
                original_key = [k for k in features.keys() if k.lower() == col.lower()][0]
                df_features[col] = features[original_key]
    
    return df_features

# Función simétrica para canchas neutrales sin alteraciones de punteros
def predict_neutral_match_symmetric(team_A, team_B, tournament_name, is_neutral, results_df, team_enc, tourn_enc):
    # Caso 1: A es local, B es visitante
    features_1 = get_match_features_live(team_A, team_B, tournament_name, is_neutral, results_df, team_enc, tourn_enc, model_home)
    pred_home_1 = model_home.predict(features_1)[0]
    pred_away_1 = model_away.predict(features_1)[0]
    
    # Caso 2: B es local, A es visitante
    features_2 = get_match_features_live(team_B, team_A, tournament_name, is_neutral, results_df, team_enc, tourn_enc, model_home)
    pred_home_2 = model_home.predict(features_2)[0]
    pred_away_2 = model_away.predict(features_2)[0]
    
    # Promedio simétrico idéntico al del notebook
    final_goles_team_A = (pred_home_1 + pred_away_2) / 2
    final_goles_team_B = (pred_away_1 + pred_home_2) / 2
    
    prode_A = int(np.round(final_goles_team_A))
    prode_B = int(np.round(final_goles_team_B))
    
    return final_goles_team_A, final_goles_team_B, prode_A, prode_B


# 4. Acción de Predicción
# 4. Acción de Predicción (CORREGIDO: Agregado model_home como referencia en el caso no neutral)
if st.button("Predecir Resultado 🔮", use_container_width=True):
    if team_A == team_B:
        st.warning("Por favor, seleccioná dos países diferentes.")
    else:
        with st.spinner("Calculando estadísticas recientes y ejecutando modelos..."):
            try:
                if is_neutral:
                    goles_A, goles_B, prode_A, prode_B = predict_neutral_match_symmetric(
                        team_A, team_B, tournament_name, is_neutral, results_df, team_encoder, tourn_encoder
                    )
                else:
                    # CORRECCIÓN: Se añade 'model_home' al final para que coincida con la firma de la función
                    features = get_match_features_live(
                        team_A, team_B, tournament_name, is_neutral, results_df, team_encoder, tourn_encoder, model_home
                    )
                    goles_A = model_home.predict(features)[0]
                    goles_B = model_away.predict(features)[0]
                    prode_A = int(np.round(goles_A))
                    prode_B = int(np.round(goles_B))
                
                # Despliegue de Resultados en la UI
                st.success("Predicción completada con éxito.")
                
                st.markdown(f"""
                <div style="background-color:#f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="margin: 0; color: #31333F;">PRODE RESULTADO FINAL</h2>
                    <h1 style="font-size: 50px; margin: 10px 0; color: #FF4B4B;">
                        {team_A} {prode_A} &mdash; {prode_B} {team_B}
                    </h1>
                    <p style="font-style: italic; color: #555;">
                        Predicción Continua Precisa: {team_A} ({goles_A:.2f}) vs {team_B} ({goles_B:.2f})
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except ValueError as val_err:
                st.error(f"Error en los datos de los equipos seleccionados: {val_err}")
            except Exception as e:
                st.error(f"Ocurrió un error inesperado: {e}")