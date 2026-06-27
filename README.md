# International Football Predictor (Machine Learning baseline)

Este proyecto desarrolla un sistema de Machine Learning end-to-end diseñado para predecir marcadores exactos de partidos de fútbol internacional, optimizado específicamente para la predicción de estructuras de tipo Prode (formatos de resultado 1X2). El dataset que se utilizó para entrenar los modelos fue obtenido de Kaggle y puede consultarse [aquí](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

## Resumen del Proyecto y Performance
* **Métrica Objetivo:** Predicción independiente de goles mediante regresión distribuida por equipo.
* **Algoritmo Base:** Random Forest Regressor.
* **Precisión Global (1X2):** **48.2%** de Acierto en el conjunto de test temporal.
* **Estrategia de Validación:** Split temporal estricto para simular escenarios de producción reales, evitando data leaks de transferencias cronológicas.

---


## Decisiones de Ingeniería y Arquitectura de Datos

### 1. Feature Engineering Móvil (Ventanas de Tiempo)
El dataset original se transformó mediante el cálculo de medias móviles basadas exclusivamente en los últimos **5 partidos globales** de cada selección antes de la disputa del encuentro objetivo. Las variables críticas construidas incluyen:
* `home_goals_for_5` / `home_goals_against_5`: Capacidad ofensiva/defensiva reciente del equipo local.
* `away_goals_for_5` / `away_goals_against_5`: Capacidad ofensiva/defensiva reciente del equipo visitante.

### 2. Ponderación Temporal de Observaciones (Decay Rate)
Para evitar que los partidos disputados en la década de los 2000 tengan el mismo impacto que los encuentros modernos, se diseñó e implementó un sistema de pesos exponenciales (`sample_weight`). La importancia de cada registro decrece matemáticamente conforme aumenta su distancia respecto a la fecha actual. Vale aclarar que para el entrenamiento de los modelos, se utilizaron datos del año 2000 en adelante.

### 3. Inferencia Espejo (Symmetric Testing)
Para neutralizar el sesgo estructural que los árboles de decisión heredan de la asimetría posicional del dataset (Local/Visitante) en partidos disputados en terreno neutral (`neutral=True`), el pipeline ejecuta una **Inferencia Espejo**. Se computan las predicciones alternando los roles de los equipos y se promedian los goles continuos resultantes, garantizando consistencia analítica estricta.

---

## Despliegue (Web App)
El modelo se encuentra productivizado mediante una interfaz web interactiva desarrollada en **Streamlit**, permitiendo a los usuarios seleccionar dos países y simular el marcador exacto proyectado en tiempo real sin interactuar con el código base.