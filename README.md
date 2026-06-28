# Prode App (Machine Learning baseline)

Este proyecto desarrolla un sistema de Machine Learning diseñado para predecir resultados exactos de partidos de fútbol internacional, optimizado específicamente para la predicción de estructuras de tipo Prode (formatos de resultado 1X2). El dataset que se utilizó para entrenar los modelos fue obtenido de Kaggle y puede consultarse [aquí](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)

## Resumen del Proyecto y Performance
* **Métrica Objetivo:** Predicción independiente de goles mediante regresión distribuida por equipo.
* **Algoritmo utilizado:** Random Forest Regressor.
* **Precisión:** **47.8%** de acierto en el conjunto de test temporal.
* **Validación:** Split temporal estricto para simular escenarios de producción reales, evitando data leaks de transferencias cronológicas.

---


## Decisiones de Ingeniería y Arquitectura de Datos

### 1. Feature Engineering:
El dataset original se transformó mediante el cálculo de medias móviles basadas exclusivamente en los últimos **5 partidos globales** de cada selección antes de la disputa del encuentro objetivo. Las variables críticas construidas incluyen:
* `home_goals_for_5` / `home_goals_against_5`: Capacidad ofensiva/defensiva reciente del equipo local.
* `away_goals_for_5` / `away_goals_against_5`: Capacidad ofensiva/defensiva reciente del equipo visitante.

### 2. Sistema de pesos:
Para evitar que los partidos disputados en la década de los 2000 tengan el mismo impacto que los encuentros modernos, se diseñó e implementó un sistema de pesos exponenciales. El peso de cada registro decrece matemáticamente conforme aumenta su distancia respecto a la fecha actual. Vale aclarar que para el entrenamiento de los modelos, se utilizaron datos del año 2000 en adelante. Además de eso, el contexto del partido determina también su peso, siendo los partidos de la Copa Mundial de la FIFA aquellos con mas peso, seguidos por torneos que no sean el mundial, y por último partidos amistosos.

### 3. Inferencia Espejo
Para neutralizar el sesgo estructural que los árboles de decisión heredan de la asimetría posicional del dataset (Local/Visitante) en partidos disputados en terreno neutral, la app ejecuta una **Inferencia Espejo**. Se computan las predicciones alternando los roles de los equipos y se promedian los goles continuos resultantes, garantizando consistencia analítica estricta.

---

## Despliegue
El modelo se encuentra productivizado mediante una interfaz web interactiva desarrollada en **Streamlit**, permitiendo a los usuarios seleccionar dos países y simular el marcador exacto proyectado en tiempo real sin interactuar con el código base.

Desarrollado por Federico Rambo, 2026.
