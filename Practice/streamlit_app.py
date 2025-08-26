import streamlit as st
import pandas as pd
import joblib

st.title('Telco Customer Churn Predictor')
st.write("""
Приложение позволяет загружать CSV-файл с данными клиентов Telco и предсказывать вероятность ухода клиента (Churn).
""")

# Загружаем модель
@st.cache_resource
def load_model(path='best_random_forest_pipeline.joblib'):
    return joblib.load(path)

model = load_model()

# Загрузка CSV
uploaded_file = st.file_uploader("Выберите CSV файл", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Первые 5 строк загруженного файла:")
    st.dataframe(df.head())

    # Проверяем, есть ли нужные колонки
    needed_cols = [c for c in model.named_steps['preprocessor'].feature_names_in_]
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        st.error(f"Отсутствуют необходимые колонки для модели: {missing_cols}")
    else:
        # Прогноз
        preds_proba = model.predict_proba(df[needed_cols])[:,1]
        df['Churn_Probability'] = preds_proba
        st.write("Предсказанная вероятность Churn (первые 20):")
        st.dataframe(df[['Churn_Probability']].head(20))

        # Визуализация распределения
        st.bar_chart(df['Churn_Probability'].head(50))
