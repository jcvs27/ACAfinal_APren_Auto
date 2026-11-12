
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Dashboard LSTM de Ventas', layout='wide')

st.title('📊 Dashboard de Ventas y Predicción LSTM')
st.write('Esta aplicación integra la base de datos de SQLite, realiza inferencias en tiempo real y evalúa cuantitativamente el desempeño del modelo.')

# 1. Función para cargar y procesar datos históricos reales de la DB
def cargar_secuencia_historica():
    try:
        conn = sqlite3.connect('sales_data.db')
        df_sql = pd.read_sql_query("SELECT * FROM sales_raw", conn)
        conn.close()

        # Procesamiento idéntico al entrenamiento
        df_sql['Order Date'] = pd.to_datetime(df_sql['Order Date'], dayfirst=True)
        df_sql = df_sql.sort_values(by='Order Date').reset_index(drop=True)
        df_diario = df_sql.groupby('Order Date')['Sales'].sum().reset_index()

        # Garantizar continuidad temporal completa
        fecha_min, fecha_max = df_diario['Order Date'].min(), df_diario['Order Date'].max()
        rango_completo = pd.date_range(start=fecha_min, end=fecha_max, freq='D')
        df_diario = df_diario.set_index('Order Date').reindex(rango_completo, fill_value=0.0)
        df_diario.index.name = 'Order Date'
        df_diario = df_diario.reset_index()
        return df_diario
    except Exception as e:
        st.error(f'Error de conexión con SQLite: {e}')
        return None

df_historico = cargar_secuencia_historica()

if df_historico is not None:
    # Configurar el Escalador usando el histórico total
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_historico[['Sales']])

    # Crear pestañas en Streamlit para organizar la aplicación
    tab1, tab2, tab3 = st.tabs(["🔮 Inferencia en Tiempo Real", "📈 Evaluación Cuantitativa del Modelo", "🖥️ Guía de la Interfaz Gráfica"])

    with tab1:
        st.subheader('🔮 Simulación y Predicción para Mañana')
        st.write('Últimos días registrados en la base de datos:')
        st.dataframe(df_historico.tail(5))

        # Obtener los últimos 30 días reales
        ultimos_30_dias = df_historico.tail(30).copy()
        secuencia_real = ultimos_30_dias['Sales'].values.copy()

        venta_hoy_simulada = st.slider('Ventas para hoy (USD):', 
                                       min_value=0.0, 
                                       max_value=float(df_historico['Sales'].max() * 1.5), 
                                       value=float(secuencia_real[-1]))

        # Reemplazar la última venta de la secuencia por la seleccionada en el slider
        secuencia_real[-1] = venta_hoy_simulada

        try:
            model = tf.keras.models.load_model('sales_model.h5')
            secuencia_escalada = scaler.transform(secuencia_real.reshape(-1, 1))
            input_tensor = secuencia_escalada.reshape((1, 30, 1))

            # Predicción
            prediccion_escalada = model.predict(input_tensor)
            prediccion_usd = scaler.inverse_transform(prediccion_escalada)[0][0]

            # Mostrar Resultado principal en una tarjeta
            st.metric(label='💵 Pronóstico de Ventas para Mañana (USD)', value=f'${prediccion_usd:,.2f} USD')

            # Gráfico de la secuencia reciente
            fig, ax = plt.subplots(figsize=(10, 4))
            fechas_grafico = ultimos_30_dias['Order Date'].dt.strftime('%Y-%m-%d').values
            ax.plot(fechas_grafico, secuencia_real, marker='o', color='teal', label='Histórico 30 días')
            ax.axhline(y=venta_hoy_simulada, color='orange', linestyle='--', label='Venta Simulada Hoy')
            ax.scatter(['Mañana (Pronosticado)'], [prediccion_usd], color='crimson', s=150, zorder=5, label='Pronóstico LSTM')
            ax.set_title('Serie Temporal de Entrada vs. Pronóstico del Modelo LSTM')
            ax.set_ylabel('Ventas (USD)')
            ax.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f'No se pudo ejecutar la predicción: {e}')

    with tab2:
        st.subheader('📊 Desempeño Cuantitativo del Pipeline de Datos')

        try:
            # Preparar de nuevo el conjunto de prueba idéntico para evaluar
            def crear_ventanas_deslizantes(data, size_ventana=30):
                X_list, y_list = [], []
                for i in range(len(data) - size_ventana):
                    X_list.append(data[i:(i + size_ventana)])
                    y_list.append(data[i + size_ventana])
                return np.array(X_list), np.array(y_list)

            sales_scaled = scaler.transform(df_historico[['Sales']]).flatten()
            X_all, y_all = crear_ventanas_deslizantes(sales_scaled, 30)
            X_all = X_all.reshape((X_all.shape[0], X_all.shape[1], 1))

            split_idx = int(len(X_all) * 0.8)
            X_test, y_test = X_all[split_idx:], y_all[split_idx:]

            model_eval = tf.keras.models.load_model('sales_model.h5')
            predicciones_escaladas = model_eval.predict(X_test)

            y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            predicciones_originales = scaler.inverse_transform(predicciones_escaladas).flatten()

            # ---- Métricas de Regresión (Negocio Real) ----
            rmse = np.sqrt(mean_squared_error(y_test_original, predicciones_originales))
            mae = mean_absolute_error(y_test_original, predicciones_originales)
            r2 = r2_score(y_test_original, predicciones_originales)

            st.markdown("### 📉 Métricas de Negocio (Regresión)")
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE (Desviación Típica del Error)", f"${rmse:,.2f} USD")
            col2.metric("MAE (Error Absoluto Medio)", f"${mae:,.2f} USD")
            col3.metric("R2 Score (Calidad de Ajuste)", f"{r2:.4f}")

            # ---- Adaptación a Métricas de Clasificación (Dirección del Mercado) ----
            st.markdown("### 🎯 Métricas de Rendimiento sobre la Tendencia Diaria")
            st.write("Para evaluar la capacidad predictiva del modelo en términos de clasificación, se analiza la dirección del cambio diario de ventas (sube/baja respecto al día anterior).")

            # Definir la tendencia real y la predicha (1 si sube, 0 si se mantiene/baja)
            tendencia_real = np.diff(y_test_original) > 0
            tendencia_predicha = np.diff(predicciones_originales) > 0

            accuracy = accuracy_score(tendencia_real, tendencia_predicha)
            f1 = f1_score(tendencia_real, tendencia_predicha)

            col_c1, col_c2 = st.columns(2)
            col_c1.metric("Accuracy (Exactitud de Tendencia)", f"{accuracy*100:.2f}%")
            col_c2.metric("F1-Score (Tendencia)", f"{f1:.4f}")

            # Gráficos de Evaluación Integrados
            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.write("**Matriz de Confusión (Dirección de Ventas):**")
                cm = confusion_matrix(tendencia_real, tendencia_predicha)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                            xticklabels=['Baja/Igual', 'Sube'], 
                            yticklabels=['Baja/Igual', 'Sube'], ax=ax_cm)
                ax_cm.set_ylabel('Realidad')
                ax_cm.set_xlabel('Predicción LSTM')
                st.pyplot(fig_cm)

            with col_g2:
                st.write("**Comparativa de Serie Temporal:**")
                fig_eval, ax_eval = plt.subplots(figsize=(6, 5.2))
                ax_eval.plot(y_test_original, label='Ventas Reales', color='teal', alpha=0.8)
                ax_eval.plot(predicciones_originales, label='Pronósticos LSTM', color='crimson', linestyle='--')
                ax_eval.set_title('Serie Real vs Pronosticada')
                ax_eval.set_xlabel('Días')
                ax_eval.set_ylabel('Ventas (USD)')
                ax_eval.legend()
                ax_eval.grid(True)
                st.pyplot(fig_eval)

        except Exception as e:
            st.error(f'Error al generar la sección de evaluación: {e}')

    with tab3:
        st.subheader("🖥️ Guía de Uso y Descripción de la Interfaz")
        st.markdown("""
### **Estructura General del Dashboard**
Este panel de control interactivo está diseñado para simplificar el análisis predictivo de ventas diarias mediante tres componentes esenciales:

1. **Inferencia en Tiempo Real:**
   - **Vista de Datos:** Te muestra los últimos registros procesados en la base de datos de SQLite.
   - **Deslizador de Ventas (Slider):** Te permite simular de forma interactiva la venta del día de hoy. El modelo utilizará automáticamente esta entrada simulada junto con los 29 días previos para proyectar de forma inmediata el pronóstico de mañana.
   - **Visualización:** Genera un gráfico dinámico donde se observa la tendencia de los últimos 30 días y se resalta con un punto rojo el pronóstico para el día siguiente.

2. **Evaluación de Rendimiento:**
   - **Métricas de Regresión (RMSE, MAE, R2):** Evalúa el error promedio diario directamente en dólares, lo que permite estimar desviaciones de caja reales de forma financiera.
   - **Métricas de Clasificación Adaptadas:** Ofrece indicadores clásicos de proyectos de analítica como el *F1-Score* e introduce la **Matriz de Confusión** que cuantifica cuántas veces el modelo acertó al predecir si el mercado se movería al alza o a la baja.
""")
