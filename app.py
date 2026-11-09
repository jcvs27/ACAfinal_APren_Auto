import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title='Dashboard LSTM Real', layout='wide')

st.title('📊 Predicción de Ventas en Producción con LSTM')
st.write('Esta aplicación se conecta a la base de datos de SQLite, extrae la secuencia histórica real de 30 días y predice de forma automática las ventas del día siguiente.')

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
    # 2. Configurar el Escalador usando el histórico total
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(df_historico[['Sales']])

    # Mostrar una sección informativa de los últimos datos de la base de datos
    st.subheader('📈 Histórico Reciente de Ventas Reales')
    st.dataframe(df_historico.tail(10))

    # 3. Interfaz de Predicción en Tiempo Real
    st.subheader('🔮 Inferencia en Tiempo Real (LSTM)')

    # Obtenemos los últimos 30 días reales
    ultimos_30_dias = df_historico.tail(30).copy()
    secuencia_real = ultimos_30_dias['Sales'].values

    # Permitir al usuario simular un valor para el día de hoy si lo desea
    st.info('A continuación puedes simular de forma interactiva la venta de hoy para ver el impacto predictivo de mañana:')
    venta_hoy_simulada = st.slider('Ventas estimadas para hoy (USD):',
                                   min_value=0.0,
                                   max_value=float(df_historico['Sales'].max() * 1.5),
                                   value=float(secuencia_real[-1]))

    # Reemplazar la última venta de la secuencia por la seleccionada en el slider para mayor interactividad
    secuencia_real[-1] = venta_hoy_simulada

    # 4. Carga del Modelo LSTM y Ejecución del Pronóstico
    try:
        model = tf.keras.models.load_model('sales_model.h5')

        # Escalar la secuencia
        secuencia_escalada = scaler.transform(secuencia_real.reshape(-1, 1))

        # Dar formato tridimensional (1 batch, 30 pasos de tiempo, 1 feature)
        input_tensor = secuencia_escalada.reshape((1, 30, 1))

        # Realizar predicción
        prediccion_escalada = model.predict(input_tensor)
        prediccion_usd = scaler.inverse_transform(prediccion_escalada)[0][0]

        # Mostrar Resultado principal en una tarjeta
        st.metric(label='💵 Pronóstico de Ventas para Mañana (USD)', value=f'${prediccion_usd:,.2f} USD')

        # 5. Visualizar Gráficamente la Secuencia Reciente y el Pronóstico Futuro
        fig, ax = plt.subplots(figsize=(10, 4))
        fechas_grafico = ultimos_30_dias['Order Date'].dt.strftime('%Y-%m-%d').values

        ax.plot(fechas_grafico, secuencia_real, marker='o', color='teal', label='Histórico 30 días')
        ax.axhline(y=venta_hoy_simulada, color='orange', linestyle='--', label='Venta Simulada Hoy')
        # Agregar punto de predicción futuro
        ax.scatter(['Mañana (Pronosticado)'], [prediccion_usd], color='crimson', s=150, zorder=5, label='Pronóstico LSTM')

        ax.set_title('Serie Temporal de Entrada vs. Pronóstico del Modelo LSTM')
        ax.set_ylabel('Ventas (USD)')
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f'Falta el modelo para realizar predicciones activas: {e}')
