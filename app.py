
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
a
# Configuración del título de la pestaña y ancho de la interfaz
st.set_page_config(page_title='Pronóstico de Ventas con LSTM', layout='wide')

st.title('📊 Aplicación de Pronóstico de Ventas (LSTM)')
st.write('Esta aplicación interactiva utiliza un modelo de red neuronal LSTM para predecir las ventas diarias.')

# Cargar datos desde SQLite para visualización en la UI
# try/except: si la base de datos no está subida a la nube, la app no se rompe;
# simplemente muestra una advertencia informativa.
try:
    conn = sqlite3.connect('sales_data.db')
    df_sql = pd.read_sql_query("SELECT * FROM sales_raw LIMIT 5", conn)
    conn.close()
    st.subheader('📋 Vista previa de registros históricos en SQLite')
    st.dataframe(df_sql)
except Exception as e:
    st.warning(f'Nota de base de datos local: {e}')

# Entrada Interactiva del simulador para el usuario
# st.slider(): control deslizante que permite elegir un valor de venta entre
# 0 y 25000 USD, con valor inicial en 2500 USD.
st.subheader('🔮 Simulación de Entrada para Predicción')
st.write('Ingrese una venta de referencia para evaluar de forma interactiva la lógica de inferencia:')
ventas_usuario = st.slider('Ventas promedio estimadas del último periodo (USD):', min_value=0.0, max_value=25000.0, value=2500.0)

# Intentar cargar el modelo entrenado; si falta el archivo, avisar al usuario.
try:
    model = tf.keras.models.load_model('sales_model.h5')
    st.success('¡Modelo LSTM principal (sales_model.h5) detectado y cargado correctamente en Streamlit!')
except Exception as e:
    st.info('Nota: Asegúrate de subir el archivo "sales_model.h5" y "sales_data.db" junto con este script a tu repositorio de GitHub para habilitar la predicción real.')
