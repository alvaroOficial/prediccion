import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import datetime

# Cargar los datos
def load_data(uploaded_file):
    df = pd.read_excel(uploaded_file)  # Carga el archivo desde el input del usuario
    df['MES'] = pd.to_datetime(df['MES'])
    df.set_index('MES', inplace=True)
    return df

# Función para construir y entrenar el modelo ARIMA
def train_arima_model(df):
    model = ARIMA(df, order=(1, 1, 1))  # Ajusta (p, d, q) según sea necesario
    model_fit = model.fit()
    return model_fit

# Función para realizar la predicción
def predict_from_date(model_fit, df, fecha_deseada):
    # Calcular la cantidad de meses entre el último dato y la fecha deseada
    ultimo_mes = df.index[-1]
    diferencia_meses = (fecha_deseada.year - ultimo_mes.year) * 12 + (fecha_deseada.month - ultimo_mes.month)

    if diferencia_meses <= 0:
        return None, "La fecha ingresada ya está en los datos."
    
    # Realizar la predicción
    forecast = model_fit.forecast(steps=diferencia_meses)
    prediccion = forecast[-1]
    
    return prediccion, None

# Función principal de la aplicación Streamlit
def main():
    st.title("Predicción de Exportaciones de Café")
    
    # Subir el archivo Excel
    uploaded_file = st.file_uploader("Sube tu archivo Excel", type=["xlsx"])
    if uploaded_file is not None:
        # Cargar los datos
        df = load_data(uploaded_file)
        
        # Mostrar los primeros datos
        st.subheader("Datos Históricos")
        st.write(df.head())
        
        # Entrenar el modelo ARIMA
        model_fit = train_arima_model(df)
        
        # Entrada del usuario para la fecha a predecir
        fecha_deseada = st.date_input("Ingrese la fecha a predecir (YYYY-MM-DD)", min_value=df.index[-1].date())
        
        # Realizar la predicción
        if st.button("Predecir"):
            prediccion, error = predict_from_date(model_fit, df, fecha_deseada)
            
            if error:
                st.error(error)
            else:
                st.success(f"La predicción para {fecha_deseada.strftime('%Y-%m-%d')} es: {prediccion:.2f}")
            
            # Graficar los datos reales y las predicciones extendidas
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Total Exportaciones'], label="Datos Reales")
            futuro_fechas = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=(fecha_deseada.year - df.index[-1].year)*12 + (fecha_deseada.month - df.index[-1].month), freq='M')
            plt.plot(futuro_fechas, [prediccion] * len(futuro_fechas), label="Predicción", linestyle="--", color="orange")
            plt.axvline(x=df.index[-1], color='red', linestyle='--', label='Inicio de Predicción')
            plt.legend()
            plt.title("Predicción de Exportaciones de Café")
            plt.xlabel("Fecha")
            plt.ylabel("Exportaciones Totales")
            plt.grid(True)
            
            st.pyplot(plt)
    else:
        st.warning("Por favor, sube un archivo Excel para continuar.")

if __name__ == "__main__":
    main()
