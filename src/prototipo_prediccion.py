import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Cargar el modelo, scaler y label encoder
model = joblib.load('data/modelo_entrenado.pkl')
scaler = joblib.load('data/scaler.pkl')
label_encoder_posicion = joblib.load('data/label_encoder_posicion.pkl')

# Cargar el DataFrame original
df_original = pd.read_excel('data/updated_data_jugadores.xlsx')

# Convertir las columnas categóricas a dummies para obtener la estructura
X_original = pd.get_dummies(df_original.drop(columns=['Posición']), drop_first=True)

# 3. Definir la función para predecir
def predict_new_player(data):
    if len(data) != 5:
        raise ValueError('Se requieren exactamente 5 características: Altura, Peso, Masa Muscular, IMC, Posición')
    
    # Codificar la posición
    try:
        posicion_encoded = label_encoder_posicion.transform([data[4]])[0]
    except ValueError:
        raise ValueError(f'Posición desconocida: {data[4]}')

    # Preparar el nuevo vector de características
    nuevo_jugador = data[:4] + [posicion_encoded]
    
    # Crear un DataFrame para el nuevo jugador con las mismas columnas que X_original
    nuevo_jugador_df = pd.DataFrame([nuevo_jugador], columns=['Altura', 'Peso', 'Masa Muscular', 'IMC', 'Posición'])

    # Convertir las variables categóricas a dummies para que coincidan con el modelo
    nuevo_jugador_dummies = pd.get_dummies(nuevo_jugador_df, drop_first=True)

    # Asegurar que las columnas coincidan con el dataset original (rellenar las columnas faltantes con 0)
    nuevo_jugador_dummies = nuevo_jugador_dummies.reindex(columns=X_original.columns, fill_value=0)

    # Escalar los datos
    data_scaled = scaler.transform(nuevo_jugador_dummies)
    
    # Realizar la predicción
    return model.predict(data_scaled)

# Datos del nuevo jugador
nuevo_jugador = [180, 80, 20.5, 22.5, 'Defensa']
resultado = predict_new_player(nuevo_jugador)
print(f'Resultado:  {resultado[0]}') 

# 5. Generar reporte
def generate_report(nuevo_jugador, resultado):
    # Crear carpeta 'reports' si no existe
    os.makedirs('reports', exist_ok=True)

    # Crear un DataFrame con el resultado
    report_df = pd.DataFrame([nuevo_jugador + [resultado]], columns=['Altura', 'Peso', 'Masa Muscular', 'IMC', 'Posición', 'Predicción'])

    # Obtener la marca de tiempo actual
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Generar un nombre único para el archivo usando la marca de tiempo
    report_file_path = os.path.join('reports', f'reporte_predicciones_{timestamp}.csv')

    # Guardar el reporte
    report_df.to_csv(report_file_path, index=False)
    print(f"Reporte generado: {report_file_path}")

# Generar el reporte
generate_report(nuevo_jugador, resultado[0]) # Reemplaza con los datos reales



# Calcular los promedios para jugadores de la misma posición
posicion = nuevo_jugador[-1]  # 'Defensa'
promedios_posicion = df_original[df_original['Posición'] == posicion][['Altura', 'Peso', 'Masa Muscular', 'IMC']].mean()

# Reemplazar valores NaN con 0 o algún valor por defecto (por ejemplo, el promedio general de la columna)
promedios_posicion = promedios_posicion.fillna(df_original[['Altura', 'Peso', 'Masa Muscular', 'IMC']].mean())
 # O usa otro valor si es más adecuado

# Crear un DataFrame con los datos del nuevo jugador y los promedios
data_comparacion = pd.DataFrame({
    'Características': ['Altura', 'Peso', 'Masa Muscular', 'IMC'],
    'Nuevo Jugador': nuevo_jugador[:4],
    'Promedio de su Posición': promedios_posicion
})

# Crear el gráfico de barras
ax = data_comparacion.set_index('Características').plot(kind='bar', figsize=(10,6), color=['skyblue', 'lightgreen'])

# Ajustar el límite superior del eje Y de acuerdo a la altura máxima (usando 210 cm como máximo)
altura_max = max(nuevo_jugador[0], 210)  # La altura del nuevo jugador o 210 si es mayor
ax.set_ylim(0, altura_max)  # Ajusta el límite superior

plt.title('Comparación de las características del nuevo jugador vs promedio de su posición')
plt.ylabel('Valor')
plt.xticks(rotation=0)
plt.show()
