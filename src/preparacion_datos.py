import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib


# Cargar el dataset
df = pd.read_excel('data/updated_data_jugadores.xlsx')

# Visualización inicial (opcional)
print("Datos originales:")
print(df.head())

# Lista de todas las posiciones posibles
todas_las_posiciones = ['Portero', 'Defensa', 'Mediocampista', 'Delantero']

# Codificar la columna de posición
label_encoder_posicion = LabelEncoder()
label_encoder_posicion.fit(todas_las_posiciones)

# Guarda el codificador
joblib.dump(label_encoder_posicion, 'data/label_encoder_posicion.pkl')

# Separar características (X) y target (y)
# Supongamos que queremos predecir 'IMC'
X = df.drop(columns=['Posición'])  # Asegúrate de ajustar el nombre según tu columna objetivo
y = df['Posición']  # Esto es solo un ejemplo; ajusta según tu objetivo

# Convertir variables categóricas en variables dummy (si las hay)
X = pd.get_dummies(X, drop_first=True)

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Verificar los resultados
print("Datos estandarizados:")
print(X_scaled[:5])  
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
print(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")# Muestra las primeras 5 filas
print(label_encoder_posicion.classes_)