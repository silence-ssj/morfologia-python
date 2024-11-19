import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Cargar el dataset
df = pd.read_excel('data/updated_data_jugadores.xlsx')

# Separar características (X) y target (y)
X = df.drop(columns=['Posición'])  # Ajusta según tu columna objetivo
y = df['Posición']

# Convertir variables categóricas en variables dummy
X = pd.get_dummies(X, drop_first=True)

# Estandarizar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir el dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Instancia del modelo
model = RandomForestClassifier()

# Entrenamiento
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

# Evaluación del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
matriz_confusion = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Valor Real')
plt.xlabel('Predicción')
plt.show()

# Importancia de características
importancias = model.feature_importances_
features = X.columns
plt.figure(figsize=(10,6))
sns.barplot(x=importancias, y=features)
plt.title('Importancia de Características')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.show()

# Guardar el modelo y scaler
joblib.dump(model, 'data/modelo_entrenado.pkl')
joblib.dump(scaler, 'data/scaler.pkl')
