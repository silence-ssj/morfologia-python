import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el dataset
df = pd.read_excel('data/updated_data_jugadores.xlsx')

# Visualización inicial
print(df.head())
print(df.info())
print(df.describe())

# Gráfico de boxplot para altura, peso, IMC y masa muscular
plt.figure(figsize=(12,6))
sns.boxplot(data=df[['Altura', 'Peso', 'IMC', 'Masa Muscular']])
plt.title('Distribución de Características Morfológicas')
plt.ylabel('Valores')
plt.xticks(rotation=0)
plt.show()

# Gráfico de dispersión entre altura y peso, coloreado por posición
plt.figure(figsize=(10,6))
sns.scatterplot(x='Altura', y='Peso', hue='Posición', data=df, palette='Set1')
plt.title('Relación entre Altura y Peso por Posición')
plt.xlabel('Altura (cm)')
plt.ylabel('Peso (kg)')
plt.show()
