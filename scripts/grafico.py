import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Configuración de estilo
sns.set(style="whitegrid")
plt.figure(figsize=(12, 7))

# Datos de tu tabla
data = {
    'Modelo': ['CNN + Bi-LSTM', 'Random Forest', 'SVM (RBF)', 'Decision Tree'],
    'Accuracy': [80.6, 89.8, 84.3, 81.0],
    'Precisión': [74.8, 88.6, 82.2, 85.2],
    'Recall': [96.8, 93.2, 90.6, 78.6],
    'F1-Score': [84.4, 90.8, 86.2, 81.8]
}

df = pd.DataFrame(data)

# Reestructurar datos para seaborn (formato largo)
df_plot = df.melt(id_vars='Modelo', var_name='Métrica', value_name='Porcentaje')

# Crear gráfico de barras
ax = sns.barplot(data=df_plot, x='Métrica', y='Porcentaje', hue='Modelo', palette='viridis')

# Añadir etiquetas de valor sobre las barras
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%', 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 9), 
                textcoords = 'offset points',
                fontsize=9, fontweight='bold')

plt.title('Comparativa de Rendimiento: CNN + Bi-LSTM vs Modelos Tradicionales', fontsize=15, pad=20)
plt.ylim(0, 110) # Espacio para las etiquetas
plt.ylabel('Porcentaje (%)')
plt.legend(title='Modelos', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Guardar la imagen
plt.savefig('comparativa_final_modelos.png', dpi=300)
print("Gráfica 'comparativa_final_modelos.png' generada con éxito.")
plt.show()