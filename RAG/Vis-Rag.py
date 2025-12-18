import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

print("Cargando datos...")
df = pd.read_csv('dataset_limpio.csv')

# Crear figura con múltiples subplots
fig = plt.figure(figsize=(16, 12))


#------------- 1. Distribución por Tema-----------
ax1 = plt.subplot(3, 3, 1)
tema_counts = df['tema'].value_counts()
colors_tema = sns.color_palette("Set2", len(tema_counts))
ax1.barh(range(len(tema_counts)), tema_counts.values, color=colors_tema)
ax1.set_yticks(range(len(tema_counts)))
ax1.set_yticklabels([t[:30] + '...' if len(t) > 30 else t for t in tema_counts.index], fontsize=8)
ax1.set_xlabel('Cantidad')
ax1.set_title('Distribución por Tema', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, v in enumerate(tema_counts.values):
    ax1.text(v + 5, i, str(v), va='center', fontsize=9)


#-------- 2. Distribución por Sentimiento ----------

ax2 = plt.subplot(3, 3, 2)
sent_counts = df['sentimiento'].value_counts()
colors_sent = ['#ff6b6b', '#95e1d3', '#feca57']
explode = (0.05, 0.05, 0.05)
ax2.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.1f%%',
        startangle=90, colors=colors_sent, explode=explode)
ax2.set_title('Distribución por Sentimiento', fontweight='bold')


#-------- 3. Distribución de Longitud de Textos ----------

ax3 = plt.subplot(3, 3, 3)
df['longitud'] = df['texto'].str.len()
ax3.hist(df['longitud'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
ax3.axvline(df['longitud'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["longitud"].mean():.0f}')
ax3.axvline(df['longitud'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df["longitud"].median():.0f}')
ax3.set_xlabel('Longitud del texto (caracteres)')
ax3.set_ylabel('Frecuencia')
ax3.set_title('Distribución de Longitud de Textos', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)


#---------- 4. Distribución de Palabras por Texto ----------

ax4 = plt.subplot(3, 3, 4)
df['num_palabras'] = df['texto'].str.split().str.len()
ax4.hist(df['num_palabras'], bins=25, color='coral', edgecolor='black', alpha=0.7)
ax4.axvline(df['num_palabras'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["num_palabras"].mean():.1f}')
ax4.axvline(df['num_palabras'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df["num_palabras"].median():.1f}')
ax4.set_xlabel('Número de palabras')
ax4.set_ylabel('Frecuencia')
ax4.set_title('Distribución de Número de Palabras', fontweight='bold')
ax4.legend()
ax4.grid(alpha=0.3)


#------------------ 5. Sentimiento por Tema (Heatmap)------------
ax5 = plt.subplot(3, 3, 5)
cross_tab = pd.crosstab(df['tema'], df['sentimiento'], normalize='index') * 100
# Acortar nombres de temas
cross_tab.index = [t[:20] + '...' if len(t) > 20 else t for t in cross_tab.index]
sns.heatmap(cross_tab, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Porcentaje (%)'})
ax5.set_title('Sentimiento por Tema (%)', fontweight='bold')
ax5.set_xlabel('Sentimiento')
ax5.set_ylabel('Tema')
plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax5.get_yticklabels(), rotation=0)


#-------------- 6. Distribución temporal -----------

ax6 = plt.subplot(3, 3, 6)
df['fecha'] = pd.to_datetime(df['fecha'])
df['año_mes'] = df['fecha'].dt.to_period('M')
temporal = df.groupby('año_mes').size()
ax6.plot(temporal.index.astype(str), temporal.values, marker='o', linewidth=2, markersize=4, color='purple')
ax6.set_xlabel('Período')
ax6.set_ylabel('Cantidad de registros')
ax6.set_title('Distribución Temporal', fontweight='bold')
ax6.grid(alpha=0.3)
plt.setp(ax6.get_xticklabels(), rotation=45, ha='right', fontsize=7)


#---------------- 7. Engagement: Likes --------------------
ax7 = plt.subplot(3, 3, 7)
ax7.boxplot([df[df['sentimiento']=='positivo']['likes'],
             df[df['sentimiento']=='neutral']['likes'],
             df[df['sentimiento']=='negativo']['likes']],
            labels=['Positivo', 'Neutral', 'Negativo'],
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', alpha=0.7))
ax7.set_ylabel('Likes')
ax7.set_title('Distribución de Likes por Sentimiento', fontweight='bold')
ax7.grid(axis='y', alpha=0.3)

#---------------- 8. Engagement: Reposts--------------------
ax8 = plt.subplot(3, 3, 8)
ax8.boxplot([df[df['sentimiento']=='positivo']['reposts'],
             df[df['sentimiento']=='neutral']['reposts'],
             df[df['sentimiento']=='negativo']['reposts']],
            labels=['Positivo', 'Neutral', 'Negativo'],
            patch_artist=True,
            boxprops=dict(facecolor='lightcoral', alpha=0.7))
ax8.set_ylabel('Reposts')
ax8.set_title('Distribución de Reposts por Sentimiento', fontweight='bold')
ax8.grid(axis='y', alpha=0.3)


#-------------- 9. Correlación Likes vs Reposts -----------

ax9 = plt.subplot(3, 3, 9)
scatter = ax9.scatter(df['likes'], df['reposts'], 
                      c=df['sentimiento'].map({'positivo': 0, 'neutral': 1, 'negativo': 2}),
                      cmap='viridis', alpha=0.6, s=30)
ax9.set_xlabel('Likes')
ax9.set_ylabel('Reposts')
ax9.set_title('Correlación Likes vs Reposts', fontweight='bold')
ax9.grid(alpha=0.3)

# Agregar línea de tendencia
z = np.polyfit(df['likes'], df['reposts'], 1)
p = np.poly1d(z)
ax9.plot(df['likes'], p(df['likes']), "r--", alpha=0.8, linewidth=2)

plt.tight_layout()
plt.savefig('visualizaciones_dataset.png', dpi=300, bbox_inches='tight')
print(" Visualización principal guardada: visualizaciones_dataset.png")


#--------------- Análisis de palabras frecuentes ----------------

print("\nGenerando análisis de palabras frecuentes...")
fig2, axes = plt.subplots(1, 3, figsize=(18, 6))

sentimientos = ['positivo', 'neutral', 'negativo']
colors_bars = ['#2ecc71', '#3498db', '#e74c3c']

# Palabras a excluir (stopwords básicas)
stopwords = {'que', 'de', 'la', 'el', 'en', 'y', 'a', 'los', 'las', 'un', 'una', 'por', 
             'con', 'para', 'del', 'se', 'es', 'su', 'al', 'lo', 'como', 'más', 'o',
             'pero', 'sus', 'le', 'ya', 'todo', 'esta', 'está', 'si', 'no', 'son'}

for idx, (sent, color) in enumerate(zip(sentimientos, colors_bars)):
    text = ' '.join(df[df['sentimiento']==sent]['texto'].str.lower())
    words = [word for word in text.split() if word not in stopwords and len(word) > 3]
    word_freq = Counter(words).most_common(15)
    
    words_list = [w[0] for w in word_freq]
    counts_list = [w[1] for w in word_freq]
    
    axes[idx].barh(range(len(words_list)), counts_list, color=color, alpha=0.7)
    axes[idx].set_yticks(range(len(words_list)))
    axes[idx].set_yticklabels(words_list)
    axes[idx].set_xlabel('Frecuencia')
    axes[idx].set_title(f'Palabras Frecuentes - {sent.capitalize()}', 
                        fontweight='bold', fontsize=12)
    axes[idx].invert_yaxis()
    axes[idx].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('palabras_frecuentes.png', dpi=300, bbox_inches='tight')
print(" Análisis de palabras frecuentes guardado: palabras_frecuentes.png")

# --------------- Análisis de similitud ----------------
print("\nGenerando análisis de similitud...")

# Cargar pares similares
df_sim = pd.read_csv('pares_similares.csv')

fig3, axes = plt.subplots(1, 2, figsize=(14, 6))

# Histograma de similitudes
ax1 = axes[0]
ax1.hist(df_sim['similitud'], bins=30, color='teal', edgecolor='black', alpha=0.7)
ax1.axvline(df_sim['similitud'].mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Media: {df_sim["similitud"].mean():.3f}')
ax1.set_xlabel('Similitud Coseno')
ax1.set_ylabel('Frecuencia')
ax1.set_title('Distribución de Similitudes (> 0.8)', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Top 10 pares más similares
ax2 = axes[1]
top_10 = df_sim.nlargest(10, 'similitud')
ax2.barh(range(10), top_10['similitud'].values, color='coral')
ax2.set_yticks(range(10))
ax2.set_yticklabels([f"Par {i+1}" for i in range(10)])
ax2.set_xlabel('Similitud Coseno')
ax2.set_title('Top 10 Pares Más Similares', fontweight='bold')
ax2.set_xlim(0.8, 1.0)
ax2.grid(axis='x', alpha=0.3)

for i, v in enumerate(top_10['similitud'].values):
    ax2.text(v - 0.02, i, f'{v:.3f}', va='center', ha='right', fontsize=9, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('analisis_similitud.png', dpi=300, bbox_inches='tight')
print(" Análisis de similitud guardado: analisis_similitud.png")

# --------------- Resumen estadístico ----------------
print("\n" + "="*80)
print("RESUMEN ESTADÍSTICO FINAL")
print("="*80)

print(f"\n DATASET LIMPIO:")
print(f"   • Total de registros: {len(df)}")
print(f"   • Registros únicos de texto: {df['texto'].nunique()}")
print(f"   • Rango de fechas: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
print(f"   • Usuarios únicos: {df['usuario'].nunique()}")

print(f"\n CONTENIDO:")
print(f"   • Longitud promedio: {df['longitud'].mean():.1f} caracteres")
print(f"   • Palabras promedio: {df['num_palabras'].mean():.1f} palabras")
print(f"   • Tema más frecuente: {df['tema'].value_counts().index[0]}")
print(f"   • Sentimiento más frecuente: {df['sentimiento'].value_counts().index[0]}")

print(f"\n ENGAGEMENT:")
print(f"   • Likes promedio: {df['likes'].mean():.0f}")
print(f"   • Reposts promedio: {df['reposts'].mean():.0f}")
print(f"   • Correlación Likes-Reposts: {df['likes'].corr(df['reposts']):.3f}")

print(f"\n SIMILITUD:")
print(f"   • Pares muy similares (>0.8): {len(df_sim)}")
print(f"   • Similitud máxima: {df_sim['similitud'].max():.4f}")
print(f"   • Similitud promedio: {df_sim['similitud'].mean():.4f}")

print("\n Todas las visualizaciones han sido generadas exitosamente!")