import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("ANÁLISIS Y PREPARACIÓN DE DATASET PARA SISTEMA RAG")
print("="*80)


# ------------ Carga y análisis exploratorio inicial -----------------------------

print("\n[PASO 1] Cargando dataset...")
df = pd.read_csv('data/dataset_sintetico_5000_ampliado.csv')

print(f"\n Dimensiones del dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"\n Columnas disponibles:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i}. {col}")

print("\n Información general del dataset:")
print(df.info())

print("\n Primeras filas del dataset:")
print(df.head(10))

print("\n Estadísticas descriptivas:")
print(df.describe())

# ------------ Análisis de calidad de datos -----------------------------
print("\n" + "="*80)
print("[PASO 2] ANÁLISIS DE CALIDAD DE DATOS")
print("="*80)

print("\n Valores nulos por columna:")
null_counts = df.isnull().sum()
for col, count in null_counts.items():
    if count > 0:
        print(f"   !  {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print(f"   ✓ {col}: 0")

print("\n  Duplicados completos:")
duplicados = df.duplicated().sum()
print(f"   Total de filas duplicadas completas: {duplicados}")

print("\n  Duplicados en campo 'texto':")
duplicados_texto = df['texto'].duplicated().sum()
print(f"   Total de textos duplicados: {duplicados_texto}")
if duplicados_texto > 0:
    print(f"   Porcentaje: {duplicados_texto/len(df)*100:.2f}%")


# ------------ Análisis de distribuciones -----------------------------

print("\n" + "="*80)
print("[PASO 3] ANÁLISIS DE DISTRIBUCIONES")
print("="*80)

print("\n  Distribución por TEMA:")
tema_dist = df['tema'].value_counts()
for tema, count in tema_dist.items():
    print(f"   • {tema}: {count} ({count/len(df)*100:.2f}%)")

print("\n  Distribución por SENTIMIENTO:")
sent_dist = df['sentimiento'].value_counts()
for sent, count in sent_dist.items():
    print(f"   • {sent}: {count} ({count/len(df)*100:.2f}%)")

print("\n  Distribución de longitud de textos:")
df['longitud_texto'] = df['texto'].str.len()
print(f"   Mínimo: {df['longitud_texto'].min()} caracteres")
print(f"   Promedio: {df['longitud_texto'].mean():.2f} caracteres")
print(f"   Mediana: {df['longitud_texto'].median():.2f} caracteres")
print(f"   Máximo: {df['longitud_texto'].max()} caracteres")

# Análisis de palabras
df['num_palabras'] = df['texto'].str.split().str.len()
print(f"\n  Distribución de número de palabras:")
print(f"   Mínimo: {df['num_palabras'].min()} palabras")
print(f"   Promedio: {df['num_palabras'].mean():.2f} palabras")
print(f"   Mediana: {df['num_palabras'].median():.2f} palabras")
print(f"   Máximo: {df['num_palabras'].max()} palabras")

# ------------ Limpieza de datos ------------
print("\n" + "="*80)
print("[PASO 4] LIMPIEZA DE DATOS")
print("="*80)

# Crear copia para limpieza
df_clean = df.copy()

print("\n  Aplicando limpieza...")

# 4.1 Eliminar duplicados exactos
duplicados_antes = len(df_clean)
df_clean = df_clean.drop_duplicates(subset=['texto'], keep='first')
duplicados_eliminados = duplicados_antes - len(df_clean)
print(f"   ✓ Duplicados exactos eliminados: {duplicados_eliminados}")

# 4.2 Limpiar espacios en blanco
df_clean['texto'] = df_clean['texto'].str.strip()
df_clean['texto'] = df_clean['texto'].str.replace(r'\s+', ' ', regex=True)

# 4.3 Eliminar textos muy cortos (menos de 20 caracteres)
textos_cortos = (df_clean['texto'].str.len() < 20).sum()
df_clean = df_clean[df_clean['texto'].str.len() >= 20]
print(f"   ✓ Textos demasiado cortos eliminados: {textos_cortos}")

# 4.4 Normalizar fechas
df_clean['fecha'] = pd.to_datetime(df_clean['fecha'], errors='coerce')

# 4.5 Validar valores numéricos
df_clean['likes'] = pd.to_numeric(df_clean['likes'], errors='coerce').fillna(0).astype(int)
df_clean['reposts'] = pd.to_numeric(df_clean['reposts'], errors='coerce').fillna(0).astype(int)

print(f"\n  Dataset limpio: {len(df_clean)} registros (eliminados: {len(df) - len(df_clean)})")

# ------------ Cálculo de similitud coseno ------------
print("\n" + "="*80)
print("[PASO 5] CÁLCULO DE SIMILITUD COSENO")
print("="*80)

print("\n  Calculando embeddings TF-IDF...")

# Crear vectorizador TF-IDF
vectorizer = TfidfVectorizer(
    max_features=1000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    strip_accents='unicode'
)

# Calcular embeddings
tfidf_matrix = vectorizer.fit_transform(df_clean['texto'])
print(f"   ✓ Matriz TF-IDF creada: {tfidf_matrix.shape}")

print("\n  Calculando matriz de similitud coseno...")
print("     Esto puede tardar un momento para datasets grandes...")

# Calcular similitud coseno
cosine_sim = cosine_similarity(tfidf_matrix)
print(f"   ✓ Matriz de similitud creada: {cosine_sim.shape}")

# Análisis de similitudes
print("\n  Análisis de similitudes:")

# Obtener valores de similitud sin la diagonal
similitudes = []
n = len(cosine_sim)
for i in range(n):
    for j in range(i+1, n):
        similitudes.append(cosine_sim[i][j])

similitudes = np.array(similitudes)

print(f"   • Total de pares comparados: {len(similitudes):,}")
print(f"   • Similitud promedio: {similitudes.mean():.4f}")
print(f"   • Similitud mínima: {similitudes.min():.4f}")
print(f"   • Similitud máxima: {similitudes.max():.4f}")
print(f"   • Similitud mediana: {np.median(similitudes):.4f}")

# Identificar pares muy similares
umbral_alta_similitud = 0.8
print(f"\n  Identificando pares con similitud > {umbral_alta_similitud}...")

pares_similares = []
indices = df_clean.index.tolist()

for i in range(len(cosine_sim)):
    for j in range(i+1, len(cosine_sim)):
        if cosine_sim[i][j] > umbral_alta_similitud:
            pares_similares.append({
                'indice_1': indices[i],
                'indice_2': indices[j],
                'similitud': cosine_sim[i][j],
                'texto_1': df_clean.iloc[i]['texto'][:100] + "...",
                'texto_2': df_clean.iloc[j]['texto'][:100] + "..."
            })

print(f"   • Pares muy similares encontrados: {len(pares_similares)}")

if len(pares_similares) > 0:
    print("\n  Ejemplos de pares muy similares:")
    for idx, par in enumerate(pares_similares[:5], 1):
        print(f"\n   Par {idx} (similitud: {par['similitud']:.4f}):")
        print(f"   Texto 1 (ID {par['indice_1']}): {par['texto_1']}")
        print(f"   Texto 2 (ID {par['indice_2']}): {par['texto_2']}")

# Decidir qué hacer con duplicados similares
print(f"\n  Recomendación: ")
if len(pares_similares) > len(df_clean) * 0.1:
    print(f"   Se encontraron muchos pares similares ({len(pares_similares)}).")
    print(f"   Considera eliminar uno de cada par para reducir redundancia.")
else:
    print(f"   El nivel de redundancia es aceptable.")

# ------------ Análisis temático y conceptual------------
print("\n" + "="*80)
print("[PASO 6] ANÁLISIS TEMÁTICO Y CONCEPTUAL")
print("="*80)

# Extraer conceptos clave del proyecto
conceptos_clave = {
    'crisis_sentido': ['vacío', 'sentido', 'propósito', 'existencial', 'crisis', 'desorientación'],
    'algoritmos': ['algoritmo', 'ia', 'inteligencia artificial', 'sistema', 'automatización'],
    'autonomia': ['autonomía', 'libertad', 'control', 'decisión', 'libre albedrío'],
    'identidad': ['identidad', 'yo', 'ser', 'autenticidad', 'perfil'],
    'hiperconectividad': ['hiperconectado', 'redes sociales', 'digital', 'online', 'conexión'],
    'rendimiento': ['productivo', 'rendimiento', 'eficiencia', 'burnout', 'agotamiento'],
    'inmediatez': ['inmediato', 'rápido', 'efímero', 'instantáneo', 'temporal']
}

print("\n  Análisis de presencia de conceptos clave:")

cobertura_conceptos = {}
for concepto, palabras in conceptos_clave.items():
    patron = '|'.join(palabras)
    matches = df_clean['texto'].str.lower().str.contains(patron, case=False, regex=True)
    count = matches.sum()
    porcentaje = (count / len(df_clean)) * 100
    cobertura_conceptos[concepto] = count
    print(f"   • {concepto}: {count} textos ({porcentaje:.2f}%)")

# Identificar lagunas conceptuales
print("\n Conceptos con baja representación (< 10%):")
for concepto, count in cobertura_conceptos.items():
    porcentaje = (count / len(df_clean)) * 100
    if porcentaje < 10:
        print(f"     {concepto}: {porcentaje:.2f}%")

# ------------Guardar resultados ------------
print("\n" + "="*80)
print("[GUARDANDO RESULTADOS]")
print("="*80)

# Guardar dataset limpio
output_clean = 'dataset_limpio.csv'
df_clean.to_csv(output_clean, index=False)
print(f"\n Dataset limpio guardado: {output_clean}")

# Guardar pares similares
if len(pares_similares) > 0:
    df_similares = pd.DataFrame(pares_similares)
    output_similares = 'pares_similares.csv'
    df_similares.to_csv(output_similares, index=False)
    print(f" Pares similares guardados: {output_similares}")

# Guardar matriz de similitud (muestra)
output_similitud = 'matriz_similitud_muestra.npy'
np.save(output_similitud, cosine_sim[:100, :100])  # Solo primeras 100 filas
print(f" Matriz de similitud (muestra) guardada: {output_similitud}")

print("\n" + "="*80)
print(" ANÁLISIS COMPLETADO")
print("="*80)
print("\nArchivos generados:")
print(f"   1. {output_clean}")
if len(pares_similares) > 0:
    print(f"   2. {output_similares}")
print(f"   3. {output_similitud}")