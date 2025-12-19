import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings
warnings.filterwarnings('ignore')

# Configuración
PRUEBAS_DIR = 'Pruebas'
DATASET_DIR = 'Data/dataset2'
IMG_SIZE = (128, 128)

CATEGORIAS = {
    'perro': 'perros',
    'gato': 'gatos', 
    'hormiga': 'hormigas',
    'mariquita': 'mariquitas',
    'tortuga': 'tortugas'
}

def analizar_imagen(ruta):
    """Extrae características de una imagen"""
    try:
        img = Image.open(ruta)
        
        # Información básica
        info = {
            'ruta': ruta,
            'formato': img.format,
            'modo': img.mode,
            'tamaño_original': img.size,
            'aspecto_ratio': img.size[0] / img.size[1] if img.size[1] > 0 else 0
        }
        
        # Convertir a RGB si es necesario
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Características de color
        img_array = np.array(img)
        info['color_promedio'] = img_array.mean(axis=(0,1))
        info['brillo_promedio'] = img_array.mean()
        info['desviacion_std'] = img_array.std()
        
        return info
    except Exception as e:
        print(f"Error analizando {ruta}: {e}")
        return None

def comparar_con_dataset(clase_prueba, imgs_prueba):
    """Compara imágenes de prueba con dataset de entrenamiento"""
    print(f"\n{'='*80}")
    print(f"ANÁLISIS: {clase_prueba.upper()}")
    print(f"{'='*80}")
    
    # Analizar imágenes de prueba
    print(f"\n📋 Imágenes de prueba ({len(imgs_prueba)}):")
    prueba_stats = []
    for img_path in imgs_prueba:
        info = analizar_imagen(img_path)
        if info:
            prueba_stats.append(info)
            print(f"  • {os.path.basename(img_path)}")
            print(f"    Tamaño: {info['tamaño_original']}, Formato: {info['formato']}")
            print(f"    Brillo: {info['brillo_promedio']:.1f}, Aspecto: {info['aspecto_ratio']:.2f}")
    
    # Analizar muestra del dataset
    dataset_path = os.path.join(DATASET_DIR, CATEGORIAS[clase_prueba])
    if os.path.exists(dataset_path):
        archivos_dataset = [f for f in os.listdir(dataset_path) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:50]
        
        print(f"\n📚 Dataset de entrenamiento (muestra de {len(archivos_dataset)}):")
        dataset_stats = []
        for archivo in archivos_dataset:
            info = analizar_imagen(os.path.join(dataset_path, archivo))
            if info:
                dataset_stats.append(info)
        
        # Estadísticas comparativas
        if dataset_stats:
            dataset_brillo = np.mean([s['brillo_promedio'] for s in dataset_stats])
            dataset_aspecto = np.mean([s['aspecto_ratio'] for s in dataset_stats])
            dataset_std = np.mean([s['desviacion_std'] for s in dataset_stats])
            
            prueba_brillo = np.mean([s['brillo_promedio'] for s in prueba_stats])
            prueba_aspecto = np.mean([s['aspecto_ratio'] for s in prueba_stats])
            prueba_std = np.mean([s['desviacion_std'] for s in prueba_stats])
            
            print(f"  Brillo promedio: {dataset_brillo:.1f}")
            print(f"  Aspecto promedio: {dataset_aspecto:.2f}")
            print(f"  Variabilidad (std): {dataset_std:.1f}")
            
            print(f"\n🔍 COMPARACIÓN:")
            print(f"  Diferencia de brillo: {abs(prueba_brillo - dataset_brillo):.1f} "
                  f"({'⚠️ ALTA' if abs(prueba_brillo - dataset_brillo) > 30 else '✓ OK'})")
            print(f"  Diferencia de aspecto: {abs(prueba_aspecto - dataset_aspecto):.2f} "
                  f"({'⚠️ ALTA' if abs(prueba_aspecto - dataset_aspecto) > 0.3 else '✓ OK'})")
            print(f"  Diferencia de variabilidad: {abs(prueba_std - dataset_std):.1f} "
                  f"({'⚠️ ALTA' if abs(prueba_std - dataset_std) > 20 else '✓ OK'})")
    
    return prueba_stats, dataset_stats if 'dataset_stats' in locals() else []

def visualizar_comparacion(clase):
    """Visualiza imágenes de prueba vs dataset"""
    # Obtener imágenes de prueba
    prueba_dir = os.path.join(PRUEBAS_DIR, clase if clase != 'gatos' else 'gatos')
    if not os.path.exists(prueba_dir):
        prueba_dir = os.path.join(PRUEBAS_DIR, clase + 's')
    if not os.path.exists(prueba_dir):
        return
    
    imgs_prueba = [os.path.join(prueba_dir, f) for f in os.listdir(prueba_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))][:5]
    
    # Obtener imágenes del dataset
    dataset_path = os.path.join(DATASET_DIR, CATEGORIAS[clase])
    if not os.path.exists(dataset_path):
        return
    
    imgs_dataset = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:5]
    
    # Visualizar
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle(f'Comparación: {clase.upper()}\nFila 1: Pruebas | Fila 2: Dataset', 
                 fontsize=16, fontweight='bold')
    
    # Fila 1: Pruebas
    for i, img_path in enumerate(imgs_prueba):
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Prueba {i+1}', fontsize=10)
            axes[0, i].axis('off')
        except:
            axes[0, i].axis('off')
    
    # Fila 2: Dataset
    for i, img_path in enumerate(imgs_dataset):
        try:
            img = load_img(img_path, target_size=IMG_SIZE)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f'Dataset {i+1}', fontsize=10)
            axes[1, i].axis('off')
        except:
            axes[1, i].axis('off')
    
    plt.tight_layout()
    os.makedirs('sources/analisis', exist_ok=True)
    plt.savefig(f'sources/analisis/comparacion_{clase}.png', dpi=200, bbox_inches='tight')
    print(f"  💾 Guardado: sources/analisis/comparacion_{clase}.png")
    plt.close()

def analisis_completo():
    """Análisis completo de todas las clases"""
    print("\n" + "="*80)
    print("ANÁLISIS DE IMÁGENES DE PRUEBA vs DATASET DE ENTRENAMIENTO")
    print("="*80)
    
    resultados = {}
    
    for clase in CATEGORIAS.keys():
        # Obtener rutas
        prueba_dir = os.path.join(PRUEBAS_DIR, clase if clase != 'gatos' else 'gatos')
        if not os.path.exists(prueba_dir):
            prueba_dir = os.path.join(PRUEBAS_DIR, clase + 's')
        
        if not os.path.exists(prueba_dir):
            print(f"\n⚠️ No se encontró carpeta de pruebas para {clase}")
            continue
        
        imgs_prueba = [os.path.join(prueba_dir, f) for f in os.listdir(prueba_dir)
                       if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        
        if imgs_prueba:
            prueba_stats, dataset_stats = comparar_con_dataset(clase, imgs_prueba)
            resultados[clase] = {
                'prueba': prueba_stats,
                'dataset': dataset_stats
            }
            
            # Visualizar
            visualizar_comparacion(clase)
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN Y RECOMENDACIONES")
    print("="*80)
    
    return resultados

if __name__ == "__main__":
    resultados = analisis_completo()
    
    print("\n📊 Análisis visual guardado en: sources/analisis/")
    print("\n💡 CONCLUSIONES PRELIMINARES:")
    print("  1. Revisa las comparaciones visuales en sources/analisis/")
    print("  2. Identifica diferencias significativas de brillo/aspecto")
    print("  3. Considera si las imágenes de prueba son representativas")
    print("\n✨ Análisis completado!")
