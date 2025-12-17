import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings
warnings.filterwarnings('ignore')

# ============= CONFIGURACIÓN =============
IMG_SIZE = (64, 64)
CLASS_NAMES = ['perro', 'gato', 'hormiga', 'mariquita', 'tortuga']
OUTPUT_DIR = r'sources/completo'

# ============= CARGAR MODELO =============
def cargar_modelo(ruta='models/mejor_modelo_animales.h5'):
    """Carga el modelo ya entrenado"""
    if not os.path.exists(ruta):
        print(f"No se encontró el modelo en: {ruta}")
        print("\nModelos disponibles:")
        if os.path.exists('models'):
            for archivo in os.listdir('models'):
                print(f"  - models/{archivo}")
        return None
    
    modelo = keras.models.load_model(ruta)
    print(f"Modelo cargado desde: {ruta}")
    return modelo


# ============= PREDICCIÓN SIMPLE =============
def predecir_imagen(ruta_imagen, modelo, mostrar=True):
    """Predice la clase de una imagen"""
    if not os.path.exists(ruta_imagen):
        print(f"No se encontró la imagen: {ruta_imagen}")
        return None, None, None
    
    # Cargar y preprocesar
    img = load_img(ruta_imagen, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predicción
    prediccion = modelo.predict(img_array, verbose=0)
    clase_predicha = np.argmax(prediccion[0])
    confianza = prediccion[0][clase_predicha]
    
    print(f"\n Predicción: {CLASS_NAMES[clase_predicha].upper()}")
    print(f" Confianza: {confianza:.2%}")
    print(f"\nProbabilidades por clase:")
    for i, nombre in enumerate(CLASS_NAMES):
        print(f"  {nombre:12s}: {prediccion[0][i]:.2%}")
    
    if mostrar:
        plt.figure(figsize=(12, 5))
        
        # Mostrar imagen
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title(f'Predicción: {CLASS_NAMES[clase_predicha]}\nConfianza: {confianza:.2%}',
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        # Mostrar probabilidades
        plt.subplot(1, 2, 2)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
        plt.barh(CLASS_NAMES, prediccion[0], color=colors)
        plt.xlabel('Probabilidad')
        plt.title('Probabilidades por Clase', fontsize=14, fontweight='bold')
        plt.xlim([0, 1])
        
        for i, v in enumerate(prediccion[0]):
            plt.text(v + 0.02, i, f'{v:.2%}', va='center')
        
        plt.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, 'prediccion.png'), dpi=300, bbox_inches='tight')
        print(f"\n Gráfica guardada: {OUTPUT_DIR}/prediccion.png")
        plt.show()
    
    return CLASS_NAMES[clase_predicha], confianza, prediccion[0]


# ============= DETECCIÓN MÚLTIPLE =============
def detectar_multiples_animales(ruta_imagen, modelo, umbral=0.3, mostrar=True):
    """Detecta múltiples animales con umbral de probabilidad"""
    if not os.path.exists(ruta_imagen):
        print(f" No se encontró la imagen: {ruta_imagen}")
        return []
    
    # Cargar y preprocesar
    img = load_img(ruta_imagen, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predicción
    prediccion = modelo.predict(img_array, verbose=0)[0]
    
    # Encontrar clases por encima del umbral
    animales_detectados = []
    for idx, prob in enumerate(prediccion):
        if prob >= umbral:
            animales_detectados.append((CLASS_NAMES[idx], prob))
    
    # Ordenar por probabilidad
    animales_detectados.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n Animales detectados (umbral={umbral}):")
    if animales_detectados:
        for animal, prob in animales_detectados:
            print(f"  ✓ {animal:12s}: {prob:.2%}")
    else:
        print(f"Ningún animal supera el umbral de {umbral:.0%}")
    
    if mostrar:
        # Visualización
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        titulo = "Animales detectados:\n"
        if animales_detectados:
            for animal, prob in animales_detectados:
                titulo += f"{animal}: {prob:.2%}\n"
        else:
            titulo += "Ninguno supera el umbral"
        plt.title(titulo, fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        colors = ['green' if p >= umbral else 'gray' for p in prediccion]
        plt.barh(CLASS_NAMES, prediccion, color=colors)
        plt.axvline(x=umbral, color='red', linestyle='--', label=f'Umbral ({umbral})')
        plt.xlabel('Probabilidad')
        plt.title('Probabilidades de Detección', fontsize=14, fontweight='bold')
        plt.legend()
        
        for i, v in enumerate(prediccion):
            plt.text(v + 0.02, i, f'{v:.2%}', va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'deteccion_multiple.png'), dpi=300, bbox_inches='tight')
        print(f" Gráfica guardada: {OUTPUT_DIR}/deteccion_multiple.png")
        plt.show()
    
    return animales_detectados


# ============= PREDICCIÓN POR LOTES =============
def predecir_carpeta(ruta_carpeta, modelo, mostrar_imagenes=False):
    """Predice todas las imágenes en una carpeta"""
    if not os.path.exists(ruta_carpeta):
        print(f" No se encontró la carpeta: {ruta_carpeta}")
        return
    
    archivos = [f for f in os.listdir(ruta_carpeta) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not archivos:
        print(" No se encontraron imágenes en la carpeta")
        return
    
    print(f"\n Analizando {len(archivos)} imágenes...\n")
    resultados = []
    
    for archivo in archivos:
        ruta = os.path.join(ruta_carpeta, archivo)
        img = load_img(ruta, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediccion = modelo.predict(img_array, verbose=0)
        clase = np.argmax(prediccion[0])
        confianza = prediccion[0][clase]
        
        resultados.append((archivo, CLASS_NAMES[clase], confianza))
        print(f"  {archivo:30s} → {CLASS_NAMES[clase]:12s} ({confianza:.2%})")
    
    return resultados


# ============= MENÚ INTERACTIVO =============
def menu_interactivo(modelo):
    """Menú para usar el detector"""
    while True:
        print("\n" + "="*60)
        print(" DETECTOR DE ANIMALES")
        print("="*60)
        print("1. Predecir una imagen")
        print("2. Detección múltiple (con umbral)")
        print("3. Analizar carpeta completa")
        print("4. Salir")
        print("="*60)
        
        opcion = input("\nSelecciona una opción (1-4): ").strip()
        
        if opcion == '1':
            ruta = input("\n Ruta de la imagen: ").strip()
            predecir_imagen(ruta, modelo)
            
        elif opcion == '2':
            ruta = input("\n Ruta de la imagen: ").strip()
            try:
                umbral = float(input("🎚️  Umbral (0.0-1.0, recomendado 0.3): ").strip())
                detectar_multiples_animales(ruta, modelo, umbral)
            except ValueError:
                print(" Umbral inválido, usando 0.3")
                detectar_multiples_animales(ruta, modelo, 0.3)
                
        elif opcion == '3':
            ruta = input("\n Ruta de la carpeta: ").strip()
            predecir_carpeta(ruta, modelo)
            
        elif opcion == '4':
            print("\n ¡Hasta luego!")
            break
        else:
            print(" Opción inválida")


# ============= MAIN =============
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" DETECTOR DE ANIMALES - MODO PREDICCIÓN")
    print("="*60)
    
    # Cargar modelo
    modelo = cargar_modelo()
    
    if modelo is None:
        print("\n No se pudo cargar el modelo. Asegúrate de haberlo entrenado primero.")
    else:
        # Iniciar menú interactivo
        menu_interactivo(modelo)
