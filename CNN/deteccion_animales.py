import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings

#---------------------TensorFlow y Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')

#-------------------------Configuracion Global
# CONFIGURACIÓN OPTIMIZADA (reducida para evitar congelamiento)
IMG_SIZE = (64, 64)          # Reducido de 150x150 (usa menos memoria)
BATCH_SIZE = 16              # Reducido de 32 (más eficiente)
EPOCHS = 15                  # Reducido de 50 (más rápido)
LEARNING_RATE = 0.001
MAX_IMAGES_PER_CLASS = 500   # LÍMITE: máximo 500 imágenes por clase

DATA_DIR = r'Data'
OUTPUT_DIR = r'sources/completo'  # Carpeta para guardar imágenes generadas

CATEGORIAS = {
    'perro': os.path.join(DATA_DIR, 'perro'),
    'gato': os.path.join(DATA_DIR, 'gato'),
    'hormiga': os.path.join(DATA_DIR, 'animals', 'hormiga'),
    'mariquita': os.path.join(DATA_DIR, 'animals', 'mariquita'),
    'tortuga': os.path.join(DATA_DIR, 'Turtle_Tortoise')
}

CLASS_NAMES = list(CATEGORIAS.keys())
NUM_CLASSES = len(CLASS_NAMES)

#----------------------Funciones de carga y preparación de datos
def verificar_dataset():
    """Verifica que existan las carpetas del dataset y cuenta las imágenes"""
    print("="*60)
    print("VERIFICACIÓN DEL DATASET")
    print("="*60)
    print(f"Categorías: {CLASS_NAMES}")
    print(f"Número de clases: {NUM_CLASSES}\n")
    
    for nombre, ruta in CATEGORIAS.items():
        if os.path.exists(ruta):
            num_imgs = len([f for f in os.listdir(ruta) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"✓ {nombre:12s}: {num_imgs:5d} imágenes")
        else:
            print(f"✗ {nombre:12s}: Carpeta no encontrada")
    print("="*60)


def cargar_imagenes_y_etiquetas():
    """Carga imágenes con LÍMITE por clase para evitar congelamiento"""
    imagenes = []
    etiquetas = []
    
    print("\n" + "="*60)
    print("CARGANDO IMÁGENES (MODO OPTIMIZADO)")
    print(f"Máximo {MAX_IMAGES_PER_CLASS} imágenes por clase")
    print("="*60)
    
    for idx, (nombre_clase, ruta_clase) in enumerate(CATEGORIAS.items()):
        print(f"\nCargando {nombre_clase}...")
        
        if not os.path.exists(ruta_clase):
            print(f"!!!!!Saltando {nombre_clase} - carpeta no encontrada!!!!!")
            continue
            
        archivos = [f for f in os.listdir(ruta_clase) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # LIMITADOR: Solo tomar MAX_IMAGES_PER_CLASS imágenes
        archivos = archivos[:MAX_IMAGES_PER_CLASS]
        total_archivos = len(archivos)
        
        for i, archivo in enumerate(archivos):
            try:
                ruta_img = os.path.join(ruta_clase, archivo)
                img = load_img(ruta_img, target_size=IMG_SIZE)
                img_array = img_to_array(img)
                imagenes.append(img_array)
                etiquetas.append(idx)
                
                if (i + 1) % 100 == 0:
                    print(f"  Cargadas {i + 1}/{total_archivos} imágenes")
            except Exception as e:
                print(f"  Error cargando {archivo}: {e}")
        
        print(f"{nombre_clase}: {len([e for e in etiquetas if e == idx])} imágenes cargadas")
    
    print("\n" + "="*60)
    print("DATASET CARGADO EXITOSAMENTE")
    print("="*60)
    print(f"Total de imágenes: {len(imagenes)}")
    print(f"Forma de X: ({len(imagenes)}, {IMG_SIZE[0]}, {IMG_SIZE[1]}, 3)")
    print("="*60 + "\n")
    
    return np.array(imagenes), np.array(etiquetas)


def preparar_datos(X, y):
    """Normaliza y divide los datos en train, validation y test"""
    print("Preparando datos...")
    
    # Normalizar (0-255 -> 0-1)
    X = X.astype('float32') / 255.0
    
    # Convertir etiquetas a one-hot encoding
    y_categorical = to_categorical(y, NUM_CLASSES)
    
    # Dividir en train, validation y test (70%, 15%, 15%)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, 
        stratify=np.argmax(y_temp, axis=1)
    )
    
    print("\nDivisión del dataset:")
    print(f"  Entrenamiento: {X_train.shape[0]} imágenes")
    print(f"  Validación:    {X_val.shape[0]} imágenes")
    print(f"  Test:          {X_test.shape[0]} imágenes\n")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


#------------------------------------Visualización

def visualizar_distribucion(y):
    """Visualiza la distribución de clases en el dataset"""
    plt.figure(figsize=(12, 5))
    
    unique, counts = np.unique(y, return_counts=True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
    plt.bar([CLASS_NAMES[i] for i in unique], counts, color=colors)
    plt.title('Distribución de Clases en el Dataset', fontsize=14, fontweight='bold')
    plt.xlabel('Clase')
    plt.ylabel('Número de Imágenes')
    plt.xticks(rotation=45)
    
    for i, v in enumerate(counts):
        plt.text(i, v + 20, str(v), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, 'distribucion_clases.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {OUTPUT_DIR}/distribucion_clases.png")
    plt.close()


def visualizar_muestras(X, y, num_muestras=10):
    """Visualiza muestras aleatorias del dataset"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Muestras del Dataset', fontsize=16, fontweight='bold')
    
    for idx in range(NUM_CLASSES):
        indices_clase = np.where(y == idx)[0]
        if len(indices_clase) > 0:
            muestras = np.random.choice(indices_clase, min(2, len(indices_clase)), replace=False)
            for i, muestra in enumerate(muestras):
                ax = axes[i, idx]
                ax.imshow(X[muestra].astype('uint8'))
                ax.set_title(CLASS_NAMES[idx], fontsize=10)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'muestras_dataset.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Gráfica guardada: {OUTPUT_DIR}/muestras_dataset.png")
    plt.close()


def visualizar_data_augmentation(X_train, train_datagen):
    """Visualiza ejemplos de data augmentation"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('Ejemplos de Data Augmentation', fontsize=16, fontweight='bold')
    
    sample_idx = np.random.randint(0, X_train.shape[0])
    sample_img = X_train[sample_idx:sample_idx+1]
    
    i = 0
    for batch in train_datagen.flow(sample_img, batch_size=1):
        ax = axes[i // 5, i % 5]
        ax.imshow(batch[0])
        ax.axis('off')
        i += 1
        if i >= 10:
            break
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'data_augmentation.png'), dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {OUTPUT_DIR}/data_augmentation.png")
    plt.close()


def visualizar_entrenamiento(history):
    """Visualiza las curvas de pérdida y precisión durante el entrenamiento"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Pérdida
    ax1.plot(history.history['loss'], label='Entrenamiento', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validación', linewidth=2)
    ax1.set_title('Pérdida del Modelo', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Pérdida')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precisión
    ax2.plot(history.history['accuracy'], label='Entrenamiento', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validación', linewidth=2)
    ax2.set_title('Precisión del Modelo', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Precisión')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'curvas_entrenamiento.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfica guardada: {OUTPUT_DIR}/curvas_entrenamiento.png")
    plt.close()
    
    # Métricas finales
    print("\nMétricas Finales:")
    print(f"  Precisión entrenamiento: {history.history['accuracy'][-1]:.4f}")
    print(f"  Precisión validación:    {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Pérdida entrenamiento:   {history.history['loss'][-1]:.4f}")
    print(f"  Pérdida validación:      {history.history['val_loss'][-1]:.4f}")


def visualizar_matriz_confusion(y_true, y_pred):
    """Visualiza la matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'matriz_confusion.png'), dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {OUTPUT_DIR}/matriz_confusion.png")
    plt.close()


def visualizar_predicciones(X, y_true, y_pred, num_ejemplos=20):
    """Visualiza predicciones correctas e incorrectas"""
    correctas = np.where(y_true == y_pred)[0]
    incorrectas = np.where(y_true != y_pred)[0]
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig.suptitle('Predicciones del Modelo', fontsize=18, fontweight='bold')
    
    # Predicciones correctas (primeras 10)
    for i in range(min(10, len(correctas))):
        idx = correctas[np.random.choice(len(correctas))]
        ax = axes[i // 5, i % 5]
        ax.imshow(X[idx])
        ax.set_title(f'Real: {CLASS_NAMES[y_true[idx]]}\nPredicho: {CLASS_NAMES[y_pred[idx]]}',
                    color='green', fontweight='bold')
        ax.axis('off')
    
    # Predicciones incorrectas (siguientes 10)
    for i in range(min(10, len(incorrectas))):
        idx = incorrectas[min(i, len(incorrectas)-1)]
        ax = axes[(i + 10) // 5, (i + 10) % 5]
        ax.imshow(X[idx])
        ax.set_title(f'✗ Real: {CLASS_NAMES[y_true[idx]]}\nPredicho: {CLASS_NAMES[y_pred[idx]]}',
                    color='red', fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'predicciones.png'), dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {OUTPUT_DIR}/predicciones.png")
    plt.close()


# ------------------------------------Modelo CNN
def crear_modelo_cnn(input_shape, num_classes):
    """Crea modelo CNN OPTIMIZADO (más ligero y rápido)"""
    model = models.Sequential([
        # Primera capa convolucional
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Segunda capa convolucional (reducida)
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Capas densas (simplificadas)
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        
        # Capa de salida
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def entrenar_modelo(model, X_train, y_train, X_val, y_val):
    """Entrena el modelo CNN con data augmentation y callbacks"""
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Crear carpeta para modelos
    os.makedirs('models', exist_ok=True)
    
    # Callbacks (optimizados para detener antes)
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,  # Reducido de 10 a 3
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,  # Reducido de 5 a 2
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'models/mejor_modelo_animales.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60)
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*60)
    print("¡ENTRENAMIENTO COMPLETADO!")
    print("="*60)
    
    return history, train_datagen


def evaluar_modelo(model, X_test, y_test):
    """Evalúa el modelo en el conjunto de test"""
    print("\n" + "="*60)
    print("EVALUACIÓN EN TEST SET")
    print("="*60)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Pérdida:   {test_loss:.4f}")
    print(f"Precisión: {test_accuracy:.4f}")
    print("="*60)
    
    # Predicciones
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Reporte de clasificación
    print("\nReporte de Clasificación:\n")
    print(classification_report(y_true_classes, y_pred_classes, target_names=CLASS_NAMES))
    
    return y_true_classes, y_pred_classes


#-------------------------------Predicción y detección
def predecir_imagen(ruta_imagen, modelo, mostrar=True):
    """Predice la clase de una nueva imagen"""
    # Cargar y preprocesar
    img = load_img(ruta_imagen, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predicción
    prediccion = modelo.predict(img_array, verbose=0)
    clase_predicha = np.argmax(prediccion[0])
    confianza = prediccion[0][clase_predicha]
    
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
        plt.savefig(os.path.join(OUTPUT_DIR, 'prediccion_individual.png'), dpi=300, bbox_inches='tight')
        print(f"✓ Gráfica guardada: {OUTPUT_DIR}/prediccion_individual.png")
        plt.close()
    
    return CLASS_NAMES[clase_predicha], confianza, prediccion[0]


def detectar_multiples_animales(ruta_imagen, modelo, umbral=0.3):
    """Detecta múltiples animales usando umbral de probabilidad"""
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
    
    # Visualización
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    titulo = "Animales detectados:\n"
    for animal, prob in animales_detectados:
        titulo += f"{animal}: {prob:.2%}\n"
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
    print(f"Gráfica guardada: {OUTPUT_DIR}/deteccion_multiple.png")
    plt.close()
    
    return animales_detectados


def guardar_modelo(model):
    """Guarda el modelo en diferentes formatos"""
    os.makedirs('models', exist_ok=True)
    
    # Modelo completo en formato nativo Keras (recomendado)
    model.save('models/modelo_deteccion_animales.keras')
    print("Modelo guardado: models/modelo_deteccion_animales.keras")
    
    # También guardar en formato H5 (legacy, para compatibilidad)
    model.save('models/modelo_deteccion_animales_completo.h5')
    print("Modelo H5 guardado: models/modelo_deteccion_animales_completo.h5")
    
    # Solo pesos (formato correcto para Keras 3.x)
    model.save_weights('models/pesos_modelo_animales.weights.h5')
    print("Pesos guardados: models/pesos_modelo_animales.weights.h5")


def cargar_modelo(ruta='models/modelo_deteccion_animales.keras'):
    """Carga un modelo guardado"""
    modelo = keras.models.load_model(ruta)
    print(f"Modelo cargado desde: {ruta}")
    return modelo


#----------------------------------Función principal
def main():
    """Función principal que ejecuta todo el pipeline"""
    
    print("\n" + "="*60)
    print("CNN - DETECCIÓN DE ANIMALES (MODO OPTIMIZADO)")
    print("="*60)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Keras version: {keras.__version__}")
    print(f"GPU disponible: {tf.config.list_physical_devices('GPU')}")
    print(f"\nCONFIGURACIÓN:")
    print(f"  • Tamaño imagen: {IMG_SIZE}")
    print(f"  • Máx imágenes/clase: {MAX_IMAGES_PER_CLASS}")
    print(f"  • Épocas: {EPOCHS}")
    print(f"  • Batch size: {BATCH_SIZE}")
    print("="*60 + "\n")
    
    # 1. Verificar dataset
    verificar_dataset()
    
    # 2. Cargar datos
    X, y = cargar_imagenes_y_etiquetas()
    
    # 3. Visualizar distribución
    visualizar_distribucion(y)
    visualizar_muestras(X, y)
    
    # 4. Preparar datos
    X_train, X_val, X_test, y_train, y_val, y_test = preparar_datos(X, y)
    
    # 5. Crear modelo
    print("\nCreando modelo CNN...")
    model = crear_modelo_cnn(input_shape=(*IMG_SIZE, 3), num_classes=NUM_CLASSES)
    model.summary()
    
    # 6. Entrenar modelo
    history, train_datagen = entrenar_modelo(model, X_train, y_train, X_val, y_val)
    
    # 7. Visualizar data augmentation
    visualizar_data_augmentation(X_train, train_datagen)
    
    # 8. Visualizar entrenamiento
    visualizar_entrenamiento(history)
    
    # 9. Evaluar modelo
    y_true_classes, y_pred_classes = evaluar_modelo(model, X_test, y_test)
    
    # 10. Visualizar resultados
    visualizar_matriz_confusion(y_true_classes, y_pred_classes)
    visualizar_predicciones(X_test, y_true_classes, y_pred_classes)
    
    # 11. Guardar modelo
    guardar_modelo(model)
    
    print("\n" + "="*60)
    print("¡PROCESO COMPLETADO!")
    print("="*60)
    print("\nArchivos generados:")
    print("  - Gráficas: distribucion_clases.png, muestras_dataset.png, etc.")
    print("  - Modelos: models/mejor_modelo_animales.h5")
    print("\nPara usar el modelo:")
    print("  modelo = cargar_modelo()")
    print("  predecir_imagen('ruta/imagen.jpg', modelo)")
    print("  detectar_multiples_animales('ruta/imagen.jpg', modelo, umbral=0.3)")
    print("="*60 + "\n")
    
    return model


if __name__ == "__main__":
    # Ejecutar el pipeline completo
    modelo = main()
    
    # Ejemplo de uso después del entrenamiento
    print("\n¿Deseas hacer predicciones? (s/n): ", end="")
    respuesta = input().lower()
    
    if respuesta == 's':
        print("\nIngresa la ruta de la imagen: ", end="")
        ruta = input()
        
        if os.path.exists(ruta):
            print("\n1. Predicción simple")
            print("2. Detección múltiple")
            print("Selecciona opción (1/2): ", end="")
            opcion = input()
            
            if opcion == '1':
                clase, confianza, probs = predecir_imagen(ruta, modelo)
                print(f"\nPredicción: {clase} (Confianza: {confianza:.2%})")
            elif opcion == '2':
                print("Ingresa el umbral (0.0-1.0, recomendado 0.3): ", end="")
                umbral = float(input())
                animales = detectar_multiples_animales(ruta, modelo, umbral)
                print(f"\nAnimales detectados: {len(animales)}")
                for animal, prob in animales:
                    print(f"  - {animal}: {prob:.2%}")
        else:
            print("La ruta no existe.")
