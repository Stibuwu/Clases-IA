import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# ============= CONFIGURACIÓN MEJORADA =============
# Opción 1: MODELO PROFUNDO (sin transfer learning)
# Opción 2: TRANSFER LEARNING (MobileNetV2 - recomendado)
# Opción 3: SUPER MODELO (VGG16/ResNet50 - requiere más tiempo)

USAR_TRANSFER_LEARNING = False  #Cambiar a False para usar modelo profundo

IMG_SIZE = (128, 128)        
BATCH_SIZE = 16
EPOCHS = 30                  
LEARNING_RATE = 0.0001       # Learning rate más bajo
MAX_IMAGES_PER_CLASS = 800   # Más imágenes por clase

DATA_DIR = r'Data'
OUTPUT_DIR = r'sources/mejorado'

CATEGORIAS = {
    'perro': os.path.join(DATA_DIR, 'perro'),
    'gato': os.path.join(DATA_DIR, 'gato'),
    'hormiga': os.path.join(DATA_DIR, 'animals', 'hormiga'),
    'mariquita': os.path.join(DATA_DIR, 'animals', 'mariquita'),
    'tortuga': os.path.join(DATA_DIR, 'Turtle_Tortoise')
}
CLASS_NAMES = list(CATEGORIAS.keys())
NUM_CLASSES = len(CLASS_NAMES)

print("="*70)
print(" ENTRENAMIENTO MEJORADO - CNN OPTIMIZADA")
print("="*70)
print(f"Modo: {'TRANSFER LEARNING (MobileNetV2)' if USAR_TRANSFER_LEARNING else 'MODELO PROFUNDO PERSONALIZADO'}")
print(f"Tamaño de imagen: {IMG_SIZE}")
print(f"Épocas máximas: {EPOCHS}")
print(f"Imágenes por clase: {MAX_IMAGES_PER_CLASS}")
print("="*70 + "\n")


# ============= CARGAR DATOS CON BALANCEO =============
def cargar_datos_balanceados():
    """Carga datos asegurando balance entre clases"""
    X, y = [], []
    conteos = {}
    
    print(" Cargando y balanceando dataset...")
    
    for idx, (nombre, ruta) in enumerate(CATEGORIAS.items()):
        if not os.path.exists(ruta):
            print(f" Saltando {nombre} - carpeta no encontrada")
            continue
        
        archivos = [f for f in os.listdir(ruta) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Mezclar y limitar
        np.random.shuffle(archivos)
        archivos = archivos[:MAX_IMAGES_PER_CLASS]
        
        for archivo in archivos:
            try:
                img = load_img(os.path.join(ruta, archivo), target_size=IMG_SIZE)
                X.append(img_to_array(img))
                y.append(idx)
            except:
                pass
        
        conteos[nombre] = sum(1 for label in y if label == idx)
        print(f" {nombre:12s}: {conteos[nombre]:4d} imágenes")
    
    # Verificar balance
    print(f"\n Balance del dataset:")
    min_count = min(conteos.values())
    max_count = max(conteos.values())
    ratio = max_count / min_count if min_count > 0 else 0
    print(f"  Rango: {min_count} - {max_count} imágenes")
    print(f"  Ratio desbalance: {ratio:.2f}x")
    
    if ratio > 2.0:
        print("!!!! ADVERTENCIA: Dataset desbalanceado (ratio > 2x)")
        print("     Considera agregar más imágenes a las clases minoritarias")
    
    return np.array(X), np.array(y)


# ============= MODELO 1: CNN PROFUNDO PERSONALIZADO =============
def crear_modelo_profundo(input_shape, num_classes):
    """Modelo CNN más profundo y robusto (sin transfer learning)"""
    model = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque 4 (nuevo)
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Capas densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        
        # Salida
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============= MODELO 2: TRANSFER LEARNING (MobileNetV2) =============
def crear_modelo_transfer_learning(input_shape, num_classes):
    """Modelo con transfer learning usando MobileNetV2 pre-entrenado"""
    
    # Cargar MobileNetV2 sin la capa superior (include_top=False)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar las primeras capas del modelo base
    base_model.trainable = False
    
    # Construir modelo completo
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ============= DATA AUGMENTATION AGRESIVO =============
def crear_generadores_mejorados():
    """Data augmentation más agresivo para mejor generalización"""
    train_datagen = ImageDataGenerator(
        rotation_range=40,           
        width_shift_range=0.3,       
        height_shift_range=0.3,
        shear_range=0.3,             
        zoom_range=0.3,              
        horizontal_flip=True,
        vertical_flip=False,         
        brightness_range=[0.7, 1.3], 
        fill_mode='nearest'
    )
    
    # Validación sin augmentation
    val_datagen = ImageDataGenerator()
    
    return train_datagen, val_datagen


# ============= ENTRENAR CON MEJORAS =============
def entrenar_modelo_mejorado(model, X_train, y_train, X_val, y_val):
    """Entrena con callbacks optimizados"""
    
    train_datagen, val_datagen = crear_generadores_mejorados()
    
    os.makedirs('models', exist_ok=True)
    
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',  # Cambiado a val_accuracy
            patience=5,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            'models/mejor_modelo_mejorado.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]
    
    print("\n" + "="*70)
    print(" INICIANDO ENTRENAMIENTO MEJORADO")
    print("="*70)
    
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n" + "="*70)
    print(" ENTRENAMIENTO COMPLETADO")
    print("="*70)
    
    return history


# ============= VISUALIZACIONES =============
def visualizar_resultados(history, y_true, y_pred, X_test):
    """Genera todas las visualizaciones"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Curvas de entrenamiento
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history.history['loss'], label='Train', linewidth=2)
    ax1.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax1.set_title('Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax2.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'entrenamiento.png'), dpi=300)
    print(f"✓ Gráfica guardada: {OUTPUT_DIR}/entrenamiento.png")
    plt.close()
    
    # 2. Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Matriz de Confusión', fontsize=16, fontweight='bold')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    
    # Calcular accuracy por clase
    accuracies = cm.diagonal() / cm.sum(axis=1)
    plt.figtext(0.02, 0.02, 
                f"Accuracy por clase:\n" + 
                "\n".join([f"{CLASS_NAMES[i]}: {acc:.1%}" for i, acc in enumerate(accuracies)]),
                fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'matriz_confusion.png'), dpi=300)
    print(f"✓ Gráfica guardada: {OUTPUT_DIR}/matriz_confusion.png")
    plt.close()
    
    # 3. Errores más comunes
    print("\n ANÁLISIS DE CONFUSIONES:")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                porcentaje = (cm[i, j] / cm[i].sum()) * 100
                if porcentaje > 10:  # Solo mostrar confusiones > 10%
                    print(f" {CLASS_NAMES[i]} → {CLASS_NAMES[j]}: {cm[i, j]} casos ({porcentaje:.1f}%)")


# ============= MAIN =============
def main():
    """Pipeline completo de entrenamiento mejorado"""
    
    # 1. Cargar datos
    X, y = cargar_datos_balanceados()
    
    # 2. Preparar datos
    X = X.astype('float32') / 255.0
    y_categorical = to_categorical(y, NUM_CLASSES)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42,
        stratify=np.argmax(y_temp, axis=1)
    )
    
    print(f"\n Datos preparados:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    # 3. Crear modelo
    print(f"\n Creando modelo...")
    if USAR_TRANSFER_LEARNING:
        model = crear_modelo_transfer_learning((*IMG_SIZE, 3), NUM_CLASSES)
        print(" Modelo con Transfer Learning (MobileNetV2)")
    else:
        model = crear_modelo_profundo((*IMG_SIZE, 3), NUM_CLASSES)
        print(" Modelo CNN profundo personalizado")
    
    print(f" Parámetros totales: {model.count_params():,}")
    
    # 4. Entrenar
    history = entrenar_modelo_mejorado(model, X_train, y_train, X_val, y_val)
    
    # 5. Evaluar
    print("\n Evaluando modelo...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Precisión en test: {test_acc:.2%}")
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\n Reporte de clasificación:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    
    # 6. Visualizar
    visualizar_resultados(history, y_true, y_pred, X_test)
    
    # 7. Guardar
    model.save('models/modelo_mejorado_completo.h5')
    print(f"\n Modelo guardado: models/modelo_mejorado_completo.h5")
    print(f" Mejor modelo: models/mejor_modelo_mejorado.h5")
    
    print("\n" + "="*70)
    print(" ¡PROCESO COMPLETADO!")
    print("="*70)
    print("\n MEJORAS APLICADAS:")
    print("   Imágenes más grandes (128x128)")
    print("   Modelo más profundo" + (" con Transfer Learning" if USAR_TRANSFER_LEARNING else ""))
    print("   Data augmentation agresivo")
    print("   Más épocas de entrenamiento")
    print("   Dataset balanceado")
    print("\n SIGUIENTE PASO:")
    print("  Usa usar_detector.py con el nuevo modelo:")
    print("  cargar_modelo('models/mejor_modelo_mejorado.h5')")
    print("="*70)
    
    return model


if __name__ == "__main__":
    modelo = main()
