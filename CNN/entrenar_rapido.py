
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# ============= CONFIGURACIÓN ULTRA RÁPIDA =============
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 10
MAX_IMAGES_PER_CLASS = 200  # Solo 200 por clase = ~1000 total

DATA_DIR = r'Data'
OUTPUT_DIR = r'sources/rapido'  # Carpeta para guardar imágenes generadas

CATEGORIAS = {
    'perro': os.path.join(DATA_DIR, 'dataset2', 'perros'),
    'gato': os.path.join(DATA_DIR, 'dataset2', 'gatos'),
    'hormiga': os.path.join(DATA_DIR, 'dataset2', 'hormigas'),
    'mariquita': os.path.join(DATA_DIR, 'dataset2', 'mariquitas'),
    'tortuga': os.path.join(DATA_DIR, 'dataset2', 'tortugas')
}
CLASS_NAMES = list(CATEGORIAS.keys())
NUM_CLASSES = len(CLASS_NAMES)

print("="*60)
print("CNN RÁPIDA - MODO PRUEBA")
print("="*60)
print(f"Configuración: {IMG_SIZE}, {MAX_IMAGES_PER_CLASS} imgs/clase, {EPOCHS} épocas")
print("="*60)

# ============= CARGAR DATOS =============
def cargar_datos_rapido():
    X, y = [], []
    print("\n Cargando imágenes...")
    
    for idx, (nombre, ruta) in enumerate(CATEGORIAS.items()):
        if not os.path.exists(ruta):
            continue
        
        archivos = [f for f in os.listdir(ruta) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:MAX_IMAGES_PER_CLASS]
        
        for archivo in archivos:
            try:
                img = load_img(os.path.join(ruta, archivo), target_size=IMG_SIZE)
                X.append(img_to_array(img))
                y.append(idx)
            except:
                pass
        
        print(f"  ✓ {nombre}: {sum(1 for label in y if label == idx)} imágenes")
    
    return np.array(X), np.array(y)

X, y = cargar_datos_rapido()
print(f"\n✓ Total: {len(X)} imágenes cargadas")

# Preparar datos
X = X.astype('float32') / 255.0
y = to_categorical(y, NUM_CLASSES)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

# ============= MODELO SIMPLE =============
print("\n Creando modelo...")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Parámetros: {model.count_params():,}")

# ============= ENTRENAR =============
print(f"\n Entrenando ({EPOCHS} épocas)...")
os.makedirs('models', exist_ok=True)

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=1
)

# ============= EVALUAR =============
print("\n Evaluando...")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"  Precisión test: {test_acc:.2%}")

y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nReporte:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))

# ============= GUARDAR =============
model.save('models/modelo_rapido.h5')
print("\n Modelo guardado: models/modelo_rapido.h5")

# ============= GRÁFICA =============
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Pérdida')
plt.legend()

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(os.path.join(OUTPUT_DIR, 'entrenamiento_rapido.png'), dpi=150)
print(f" Gráfica guardada: {OUTPUT_DIR}/entrenamiento_rapido.png")

print("\n" + "="*60)
print(" ¡COMPLETADO!")
print("="*60)
print("\nPara usar el modelo:")
print("  from tensorflow import keras")
print("  modelo = keras.models.load_model('models/modelo_rapido.h5')")
print("="*60)
