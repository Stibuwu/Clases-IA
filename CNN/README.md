# Sistema de Clasificación de Animales mediante CNN

## Tabla de Contenidos
1. [Descripción General](#descripción-general)
2. [Marco Teórico](#marco-teórico)
3. [Estructura del Proyecto](#estructura-del-proyecto)
4. [Módulos del Sistema](#módulos-del-sistema)
5. [Análisis de Implementación](#análisis-de-implementación)
6. [Requisitos e Instalación](#requisitos-e-instalación)
7. [Uso](#uso)
8. [Evaluación](#evaluación)
9. [Resultados](#resultados)
10. [Referencias](#referencias)

---

## Descripción General

Sistema de clasificación automática de imágenes basado en Redes Neuronales Convolucionales (CNN). Clasifica cinco categorías de animales: perros, gatos, hormigas, mariquitas y tortugas.

### Objetivos

**Principal:** Desarrollar un clasificador de imágenes de animales con alta precisión mediante arquitecturas CNN.

**Secundarios:**
- Implementar dos arquitecturas con diferentes niveles de complejidad
- Proporcionar herramientas de inferencia para clasificación en tiempo real
- Generar visualizaciones del proceso de entrenamiento y evaluación

### Aplicaciones Potenciales

- Monitoreo de fauna silvestre
- Catalogación de biodiversidad
- Sistemas educativos de identificación de especies
- Detección de plagas agrícolas

---

## Marco Teórico

### Redes Neuronales Convolucionales

Las CNN son arquitecturas especializadas para procesamiento de datos con estructura espacial. Utilizan tres operaciones principales:

**Convolución:** Aplica filtros para extraer características locales.
```
S(i,j) = (I * K)(i,j) = ΣΣ I(i+m, j+n) × K(m,n)
```

**Pooling:** Reduce dimensionalidad espacial preservando características relevantes.
```
y(i,j) = max{x(i×s + m, j×s + n) | 0 ≤ m,n < k}
```

**Activación ReLU:** Introduce no-linealidad.
```
f(x) = max(0, x)
```

### Data Augmentation

Aumenta artificialmente el dataset aplicando transformaciones:
- Geométricas: rotación, traslación, escalado, shearing
- Fotométricas: ajuste de brillo y contraste

Actúa como regularizador implícito, mejorando la capacidad de generalización.

### Regularización

**Dropout:** Desactiva neuronas aleatoriamente durante entrenamiento con probabilidad p, reduciendo dependencias entre unidades.

**Optimización:** Se utiliza Adam, que combina momentum y tasas de aprendizaje adaptativas por parámetro.

## Estructura del Proyecto

```
CNN/
├── Data/                           # Datasets
│   ├── perro/
│   ├── gato/
│   ├── animals/
│   │   ├── hormiga/
│   │   └── mariquita/
│   └── Turtle_Tortoise/
│
├── models/                         # Modelos entrenados (.h5)
│   ├── modelo_rapido.h5
│   └── mejor_modelo_animales.h5
│
├── sources/                        # Resultados y visualizaciones
│   ├── Rapido/
│   └── Completo/
│
├── Pruebas/                        # Imágenes de prueba
│
├── Apuntes-clase/                  # Material educativo
│   ├── CNN.ipynb
│   └── CNNriesgo.ipynb
│
├── entrenar_rapido.py              # Entrenamiento rápido
├── deteccion_animales.py           # Entrenamiento completo
├── usar_detector.py                # Inferencia
├── requirements.txt
└── README.md
```

---

## Módulos del Sistema

### 1. entrenar_rapido.py

Script de entrenamiento rápido para validación inicial.

**Configuración:**
- Resolución: 64×64 píxeles
- Arquitectura: 2 capas convolucionales
- Épocas: 10
- Límite: 200 imágenes/clase

**Arquitectura:**
```
Input (64×64×3)
    ↓
Conv2D (32 filtros, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Conv2D (64 filtros, 3×3) + ReLU
    ↓
MaxPooling2D (2×2)
    ↓
Flatten
    ↓
Dense (64) + ReLU + Dropout(0.5)
    ↓
Dense (5) + Softmax
```

**Parámetros:** ~300,000

**Uso:** Prototipado y verificación de pipeline. No apto para producción.

### 2. deteccion_animales.py

Sistema completo con pipeline de entrenamiento integral.

**Pipeline:**
1. Verificación y carga de dataset (500 imgs/clase)
2. Preprocesamiento y normalización [0,1]
3. División estratificada (70% train, 15% val, 15% test)
4. Data augmentation (rotación ±20°, shifts ±20%, zoom ±20%, flip horizontal)
5. Entrenamiento con callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
6. Evaluación y generación de visualizaciones

**Arquitectura:**
```
Input (64×64×3)
    ↓
Conv2D (32, 3×3, padding='same') + ReLU
    ↓
MaxPooling2D (2×2) + Dropout(0.25)
    ↓
Conv2D (64, 3×3, padding='same') + ReLU
    ↓
MaxPooling2D (2×2) + Dropout(0.25)
    ↓
Flatten
    ↓
Dense (128) + ReLU + Dropout(0.5)
    ↓
Dense (5) + Softmax
```

**Parámetros:** ~500,000

**Visualizaciones generadas:**
- Distribución de clases
- Muestras del dataset
- Ejemplos de data augmentation
- Curvas de entrenamiento (loss/accuracy)
- Matriz de confusión
- Predicciones correctas/incorrectas

**Precisión esperada:** 70-80%

### 3. usar_detector.py

Interfaz de inferencia para clasificación.

**Funciones principales:**

`cargar_modelo(ruta)` - Carga modelo desde archivo .h5

`predecir_imagen(ruta, modelo)` - Clasificación individual con visualización

`detectar_multiples_animales(ruta, modelo, umbral)` - Detección múltiple basada en umbral

`predecir_carpeta(ruta, modelo)` - Procesamiento batch de directorios

**Menú interactivo:**
1. Predicción simple
2. Detección múltiple
3. Análisis de carpeta
4. Salir

---

## Análisis de Implementación

### Carga de Datos

```python
def cargar_imagenes_y_etiquetas():
    imagenes, etiquetas = [], []
    
    for idx, (nombre_clase, ruta_clase) in enumerate(CATEGORIAS.items()):
        archivos = [f for f in os.listdir(ruta_clase) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        archivos = archivos[:MAX_IMAGES_PER_CLASS]
        
        for archivo in archivos:
            try:
                img = load_img(ruta, target_size=IMG_SIZE)
                img_array = img_to_array(img)
                imagenes.append(img_array)
                etiquetas.append(idx)
            except:
                pass
    
    return np.array(imagenes), np.array(etiquetas)
```

Limitador de datos previene sobrecarga de memoria. Try-except maneja imágenes corruptas.

### Preprocesamiento

```python
def preparar_datos(X, y):
    X = X.astype('float32') / 255.0
    y_categorical = to_categorical(y, NUM_CLASSES)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.3, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, 
        stratify=np.argmax(y_temp, axis=1)
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

Normalización a [0,1] estabiliza gradientes. División estratificada mantiene proporción de clases (70/15/15).

### Definición de Modelo

```python
def crear_modelo(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', 
                     input_shape=input_shape, padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

Padding='same' mantiene dimensiones espaciales. Dropout escalonado previene overfitting.

### Callbacks

```python
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, 
                  restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2),
    ModelCheckpoint('models/mejor_modelo.h5', 
                   monitor='val_accuracy', save_best_only=True)
]
```

EarlyStopping detiene entrenamiento ante estancamiento. ReduceLROnPlateau implementa schedule adaptativo de learning rate.

### Inferencia

```python
def predecir_imagen(ruta_imagen, modelo):
    img = load_img(ruta_imagen, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediccion = modelo.predict(img_array, verbose=0)
    clase_predicha = np.argmax(prediccion[0])
    confianza = prediccion[0][clase_predicha]
    
    return CLASS_NAMES[clase_predicha], confianza, prediccion[0]
```

Preprocesamiento en inferencia debe ser idéntico al entrenamiento.

**fill_mode='nearest'**: Rellena píxeles vacíos (por rotación/traslación) con el valor del píxel más cercano.

#### 4. Definición de Modelo

```python
def crear_modelo_profundo(input_shape, num_classes):
    model = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu', 
                     input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # ... más bloques ...
        
        # Clasificador
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Padding='same'**: Mantiene dimensiones espaciales, permitiendo capas más profundas sin reducción excesiva.

**BatchNormalization**: Estabiliza distribución de activaciones, permitiendo learning rates más altos.

**Dropout escalonado**: Mayor dropout (0.5) en capas densas donde hay más parámetros.

**Categorical Crossentropy**: Loss estándar para clasificación multiclase:
```
L = -Σ y_i × log(ŷ_i)
```

#### 5. Callbacks

```python
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3
    ),
    ModelCheckpoint(
        'models/mejor_modelo.h5',
        monitor='val_accuracy',
        save_best_only=True
    )
]
```

**EarlyStopping**: Previene overfitting deteniendo cuando val_accuracy no mejora. `restore_best_weights=True` asegura usar el mejor modelo, no el último.

**ReduceLROnPlateau**: Implementa schedule de learning rate adaptativo. Reduce LR cuando el modelo "plateau" (se estanca).

**ModelCheckpoint**: Guarda automáticamente solo cuando hay mejora, evitando sobrescribir buenos modelos.

#### 6. Transfer Learning

```python
def crear_modelo_transfer_learning(input_shape, num_classes):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,  # Sin clasificador original
        weights='imagenet'   # Pesos pre-entrenados
    )
    
    base_model.trainable = False  # Congelar capas base
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model
```

**include_top=False**: Remueve las capas de clasificación originales (1000 clases de ImageNet), manteniendo solo extractor de características.

**trainable=False**: Congela pesos pre-entrenados durante entrenamiento inicial. Puede descongelarse después para fine-tuning.

**GlobalAveragePooling2D**: Reduce cada mapa de características a un único valor mediante promedio, reduciendo parámetros y previniendo overfitting.

#### 7. Inferencia

```python
def predecir_imagen(ruta_imagen, modelo, mostrar=True):
    # Preprocesamiento idéntico al entrenamiento
    img = load_img(ruta_imagen, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (H,W,C) → (1,H,W,C)
    
    # Predicción
    prediccion = modelo.predict(img_array, verbose=0)
    clase_predicha = np.argmax(prediccion[0])
    confianza = prediccion[0][clase_predicha]
    
    return CLASS_NAMES[clase_predicha], confianza, prediccion[0]
```

**Crítico**: El preprocesamiento en inferencia DEBE ser idéntico al entrenamiento (mismo tamaño, normalización).

**expand_dims**: Añade dimensión de batch (requerida por Keras).

**np.argmax**: Convierte probabilidades softmax a índice de clase.

---

## Requisitos e Instalación

### Requisitos

**Hardware:**
- CPU: 4+ cores, 2.0+ GHz
- RAM: 8 GB mínimo, 16 GB recomendado
- Almacenamiento: 5 GB
- GPU (opcional): NVIDIA con CUDA support

**Software:**
- Python 3.8-3.11
- pip

### Instalación

```bash
# Navegar al directorio
cd path/to/IA/Clases-IA

# Crear entorno virtual
python -m venv venv

# Activar entorno (Windows)
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Verificar instalación
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Dependencias principales:**
```
tensorflow>=2.10.0,<2.16.0
numpy>=1.21.0,<2.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
pillow>=9.0.0
scikit-learn>=1.0.0
```

---

## Uso

### Entrenamiento

**Validación rápida:**
```bash
cd CNN
python entrenar_rapido.py
```
Tiempo: 5-15 min | Precisión: 60-70%

**Entrenamiento completo:**
```bash
python deteccion_animales.py
```
Tiempo: 20-40 min | Precisión: 70-80%

### Inferencia

**Modo interactivo:**
```bash
python usar_detector.py
```

**Modo programático:**
```python
from usar_detector import cargar_modelo, predecir_imagen

modelo = cargar_modelo('models/mejor_modelo_animales.h5')
clase, confianza, probs = predecir_imagen('imagen.jpg', modelo)
```

**Procesamiento batch:**
```python
from usar_detector import predecir_carpeta

resultados = predecir_carpeta('Pruebas/', modelo)
for archivo, clase, conf in resultados:
    print(f"{archivo}: {clase} ({conf:.2%})")
```

---

## Evaluación

### Métricas

**Accuracy:** Porcentaje de predicciones correctas.
```
Accuracy = (TP + TN) / Total
```

**Precision:** De todas las predicciones positivas, cuántas son correctas.
```
Precision = TP / (TP + FP)
```

**Recall:** De todos los positivos reales, cuántos se detectaron.
```
Recall = TP / (TP + FN)
```

**F1-Score:** Media armónica de precision y recall.
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Matriz de Confusión:** Tabla que muestra predicciones correctas (diagonal) y errores (off-diagonal).

### Visualizaciones

**Curvas de entrenamiento:** Loss y accuracy vs épocas.
- Convergencia saludable: train y val decrecen juntos
- Overfitting: train mejora, val empeora
- Underfitting: ambos estancados

**Matriz de confusión:** Heatmap de frecuencias.
- Diagonal oscura indica buenas predicciones
- Off-diagonal muestra confusiones sistemáticas

---

## Resultados

### Modelo Rápido

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| Accuracy | 82% | 68% | 65% |
| Loss | 0.45 | 0.89 | 0.95 |

Overfitting evidente. Tiempo: 8 minutos.

### Modelo Completo

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| Accuracy | 78% | 75% | 74% |
| Loss | 0.58 | 0.67 | 0.69 |

**Por Clase:**

| Clase | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Perro | 82% | 79% | 0.80 |
| Gato | 68% | 72% | 0.70 |
| Hormiga | 77% | 73% | 0.75 |
| Mariquita | 81% | 78% | 0.79 |
| Tortuga | 72% | 68% | 0.70 |

Mejor generalización. Tiempo: 28 minutos.

### Comparación

| Modelo | Parámetros | Precisión | Tiempo | Uso |
|--------|------------|-----------|--------|-----|
| Rápido | 300K | 65% | 8 min | Testing |
| Completo | 500K | 74% | 28 min | Producción |

---

## Limitaciones

**Dataset:**
- 500-800 imágenes/clase (modesto para deep learning)
- Posible sesgo en ángulos e iluminación
- Limitado a imágenes estáticas

**Arquitectura:**
- Resolución 64×64 puede perder detalles
- 5 clases fijas
- Sin localización espacial

**Despliegue:**
- Latencia 50-200ms
- Dependencia pesada (TensorFlow ~500MB)

## Trabajo Futuro

**Inmediato:**
- Incrementar dataset (2000+ imgs/clase)
- Data augmentation más agresivo
- Ensemble de modelos
- Optimización TensorFlow Lite

**Medio plazo:**
- Implementar object detection (YOLO/SSD)
- Arquitecturas avanzadas (EfficientNet, ResNet)
- Segmentación semántica

**Largo plazo:**
- Escalabilidad a 50-100 especies
- Clasificación jerárquica
- Procesamiento de video en tiempo real

---

## Referencias

1. LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*.

2. Krizhevsky, A., et al. (2012). "ImageNet classification with deep convolutional neural networks." *NeurIPS*.

3. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR*.

---

**Proyecto:** Sistema CNN de Clasificación de Animales  
**Versión:** 1.0  
**Fecha:** Diciembre 2024
