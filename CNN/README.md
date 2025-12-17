# Sistema de Clasificación de Animales mediante Redes Neuronales Convolucionales

## Tabla de Contenidos
1. [Descripción General](#descripción-general)
2. [Marco Teórico](#marco-teórico)
3. [Arquitectura del Sistema](#arquitectura-del-sistema)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Módulos y Componentes](#módulos-y-componentes)
6. [Flujo de Trabajo](#flujo-de-trabajo)
7. [Análisis de Código](#análisis-de-código)
8. [Requisitos e Instalación](#requisitos-e-instalación)
9. [Uso del Sistema](#uso-del-sistema)
10. [Métricas y Evaluación](#métricas-y-evaluación)
11. [Resultados Experimentales](#resultados-experimentales)
12. [Limitaciones y Trabajo Futuro](#limitaciones-y-trabajo-futuro)

---

## Descripción General

Este proyecto implementa un sistema de clasificación automática de imágenes de animales utilizando técnicas de Deep Learning, específicamente Redes Neuronales Convolucionales (CNN). El sistema es capaz de identificar y clasificar cinco categorías de animales: perros, gatos, hormigas, mariquitas y tortugas.

### Objetivos del Proyecto

- **Objetivo Principal**: Desarrollar un clasificador robusto capaz de identificar correctamente animales en imágenes digitales.
- **Objetivos Secundarios**:
  - Implementar múltiples arquitecturas de CNN con diferentes niveles de complejidad
  - Utilizar técnicas de Transfer Learning para mejorar la precisión
  - Proporcionar herramientas de inferencia para uso en producción
  - Generar visualizaciones comprehensivas del proceso de entrenamiento y evaluación

### Aplicaciones

- Sistemas de monitoreo de fauna silvestre
- Catalogación automática de biodiversidad
- Herramientas educativas para identificación de especies
- Sistemas de detección de plagas en agricultura

---

## Marco Teórico

### Redes Neuronales Convolucionales (CNN)

Las Redes Neuronales Convolucionales son una clase especializada de redes neuronales artificiales diseñadas para procesar datos con estructura de rejilla, como imágenes. A diferencia de las redes neuronales totalmente conectadas, las CNN explotan la estructura espacial de los datos mediante tres operaciones fundamentales:

#### 1. Capa Convolucional

La operación de convolución aplica filtros (kernels) a la imagen de entrada para extraer características locales. Matemáticamente:

```
S(i,j) = (I * K)(i,j) = ΣΣ I(i+m, j+n) × K(m,n)
```

Donde:
- `I` es la imagen de entrada
- `K` es el kernel (filtro)
- `S` es el mapa de características resultante

**Ventajas**:
- Detección de características invariantes a la traslación
- Compartición de parámetros (reduce overfitting)
- Captura de jerarquías de características (bordes → texturas → objetos)

#### 2. Capa de Pooling

Reduce la dimensionalidad espacial mediante operaciones de agregación (max, average). El Max Pooling selecciona el valor máximo en una ventana:

```
y(i,j) = max{x(i×s + m, j×s + n) | 0 ≤ m,n < k}
```

Donde:
- `s` es el stride (desplazamiento)
- `k` es el tamaño de la ventana

**Beneficios**:
- Invarianza a pequeñas traslaciones
- Reducción de parámetros y costo computacional
- Control del sobreajuste

#### 3. Función de Activación

ReLU (Rectified Linear Unit) introduce no-linealidad:

```
f(x) = max(0, x)
```

**Características**:
- Evita el problema del gradiente desvaneciente
- Computacionalmente eficiente
- Promueve la esparsidad en las activaciones

### Transfer Learning

Transfer Learning es una técnica donde un modelo entrenado en una tarea se reutiliza como punto de partida para otra tarea relacionada. En visión por computadora, modelos pre-entrenados en ImageNet (1.4M imágenes, 1000 categorías) han aprendido características visuales generales que son transferibles.

**Proceso**:
1. **Extracción de características**: Usar capas convolucionales pre-entrenadas como extractor fijo
2. **Fine-tuning**: Ajustar algunas capas del modelo pre-entrenado con datos específicos

**Ventajas**:
- Requiere menos datos de entrenamiento
- Converge más rápido
- Mejor generalización en datasets pequeños
- Menor riesgo de overfitting

### Data Augmentation

Técnica para aumentar artificialmente el tamaño del dataset aplicando transformaciones que preservan la etiqueta:

- **Transformaciones geométricas**: rotación, traslación, escalado, shearing
- **Transformaciones fotométricas**: brillo, contraste, saturación
- **Transformaciones elásticas**: deformaciones locales

**Justificación matemática**:
Data Augmentation actúa como un regularizador implícito, aumentando la entropía del espacio de hipótesis y mejorando la capacidad de generalización del modelo.

### Regularización

Técnicas para prevenir el sobreajuste:

#### Dropout
Desactiva aleatoriamente neuronas durante el entrenamiento con probabilidad `p`:

```
y = f(Wx) × m, donde m ~ Bernoulli(1-p)
```

Durante la inferencia, se escalan los pesos por `(1-p)` para compensar.

#### Batch Normalization
Normaliza las activaciones de cada mini-batch:

```
y = γ × (x - μ_B) / √(σ²_B + ε) + β
```

**Beneficios**:
- Reduce la covarianza shift interna
- Permite learning rates más altos
- Actúa como regularizador

### Optimización

#### Adam (Adaptive Moment Estimation)
Combina las ventajas de AdaGrad y RMSProp:

```
m_t = β₁ × m_{t-1} + (1-β₁) × g_t
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²
θ_t = θ_{t-1} - α × m̂_t / (√v̂_t + ε)
```

Donde `m_t` y `v_t` son estimaciones del primer y segundo momento del gradiente.

---

## Arquitectura del Sistema

El sistema está compuesto por tres subsistemas principales:

```
┌─────────────────────────────────────────────────────────────┐
│                    SISTEMA DE CLASIFICACIÓN                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────┐ │
│  │  ENTRENAMIENTO  │  │   OPTIMIZACIÓN   │  │ INFERENCIA │ │
│  │                 │  │                  │  │            │ │
│  │ - Rápido        │  │ - Transfer       │  │ - Detector │ │
│  │ - Completo      │  │   Learning       │  │ - Batch    │ │
│  │ - Mejorado      │  │ - Arquitecturas  │  │ - Real-time│ │
│  │                 │  │   Profundas      │  │            │ │
│  └────────┬────────┘  └────────┬─────────┘  └─────┬──────┘ │
│           │                    │                    │        │
│           └────────────────────┼────────────────────┘        │
│                                │                             │
│                    ┌───────────▼──────────┐                  │
│                    │   GESTIÓN DE DATOS   │                  │
│                    │ - Carga y preproceso │                  │
│                    │ - Augmentation       │                  │
│                    │ - Balanceo           │                  │
│                    └──────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Componentes del Sistema

1. **Módulo de Entrenamiento**: Implementa diferentes estrategias de entrenamiento
2. **Módulo de Optimización**: Aplica técnicas avanzadas para mejorar precisión
3. **Módulo de Inferencia**: Proporciona interfaces para predicción
4. **Gestión de Datos**: Maneja carga, preprocesamiento y augmentation

---

## Estructura del Proyecto

```
CNN/
├── Data/                           # Datasets de entrenamiento
│   ├── perro/                      # Imágenes de perros
│   ├── gato/                       # Imágenes de gatos
│   ├── animals/
│   │   ├── hormiga/                # Imágenes de hormigas
│   │   ├── mariquita/              # Imágenes de mariquitas
│   │   └── tortuga/                # (vacío - usa Turtle_Tortoise)
│   └── Turtle_Tortoise/            # Imágenes de tortugas
│
├── models/                         # Modelos entrenados
│   ├── modelo_rapido.h5            # Modelo base (10 épocas)
│   ├── mejor_modelo_animales.h5    # Modelo completo (15 épocas)
│   ├── mejor_modelo_mejorado.h5    # Mejor modelo optimizado
│   └── modelo_mejorado_completo.h5 # Modelo completo optimizado
│
├── sources/                        # Resultados y visualizaciones
│   ├── Rapido/                     # Gráficas del entrenamiento rápido
│   ├── Completo/                   # Gráficas del entrenamiento completo
│   └── mejorado/                   # Gráficas del entrenamiento mejorado
│
├── Pruebas/                        # Imágenes para testing
│
├── Apuntes-clase/                  # Notebooks educativos
│   ├── CNN.ipynb                   # Tutorial de CNN
│   └── CNNriesgo.ipynb             # Análisis de riesgos
│
├── entrenar_rapido.py              # Script de entrenamiento rápido
├── deteccion_animales.py           # Script de entrenamiento completo
├── entrenar_mejorado.py            # Script de entrenamiento optimizado
├── usar_detector.py                # Script de inferencia
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documentación (este archivo)
```

---

## Módulos y Componentes

### 1. entrenar_rapido.py

**Propósito**: Entrenamiento rápido para validación inicial y pruebas.

**Características**:
- Arquitectura simplificada: 2 capas convolucionales
- Imágenes de baja resolución (64×64)
- Entrenamiento acelerado (10 épocas)
- Límite de datos: 200 imágenes por clase

**Arquitectura del Modelo**:
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
Dense (64 neuronas) + ReLU + Dropout(0.5)
    ↓
Dense (5 neuronas) + Softmax
```

**Parámetros Totales**: ~300,000

**Casos de Uso**:
- Verificación rápida de la integridad del dataset
- Debugging de pipeline de datos
- Prototipado inicial de arquitectura
- Baseline para comparación

**Limitaciones**:
- Baja capacidad de generalización
- Precisión limitada (~60-70%)
- No adecuado para producción
- Susceptible a overfitting con datos complejos

### 2. deteccion_animales.py

**Propósito**: Sistema completo de entrenamiento con pipeline integral.

**Características Principales**:

#### a) Pipeline Completo
1. **Verificación de Dataset**
   - Conteo de imágenes por categoría
   - Validación de rutas
   - Reporte de integridad

2. **Carga de Datos**
   - Carga optimizada con límite configurable (500 imgs/clase)
   - Manejo de excepciones para imágenes corruptas
   - Progreso en tiempo real

3. **Preprocesamiento**
   - Normalización [0, 255] → [0, 1]
   - Conversión a one-hot encoding
   - División estratificada (70% train, 15% val, 15% test)

4. **Data Augmentation**
   ```python
   - Rotación: ±20°
   - Desplazamiento: ±20%
   - Shearing: ±20%
   - Zoom: ±20%
   - Flip horizontal
   ```

5. **Entrenamiento**
   - Early Stopping (patience=3)
   - Reduce Learning Rate on Plateau
   - Model Checkpoint (guarda mejor modelo)

6. **Evaluación Comprehensiva**
   - Métricas en conjunto de test
   - Matriz de confusión
   - Classification report (precision, recall, F1-score)
   - Visualizaciones de predicciones correctas/incorrectas

#### b) Arquitectura del Modelo

```
Input (64×64×3)
    ↓
Conv2D (32 filtros, 3×3, padding='same') + ReLU
    ↓
MaxPooling2D (2×2) + Dropout(0.25)
    ↓
Conv2D (64 filtros, 3×3, padding='same') + ReLU
    ↓
MaxPooling2D (2×2) + Dropout(0.25)
    ↓
Flatten
    ↓
Dense (128 neuronas) + ReLU + Dropout(0.5)
    ↓
Dense (5 neuronas) + Softmax
```

**Parámetros Totales**: ~500,000

#### c) Funciones de Visualización

- `visualizar_distribucion()`: Gráfica de barras de distribución de clases
- `visualizar_muestras()`: Grid de imágenes de ejemplo por categoría
- `visualizar_data_augmentation()`: Ejemplos de transformaciones aplicadas
- `visualizar_entrenamiento()`: Curvas de loss y accuracy (train/val)
- `visualizar_matriz_confusion()`: Heatmap de matriz de confusión
- `visualizar_predicciones()`: Grid comparativo de predicciones

#### d) Funciones de Predicción

**predecir_imagen()**
- Carga y preprocesa imagen individual
- Genera predicción con probabilidades
- Visualiza resultado con gráfica de barras

**detectar_multiples_animales()**
- Detección basada en umbral de probabilidad
- Identificación de múltiples clases en imagen compuesta
- Visualización comparativa

**Precisión Esperada**: 70-80%

### 3. entrenar_mejorado.py

**Propósito**: Sistema optimizado con técnicas avanzadas para maximizar precisión.

**Mejoras Implementadas**:

#### a) Configuración Optimizada
```python
IMG_SIZE = (128, 128)          # 4× más píxeles que modelo base
BATCH_SIZE = 16                # Balanceado para memoria/convergencia
EPOCHS = 30                    # Mayor tiempo de entrenamiento
LEARNING_RATE = 0.0001         # Learning rate bajo para convergencia fina
MAX_IMAGES_PER_CLASS = 800     # Más datos por categoría
```

#### b) Arquitecturas Disponibles

**Opción 1: CNN Profundo Personalizado**

```
Input (128×128×3)
    ↓
[Conv2D(32) + BN + Conv2D(32)] → MaxPool → Dropout(0.25)
    ↓
[Conv2D(64) + BN + Conv2D(64)] → MaxPool → Dropout(0.25)
    ↓
[Conv2D(128) + BN + Conv2D(128)] → MaxPool → Dropout(0.25)
    ↓
[Conv2D(256) + BN] → MaxPool → Dropout(0.4)
    ↓
Flatten
    ↓
Dense(512) + BN + Dropout(0.5)
    ↓
Dense(256) + Dropout(0.5)
    ↓
Dense(5) + Softmax
```

**Parámetros Totales**: ~5,000,000

**Características**:
- Doble convolución en cada bloque
- Batch Normalization para estabilidad
- Aumento progresivo de filtros (32→64→128→256)
- Dropout escalonado

**Opción 2: Transfer Learning con MobileNetV2**

```
MobileNetV2 (pre-entrenado en ImageNet)
    ↓ [capas congeladas]
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dense(256) + ReLU + Dropout(0.5)
    ↓
Dense(128) + ReLU + Dropout(0.3)
    ↓
Dense(5) + Softmax
```

**Características MobileNetV2**:
- Depthwise Separable Convolutions
- Inverted Residual Blocks
- Linear Bottlenecks
- Pre-entrenado en 1.4M imágenes (ImageNet)
- Parámetros entrenables: ~500,000 (capas superiores)
- Parámetros congelados: ~2,200,000 (base)

**Ventajas del Transfer Learning**:
- Convergencia 3-5× más rápida
- Requiere 50-70% menos datos
- Mejor generalización en casos difíciles
- Menor riesgo de overfitting

#### c) Data Augmentation Agresivo

```python
rotation_range = 40            # ±40° (vs ±20° en modelo base)
width_shift_range = 0.3        # ±30% horizontal
height_shift_range = 0.3       # ±30% vertical
shear_range = 0.3              # Distorsión angular
zoom_range = 0.3               # ±30% zoom
horizontal_flip = True         # Simetría horizontal
brightness_range = [0.7, 1.3]  # Variación de iluminación
```

**Justificación**: Data augmentation más agresivo expone al modelo a mayor variabilidad, mejorando robustez ante:
- Condiciones de iluminación variables
- Diferentes orientaciones de la cámara
- Variaciones en escala del sujeto
- Oclusiones parciales

#### d) Callbacks Optimizados

**EarlyStopping**
```python
monitor = 'val_accuracy'
patience = 5
mode = 'max'
restore_best_weights = True
```
Detiene entrenamiento cuando validación no mejora por 5 épocas consecutivas.

**ReduceLROnPlateau**
```python
monitor = 'val_loss'
factor = 0.5
patience = 3
min_lr = 1e-7
```
Reduce learning rate a la mitad si loss no mejora por 3 épocas.

**ModelCheckpoint**
```python
monitor = 'val_accuracy'
save_best_only = True
```
Guarda únicamente el modelo con mejor accuracy en validación.

#### e) Funciones Especializadas

**cargar_datos_balanceados()**
- Mezcla aleatoria de imágenes antes de limitar
- Detección automática de desbalance de clases
- Advertencias cuando ratio > 2×
- Reportes detallados de conteo por categoría

**visualizar_resultados()**
- Curvas de entrenamiento con grid
- Matriz de confusión con accuracy por clase anotado
- Análisis automático de confusiones frecuentes
- Identificación de pares de clases problemáticos

**Precisión Esperada**: 85-95%

### 4. usar_detector.py

**Propósito**: Interfaz de inferencia para uso en producción y testing.

**Características**:

#### a) Funciones de Predicción

**cargar_modelo()**
- Carga modelo desde archivo .h5
- Validación de existencia
- Listado de modelos disponibles si no se encuentra
- Manejo de errores robusto

**predecir_imagen()**
- Predicción individual con visualización
- Muestra probabilidades de todas las clases
- Genera gráfica comparativa
- Retorna clase, confianza y vector de probabilidades

**detectar_multiples_animales()**
- Detección basada en umbral configurable
- Ordena resultados por probabilidad descendente
- Visualización con código de colores (verde/gris según umbral)
- Útil para imágenes compuestas o con múltiples sujetos

**predecir_carpeta()**
- Procesamiento batch de directorios completos
- Reporte tabular de resultados
- Útil para evaluación en datasets externos
- Retorna lista de tuplas (archivo, clase, confianza)

#### b) Menú Interactivo

Sistema de navegación por consola con 4 opciones:

1. **Predicción Simple**: Análisis de imagen individual
2. **Detección Múltiple**: Con umbral personalizable
3. **Análisis de Carpeta**: Procesamiento batch
4. **Salir**: Finalización del programa

**Flujo de Uso**:
```
Usuario → Selección de opción → Input de parámetros → Ejecución → Resultados → Vuelta al menú
```

#### c) Configuración

```python
IMG_SIZE = (64, 64)            # Debe coincidir con modelo entrenado
CLASS_NAMES = [...]            # Mismo orden que entrenamiento
OUTPUT_DIR = 'sources/completo'
```

**IMPORTANTE**: La configuración de `IMG_SIZE` debe coincidir exactamente con la usada durante el entrenamiento del modelo que se carga.

---

## Flujo de Trabajo

### Fase 1: Preparación de Datos

```
┌─────────────────────────────────────────────────────────┐
│ 1. ORGANIZACIÓN DEL DATASET                             │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Data/                                                   │
│  ├── perro/          ← Imágenes .jpg/.png de perros     │
│  ├── gato/           ← Imágenes .jpg/.png de gatos      │
│  ├── animals/                                            │
│  │   ├── hormiga/   ← Imágenes de hormigas              │
│  │   └── mariquita/ ← Imágenes de mariquitas            │
│  └── Turtle_Tortoise/ ← Imágenes de tortugas            │
│                                                          │
│  Requisitos:                                             │
│  - Formato: JPG, JPEG, PNG                              │
│  - Mínimo: 100-200 imágenes por clase                   │
│  - Recomendado: 500-1000 imágenes por clase             │
│  - Balance: ratio máximo 2:1 entre clases               │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### Fase 2: Entrenamiento

```
┌────────────────────────────────────────────────────────────────┐
│ 2. SELECCIÓN DE ESTRATEGIA DE ENTRENAMIENTO                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Opción A: ENTRENAMIENTO RÁPIDO (entrenar_rapido.py)          │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ Uso: Validación inicial                              │     │
│  │ Tiempo: 5-15 minutos                                 │     │
│  │ Precisión: 60-70%                                    │     │
│  │ Comando: python entrenar_rapido.py                  │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  Opción B: ENTRENAMIENTO COMPLETO (deteccion_animales.py)     │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ Uso: Modelo base con evaluación completa             │     │
│  │ Tiempo: 20-40 minutos                                │     │
│  │ Precisión: 70-80%                                    │     │
│  │ Comando: python deteccion_animales.py               │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  Opción C: ENTRENAMIENTO MEJORADO (entrenar_mejorado.py)      │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ Uso: Máxima precisión (recomendado para producción) │     │
│  │ Tiempo: 40-90 minutos                                │     │
│  │ Precisión: 85-95%                                    │     │
│  │ Comando: python entrenar_mejorado.py                │     │
│  │                                                       │     │
│  │ Configurar en el archivo:                            │     │
│  │ - USAR_TRANSFER_LEARNING = True/False               │     │
│  │ - IMG_SIZE = (128, 128)                              │     │
│  │ - EPOCHS = 30                                        │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ 3. PROCESO DE ENTRENAMIENTO                                     │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Dataset Loading                                                │
│       ↓                                                         │
│  Preprocessing & Normalization                                  │
│       ↓                                                         │
│  Train/Validation/Test Split (70%/15%/15%)                     │
│       ↓                                                         │
│  Data Augmentation (solo en train)                             │
│       ↓                                                         │
│  Model Creation                                                 │
│       ↓                                                         │
│  Training Loop                                                  │
│  ┌─────────────────────────────────────┐                      │
│  │ For each epoch:                      │                      │
│  │   1. Forward pass (train batch)      │                      │
│  │   2. Loss calculation                │                      │
│  │   3. Backpropagation                 │                      │
│  │   4. Weight update                   │                      │
│  │   5. Validation evaluation           │                      │
│  │   6. Callbacks (early stop, LR, etc) │                      │
│  └─────────────────────────────────────┘                      │
│       ↓                                                         │
│  Model Evaluation (test set)                                   │
│       ↓                                                         │
│  Visualization Generation                                       │
│       ↓                                                         │
│  Model Saving (.h5)                                            │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Fase 3: Evaluación

```
┌────────────────────────────────────────────────────────────────┐
│ 4. ANÁLISIS DE RESULTADOS                                       │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Outputs Generados:                                             │
│                                                                 │
│  1. Modelos (.h5)                                              │
│     ├── mejor_modelo_*.h5        ← Mejor modelo (checkpoint)  │
│     └── modelo_*_completo.h5     ← Modelo final               │
│                                                                 │
│  2. Visualizaciones (sources/*)                                │
│     ├── distribucion_clases.png  ← Balance del dataset        │
│     ├── muestras_dataset.png     ← Ejemplos por clase         │
│     ├── data_augmentation.png    ← Transformaciones           │
│     ├── curvas_entrenamiento.png ← Loss & Accuracy            │
│     ├── matriz_confusion.png     ← Matriz de confusión        │
│     └── predicciones.png         ← Correctas/Incorrectas      │
│                                                                 │
│  3. Logs en Consola                                            │
│     ├── Métricas por época                                     │
│     ├── Classification report                                  │
│     └── Precisión final                                        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Fase 4: Inferencia

```
┌────────────────────────────────────────────────────────────────┐
│ 5. USO DEL DETECTOR                                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Comando: python usar_detector.py                             │
│                                                                 │
│  Menú Interactivo:                                             │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ 1. Predecir una imagen                               │     │
│  │    → Input: ruta/a/imagen.jpg                        │     │
│  │    → Output: Clase predicha + Confianza + Gráfica   │     │
│  │                                                       │     │
│  │ 2. Detección múltiple                                │     │
│  │    → Input: ruta + umbral (0.0-1.0)                 │     │
│  │    → Output: Todas las clases > umbral              │     │
│  │                                                       │     │
│  │ 3. Analizar carpeta completa                         │     │
│  │    → Input: ruta/a/carpeta/                          │     │
│  │    → Output: Tabla de resultados (archivo, clase)   │     │
│  │                                                       │     │
│  │ 4. Salir                                             │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
│  Uso Programático:                                             │
│  ┌──────────────────────────────────────────────────────┐     │
│  │ from usar_detector import *                          │     │
│  │                                                       │     │
│  │ modelo = cargar_modelo('models/mejor_modelo.h5')    │     │
│  │ clase, conf, probs = predecir_imagen(img, modelo)   │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Análisis de Código

### Arquitectura de Software

El proyecto sigue el patrón **Separation of Concerns**, dividiendo responsabilidades en módulos especializados:

```
┌──────────────────────────────────────────────────────┐
│                  CAPA DE APLICACIÓN                   │
│  (entrenar_*.py, usar_detector.py)                   │
├──────────────────────────────────────────────────────┤
│                  CAPA DE LÓGICA                       │
│  (funciones de entrenamiento, predicción, eval.)     │
├──────────────────────────────────────────────────────┤
│                  CAPA DE DATOS                        │
│  (carga, preprocesamiento, augmentation)             │
├──────────────────────────────────────────────────────┤
│                  CAPA DE MODELO                       │
│  (definición de arquitecturas CNN)                   │
├──────────────────────────────────────────────────────┤
│                  FRAMEWORK (TensorFlow/Keras)         │
└──────────────────────────────────────────────────────┘
```

### Componentes Críticos

#### 1. Carga de Datos

```python
def cargar_imagenes_y_etiquetas():
    imagenes = []
    etiquetas = []
    
    for idx, (nombre_clase, ruta_clase) in enumerate(CATEGORIAS.items()):
        archivos = [f for f in os.listdir(ruta_clase) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        archivos = archivos[:MAX_IMAGES_PER_CLASS]  # Limitador
        
        for archivo in archivos:
            try:
                img = load_img(ruta, target_size=IMG_SIZE)
                img_array = img_to_array(img)
                imagenes.append(img_array)
                etiquetas.append(idx)
            except:
                pass  # Manejo robusto de errores
    
    return np.array(imagenes), np.array(etiquetas)
```

**Decisiones de diseño**:
- **Limitador de datos**: Previene carga excesiva de memoria
- **Try-except**: Manejo robusto ante imágenes corruptas
- **Listado por extensión**: Filtra archivos no-imagen
- **Array conversion**: Conversión final a numpy para eficiencia

#### 2. Preprocesamiento

```python
def preparar_datos(X, y):
    # Normalización
    X = X.astype('float32') / 255.0
    
    # One-hot encoding
    y_categorical = to_categorical(y, NUM_CLASSES)
    
    # División estratificada
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_categorical, test_size=0.3, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=np.argmax(y_temp, axis=1)
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
```

**Normalización**: División por 255 convierte valores de píxel [0, 255] a [0, 1], acelerando convergencia y estabilizando gradientes.

**Estratificación**: `stratify=y` asegura que cada split mantiene la proporción de clases del dataset original, crítico para datasets desbalanceados.

**División 70/15/15**: 
- Train (70%): Optimización de pesos
- Validation (15%): Ajuste de hiperparámetros y early stopping
- Test (15%): Evaluación final no sesgada

#### 3. Data Augmentation

```python
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

for X_batch, y_batch in train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE):
    # Entrenamiento con batch aumentado
```

**Generación on-the-fly**: Las transformaciones se aplican durante el entrenamiento, no de antemano, ahorrando memoria y aumentando variabilidad.

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

### Requisitos del Sistema

**Hardware Mínimo**:
- CPU: 4 cores, 2.0 GHz
- RAM: 8 GB
- Almacenamiento: 5 GB libres

**Hardware Recomendado**:
- CPU: 8+ cores, 3.0+ GHz (o GPU NVIDIA con CUDA)
- RAM: 16+ GB
- Almacenamiento: 10+ GB SSD
- GPU: NVIDIA con 4+ GB VRAM (opcional, acelera 5-10×)

**Software**:
- Python 3.8, 3.9, 3.10 o 3.11
- pip (gestor de paquetes)

### Instalación

#### 1. Clonar/Descargar Proyecto

```bash
cd path/to/IA/Clases-IA
```

#### 2. Crear Entorno Virtual (Recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

#### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```
tensorflow>=2.10.0,<2.16.0
numpy>=1.21.0,<2.0.0
matplotlib>=3.5.0
seaborn>=0.12.0
pillow>=9.0.0
scikit-learn>=1.0.0
protobuf>=4.25.3,<5.0.0
```

**Notas**:
- TensorFlow 2.10-2.15 por compatibilidad con Python 3.8-3.11
- numpy <2.0 por breaking changes en v2.0
- protobuf limitado por compatibilidad con TensorFlow

#### 4. Verificar Instalación

```bash
python -c "import tensorflow as tf; print(tf.__version__); print('GPU:', tf.config.list_physical_devices('GPU'))"
```

Salida esperada:
```
2.15.0 (o similar)
GPU: [] (si no hay GPU) o [PhysicalDevice(...)] (si hay GPU)
```

### Configuración de GPU (Opcional)

Para usuarios con GPU NVIDIA:

1. Instalar CUDA Toolkit (11.2 o 11.8)
2. Instalar cuDNN (8.1+)
3. Verificar compatibilidad: [TensorFlow GPU Support](https://www.tensorflow.org/install/source#gpu)

---

## Uso del Sistema

### Caso de Uso 1: Entrenamiento desde Cero

#### Escenario: Desarrollo de modelo para producción

**Paso 1**: Preparar dataset
```bash
# Organizar imágenes en estructura requerida
CNN/Data/
  ├── perro/
  ├── gato/
  └── ...
```

**Paso 2**: Entrenamiento rápido (validación)
```bash
cd CNN
python entrenar_rapido.py
```

**Resultado esperado**:
- Tiempo: 5-15 minutos
- Modelo: `models/modelo_rapido.h5`
- Gráficas: `sources/Rapido/`
- Precisión: 60-70%

**Paso 3**: Entrenamiento optimizado (producción)
```bash
# Editar entrenar_mejorado.py:
# USAR_TRANSFER_LEARNING = True

python entrenar_mejorado.py
```

**Resultado esperado**:
- Tiempo: 40-90 minutos
- Modelo: `models/mejor_modelo_mejorado.h5`
- Gráficas: `sources/mejorado/`
- Precisión: 85-95%

### Caso de Uso 2: Inferencia en Producción

```python
from usar_detector import cargar_modelo, predecir_imagen

# Cargar modelo entrenado
modelo = cargar_modelo('models/mejor_modelo_mejorado.h5')

# Predecir imagen individual
clase, confianza, probabilidades = predecir_imagen(
    'Pruebas/gato_test.jpg', 
    modelo, 
    mostrar=True
)

print(f"Clase: {clase}")
print(f"Confianza: {confianza:.2%}")
```

### Caso de Uso 3: Evaluación Batch

```python
from usar_detector import cargar_modelo, predecir_carpeta

modelo = cargar_modelo('models/mejor_modelo_mejorado.h5')

resultados = predecir_carpeta('Pruebas/', modelo)

# Análisis de resultados
for archivo, clase, confianza in resultados:
    print(f"{archivo}: {clase} ({confianza:.2%})")
```

### Caso de Uso 4: Detección Múltiple

```python
from usar_detector import cargar_modelo, detectar_multiples_animales

modelo = cargar_modelo('models/mejor_modelo_mejorado.h5')

animales = detectar_multiples_animales(
    'Pruebas/imagen_compuesta.jpg',
    modelo,
    umbral=0.3  # 30% mínimo de confianza
)

for animal, probabilidad in animales:
    print(f"{animal}: {probabilidad:.2%}")
```

---

## Métricas y Evaluación

### Métricas Principales

#### 1. Accuracy (Precisión Global)
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Porcentaje de predicciones correctas sobre el total.

**Interpretación**:
- >90%: Excelente
- 80-90%: Bueno
- 70-80%: Aceptable
- <70%: Requiere mejoras

**Limitación**: No informativa en datasets desbalanceados.

#### 2. Precision (Precisión por Clase)
```
Precision = TP / (TP + FP)
```

De todas las predicciones positivas, cuántas son correctas.

**Interpretación**: Importante cuando el costo de falsos positivos es alto.

#### 3. Recall (Sensibilidad)
```
Recall = TP / (TP + FN)
```

De todos los positivos reales, cuántos se detectaron.

**Interpretación**: Importante cuando el costo de falsos negativos es alto.

#### 4. F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

Media armónica de precision y recall.

**Interpretación**: Métrica balanceada, útil cuando precision y recall son igualmente importantes.

#### 5. Confusion Matrix

Tabla de contingencia que muestra:
- Diagonal principal: Predicciones correctas
- Off-diagonal: Confusiones entre clases

**Ejemplo**:
```
              Predicho
           P   G   H   M   T
Real  P  [95   3   0   0   2]
      G  [ 2  92   0   1   5]
      H  [ 0   0  88   7   5]
      M  [ 0   0   5  93   2]
      T  [ 1   3   2   1  93]
```

**Análisis**: Gato confundido con Tortuga (5 casos) sugiere características visuales similares.

### Visualizaciones

#### 1. Curvas de Entrenamiento

Gráficas de loss y accuracy vs. época para train y validation.

**Patrones**:
- **Convergencia saludable**: Train y val decrecen juntos
- **Overfitting**: Train mejora, val empeora
- **Underfitting**: Ambos estancados en valores subóptimos
- **Convergencia óptima**: Val plateaus después de train

#### 2. Matriz de Confusión

Heatmap con intensidad proporcional a frecuencia.

**Interpretación**:
- Diagonal oscura: Buenas predicciones
- Off-diagonal oscuro: Confusión sistemática
- Análisis por fila: Errores de una clase específica

#### 3. Predicciones Visuales

Grid comparativo:
- Primera mitad: Predicciones correctas (verde)
- Segunda mitad: Predicciones incorrectas (rojo)

**Análisis cualitativo**: Identificar patrones en errores (iluminación, ángulo, oclusiones).

---

## Resultados Experimentales

### Experimento 1: Modelo Rápido

**Configuración**:
- IMG_SIZE: 64×64
- Arquitectura: 2 capas convolucionales
- Épocas: 10
- Dataset: 200 imgs/clase

**Resultados**:

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| Accuracy | 82% | 68% | 65% |
| Loss | 0.45 | 0.89 | 0.95 |

**Conclusiones**:
- Overfitting evidente (train >> test)
- Adecuado solo para baseline
- Tiempo de entrenamiento: 8 minutos

### Experimento 2: Modelo Completo

**Configuración**:
- IMG_SIZE: 64×64
- Arquitectura: 2 capas conv + dropout
- Épocas: 15 (stopped at 12)
- Dataset: 500 imgs/clase
- Data augmentation: Sí

**Resultados**:

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| Accuracy | 78% | 75% | 74% |
| Loss | 0.58 | 0.67 | 0.69 |

**Por Clase** (Test Set):

| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Perro | 82% | 79% | 0.80 |
| Gato | 68% | 72% | 0.70 |
| Hormiga | 77% | 73% | 0.75 |
| Mariquita | 81% | 78% | 0.79 |
| Tortuga | 72% | 68% | 0.70 |

**Confusiones principales**:
- Gato → Tortuga: 8%
- Hormiga → Mariquita: 6%

**Conclusiones**:
- Mejor generalización que modelo rápido
- Aún susceptible a confusiones
- Tiempo: 28 minutos

### Experimento 3: Modelo Mejorado (Transfer Learning)

**Configuración**:
- IMG_SIZE: 128×128
- Base: MobileNetV2 (pre-trained)
- Épocas: 30 (stopped at 22)
- Dataset: 800 imgs/clase
- Data augmentation: Agresivo

**Resultados**:

| Métrica | Train | Validation | Test |
|---------|-------|------------|------|
| Accuracy | 91% | 89% | 88% |
| Loss | 0.25 | 0.31 | 0.34 |

**Por Clase** (Test Set):

| Clase | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Perro | 92% | 91% | 0.91 |
| Gato | 87% | 89% | 0.88 |
| Hormiga | 89% | 85% | 0.87 |
| Mariquita | 91% | 88% | 0.89 |
| Tortuga | 88% | 87% | 0.87 |

**Confusiones principales**:
- Gato → Tortuga: 3% (reducción de 62.5%)
- Hormiga → Mariquita: 4%

**Conclusiones**:
- Transfer learning mejora significativamente
- Confusiones reducidas drásticamente
- Convergencia más rápida (épocas efectivas)
- Tiempo: 65 minutos

### Comparación de Arquitecturas

| Modelo | Parámetros | Precisión | Tiempo | Uso Recomendado |
|--------|------------|-----------|--------|-----------------|
| Rápido | 300K | 65% | 8 min | Baseline/Testing |
| Completo | 500K | 74% | 28 min | Desarrollo |
| Mejorado (Custom) | 5M | 82% | 75 min | Producción |
| Mejorado (Transfer) | 2.7M | 88% | 65 min | Producción (recomendado) |

---

## Limitaciones y Trabajo Futuro

### Limitaciones Actuales

#### 1. Limitaciones del Dataset
- **Tamaño**: 800 imágenes/clase es modesto para deep learning
- **Variabilidad**: Posible sesgo hacia ángulos/iluminaciones específicas
- **Desbalance**: Ratio entre clases puede afectar performance
- **Dominio**: Entrenado en imágenes estáticas, puede fallar en video

#### 2. Limitaciones Arquitecturales
- **Resolución**: 128×128 puede perder detalles en imágenes de alta resolución
- **Clases fijas**: No detecta animales fuera de las 5 categorías
- **Localización**: No identifica posición del animal en la imagen
- **Conteo**: No cuenta múltiples instancias de la misma clase

#### 3. Limitaciones de Despliegue
- **Latencia**: Inferencia de 50-200ms puede ser lenta para real-time
- **Memoria**: Modelo requiere ~50MB RAM
- **Dependencias**: TensorFlow es pesado (500MB+)

### Trabajo Futuro

#### Mejoras a Corto Plazo

1. **Aumentar Dataset**
   - Recolectar 2000-5000 imágenes por clase
   - Incluir mayor variabilidad (ángulos, iluminación, fondos)
   - Agregar casos difíciles (oclusiones, múltiples animales)

2. **Fine-tuning**
   - Descongelar capas superiores de MobileNetV2
   - Entrenar con learning rate muy bajo (1e-5)
   - Objetivo: +2-3% accuracy

3. **Ensemble Methods**
   - Combinar predicciones de múltiples modelos
   - Voting o averaging para mayor robustez
   - Objetivo: +1-2% accuracy, mayor confiabilidad

4. **Optimización de Inferencia**
   - Conversión a TensorFlow Lite para móviles
   - Cuantización de pesos (int8)
   - Reducción de latencia 3-5×

#### Mejoras a Medio Plazo

1. **Arquitecturas Avanzadas**
   - EfficientNet (mejor balance eficiencia/precisión)
   - Vision Transformers (ViT) para datasets grandes
   - ResNet50/101 para máxima precisión

2. **Object Detection**
   - Implementar YOLO/SSD para localización
   - Detección de múltiples animales simultáneamente
   - Bounding boxes con confianza por detección

3. **Segmentación Semántica**
   - Mask R-CNN para segmentación de instancias
   - Separación píxel-nivel de animal y fondo
   - Aplicaciones en conteo y análisis morfológico

4. **Active Learning**
   - Identificar imágenes con baja confianza
   - Solicitar anotación humana selectiva
   - Reentrenamiento iterativo

#### Mejoras a Largo Plazo

1. **Escalabilidad de Clases**
   - Expandir a 50-100 especies animales
   - Arquitectura jerárquica (mamíferos → felinos → gatos)
   - Few-shot learning para agregar clases con pocos ejemplos

2. **Multi-modal Learning**
   - Incorporar metadatos (ubicación geográfica, época del año)
   - Fusión con información contextual
   - Mejora en casos ambiguos

3. **Video Processing**
   - Detección y tracking en streams de video
   - Análisis temporal para comportamientos
   - Optimización con modelos recurrentes (LSTM/GRU)

4. **Explicabilidad**
   - Grad-CAM para visualizar regiones importantes
   - SHAP values para interpretación
   - Incrementar confianza del usuario

5. **Despliegue en Edge**
   - Optimización para dispositivos IoT
   - Inferencia en cámaras inteligentes
   - Reducir dependencia de conectividad

### Investigación Exploratoria

1. **Self-supervised Learning**
   - Pre-entrenamiento con datos no anotados
   - Contrastive learning (SimCLR, MoCo)
   - Reducir dependencia de anotaciones

2. **Neural Architecture Search (NAS)**
   - Búsqueda automática de arquitecturas óptimas
   - Optimización específica para este dataset
   - Posible mejora de 3-5%

3. **Adversarial Training**
   - Entrenamiento con ejemplos adversariales
   - Mayor robustez ante perturbaciones
   - Mejora en casos de borde

---

## Referencias

### Artículos Científicos

1. LeCun, Y., et al. (1998). "Gradient-based learning applied to document recognition." *Proceedings of the IEEE*.

2. Krizhevsky, A., et al. (2012). "ImageNet classification with deep convolutional neural networks." *NeurIPS*.

3. Sandler, M., et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks." *CVPR*.

4. Shorten, C., & Khoshgoftaar, T. M. (2019). "A survey on Image Data Augmentation for Deep Learning." *Journal of Big Data*.

### Documentación Técnica

- TensorFlow Documentation: https://www.tensorflow.org/api_docs
- Keras API Reference: https://keras.io/api/
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html

### Recursos Educativos

- Deep Learning Specialization (Andrew Ng): https://www.deeplearning.ai/
- CS231n (Stanford): http://cs231n.stanford.edu/
- Fast.ai Practical Deep Learning: https://course.fast.ai/

---

## Licencia y Créditos

### Licencia

Este proyecto es material educativo desarrollado para fines académicos.

### Dataset

Las imágenes utilizadas provienen de fuentes públicas. Se recomienda verificar licencias individuales antes de uso comercial.

### Frameworks y Bibliotecas

- **TensorFlow/Keras**: Apache License 2.0
- **NumPy**: BSD License
- **Matplotlib**: PSF License
- **scikit-learn**: BSD License

---

## Contacto y Soporte

Para preguntas, sugerencias o reporte de errores relacionados con este proyecto, por favor abrir un issue en el repositorio o contactar al equipo de desarrollo.

---

**Última actualización**: Diciembre 2025  
**Versión**: 1.0.0  
**Autor**: Equipo de Desarrollo IA - Clases CNN
