# Documentación - Visualizador A* (a-asterisco.py)

> Archivo principal: `a-asterisco.py`

---

## Índice

1. Todo su funcionamiento
2. Procesos clave (clases y funciones)
3. Explicación del algoritmo A*
4. Cómo se muestra la información (UI/visualización)
5. Casos especiales y consideraciones
6. Ubicaciones sugeridas para capturas de pantalla
7. Cómo ejecutar

---

## 1. Todo su funcionamiento

Esta pequeña aplicación visualiza la ejecución del algoritmo A* sobre una cuadrícula interactiva. Flujo de uso:

- Al iniciar el script se abre una ventana redimensionable con un área de juego (grid) y un panel lateral con información.
- El usuario interactúa con el grid mediante el ratón y el teclado:
  - Click izquierdo: colocar elementos (primero inicio, luego fin, luego paredes)
  - Click derecho: borrar una celda (restablecer)
  - ESPACIO: iniciar la ejecución del algoritmo A*
  - C: limpiar el tablero (restablecer inicio/fin/paredes)
  - R: reiniciar la ventana a las dimensiones iniciales
- Durante la ejecución, la aplicación muestra en tiempo real los nodos explorados y colores para distinguir estados (visitado, abierto, camino óptimo, caminos alternativos).
- Al finalizar la búsqueda se reconstruye y muestra el camino óptimo en verde y, de forma secuencial, los caminos alternativos en azul. En el panel lateral se muestra información (costos, nodo actual y resumen final).

Inputs/Outputs:
- Input: interacciones del usuario (clicks, teclas) para definir inicio/fin/paredes.
- Output: visualización en pantalla (grid + panel lateral) y mensajes en consola (si aplica) con resumen.

Contrato mínimo:
- Entradas: posición de clicks dentro del grid; teclas de control.
- Salidas: actualización de `grid` con colores y textos; panel lateral con línea de información.
- Error modes: clicks fuera del grid son ignorados; si no hay camino posible se muestra mensaje y panel con "No se encontró un camino posible".


## 2. Procesos clave (clases y funciones)

- Clase `Nodo` (representación de celda)
  - Atributos importantes:
    - `fila`, `col`: coordenadas lógicas en la matriz
    - `x`, `y`: posición en píxeles (actualizada en `dibujar` según offsets)
    - `ancho`: tamaño en píxeles (ancho del nodo)
    - `color`: color actual que determina el estado visual
    - `vecinos`: lista de nodos accesibles (actualizada por `actualizar_vecinos`)
  - Métodos relevantes:
    - `get_pos()` → devuelve (fila, col)
    - `es_pared()`, `es_inicio()`, `es_fin()` → checks de estado
    - `hacer_inicio()`, `hacer_fin()`, `hacer_pared()`, `hacer_camino()`, `hacer_visitado()`, `hacer_alternativo()`, `hacer_abierto()` → cambian `color`
    - `actualizar_vecinos(grid)` → recalcula vecinos no bloqueados (arriba/abajo/izq/der)
    - `dibujar(ventana)` → dibuja el rectángulo redondeado y, si corresponde, muestra los costos "g" | "h" | "f" centrados en la celda.

- `crear_grid(filas, ancho)`
  - Crea y devuelve una matriz `grid` (lista de listas) de `Nodo` con dimensiones `filas x filas`. Calcula `ancho_nodo` inicial.

- `h(p1, p2)`
  - Heurística: distancia Manhattan entre posiciones de nodos (admisible para movimientos 4-direcciones).

- `algoritmo_a_estrella(dibujar_fn, grid, inicio, fin)`
  - Implementación del A* con:
    - `PriorityQueue` para la lista abierta (frontera), con (f_score, contador, nodo) para evitar colisiones
    - `g_score` y `f_score` dicts por nodo
    - `vino_de` para reconstruir caminos
    - `conjunto_abierto_hash` para checks rápidos de inclusión
    - `caminos_alternativos` colección de caminos parciales detectados durante la expansión
  - La función acepta `dibujar_fn(info)` callable para actualizar la UI con información en tiempo real.
  - Retorna: (encontrado: bool, caminos_alternativos: list, info_final: list, costo_optimo: int)

- `reconstruir_camino(vino_de, actual, inicio, dibujar_fn)`
  - Recorre `vino_de` desde `actual` hasta `inicio`, marcando nodos del camino (verde) y llamando a `dibujar_fn()` con pausas para visualización

- Funciones de UI
  - `dibujar(ventana, grid, filas, ancho_ventana, alto_ventana, panel_lateral, info)` → maneja el cálculo del tamaño de cada nodo, offsets, dibuja nodos y el panel lateral con `mostrar_info_lateral`.
  - `mostrar_info_lateral(ventana, info, ancho_ventana, alto_ventana, panel_lateral)` → pinta el panel lateral y renderiza líneas de texto con la información actual.
  - `obtener_click_pos(pos, filas, offset_x, offset_y, ancho_juego)` → convierte coordenadas del mouse en índice de `grid` teniendo en cuenta offsets y tamaño real del grid; valida clicks fuera del área.

- `main(ventana, ancho_ventana)`
  - Bucle principal: maneja eventos (quit, resize, clicks, keys)
  - Redimensionamiento: ventana resizable; recalcula `panel_lateral`, offsets y tamaño de nodos
  - Al presionar ESPACIO: actualiza `vecinos` para cada nodo y ejecuta `algoritmo_a_estrella`, luego muestra caminos alternativos secuencialmente


## 3. Explicación del algoritmo (A*)

Resumen rápido:
- A* busca el camino de costo mínimo desde `inicio` hasta `fin` en un grafo (aquí, celdas con adyacencia 4-direcciones).
- Para cada nodo `n` se mantiene:
  - `g(n)`: costo real desde `inicio` hasta `n` (en esta implementación cada paso tiene costo 1)
  - `h(n)`: heurística (estimación) del costo de `n` hasta `fin` (distancia Manhattan)
  - `f(n) = g(n) + h(n)`: estimación del costo total pasando por `n`
- La frontera (nodos por explorar) es una cola de prioridad ordenada por `f(n)`. Se extrae el nodo con menor `f` y se expande.
- Si la heurística `h` es admisible (nunca sobreestima), A* es óptimo: el primer momento en que extraemos el `fin` de la frontera implica que encontramos el camino de costo mínimo.

Implementación específica en el código:
- `PriorityQueue` almacena tuplas `(f_score, contador, nodo)`; `contador` evita comparaciones entre nodos si `f_score` empata.
- `g_score` y `f_score` inicializados a `inf` para todos los nodos salvo `inicio`.
- Al evaluar un vecino `v` desde `actual` se calcula `temp_g_score = g_score[actual] + 1`.
  - Si `temp_g_score < g_score[v]`, se ha encontrado un mejor camino hacia `v`, se actualiza `vino_de[v] = actual`, `g_score[v]`, `f_score[v]`.
  - Si `v` no está en la frontera, se añade y se marca visualmente como abierto (amarillo).
- `reconstruir_camino` recorre `vino_de` desde `fin` hasta `inicio` marcando el camino final en verde y mostrando la animación.

Caminos alternativos:
- Mientras se agregan vecinos a la frontera se construyen caminos parciales a partir de `vino_de` y se almacenan en `caminos_alternativos`.
- Al terminar la búsqueda (camino óptimo encontrado), esos caminos alternativos se muestran secuencialmente en azul; el panel lateral muestra la diferencia de costo con respecto al óptimo.

Limitaciones y supuestos:
- Movimiento permitido: 4 direcciones (N, S, E, W). No hay diagonales.
- Costo por movimiento: 1 (constante). Si quieres pesos distintos, habría que cambiar la suma en `temp_g_score`.
- Heurística: Manhattan (admisible y consistente para este grafo simple).


## 4. Cómo se muestra la información (UI/visualización)

- Ventana principal: área de juego a la izquierda, panel lateral a la derecha (20% del ancho, recalculado al redimensionar)
- Panel lateral muestra:
  - Controles
  - Leyenda de colores
  - Información del proceso (lista de líneas): nodo que se está explorando, g/h/f actuales, nodos visitados, resumen final
- Colores y significados:
  - Naranja: Inicio
  - Morado: Final
  - Negro: Pared
  - Amarillo: Nodo abierto (en frontera)
  - Rojo: Nodo visitado / expandido
  - Verde: Camino óptimo encontrado
  - Azul: Caminos alternativos mostrados secuencialmente
- Visualización de costos en celdas:
  - Para celdas en estado abierto/visitado se muestra el texto `g|h|f` centrado en la celda
  - El tamaño de la fuente se calcula dinámicamente en `Nodo.dibujar` según el ancho de la celda (mantenemos un mínimo razonable)
  - Se dibuja un pequeño fondo semi-transparente detrás del texto para mejorar la legibilidad
- Animaciones y timing:
  - Cada actualización importante llama a `dibujar_fn(info)` para refrescar la interfaz
  - Reconstrucción del camino introduce pausas cortas (`pygame.time.delay(50)`) para que el usuario vea el proceso
  - Al mostrar caminos alternativos se usa un loop con `pygame.time.get_ticks()` y procesado de eventos para evitar congelamientos y permitir cerrar o redimensionar la ventana durante la animación


## 5. Casos especiales y consideraciones

- Click fuera del grid: ignorado (la función `obtener_click_pos` valida límites y offsets).
- Redimensionamiento: el grid se recentra y el tamaño de cada celda se recalcula; los nodos actualizan sus `x`, `y`, `ancho` en cada `dibujar`.
- Congelamiento: el código maneja eventos durante animaciones (caminos alternativos) para responder a `QUIT` y `VIDEORESIZE`.
- Rendimiento: con muchas filas y altas resoluciones, la visibilidad de números puede quedar pequeña. Recomendación: usar FILAS = 20–40 según pantalla.
- Extensiones fáciles:
  - Soportar pesos diferentes por celda (almacenar `peso` en `Nodo` y usar `g += peso`)
  - Movimiento diagonal (cambiar vecinos y heurística a Euclidiana o Chebyshev)


## 7. Cómo ejecutar

1. Requisitos: Python 3.8+ y `pygame`.

Instalar pygame (si hace falta):

```powershell
pip install pygame
```

2. Ejecutar app (desde la carpeta `Algoritmo A`):

```powershell
python a-asterisco.py
```

3. Recomendaciones:
- Si los números en las celdas se ven muy pequeños, se puede aumentar `ANCHO_VENTANA` y `ALTO_VENTANA` o reducir `FILAS` (valor por defecto 20), editando `main()` si se busca otro valor por defecto.

---

### Notas finales y próximos pasos sugeridos

- Se pueden añadir más datos al panel lateral (por ejemplo, tablas con `g_score`/`f_score` para algunos nodos seleccionados).
- Añadir opción para exportar la lista de nodos visitados y el camino como JSON/CSV para análisis posterior.
- Añadir control de velocidad de animación (slider o teclas + / -).
