import pygame
from queue import PriorityQueue
import math

pygame.init()

info_pantalla = pygame.display.Info()
ANCHO_PANTALLA = info_pantalla.current_w
ALTO_PANTALLA = info_pantalla.current_h

ANCHO_VENTANA = min(int(ANCHO_PANTALLA * 0.8), 1200)  
ALTO_VENTANA = min(int(ALTO_PANTALLA * 0.8), 900)    
PANEL_LATERAL = int(ANCHO_VENTANA * 0.2)  

VENTANA = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA), pygame.RESIZABLE)
pygame.display.set_caption("Algoritmo A* - Búsqueda de Caminos")

FUENTE = pygame.font.Font(None, 24)
FUENTE_PEQUEÑA = pygame.font.Font(None, 20)

BLANCO = (248, 248, 255)  
NEGRO = (28, 28, 28)      
GRIS = (169, 169, 169)    
VERDE = (34, 139, 34)     
ROJO = (220, 20, 60)      
NARANJA = (255, 140, 0)   
PURPURA = (147, 112, 219) 
AZUL = (65, 105, 225)     
AMARILLO = (255, 215, 0)  
GRIS_CLARO = (211, 211, 211)  

class Nodo:
    def __init__(self, fila, col, ancho, total_filas):
        self.fila = fila
        self.col = col
        self.x = fila * ancho
        self.y = col * ancho
        self.color = BLANCO
        self.ancho = ancho
        self.total_filas = total_filas

    def get_pos(self):
        return self.fila, self.col

    def es_pared(self):
        return self.color == NEGRO

    def es_inicio(self):
        return self.color == NARANJA

    def es_fin(self):
        return self.color == PURPURA

    def restablecer(self):
        self.color = BLANCO

    def hacer_inicio(self):
        self.color = NARANJA

    def hacer_pared(self):
        self.color = NEGRO

    def hacer_fin(self):
        self.color = PURPURA

    def hacer_camino(self):
        self.color = VERDE

    def hacer_visitado(self):
        self.color = ROJO

    def hacer_alternativo(self):
        self.color = AZUL

    def hacer_abierto(self):
        self.color = AMARILLO

    def dibujar(self, ventana):
        # Dibuja el nodo con bordes redondeados
        pygame.draw.rect(ventana, self.color, (self.x, self.y, self.ancho, self.ancho), border_radius=3)
        
        # Si el nodo está siendo evaluado (amarillo o rojo), mostrar los costos
        if self.color in [AMARILLO, ROJO] and hasattr(self, 'g_cost') and hasattr(self, 'h_cost'):
            # Calcular el tamaño de fuente basado en el tamaño del nodo
            tamano_fuente = max(int(self.ancho / 4), 10) 
            fuente = pygame.font.Font(None, tamano_fuente)
            
            # Crear textos
            f_valor = self.g_cost + self.h_cost
            texto_valores = fuente.render(f'{self.g_cost}|{self.h_cost}|{f_valor}', True, NEGRO)
            
            # Centrar el texto en el nodo
            texto_rect = texto_valores.get_rect()
            texto_rect.center = (self.x + self.ancho/2, self.y + self.ancho/2)
            
            # Dibujar un fondo semi-transparente para mejorar la legibilidad
            s = pygame.Surface((texto_rect.width + 4, texto_rect.height + 4))
            s.set_alpha(128)
            s.fill(BLANCO)
            ventana.blit(s, (texto_rect.x - 2, texto_rect.y - 2))
            
            # Dibujar el texto
            ventana.blit(texto_valores, texto_rect)

    def actualizar_vecinos(self, grid):
        self.vecinos = []
        # Verificar vecino abajo
        if self.fila < self.total_filas - 1 and not grid[self.fila + 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila + 1][self.col])
        # Verificar vecino arriba
        if self.fila > 0 and not grid[self.fila - 1][self.col].es_pared():
            self.vecinos.append(grid[self.fila - 1][self.col])
        # Verificar vecino derecha
        if self.col < self.total_filas - 1 and not grid[self.fila][self.col + 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col + 1])
        # Verificar vecino izquierda
        if self.col > 0 and not grid[self.fila][self.col - 1].es_pared():
            self.vecinos.append(grid[self.fila][self.col - 1])

def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruir_camino(vino_de, actual, inicio, dibujar_fn):
    camino = []
    pasos_reconstruccion = 0
    while actual in vino_de:
        actual = vino_de[actual]
        if actual != inicio:
            actual.hacer_camino()
            pasos_reconstruccion += 1
        camino.append(actual)
        dibujar_fn()
        pygame.time.delay(50)  # Pequeña pausa para visualizar la reconstrucción
    return camino

def algoritmo_a_estrella(dibujar_fn, grid, inicio, fin):
    cuenta = 0
    conjunto_abierto = PriorityQueue()
    conjunto_abierto.put((0, cuenta, inicio))
    vino_de = {}
    g_score = {nodo: float("inf") for fila in grid for nodo in fila}
    g_score[inicio] = 0
    f_score = {nodo: float("inf") for fila in grid for nodo in fila}
    f_score[inicio] = h(inicio.get_pos(), fin.get_pos())
    
    # Info para mostrar en la interfaz
    info = []
    pasos = 0

    conjunto_abierto_hash = {inicio}
    caminos_alternativos = []

    while not conjunto_abierto.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        actual = conjunto_abierto.get()[2]
        conjunto_abierto_hash.remove(actual)

        if actual == fin:
            info.append(f"Reconstruyendo camino óptimo...")
            camino = reconstruir_camino(vino_de, actual, inicio, lambda: dibujar_fn(info))
            fin.hacer_fin()
            inicio.hacer_inicio()
            
            # Calcular el costo total del camino óptimo
            costo_total = g_score[fin]
            info = [
                f"¡Camino encontrado!",
                f"Longitud: {len(camino)} nodos",
                f"Costo total: {costo_total}",
                f"Nodos explorados: {pasos}"
            ]
            
            dibujar_fn(info)
            return True, caminos_alternativos, info, g_score[fin]

        pasos += 1
        for vecino in actual.vecinos:
            temp_g_score = g_score[actual] + 1

            if temp_g_score < g_score[vecino]:
                vino_de[vecino] = actual
                g_score[vecino] = temp_g_score
                h_cost = h(vecino.get_pos(), fin.get_pos())
                f_score[vecino] = temp_g_score + h_cost
                
                # Guardar los costos en el nodo para visualización
                vecino.g_cost = temp_g_score
                vecino.h_cost = h_cost
                
                if vecino not in conjunto_abierto_hash:
                    cuenta += 1
                    conjunto_abierto.put((f_score[vecino], cuenta, vecino))
                    conjunto_abierto_hash.add(vecino)
                    vecino.hacer_abierto()
                    
                    # Actualizar información
                    info = [
                        f"Explorando nodo: ({vecino.fila}, {vecino.col})",
                        f"g(n) = {temp_g_score}",
                        f"h(n) = {h_cost}",
                        f"f(n) = {f_score[vecino]}",
                        f"Nodos visitados: {pasos}"
                    ]
                    dibujar_fn(info)
                    
                    # Guardar camino alternativo
                    if vecino != fin:
                        camino_alt = []
                        temp = vecino
                        while temp in vino_de:
                            camino_alt.append(temp)
                            temp = vino_de[temp]
                        caminos_alternativos.append(camino_alt)

        if actual != inicio:
            actual.hacer_visitado()
            info = [
                f"Nodo actual: ({actual.fila}, {actual.col})",
                f"Nodos explorados: {pasos}",
                "Buscando camino..."
            ]
            dibujar_fn(info)

    info = ["No se encontró un camino posible"]
    dibujar_fn(info)
    return False, [], info, 0

def crear_grid(filas, ancho):
    grid = []
    ancho_nodo = ancho // filas
    for i in range(filas):
        grid.append([])
        for j in range(filas):
            nodo = Nodo(i, j, ancho_nodo, filas)
            grid[i].append(nodo)
    return grid

def dibujar_grid(ventana, filas, ancho):
    ancho_nodo = ancho // filas
    for i in range(filas):
        pygame.draw.line(ventana, GRIS, (0, i * ancho_nodo), (ancho, i * ancho_nodo))
        for j in range(filas):
            pygame.draw.line(ventana, GRIS, (j * ancho_nodo, 0), (j * ancho_nodo, ancho))

def mostrar_info_lateral(ventana, info, ancho_ventana, alto_ventana, panel_lateral):
    # Área lateral para información
    x_panel = ancho_ventana - panel_lateral
    pygame.draw.rect(ventana, GRIS_CLARO, (x_panel, 0, panel_lateral, alto_ventana))
    
    # Título
    titulo = FUENTE.render("Información", True, NEGRO)
    ventana.blit(titulo, (x_panel + 10, 10))
    
    # Mostrar información línea por línea
    y = 50
    for linea in info:
        texto = FUENTE_PEQUEÑA.render(linea, True, NEGRO)
        ventana.blit(texto, (x_panel + 10, y))
        y += 25

def dibujar(ventana, grid, filas, ancho_ventana, alto_ventana, panel_lateral, info=None):
    ventana.fill(BLANCO)
    
    # Calcular el tamaño del área de juego (excluyendo el panel lateral)
    ancho_juego = ancho_ventana - panel_lateral
    alto_juego = alto_ventana
    ancho_nodo = min(ancho_juego, alto_juego) // filas
    
    # Centrar el grid
    offset_x = (ancho_juego - (ancho_nodo * filas)) // 2
    offset_y = (alto_juego - (ancho_nodo * filas)) // 2
    
    # Actualizar las posiciones de los nodos
    for i, fila in enumerate(grid):
        for j, nodo in enumerate(fila):
            nodo.x = offset_x + (j * ancho_nodo)
            nodo.y = offset_y + (i * ancho_nodo)
            nodo.ancho = ancho_nodo
            nodo.dibujar(ventana)

    # Dibujar el grid
    for i in range(filas + 1):
        pygame.draw.line(ventana, GRIS, 
                        (offset_x + i * ancho_nodo, offset_y),
                        (offset_x + i * ancho_nodo, offset_y + filas * ancho_nodo))
        pygame.draw.line(ventana, GRIS,
                        (offset_x, offset_y + i * ancho_nodo),
                        (offset_x + filas * ancho_nodo, offset_y + i * ancho_nodo))
    
    # Mostrar leyenda
    leyenda = [
        "Controles:",
        "Click Izq: Colocar",
        "Click Der: Borrar",
        "ESPACIO: Iniciar",
        "C: Limpiar todo",
        "R: Reiniciar ventana",
        "",
        "Colores:",
        "Naranja: Inicio",
        "Morado: Final",
        "Negro: Pared",
        "Rojo: Visitado",
        "Amarillo: En考",
        "Verde: Óptimo",
        "Azul: Alternativo"
    ]
    
    if info:
        leyenda.extend(["", "Proceso:"] + info)
    
    mostrar_info_lateral(ventana, leyenda, ancho_ventana, alto_ventana, panel_lateral)
    pygame.display.update()

def obtener_click_pos(pos, filas, offset_x, offset_y, ancho_juego):
    x, y = pos
    x = x - offset_x
    y = y - offset_y
    if x <= 0 or y <= 0:
        return None, None
    
    ancho_nodo = ancho_juego // filas
    fila = y // ancho_nodo
    col = x // ancho_nodo
    
    if fila >= filas or col >= filas:
        return None, None
        
    return fila, col

def main(ventana, ancho_ventana):
    FILAS = 20  # Número de filas por defecto
    alto_ventana = ALTO_VENTANA
    panel_lateral = PANEL_LATERAL
    grid = crear_grid(FILAS, ancho_ventana - panel_lateral)

    inicio = None
    fin = None
    corriendo = True

    while corriendo:
        # Calcular dimensiones actuales
        ancho_juego = ancho_ventana - panel_lateral
        ancho_nodo = min(ancho_juego, alto_ventana) // FILAS
        offset_x = (ancho_juego - (ancho_nodo * FILAS)) // 2
        offset_y = (alto_ventana - (ancho_nodo * FILAS)) // 2

        dibujar(ventana, grid, FILAS, ancho_ventana, alto_ventana, panel_lateral)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                corriendo = False

            if event.type == pygame.VIDEORESIZE:
                ancho_ventana = event.w
                alto_ventana = event.h
                ventana = pygame.display.set_mode((ancho_ventana, alto_ventana), pygame.RESIZABLE)
                panel_lateral = int(ancho_ventana * 0.2)  # Ajustar panel lateral

            if pygame.mouse.get_pressed()[0]:  # Click izquierdo
                pos = pygame.mouse.get_pos()
                resultado = obtener_click_pos(pos, FILAS, offset_x, offset_y, min(ancho_juego, alto_ventana))
                if resultado[0] is not None:
                    fila, col = resultado
                    nodo = grid[fila][col]
                    if not inicio and nodo != fin:
                        inicio = nodo
                        inicio.hacer_inicio()
                    elif not fin and nodo != inicio:
                        fin = nodo
                        fin.hacer_fin()
                    elif nodo != fin and nodo != inicio:
                        nodo.hacer_pared()

            elif pygame.mouse.get_pressed()[2]:  # Click derecho
                pos = pygame.mouse.get_pos()
                resultado = obtener_click_pos(pos, FILAS, offset_x, offset_y, min(ancho_juego, alto_ventana))
                if resultado[0] is not None:
                    fila, col = resultado
                    nodo = grid[fila][col]
                    nodo.restablecer()
                    if nodo == inicio:
                        inicio = None
                    elif nodo == fin:
                        fin = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and inicio and fin:
                    for fila in grid:
                        for nodo in fila:
                            nodo.actualizar_vecinos(grid)

                    encontrado, caminos_alt, info_final, costo_optimo = algoritmo_a_estrella(
                        lambda info: dibujar(ventana, grid, FILAS, ancho_ventana, alto_ventana, panel_lateral, info),
                        grid, inicio, fin)
                    
                    if encontrado:
                        info_final.append("")
                        info_final.append(f"Caminos alternativos: {len(caminos_alt)}")
                        
                        # Mostrar caminos alternativos
                        tiempo_inicial = pygame.time.get_ticks()
                        for i, camino in enumerate(caminos_alt):
                            # Manejar eventos mientras se muestran los caminos
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    return
                                if event.type == pygame.VIDEORESIZE:
                                    ancho_ventana = event.w
                                    alto_ventana = event.h
                                    ventana = pygame.display.set_mode((ancho_ventana, alto_ventana), pygame.RESIZABLE)
                                    panel_lateral = int(ancho_ventana * 0.2)
                            
                            costo_alt = len(camino)
                            diferencia = costo_alt - costo_optimo
                            
                            info_actual = info_final.copy()
                            info_actual.append(f"Camino alt. {i+1}:")
                            info_actual.append(f"Costo: {costo_alt}")
                            info_actual.append(f"Diferencia: +{diferencia}")
                            
                            # Limpiar caminos alternativos anteriores
                            for fila in grid:
                                for nodo in fila:
                                    if nodo.color == AZUL:
                                        nodo.restablecer()
                                    if nodo == inicio:
                                        nodo.hacer_inicio()
                                    elif nodo == fin:
                                        nodo.hacer_fin()
                                    elif nodo.color == VERDE:
                                        nodo.hacer_camino()
                            
                            # Mostrar nuevo camino alternativo
                            for nodo in camino:
                                if nodo != inicio and nodo != fin:
                                    nodo.hacer_alternativo()
                            
                            dibujar(ventana, grid, FILAS, ancho_ventana, alto_ventana, panel_lateral, info_actual)
                            
                            # Usar tick para control de tiempo más preciso
                            while pygame.time.get_ticks() - tiempo_inicial < 1500:
                                for event in pygame.event.get():
                                    if event.type == pygame.QUIT:
                                        pygame.quit()
                                        return
                                pygame.time.delay(50)  # Pequeñas pausas para no consumir CPU
                            
                            tiempo_inicial = pygame.time.get_ticks()
                    else:
                        print("\nNo se encontró un camino!")

                if event.key == pygame.K_c:
                    inicio = None
                    fin = None
                    grid = crear_grid(FILAS, ancho_ventana - panel_lateral)
                
                if event.key == pygame.K_r:
                    # Restablecer dimensiones de la ventana
                    ancho_ventana = ANCHO_VENTANA
                    alto_ventana = ALTO_VENTANA
                    panel_lateral = PANEL_LATERAL
                    ventana = pygame.display.set_mode((ancho_ventana, alto_ventana), pygame.RESIZABLE)
                    grid = crear_grid(FILAS, ancho_ventana - panel_lateral)
                    inicio = None
                    fin = None

    pygame.quit()

main(VENTANA, ANCHO_VENTANA)
