import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("="*80)
print("Generador de comentarios sinteticos para ampliar dataset RAG")
print("="*80)

#-------------- Templates de comentarios por concepto filosofico ------------------------

# Basados en los ejes teóricos del proyecto

templates_vacío_existencial = [
    "A veces siento que mi vida no tiene dirección clara. Las redes sociales muestran vidas perfectas, pero yo solo siento vacío.",
    "¿Para qué trabajar tanto si al final todo pierde sentido? La presión constante me agota y no veo un propósito real.",
    "Camus hablaba del absurdo de la existencia. Hoy lo vivo scrolleando Instagram a las 3 AM sin saber por qué.",
    "Me pregunto si realmente importo o si solo soy otro perfil más en el algoritmo. El vacío existencial es real.",
    "La sensación de que nada tiene sentido permanente es constante. ¿Qué queda cuando las likes desaparecen?",
    "Sartre decía que estamos condenados a ser libres, pero yo me siento condenado a ser productivo sin propósito.",
    "No encuentro un 'para qué' en mi día a día. Solo reproduzco rutinas que vi en TikTok.",
    "El vacío que siento no se llena con más contenido. Paradójicamente, mientras más consumo, menos significado encuentro.",
    "¿Existe algo más allá de las métricas? Mi valor parece medirse en followers, no en lo que realmente soy.",
    "Lyotard predijo el fin de los metarrelatos. Yo lo confirmo: no creo en ningún proyecto colectivo grande.",
]

templates_identidad_líquida = [
    "Mi identidad cambia según la plataforma. En LinkedIn soy profesional, en Instagram creativo, en TikTok gracioso. ¿Quién soy realmente?",
    "Bauman tenía razón: somos identidades líquidas. Me reinvento cada semana según lo que esté trending.",
    "Cambio mi bio cada mes. No sé si es adaptabilidad o falta de una identidad sólida.",
    "Los algoritmos me muestran versiones de mí que ni siquiera reconozco. ¿Me estoy construyendo yo o me construye la IA?",
    "No tengo certezas sobre quién soy. Solo tengo versiones temporales que duran hasta el siguiente trend.",
    "Mi 'yo' es un proyecto en constante edición. Nada parece definitivo ni duradero.",
    "La identidad digital es performativa. Actuamos roles según lo que el algoritmo recompensa con visibilidad.",
    "¿Soy auténtico o solo imito modelos que vi en mi feed? La línea es cada vez más difusa.",
    "Construyo y destruyo mi identidad online como si fuera un juego. Pero a veces me pregunto si perdí mi esencia.",
    "Bauman advertía sobre la modernidad líquida. Hoy la vivo: todo es temporal, nada es sólido, ni siquiera yo.",
]

templates_algoritmos_autonomía = [
    "¿Elijo yo qué ver o el algoritmo decide por mí? La línea entre mi voluntad y la manipulación es borrosa.",
    "Netflix me dice qué ver, Spotify qué escuchar, Instagram qué comprar. ¿Sigo siendo autónomo?",
    "Los algoritmos me conocen mejor que yo mismo. Predicen mis deseos antes de que los tenga.",
    "Foucault hablaría de biopoder algorítmico. Nos vigilan, analizan y dirigen sin que lo notemos.",
    "¿Es libertad si todas mis opciones están pre-filtradas por un algoritmo?",
    "Las burbujas de filtro me encierran en mi propia cámara de eco. Ya no veo perspectivas diferentes.",
    "Heidegger diría que la tecnología nos des-oculta, pero también nos oculta otras posibilidades.",
    "Mi libertad de elección es una ilusión. Solo elijo dentro del menú que el algoritmo me presenta.",
    "Los algoritmos crean mis deseos. Compro cosas que ni sabía que quería hasta que las vi recomendadas.",
    "La IA diseña mi realidad. ¿Hasta qué punto soy un agente libre y hasta qué punto un sujeto programado?",
]

templates_rendimiento_burnout = [
    "Byung-Chul Han describe perfectamente mi vida: autoexplotación disfrazada de autorrealización.",
    "La cultura del rendimiento me está quemando. No puedo parar de producir, optimizar, mejorar.",
    "Estoy exhausto de intentar ser productivo 24/7. El burnout es el precio de esta sociedad del rendimiento.",
    "No basta con trabajar. También debo hacer ejercicio, meditar, leer, aprender idiomas. Es agotador.",
    "La sociedad del cansancio es real. Me siento culpable cuando descanso.",
    "El hustle culture me vendió el éxito. Lo que obtuve fue ansiedad y agotamiento crónico.",
    "No puedo ser productivo todo el tiempo, pero la presión social me hace sentir que debo serlo.",
    "Mi valor como persona parece estar atado a mi productividad. Si no produzco, no valgo.",
    "El burnout juvenil es una crisis silenciosa. Todos actuamos como si estuviéramos bien, pero estamos rotos.",
    "Han lo explicó: nos autoexplotamos pensando que es libertad. Es esclavitud disfrazada.",
]

templates_hiperconectividad_soledad = [
    "Tengo 500 amigos online y me siento completamente solo. La hiperconectividad nos desconecta.",
    "Estamos conectados 24/7 pero nunca hemos estado más aislados. Paradoja de la era digital.",
    "Las redes sociales prometen conexión pero entregan soledad algorítmica.",
    "Me siento más cerca de extraños en Twitter que de personas reales. Algo está mal.",
    "La hiperconectividad es una falsa promesa. Tenemos miles de contactos pero ninguna conexión real.",
    "Pasamos horas en redes sociales buscando conexión humana, pero solo encontramos perfiles, no personas.",
    "La soledad digital es única: estás rodeado de gente pero completamente aislado.",
    "Habermas tenía razón: el espacio público se erosiona. Hoy es un espacio algorítmico sin comunidad real.",
    "Chateamos, no conversamos. Nos etiquetamos, no nos conocemos. Estamos solos juntos.",
    "La ironía de nuestra era: nunca fue tan fácil conectar y nunca fue tan difícil sentirnos cerca.",
]

templates_inmediatez_proyectos = [
    "Todo debe ser instantáneo. No puedo comprometerme con proyectos a largo plazo.",
    "La gratificación inmediata arruinó mi capacidad de esperar. Quiero todo ya.",
    "Los proyectos de vida a largo plazo son un concepto muerto. Vivimos el eterno presente.",
    "No puedo pensar en mi futuro a 10 años. Apenas puedo planear la próxima semana.",
    "La cultura de lo inmediato nos robó la paciencia. Queremos resultados sin proceso.",
    "TikTok me entrenó para consumir contenido en ráfagas de 15 segundos. Ya no puedo leer libros.",
    "La inmediatez destruye la profundidad. Todo es superficial, rápido, desechable.",
    "No hay tiempo para reflexionar cuando todo cambia tan rápido. Vivimos en modo reactivo constante.",
    "Los planes a futuro son obsoletos. Vivimos de trend en trend, de momento en momento.",
    "La paciencia es un lujo que nuestra generación no puede permitirse en la era de lo instantáneo.",
]

templates_vigilancia_privacidad = [
    "Foucault describió el panóptico. Hoy es digital: nos vigilan constantemente sin que lo notemos.",
    "Mi teléfono sabe más de mí que mi familia. Es el perfecto dispositivo de vigilancia voluntaria.",
    "Aceptamos términos y condiciones sin leer. Firmamos nuestra propia vigilancia masiva.",
    "La privacidad es un concepto del pasado. Ahora todo es datos, perfiles, predicciones.",
    "¿A quién le importa la privacidad cuando podemos tener servicios gratis? Ese es el trato faustiano digital.",
    "Nos vigilan para vendernos cosas, para predecir comportamientos, para controlarnos sutilmente.",
    "El biopoder de Foucault era limitado. El poder algorítmico es total e invisible.",
    "Cada click, cada búsqueda, cada like es registrado. Somos perfiles permanentemente monitoreados.",
    "La vigilancia digital no necesita guardias. El algoritmo es el nuevo panóptico.",
    "Renunciamos a nuestra privacidad a cambio de conveniencia. Fue un mal trato.",
]

templates_pensamiento_crítico = [
    "Las burbujas de filtro destruyen el pensamiento crítico. Solo veo lo que confirma mis creencias.",
    "Ya no evalúo información, solo consumo lo que el algoritmo me sirve pre-digerido.",
    "El pensamiento crítico requiere exposición a ideas diferentes. Los algoritmos me dan lo contrario.",
    "¿Cómo pensar críticamente si nunca veo perspectivas que desafíen mi visión del mundo?",
    "Los algoritmos optimizan para engagement, no para verdad. El resultado es una sociedad menos crítica.",
    "Me preocupa que mis opiniones sean solo ecos de mi burbuja algorítmica.",
    "El pensamiento profundo requiere tiempo y silencio. Ambos son escasos en la era digital.",
    "Ya no buscamos información, la recibimos. Esa pasividad mata el pensamiento crítico.",
    "Habermas advertía sobre la erosión del espacio público racional. Los algoritmos lo consumaron.",
    "No puedo pensar críticamente cuando mi atención está secuestrada por notificaciones constantes.",
]

templates_ansiedad_futuro = [
    "El futuro me genera ansiedad paralizante. Demasiadas opciones, ninguna certeza.",
    "No sé qué estudiar, dónde trabajar, cómo vivir. La incertidumbre es abrumadora.",
    "Nuestros padres tenían trayectorias lineales. Nosotros tenemos laberintos infinitos de posibilidades.",
    "La paradoja de la elección nos paraliza. Tenemos todas las opciones pero no sabemos cuál tomar.",
    "El futuro se siente apocalíptico: crisis climática, colapso económico, automatización. ¿Para qué planear?",
    "La ansiedad generacional es real. El mundo nos entregó una crisis múltiple sin manual de instrucciones.",
    "¿Cómo planear un futuro cuando todo cambia tan rápido que los planes son obsoletos antes de ejecutarlos?",
    "La incertidumbre constante se volvió nuestra única certeza. Es agotador vivir así.",
    "No puedo imaginar mi vida a futuro. Solo sobrevivo el presente.",
    "La ansiedad por el futuro me roba el presente. Pero no planear también genera ansiedad. Es un círculo vicioso.",
]

templates_autenticidad_performance = [
    "¿Soy auténtico o solo interpreto la versión de mí que genera más likes?",
    "Las redes sociales me obligaron a ser performativo. Actúo mi vida más que vivirla.",
    "La autenticidad es imposible cuando todo es público y cuantificable.",
    "Dejo de hacer cosas auténticas porque 'no se ven bien en Instagram'. Eso es triste.",
    "Mi vida se volvió una curaduría constante. Filtro, edito, publico. ¿Dónde quedó lo real?",
    "La presión por mostrar una vida perfecta me aleja de mi verdadero yo.",
    "Performamos felicidad, éxito, satisfacción. Pero detrás de los filtros hay vacío.",
    "¿Puedo ser yo mismo en un espacio donde todo es espectáculo y métrica?",
    "La autenticidad no genera engagement. El algoritmo premia la performance, no la verdad.",
    "Vivimos en el teatro digital de Erving Goffman. Todos somos actores de nuestra propia vida.",
]

# -------------- Funcion para generar comentarios --------------

def generar_dataset_ampliado(n_comentarios=200):
    """
    Genera n_comentarios nuevos basados en los templates filosóficos
    """
    
    # Combinar todos los templates
    todos_templates = {
        'Generación Z y crisis de sentido': (
            templates_vacío_existencial + 
            templates_identidad_líquida + 
            templates_ansiedad_futuro +
            templates_autenticidad_performance
        ),
        'IA y pérdida de autonomía humana': (
            templates_algoritmos_autonomía + 
            templates_vigilancia_privacidad +
            templates_pensamiento_crítico
        ),
        'Cultura de lo efímero y proyectos de vida': (
            templates_inmediatez_proyectos + 
            templates_hiperconectividad_soledad +
            templates_rendimiento_burnout
        )
    }
    
    # Lista para almacenar nuevos comentarios
    nuevos_comentarios = []
    
    # Generar fechas aleatorias entre 2022 y 2024
    fecha_inicio = datetime(2022, 1, 1)
    fecha_fin = datetime(2024, 6, 30)
    
    sentimientos = ['positivo', 'neutral', 'negativo']
    
    for i in range(n_comentarios):
        # Seleccionar tema aleatorio
        tema = random.choice(list(todos_templates.keys()))
        
        # Seleccionar template del tema
        texto = random.choice(todos_templates[tema])
        
        # Generar datos aleatorios
        fecha_random = fecha_inicio + timedelta(
            days=random.randint(0, (fecha_fin - fecha_inicio).days)
        )
        
        usuario = f"user_{random.randint(1000, 99999)}"
        sentimiento = random.choice(sentimientos)
        likes = random.randint(50, 20000)
        reposts = random.randint(0, 5000)
        
        nuevos_comentarios.append({
            'id': 5001 + i,
            'fecha': fecha_random.strftime('%Y-%m-%d'),
            'usuario': usuario,
            'texto': texto,
            'tema': tema,
            'sentimiento': sentimiento,
            'likes': likes,
            'reposts': reposts
        })
    
    return pd.DataFrame(nuevos_comentarios)

# -------------- Generar y combinar datasets ---------------

print("\n Generando nuevos comentarios sintéticos...")

# Cargar dataset limpio
df_limpio = pd.read_csv('dataset_limpio.csv')
print(f"   • Dataset limpio: {len(df_limpio)} registros")

# Generar nuevos comentarios
df_nuevos = generar_dataset_ampliado(n_comentarios=200)
print(f"   • Comentarios nuevos generados: {len(df_nuevos)}")

# Combinar datasets
df_final = pd.concat([df_limpio, df_nuevos], ignore_index=True)
print(f"   • Dataset final: {len(df_final)} registros")

# Verificar distribución
print("\n Distribución final por tema:")
for tema, count in df_final['tema'].value_counts().items():
    print(f"   • {tema}: {count}")

print("\n Distribución final por sentimiento:")
for sent, count in df_final['sentimiento'].value_counts().items():
    print(f"   • {sent}: {count}")

# -------------- Guardar dataset final ---------------

output_path = 'dataset_final_para_rag.csv'
df_final.to_csv(output_path, index=False)

print(f"\n Dataset final guardado: {output_path}")

# También guardar solo los nuevos comentarios
output_nuevos = 'comentarios_nuevos_generados.csv'
df_nuevos.to_csv(output_nuevos, index=False)
print(f" Comentarios nuevos guardados: {output_nuevos}")

# -------------- Análisis de cobertura conceptual ---------------

print("\n" + "="*80)
print("ANÁLISIS DE COBERTURA CONCEPTUAL DEL DATASET FINAL")
print("="*80)

conceptos_clave = {
    'vacío_existencial': ['vacío', 'sentido', 'propósito', 'existencial'],
    'identidad': ['identidad', 'yo', 'ser', 'auténtico'],
    'algoritmos': ['algoritmo', 'ia', 'inteligencia artificial'],
    'autonomía': ['autonomía', 'libertad', 'elección', 'libre'],
    'rendimiento': ['rendimiento', 'productivo', 'burnout', 'agotamiento'],
    'hiperconectividad': ['hiperconectado', 'redes sociales', 'conectado', 'soledad'],
    'inmediatez': ['inmediato', 'instantáneo', 'rápido', 'efímero'],
    'vigilancia': ['vigilancia', 'privacidad', 'datos', 'panóptico'],
    'ansiedad': ['ansiedad', 'angustia', 'preocupación', 'estrés'],
    'performance': ['performativo', 'actuar', 'espectáculo', 'show']
}

print("\n Cobertura por concepto filosófico:")
for concepto, palabras in conceptos_clave.items():
    patron = '|'.join(palabras)
    matches = df_final['texto'].str.lower().str.contains(patron, case=False, regex=True)
    count = matches.sum()
    porcentaje = (count / len(df_final)) * 100
    print(f"   • {concepto:25s}: {count:4d} textos ({porcentaje:5.1f}%)")

print("\n Dataset enriquecido y listo para sistema RAG!")
print("="*80)