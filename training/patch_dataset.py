"""Script one-shot: normaliza Unicode y añade ejemplos [SIN INFORMACION] al dataset.py"""
import re

# ── 1. NORMALIZAR UNICODE EN DATASET EXISTENTE ─────────────────────────────
UNICODE_MAP = {
    '≤': '<=',
    '≥': '>=',
    '→': '->',
    '←': '<-',
    '±': '+/-',
    '×': 'x',
    '÷': '/',
    '°': ' grados',
    '²': '^2',
    '³': '^3',
    '\u2022': '-',   # bullet
    '\u2013': '-',   # en-dash
    '\u2014': '-',   # em-dash
    '\u2018': "'",
    '\u2019': "'",
    '\u201c': '"',
    '\u201d': '"',
}

# ── 2. EJEMPLOS [SIN INFORMACION] ──────────────────────────────────────────
# Estos ejemplos ensenan al modelo a responder cuando el texto NO menciona la columna
NO_INFO_EXAMPLES = [
    # Columna presente pero texto es de otro dominio completamente
    {"input": "Extrae [Modelo predictivo utilizado]: Este articulo analiza el impacto del salario minimo en el empleo. Los autores utilizaron un diseno de diferencias en diferencias con datos de panel de 12 estados.",
     "output": "[SIN INFORMACION] El texto no describe ningun modelo predictivo; el diseno es cuasi-experimental (DiD)."},

    {"input": "Extrae [Escala psicometrica de ansiedad]: El estudio evalua el efecto de la lluvia acida en biodiversidad de macroinvertebrados en rios andinos usando indices de Shannon-Wiener y Pielou.",
     "output": "[SIN INFORMACION] El texto no contiene escalas psicometricas; se trata de un estudio ambiental con indices ecologicos."},

    {"input": "Extrae [Tasa de mortalidad cardiovascular]: Los autores comparan tres algoritmos de machine learning (Random Forest, XGBoost, SVM) para predecir el abandono escolar en educacion secundaria.",
     "output": "[SIN INFORMACION] El texto no reporta datos de mortalidad cardiovascular; el dominio es prediccion educativa."},

    {"input": "Extrae [Variables de resultado farmacologico]: Este articulo desarrolla un marco conceptual para analizar la adopcion de tecnologia movil en PYMEs de paises en desarrollo. No incluye datos empiricos.",
     "output": "[SIN INFORMACION] El articulo es teorico/conceptual y no reporta variables de resultado farmacologico."},

    {"input": "Extrae [Endpoint primario del ensayo]: La investigacion utiliza analisis de redes sociales para estudiar la difusion de desinformacion en Twitter durante elecciones presidenciales en Mexico 2024.",
     "output": "[SIN INFORMACION] El texto no es un ensayo clinico; no existe endpoint primario. Se trata de un estudio observacional en redes sociales."},

    {"input": "Extrae [Dosis del farmaco evaluado]: El articulo realiza una revision sistematica de estrategias pedagogicas para la educacion STEM en educacion primaria, analizando 45 estudios de 2010 a 2023.",
     "output": "[SIN INFORMACION] El articulo es una revision sistematica en educacion; no evalua farmacos ni dosis."},

    {"input": "Extrae [Modelo GARCH o VAR utilizado]: El estudio compara la eficacia de tres intervenciones psicologicas (TCC, ACT, mindfulness) para el tratamiento del trastorno de ansiedad generalizada en adultos.",
     "output": "[SIN INFORMACION] El texto describe un ECA de intervenciones psicologicas; no utiliza modelos de series de tiempo financieras."},

    {"input": "Extrae [Indice de Shannon-Wiener de biodiversidad]: El articulo evalua el desempeno de cuatro modelos de lenguaje grande (GPT-4, Claude, Gemini, Llama-3) en tareas de razonamiento matematico.",
     "output": "[SIN INFORMACION] El articulo compara LLMs para razonamiento matematico; no reporta indices de biodiversidad."},

    # Columna relacionada pero el texto simplemente no menciona ese aspecto
    {"input": "Extrae [Tamano del efecto de la intervencion]: Los autores describen el diseno y la implementacion de un prototipo de sensor IoT para monitoreo de humedad del suelo en cultivos de papa. No presentan resultados de campo.",
     "output": "[SIN INFORMACION] El articulo describe el diseno del prototipo pero no reporta resultados empiricos ni tamano del efecto."},

    {"input": "Extrae [Criterios de inclusion y exclusion de participantes]: Este trabajo propone una arquitectura de microservicios para sistemas de recomendacion en e-commerce. Es un articulo de ingenieria de software sin participantes humanos.",
     "output": "[SIN INFORMACION] No existen criterios de inclusion/exclusion: el articulo no involucra participantes humanos; es un estudio tecnico de arquitectura de software."},

    {"input": "Extrae [Estrategia de reclutamiento de pacientes]: El articulo presenta un analisis bibliometrico de la produccion cientifica sobre inteligencia artificial en el Peru durante 2015-2023 usando VOSviewer.",
     "output": "[SIN INFORMACION] El articulo es un bibliometrico; no recluta pacientes ni participantes experimentales."},

    {"input": "Extrae [Metricas de rendimiento del modelo de ML]: El estudio es una etnografia de 18 meses sobre practicas de subsistencia en comunidades indigenas amazonicas. Metodos: observacion participante y entrevistas.",
     "output": "[SIN INFORMACION] El articulo es un estudio etnografico cualitativo; no desarrolla ni evalua modelos de machine learning."},

    # El texto menciona el tema pero NO da el dato especifico
    {"input": "Extrae [Tamano de la muestra del estudio]: El articulo discute la importancia de definir correctamente los criterios de inclusion en estudios clinicos y propone una guia metodologica. No reporta un estudio empirico propio.",
     "output": "[SIN INFORMACION] El articulo es metodologico/conceptual; no reporta un estudio empirico con muestra propia."},

    {"input": "Extrae [Resultado primario de la intervencion educativa]: La investigacion describe el desarrollo y validacion de un cuestionario para medir competencias digitales en docentes universitarios (fase de diseno, n=8 expertos).",
     "output": "[SIN INFORMACION] El articulo reporta la fase de diseno/validacion del instrumento, no resultados de una intervencion educativa implementada."},

    {"input": "Extrae [Porcentaje de reduccion de CO2 logrado]: El estudio es una revision narrativa de politicas de precios al carbono en America Latina, discutiendo sus limitaciones y oportunidades. No reporta datos empiricos originales.",
     "output": "[SIN INFORMACION] El articulo es una revision narrativa de politicas; no reporta datos empiricos de reduccion de CO2."},

    # Casos donde el articulo menciona el concepto pero en otro contexto
    {"input": "Extrae [Escala de depresion utilizada en el screening]: El articulo analiza la viabilidad financiera de proyectos de energia solar en zonas rurales de Bolivia usando TIR, VPN y periodo de retorno.",
     "output": "[SIN INFORMACION] El articulo es de evaluacion financiera de proyectos energeticos; no utiliza escalas de depresion."},

    {"input": "Extrae [Algoritmo de clasificacion de imagenes medicas]: Los autores realizan un experimento de campo evaluando el efecto de tres densidades de siembra en el rendimiento del arroz en la costa del Peru.",
     "output": "[SIN INFORMACION] El articulo es un experimento agricola de campo; no desarrolla ni aplica algoritmos de clasificacion de imagenes medicas."},

    # Casos de respuesta parcial (menciona el concepto pero sin datos)
    {"input": "Extrae [Intervalo de confianza del efecto principal]: Los autores concluyen que la intervencion fue efectiva y estadisticamente significativa (p<0.05), pero no reportan el intervalo de confianza en el resumen ni en los resultados principales.",
     "output": "[SIN INFORMACION] El texto reporta significancia estadistica (p<0.05) pero no proporciona el intervalo de confianza del efecto principal."},

    {"input": "Extrae [Nombre del dataset publico utilizado]: El articulo describe un sistema de deteccion de fraude bancario usando redes neuronales. Los autores mencionan que usaron datos reales de un banco europeo bajo acuerdo de confidencialidad.",
     "output": "[SIN INFORMACION] Los datos son propietarios bajo confidencialidad; no se utiliza un dataset publico nombrable."},

    {"input": "Extrae [Porcentaje de acuerdo entre evaluadores]: El articulo reporta el proceso de analisis de contenido pero omite los indices de confiabilidad entre codificadores, mencionando solo que 'se alcanzaron acuerdos satisfactorios'.",
     "output": "[SIN INFORMACION] El texto menciona acuerdos entre evaluadores de forma cualitativa pero no reporta el porcentaje o coeficiente especifico (ej. kappa de Cohen)."},

    # Casos extremos: texto completamente irrelevante al campo de la columna
    {"input": "Extrae [Protocolo de quimioterapia evaluado]: Este articulo estudia la adopcion de practicas agiles (Scrum, Kanban) en empresas de desarrollo de software en Chile. Analiza 120 equipos de desarrollo.",
     "output": "[SIN INFORMACION] El articulo es sobre metodologias agiles en desarrollo de software; no evalua protocolos de quimioterapia ni ningun tratamiento oncologico."},

    {"input": "Extrae [Tasa de supervivencia a 5 anos]: El estudio analiza el sentido del humor como estrategia pedagogica en el aula universitaria, usando grupos focales (n=6) y observacion de clases.",
     "output": "[SIN INFORMACION] El articulo es un estudio cualitativo sobre pedagogia y humor; no reporta tasas de supervivencia clinica."},

    {"input": "Extrae [Coeficiente de Gini del pais]: El articulo evalua la eficacia de un chatbot de IA para soporte tecnico en una empresa de telecomunicaciones, midiendo satisfaccion del usuario (CSAT) y tiempo de resolucion.",
     "output": "[SIN INFORMACION] El articulo es sobre evaluacion de chatbot corporativo; no reporta el coeficiente de Gini ni datos de desigualdad economica."},

    {"input": "Extrae [Criterios DSM-5 para el diagnostico]: El trabajo presenta un modelo matematico para optimizar rutas de distribucion de vacunas en zonas rurales usando programacion entera mixta.",
     "output": "[SIN INFORMACION] El articulo es de investigacion de operaciones y logistica; no aplica criterios DSM-5 ni realiza diagnosticos psiquiatricos."},

    {"input": "Extrae [Fraccion de eyeccion ventricular izquierda]: El articulo analiza el impacto de las redes sociales en la autoestima de adolescentes usando la Escala de Autoestima de Rosenberg (EAR) en 340 estudiantes.",
     "output": "[SIN INFORMACION] El articulo estudia autoestima en adolescentes; no mide parametros cardiacos como la fraccion de eyeccion ventricular."},

    {"input": "Extrae [Tipo de reactor quimico utilizado]: La investigacion realiza un analisis de correspondencias multiples sobre los factores asociados a la desercion universitaria en universidades publicas del Peru.",
     "output": "[SIN INFORMACION] El articulo es un estudio estadistico sobre desercion universitaria; no describe ningun reactor quimico."},
]

print(f"Ejemplos [SIN INFORMACION] preparados: {len(NO_INFO_EXAMPLES)}")

# ── 3. LEER, PARCHEAR Y REESCRIBIR dataset.py ──────────────────────────────
with open('dataset.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Normalizar Unicode en todo el archivo
for char, replacement in UNICODE_MAP.items():
    content = content.replace(char, replacement)

# Encontrar donde termina el DATASET (antes del cierre ']')
# Insertar los nuevos ejemplos antes del ']' final
insert_marker = '\n]\n'
if insert_marker not in content:
    # Intentar con espacios extra
    insert_marker = '\n]\r\n'

# Construir el bloque de nuevos ejemplos
new_examples_str = "\n    # ================================================================\n"
new_examples_str += "    # DOMINIO 8: EJEMPLOS [SIN INFORMACION] — ANTI-ALUCINACION\n"
new_examples_str += "    # Ensenian al modelo a responder cuando el texto NO contiene datos\n"
new_examples_str += "    # para la columna pedida. Critico para fidelidad en produccion.\n"
new_examples_str += "    # ================================================================\n"

for ex in NO_INFO_EXAMPLES:
    inp = ex['input'].replace('\\', '\\\\').replace('"', '\\"')
    out = ex['output'].replace('\\', '\\\\').replace('"', '\\"')
    new_examples_str += f'    {{"input": "{inp}",\n     "output": "{out}"}},\n\n'

# Insertar antes del cierre del array
if '\n]\n' in content:
    content = content.replace('\n]\n', new_examples_str + '\n]\n', 1)
elif content.rstrip().endswith(']'):
    content = content.rstrip()[:-1] + new_examples_str + ']\n'

with open('dataset.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("OK: dataset.py actualizado")

# Verificar
from dataset import DATASET
print(f"Total ejemplos: {len(DATASET)}")
sin_info = [d for d in DATASET if '[SIN INFORMACION]' in d['output']]
print(f"Ejemplos [SIN INFORMACION]: {len(sin_info)}")
print(f"Ejemplos normales: {len(DATASET) - len(sin_info)}")
