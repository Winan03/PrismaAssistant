import logging
import re
from collections import Counter
import sys
import os

# Añadir el directorio raíz al path para importar módulos
sys.path.append(os.getcwd())

from modules.synthesis import refine_introduction
import modules.synthesis as synthesis
print(f"DEBUG: synthesis file path: {synthesis.__file__}")
from modules.ai_model import LocalModel

# Configurar logging para ver los detalles del refinamiento
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_refinement():
    bad_text = """
    I. INTRODUCCIÓN
    La seguridad del código fuente constituye un pilar crítico. Se analizan los pipelines de desarrollo continuo para detectar fallos. Zhu (2020) Evaluaron la homología de firmware (Zhu, 2020). Pooja et al. (2022) Propusieron una hoja de ruta. Akter (2022) contraste redes (Akter, 2022). Se detectó el F1score bajo. Aunque persiste una brecha. Esta carencia justifica este estudio. La metodología usa PRISMA. Los objetivos son: 1) analizar CI/CD.
    """
    
    print("=== TEXTO ORIGINAL ===")
    print(bad_text)
    print("\n=== INICIANDO REFINAMIENTO ===\n")
    
    # Simular términos protegidos
    protected = ["inteligencia artificial"]
    
    refined = refine_introduction(bad_text, protected_terms=protected)
    
    print("\n=== TEXTO REFINADO ===")
    print(refined)
    
    # Verificaciones básicas
    print("\n=== VERIFICACIONES ===")
    
    status = []
    
    # Banned words (with word boundaries)
    banned = ["actualmente", "en la actualidad", "ya que", "nos", "nuestro", "nuestra", "hoy", "tiempos", "seleccionamos", "analizamos"]
    found_banned = [w for w in banned if re.search(r'\b' + re.escape(w) + r'\b', refined, re.IGNORECASE)]
    if found_banned:
        status.append(f"❌ FALLO: Palabras prohibidas detectadas: {found_banned}")
    else:
        status.append("✅ ÉXITO: Palabras prohibidas eliminadas/corregidas")
        
    # Formato et al.
    if "etal" in refined.lower() or "et. al" in refined.lower():
        status.append("❌ FALLO: Formato 'et al.' incorrecto detectado")
    else:
        status.append("✅ ÉXITO: Formato 'et al.' correcto")

    # Repeticiones
    counts = Counter(re.findall(r'inteligencia artificial', refined.lower()))
    rep_count = counts['inteligencia artificial']
    if rep_count > 3:
        status.append(f"❌ FALLO: Demasiadas repeticiones ({rep_count})")
    else:
        status.append(f"✅ ÉXITO: Repeticiones reducidas ({rep_count})")
        
    # Typos
    if "falos" in refined.lower():
        status.append("❌ FALLO: Typo 'falos' no corregido")
    else:
        status.append("✅ ÉXITO: Typo 'falos' corregido")
        
    # PUNTUACIÓN Y FLUJO (NUEVO)
    # Check "Zhu et al. (2020) propusieron" (no period, lowercase verb)
    if re.search(r'\(2020\)\s+propusieron', refined, re.IGNORECASE):
        status.append("✅ ÉXITO: Puntuación corregida tras año (sujeto-verbo continuo)")
    elif re.search(r'\(2020\)\.\s+Propusieron', refined):
        status.append("❌ FALLO: Punto detectado tras el año (rompe la oración)")
    else:
        status.append("⚠️ AVISO: No se pudo validar la continuidad sujeto-verbo de Zhu")

    # Check abrupt breaks
    if "Akter et al. (2022)" in refined:
        status.append("✅ ÉXITO: Salto de línea abrupto en cita corregido")
    elif "Akter et al.\n" in refined:
        status.append("❌ FALLO: Salto de línea detectado en medio de la cita de Akter")

    # Redundancia
    if "(Akter et al., 2022)" in refined:
         status.append("❌ FALLO: Cita parentética redundante detectada")
    else:
         status.append("✅ ÉXITO: Cita redundante eliminada")

    # Orden Cronológico (Jones 2020 debe estar antes que Smith 2024)
    pos_jones = refined.find("2020")
    pos_smith = refined.find("2024")
    if pos_jones != -1 and pos_smith != -1:
        if pos_jones < pos_smith:
            status.append("✅ ÉXITO: Orden cronológico mantenido (2020 antes que 2024)")
        else:
            status.append("❌ FALLO: Orden cronológico roto (2024 antes que 2020)")
    else:
        status.append("⚠️ AVISO: No se encontraron las citas cronológicas esperadas para validar")

    # GAP y Objetivos (El GAP debe estar presente y después de la revisión literaria)
    gap_found = re.search(r'falta|brecha|limitación|vacio|gap|carencia|necesaria?|necesidad|carestía', refined, re.IGNORECASE)
    obj_found = re.search(r'objetivo|busco|pretende|finalidad', refined, re.IGNORECASE)
    
    if gap_found and obj_found:
        if gap_found.start() < obj_found.start():
            status.append("✅ ÉXITO: GAP detectado como puente antes de los objetivos")
        else:
            status.append("❌ FALLO: Los objetivos aparecen antes que el GAP")
    elif not gap_found:
        status.append("❌ FALLO: No se detectó el GAP")
    elif not obj_found:
        status.append("❌ FALLO: No se detectaron los OBJETIVOS")

    # CAPITALIZACIÓN (Harzevili, Bommareddy)
    for auth in ["Harzevili", "Bommareddy", "Chittibala"]:
        if auth in refined:
            status.append(f"✅ ÉXITO: {auth} correctamente capitalizado")
        elif auth.lower() in refined:
            status.append(f"❌ FALLO: {auth} aparece en minúsculas")

    # FORMATO DE FECHA
    if "2021-2025" in refined:
        status.append("✅ ÉXITO: Formato de fecha 2021-2025 correcto")
    elif "20212025" in refined:
        status.append("❌ FALLO: Formato de fecha 20212025 detectado (sin guion)")

    # EXTENSIÓN DE PÁRRAFO
    paragraphs = [p.strip() for p in refined.split('\n\n') if len(p.strip()) > 30]
    lengths = [len(p.split()) for p in paragraphs]
    over_limit = [l for l in lengths if l > 120]
    if over_limit:
        status.append(f"❌ FALLO: Párrafos demasiado largos detectados: {over_limit} palabras")
    else:
        status.append(f"✅ ÉXITO: Extensión de párrafos controlada (max {max(lengths) if lengths else 0} palabras)")

    # ALUCINACIONES (CoronaFraga)
    if "CoronaFraga" in refined:
        status.append("❌ FALLO: Fuente alucinada 'CoronaFraga' detectada")
    else:
        status.append("✅ ÉXITO: No se detectaron fuentes alucinadas")

    # REDUNDANCIA DE CITAS (V5)
    # Ejemplo: "Kiani y Sheng (2024) ... (Kiani y Sheng, 2024)"
    if re.search(r'Kiani y Sheng \(2024\).*\(Kiani y Sheng, 2024\)', refined):
        status.append("❌ FALLO: Citación redundante detectada (Narrativa + Parentética)")
    else:
        status.append("✅ ÉXITO: Citación redundante eliminada")

    # TERMINOLOGÍA Y UNIFICACIÓN (V6)
    if "pipelines de CI/CD" in refined:
        status.append("✅ ÉXITO: Terminología 'pipelines de CI/CD' unificada")
    if "pipelines de desarrollo continuo" in refined.lower():
        status.append("❌ FALLO: 'pipelines de desarrollo continuo' detectado (usar CI/CD)")

    if "falsos positivos" in refined.lower():
        status.append("✅ ÉXITO: Terminología 'falsos positivos' priorizada")

    # MAYÚSCULAS TRAS CITA NARRATIVA (V6)
    # Ejemplo: "Zhu et al. (2020) Evaluaron" -> "Zhu et al. (2020) evaluaron"
    if re.search(r'[A-Z][a-z]+\s+\(\d{4}\)\s+[A-Z][a-z]+', refined):
        # Permitir si la palabra post-cita es un nombre propio (et al. es común)
        # Pero si el verbo común está en mayúscula, es fallo
        if re.search(r'\(\d{4}\)\s+(?:Evaluaron|Propusieron|Analizaron|Revisaron)', refined):
            status.append("❌ FALLO: Mayúscula incorrecta tras cita narrativa (Sujeto-Verbo roto)")
        else:
            status.append("✅ ÉXITO: Capitalización tras cita narrativa parece correcta")
    else:
        status.append("✅ ÉXITO: Capitalización tras cita narrativa fluida")

    # FORMATO DE AÑOS (V5)
    if re.search(r'\d{4}-\d{4}', refined):
        status.append("✅ ÉXITO: Formato de rango de años (guion) detectado")

    # REPETICIÓN DE AUTORES (V5)
    paragraphs = refined.split('\n\n')
    found_reps = []
    for i in range(len(paragraphs)-1):
        p1 = paragraphs[i]
        p2 = paragraphs[i+1]
        authors1 = set(re.findall(r'\b[A-Z][a-z]+\b', p1))
        authors2 = set(re.findall(r'\b[A-Z][a-z]+\b', p2))
        common = authors1.intersection(authors2)
        common = {a for a in common if a not in ["En", "La", "El", "Los", "Las", "Aunque", "Para", "Este", "Zhu", "SAST", "LLM"]}
        if common:
            found_reps.append(list(common))
    
    if len(found_reps) > 2:
        status.append(f"❌ FALLO: Excesiva repetición de autores en párrafos contiguos ({found_reps})")
    else:
        status.append("✅ ÉXITO: Distribución de autores aceptable")

    for s in status:
        print(s)

if __name__ == "__main__":
    test_refinement()
