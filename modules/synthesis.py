"""
Síntesis MEJORADA - Estilo Consensus
Genera síntesis estructuradas y profesionales
"""
import requests
import config
import logging
import time

def format_apa_reference(article):
    """Formato APA simplificado"""
    authors = article.get('authors', [])
    
    author_list = []
    for full_name in authors:
        parts = full_name.split()
        if len(parts) > 1:
            last_name = parts[-1]
            initials = "".join([p[0] + "." for p in parts[:-1]])
            author_list.append(f"{last_name}, {initials}")
        else:
            author_list.append(full_name)

    if len(author_list) > 1:
        authors_str = ", ".join(author_list[:-1]) + f" & {author_list[-1]}"
    elif len(author_list) == 1:
        authors_str = author_list[0]
    else:
        authors_str = "Anon."
    
    year = article['year'] if article.get('year') else 'n.d.'
    title = article.get('title', '')
    journal = article.get('journal', 'Sin revista')
    doi = article.get('doi', '')
    
    return f"{authors_str} ({year}). {title}. *{journal}*. DOI: {doi}"


def generate_synthesis(rag_results, question):
    """
    ✅ NUEVA SÍNTESIS - ESTILO CONSENSUS
    Genera síntesis con CONTEXTO + EVIDENCIA + GAPS + CONCLUSIÓN
    """
    if not rag_results:
        return "No hay artículos relevantes para sintetizar."

    # 1. Preparar contexto para IA
    context_parts = []
    apa_references = []
    
    for i, r in enumerate(rag_results):
        authors = r.get('authors', [])
        first_author_surname = "et al."
        if authors:
            first_author_name = authors[0]
            first_author_surname = first_author_name.split()[-1] if first_author_name.split() else "Anon."
        
        year = r['metadata']['year'] if r.get('metadata') else 'n.d.'
        citation_tag = f"({first_author_surname}, {year})"
        
        context_parts.append(
            f"--- Artículo {i+1} {citation_tag} ---\n"
            f"Título: {r['title']}\n"
            f"Autores: {', '.join(authors)}\n"
            f"Resumen: {r['abstract'][:1500]}\n"  # ✅ MÁS CONTEXTO
        )
        apa_references.append(format_apa_reference(r))

    context = "\n\n".join(context_parts)
    references_text = "\n".join([f"{i+1}. {ref}" for i, ref in enumerate(apa_references)])
    
    # ✅ NUEVO PROMPT - ESTILO CONSENSUS
    prompt = f"""Eres un revisor sistemático experto. Genera una síntesis narrativa PROFESIONAL en español siguiendo el formato de revisiones sistemáticas publicadas en revistas de alto impacto.

PREGUNTA DE INVESTIGACIÓN:
{question}

ARTÍCULOS PARA ANÁLISIS:
{context}

ESTRUCTURA REQUERIDA:

## 1. Contexto y Antecedentes (2-3 párrafos)
- Explica la importancia del tema
- Presenta el estado actual del conocimiento
- Justifica la necesidad de esta revisión

## 2. Evidencia Científica Encontrada (3-4 párrafos)
- OBLIGATORIO: Cita estudios específicos usando (Apellido, Año)
- Agrupa hallazgos por temas o metodologías
- Menciona tamaños de muestra y resultados cuantitativos cuando estén disponibles
- Compara resultados entre estudios

## 3. Brechas de Conocimiento y Limitaciones
- Identifica qué falta por investigar
- Menciona limitaciones metodológicas de los estudios revisados
- Señala inconsistencias entre estudios si existen

## 4. Conclusión y Recomendaciones
- Resume los hallazgos principales
- Da recomendaciones para la práctica clínica (si aplica)
- Sugiere direcciones futuras de investigación

## 5. Referencias
{references_text}

INSTRUCCIONES CRÍTICAS:
- Escribe de forma académica pero clara
- USA CITAS EN EL TEXTO: (Apellido, Año) - OBLIGATORIO
- Menciona DATOS CONCRETOS: porcentajes, tamaños de muestra (n=X), p-values
- NO repitas información, sintetiza
- Longitud: 800-1200 palabras
"""

    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": config.OPENROUTER_MODEL,
        "messages": [
            {
                'role': 'system', 
                'content': 'Eres un revisor sistemático experto. Escribes síntesis académicas con citas apropiadas y datos concretos.'
            },
            {'role': 'user', 'content': prompt}
        ],
        "temperature": 0.4,  # ✅ Más creatividad que antes (0.3)
        "max_tokens": 3000   # ✅ Más tokens para síntesis largas
    }

    # Reintentos
    max_retries = 5
    for attempt in range(max_retries):
        try:
            logging.info(f"📝 Generando síntesis profesional (Intento {attempt + 1}/{max_retries})...")
            
            resp = requests.post(
                f"{config.OPENROUTER_BASE_URL}/chat/completions", 
                headers=headers, 
                json=data, 
                timeout=120 
            )
            
            if resp.status_code == 200:
                synthesis = resp.json()["choices"][0]["message"]["content"]
                
                # ✅ VALIDACIÓN: Verificar que tenga citas
                if "(" not in synthesis or ")" not in synthesis:
                    logging.warning("⚠️ Síntesis sin citas, reintentando...")
                    continue
                
                logging.info("✅ Síntesis generada exitosamente")
                return synthesis
            
            elif resp.status_code >= 500 or resp.status_code == 429:
                wait_time = 5 * (attempt + 1)
                logging.error(f"❌ OpenRouter error: {resp.status_code}. Reintentando en {wait_time}s...")
                time.sleep(wait_time)
            
            else:
                logging.error(f"❌ OpenRouter error: {resp.status_code} {resp.text}")
                return "Error en la generación de síntesis (Revisar logs)."

        except requests.exceptions.Timeout:
            wait_time = 5 * (attempt + 1)
            logging.error(f"❌ Timeout agotado. Reintentando en {wait_time}s...")
            time.sleep(wait_time)
        except Exception as e:
            logging.error(f"❌ Error inesperado: {e}")
            time.sleep(5)
            
    logging.error(f"🚨 Fallo permanente después de {max_retries} intentos")
    return "Fallo en la síntesis: No se pudo conectar con el modelo de IA después de múltiples intentos."