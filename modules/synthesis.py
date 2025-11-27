"""
Síntesis Narrativa Avanzada - Dual Output (HTML Breve + PDF Completo)
Genera síntesis estilo Elicit para web y reporte completo para descarga.
"""
import requests
import config
import logging
import time

def format_apa7_reference(article):
    """Genera referencia APA 7 con manejo robusto de caracteres especiales."""
    authors = article.get('authors', [])
    if isinstance(authors, str):
        authors = [authors]
        
    formatted_authors = []
    
    def clean_name(name):
        # Limpieza de caracteres especiales comunes
        name = name.replace('ł', 'l').replace('ń', 'n').replace('ą', 'a')
        name = name.replace('ć', 'c').replace('ę', 'e').replace('ó', 'o')
        name = name.replace('ś', 's').replace('ź', 'z').replace('ż', 'z')
        
        parts = name.replace(',', '').split()
        if not parts: return "Anon"
        if len(parts) == 1: return parts[0]
        surname = parts[-1]
        initials = "".join([f"{p[0]}." for p in parts[:-1]])
        return f"{surname}, {initials}"

    for auth in authors:
        formatted_authors.append(clean_name(auth))

    if len(formatted_authors) == 0:
        auth_str = "Anonymous."
    elif len(formatted_authors) == 1:
        auth_str = f"{formatted_authors[0]}"
    elif len(formatted_authors) == 2:
        auth_str = f"{formatted_authors[0]} & {formatted_authors[1]}"
    elif len(formatted_authors) <= 20:
        auth_str = ", ".join(formatted_authors[:-1]) + f", & {formatted_authors[-1]}"
    else:
        auth_str = ", ".join(formatted_authors[:19]) + f" ... {formatted_authors[-1]}"

    year = article.get('year') or "n.d."
    title = article.get('title', 'Untitled').strip()
    
    # Limpiar caracteres especiales del título
    title = title.encode('ascii', 'ignore').decode('ascii')
    if not title.endswith('.'): title += '.'

    journal = article.get('journal', 'Source Unknown')
    volume = article.get('volume', '')
    issue = article.get('issue', '')
    pages = article.get('pages', '')
    
    journal_vol = f"*{journal}*"
    if volume:
        journal_vol += f", *{volume}*"
    
    source_parts = [journal_vol]
    if issue:
        source_parts.append(f"({issue})")
    if pages:
        source_parts.append(f", {pages}")
        
    source_str = "".join(source_parts)
    if not source_str.endswith('.'): source_str += '.'

    doi = article.get('doi', '')
    url = article.get('url', '')
    link = ""
    if doi:
        link = f"https://doi.org/{doi}" if "doi.org" not in doi else doi
    elif url:
        link = url

    reference = f"{auth_str} ({year}). {title} {source_str} {link}"
    return reference

def get_in_text_citation(article):
    """Genera cita parentética (Smith et al., 2023)."""
    authors = article.get('authors', [])
    year = article.get('year', 'n.d.')
    
    if not authors:
        return f"(Anonymous, {year})"
    
    try:
        if isinstance(authors, str): 
            authors = [authors]
        
        last_names = []
        for auth in authors:
            # Limpieza de caracteres especiales
            auth = auth.replace('ł', 'l').replace('ń', 'n')
            parts = auth.replace(',', '').split()
            last_names.append(parts[-1] if parts else "Anon")

        if len(last_names) == 1:
            return f"({last_names[0]}, {year})"
        elif len(last_names) == 2:
            return f"({last_names[0]} & {last_names[1]}, {year})"
        else:
            return f"({last_names[0]} et al., {year})"
    except:
        return f"(Autor, {year})"

def generate_synthesis_brief(articles, question):
    """
    Genera síntesis BREVE estilo Elicit (3-4 párrafos) para mostrar en HTML.
    """
    if not articles:
        return "⚠️ No hay artículos para sintetizar."

    # Limitamos a los 4-8 más relevantes como Elicit
    top_articles = sorted(articles, key=lambda x: x.get('similarity', 0), reverse=True)[:6]
    
    context_parts = []
    for i, art in enumerate(top_articles):
        citation = get_in_text_citation(art)
        context_parts.append(
            f"ARTÍCULO {i+1} - {citation}:\n"
            f"Título: {art.get('title', '')}\n"
            f"Hallazgos clave: {art.get('key_findings', '') or art.get('abstract', '')[:300]}\n"
        )

    context_text = "\n".join(context_parts)
    
    system_prompt = (
        "Eres un experto investigador escribiendo un resumen ejecutivo estilo Elicit. "
        "Tu respuesta debe ser BREVE (3-4 párrafos), directa y enfocada en responder la pregunta."
    )
    
    user_prompt = f"""
PREGUNTA DE INVESTIGACIÓN: "{question}"

Basándote en estos {len(top_articles)} estudios principales, escribe un resumen ejecutivo que:
1. Responda directamente la pregunta (primer párrafo)
2. Cite los hallazgos más importantes con formato (Autor, Año)
3. Mencione tendencias o consensos (segundo párrafo)
4. Identifique gaps o limitaciones (tercer párrafo)

ESTUDIOS:
{context_text}

FORMATO: Usa Markdown simple. Máximo 300 palabras. Estilo académico pero accesible.
"""

    return _call_openrouter(system_prompt, user_prompt, max_tokens=800)

def generate_synthesis_full(articles, question):
    """
    Genera síntesis COMPLETA para el PDF (detallada, con secciones, referencias).
    """
    if not articles:
        return "⚠️ No hay artículos seleccionados."

    context_parts = []
    references_list = []
    
    for i, art in enumerate(articles):
        citation_key = get_in_text_citation(art)
        ref_entry = format_apa7_reference(art)
        references_list.append(ref_entry)
        
        context_parts.append(
            f"--- ARTÍCULO {i+1} ---\n"
            f"CITA: {citation_key}\n"
            f"TÍTULO: {art.get('title', '')}\n"
            f"RESUMEN: {art.get('abstract', '')}\n"
            f"METODOLOGÍA: {art.get('methodology', '')}\n"
            f"HALLAZGOS: {art.get('key_findings', '')}\n"
            f"CONCLUSIONES: {art.get('conclusions', '')}\n"
        )

    context_text = "\n".join(context_parts)
    formatted_refs = "\n".join([f"{i+1}. {ref}" for i, ref in enumerate(references_list)])
    
    system_prompt = (
        "Eres un experto redactando la sección de 'Resultados y Discusión' de una revisión sistemática. "
        "Escribe de forma exhaustiva, crítica y académica."
    )
    
    user_prompt = f"""
PREGUNTA: "{question}"

Escribe una síntesis narrativa completa basada en los {len(articles)} estudios incluidos.

REGLAS:
1. NO listes artículos. Agrupa por TEMAS (ej: "Eficacia de Algoritmos", "Limitaciones Metodológicas")
2. Citas APA 7 obligatorias en cada afirmación
3. Contrasta resultados entre estudios
4. Incluye datos específicos (porcentajes, n, p-valores)
5. Estructura clara con secciones numeradas

ARTÍCULOS:
{context_text}

ESTRUCTURA REQUERIDA:
# Síntesis Narrativa

## 1. Introducción
(Visión general de los hallazgos)

## 2. [Tema Principal 1]
(Sintetiza estudios relacionados, contrasta resultados)

## 3. [Tema Principal 2]
(Continúa análisis)

## 4. Metodologías Empleadas
(Métodos predominantes, calidad de evidencia)

## 5. Limitaciones y Gaps
(Limitaciones reportadas, áreas sin explorar)

## 6. Conclusiones
(Respuesta directa a la pregunta de investigación)

## Referencias Bibliográficas
{formatted_refs}
"""

    return _call_openrouter(system_prompt, user_prompt, max_tokens=4000)

def _call_openrouter(system_prompt, user_prompt, max_tokens=2000):
    """Función auxiliar para llamadas a OpenRouter."""
    headers = {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://upao.edu.pe",
        "X-Title": "PRISMA Assistant"
    }
    
    payload = {
        "model": config.OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens
    }

    try:
        start_t = time.time()
        response = requests.post(
            f"{config.OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=180
        )
        
        if response.status_code == 200:
            content = response.json()['choices'][0]['message']['content']
            logging.info(f"✅ Síntesis generada en {time.time() - start_t:.2f}s")
            return content
        else:
            logging.error(f"❌ Error API: {response.status_code}")
            return f"Error al generar síntesis (código {response.status_code})"

    except Exception as e:
        logging.error(f"❌ Excepción: {e}")
        return "Error técnico al conectar con IA."

# Función de compatibilidad (mantiene la interfaz anterior)
def generate_synthesis(articles, question):
    """Genera síntesis completa (usada por el endpoint de PDF)."""
    return generate_synthesis_full(articles, question)