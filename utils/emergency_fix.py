# emergency_fix.py - CORRECCIONES DE EMERGENCIA
import re

def emergency_clean_metadata(metadata):
    """Correcciones de emergencia para metadata corrupta."""
    if not metadata:
        return metadata
    
    # Título en español
    if 'title_es' in metadata and metadata['title_es']:
        title = metadata['title_es']
        # Eliminar texto corrupto
        if "La inteligencia artificial" in title and ":" in title:
            # Extraer solo después de los dos puntos
            parts = title.split(":")
            if len(parts) > 1:
                title = parts[1].strip()
                if not title:
                    title = "Inteligencia Artificial Generativa en Educación Superior: Una Revisión Sistemática de la Literatura"
        
        # Asegurar formato
        if ":" not in title:
            title = f"{title}: Una Revisión Sistemática de la Literatura"
        
        metadata['title_es'] = title
    
    # Título en inglés
    if 'title_en' in metadata and metadata['title_en']:
        title = metadata['title_en']
        # Si está en español, traducir
        if any(word in title.lower() for word in ['inteligencia', 'artificial', 'revisión']):
            title = metadata.get('title_es', '').replace(
                "Inteligencia Artificial Generativa",
                "Generative Artificial Intelligence"
            ).replace(
                "Una Revisión Sistemática de la Literatura",
                "A Systematic Literature Review"
            )
        metadata['title_en'] = title
    
    # Abstract en inglés
    if 'abstract' in metadata and metadata['abstract']:
        abstract = metadata['abstract']
        # Si está en español, marcar como pendiente
        if any(word in abstract.lower() for word in ['este', 'artículo', 'para', 'los']):
            metadata['abstract'] = "[EN PROCESO DE TRADUCCIÓN] " + metadata.get('resumen', '')[:200] + "..."
    
    return metadata