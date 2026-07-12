"""
Exportación de resultados a CSV con campos adicionales
"""
import csv
import logging
from pathlib import Path

def export_to_csv(articles, filename, include_additional_fields=True):
    """
    Exporta artículos a CSV con todos los campos disponibles.
    
    Args:
        articles: Lista de diccionarios con información de artículos
        filename: Nombre del archivo de salida
        include_additional_fields: Si incluir campos adicionales del screening
    """
    if not articles:
        logging.warning(f"⚠️ No hay artículos para exportar a {filename}")
        return
    
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    filepath = logs_dir / filename
    
    # Definir campos base
    base_fieldnames = [
        'title',
        'authors',
        'year',
        'journal',
        'doi',
        'abstract',
        'url',
        'source',
        'similarity',
        'relevance'
    ]
    
    # Campos adicionales del screening manual
    additional_fieldnames = [
        'researcher_notes',
        'exclusion_reason',
        'methodology',
        'population',
        'intervention',
        'key_findings',
        'limitations',
        'conclusions',
        'translation'
    ]
    
    # Combinar campos
    if include_additional_fields:
        fieldnames = base_fieldnames + additional_fieldnames
    else:
        fieldnames = base_fieldnames
    
    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            
            for article in articles:
                # Preparar fila
                row = {}
                
                # Campos base
                row['title'] = article.get('title', '')
                row['authors'] = ', '.join(article.get('authors', []))
                row['year'] = article.get('year', '')
                row['journal'] = article.get('journal', '')
                row['doi'] = article.get('doi', '')
                row['abstract'] = article.get('abstract', '')
                row['url'] = article.get('url', '')
                row['source'] = article.get('source', '')
                row['similarity'] = article.get('similarity', '')
                row['relevance'] = article.get('relevance', '')
                
                # Campos adicionales si están habilitados
                if include_additional_fields:
                    row['researcher_notes'] = article.get('researcher_notes', '')
                    row['exclusion_reason'] = article.get('exclusion_reason', '')
                    row['methodology'] = article.get('methodology', '')
                    row['population'] = article.get('population', '')
                    row['intervention'] = article.get('intervention', '')
                    row['key_findings'] = article.get('key_findings', '')
                    row['limitations'] = article.get('limitations', '')
                    row['conclusions'] = article.get('conclusions', '')
                    row['translation'] = article.get('translation', '')
                
                writer.writerow(row)
        
        logging.info(f"✅ Exportado: {len(articles)} artículos → {filename}")
        
    except Exception as e:
        logging.error(f"❌ Error exportando a CSV {filename}: {e}")


def export_synthesis_to_txt(synthesis_text, filename="synthesis_output.txt"):
    """
    Exporta la síntesis generada a un archivo de texto.
    
    Args:
        synthesis_text: Texto de la síntesis
        filename: Nombre del archivo de salida
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    filepath = logs_dir / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(synthesis_text)
        
        logging.info(f"✅ Síntesis exportada → {filename}")
        
    except Exception as e:
        logging.error(f"❌ Error exportando síntesis: {e}")