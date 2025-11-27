"""
Generador de Reportes PDF PROFESIONAL - Estilo Paper Científico
Incluye: Diagrama PRISMA, Síntesis Narrativa, Top 2 Artículos con Figuras
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from fpdf import FPDF
import os
import textwrap
import requests
from io import BytesIO
from PIL import Image
import logging

class PDFReport(FPDF):
    """PDF personalizado con header/footer profesional."""
    
    def header(self):
        self.set_font('Arial', 'B', 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 8, 'Revision Sistematica PRISMA - UPAO', 0, 1, 'R')
        self.ln(3)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

def clean_text_for_pdf(text):
    """Limpia texto de caracteres no soportados por FPDF."""
    # Reemplazos de caracteres especiales comunes
    replacements = {
        'ł': 'l', 'ń': 'n', 'ą': 'a', 'ć': 'c', 'ę': 'e',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z', 'Ł': 'L',
        'Ń': 'N', 'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ó': 'O',
        'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z',
        # Otros caracteres problemáticos
        ''': "'", ''': "'", '"': '"', '"': '"', '–': '-', '—': '-',
        '…': '...', '×': 'x', '°': ' grados', 'μ': 'u', 'α': 'alpha',
        'β': 'beta', 'γ': 'gamma', '≤': '<=', '≥': '>=', '±': '+/-'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Eliminar cualquier carácter no-ASCII restante
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text

def generate_prisma_diagram(metrics, filename="static/prisma_flow.png"):
    """Genera diagrama PRISMA profesional."""
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    box_props = dict(boxstyle='round,pad=0.5', facecolor='#e0f2fe', 
                     edgecolor='#0369a1', linewidth=2.5)
    arrow_props = dict(facecolor='#64748b', arrowstyle='->', linewidth=2)
    exclude_props = dict(boxstyle='round,pad=0.4', facecolor='#fee2e2', 
                         edgecolor='#dc2626', linewidth=2)
    
    def draw_box(x, y, title, count, subtitle="", is_excluded=False):
        text = f"{title}\n(n = {count})"
        if subtitle: text += f"\n{subtitle}"
        props = exclude_props if is_excluded else box_props
        ax.text(x, y, text, ha='center', va='center', fontsize=11, 
                bbox=props, family='sans-serif', weight='bold', color='#0f172a')

    # 1. Identificación
    draw_box(5, 9, "IDENTIFICACION", metrics.get('total', 0), 
             "Registros identificados en bases de datos")
    ax.annotate('', xy=(5, 8), xytext=(5, 8.5), arrowprops=arrow_props)

    # 2. Cribado
    draw_box(5, 7.3, "CRIBADO", metrics.get('after_filter', 0), 
             "Tras aplicar filtros temporales")

    excluded_filters = metrics.get('total', 0) - metrics.get('after_filter', 0)
    if excluded_filters > 0:
        ax.annotate('', xy=(7.5, 7.3), xytext=(6.3, 7.3), arrowprops=arrow_props)
        draw_box(8.5, 7.3, "Excluidos", excluded_filters, 
                 "Fuera de rango temporal", is_excluded=True)

    ax.annotate('', xy=(5, 6.2), xytext=(5, 6.9), arrowprops=arrow_props)

    # 3. Deduplicación
    draw_box(5, 5.6, "DEDUPLICACION", metrics.get('after_dedup', 0), 
             "Registros unicos")

    excluded_dups = metrics.get('after_filter', 0) - metrics.get('after_dedup', 0)
    if excluded_dups > 0:
        ax.annotate('', xy=(7.5, 5.6), xytext=(6.3, 5.6), arrowprops=arrow_props)
        draw_box(8.5, 5.6, "Excluidos", excluded_dups, 
                 "Duplicados detectados", is_excluded=True)

    ax.annotate('', xy=(5, 4.5), xytext=(5, 5.2), arrowprops=arrow_props)

    # 4. Elegibilidad
    draw_box(5, 3.9, "ELEGIBILIDAD", metrics.get('relevant', 0), 
             "Evaluados por relevancia (>=70%)")

    excluded_relevance = metrics.get('after_dedup', 0) - metrics.get('final_included', 0)
    ax.annotate('', xy=(7.5, 3.9), xytext=(6.3, 3.9), arrowprops=arrow_props)
    draw_box(8.5, 3.9, "Excluidos", excluded_relevance, 
             "Baja relevancia o criterios", is_excluded=True)

    ax.annotate('', xy=(5, 2.8), xytext=(5, 3.5), arrowprops=arrow_props)

    # 5. Inclusión Final
    final_props = dict(boxstyle='round,pad=0.5', facecolor='#d1fae5', 
                       edgecolor='#059669', linewidth=3)
    ax.text(5, 2.3, f"INCLUIDOS\n(n = {metrics.get('final_included', 0)})\n"
            "Estudios en sintesis narrativa",
            ha='center', va='center', fontsize=12, bbox=final_props,
            family='sans-serif', weight='bold', color='#065f46')

    # Título del diagrama
    ax.text(5, 9.7, "DIAGRAMA DE FLUJO PRISMA", ha='center', va='center',
            fontsize=14, weight='bold', color='#1e293b')

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    return filename

def download_article_figure(article):
    """
    Intenta descargar la primera figura relevante del artículo.
    Prioriza: pdf_url > doi > url. Si falla, retorna placeholder.
    """
    try:
        # NOTA: Esto es conceptual. En producción necesitarías:
        # 1. Un parser de PDFs que extraiga imágenes
        # 2. O usar APIs de publicadores (Elsevier, Springer, etc.)
        # 3. O scraping de páginas HTML del artículo
        
        # Por ahora, creamos un placeholder con metadata del artículo
        return create_placeholder_figure(article)
        
    except Exception as e:
        logging.warning(f"No se pudo obtener figura para {article.get('title', '')[:30]}: {e}")
        return create_placeholder_figure(article)

def create_placeholder_figure(article):
    """Crea figura placeholder con metadata del artículo."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis('off')
    
    # Información visual del artículo
    title = article.get('title', 'Artículo')[:80]
    authors = article.get('authors', [])
    if isinstance(authors, list):
        auth_str = ', '.join(authors[:2])
        if len(authors) > 2: auth_str += ' et al.'
    else:
        auth_str = str(authors)[:50]
    
    year = article.get('year', 'N/A')
    journal = article.get('journal', 'Journal')[:50]
    
    # Diseño
    ax.add_patch(patches.Rectangle((0.1, 0.1), 0.8, 0.8, 
                                    linewidth=2, edgecolor='#0369a1', 
                                    facecolor='#e0f2fe'))
    
    ax.text(0.5, 0.7, title, ha='center', va='center', 
            fontsize=11, weight='bold', wrap=True, color='#0f172a')
    ax.text(0.5, 0.5, f"{auth_str} ({year})", ha='center', va='center',
            fontsize=9, style='italic', color='#475569')
    ax.text(0.5, 0.35, journal, ha='center', va='center',
            fontsize=8, color='#64748b')
    ax.text(0.5, 0.15, "Relevancia: {:.0%}".format(article.get('similarity', 0)),
            ha='center', va='center', fontsize=10, weight='bold', color='#059669')
    
    filename = f"static/fig_{hash(article.get('title', ''))}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    return filename

def create_pdf_report(synthesis_text, metrics, articles, question, 
                      pdf_path="sintesis_prisma.pdf"):
    """
    Genera reporte PDF completo con:
    - Portada
    - Diagrama PRISMA
    - Síntesis Narrativa
    - Top 2 Artículos con figuras
    """
    try:
        pdf = PDFReport()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # ===== PORTADA =====
        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.set_text_color(15, 23, 42)
        pdf.ln(40)
        pdf.multi_cell(0, 12, 'REVISION SISTEMATICA\nMETODOLOGIA PRISMA', 0, 'C')
        
        pdf.ln(15)
        pdf.set_font('Arial', '', 12)
        pdf.set_text_color(71, 85, 105)
        question_clean = clean_text_for_pdf(question)
        pdf.multi_cell(0, 8, f'Pregunta de Investigacion:\n{question_clean}', 0, 'C')
        
        pdf.ln(30)
        pdf.set_font('Arial', 'B', 14)
        pdf.set_text_color(5, 150, 105)
        pdf.cell(0, 10, f'{metrics.get("final_included", 0)} Estudios Incluidos', 0, 1, 'C')
        
        pdf.ln(50)
        pdf.set_font('Arial', 'I', 10)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 10, 'Universidad Privada Antenor Orrego', 0, 1, 'C')
        pdf.cell(0, 10, 'Asistente de Revision Sistematica con IA', 0, 1, 'C')
        
        # ===== DIAGRAMA PRISMA =====
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.set_text_color(30, 41, 59)
        pdf.cell(0, 10, '1. Diagrama de Flujo PRISMA', 0, 1, 'L')
        pdf.ln(5)
        
        prisma_img = generate_prisma_diagram(metrics)
        pdf.image(prisma_img, x=25, w=160)
        pdf.ln(5)
        
        # ===== SÍNTESIS NARRATIVA =====
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '2. Sintesis Narrativa', 0, 1, 'L')
        pdf.ln(5)
        
        # Limpiar texto de Markdown y caracteres especiales
        clean_synth = clean_text_for_pdf(synthesis_text)
        clean_synth = clean_synth.replace('**', '').replace('##', '').replace('#', '')
        clean_synth = clean_synth.replace('*', '')
        
        pdf.set_font('Arial', '', 10)
        pdf.set_text_color(51, 65, 85)
        pdf.multi_cell(0, 6, clean_synth)
        
        # ===== TOP 2 ARTÍCULOS CON FIGURAS =====
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '3. Articulos Destacados', 0, 1, 'L')
        pdf.ln(5)
        
        # Ordenar por relevancia y tomar top 2
        top_articles = sorted(articles, key=lambda x: x.get('similarity', 0), reverse=True)[:2]
        
        for i, art in enumerate(top_articles, 1):
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(5, 150, 105)
            pdf.cell(0, 8, f'Articulo {i} (Relevancia: {art.get("similarity", 0)*100:.1f}%)', 0, 1, 'L')
            
            pdf.set_font('Arial', 'B', 11)
            pdf.set_text_color(30, 41, 59)
            title_clean = clean_text_for_pdf(art.get('title', ''))
            pdf.multi_cell(0, 6, title_clean)
            
            pdf.set_font('Arial', 'I', 9)
            pdf.set_text_color(100, 116, 139)
            authors = art.get('authors', [])
            if isinstance(authors, list):
                auth_str = ', '.join(authors[:3])
                if len(authors) > 3: auth_str += ' et al.'
            else:
                auth_str = str(authors)[:100]
            auth_str = clean_text_for_pdf(auth_str)
            pdf.cell(0, 5, f'{auth_str} ({art.get("year", "N/A")})', 0, 1, 'L')
            
            pdf.ln(3)
            
            # Intentar agregar figura del artículo
            try:
                fig_path = download_article_figure(art)
                if os.path.exists(fig_path):
                    pdf.image(fig_path, x=40, w=130)
                    pdf.ln(5)
            except Exception as e:
                logging.warning(f"No se pudo agregar figura: {e}")
            
            pdf.ln(5)
        
        # Guardar PDF
        pdf.output(pdf_path)
        logging.info(f"✅ PDF generado: {pdf_path}")
        return pdf_path
        
    except Exception as e:
        logging.error(f"❌ Error generando PDF: {e}")
        raise