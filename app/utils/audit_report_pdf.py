from fpdf import FPDF
import datetime

class AuditReportPDF(FPDF):
    def __init__(self, question):
        super().__init__()
        self.question = self.clean_text(question)
        self.set_auto_page_break(auto=True, margin=15)

    def clean_text(self, text):
        """Limpia caracteres Unicode no soportados por las fuentes estándar de PDF."""
        if not text: return ""
        if not isinstance(text, str): text = str(text)
        
        replacements = {
            '\u2014': '-', # em dash
            '\u2013': '-', # en dash
            '\u201c': '"', # smart quotes
            '\u201d': '"',
            '\u2018': "'",
            '\u2019': "'",
            '\u2026': '...', # ellipsis
            '\u00a9': '(c)',
            '\u00ae': '(r)',
            '\u2122': '(tm)',
            '\u2022': '*', # bullet
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        # Forzar a latin-1 para compatibilidad con Helvetica, reemplazando lo desconocido con '?'
        return text.encode('latin-1', 'replace').decode('latin-1')

    def header(self):
        # Logo placeholder or Title
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(30, 41, 59) # Slate 800
        self.cell(0, 10, 'Reporte de Auditoria - PRISMA Assistant', 0, 1, 'L')
        
        self.set_font('Helvetica', '', 9)
        self.set_text_color(100, 116, 139) # Slate 500
        self.cell(0, 5, f'Generado el: {datetime.datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'L')
        self.ln(5)
        
        # Research Question Box
        self.set_fill_color(248, 250, 252)
        self.set_draw_color(226, 232, 240)
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(30, 41, 59)
        self.cell(0, 8, ' Pregunta de Investigacion:', 'TLR', 1, 'L', fill=True)
        self.set_font('Helvetica', 'I', 9)
        self.multi_cell(0, 6, f' "{self.question}"', 'BLR', 'L', fill=True)
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 10, f'Pagina {self.page_no()} | PrismaAssistant Audit System', 0, 0, 'C')

    def add_stats_summary(self, raw, excluded, final):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(15, 23, 42)
        self.cell(0, 10, 'Resumen de Flujo de Identificacion', 0, 1, 'L')
        
        # Table of stats
        self.set_font('Helvetica', '', 10)
        col_width = self.epw / 3
        
        self.set_fill_color(241, 245, 249)
        self.cell(col_width, 10, ' Total Inicial', 1, 0, 'C', fill=True)
        self.cell(col_width, 10, ' Excluidos (Filtro)', 1, 0, 'C', fill=True)
        self.cell(col_width, 10, ' Candidatos Finales', 1, 1, 'C', fill=True)
        
        self.set_font('Helvetica', 'B', 14)
        self.cell(col_width, 12, str(raw), 1, 0, 'C')
        self.set_text_color(220, 38, 38)
        self.cell(col_width, 12, str(excluded), 1, 0, 'C')
        self.set_text_color(22, 163, 74)
        self.cell(col_width, 12, str(final), 1, 1, 'C')
        self.ln(10)

    def add_article_entry(self, art, index):
        # 1. Preparar datos y limpiar texto
        title = self.clean_text(art.get('title', 'Sin titulo'))
        year = art.get('year', 'N/D')
        venue = self.clean_text(art.get('journal') or art.get('venue', 'N/D'))
        matched = art.get('_concepts_matched_list', [])
        missing = art.get('_concepts_missing_list', [])
        
        # 2. Control de salto de página
        if self.get_y() > 220:
            self.add_page()

        margin_x = 10
        width = 190
        
        # --- ENCABEZADO UNIFICADO (#index | Titulo) ---
        self.set_x(margin_x)
        self.set_font('Helvetica', 'B', 10)
        self.set_fill_color(241, 245, 249) # Azul grisáceo muy claro
        
        # Combinamos para que el salto de linea no rompa el diseño
        full_header = f'  #{index}  |  {title}'
        self.set_text_color(30, 58, 138) # Azul profundo
        self.multi_cell(width, 7, full_header, 'TLRB', 'L', fill=True)
        
        # --- METADATOS ---
        self.set_x(margin_x)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(100, 116, 139)
        self.multi_cell(width, 5, f' Sede/Revista: {venue} ({year})', 'LR', 'L')
        
        # --- ESTADO DE EXCLUSION ---
        self.set_x(margin_x)
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(185, 28, 28) # Rojo
        status_line = f' EXCLUIDO - Filtro Tematico | Coincidencia: {art.get("_concepts_matched_count", 0)}/{art.get("_concepts_total_count", 0)}'
        self.cell(width, 6, status_line, 'LR', 1, 'L')
        
        # --- EVIDENCIA (COLORES DIFERENCIADOS) ---
        self.set_x(margin_x)
        self.set_font('Helvetica', '', 7)
        
        # Parte 1: Presentes (Verde)
        if matched:
            self.set_x(margin_x)
            self.set_text_color(21, 128, 61) # Verde esmeralda
            txt = " PRESENTES: " + ", ".join(matched)
            self.multi_cell(width, 4, self.clean_text(txt), 'LR', 'L')
            
        # Parte 2: Ausentes (Rojo suave)
        if missing:
            self.set_x(margin_x)
            self.set_text_color(153, 27, 27) # Rojo coral
            txt = " AUSENTES: " + ", ".join(missing)
            self.multi_cell(width, 4, self.clean_text(txt), 'LR', 'L')
        
        # Borde inferior
        self.set_x(margin_x)
        self.cell(width, 1, '', 'B', 1)
        
        self.ln(5)

def generate_audit_pdf(session_data):
    question = session_data.get('question', 'Investigacion')
    articles = session_data.get('articles', [])
    excluded = session_data.get('excluded_articles', [])
    
    pdf = AuditReportPDF(question)
    pdf.add_page()
    
    raw_total = session_data.get('raw_count', len(articles) + len(excluded))
    pdf.add_stats_summary(raw_total, len(excluded), len(articles))
    
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 10, 'Detalle de Articulos Excluidos (Evidencia Conceptual)', 0, 1, 'L')
    pdf.ln(2)
    
    # Generar listado completo sin limites para transparencia total
    for i, art in enumerate(excluded, 1):
        pdf.add_article_entry(art, i)
            
    return pdf.output()
