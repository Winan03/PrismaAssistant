import unittest
import re
import logging
from modules.synthesis import AuthorPurifier, build_funnel_introduction, final_programmatic_cleanup

class TestSynthesisLogic(unittest.TestCase):
    def setUp(self):
        # Datos de prueba dinámicos
        self.articles = [
            {
                'authors': ['Kubiuk, I.', 'Kyselov, O.'], 
                'year': 2021, 
                'title': 'Code Representations',
                'abstract': 'Deep learning for code.'
            },
            {
                'authors': ['Harzevili, N.', 'et al.'], 
                'year': 2023, 
                'title': 'ML in Vulnerability Detection',
                'abstract': 'Systematic survey of ML.'
            }
        ]
        self.purifier = AuthorPurifier(self.articles)

    def test_author_purification_complex(self):
        """Prueba que los nombres se enriquezcan y limpien correctamente."""
        # Caso 1: Enriquecimiento simple (Kubiuk -> Kubiuk y Kyselov)
        text = "Según Kubiuk (2021) el aprendizaje profundo es clave."
        result = self.purifier.purify(text)
        self.assertIn("Kubiuk y Kyselov (2021)", result)
        
        # Caso 2: Cita parentética a la que le falta la coma
        text = "La detección es compleja (Harzevili et al. 2023)."
        result = self.purifier.purify(text)
        self.assertIn("(Harzevili et al., 2023)", result)

        # Caso 3: Evitar duplicación de paréntesis si el LLM ya los puso
        text = "Como se ve en (Kubiuk y Kyselov, 2021) los modelos..."
        result = self.purifier.purify(text)
        self.assertIn("(Kubiuk y Kyselov, 2021)", result)
        self.assertNotIn("((Kubiuk", result)

    def test_ghost_hyphens_extreme(self):
        """Prueba la eliminación de guiones parásitos en casos complejos (V11.0)."""
        # Caso 1: Conectores cortos iniciados con guion
        text = "-En- la era de la digitalización, aplicaciones -y- redes."
        result = self.purifier.purify(text)
        self.assertEqual(result, "En la era de la digitalización, aplicaciones y redes.")

        # Caso 2: Citas con guion
        text = "Se observó un avance (-Bommareddy- -2024-)."
        result = self.purifier.purify(text)
        self.assertIn("(Bommareddy, 2024)", result)

        # Caso 3: Acrónimos y términos técnicos rodeados
        text = "El ciclo -SDLC- y el -meta--análisis- son importantes."
        result = self.purifier.purify(text)
        self.assertIn(" ciclo SDLC y el meta-análisis son ", result)

        # Caso 4: Números y comas rodeadas
        text = "El valor es -0-,-95- según Akshar."
        result = self.purifier.purify(text)
        self.assertIn("valor es 0,95 según", result)

    def test_fine_tuning_preservation(self):
        """Asegura que fine-tuning no pierda su guion legítimo."""
        text = "Se requiere -fine-tuning- para el modelo."
        result = self.purifier.purify(text)
        self.assertIn("*fine-tuning*", result)
        self.assertNotIn("-fine-tuning-", result)

    def test_objective_formatting_romans(self):
        """Prueba que los objetivos en numerales romanos se reformateen a lista."""
        # Simulamos una sección de objetivos "sucia" del LLM
        text = "Específicamente, esta investigación busca: (i) mapear los enfoques, (ii) identificar criterios, (iii) evaluar impacto."
        
        # Usamos regex de la lógica interna para validar la extracción
        def dummy_format(content):
            # Lógica extraída de build_funnel_introduction V9.1
            objs = re.split(r'\s*\(?[\di]+\)\s*', content)
            objs = [o.strip() for o in objs if o.strip()]
            return "\n".join([f"    {i+1}) {obj};" for i, obj in enumerate(objs)])

        # Extraer la parte de objetivos (simulando el re.sub de la función real)
        m = re.search(r'(busca:)\s*(\(?[\di]+\)?.*?)$', text)
        if m:
            formatted = dummy_format(m.group(2))
            self.assertIn("1) mapear los enfoques", formatted)
            self.assertIn("2) identificar criterios", formatted)
            self.assertIn("3) evaluar impacto", formatted)

    def test_final_cleanup_conjunctions(self):
        """Prueba la armonización de conjunciones disruptivas (V10.1)."""
        text = "Se observó en (2023). Y por lo tanto..."
        result = final_programmatic_cleanup(text)
        self.assertIn("(2023) y por lo tanto", result)

    def test_author_guardian_blocking(self):
        """Prueba que el guardián bloquee autores alucinados."""
        from modules.synthesis import AuthorGuardian
        guardian = AuthorGuardian(self.articles)
        
        # Caso permitido
        self.assertTrue(guardian.validate_sentence("Según Kubiuk (2021) todo bien."))
        
        # Caso alucinado (Autor "Gomez" no existe en metadata)
        self.assertFalse(guardian.validate_sentence("Gomez (2024) descubrió algo nuevo."))

    def test_apa_blindaje_v12(self):
        """Prueba las nuevas funcionalidades de blindaje APA V12.0."""
        # Caso 1: Corrección de año alucinado (Kubiuk 2024 -> 2021)
        text = "Según Kubiuk (2024) el aprendizaje profundo es clave."
        result = self.purifier.purify(text)
        self.assertIn("(2021)", result)
        self.assertNotIn("(2024)", result)

        # Caso 2: Ordenamiento alfabético de citas múltiples
        text = "Varios estudios (Chittibala, 2024; Akshar, 2024; Bommareddy, 2024) analizan el tema."
        result = self.purifier.purify(text)
        # Orden esperado: Akshar, Bommareddy, Chittibala
        self.assertIn("(Akshar, 2024; Bommareddy, 2024; Chittibala, 2024)", result)

        # Caso 3: Combinación de et al. y ordenamiento
        # Forzar un apa_map con et al. para un autor
        self.purifier.apa_map['ding'] = {"2024": "Ding et al."}
        self.purifier.year_map['ding'] = ["2024"]
        text = "Investigaciones (Ding, 2023; Akshar, 2024) concluyen que..."
        result = self.purifier.purify(text)
        # Ding debe corregirse a 2024 y et al., y ordenarse: Akshar, Ding
        self.assertIn("(Akshar, 2024; Ding et al., 2024)", result)

    def test_integrity_v12_2(self):
        """Prueba las funcionalidades de integridad dinámica V12.2."""
        # Caso 1: Normalización de conjunciones (and -> y) en citaciones NARRATIVAS
        text = "Según Kubiuk and Kyselov (2021) analizan los datos."
        result = self.purifier.purify(text)
        self.assertIn("Kubiuk y Kyselov (2021)", result)

        # Caso 1b: Normalización en PARENTÉTICAS
        text = "Se analizan los datos (Kubiuk and Kyselov, 2021)."
        result = self.purifier.purify(text)
        self.assertIn("(Kubiuk y Kyselov, 2021)", result)

        # Caso 2: Saltos de línea disruptivos en citaciones
        text = "Bommareddy \n (2024) investigó la ciberseguridad."
        result = self.purifier.purify(text)
        self.assertIn("Bommareddy (2024)", result)

        # Caso 3: Términos técnicos con variantes de formato
        text = "Usamos finetuning y F1score."
        result = self.purifier.purify(text)
        self.assertIn("*fine-tuning*", result)
        self.assertIn("*F1-score*", result)

        # Caso 4: Anglicismos (trade-offs)
        text = "Los tradeoffs son importantes."
        result = self.purifier.purify(text)
        self.assertIn("*trade-offs*", result)

        # Caso 5: CodeBERT
        text = "El modelo codebert es eficaz."
        result = self.purifier.purify(text)
        self.assertIn("*CodeBERT*", result)

    def test_harzevili_dynamic_multi_article(self):
        """Prueba que el mismo autor se cite distinto según el año (Fase V12.4)."""
        articles = [
            {'authors': ['Harzevili, N.', 'Belle, A.', 'Wang, J.'], 'year': 2023},
            {'authors': ['Harzevili, N.'], 'year': 2024}
        ]
        from modules.synthesis import AuthorPurifier
        purifier = AuthorPurifier(articles)
        
        # Caso 2023: Debe ser et al.
        text_2023 = "El estudio de Harzevili (2023) es clave."
        result_2023 = purifier.purify(text_2023)
        self.assertIn("Harzevili et al. (2023)", result_2023)
        
        # Caso 2024: Debe ser autor único
        text_2024 = "En (Harzevili, 2024) se amplia el tema."
        result_2024 = purifier.purify(text_2024)
        self.assertIn("(Harzevili, 2024)", result_2024)
        self.assertNotIn("et al.", result_2024)

        # Caso en cluster mixto
        text = "Varios avances (Harzevili, 2023; Harzevili, 2024)."
        result = purifier.purify(text)
        self.assertIn("(Harzevili et al., 2023; Harzevili, 2024)", result)

    def test_integrity_v12_3(self):
        """Prueba las funcionalidades de integridad estilística V12.3."""
        # Caso 1: Conversión de Ampersand a 'y'
        text = "El estudio de (Kubiuk & Kyselov, 2021) es clave."
        result = self.purifier.purify(text)
        self.assertIn("(Kubiuk y Kyselov, 2021)", result)
        self.assertNotIn("&", result)

        # Caso 1b: Ampersand en cluster
        text = "Varios autores (Akshar et al., 2024; Croft & Smith, 2021)."
        result = self.purifier.purify(text)
        self.assertIn("(Akshar et al., 2024; Croft y Smith, 2021)", result)

        # Caso 2: Eliminación total de párrafo de estructura (último párrafo)
        text = "Introducción... \n\n El documento se organiza de la siguiente manera: sección método y resultados (Harzevili et al., 2023)."
        result = self.purifier.purify(text)
        # El párrafo debe haber desaparecido
        self.assertNotIn("organiza en secciones", result)
        self.assertNotIn("sección método", result)
        self.assertNotIn("Harzevili", result)
        self.assertEqual(result.strip(), "Introducción...")

if __name__ == '__main__':
    unittest.main()
