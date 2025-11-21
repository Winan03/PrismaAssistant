import requests
import config
import logging
import json
import re
from typing import Dict, List, Optional
import time

# NUEVA IMPORTACIÓN: Necesitamos la función de embedding y la colección de ChromaDB
from modules.screening import get_embedding 
from modules.database import ensure_collection

class PRISMAScorer:
    """
    Evalúa artículos con criterios PRISMA académicos y genera un score real (0-100%).
    
    Criterios evaluados:
    1. Población relevante (20 puntos)
    2. Intervención/Exposición relevante (20 puntos)
    3. Outcomes/Resultados relevantes (20 puntos)
    4. Diseño del estudio apropiado (15 puntos)
    5. Metodología clara (15 puntos)
    6. Año de publicación dentro del rango (10 puntos)
    
    TOTAL: 100 puntos
    
    Threshold recomendado: ≥70 puntos = INCLUIR
    """
    
    def __init__(self):
        self.criteria_weights = {
            "population": 20,
            "intervention": 20,
            "outcomes": 20,
            "study_design": 15,
            "methodology": 15,
            "recency": 10
        }
        
        # Cache para evitar re-evaluar el mismo artículo
        self.cache = {}
    
    def evaluate_article(self, article: Dict, research_question: str, 
                         required_population: List[str] = None,
                         required_outcomes: List[str] = None,
                         acceptable_study_designs: List[str] = None,
                         year_range: tuple = None) -> Dict:
        """
        Evalúa un artículo contra criterios PRISMA específicos.
        """
        title = article.get('title', '')
        
        # Cache check
        cache_key = f"{title}_{research_question}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Extraer valores por defecto de la pregunta si no se especifican
        if required_population is None:
            required_population = self._extract_population_from_question(research_question)
        
        if required_outcomes is None:
            required_outcomes = self._extract_outcomes_from_question(research_question)
        
        if acceptable_study_designs is None:
            acceptable_study_designs = ["randomized controlled trial", "systematic review", 
                                       "meta-analysis", "cohort study", "case-control study", 
                                       "cross-sectional", "observational", "pilot study",
                                       "performance evaluation", "validation study"]
        
        # ✅ CORREGIDO: Si no se pasa year_range, usar los últimos 5 años
        if year_range is None:
            from datetime import datetime
            current_year = datetime.now().year
            year_range = (current_year - 4, current_year)
        
        # Evaluar cada criterio
        scores = {}
        justifications = []
        
        # 1. Población (20 puntos)
        pop_score, pop_just = self._evaluate_population(article, required_population)
        scores["population"] = pop_score
        justifications.append(f"**Población ({pop_score}/20):** {pop_just}")
        
        # 2. Intervención (20 puntos)
        int_score, int_just = self._evaluate_intervention(article, research_question)
        scores["intervention"] = int_score
        justifications.append(f"**Intervención ({int_score}/20):** {int_just}")
        
        # 3. Outcomes (20 puntos)
        out_score, out_just = self._evaluate_outcomes(article, required_outcomes)
        scores["outcomes"] = out_score
        justifications.append(f"**Outcomes ({out_score}/20):** {out_just}")
        
        # 4. Diseño del estudio (15 puntos)
        design_score, design_just = self._evaluate_study_design(article, acceptable_study_designs)
        scores["study_design"] = design_score
        justifications.append(f"**Diseño ({design_score}/15):** {design_just}")
        
        # 5. Metodología (15 puntos)
        method_score, method_just = self._evaluate_methodology(article)
        scores["methodology"] = method_score
        justifications.append(f"**Metodología ({method_score}/15):** {method_just}")
        
        # 6. Recency (10 puntos)
        recency_score, recency_just = self._evaluate_recency(article, year_range)
        scores["recency"] = recency_score
        justifications.append(f"**Actualidad ({recency_score}/10):** {recency_just}")
        
        # Calcular total
        total_score = sum(scores.values())
        passes_threshold = total_score >= 70  # ✅ CAMBIO: Threshold más realista
        
        result = {
            "total_score": total_score,
            "passes_threshold": passes_threshold,
            "criteria_scores": scores,
            "justification": "\n".join(justifications),
            "recommendation": "✅ INCLUIR" if passes_threshold else "❌ EXCLUIR"
        }
        
        # Guardar en cache
        self.cache[cache_key] = result
        
        return result
    
    # ==========================
    # Evaluadores por criterio
    # ==========================
    
    def _evaluate_population(self, article: Dict, required_terms: List[str]) -> tuple:
        """Evalúa si la población del estudio es relevante (20 puntos max)"""
        text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
        
        if not required_terms:
            return (15, "No se especificó población requerida - asumiendo parcialmente relevante")
        
        matches = [term for term in required_terms if term.lower() in text]
        
        if len(matches) >= 2:
            return (20, f"Población claramente relevante. Menciona: {', '.join(matches)}")
        elif len(matches) == 1:
            return (12, f"Población parcialmente relevante. Menciona: {matches[0]}")
        else:
            # Usar RAG Semántico para mejorar el score implícito
            implicit_score = self._check_implicit_match(article, "population", required_terms)
            if implicit_score > 0.7:
                return (15, f"Población implícitamente relevante (score RAG: {implicit_score:.2f})")
            return (5, "Población no claramente especificada")
    
    def _evaluate_intervention(self, article: Dict, research_question: str) -> tuple:
        """Evalúa si la intervención/exposición es relevante (20 puntos max)"""
        text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
        
        intervention_keywords = self._extract_intervention_keywords(research_question)
        
        matches = [kw for kw in intervention_keywords if kw.lower() in text]
        
        if len(matches) >= 2:
            return (20, f"Intervención directamente relevante. Menciona: {', '.join(matches[:3])}")
        elif len(matches) == 1:
            return (14, f"Intervención parcialmente relevante. Menciona: {matches[0]}")
        else:
            # Usar RAG Semántico para mejorar el score implícito
            implicit_score = self._check_implicit_match(article, "intervention", intervention_keywords)
            if implicit_score > 0.7:
                return (16, f"Intervención implícitamente relevante (score RAG: {implicit_score:.2f})")
            return (8, "Intervención no claramente especificada")
    
    def _evaluate_outcomes(self, article: Dict, required_outcomes: List[str]) -> tuple:
        """Evalúa si mide los outcomes relevantes (20 puntos max)"""
        text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
        
        if not required_outcomes:
            generic_outcomes = ["accuracy", "sensitivity", "specificity", "performance", 
                              "effectiveness", "efficacy", "diagnostic", "prediction"]
            matches = [o for o in generic_outcomes if o in text]
            if matches:
                return (15, f"Outcomes genéricos detectados: {', '.join(matches[:2])}")
            return (10, "Outcomes no especificados claramente")
        
        matches = [o for o in required_outcomes if o.lower() in text]
        
        if len(matches) >= 2:
            return (20, f"Outcomes claramente relevantes. Mide: {', '.join(matches)}")
        elif len(matches) == 1:
            return (13, f"Outcome parcialmente relevante. Mide: {matches[0]}")
        else:
            return (7, "Outcomes no alineados con la pregunta de investigación")
    
    # ✅ CORRECCIÓN: Scoring de diseño MÁS FLEXIBLE
    def _evaluate_study_design(self, article: Dict, acceptable_designs: List[str]) -> tuple:
        """Evalúa si el diseño del estudio es apropiado (15 puntos max)"""
        text = f"{article.get('title', '')} {article.get('abstract', '')}".lower()
        
        # Buscar diseños específicos
        detected_designs = [design for design in acceptable_designs if design.lower() in text]
        
        # ✅ CAMBIO: Niveles de evidencia más realistas para LLMs médicos
        if "randomized controlled trial" in detected_designs or "rct" in text:
            return (15, "Diseño de la más alta calidad: Ensayo Controlado Aleatorizado (RCT)")
        
        elif "systematic review" in detected_designs or "meta-analysis" in detected_designs:
            return (15, "Diseño de la más alta calidad: Revisión Sistemática/Meta-análisis")
        
        elif "cohort study" in detected_designs:
            return (14, "Diseño Fuerte: Estudio de Cohorte")
        
        elif "case-control" in text:
            return (12, "Diseño Bueno: Estudio de Caso-Control")
        
        elif "cross-sectional" in text or "observational" in text:
            return (11, "Diseño Aceptable: Estudio Transversal/Observacional")
        
        elif "pilot study" in text or "feasibility" in text:
            return (10, "Diseño Moderado: Estudio Piloto o Factibilidad")
        
        # ✅ NUEVO: Reconocer estudios de evaluación de rendimiento (comunes en LLMs)
        elif any(kw in text for kw in ["performance evaluation", "benchmark", "validation study", "diagnostic accuracy"]):
            return (12, "Diseño Bueno: Estudio de Evaluación de Rendimiento/Validación")
        
        # ✅ CAMBIO: Base más generosa (antes era 5, ahora 8)
        else:
            return (8, "Diseño del estudio no especificado claramente (Base generosa)")
    
    def _evaluate_methodology(self, article: Dict) -> tuple:
        """Evalúa si la metodología es clara y rigurosa (15 puntos max)"""
        abstract = article.get('abstract', '').lower()
        
        methodology_indicators = {
            "sample_size": ["n=", "participants", "patients", "subjects"],
            "statistical_analysis": ["p-value", "confidence interval", "statistical", "regression"],
            "data_collection": ["data collection", "measurement", "assessment", "evaluation"],
            "validation": ["validation", "validated", "reliability", "reproducibility"]
        }
        
        score = 0
        details = []
        
        for category, keywords in methodology_indicators.items():
            if any(kw in abstract for kw in keywords):
                score += 4
                details.append(category.replace("_", " "))
        
        if score >= 12:
            return (15, f"Metodología rigurosa. Incluye: {', '.join(details)}")
        elif score >= 8:
            return (12, f"Metodología adecuada. Incluye: {', '.join(details)}")
        elif score >= 4:
            return (8, f"Metodología básica. Incluye: {', '.join(details)}")
        else:
            return (5, "Metodología no suficientemente detallada en el abstract")
    
    # ✅ CORRECCIÓN: Recency más flexible
    def _evaluate_recency(self, article: Dict, year_range: tuple) -> tuple:
        """
        Evalúa si es suficientemente reciente (10 puntos max).
        ✅ CORREGIDO: Usa year_range del formulario, no hardcoded.
        """
        year = article.get('year', 0)
        min_year, max_year = year_range
        
        if year >= min_year and year <= max_year:
            return (10, f"✅ Dentro del rango temporal ({min_year}-{max_year}). Año: {year}")
        
        elif year == min_year - 1:  # ✅ NUEVO: Tolerancia de 1 año
            return (8, f"⚠️ Apenas fuera del rango ({year}), pero reciente. Tolerado")
        
        elif year > max_year:
            return (5, f"Fecha posterior al rango ({year}). Puntuación reducida")
        
        else:
            return (2, f"Fuera del rango temporal ({year} < {min_year}). Falla Recencia")
    
    # ==========================
    # Funciones auxiliares con IA/RAG
    # ==========================
    
    def _check_implicit_match(self, article: Dict, criterion: str, required_terms: List[str]) -> float:
        """
        Usa la búsqueda vectorial (RAG) en ChromaDB para verificar la relevancia semántica del artículo
        contra un criterio específico (Población, Outcomes) si falla el match directo.
        Retorna el score de similaridad (0.0-1.0).
        """
        try:
            # 1. Crear una query combinando el título del artículo y el criterio.
            query = f"{article.get('title', '')} {criterion} {', '.join(required_terms)}"
            
            # 2. Generar embedding de la consulta.
            query_emb = get_embedding(query)
            if query_emb is None:
                return 0.5

            collection = ensure_collection()
            
            # 3. Buscar documentos similares en la base de datos (RAG)
            results = collection.query(
                query_embeddings=[query_emb.tolist()],
                n_results=1,
                include=["documents", "distances"]
            )

            if results and results.get("distances") and len(results["distances"][0]) > 0:
                distance = results["distances"][0][0]
                # El score es 1.0 - distance. Si la distancia es pequeña, el score es alto.
                score = 1.0 - distance
                return score 
            
        except Exception as e:
            # Si falla la conexión o el RAG, caemos en el fallback neutral
            logging.warning(f"⚠️ Error en _check_implicit_match (RAG): {e}")
        
        return 0.5  # Neutral fallback, asumiendo un match decente si ya pasó el screening semántico
    
    def _extract_population_from_question(self, question: str) -> List[str]:
        """Extrae términos de población de la pregunta"""
        population_terms = []
        medical_populations = ["patients", "adults", "children", "elderly", "diabetes", 
                              "cancer", "cardiovascular", "pregnant", "newborn", "clinical"]
        question_lower = question.lower()
        for term in medical_populations:
            if term in question_lower:
                population_terms.append(term)
        return population_terms if population_terms else ["patients", "clinical"]
    
    def _extract_outcomes_from_question(self, question: str) -> List[str]:
        """Extrae outcomes esperados de la pregunta"""
        outcome_keywords = []
        common_outcomes = ["diagnosis", "diagnostic", "accuracy", "sensitivity", 
                          "specificity", "treatment", "prognosis", "mortality", 
                          "efficacy", "effectiveness", "performance"]
        question_lower = question.lower()
        for outcome in common_outcomes:
            if outcome in question_lower:
                outcome_keywords.append(outcome)
        return outcome_keywords if outcome_keywords else ["diagnosis", "accuracy"]
    
    def _extract_intervention_keywords(self, question: str) -> List[str]:
        """Extrae keywords de intervención de la pregunta"""
        keywords = []
        try:
            prompt = f"""Extract the main intervention/technology keywords from this research question.
Output ONLY a comma-separated list of 3-5 keywords.

Question: {question}

Example output: machine learning, deep learning, artificial intelligence"""

            headers = {
                "Authorization": f"Bearer {config.GITHUB_MODELS_TOKEN}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": config.PROMPT_GENERATION_MODEL,
                "messages": [
                    {"role": "system", "content": "Output only comma-separated keywords."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 50
            }
            
            response = requests.post(
                f"{config.GITHUB_MODELS_ENDPOINT}/chat/completions",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                content = response.json()["choices"][0]["message"]["content"].strip()
                keywords = [k.strip() for k in content.split(',')]
            
        except Exception as e:
            logging.warning(f"⚠️ Error extrayendo keywords: {e}")
        
        if not keywords:
            fallback_terms = ["machine learning", "deep learning", "artificial intelligence", 
                            "neural network", "algorithm", "large language model", "llm"]
            keywords = [t for t in fallback_terms if t.lower() in question.lower()]
        
        return keywords if keywords else ["machine learning", "artificial intelligence"]


# ==========================
# Función principal de scoring en batch
# ==========================

def score_articles_with_prisma(articles: List[Dict], research_question: str,
                                 threshold: int = 70,  # ✅ CAMBIO: 70% por defecto
                                 **criteria_params) -> tuple:
    """
    Evalúa una lista de artículos con criterios PRISMA y filtra por threshold.
    ✅ CAMBIO: Threshold por defecto 70% (más realista para LLMs médicos)
    """
    scorer = PRISMAScorer()
    
    included = []
    excluded = []
    
    logging.info(f"📋 Evaluando {len(articles)} artículos con criterios PRISMA (threshold: {threshold}%)...")
    
    for i, article in enumerate(articles):
        try:
            evaluation = scorer.evaluate_article(article, research_question, **criteria_params)
            
            article["prisma_score"] = evaluation["total_score"]
            article["prisma_criteria"] = evaluation["criteria_scores"]
            article["prisma_justification"] = evaluation["justification"]
            article["prisma_recommendation"] = evaluation["recommendation"]
            
            if evaluation["passes_threshold"]:
                included.append(article)
            else:
                article["exclusion_reason"] = f"Score PRISMA insuficiente ({evaluation['total_score']}/100 < {threshold})"
                excluded.append(article)
            
            if (i + 1) % 10 == 0:
                logging.info(f"   Progreso: {i + 1}/{len(articles)} evaluados...")
        
        except Exception as e:
            logging.error(f"❌ Error evaluando artículo {article.get('title', '')[:50]}: {e}")
            # ✅ CAMBIO: Fallback más generoso (70 en vez de auto-incluir)
            article["prisma_score"] = 70
            article["prisma_justification"] = f"Error en evaluación: {str(e)}"
            included.append(article)
    
    included.sort(key=lambda x: x.get("prisma_score", 0), reverse=True)
    
    if included:
        scores = [a["prisma_score"] for a in included]
        logging.info(f"✅ Evaluación PRISMA completada:")
        logging.info(f"   - Incluidos (≥{threshold}%): {len(included)}")
        logging.info(f"   - Excluidos (<{threshold}%): {len(excluded)}")
        logging.info(f"   - Score promedio (incluidos): {sum(scores)/len(scores):.1f}/100")
        logging.info(f"   - Score más alto: {max(scores)}/100")
        logging.info(f"   - Score más bajo: {min(scores)}/100")
    
    return included, excluded