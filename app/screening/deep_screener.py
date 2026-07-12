import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import config
from app.llm.ai_model import LocalModel
from app.domain.eligibility_contract import generate_eligibility_contract, required_atom_labels

logger = logging.getLogger(__name__)

STRICT_CRITERIA = (
    "population",
    "intervention",
    "outcome",
    "context",
    "comparator",
    "study_design",
)
TEMPORAL_METADATA_RE = re.compile(
    r"\b(publication\s+(date|year)|published\s+(between|from|in)|"
    r"date\s+range|year\s+range|fecha|periodo|a(?:n|\u00f1)o|202[0-9]|201[0-9])\b",
    re.IGNORECASE,
)


def _is_yes(value: object) -> bool:
    return str(value or "").strip().upper() == "YES"


def _is_metadata_atom(atom: Dict) -> bool:
    text = " ".join(str(value or "") for value in [
        atom.get("label"),
        *(atom.get("query_terms") or []),
        *(atom.get("evidence_terms") or []),
        *(atom.get("terms") or []),
    ])
    return bool(TEMPORAL_METADATA_RE.search(text))


def _loads_llm_json(content: str) -> Dict:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return json.loads(content, strict=False)


def _contract_block(eligibility_contract: Optional[Dict]) -> str:
    if not eligibility_contract:
        return ""
    atoms = eligibility_contract.get("required_atoms", [])
    if not isinstance(atoms, list) or not atoms:
        return ""
    compact_atoms = []
    for atom in atoms:
        if not isinstance(atom, dict):
            continue
        if _is_metadata_atom(atom):
            continue
        if not bool(atom.get("required")):
            continue
        compact_atoms.append({
            "id": atom.get("id"),
            "category": atom.get("category"),
            "label": atom.get("label"),
            "required": bool(atom.get("required")),
            "must_be_explicit": bool(atom.get("must_be_explicit", True)),
            "not_acceptable": atom.get("not_acceptable", []),
        })
    if not compact_atoms:
        return ""
    return (
        "\n\nEligibility contract atoms. These are the ground truth for inclusion decisions:\n"
        f"{json.dumps(compact_atoms, ensure_ascii=False)}\n"
        "For each atom: evaluate whether the article text discusses this concept, "
        "including closely related terminology and subdomains within the same field. "
        "Mark 'met' when the text addresses the concept even if using different wording. "
        "Mark 'unclear' when the connection is indirect or debatable. "
        "Mark 'not_met' only when the concept is completely absent."
    )


def _criteria_mentions_review_exclusion(exclusion_criteria: str) -> bool:
    text = str(exclusion_criteria or "").lower()
    return bool(re.search(r"\b(review|survey|mapping|revisi[oó]n|estado del arte)\b", text))


def _study_design_instruction(is_abstract: bool, exclusion_criteria: str) -> str:
    text_source = "abstract" if is_abstract else "text"
    if _criteria_mentions_review_exclusion(exclusion_criteria):
        return (
            f"Based on the {text_source}, does this study satisfy the required study design, "
            "excluding literature reviews/surveys/mapping studies because the exclusion criteria say so?"
        )
    return (
        f"Based on the {text_source}, does this article satisfy any explicit study-design requirement? "
        "If no specific study design is required, answer YES. Literature reviews, surveys, and mapping "
        "studies may satisfy this criterion unless the exclusion criteria explicitly reject them."
    )


def _atom_guard(data: Dict, eligibility_contract: Optional[Dict]) -> Tuple[bool, float, List[str], List[Dict]]:
    if not eligibility_contract:
        return True, 1.0, [], []

    required_atoms = [
        atom for atom in eligibility_contract.get("required_atoms", [])
        if isinstance(atom, dict) and bool(atom.get("required")) and not _is_metadata_atom(atom)
    ]
    if not required_atoms:
        return True, 1.0, [], []

    raw_evaluations = data.get("atom_evaluations", [])
    evaluations = raw_evaluations if isinstance(raw_evaluations, list) else []
    by_id = {
        str(item.get("id") or "").strip(): item
        for item in evaluations
        if isinstance(item, dict)
    }

    missing: List[str] = []
    met_count = 0
    normalised: List[Dict] = []

    for atom in required_atoms:
        atom_id = str(atom.get("id") or "").strip()
        label = str(atom.get("label") or atom_id or "required atom")
        item = by_id.get(atom_id, {})
        status = str(item.get("status") or "").strip().lower()
        evidence = str(item.get("evidence") or "").strip()
        reason = str(item.get("reason") or "").strip()

        if status == "met" and evidence:
            met_count += 1
        else:
            missing.append(label)
            if status == "met" and not evidence:
                status = "unclear"
        normalised.append({
            "id": atom_id,
            "label": label,
            "category": atom.get("category", ""),
            "required": True,
            "status": status or "unclear",
            "evidence": evidence,
            "reason": reason,
        })

    score = met_count / len(required_atoms)
    return not missing, score, missing, normalised


def build_screening_prompt(
    cropped_text: str,
    original_question: str,
    paper_desc: str,
    study_design_text: str,
    criterios_str: str,
    abstract_instruction: str,
    title: str = "",
    eligibility_contract: Optional[Dict] = None,
) -> str:
    contract_block = _contract_block(eligibility_contract)
    if eligibility_contract and eligibility_contract.get("required_atoms"):
        atom_instruction = (
            "Step 4. Evaluate eligibility atoms. For each atom in the eligibility contract above, "
            "search the text for evidence. Mark 'met' when the concept is present (including closely "
            "related subdomains and terminology variants). Mark 'unclear' when the connection exists but "
            "is indirect. Mark 'not_met' only when the concept is completely absent."
        )
        decision_rules = (
            "- INCLUDE: ALL six criteria are YES AND every required eligibility atom is 'met' with evidence.\n"
            "- BACKGROUND_ONLY: The article is useful background or discusses the general topic, but at least "
            "one required criterion is NO or one required atom is not 'met'.\n"
            "- EXCLUDE: The article does not address the research question in a meaningful way."
        )
        json_format = (
            '{"population_yes_no":"YES/NO","intervention_yes_no":"YES/NO","outcome_yes_no":"YES/NO",'
            '"context_yes_no":"YES/NO","comparator_yes_no":"YES/NO","study_design_yes_no":"YES/NO",'
            '"atom_evaluations":[{"id":"atom_id","status":"met/unclear/not_met","evidence":"quote from text"}],'
            '"decision":"INCLUDE/BACKGROUND_ONLY/EXCLUDE","justification":"max 30 words explaining the decision"}'
        )
    else:
        atom_instruction = "Step 4. Output an empty list `[]` for the field 'atom_evaluations'."
        decision_rules = (
            "- INCLUDE: ALL six criteria are YES.\n"
            "- BACKGROUND_ONLY: The article is useful background or discusses the general topic, but at least "
            "one required criterion is NO.\n"
            "- EXCLUDE: The article does not address the research question in a meaningful way."
        )
        json_format = (
            '{"population_yes_no":"YES/NO","intervention_yes_no":"YES/NO","outcome_yes_no":"YES/NO",'
            '"context_yes_no":"YES/NO","comparator_yes_no":"YES/NO","study_design_yes_no":"YES/NO",'
            '"atom_evaluations":[],'
            '"decision":"INCLUDE/BACKGROUND_ONLY/EXCLUDE","justification":"max 30 words explaining the decision"}'
        )

    return f"""You are a systematic literature review expert performing PRISMA-compliant screening.

Research Question: "{original_question}"
{criterios_str}{contract_block}{abstract_instruction}

--- ARTICLE TEXT ---
Title: {title}
{cropped_text}
-----------------------

REASONING PROCESS — Evaluate step by step:

Step 1. Interpret the research question. What population, intervention, outcome, context, and comparator does it strictly require? Write down the exact requirement for each PICO element.

Step 2. Read the article text carefully. Identify what the article actually studies — its population, the technology/method it uses, what it measures, its setting, and its comparison baseline.

Step 3. Match each criterion. For each of the six criteria below, compare the requirement from Step 1 against the article evidence from Step 2. Answer YES only when there is explicit alignment. Do not substitute a similar-but-different concept — the match must be genuine, not merely adjacent.

CRITERIA — Answer YES or NO:
1. population_yes_no — Does the article study the population that the research question targets?
2. intervention_yes_no — Does the article employ/evaluate the intervention, technology, system, or method that the research question concerns, including all mandatory components?
3. outcome_yes_no — Does the article measure or analyze the dependent variable or outcome that the research question requires?
4. context_yes_no — Does the article match the required setting, environment, domain, or contextual constraint? If the question imposes no context requirement, answer YES.
5. comparator_yes_no — Does the article include the required comparator or control condition? If the question imposes no comparator, answer YES.
6. study_design_yes_no — {study_design_text}

{atom_instruction}

DECISION RULES:
{decision_rules}

Respond with ONLY this JSON:
{json_format}"""


def screen_cropped_pdf_with_llm(
    cropped_text: str,
    original_question: str,
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
    title: str = "",
    eligibility_contract: Optional[Dict] = None,
    judge_model: Optional[str] = None,
    skip_atom_guard: bool = False,
) -> Tuple[bool, float, Dict]:
    """
    Stage 4: evalua el texto recortado con criterios estrictos de elegibilidad.

    Returns:
        Tuple[List, float, Dict]: (passed, score_percent, details_dict)
    """
    if not cropped_text or len(cropped_text.strip()) < 800:
        return False, 0.0, {
            "error": "Texto insuficiente para screening",
            "population": "NO",
            "intervention": "NO",
            "outcome": "NO",
            "context": "NO",
            "comparator": "NO",
            "study_design": "NO",
            "decision": "EXCLUDE",
            "near_miss": False,
            "q1": "NO",
            "q2": "NO",
            "q3": "NO",
            "justification": "Texto extraido del PDF esta vacio o es demasiado corto.",
        }

    inc_list = [line.strip() for line in inclusion_criteria.splitlines() if line.strip()] if inclusion_criteria else []
    exc_list = [line.strip() for line in exclusion_criteria.splitlines() if line.strip()] if exclusion_criteria else []
    if skip_atom_guard:
        eligibility_contract = None
    elif eligibility_contract is None:
        eligibility_contract = generate_eligibility_contract(
            original_question,
            inclusion_criteria,
            exclusion_criteria,
        )

    criterios_str = ""
    if inc_list:
        criterios_str += "\nInclusion Criteria:\n" + "\n".join(f"- {criterion}" for criterion in inc_list)
    if exc_list:
        criterios_str += "\nExclusion Criteria:\n" + "\n".join(f"- {criterion}" for criterion in exc_list)

    is_abstract = len(cropped_text) < 4000
    paper_desc = "abstract of an academic article" if is_abstract else "cropped academic article sections"
    study_design_text = _study_design_instruction(is_abstract, exclusion_criteria)
    instruction_type = "an ABSTRACT" if is_abstract else "CROPPED SECTIONS of a paper"
    abstract_instruction = (
        f"\n\nIMPORTANT: You are evaluating {instruction_type}. Make a definitive YES/NO decision "
        "for each criterion based solely on the provided text. If the text does not explicitly support a "
        "required criterion, answer NO. Do not infer missing required details from broad topical language."
    )

    prompt = build_screening_prompt(
        cropped_text=cropped_text,
        original_question=original_question,
        paper_desc=paper_desc,
        study_design_text=study_design_text,
        criterios_str=criterios_str,
        abstract_instruction=abstract_instruction,
        title=title,
        eligibility_contract=eligibility_contract,
    )

    system_prompt = (
        "You are a conservative PRISMA screening assistant. "
        "Evaluate only explicit evidence in the provided text. "
        "Never include an article that fails any required criterion. "
        "Output only raw valid JSON."
    )

    try:
        lm = LocalModel.get_instance()
        content = lm.generate(
            instruction=prompt,
            input_text="",
            max_tokens=int(getattr(config, "STAGE4_LLM_MAX_TOKENS", 512)),
            system_prompt=system_prompt,
        )
        content = (content or "").strip()

        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\n```$", "", content)
            content = re.sub(r"\s*```$", "", content)
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            content = match.group(0)

        data = _loads_llm_json(content)
        criteria = {}
        for key in STRICT_CRITERIA:
            val = data.get(key) or data.get(f"{key}_yes_no") or "NO"
            criteria[key] = str(val).strip().upper()
        yes_count = sum(1 for value in criteria.values() if _is_yes(value))
        criteria_score = yes_count / len(STRICT_CRITERIA)
        atom_ok, atom_score, missing_atoms, atom_evaluations = _atom_guard(data, eligibility_contract)
        score = min(criteria_score, atom_score)
        passed = all(_is_yes(criteria[key]) for key in STRICT_CRITERIA) and atom_ok

        decision = str(data.get("decision", "INCLUDE" if passed else "EXCLUDE")).strip().upper()
        if missing_atoms and decision == "INCLUDE":
            decision = "BACKGROUND_ONLY" if score >= 0.5 else "EXCLUDE"
        if decision != "INCLUDE":
            passed = False

        overall_reason = str(data.get("reason") or data.get("justification") or "")
        details = {
            **criteria,
            "decision": decision if decision in {"INCLUDE", "EXCLUDE", "BACKGROUND_ONLY"} else ("INCLUDE" if passed else "EXCLUDE"),
            "near_miss": (not passed and score >= 0.5),
            "q1": criteria["population"],
            "q2": criteria["intervention"],
            "q3": criteria["study_design"],
            "justification": overall_reason,
            "score": round(score, 4),
            "atom_score": round(atom_score, 4),
            "criteria_score": round(criteria_score, 4),
            "atom_evaluations": atom_evaluations,
            "missing_required_atoms": missing_atoms,
            "required_atoms": required_atom_labels(eligibility_contract or {}),
        }
        return passed, round(score * 100.0, 1), details

    except Exception as exc:
        logger.error("DeepScreener error evaluando paper con LLM: %s", exc)
        return False, 0.0, {
            "error": f"Fallo en evaluacion con LLM: {str(exc)}",
            "population": "NO",
            "intervention": "NO",
            "outcome": "NO",
            "context": "NO",
            "comparator": "NO",
            "study_design": "NO",
            "decision": "EXCLUDE",
            "near_miss": False,
            "q1": "NO",
            "q2": "NO",
            "q3": "NO",
            "justification": "Error de comunicacion o parseo con el LLM.",
        }


def screen_candidates_cascade(
    articles: List[Dict],
    original_question: str,
    inclusion_criteria: str = "",
    exclusion_criteria: str = "",
    target_n: int = 100,
    eligibility_contract: Optional[Dict] = None,
    judge_model: Optional[str] = None,
    skip_atom_guard: bool = False,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Orquesta la evaluacion Stage 4 sobre los articulos con texto disponible.
    """
    passed: List[Dict] = []
    excluded: List[Dict] = []
    if skip_atom_guard:
        eligibility_contract = None
    elif eligibility_contract is None:
        eligibility_contract = generate_eligibility_contract(
            original_question,
            inclusion_criteria,
            exclusion_criteria,
        )

    active_model = judge_model or getattr(config, "OLLAMA_MODEL_JUDGE", "")
    logger.info(
        "[Stage 4] Iniciando cribado profundo estricto sobre %d candidatos con %s...",
        len(articles),
        active_model or "modelo configurado",
    )

    def process_article_vote(idx_a: Tuple[int, Dict]) -> Tuple[Dict, bool, float, str]:
        _idx, article = idx_a
        text = article.get("full_text") or article.get("abstract") or ""
        title = article.get("title", "")
        llm_start = time.perf_counter()
        
        ok, pct, details = screen_cropped_pdf_with_llm(
            text,
            original_question,
            inclusion_criteria=inclusion_criteria,
            exclusion_criteria=exclusion_criteria,
            title=title,
            eligibility_contract=eligibility_contract,
            judge_model=active_model,
            skip_atom_guard=skip_atom_guard,
        )
        
        decision = str(details.get("decision", "INCLUDE" if ok else "EXCLUDE")).strip().upper()
        near_miss = bool(details.get("near_miss", False)) or (decision == "BACKGROUND_ONLY")
        
        passed = ok
        is_borderline = False
        
        core_ok = (
            str(details.get("population") or details.get("population_yes_no") or "NO").strip().upper() == "YES" and
            str(details.get("intervention") or details.get("intervention_yes_no") or "NO").strip().upper() == "YES" and
            str(details.get("study_design") or details.get("study_design_yes_no") or "NO").strip().upper() == "YES"
        )
        
        if not passed and near_miss and core_ok:
            passed = True
            is_borderline = True
            
        article["stage4_llm_seconds"] = round(time.perf_counter() - llm_start, 3)
        article["deep_screening_passed"] = passed
        article["deep_screening_score"] = pct
        article["deep_screening_details"] = details
        article["similarity"] = round(pct / 100.0, 4)
        
        if passed:
            if is_borderline:
                article["evidence_bucket"] = "BACKGROUND_ONLY"
                article["exclusion_reason"] = "Borderline: Revisar a profundidad"
            else:
                article["evidence_bucket"] = "DIRECT_INCLUDED"
                article["exclusion_reason"] = None
            reason = ""
        else:
            article["evidence_bucket"] = decision
            missing_atoms = details.get("missing_required_atoms") or []
            if missing_atoms:
                reason = "Faltan atomos obligatorios: " + ", ".join(missing_atoms[:5])
            else:
                reason = details.get("justification") or f"No cumple todos los criterios estrictos (Puntaje: {pct}%)"
            article["exclusion_reason"] = reason
            
        article["_voting_include_votes"] = 1 if ok else 0
        return article, passed, pct, reason

    max_workers = max(1, int(getattr(config, "OLLAMA_STAGE4_CONCURRENCY", 5)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_article_vote, enumerate(articles)))

    for i, (article, ok, pct, reason) in enumerate(results):
        if ok:
            passed.append(article)
            logger.info("[Stage 4] [%d/%d] INCLUIDO (%d%%): %s", i + 1, len(articles), pct, article.get("title", "")[:50])
        else:
            excluded.append(article)
            logger.info("[Stage 4] [%d/%d] EXCLUIDO (%d%%): %s. Razon: %s", i + 1, len(articles), pct, article.get("title", "")[:50], reason[:80])

    passed.sort(key=lambda x: x.get("deep_screening_score", 0), reverse=True)
    excluded.sort(key=lambda x: x.get("deep_screening_score", 0), reverse=True)

    return passed, excluded
