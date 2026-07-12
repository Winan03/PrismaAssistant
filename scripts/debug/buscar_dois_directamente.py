import pandas as pd
import requests
import time
import os

df = pd.read_csv("evaluation/gold_standard.csv")
headers = {}
if "SEMANTIC_SCHOLAR_API_KEY" in os.environ:
    headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
else:
    # Intenta obtener de modules.core.config
    try:
        from modules.core.config import SEMANTIC_SCHOLAR_API_KEY
        if SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY
    except Exception as e:
        print("No se pudo cargar la API key de SS:", e)

results = []
for idx, row in df.iterrows():
    title = row["title"]
    doi = str(row["doi"]).strip()
    
    # Limpiar DOI
    doi_clean = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
    if not doi_clean or doi_clean == "nan":
        print(f"[{idx+1}] Sin DOI: {title[:50]}...")
        results.append({"idx": idx+1, "title": title, "doi": doi, "status": "Sin DOI"})
        continue
        
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi_clean}"
    params = {"fields": "title,year,abstract,authors,citationCount,openAccessPdf"}
    
    time.sleep(1.0) # Respetar rate limits
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            has_abstract = "abstract" in data and data["abstract"] is not None and len(data["abstract"]) >= 100
            print(f"[{idx+1}] ENCONTRADO por DOI: {data.get('title')[:50]}... | Abstract: {has_abstract} | Citaciones: {data.get('citationCount')}")
            results.append({
                "idx": idx+1,
                "title": title,
                "doi": doi,
                "status": "Encontrado",
                "has_abstract": has_abstract,
                "citations": data.get("citationCount", 0),
                "year": data.get("year", 0)
            })
        else:
            print(f"[{idx+1}] NO ENCONTRADO por DOI ({r.status_code}): {title[:50]}...")
            results.append({"idx": idx+1, "title": title, "doi": doi, "status": f"No Encontrado ({r.status_code})"})
    except Exception as e:
        print(f"[{idx+1}] Error consultando DOI: {e}")
        results.append({"idx": idx+1, "title": title, "doi": doi, "status": f"Error: {e}"})

df_res = pd.DataFrame(results)
df_res.to_csv("evaluation/auditoria_dois_ss.csv", index=False)
print("Auditoría completada y guardada en evaluation/auditoria_dois_ss.csv")
