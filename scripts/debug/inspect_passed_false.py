import pandas as pd
df = pd.read_csv("evaluation/analisis_articulos_gold_standard.csv")
print("=== ARTICULOS RECHAZADOS ===")
rejected = df[~df["passed"]]
for idx, row in rejected.iterrows():
    print(f"Index: {idx+1}")
    print(f"Title: {row['title']}")
    print(f"Score: {row['score']}")
    print(f"Passed: {row['passed']}")
    print(f"Raw Similarity: {row['raw_similarity']}")
    print(f"Domain Relevance: {row['domain_relevance']}")
    print(f"Fuzzy Score: {row['fuzzy_score']}")
    print(f"Reason: {row['reason']}")
    print("-" * 50)
