import pandas as pd
df = pd.read_csv("evaluation/analisis_articulos_gold_standard.csv")
print("Title | Score | Passed | RawSim | DomRel | Fuzzy | Reason")
print("="*100)
for idx, row in df.iterrows():
    print(f"{row['title'][:40]}... | {row['score']:.4f} | {row['passed']} | {row['raw_similarity']:.4f} | {row['domain_relevance']:.2f} | {row['fuzzy_score']:.2f} | {row['reason']}")
