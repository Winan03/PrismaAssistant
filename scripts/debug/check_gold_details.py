import os
import pandas as pd

files = os.listdir("evaluation")
for file in files:
    if file.endswith(".csv"):
        path = os.path.join("evaluation", file)
        try:
            df = pd.read_csv(path)
            # Check if any title or abstract contains toy or juguete
            toy_matches = df[df.apply(lambda row: row.astype(str).str.contains('toy|juguete|preschool|ludic|child', case=False).any(), axis=1)]
            if len(toy_matches) > 0:
                print(f"File {file} has {len(toy_matches)} rows containing toy/juguete/preschool/ludic/child words out of {len(df)} total rows.")
            else:
                print(f"File {file} has 0 matches out of {len(df)} rows.")
        except Exception as e:
            print(f"Could not read {file}: {e}")
