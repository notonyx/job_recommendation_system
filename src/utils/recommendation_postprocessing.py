import pandas as pd

def unique_by_title(results):
    seen = set()
    unique_rows = []

    for _, row in results.iterrows():
        title = row.get("title", "")

        if title not in seen:
            seen.add(title)
            unique_rows.append(row)

    return pd.DataFrame(unique_rows).reset_index(drop=True)