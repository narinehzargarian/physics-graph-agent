"""Cleans the raw concept and relation files with Pandas before they reach the databases."""
import pathlib

import pandas as pd

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def _key(series):
    """Case and whitespace-insensitive key, for matching names that were typed differently."""
    return series.str.strip().str.casefold().str.replace(r"\s+", " ", regex=True)


def clean_concepts(raw):
    df = raw.copy()
    df["name"] = df["name"].str.strip()
    df["description"] = df["description"].str.strip()
    df = df.dropna(subset=["name", "description"])
    df["field"] = df["field"].fillna("Unclassified").str.strip().str.title()
    df["key"] = _key(df["name"])
    # Rows are ordered by trust, so the first spelling of a concept wins.
    df = df.drop_duplicates(subset="key", keep="first")
    return df.reset_index(drop=True)


def clean_relations(raw, concepts):
    df = raw.copy()
    df["source_key"] = _key(df["source"])
    df["target_key"] = _key(df["target"])
    df["relation"] = df["relation"].str.strip().str.upper()
    known = set(concepts["key"])
    report = {
        "dangling": int((~df["source_key"].isin(known) | ~df["target_key"].isin(known)).sum()),
        "self_loops": int((df["source_key"] == df["target_key"]).sum()),
    }
    df = df[df["source_key"].isin(known) & df["target_key"].isin(known)]
    df = df[df["source_key"] != df["target_key"]]
    before = len(df)
    df = df.drop_duplicates(subset=["source_key", "relation", "target_key"])
    report["duplicates"] = before - len(df)
    # Relation types become Neo4j labels in a query, so only safe names may pass.
    invalid = df[~df["relation"].str.fullmatch(r"[A-Z_]+")]
    if not invalid.empty:
        raise ValueError(f"invalid relation names: {sorted(invalid['relation'].unique())}")
    return df[["source_key", "relation", "target_key"]].reset_index(drop=True), report


def load(data_dir=DATA_DIR):
    """Returns (concepts, relations, report)."""
    raw_concepts = pd.read_csv(pathlib.Path(data_dir) / "concepts_raw.csv")
    raw_relations = pd.read_csv(pathlib.Path(data_dir) / "relations_raw.csv")
    concepts = clean_concepts(raw_concepts)
    relations, report = clean_relations(raw_relations, concepts)
    report["duplicate_concepts"] = len(raw_concepts) - len(concepts)
    return concepts, relations, report


def field_summary(concepts, relations):
    """Concepts and links per field, using a merge and group-by."""
    links = relations.merge(concepts[["key", "field"]], left_on="source_key", right_on="key")
    return (
        concepts.groupby("field").size().to_frame("concepts")
        .join(links.groupby("field").size().to_frame("outgoing_links"))
        .fillna(0).astype(int).sort_values("concepts", ascending=False)
    )

