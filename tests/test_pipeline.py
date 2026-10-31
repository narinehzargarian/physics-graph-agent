import pandas as pd
import pytest

from agent import data_pipeline


def test_merges_duplicate_spellings_and_fills_missing_fields():
    raw = pd.DataFrame({
        "name": ["Plasma", " plasma ", "Photon", "Dangling"],
        "field": ["plasma physics", "Plasma Physics", None, "x"],
        "description": ["Ionized gas.", "Ionized gas.", "Light quantum.", None],
    })
    cleaned = data_pipeline.clean_concepts(raw)
    assert list(cleaned["name"]) == ["Plasma", "Photon"]
    assert list(cleaned["field"]) == ["Plasma Physics", "Unclassified"]


def test_drops_bad_relations_and_reports_them():
    concepts = data_pipeline.clean_concepts(pd.DataFrame({
        "name": ["A", "B"], "field": ["F", "F"], "description": ["a", "b"],
    }))
    raw = pd.DataFrame({
        "source": ["A", " a ", "A", "A"],
        "relation": ["uses", "USES", "USES", "USES"],
        "target": ["B", "B", "Missing", "A"],
    })
    relations, report = data_pipeline.clean_relations(raw, concepts)
    assert len(relations) == 1
    assert report == {"dangling": 1, "self_loops": 1, "duplicates": 1}


def test_rejects_relation_names_that_could_inject_cypher():
    concepts = data_pipeline.clean_concepts(pd.DataFrame({
        "name": ["A", "B"], "field": ["F", "F"], "description": ["a", "b"],
    }))
    raw = pd.DataFrame({"source": ["A"], "relation": ["USES]->(b) DETACH DELETE b //"], "target": ["B"]})
    with pytest.raises(ValueError):
        data_pipeline.clean_relations(raw, concepts)


def test_real_data_is_clean():
    concepts, relations, report = data_pipeline.load()
    assert concepts["key"].is_unique
    assert report["duplicate_concepts"] == 1
    assert report["dangling"] == 1
    assert report["self_loops"] == 1
    assert set(relations["source_key"]) <= set(concepts["key"])
    assert "Unclassified" in set(concepts["field"])


def test_field_summary_counts_add_up():
    concepts, relations, _ = data_pipeline.load()
    summary = data_pipeline.field_summary(concepts, relations)
    assert summary["concepts"].sum() == len(concepts)
    assert summary["outgoing_links"].sum() == len(relations)
