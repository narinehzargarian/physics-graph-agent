"""The three data stores: Neo4j for connections, Elasticsearch for search, Redis for caching."""
import hashlib
import json
import os

import redis
from elasticsearch import Elasticsearch, helpers
from neo4j import GraphDatabase

INDEX = "physics-concepts"


class GraphStore:
    def __init__(self, uri=None, user=None, password=None):
        self.driver = GraphDatabase.driver(
            uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            auth=(user or os.environ.get("NEO4J_USER", "neo4j"), password or os.environ["NEO4J_PASSWORD"]),
        )

    def close(self):
        self.driver.close()

    def load(self, concepts, relations):
        """Safe to run repeatedly, because MERGE doesn't create duplicates."""
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT concept_key IF NOT EXISTS FOR (c:Concept) REQUIRE c.key IS UNIQUE")
            session.run(
                "UNWIND $rows AS row MERGE (c:Concept {key: row.key}) "
                "SET c.name = row.name, c.field = row.field, c.description = row.description",
                rows=concepts[["key", "name", "field", "description"]].to_dict("records"),
            )
            # Cypher can't take a relationship type as a parameter. The pipeline has already
            # restricted these names to capital letters and underscores.
            for relation, group in relations.groupby("relation"):
                session.run(
                    "UNWIND $rows AS row MATCH (a:Concept {key: row.source_key}), (b:Concept {key: row.target_key}) "
                    f"MERGE (a)-[:{relation}]->(b)",
                    rows=group.to_dict("records"),
                )

    def neighbors(self, key):
        query = (
            "MATCH (a:Concept {key: $key})-[r]-(b:Concept) "
            "RETURN a.name AS source, type(r) AS relation, b.name AS target, "
            "startNode(r) = a AS outgoing ORDER BY b.name LIMIT 12"
        )
        with self.driver.session() as session:
            return [
                f"{row['source']} {row['relation']} {row['target']}" if row["outgoing"]
                else f"{row['target']} {row['relation']} {row['source']}"
                for row in session.run(query, key=key)
            ]

    def shortest_path(self, key_a, key_b):
        """Facts connecting two concepts, each stated in its real direction."""
        query = (
            "MATCH p = shortestPath((a:Concept {key: $a})-[*..6]-(b:Concept {key: $b})) "
            "UNWIND relationships(p) AS r "
            "RETURN startNode(r).name AS source, type(r) AS relation, endNode(r).name AS target"
        )
        with self.driver.session() as session:
            return [f"{row['source']} {row['relation']} {row['target']}" for row in session.run(query, a=key_a, b=key_b)]


class SearchStore:
    def __init__(self, url=None):
        self.client = Elasticsearch(url or os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))

    def load(self, concepts):
        if self.client.indices.exists(index=INDEX):
            self.client.indices.delete(index=INDEX)
        self.client.indices.create(
            index=INDEX,
            mappings={"properties": {
                "key": {"type": "keyword"},
                # The English analyzer drops words like "what" and "best", which otherwise
                # fuzzy-match concept text and let off-topic questions through.
                "name": {"type": "text", "analyzer": "english"},
                "field": {"type": "keyword"},
                "description": {"type": "text", "analyzer": "english"},
            }},
        )
        helpers.bulk(self.client, (
            {"_index": INDEX, "_id": row.key, "key": row.key, "name": row.name,
             "field": row.field, "description": row.description}
            for row in concepts.itertuples()
        ), refresh=True)

    def search(self, text, size=4):
        """Ranks a match in the concept's name three times higher than one in its description."""
        result = self.client.search(
            index=INDEX, size=size,
            query={"multi_match": {"query": text, "fields": ["name^3", "description"], "fuzziness": "AUTO"}},
        )
        return [
            {"key": hit["_source"]["key"], "name": hit["_source"]["name"],
             "description": hit["_source"]["description"], "score": hit["_score"]}
            for hit in result["hits"]["hits"]
        ]


class AnswerCache:
    """Caches finished answers, so a repeated question skips the model."""

    def __init__(self, url=None, ttl_seconds=3600):
        self.client = redis.Redis.from_url(url or os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
        self.ttl = ttl_seconds

    @staticmethod
    def _key(question):
        normalized = " ".join(question.casefold().split())
        return "answer:" + hashlib.sha256(normalized.encode()).hexdigest()

    def get(self, question):
        cached = self.client.get(self._key(question))
        return json.loads(cached) if cached else None

    def put(self, question, answer):
        self.client.set(self._key(question), json.dumps(answer), ex=self.ttl)
