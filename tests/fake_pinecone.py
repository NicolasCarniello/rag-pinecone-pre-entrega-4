"""Pinecone falso en memoria con la misma interfaz que usa el proyecto (SDK v7).

Permite correr la ingesta, el PineconeVectorStore real de LangChain y la
evaluación sin cuenta ni red.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace


class FakeIndex:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.data: dict[str, dict[str, dict]] = {}  # namespace -> id -> {values, metadata}
        self.upsert_calls = 0
        self.config = SimpleNamespace(host="fake-index.svc.pinecone.io", api_key="fake")  # como el SDK real

    def upsert(self, vectors, namespace=""):
        self.upsert_calls += 1
        ns = self.data.setdefault(namespace, {})
        for v in vectors:
            if len(v["values"]) != self.dimension:
                raise ValueError(f"Vector dimension {len(v['values'])} does not match {self.dimension}")
            ns[v["id"]] = {"values": list(v["values"]), "metadata": copy.deepcopy(v["metadata"])}

    @staticmethod
    def _match(meta, flt):
        if not flt:
            return True
        for key, cond in flt.items():
            if isinstance(cond, dict):
                if "$eq" in cond and meta.get(key) != cond["$eq"]:
                    return False
                if "$in" in cond and meta.get(key) not in cond["$in"]:
                    return False
            elif meta.get(key) != cond:
                return False
        return True

    def query(self, vector, top_k=10, include_metadata=True, namespace="", filter=None, **_):
        def cos(a, b):
            na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
            return sum(x * y for x, y in zip(a, b)) / (na * nb or 1)

        items = [(vid, rec) for vid, rec in self.data.get(namespace, {}).items()
                 if self._match(rec["metadata"], filter)]
        scored = sorted(((cos(vector, rec["values"]), vid, rec) for vid, rec in items), reverse=True)
        return {"matches": [{"id": vid, "score": s, "metadata": copy.deepcopy(rec["metadata"])}
                            for s, vid, rec in scored[:top_k]]}

    def list(self, namespace="", prefix=None, limit=100):
        ids = sorted(i for i in self.data.get(namespace, {}) if prefix is None or i.startswith(prefix))
        for i in range(0, len(ids), limit):
            yield ids[i:i + limit]

    def fetch(self, ids, namespace=""):
        ns = self.data.get(namespace, {})
        return SimpleNamespace(vectors={
            i: SimpleNamespace(id=i, values=ns[i]["values"], metadata=copy.deepcopy(ns[i]["metadata"]))
            for i in ids if i in ns
        })

    def delete(self, ids=None, delete_all=False, namespace=""):
        if delete_all:
            if namespace not in self.data:
                raise KeyError("Namespace not found")
            self.data.pop(namespace)
            return
        for i in ids or []:
            self.data.get(namespace, {}).pop(i, None)

    def describe_index_stats(self):
        return SimpleNamespace(
            dimension=self.dimension,
            total_vector_count=sum(len(v) for v in self.data.values()),
            namespaces={ns: SimpleNamespace(vector_count=len(v)) for ns, v in self.data.items()},
        )


class FakePinecone:
    def __init__(self):
        self.indexes: dict[str, dict] = {}
        self.created: list[dict] = []

    def has_index(self, name):
        return name in self.indexes

    def create_index(self, name, dimension, metric, spec):
        self.created.append({"name": name, "dimension": dimension, "metric": metric, "spec": spec})
        self.indexes[name] = {"dimension": dimension, "metric": metric, "index": FakeIndex(dimension),
                              "ready_checks": 1}

    def describe_index(self, name):
        info = self.indexes[name]
        ready = info["ready_checks"] <= 0
        info["ready_checks"] -= 1  # el primer describe devuelve "no listo"
        return SimpleNamespace(dimension=info["dimension"], metric=info["metric"], status={"ready": ready})

    def Index(self, name):  # noqa: N802 (misma firma que el SDK)
        return self.indexes[name]["index"]
