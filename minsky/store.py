from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa

TABLE_NAME = "chunks"


def _schema(vector_dim: int) -> pa.Schema:
    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("project", pa.string()),
            pa.field("path", pa.string()),
            pa.field("kind", pa.string()),
            pa.field("content", pa.string()),
            pa.field("mtime", pa.float64()),
            pa.field("start_line", pa.int64()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
        ]
    )


def _escape(value: str) -> str:
    return value.replace("'", "''")


class Store:
    def __init__(self, db_path: Path, vector_dim: int):
        db_path.mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(str(db_path))
        self.vector_dim = vector_dim
        if TABLE_NAME in self.db.table_names():
            self.table = self.db.open_table(TABLE_NAME)
        else:
            self.table = self.db.create_table(TABLE_NAME, schema=_schema(vector_dim))

    def delete_path(self, project: str, path: str) -> None:
        self.table.delete(f"project = '{_escape(project)}' AND path = '{_escape(path)}'")

    def delete_paths(self, project: str, paths: list[str]) -> None:
        if not paths:
            return
        in_list = ", ".join(f"'{_escape(p)}'" for p in paths)
        self.table.delete(f"project = '{_escape(project)}' AND path IN ({in_list})")

    def delete_project(self, project: str) -> None:
        self.table.delete(f"project = '{_escape(project)}'")

    def add(self, rows: list[dict[str, Any]]) -> None:
        if rows:
            self.table.add(rows)

    def search(
        self,
        vector: list[float],
        top_k: int = 8,
        project: str | None = None,
        kind: str | None = None,
    ) -> list[dict[str, Any]]:
        query = self.table.search(vector).limit(top_k)
        clauses = []
        if project:
            clauses.append(f"project = '{_escape(project)}'")
        if kind:
            clauses.append(f"kind = '{_escape(kind)}'")
        if clauses:
            query = query.where(" AND ".join(clauses))
        results = query.to_list()
        for r in results:
            r.pop("vector", None)
        return results

    def optimize(self) -> None:
        """Compact fragments and drop tombstoned old versions to reclaim disk space."""
        self.table.optimize(cleanup_older_than=timedelta(seconds=0), delete_unverified=True)

    def list_projects(self) -> list[str]:
        if self.table.count_rows() == 0:
            return []
        tbl = self.table.to_lance().to_table(columns=["project"])
        return sorted(set(tbl.column("project").to_pylist()))
