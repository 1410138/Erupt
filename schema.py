from typing import Dict, List
import os
import json
from config import DATASET_DIR

class SchemaProvider:
    """
    { "tables": { "table1": ["c1","c2",...], "table2": [...] } }
    """
    def __init__(self, tables_to_columns: Dict[str, List[str]]):
        self._t2c = {t.lower(): [c for c in cols] for t, cols in tables_to_columns.items()}

    @classmethod
    def from_file(cls, db_name: str):
        path = os.path.join(DATASET_DIR, db_name, "schema.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return cls(obj.get("tables", {}))
        raise FileNotFoundError(f"Schema file not found: {path}")

    def table_names(self) -> List[str]:
        return list(self._t2c.keys())

    def columns_of(self, table: str) -> List[str]:
        return self._t2c.get(table.lower(), [])