from typing import Optional
import psycopg2
from sqlglot import expressions as exp

class SelectivityProvider:
    def __init__(
        self,
        db_name: Optional[str] = None,
        host: Optional[str] = "localhost",
        port: Optional[int] = 5432,
        user: Optional[str] = "root",
        password: Optional[str] = "",
        timeout_ms: int = 5000,
    ):
        self.db_name = f"{db_name}_10_1"
        self.host = host or "localhost"
        self.port = int(port or 5432)
        self.user = user or "root"
        self.password = password or ""
        self.timeout_ms = int(timeout_ms)

        self.conn = psycopg2.connect(
            dbname=self.db_name,
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
        )
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        self.cur.execute(f"SET statement_timeout = {int(self.timeout_ms)};")

    @classmethod
    def from_env(cls, db_name: str, default_selectivity: float = 0.5, **kwargs):
        return cls(
            default_selectivity=default_selectivity,
            db_name=f"{db_name}_10_1",
            host="localhost",
            port=5432,
            user="root",
            password="",
            **kwargs,
        )

    def _predicate_sql(self, predicate: exp.Expression) -> str:
        try:
            return predicate.sql(dialect="postgres")
        except Exception:
            return str(predicate)

    def _explain_rows(self, sql: str) -> Optional[int]:
        self.cur.execute("EXPLAIN " + sql)
        row = self.cur.fetchall()[0][0]
        if "rows=" in row:
            part = row.split("rows=")[-1].split(" ")[0]
            return int(part)
        return None

    def estimate_selectivity(self, table: str, column: str, predicate: Optional[exp.Expression]) -> float:
        t = table.strip()
        pred_sql = self._predicate_sql(predicate)

        total_sql = f"SELECT * FROM \"{t}\";"
        select_sql = f"SELECT * FROM \"{t}\" WHERE {pred_sql};"

        total_rows = self._explain_rows(total_sql)
        select_rows = self._explain_rows(select_sql)

        p = float(select_rows) / float(total_rows)
        p = max(1e-12, min(1.0, p))
        return p