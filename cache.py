import os
import math
from typing import Optional
from utils import ensure_dir, md5_of_text, write_json, read_json

class PlanCache:
    def __init__(self, root_dir: str = "cache", db_name: str = "tpc_h"):
        self.root_dir = root_dir
        self.db_name = db_name

    def _qdir(self, query_sql: str) -> str:
        q_md5 = md5_of_text(query_sql)
        return os.path.join(self.root_dir, self.db_name, q_md5)

    def _pfile(self, query_sql: str, shape_str: str) -> str:
        p_md5 = md5_of_text(shape_str)
        return os.path.join(self._qdir(query_sql), f"{p_md5}.json")

    def _mfile(self, query_sql: str) -> str:
        return os.path.join(self._qdir(query_sql), "metadata.json")

    def save_plan(self, query_sql: str, shape_str: str, rewritten_sql: str, exec_time_sec: float) -> None:
        if math.isinf(exec_time_sec) and exec_time_sec < 0:
            return
        
        qdir = self._qdir(query_sql)
        ensure_dir(qdir)

        plan_path = self._pfile(query_sql, shape_str)
        write_json(plan_path, {
            "shape": shape_str,
            "execution_time": exec_time_sec,
            "rewritten_sql": rewritten_sql,
        })

        # metadata
        meta_path = self._mfile(query_sql)
        if os.path.exists(meta_path):
            meta = read_json(meta_path)
        else:
            meta = {
                "max_execution_time": None,
                "min_execution_time": None,
                "query_sql": query_sql,
                "plan_count": 0,
            }

        plan_files = [fn for fn in os.listdir(qdir) if fn.endswith(".json") and fn != "metadata.json" and fn != "baseline.json" and fn != "pullup.json"]
        meta["plan_count"] = len(plan_files)

        t = exec_time_sec
        if meta["max_execution_time"] is None or t > meta["max_execution_time"]:
            meta["max_execution_time"] = t
        if meta["min_execution_time"] is None or t < meta["min_execution_time"]:
            meta["min_execution_time"] = t

        write_json(meta_path, meta)

    def load_plan(self, query_sql: str, shape_str: str):
        path = self._pfile(query_sql, shape_str)
        if os.path.exists(path):
            return read_json(path)
        return None