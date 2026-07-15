import os
import json
import subprocess
import sys
import re
from utils import md5_of_text, read_json, write_json
from typing import Dict, Optional, Tuple
from config import DATASET_DIR
from udf_placement import build_pullup_sql


def _parse_subprocess_json(stdout: str, stderr: str, returncode: int) -> Optional[Dict]:
    if stdout and stdout.strip():
        try:
            result = json.loads(stdout.strip())
            if result.get("success"):
                return result
        except json.JSONDecodeError:
            pass

    if stderr:
        print(stderr, file=sys.stderr)
    if returncode != 0 or (stderr and "Traceback" in stderr):
        return None
    return None


class Executor:
    def __init__(self, db_name, timeout_sec: int = 300):
        self.db_name = db_name
        self.timeout_sec = timeout_sec
        self.db_file = os.path.join(DATASET_DIR, db_name, f"{db_name}_10_1.db")
        self.udf_file = os.path.join(DATASET_DIR, db_name, f"{db_name}_udf.sql")
        
        # Execution scripts.
        self.script_path = os.path.join(os.path.dirname(__file__), "exec_sql.py")
        self.baseline_script_path = os.path.join(os.path.dirname(__file__), "baseline_sql.py")
        self.pullup_script_path = os.path.join(os.path.dirname(__file__), "pullup_sql.py")

        # Load UDFs.
        self.udfs = {}
        self._load_udfs()

    def _load_udfs(self):
        with open(self.udf_file, 'r', encoding='utf-8') as infile:
            udf_content = infile.read()

        parts = re.split(r'(def\s+func_\w+\s*\()', udf_content)
        for i in range(1, len(parts), 2):
            func_name = parts[i].split()[1].split("(")[0]
            full_block = parts[i].strip() + parts[i+1].strip()
            self.udfs[func_name] = full_block

    def get_baseline_time(self, sql: str) -> float:
        q_md5 = md5_of_text(sql)
        if not os.path.exists(os.path.join("cache", self.db_name, q_md5)):
            os.makedirs(os.path.join("cache", self.db_name, q_md5))
        baseline_path = os.path.join("cache", self.db_name, q_md5, "baseline.json")
        if os.path.exists(baseline_path):
            return read_json(baseline_path).get("time")
        else:
            used_udfs = set()
            func_pattern = re.compile(r'\b(func_\w+)\b', re.IGNORECASE)
            matches = func_pattern.findall(sql)
            used_udfs.update(matches)
            used_udfs_content = "import numpy, math, re\n"
            for func_name in used_udfs:
                used_udfs_content += self.udfs[func_name.lower()] + "\n"

            process = subprocess.Popen(
                [sys.executable, self.baseline_script_path, self.db_file, used_udfs_content, sql],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
                
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_sec)
                result = _parse_subprocess_json(stdout, stderr, process.returncode)
                if result is None:
                    return -float('inf')
                write_json(baseline_path, result)
                return result["time"]
                    
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                write_json(baseline_path, {"time": float(self.timeout_sec), "success": False})
                return float(self.timeout_sec)
            
    def get_pullup_time(self, sql: str) -> float:
        q_md5 = md5_of_text(sql)
        qdir = os.path.join("cache", self.db_name, q_md5)
        if not os.path.exists(qdir):
            os.makedirs(qdir)
        pullup_path = os.path.join(qdir, "pullup.json")
        if os.path.exists(pullup_path):
            return read_json(pullup_path).get("time")

        rewritten_sql = build_pullup_sql(sql)

        used_udfs = set()
        func_pattern = re.compile(r'\b(func_\w+)\b', re.IGNORECASE)
        matches = func_pattern.findall(rewritten_sql)
        used_udfs.update(matches)
        used_udfs_content = "import numpy, math, re\n"
        for func_name in used_udfs:
            used_udfs_content += self.udfs[func_name.lower()] + "\n"

        process = subprocess.Popen(
            [sys.executable, self.pullup_script_path, self.db_file, used_udfs_content, rewritten_sql],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_sec)
            result = _parse_subprocess_json(stdout, stderr, process.returncode)
            if result is None:
                print(rewritten_sql)
                return -float('inf')
            t = result["time"]
            write_json(pullup_path, {"rewritten_sql": rewritten_sql, "time": t})
            return t
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            write_json(pullup_path, {"rewritten_sql": rewritten_sql, "time": float(self.timeout_sec)})
            return float(self.timeout_sec)

    def explain_analyze_time(self, sql: str) -> Tuple[float, str]:
        """
        Return (time_in_seconds, plan_text).
        """
        used_udfs = set()
        func_pattern = re.compile(r'\b(func_\w+)\b', re.IGNORECASE)
        matches = func_pattern.findall(sql)
        used_udfs.update(matches)
        used_udfs_content = "import numpy, math, re\n"
        for func_name in used_udfs:
            used_udfs_content += self.udfs[func_name.lower()] + "\n"

        process = subprocess.Popen(
            [sys.executable, self.script_path, self.db_file, used_udfs_content, sql],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
            
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_sec)
            result = _parse_subprocess_json(stdout, stderr, process.returncode)
            if result is None:
                return -float('inf'), "ERROR"
            return result["time"], result["plan"]
                
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return float(self.timeout_sec), "TIMEOUT"
