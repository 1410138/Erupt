import sys
import json
import duckdb
import datetime
import re

TOTAL_TIME_RE = re.compile(r"Total Time:\s*([0-9.]+)s", re.IGNORECASE)

def main():
    if len(sys.argv) != 4:
        # Raise an error for invalid arguments.
        raise ValueError("Usage: baseline_sql.py <db_file> <udf_content> <sql>")
    
    db_file = sys.argv[1]
    udf_content = sys.argv[2] 
    sql = sys.argv[3]
    
    # Connect to the database.
    con = duckdb.connect(db_file)
    
    # Load UDFs.
    exec(udf_content, {
        'db_conn': con, 
        'date': datetime.date, 
        'time': datetime.time,
        'VARCHAR': duckdb.typing.VARCHAR, 
        'INTEGER': duckdb.typing.INTEGER,
        'DOUBLE': duckdb.typing.DOUBLE, 
        'TIMESTAMP': duckdb.typing.TIMESTAMP,
        'DATE': duckdb.typing.DATE, 
        'TIME': duckdb.typing.TIME,
        'BOOLEAN': duckdb.typing.BOOLEAN, 
        'HUGEINT': duckdb.typing.HUGEINT,
        'FLOAT': duckdb.typing.FLOAT
    })
    
    # Run EXPLAIN ANALYZE.
    res = con.execute(f'EXPLAIN ANALYZE {sql}')
    rows = res.fetchall()
    texts = []
    for row in rows:
        texts.append(str(row[1]))
    raw = "\n".join(texts)
    
    # Extract execution time.
    m = TOTAL_TIME_RE.search(raw)
    t = float(m.group(1))
    
    # Print the result.
    result = {"time": t, "plan": raw, "success": True}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
