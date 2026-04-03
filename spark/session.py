from __future__ import annotations

import os
import subprocess
from pathlib import Path

from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_java_home() -> str | None:
    current_java_home = os.environ.get("JAVA_HOME")
    if current_java_home:
        return current_java_home

    if os.name != "posix":
        return None

    java_home_cmd = ["/usr/libexec/java_home", "-v", "21"]
    try:
        result = subprocess.run(
            java_home_cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    resolved = result.stdout.strip()
    return resolved or None


def build_spark_session(app_name: str = "ReadingAnalyticsSpark") -> SparkSession:
    warehouse_dir = PROJECT_ROOT / "data" / "warehouse" / "spark_warehouse"
    warehouse_dir.mkdir(parents=True, exist_ok=True)

    java_home = _resolve_java_home()
    if java_home:
        os.environ.setdefault("JAVA_HOME", java_home)
        os.environ["PATH"] = f"{java_home}/bin:{os.environ['PATH']}"

    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    return (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.sql.warehouse.dir", str(warehouse_dir))
        .getOrCreate()
    )
