import os
import subprocess
import sys
from pathlib import Path


def test_database_import_works_from_repository_root():
    project_root = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)

    result = subprocess.run(
        [sys.executable, "-c", "from database import DatasetDatabase; print(DatasetDatabase.__name__)"],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "DatasetDatabase"
