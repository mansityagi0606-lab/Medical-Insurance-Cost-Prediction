import os
from pathlib import Path

project_name = "mlProject"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/utils.py",

    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",

    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/train_pipeline.py",
    f"src/{project_name}/pipeline/predict_pipeline.py",

    "notebooks/EDA.ipynb",
    "artifacts/.gitkeep",

    "app.py",
    "main.py",
    "params.yaml",
    "README.md"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    if filedir != Path("."):
        os.makedirs(filedir, exist_ok=True)

    if not filepath.exists():
        with open(filepath, "w"):
            pass
