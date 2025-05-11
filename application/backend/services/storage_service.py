import os
from typing import Final, List
from pathlib import Path
import shutil
import zipfile
from fastapi import UploadFile, File, HTTPException

DATASETS_PATH: Final[str] = Path(__file__).parent.parent / "datasets"
MODELS_PATH: Final[str] = Path(__file__).parent.parent / "models_train"


def delete_model(id_model: str) -> bool:
    try:
        os.remove(MODELS_PATH / f"{id_model}.joblib")
        return True
    except FileNotFoundError:
        return False


def delete_all_models():
    try:
        shutil.rmtree(MODELS_PATH)
    except FileNotFoundError:
        return


def list_models() -> List[str]:
    try:
        return [file.replace('.joblib', '') for file in os.listdir(MODELS_PATH) if file.endswith('.joblib')]
    except FileNotFoundError:
        return []


async def load_dataset(file: UploadFile = File(...)) -> str:
    try:
        if not file.filename.endswith(".zip"):
            raise HTTPException(
                status_code=400, detail="Только ZIP архив необходимо загружать")

        dataset_name = file.filename.replace(".zip", "")

        if not os.path.exists(DATASETS_PATH):
            os.makedirs(DATASETS_PATH)

        dir_datasets = [dir for dir in os.listdir(
            DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, dir))]
        if dataset_name in dir_datasets:
            raise HTTPException(
                status_code=400, detail="Датасет с таким именем уже есть")

        zip_path = os.path.join(DATASETS_PATH, file.filename)
        with open(zip_path, "wb") as temp:
            content = await file.read()
            temp.write(content)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(DATASETS_PATH)

        os.remove(zip_path)
    # remove artifacts of mac os zip files
        shutil.rmtree(DATASETS_PATH / "__MACOSX")
    except FileNotFoundError:
        return dataset_name


def delete_dataset(name_dataset: str) -> bool:
    try:
        shutil.rmtree(DATASETS_PATH / f"{name_dataset}")
        return True
    except FileNotFoundError:
        return False


def delete_all_datasets():
    try:
        shutil.rmtree(DATASETS_PATH)
    except FileNotFoundError:
        return


def list_datasets() -> List[str]:
    try:
        return [dir for dir in os.listdir(DATASETS_PATH) if os.path.isdir(os.path.join(DATASETS_PATH, dir))]
    except FileNotFoundError:
        return []
