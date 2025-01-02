from fastapi import UploadFile, File, HTTPException
import os
from joblib import load
from typing import Final
from pathlib import Path
import zipfile
import torch
import shutil
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2

from models.request_models import ModelLoadRequest
from models.response_models import ModelLoadResponse, PredictionResponse
from trainer_core.upload_dataset import is_image
from trainer_core.extraction import extract_features_predict

MODELS_PATH: Final[str] = Path(__file__).parent.parent / "models_train"
PREDICT_PATH: Final[str] = Path(__file__).parent.parent / "data_predict"

async def load_model_inference(request: ModelLoadRequest) -> dict:
    model_inference = {}
    model_id = request.id_model
    model_path = f"{MODELS_PATH}/{model_id}.joblib"
    if model_id in model_inference:
        return ModelLoadResponse(message=f"Модель '{model_id}' уже загружена.")
    model_inference[model_id] = load(model_path)
    if not model_inference:
        raise HTTPException(
            status_code=400, detail=f"Модель {request.id_model} не найдена")
    return model_inference
    # except FileNotFoundError:
    #     return False

async def predict_inference(model_inference: dict, file: UploadFile = File(...)) -> PredictionResponse:
    if not model_inference:
        raise HTTPException(
            status_code=400, detail="Загрузите модель в инференс")

    if not file.filename.endswith(".zip"):
        raise HTTPException(
            status_code=400, detail="Только ZIP архив необходимо загружать")
    os.makedirs(PREDICT_PATH, exist_ok=True)
    name_dir_predict = file.filename.replace(".zip", "")
    zip_path = os.path.join(PREDICT_PATH, file.filename)
    with open(zip_path, "wb") as temp:
        content = await file.read()
        temp.write(content)

    data_path = f"{PREDICT_PATH}/{name_dir_predict}/"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(PREDICT_PATH)
    os.remove(zip_path)

    image_names = [filename for filename in os.listdir(
        data_path) if is_image(f"{data_path}{filename}")]
    if not image_names:
        raise HTTPException(
            status_code=400, detail="В корне каталога нет изображений")
    transform = v2.Compose([
        v2.Resize(size=(224, 224)),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    feature_extractor.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor.to(device)

    image_paths = [os.path.join(data_path, filename) for filename in os.listdir(
        data_path) if is_image(f"{data_path}{filename}")]
    test_features = extract_features_predict(image_paths, transform, feature_extractor, device)
    model_id = next(iter(model_inference))
    data = model_inference[model_id]
    model = data["model"]
    labels = data["labels"]
    predictions = model.predict(test_features)
    predictions_emotions = {}
    for image_name, number in zip(image_names, predictions):
        label = [key for key, value in labels.items() if value == number][0]
        predictions_emotions[image_name] = label
    shutil.rmtree(PREDICT_PATH)
    
    return PredictionResponse(id=model_id, prediction=predictions_emotions)