import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import zipfile
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from typing import List
from PIL import Image
import joblib
from joblib import load
import torch
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.transforms import v2
from models.request_models import FitRequest, ModelLoadRequest, ModelRemoveRequest, DatasetRemoveRequest, EDARequest
from models.response_models import DatasetLoadResponse, FitResponse, ModelLoadResponse, PredictionResponse 
from models.response_models import ModelsListResponse, DatasetsListResponse, ModelRemoveResponse, EDAResponse
from models.response_models import DatasetRemoveResponse, AllModelsRemoveResponse, AllDatasetsRemoveResponse
from trainer_core.upload_dataset import upload_emotion_class, upload_dataset_inframe, is_image
from trainer_core.dataset import Dataset
from trainer_core.extraction import Extraction, extract_features_predict

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

process_status = {}
model_active = {}



@app.post("/load_dataset", response_model=DatasetLoadResponse, tags=["upload_file"])
async def load_dataset(file: UploadFile = File(...)):
    
    process_status["load_dataset"] = file.filename
    os.makedirs("datasets", exist_ok=True)
    dir_datasets = [dir for dir in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", dir))]
    if file.filename.replace(".zip", "") in dir_datasets:
        raise HTTPException(status_code=400, detail="Датасет с таким именем уже есть")
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Только ZIP архив необходимо загружать")
    
    zip_path = os.path.join("datasets", file.filename)
    with open(zip_path, "wb") as temp:
        content = await file.read()
        temp.write(content)
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("datasets")
        
    os.remove(zip_path)
    
    process_status["load_dataset"] = 0
    
    return DatasetLoadResponse(message=f"Dataset {file.filename} загружен!")

@app.post("/eda", response_model=EDAResponse, tags=["trainer"])
async def eda(requests: EDARequest):
    
    dataset = requests.name_dataset
    datasets_list = os.listdir("datasets")
    
    if dataset not in datasets_list:
        raise HTTPException(
            status_code=400, detail=f"Датасета '{dataset}' нет на сервере")
        
    list_emotion = upload_emotion_class(dataset)
     
    if list_emotion['emotions_count'] < 2:
        raise HTTPException(
            status_code=400, detail="Необходимо загрузить датасет с двумя классами или более")
        
    df = upload_dataset_inframe(dataset, list_emotion['emotions_list'])
    path_dataset = f"datasets/{dataset}"
    
    list_width = []
    list_height = []
    list_categ_RGB = []
    list_cat_R=[]
    list_cat_G=[]
    list_cat_B=[]
    
    for root, _, images in os.walk(path_dataset):
        for image in images:
            image_path = os.path.join(root, image)
            with Image.open(image_path) as img:
                width, height = img.size
                list_width.append(width)
                list_height.append(height)
                img_array = np.array(img)

                if (img_array[:, :, 0] == img_array[:, :, 1]).all() and (img_array[:, :, 1] == img_array[:, :, 2]).all():
                    list_categ_RGB.append('черно_белый')
                    list_cat_R.append(img_array[:, :, 0].flatten())
                    list_cat_G.append(img_array[:, :, 1].flatten())
                    list_cat_B.append(img_array[:, :, 2].flatten())

                elif img_array.ndim == 3 and img_array.shape[2] == 3:
                    list_categ_RGB.append('цветной')
                    list_cat_R.append(img_array[:, :, 0].flatten())
                    list_cat_G.append(img_array[:, :, 1].flatten())
                    list_cat_B.append(img_array[:, :, 2].flatten())
                    
    eda_dict = {
        "count_classes" : df['emotion'].nunique(),
        "count_images" : df.shape[0],
        "mean_R" : int(np.mean(list_cat_R)),
        "mean_G" : int(np.mean(list_cat_G)),
        "mean_B" : int(np.mean(list_cat_B)),
        "std_R" : int(np.std(list_cat_R)),
        "std_G" : int(np.std(list_cat_G)),
        "std_B" : int(np.std(list_cat_B)),
        "mean_width":  int(np.mean(list_width)),
        "mean_height": int(np.mean(list_height)),
        "min_width": int(np.min(list_width)),
        "min_height" : int(np.min(list_height)),
        "max_width": int(np.max(list_width)),
        "max_height": int(np.max(list_height)),
        "std_width" : int(np.std(list_width)),
        "std_height": int(np.std(list_height))
    }
    
    return EDAResponse(EDA = eda_dict)
    

@app.post("/fit", response_model=FitResponse, tags=["trainer"])
async def fit(requests: FitRequest):
    
    dataset = requests.name_dataset
    os.makedirs("datasets", exist_ok=True)
    datasets_list = os.listdir("datasets")
    if dataset not in datasets_list:
            raise HTTPException(status_code=400, detail=f"Датасета '{dataset}' нет на сервере")
    
    list_emotion = upload_emotion_class(dataset)
    model_id = requests.config.id_model
    
    os.makedirs("models_train", exist_ok=True)
    models_list = [os.path.splitext(file)[0] for file in os.listdir("models_train")]
    
    if model_id in models_list:
        raise HTTPException(status_code=400, detail=f"Модель '{model_id}' уже существует")
    
    if list_emotion['emotions_count'] < 2:
        raise HTTPException(status_code=400, detail="Необходимо загрузить датасет с двумя классами или более")
    
    if not list_emotion['emotions_list']:
        raise HTTPException(status_code=400, detail="В директории с датасетом должны быть вложенные директории, название которых соответсвуют названию класса")
    
    emotions = list_emotion['emotions_list']    
    df = upload_dataset_inframe(dataset, emotions)
    
    if df.empty:
        raise HTTPException(status_code=400, detail="В датасете нет изображений")
    
    emotion_labels_map = {}

    unique_emotions = df['emotion'].unique()
    for i in range(len(unique_emotions)):
        emotion_labels_map[unique_emotions[i]] = i
    df['emotion_label'] = df['emotion'].apply(lambda x: emotion_labels_map[x])
    
    y_train = df["emotion_label"]
    X_train = df["img"]
    X_train.index = np.arange(len(X_train))
    y_train.index = np.arange(len(y_train))
    
    transform_test = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_ds = Dataset(pd.concat([X_train, y_train], axis=1), transform_test)
    train_loader = DataLoader(train_ds, batch_size=32, num_workers=4)
    
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    
    ext = Extraction(feature_extractor)
    train_feature, train_label = ext.extract(train_loader)
    
    param = {
    "kernel": requests.config.hyperparameters.kernal,
    'C': requests.config.hyperparameters.C
    }
    
    svc = svm.SVC()
    svm_grid = GridSearchCV(svc, param_grid=param, verbose=2, n_jobs=-1)

    svm_grid.fit(train_feature, train_label)
    model_with_labels = {
        "model": svm_grid,
        "labels": emotion_labels_map
    }
    
    model_path = f"models_train/{model_id}.joblib"
    joblib.dump(model_with_labels, model_path)
    
    return FitResponse(message=f"Модель '{requests.config.id_model}' обучена и сохранена.")



@app.post("/load_model", response_model=ModelLoadResponse, tags=["upload_file"])
async def load_model(requests: ModelLoadRequest):
    
    global model_active
    model_id = requests.id_model
    model_path = f"models_train/{model_id}.joblib"
    if model_id in model_active:
        return ModelLoadResponse(message=f"Модель '{model_id}' уже загружена.")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Модель '{model_id}' не найдена.")
    model_active.clear()        
    model_active[model_id] = load(model_path)
    return ModelLoadResponse(message=f"Модель '{requests.id_model}' загружена.")

@app.get("/list_models", response_model=ModelsListResponse, tags=["upload_file"])
async def list_models():
    model_list = []
    return ModelsListResponse(models=model_list)

@app.get("/list_datasets", response_model=DatasetsListResponse, tags=["upload_file"])
async def list_datasets():
    model_list = []
    return ModelsListResponse(models=model_list)

@app.delete("/remove_model", response_model=ModelRemoveResponse, tags=["upload_file"])
async def remove_model(requests: ModelRemoveRequest):
    
    return ModelRemoveResponse(message=f"Модель {requests.id_model} удалена")

@app.delete("/remove_dataset", response_model=DatasetRemoveResponse, tags=["upload_file"])
async def remove_dataset(requests: DatasetRemoveRequest):
    return DatasetRemoveResponse(message=f"Датасет {requests.name_dataset} удален")

@app.delete("/remove_all_models", response_model = AllModelsRemoveResponse, tags=["upload_file"])
async def remove_all_models():
    return AllModelsRemoveResponse(message = f"Все модели удалены")

@app.delete("/remove_all_datasets", response_model = AllDatasetsRemoveResponse, tags=["upload_file"])
async def remove_all_datasets():
    return AllDatasetsRemoveResponse(message = f"Все датасеты удалены")

@app.post("/predict", response_model=PredictionResponse, tags=["trainer"])
async def predict(file: UploadFile = File(...)):
    os.makedirs("data_predict", exist_ok=True)
    global model_active
    if not model_active:
        raise HTTPException(status_code=400, detail="Загрузите модель в инференс")
    
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Только ZIP архив необходимо загружать")
    name_dir_predict = file.filename.replace(".zip", "")
    zip_path = os.path.join("data_predict", file.filename)
    with open(zip_path, "wb") as temp:
        content = await file.read()
        temp.write(content)
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("data_predict")
    os.remove(zip_path)

    data_path = f"data_predict/{name_dir_predict}/"
    image_names = [filename for filename in os.listdir(data_path) if is_image(f"{data_path}{filename}")]
    if not image_names:
        raise HTTPException(status_code=400, detail="В корне каталога нет изображений")
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
    
    image_paths = [os.path.join(data_path, filename) for filename in os.listdir(data_path) if is_image(f"{data_path}{filename}")]
    test_features = await extract_features_predict(image_paths, transform, feature_extractor, device)
    model_id = next(iter(model_active))
    data = model_active[model_id]
    model = data["model"]
    labels = data["labels"]
    predictions = await model.predict(test_features)
    predictions_emotions = {}
    for image_name, number in zip(image_names, predictions):
        label = [key for key, value in labels.items() if value == number][0]
        predictions_emotions[image_name] = label
    
    return PredictionResponse(id=model_id, prediction=predictions_emotions)

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
