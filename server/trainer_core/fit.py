import os
from typing import Final
from pathlib import Path
from fastapi import HTTPException
import numpy as np
import joblib
from torchvision.transforms import v2
import torch
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from trainer_core.dataset import Dataset
from torch.utils.data import DataLoader
from trainer_core.extraction import Extraction
from torchvision.models import resnet18, ResNet18_Weights
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV

from models.request_models import FitRequest
from models.response_models import FitResponse
from trainer_core.upload_dataset import upload_emotion_class
from trainer_core.upload_dataset import upload_dataset_inframe

DATASETS_PATH: Final[str] = Path(__file__).parent.parent / "datasets"
MODELS_PATH: Final[str] = Path(__file__).parent.parent / "models_train"

def fit_train(request: FitRequest) -> FitResponse:
    dataset = request.name_dataset
    os.makedirs(DATASETS_PATH, exist_ok=True)
    datasets_list = os.listdir(DATASETS_PATH)
    if dataset not in datasets_list:
        raise HTTPException(
            status_code=400, detail=f"Датасета '{dataset}' нет на сервере")

    list_emotion = upload_emotion_class(dataset)
    model_id = request.config.id_model

    os.makedirs(MODELS_PATH, exist_ok=True)
    models_list = [os.path.splitext(file)[0]
                   for file in os.listdir(MODELS_PATH)]

    if model_id in models_list:
        raise HTTPException(
            status_code=400, detail=f"Модель '{model_id}' уже существует")

    if list_emotion['emotions_count'] < 2:
        raise HTTPException(
            status_code=400, detail="Необходимо загрузить датасет с двумя классами или более")

    if not list_emotion['emotions_list']:
        raise HTTPException(
            status_code=400,
            detail="Нет вложенных директорий соответсвующих классам"
        )

    emotions = list_emotion['emotions_list']
    df = upload_dataset_inframe(dataset, emotions)

    if df.empty:
        raise HTTPException(
            status_code=400, detail="В датасете нет изображений")

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
        "kernel": request.config.hyperparameters.kernel,
        'C': request.config.hyperparameters.C
    }

    svc = svm.SVC()
    svm_grid = GridSearchCV(svc, param_grid=param, verbose=2, n_jobs=-1)

    svm_grid.fit(train_feature, train_label)
    model_with_labels = {
        "model": svm_grid,
        "labels": emotion_labels_map
    }

    model_path = f"{MODELS_PATH}/{model_id}.joblib"
    joblib.dump(model_with_labels, model_path)

    # calculate micro-average roc using one-vs-rest strategy
    pred_score = svm_grid.decision_function(train_feature)
    label_binarizer = LabelBinarizer().fit(train_label)
    train_onehot_label = label_binarizer.transform(train_label)

    
    fpr, tpr, _ = roc_curve(train_onehot_label.ravel(), pred_score.ravel())
    roc_auc = auc(fpr, tpr)
    
    return FitResponse(
        message=f"Модель '{request.config.id_model}' обучена и сохранена.",
        roc_auc_ovr = roc_auc,
        true_positive_rate_ovr = tpr,
        false_positive_rate_ovr = fpr,
    )