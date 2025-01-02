import uvicorn
from typing import Annotated
from fastapi import FastAPI, UploadFile, File, HTTPException

from models.request_models import FitRequest, ModelLoadRequest
from models.request_models import ModelRemoveRequest, DatasetRemoveRequest
from models.request_models import EDARequest
from models.response_models import DatasetLoadResponse, FitResponse
from models.response_models import ModelLoadResponse, PredictionResponse
from models.response_models import ModelsListResponse, DatasetsListResponse
from models.response_models import ModelRemoveResponse, DatasetRemoveResponse
from models.response_models import AllModelsRemoveResponse
from models.response_models import AllDatasetsRemoveResponse
from models.response_models import EDAResponse

from trainer_core.eda import eda_info
from trainer_core.fit import fit_train
from trainer_core.predict import load_model_inference, predict_inference

from core.logger import CustomizeLogger
from services import storage_service

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)
logger = CustomizeLogger.make_logger("server")
app.logger = logger

model_active = {}


@app.post("/load_dataset", response_model=DatasetLoadResponse, tags=["upload_file"])
async def load_dataset(file: Annotated[UploadFile , File(...)]) -> DatasetLoadResponse:
    dataset_name = await storage_service.load_dataset(file)
    return DatasetLoadResponse(message=f"Dataset {dataset_name} загружен!")


@app.post("/eda", response_model=EDAResponse, tags=["trainer"])
async def eda(requests: EDARequest) -> EDAResponse:
    eda_dict = await eda_info(requests)   
    return EDAResponse(EDA = eda_dict)


@app.post("/fit", response_model=FitResponse, tags=["trainer"])
async def fit(request: FitRequest) -> FitResponse:
    fit_data = fit_train(request)    
    return fit_data   



@app.post("/load_model", response_model=ModelLoadResponse, tags=["predict"])
async def load_model(request: ModelLoadRequest) -> ModelLoadResponse:
    global model_active
    model_active.clear()
    model_inference = await load_model_inference(request)
    model_active = model_inference
    return ModelLoadResponse(message=f"Модель {request.id_model} загружена")

@app.get("/list_models", response_model=ModelsListResponse, tags=["upload_file"])
async def list_models() -> ModelsListResponse:
    model_list = storage_service.list_models()
    return ModelsListResponse(models=model_list)


@app.get("/list_datasets", response_model=DatasetsListResponse, tags=["upload_file"])
async def list_datasets() -> DatasetsListResponse:
    dataset_list = storage_service.list_datasets()
    return DatasetsListResponse(datasets=dataset_list)


@app.delete("/remove_model", response_model=ModelRemoveResponse, tags=["upload_file"])
async def remove_model(request: ModelRemoveRequest) -> ModelRemoveResponse:
    existed = storage_service.delete_model(request.id_model)
    if not existed:
        raise HTTPException(
            status_code=400, detail=f"Модель {request.id_model} не загружена")
    return ModelRemoveResponse(message=f"Модель {request.id_model} удалена")


@app.delete("/remove_dataset", response_model=DatasetRemoveResponse, tags=["upload_file"])
async def remove_dataset(request: DatasetRemoveRequest) -> DatasetRemoveResponse:
    existed = storage_service.delete_dataset(request.name_dataset)
    if not existed:
        raise HTTPException(
            status_code=400, detail=f"Датасет {request.name_dataset} не загружен")
    return DatasetRemoveResponse(message=f"Датасет {request.name_dataset} удален")


@app.delete("/remove_all_models", response_model=AllModelsRemoveResponse, tags=["upload_file"])
async def remove_all_models() -> AllModelsRemoveResponse:
    storage_service.delete_all_models()
    return AllModelsRemoveResponse(message=f"Все модели удалены")


@app.delete("/remove_all_datasets", response_model=AllDatasetsRemoveResponse, tags=["upload_file"])
async def remove_all_datasets() -> AllDatasetsRemoveResponse:
    storage_service.delete_all_datasets()
    return AllDatasetsRemoveResponse(message=f"Все датасеты удалены")


@app.post("/predict", response_model=PredictionResponse, tags=["predict"])
async def predict(file: Annotated[UploadFile , File(...)]) -> PredictionResponse:
    global model_active
    predict = await predict_inference(model_active, file)
    return predict

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
