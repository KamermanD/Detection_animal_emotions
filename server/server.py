import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import zipfile
from typing import List

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

process_status = {}
 
class DatasetLoadResponse(BaseModel):
    message: str
    
# class DataLoadRequest(BaseModel):
#     folder: UploadFile = File(...)

class Hyperparameters(BaseModel):
    C: List[float]
    kernal: List[str]

class ModelConfig(BaseModel):
    hyperparameters: Hyperparameters 
    id_model: str
    
class FitResponse(BaseModel):
    message: str

class FitRequest(BaseModel):
    name_dataset: str
    config: ModelConfig

class ModelLoadRequest(BaseModel):
    id_model:str
    
class ModelLoadResponse(BaseModel):
    message: str
    
# class PredictionResponse(BaseModel):
#     message: str
    
# class PredictionRequest(BaseModel):
#     name_dataset: str

class ModelsListResponse(BaseModel):
    models: List[str]

class DatasetsListResponse(BaseModel):
    datasets: List[str]
    
class ModelRemoveResponse(BaseModel):
    message: str
    
class ModelRemoveRequest(BaseModel):
    id_model: str
    
class DatasetRemoveResponse(BaseModel):
    message: str
    
class DatasetRemoveRequest(BaseModel):
    name_dataset: str
    
class AllModelsRemoveResponse(BaseModel):
    message: str
    
class AllDatasetsRemoveResponse(BaseModel):
    message: str

@app.post("/load_dataset", response_model=DatasetLoadResponse, tags=["upload_file"])
async def load_dataset(file: UploadFile = File(...)):
    
    process_status["load_dataset"] = file.filename
    
    if not os.path.exists("datasets"):
        os.makedirs("datasets")
    
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


@app.post("/fit", response_model=FitResponse, tags=["trainer"])
async def fit(requests: FitRequest):
    return FitResponse(message=f"Модель '{requests.config.id_model}' обучена и сохранена.") 


@app.post("/load_model", response_model=ModelLoadResponse, tags=["upload_file"])
async def load_model(requests: ModelLoadRequest):
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

# @app.post("/predict", response_model=PredictionResponse, tags=["trainer"])
# async def predict(request: PredictRequest):

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
