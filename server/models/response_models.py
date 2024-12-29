from pydantic import BaseModel
from typing import List, Dict

class DatasetLoadResponse(BaseModel):
    message: str
    
class FitResponse(BaseModel):
    message: str
    roc_auc_ovr: float
    true_positive_rate_ovr: List[float]
    false_positive_rate_ovr: List[float]
    
class ModelLoadResponse(BaseModel):
    message: str
    
class ModelsListResponse(BaseModel):
    models: List[str]

class DatasetsListResponse(BaseModel):
    datasets: List[str]
    
class ModelRemoveResponse(BaseModel):
    message: str
    
class DatasetRemoveResponse(BaseModel):
    message: str
    
class AllModelsRemoveResponse(BaseModel):
    message: str
    
class AllDatasetsRemoveResponse(BaseModel):
    message: str

class PredictionResponse(BaseModel):
    id: str
    prediction: Dict[str, str]