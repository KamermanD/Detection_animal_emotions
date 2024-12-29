from pydantic import BaseModel
from typing import List


class Hyperparameters(BaseModel):
    C: List[float]
    kernel: List[str]

class ModelConfig(BaseModel):
    hyperparameters: Hyperparameters 
    id_model: str

class FitRequest(BaseModel):
    name_dataset: str
    config: ModelConfig
    
class ModelLoadRequest(BaseModel):
    id_model:str
    
class ModelRemoveRequest(BaseModel):
    id_model: str
    
class DatasetRemoveRequest(BaseModel):
    name_dataset: str
    
