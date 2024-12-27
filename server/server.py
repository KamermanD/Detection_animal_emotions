import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import zipfile

app = FastAPI(
    docs_url="/api/openapi",
    openapi_url="/api/openapi.json",
)

process_status = {}

class DatasetLoadResponse(BaseModel):
    message: str

@app.post("/load_dataset", response_model=DatasetLoadResponse, tags=["load_dataset"])
async def load_dataset(folder: UploadFile = File(...)):
    
    process_status["load_dataset"] = folder.filename
    
    dir_datasets = [dir for dir in os.listdir("datasets") if os.path.isdir(os.path.join("datasets", dir))]
    if folder.filename in dir_datasets:
        raise HTTPException(status_code=400, detail="Датасет с таким именем уже есть")
    if not folder.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Только ZIP архив необходимо загружать")
    
    zip_path = os.path.join("datasets", folder.filename)
    with open(zip_path, "wb") as temp:
        temp.write(await folder.read())
        
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("datasets")
        
    os.remove(zip_path)
    
    process_status["load_dataset"] = ""
    
    return DatasetLoadResponse(message=f"Dataset {folder.filename} загружен!")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)