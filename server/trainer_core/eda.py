from typing import Dict
import os
from pathlib import Path
from typing import Final
from fastapi import HTTPException
from PIL import Image
import numpy as np

from trainer_core.upload_dataset import upload_emotion_class
from trainer_core.upload_dataset import upload_dataset_inframe
from models.request_models import EDARequest

DATASETS_PATH: Final[str] = Path(__file__).parent.parent / "datasets"

async def eda_info(requests: EDARequest) -> Dict[str, int]:
    dataset = requests.name_dataset
    os.makedirs(DATASETS_PATH, exist_ok=True)
    datasets_list = os.listdir(DATASETS_PATH)
    
    if dataset not in datasets_list:
        raise HTTPException(
            status_code=400, detail=f"Датасета '{dataset}' нет на сервере")
        
    list_emotion = upload_emotion_class(dataset)
     
    if list_emotion['emotions_count'] < 2:
        raise HTTPException(
            status_code=400, detail="Необходимо загрузить датасет с двумя классами или более")
        
    df = upload_dataset_inframe(dataset, list_emotion['emotions_list'])
    path_dataset = f"{DATASETS_PATH}/{dataset}"
    
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
    
    return eda_dict