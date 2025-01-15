from typing import Dict, List, Callable
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


def calculate_statistics(data: List[int], agg_func: Callable[[List[int]], float]) -> int:
    return int(agg_func(data))


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
    list_cat_R = []
    list_cat_G = []
    list_cat_B = []

    for root, _, images in os.walk(path_dataset):
        for image in images:
            if not (image.endswith('jpg') or image.endswith('png') or image.endswith('jpeg')):
                continue
            image_path = os.path.join(root, image)
            with Image.open(image_path).convert('RGB') as img:
                width, height = img.size
                list_width.append(width)
                list_height.append(height)
                img_array = np.array(img)

                list_cat_R += img_array[:, :, 0].flatten().tolist()
                list_cat_G += img_array[:, :, 1].flatten().tolist()
                list_cat_B += img_array[:, :, 2].flatten().tolist()

    eda_dict = {
        "count_classes": df['emotion'].nunique(),
        "count_images": df.shape[0],
    }

    stats_size = {
        "width": list_width,
        "height": list_height,
    }

    agg_funcs = {
        "mean": np.mean,
        "std": np.std,
        "min": np.min,
        "max": np.max,
    }

    for key, data in stats_size.items():
        for stat_name, agg_func in agg_funcs.items():
            eda_dict[f"{stat_name}_{key}"] = calculate_statistics(data, agg_func)

    for channel, data in zip(["R", "G", "B"], [list_cat_R, list_cat_G, list_cat_B]):
        eda_dict[f"mean_{channel}"] = calculate_statistics(data, np.mean)
        eda_dict[f"std_{channel}"] = calculate_statistics(data, np.std)

    return eda_dict
