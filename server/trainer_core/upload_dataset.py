import os
from typing import List
import pandas as pd
from PIL import Image
from fastapi import HTTPException


def upload_emotion_class(name_dataset: str):
    emotions_dict = {
        'emotions_list': [],
        'emotions_count': 0
    }
    path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), f"datasets/{name_dataset}")
    dir_names = [dir for dir in os.listdir(
        path) if os.path.isdir(os.path.join(path, dir))]

    emotions_dict['emotions_list'] = dir_names
    emotions_dict['emotions_count'] = len(dir_names)

    return emotions_dict


def is_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def upload_dataset_inframe(name_dataset: str, emotion_list: List[str]):
    data = {'img': [], 'emotion': []}

    path = os.path.join(os.path.dirname(
        os.path.dirname(__file__)), f"datasets/{name_dataset}/")

    for emotion in emotion_list:
        for dirname, _, filenames in os.walk(path + emotion + '/'):
            if len(filenames) <= 5:
                raise HTTPException(
                    status_code=400, detail="В каждом должно быть больше 5 изображений!")
            for filename in filenames:
                file_path = os.path.join(dirname, filename)
                if is_image(file_path):
                    data['img'].append(os.path.join(dirname, filename))
                    data['emotion'].append(emotion)

    df = pd.DataFrame.from_dict(data)

    return df
