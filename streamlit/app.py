import streamlit as st
import json
import requests

URL = "https://98076d76d2767b43b39d343e17803b76.serveo.net/"


def conduct_eda():
    pass

def compute_fit_metrics():
    pass

def dataset_uploader():

    uploaded_file = st.file_uploader("Загрузите zip-файл с датасетом", type=['zip'])

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        files = {
            "file": (uploaded_file.name, file_bytes, "application/zip")
        }
        if st.button('Загрузить датасет'):
            try:
                load_dataset_response = requests.post(URL + '/load_dataset', files=files)
                if load_dataset_response.status_code == 200:
                    st.success(f"Успех: {load_dataset_response.json().get('message')}")
                else:
                    st.error(f"Ошибка: {load_dataset_response.status_code} - {load_dataset_response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Произошла ошибка при запросе: {e}")


def dataset_remover():
    list_datasets_response = requests.get(URL + '/list_datasets')
    available_datasets = list_datasets_response.json()["datasets"]

    selected_dataset = st.selectbox('Выберите датасет для удаления', available_datasets)
    
    if st.button('Удалить датасет'):
        try:
            params = {
                "name_dataset": selected_dataset
            }
            delete_dataset_response = requests.delete(URL + '/remove_dataset', json=params)
            if delete_dataset_response.status_code == 200:
                st.success(f"Успех: {delete_dataset_response.json().get('message')}")
            else:
                st.error(f"Ошибка: {delete_dataset_response.status_code} - {delete_dataset_response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Произошла ошибка при запросе: {e}")

def model_remover():
    list_datasets_response = requests.get(URL + '/list_models')
    available_models = list_datasets_response.json()["models"]

    selected_model = st.selectbox('Выберите модель для удаления', available_models)
    
    if st.button('Удалить модель'):
        try:
            params = {
                "id_model": selected_model
            }
            delete_dataset_response = requests.delete(URL + '/remove_model', json=params)
            if delete_dataset_response.status_code == 200:
                st.success(f"Успех: {delete_dataset_response.json().get('message')}")
            else:
                st.error(f"Ошибка: {delete_dataset_response.status_code} - {delete_dataset_response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Произошла ошибка при запросе: {e}")


def fit_model():
    list_datasets_response = requests.get(URL + '/list_datasets')
    available_datasets = list_datasets_response.json()["datasets"]

    if available_datasets is None:
        available_datasets = []

    selected_dataset = st.selectbox('Выберите датасет', available_datasets)

    id_model = st.text_input("Введите название модели:")

    selected_kernel = st.selectbox('Ядро для SVM', ['linear', 'poly', 'rbf'])

    svc_c = st.text_input("Гиперпарметр регуляризации:")

    if st.button('Обучить модель'):
        if id_model == '':
            st.text('Укажите название модели')
        elif selected_kernel is None:
            st.text('Выберите ядро для SVM')
        elif svc_c == '':
            st.text("Укажите коэффициент регуляризации")
        else:
            try:
                request_params = {
                    "name_dataset": selected_dataset,
                    "config": {
                        "hyperparameters": {
                            "C": [float(svc_c)],
                            "kernel": [selected_kernel]
                        },
                        "id_model": id_model
                    }
                }

                fit_model_response = requests.post(URL + '/fit', json=request_params)
                if fit_model_response.status_code == 200:
                    st.success(f"Успех: {fit_model_response.json().get('message')}")
                else:
                    st.error(f"Ошибка: {fit_model_response.status_code} - {fit_model_response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Произошла ошибка при запросе: {e}")

def inference_model():
    list_models_response = requests.get(URL + '/list_models')
    available_models= list_models_response.json()["models"]

    if available_models is None:
        available_models = []

    selected_model = st.selectbox('Выберите модель', available_models)

    uploaded_file = st.file_uploader("Загрузите zip-файл с датасетом для инференса", type=['zip'])

    if st.button(f'Инференс модели'):
            if uploaded_file is None:
                st.error(f"Ошибка: файл не загружен")
            else:
                try:
                    predict_response = requests.post(URL + '/predict')
                    if predict_response.status_code == 200:
                        st.success(f"Успех: {predict_response.json().get('message')}")
                    else:
                        st.error(f"Ошибка: {predict_response.status_code} - {predict_response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Произошла ошибка при запросе: {e}")


st.title('Animal Emotion Classifier')

st.header('Загрузка нового датасета')
dataset_uploader()

st.header('Обучение модели')
fit_model()

st.header('Инференс модели')
inference_model()

st.header('Удаление Датасета и моделей')
dataset_remover()
model_remover()