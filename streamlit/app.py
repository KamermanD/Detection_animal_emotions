import streamlit as st
import json
import matplotlib.pyplot as plt
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
                    st.success(f"{load_dataset_response.json().get('message')}")
                else:
                    st.error(f"Ошибка: {load_dataset_response.status_code} - {load_dataset_response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Произошла ошибка при запросе: {e}")


def dataset_remover():
    list_datasets_response = requests.get(URL + '/list_datasets')
    available_datasets = list_datasets_response.json()["datasets"]

    selected_dataset = st.selectbox('Выберите датасет для удаления', available_datasets)
    
    if st.button('Удалить датасет'):
        params = {
                "name_dataset": selected_dataset
            }
        try:
            delete_dataset_response = requests.delete(URL + '/remove_dataset', json=params)
            if delete_dataset_response.status_code == 200:
                st.success(f"{delete_dataset_response.json().get('message')}")
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
                st.success(f"{delete_dataset_response.json().get('message')}")
            else:
                st.error(f"Ошибка: {delete_dataset_response.status_code} - {delete_dataset_response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Произошла ошибка при запросе: {e}")


def show_fitting_info(response):
    roc_auc_score = response.json().get('roc_auc_ovr')
    tpr = response.json().get('true_positive_rate_ovr')
    fpr = response.json().get('false_positive_rate_ovr')

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc_score:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Chance')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC)')
    ax.legend(loc="lower right")

    st.pyplot(fig)

    st.write(f'ROC-AUC: {roc_auc_score}')

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
            try:
                fit_model_response = requests.post(URL + '/fit', json=request_params)
                if fit_model_response.status_code == 200:
                    st.success(f"{fit_model_response.json().get('message')}")
                    show_fitting_info(fit_model_response)
                else:
                    st.error(f"Ошибка: {fit_model_response.status_code} - {fit_model_response.text}")
            except requests.exceptions.RequestException as e:
                st.error(f"Произошла ошибка при запросе: {e}")

def load_model():
    list_models_response = requests.get(URL + '/list_models')
    available_models= list_models_response.json()["models"]

    if available_models is None:
        available_models = []
    
    selected_model = st.selectbox('Выберите модель', available_models)

    if st.button(f"Загрузить модель"):
        params = {"id_model": selected_model}
        try:
            load_model_response = requests.post(URL + '/load_model', json=params)
            if load_model_response.status_code == 200:
                st.success(f"{load_model_response.json().get('message')}")
            else:
                st.error(f"Ошибка: {load_model_response.status_code} - {load_model_response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Произошла ошибка при запросе: {e}")

def inference_model():

    uploaded_file = st.file_uploader("Загрузите zip-файл с датасетом для инференса", type=['zip'])

    if st.button(f'Инференс модели'):
            if uploaded_file is None:
                st.error(f"Ошибка: файл не загружен")
            else:
                file_bytes = uploaded_file.read()
                files = {"file": (uploaded_file.name, file_bytes, "application/zip")}
                try:
                    predict_response = requests.post(URL + '/predict', files=files)
                    if predict_response.status_code == 200:
                        st.success(f"{predict_response.json().get('message')}")
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
load_model()
inference_model()

st.header('Удаление датасетов и моделей')
dataset_remover()
model_remover()