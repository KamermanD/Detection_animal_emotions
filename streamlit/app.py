import streamlit as st
import matplotlib.pyplot as plt
import requests
import logging
import os
import zipfile
from PIL import Image
import io
import shutil
from core.logger import CustomizeLogger


URL = "https://4f2ab330d0b41e4029b8d30f6ee57588.serveo.net"


def dataset_uploader():
    if 'upload_dataset_button' not in st.session_state:
        st.session_state.upload_dataset_button = -1
        st.session_state.upload_dataset_response = None

    uploaded_file = st.file_uploader(
        "Загрузите zip-файл с датасетом",
        type=['zip'])
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        files = {
            "file": (uploaded_file.name, file_bytes, "application/zip")
        }
        if st.button('Загрузить датасет'):
            try:
                dataset_response = requests.post(
                    URL + '/load_dataset',
                    files=files)
                if dataset_response.status_code == 200:
                    st.session_state.upload_dataset_button = 200
                    st.session_state.upload_dataset_response = dataset_response
                else:
                    st.session_state.upload_dataset_button = dataset_response.status_code
                    st.session_state.upload_dataset_response = dataset_response
            except requests.exceptions.RequestException as e:
                st.session_state.upload_dataset_button = 700
                st.session_state.upload_dataset_response = e
    
    if st.session_state.upload_dataset_button == 200:
        st.success(f"{st.session_state.upload_dataset_response.json().get('message')}")
    elif st.session_state.upload_dataset_button == 700:
        st.error(f"Произошла ошибка при запросе: {st.session_state.upload_dataset_response}")
    elif st.session_state.upload_dataset_button != -1:
        st.error(f"Ошибка: {st.session_state.upload_dataset_response.status_code} \
                  - {st.session_state.upload_dataset_response.text}")
        

def dataset_remover():
    if 'remove_dataset_button' not in st.session_state:
        st.session_state.remove_dataset_button = -1
        st.session_state.remove_dataset_response = None

    list_datasets_response = requests.get(URL + '/list_datasets')
    available_datasets = list_datasets_response.json()["datasets"]
    selected_dataset = st.selectbox(
        'Выберите датасет для удаления',
        available_datasets)
    if st.button('Удалить датасет'):
        params = {
                "name_dataset": selected_dataset
            }
        try:
            response = requests.delete(
                URL + '/remove_dataset',
                json=params)
            if response.status_code == 200:
                st.session_state.remove_dataset_button = 200
                st.session_state.remove_dataset_response = response
            else:
                st.session_state.remove_datset_button = response.status_code
                st.session_state.remove_dataset_response = response
        except requests.exceptions.RequestException as e:
            st.session_state.remove_datset_button = response.status_code
            st.session_state.remove_dataset_response = e

    if st.session_state.remove_dataset_button == 200:
        st.success(f"{st.session_state.remove_dataset_response.json().get('message')}")
    elif st.session_state.remove_dataset_button == 700:
        st.error(f"Произошла ошибка при запросе: {st.session_state.remove_dataset_response}")
    elif st.session_state.remove_dataset_button != -1:
        st.error(f"Ошибка: {st.session_state.remove_dataset_response.status_code} \
                  - {st.session_state.remove_dataset_response.text}")


def model_remover():
    if 'remove_model_button' not in st.session_state:
        st.session_state.remove_model_button = -1
        st.session_state.remove_model_response = None

    list_models_response = requests.get(URL + '/list_models')
    available_models = list_models_response.json()["models"]
    selected_model = st.selectbox(
        'Выберите модель для удаления',
        available_models)
    if st.button('Удалить модель'):
        try:
            params = {
                "id_model": selected_model
            }
            response = requests.delete(
                URL + '/remove_model',
                json=params)
            if response.status_code == 200:
                st.session_state.remove_model_button = 200
                st.session_state.remove_model_response = response
            else:
                st.session_state.remove_model_button = response.status_code
                st.session_state.remove_model_response = response
        except requests.exceptions.RequestException as e:
            st.session_state.remove_model_button = 700
            st.session_state.remove_model_response = e

    if st.session_state.remove_model_button == 200:
        st.success(f"{st.session_state.remove_model_response.json().get('message')}")
    elif st.session_state.remove_model_button == 700:
        st.error(f"Произошла ошибка при запросе: {st.session_state.remove_model_response}")
    elif st.session_state.remove_model_button != -1:
        st.error(f"Ошибка: {st.session_state.remove_model_response.status_code} \
                  - {st.session_state.remove_model_response.text}")


def show_fitting_info(response):
    roc_auc_score = response.json().get('roc_auc_ovr')
    tpr = response.json().get('true_positive_rate_ovr')
    fpr = response.json().get('false_positive_rate_ovr')
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve')
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
    if "fit_button" not in st.session_state:
        st.session_state.fit_button = -1
        st.session_state.fit_response = None

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
                response = requests.post(
                    URL + '/fit',
                    json=request_params)
                if response.status_code == 200:
                    st.session_state.fit_button = 200
                    st.session_state.fit_response = response
                else:
                    st.session_state.fit_button = response.status_code
                    st.session_state.fit_response = response
            except requests.exceptions.RequestException as e:
                    st.session_state.fit_button = 700
                    st.session_state.fit_response = e

    if st.session_state.fit_button == 200:
        st.success(f"{st.session_state.fit_response.json().get('message')}")
        show_fitting_info(st.session_state.fit_response)
    elif st.session_state.fit_button == 700:
        st.error(f"Произошла ошибка при запросе: {st.session_state.fit_response}")
    elif st.session_state.fit_button != -1:
        st.error(f"Ошибка: {st.session_state.fit_response.status_code} - {st.session_state.fit_response.text}")


def load_model():
    if "load_moedl_button" not in st.session_state:
        st.session_state.load_model_button = -1
        st.session_state.load_model_response = None

    list_models_response = requests.get(URL + '/list_models')
    available_models = list_models_response.json()["models"]
    if available_models is None:
        available_models = []
    selected_model = st.selectbox('Выберите модель', available_models)
    if st.button(f"Загрузить модель"):
        params = {"id_model": selected_model}
        try:
            response = requests.post(
                URL + '/load_model',
                json=params)
            if response.status_code == 200:
                st.session_state.load_model_button = 200
                st.session_state.load_model_response = response
            else:
                st.session_state.load_model_button = response.status_code
                st.session_state.load_model_response = response
        except requests.exceptions.RequestException as e:
            st.session_state.load_model_button = 700
            st.session_state.load_model_response = e

    if st.session_state.load_model_button == 200:
        st.success(f"{st.session_state.load_model_response.json().get('message')}")
    elif st.session_state.load_model_button == 700:
        st.error(f"Произошла ошибка при запросе: {st.session_state.load_model_response}")
    elif st.session_state.load_model_button != -1:
        st.error(f"Ошибка: {st.session_state.laod_model_response.status_code} - {st.session_state.load_model_response.text}")

def show_predictions(predictions, file):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall("temp_images")
    extracted_path = "temp_images"
    sub_dirs = [
        d for d in os.listdir(extracted_path)
        if os.path.isdir(os.path.join(extracted_path, d)) and d != "__MACOSX"
    ]
    if len(sub_dirs) == 1:
        images_path = os.path.join(extracted_path, sub_dirs[0])
    else:
        images_path = extracted_path  

    for key, value in predictions.items():
        st.write(f"**{key}:** {value}")
        image_path = os.path.join(images_path, key)
        image = Image.open(image_path)
        st.image(image, caption=f"{key}", use_column_width=True)

    shutil.rmtree("temp_images")


def inference_model():
    if "predict_button" not in st.session_state:
        st.session_state.predict_button = -1
        st.session_state.predict_response = None

    uploaded_file = st.file_uploader(
        "Загрузите zip-файл с датасетом для инференса",
        type=['zip'])
    if st.button(f'Инференс модели'):
        if uploaded_file is None:
            st.session_state.predict_button = 701
        else:
            file_bytes = uploaded_file.read()
            files = {
                "file": (uploaded_file.name, file_bytes, "application/zip")
                }
            try:
                response = requests.post(URL + '/predict', files=files)
                if response.status_code == 200:
                    st.session_state.predict_button = 200
                    st.session_state.predict_response = response
                else:
                    st.session_state.predict_button = response.status_code
                    st.session_state.predict_response = response
            except requests.exceptions.RequestException as e:
                st.session_state.predict_button = 700
                st.session_state.predict_response = e
    
    if st.session_state.predict_button == 200:
        show_predictions(
            st.session_state.predict_response.json().get('prediction'),
            uploaded_file)
    elif st.session_state.predict_button == 700:
        st.error(f"Произошла ошибка при запросе: {st.session_state.predict_response}")
    elif st.session_state.predict_button == 701:
        st.error(f"Ошибка: файл не загружен")
    elif st.session_state.predict_button != -1:
        st.error(f"Ошибка: {st.session_state.predict_response.status_code} - {st.session_state.predict_response.text}")


def eda_show_stats(d_info):
    st.subheader("General Statistics")
    st.write(f"**Number of Classes:** {d_info.get('count_classes')}")
    st.write(f"**Number of Images:** {d_info.get('count_images')}")

    st.subheader("Color Channel Statistics (RGB)")
    st.write(f"**Mean R:** {d_info.get('mean_R')}")
    st.write(f"**Mean G:** {d_info.get('mean_G')}")
    st.write(f"**Mean B:** {d_info.get('mean_B')}")
    st.write(f"**Std R:** {d_info.get('std_R')}")
    st.write(f"**Std G:** {d_info.get('std_G')}")
    st.write(f"**Std B:** {d_info.get('std_B')}")

    st.subheader("Image Dimensions")
    st.write(f"**Mean Width:** {d_info.get('mean_width')} px")
    st.write(f"**Mean Height:** {d_info.get('mean_height')} px")
    st.write(f"**Min Width:** {d_info.get('min_width')} px")
    st.write(f"**Min Height:** {d_info.get('min_height')} px")
    st.write(f"**Max Width:** {d_info.get('max_width')} px")
    st.write(f"**Max Height:** {d_info.get('max_height')} px")
    st.write(f"**Std Width:** {d_info.get('std_width')} px")
    st.write(f"**Std Height:** {d_info.get('std_height')} px")


def dataset_eda():
    if 'eda_button' not in st.session_state:
        st.session_state.eda_button = -1
        st.session_state.eda_response = None
    response = requests.get(URL + '/list_datasets')
    available_datasets = response.json()["datasets"]
    selected_dataset = st.selectbox(
        'Выберите датасет для EDA',
        available_datasets)
    if st.button('Провести EDA'):
        try:
            response = requests.post(
                URL + '/eda',
                json={'name_dataset': selected_dataset})
            if response.status_code == 200:
                st.session_state.eda_button = 200
                st.session_state.eda_response = response
            else:
                st.session_state.eda_button = response.status_code
                st.session_state.eda_response = response
        except requests.exceptions.RequestException as e:
            st.session_state.eda_button = 700
            st.session_state.eda_response = e
    
    if st.session_state.eda_button == 200:
        eda_show_stats(st.session_state.eda_response.json().get('EDA'))
    elif st.session_state.eda_button == 700:
        st.error(f"Произошла ошибка при запросе: {st.session_state.eda_response}")
    elif st.session_state.eda_button != -1:
        st.error(f"Ошибка: {st.session_state.eda_response.status_code} - {st.session_state.eda_response.text}")



def main():
    st.title('Animal Emotion Classifier')
    st.header('Загрузка нового датасета')
    dataset_uploader()

    st.header('Exploratory data analysis')
    dataset_eda()

    st.header('Обучение модели')
    fit_model()

    st.header('Инференс модели')
    load_model()
    inference_model()

    st.header('Удаление датасетов и моделей')
    dataset_remover()
    model_remover()


if __name__ == '__main__':
    logger = CustomizeLogger.make_logger("front")

    main()
