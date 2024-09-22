import io
import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Функция для загрузки изображения через Streamlit
def load_image():
    # Виджет для загрузки файла с изображением
    uploaded_file = st.file_uploader(label='Выберите изображение-видео для распознавания')

    # Проверка, было ли выбрано изображение
    if uploaded_file is not None:
        # Открытие изображения с использованием библиотеки PIL
        image = Image.open(io.BytesIO(uploaded_file.read()))

        # Отображение изображения в интерфейсе Streamlit
        st.image(image)

        # Возвращение объекта изображения
        return image
    else:
        return None

# Загрузка модели при запуске приложения
model = YOLO('yolov8s.pt')

# Заголовок Streamlit
st.title('Детектирование объектов на изображении')

# Загрузка изображения через пользовательский интерфейс
img = load_image()

# Кнопка для запуска распознавания объектов на изображении
result = st.button('Распознать изображение')

# Проверка, была ли нажата кнопка, и выбрано ли изображение
if result and img is not None:
    # Отображение результатов распознавания
    st.write('**Результаты распознавания:**')

    # Вызов функции для распознавания
    output = model.predict(img)  # , save=True

    # Отображение изображений сегментации
    for detect_image in output:
        # Отображение изображения в интерфейсе Streamlit
        img_array = detect_image.plot()
        det_img = Image.fromarray(img_array[..., ::-1])
        st.image(det_img)
