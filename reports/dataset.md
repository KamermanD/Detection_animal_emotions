[Ссылка на датасеты](https://drive.google.com/drive/folders/1V5UZb6KxIqVrpQ09dsf2CTwWkRFmns6R?usp=drive_link)

Описание датасетов:

**Cat's emotions** **- выбран основным**
В датасете всего 671 изображение, 7 классов ('Happy', 'Angry', 'Scared', 'Normal', 'Sad', 'Disgusted', 'Surprised'), по классам изображения распределены более-менее равномерно. Самый маленький класс - Disgusted, 79 изображений, самый большой - Surprised, 100 изображений.
Размеры изображений одинаковый: 640х640.
По анализу каналов RGB есть вылеты на 0 и 255 - возможно, из-за присутстия изображений на белом/чёрном фоне. Есть изображения, будто сгенерированные нейросетью, а не фотки.
Есть разбиение на train/valid

**Dog's emotions** **- выбран основным**
В датасете всего 4000 изображений, 4 класса ('angry', 'happy', 'relaxed', 'sad'), по классам изображения распределены равномерно.
Размеры изображений очень разные, не все квадратные.
Есть повороты изображений.
По анализу каналов RGB тоже есть вылеты на 0 и 255 - возможно, из-за присутстия изображений на белом/чёрном фоне.

**Pets emotions**
В датасете всего 1000 изображений, 4 класса, по классам изображения распределены равномерно
Самих классов эмоций мало: angry, happy, sad и other. В датасете изображения разных животных (кошки, собаки, морские свинки и др), по животным не классифицирован
Размеры изображений либо 224х224 (большинство), либо 179х179
По анализу каналов RGB есть вылеты на 0 и 255 - возможно, из-за присутстия ч/б изображений в датасете или есть изображения с проблемами с экспозицией
Есть разбиение на test/train/valid

**Stanford Dogs Dataset**
Породы собак
Имеется достаточное количество картинок (20580 суммарно, около 150 картинок на каждый класс), изображения согласуются с разметкой, распределение значений по каналам похоже на распределение для ImageNet.
Так же в датасете есть разметка по bounding box'ам.
Количество изображений и согласованность распределений значений с ImageNet благоволит к использованию fine tuning'а модели, основанной на ImageNet - например ResNet18 или ResNet34.
По ресайзу изображений предлагается использовать размер 224 * 224, что является стандартной практикой при работе с моделями ImageNet/ResNet18.
