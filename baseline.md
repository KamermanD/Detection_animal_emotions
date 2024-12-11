# Построение бейзлайна

В рамках данного этапа каждый из участников проекта взял отдельный датасет для построения бейзлайна. В качестве базовой модели использовался SVM с различными вариантами feature extractor'а. Лучшие результаты представлены ниже в виде таблицы. 

| model                            | dataset                            | accuracy | student  |
|----------------------------------|------------------------------------|----------|----------|
| HOG + SVM                        | Cat's emotions                     | 0.284    | Ksenia   |
| ResNet18 (Feature Extract) + SVM | Dog's emotions                     | 0.734    | Mikhail  |
| ResNet18 (Feature Extract) + SVM | Stanford Dogs Dataset              | 0.79     | Vladimir |
| SVM                              | Dog's emotions (DKamerman Dataset) | 0.415    | Dmitry   |

