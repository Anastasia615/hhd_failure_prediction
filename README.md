# hhd_failure_prediction

Учитывая данные мониторинга состояния диска S.M.A.R.T (https://ru.wikipedia.org/wiki/S.M.A.R.T.) и данные о неисправностях, необходимо разработать собственное решение для определения, выйдет ли из строя каждый диск в течение следующих 30 дней.

## Ссылка на датасет:
* https://www.mediafire.com/file/r6l8cmerp8prcfx/ST14000NM001G.tar/file

## Требования:
* Посмотреть список зависимостей в `requirements.txt` и установить их перед запуском.

## Полезные материалы:
* https://tianchi.aliyun.com/dataset/70251
* https://github.com/KarthikNA/Prediction-of-Hard-Drive-Failure
* https://tianchi.aliyun.com/competition/entrance/231775/information
* https://harishkumar-69065.medium.com/hdd-failure-detection-4a4797fae7e
* https://scotcomp.medium.com/preemptive-measures-predicting-hard-drive-failures-01210c7e00a5

## Возможности:
* Обработка пропущенных значений с использованием KNN Imputer.
* Создание циклических временных признаков для учета сезонности.
* Взаимодействующие признаки для повышения информативности модели.
* Использование нескольких моделей в ансамбле для улучшения качества предсказаний.
* Оптимизация весов моделей на основе ROC AUC.
* Интерпретация модели с помощью SHAP для понимания вклада признаков.
