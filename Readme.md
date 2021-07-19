#FaceRecognition

Небольшой проект по распознаванию лиц в потоке видео.

Для детектированя и распозавания лица использовал insightface.

Для реализации поиска - faiss.


insightfacevideothread.py - основной скрипт.

server2.py - для запуска медиасервера

postgres_data - дамп данных лиц актеров для postgres (docker). [manual for start](https://techexpert.tips/postgresql/postgresql-docker-installation/)

[manual for cuda](https://illya13.github.io/RL/tutorial/2020/04/26/installing-tensorflow-on-ubuntu-20.html)