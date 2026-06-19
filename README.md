# Курсовой проект ClearML Text Classification

Этот проект реализует MLOps-жизненный цикл для игрушечного классификатора тональности текста:

1. версионированный ClearML Dataset,
2. удаленное обучение через ClearML Agent,
3. логирование метрик и артефактов,
4. публикация модели в Model Registry,
5. HTTP endpoint через ClearML Serving,
6. Streamlit UI, который вызывает serving endpoint по HTTP.

Модель специально сделана легкой: scikit-learn pipeline `TF-IDF + Logistic Regression` или `LinearSVC`

## 1. Локальное окружение

Используйте Python 3.11 или 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## 2. Запуск ClearML Server

```bash
docker compose up -d apiserver webserver fileserver elasticsearch mongodb redis
```

Откройте:

- ClearML Web UI: http://localhost:18080
- API server: http://localhost:8008
- File server: http://localhost:8081

Создайте credentials в ClearML UI, затем настройте SDK:

```bash
clearml-init
```

Используйте:

```text
api_server: http://localhost:8008
web_server: http://localhost:18080
files_server: http://localhost:8081
```

## 3. Запуск ClearML Agent

Обязательное имя очереди: `students`.

```bash
clearml-agent daemon --queue students --foreground
```

Альтернативный запуск агента через Docker:

```bash
docker compose --profile agent up agent
```

В ClearML UI проверьте, что агент виден.

В этом локальном запуске агент использовал локальный git remote через `file://`, чтобы клонировать проект без GitHub/GitLab:

```bash
git init --bare ../mlops-aith-2026-kachaev.git
git remote add origin "file://$(cd .. && pwd)/mlops-aith-2026-kachaev.git"
git push -u origin main
```

## 4. Создание ClearML Dataset

```bash
python -m src.create_dataset \
  --dataset-name sentiment_texts \
  --dataset-version 1.0.0 \
  --data-path data/raw/sentiment.csv
```

Скрипт печатает и сохраняет dataset id:

```bash
cat outputs/dataset_id.txt
```

Выполненный запуск создал:

```text
caea8b717da64c0ba645cadecbf39ddd
```

## 5. Запуск двух экспериментов через агент

```bash
git init
git add .
git commit -m "Initial ClearML course project"
```

В этом запуске ClearML Agent выполнял root entrypoints, которые вызывают `src.train` с фиксированными параметрами эксперимента:

```bash
clearml-task --project mlops-aith-2026-kachaev --name sentiment-logreg-agent \
  --folder . --script train_logreg_agent.py --cwd . \
  --requirements requirements-agent.txt --queue students \
  --task-type training --skip-task-init

clearml-task --project mlops-aith-2026-kachaev --name sentiment-linearsvc-agent \
  --folder . --script train_linearsvc_agent.py --cwd . \
  --requirements requirements-agent.txt --queue students \
  --task-type training --skip-task-init
```

Завершенные задачи:

- `231ce563bef246078ac128745ff23999` - `sentiment-logreg-agent`
- `006e49c6e6c84325911800db5542de81` - `sentiment-linearsvc-agent`


## 6. Публикация лучшей модели в Model Registry

Выберите лучший `clearml_model_id` из логов winning training task.

```bash
python -m src.register_model \
  --model-id "b61a19aa3db44babb4e4d61de652159a" \
  --version 1.0.0
```

Для ClearML Serving веса модели должны быть доступны изнутри Docker. В этом запуске также была опубликована serving-ready копия модели, веса которой загружены в ClearML fileserver:

```text
92a491b058ab4ada84b6c867afc5cd1e
```

## 7. Деплой ClearML Serving

Установите ClearML Serving, если он еще не установлен:

```bash
pip install -r requirements-serving.txt
```

Создайте serving service и добавьте зарегистрированную модель:

```bash
sed 's#http://localhost:#http://host.docker.internal:#g' clearml.conf > clearml.docker.conf
CLEARML_SERVING_TASK_ID=ffe0156edd134f28bfb42ef68f3bfbb2 \
  docker compose --profile serving up -d serving-inference
```

Рабочие serving values из этого запуска:

```text
Serving task id: ffe0156edd134f28bfb42ef68f3bfbb2
Serving model id: 92a491b058ab4ada84b6c867afc5cd1e
Endpoint: http://localhost:8090/serve/sentiment
```

Примеры запросов:

```bash
curl -X POST http://localhost:8090/serve/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "The interface is clean and reliable"}'

curl -X POST http://localhost:8090/serve/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "The app crashes and the support is terrible"}'
```

## 8. Запуск Streamlit UI

```bash
export CLEARML_SERVING_URL=http://localhost:8090/serve/sentiment
streamlit run ui/streamlit_app.py
```


## 9. Артефакты 

<img width="1280" height="882" alt="proj_1" src="https://github.com/user-attachments/assets/df48ac04-8d2e-4fd9-a654-9a9b2bd08bcf" />

<img width="1280" height="881" alt="proj_2" src="https://github.com/user-attachments/assets/d359a269-0f2a-4d49-add1-7d063cc19c43" />


<img width="1280" height="880" alt="proj_3" src="https://github.com/user-attachments/assets/30b1016f-1312-4b66-9e43-8fa1fc198349" />

<img width="1280" height="878" alt="proj_4" src="https://github.com/user-attachments/assets/e3ff72d1-d3d1-49b2-8ee0-c259b36b80f7" />


<img width="1280" height="882" alt="proj_5" src="https://github.com/user-attachments/assets/f506b0f8-fb48-43b9-9277-9e4307b2d183" />

<img width="1280" height="880" alt="proj6" src="https://github.com/user-attachments/assets/63d584ae-d6df-4f13-832c-b7fd4f2147e0" />


<img width="1280" height="881" alt="proj8" src="https://github.com/user-attachments/assets/a9f1b7f1-7535-49c6-b5d1-81791bd72c76" />
<img width="1280" height="883" alt="proj7" src="https://github.com/user-attachments/assets/9a47e6de-e277-453d-b55f-38c63d43dbe8" />

<img width="1280" height="875" alt="proj9" src="https://github.com/user-attachments/assets/1d8d5058-0d90-453d-b42b-db5daa1a9e04" />

<img width="1280" height="881" alt="proj11" src="https://github.com/user-attachments/assets/3a925074-c527-4b49-baec-92a7cc156cfb" />
<img width="1280" height="880" alt="proj10" src="https://github.com/user-attachments/assets/63a5f3e8-d8bb-44d3-a58d-d99c1437a91e" />



<img width="1280" height="228" alt="proj12" src="https://github.com/user-attachments/assets/c3308e01-7346-42fb-bb01-c09e02a9ed99" />










