FROM python:3.9.13-slim-bullseye

WORKDIR /app
COPY poetry.lock pyproject.toml ./

RUN apt-get update && apt-get install -y --no-install-recommends gcc linux-libc-dev libc6-dev && \
    pip install "poetry==1.6.1" && poetry config virtualenvs.create false && \
    poetry install --no-dev --no-root --no-interaction -vvv


COPY netflix_recommender_system/ netflix_recommender_system/
COPY data/ data/
COPY models/ models/
COPY config.yaml config.yaml

EXPOSE 8000

CMD [ "sh", "-c", "fastapi run netflix_recommender_system/server.py"]
