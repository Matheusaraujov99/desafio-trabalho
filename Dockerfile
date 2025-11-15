FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y build-essential libpoppler-cpp-dev pkg-config poppler-utils && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /app

RUN python -c "import nltk; nltk.download('stopwords')"

ENV PORT=10000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:$PORT", "--workers", "2"]