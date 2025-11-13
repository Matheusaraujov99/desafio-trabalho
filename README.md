# Email Classifier (Produtivo / Improdutivo)

## Resumo
Aplicação web simples que classifica emails em **Produtivo** ou **Improdutivo** e sugere uma resposta automática. Backend em Python (Flask) e classifier TF-IDF + LogisticRegression.

## Estrutura
Veja `train.py`, `app.py`, `utils.py`, `data/examples.csv`, `templates/`, `static/`.

## Setup local
1. Criar virtualenv:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix / Mac
source venv/bin/activate