import os
import tempfile
from flask import Flask, request, render_template, redirect, url_for, flash
from joblib import load
from utils import read_pdf, read_txt, normalize_text
import openai

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "mude-essa-senha")

# Caminhos dos modelos
VEC_PATH = 'model/vectorizer.pkl'
CLF_PATH = 'model/classifier.pkl'

# Verificação de existência dos modelos
if not os.path.exists(VEC_PATH) or not os.path.exists(CLF_PATH):
    raise RuntimeError("Modelos não encontrados. Rode 'python train.py' antes de iniciar a aplicação.")

vectorizer = load(VEC_PATH)
clf = load(CLF_PATH)

OPENAI_KEY = os.getenv('OPENAI_API_KEY')
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY


def classify_text(text):
    """Normaliza texto e retorna categoria + probabilidade"""
    clean = normalize_text(text)
    X = vectorizer.transform([clean])
    pred = clf.predict(X)[0]
    prob = clf.predict_proba(X).max()
    return pred, float(prob)


def generate_response_openai(text, category):
    """Gera resposta via OpenAI, caso API key exista"""
    if not OPENAI_KEY:
        return None

    prompt = (
        f"Você é um assistente que escreve respostas profissionais em português.\n\n"
        f"Email recebido:\n\"\"\"{text}\"\"\"\n\n"
        f"Categoria: {category}\n\n"
        "Escreva uma resposta objetiva com 2 a 4 frases."
    )

    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.2
        )
        return resp['choices'][0]['message']['content'].strip()

    except Exception as e:
        print("Erro OpenAI:", e)
        return None


def generate_response_template(category, text):
    """Gera resposta padrão caso OpenAI não esteja configurado."""
    if category == "Produtivo":
        return (
            "Olá,\n\n"
            "Recebemos sua mensagem e iniciamos a análise. "
            "Se possível, envie detalhes adicionais como prints ou número de protocolo. "
            "Retornaremos em breve.\n\nAtenciosamente,"
        )
    else:
        return (
            "Olá,\n\nAgradecemos o contato. Registramos sua mensagem. "
            "Se precisar de suporte, por favor descreva sua solicitação.\n\nAtenciosamente,"
        )


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        input_text = request.form.get('email_text', '').strip()
        file = request.files.get('file')
        text = input_text

        # Prioriza arquivo, caso enviado
        if file and file.filename != '':
            fname = file.filename.lower()
            tmp = tempfile.NamedTemporaryFile(delete=False)
            file.save(tmp.name)

            if fname.endswith('.pdf'):
                text = read_pdf(tmp.name)
            elif fname.endswith('.txt'):
                text = read_txt(tmp.name)
            else:
                flash("Formato inválido. Envie .txt ou .pdf.")
                return redirect(url_for('index'))

        if not text:
            flash("Cole um texto ou envie um arquivo (.txt ou .pdf).")
            return redirect(url_for('index'))

        category, prob = classify_text(text)

        suggested = generate_response_openai(text, category)
        if not suggested:
            suggested = generate_response_template(category, text)

        result = {
            'category': category,
            'probability': round(prob, 3),
            'suggested_response': suggested,
            'original_text': text
        }

    return render_template('index.html', result=result)


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.getenv("PORT", 5000)),
        debug=False
    )