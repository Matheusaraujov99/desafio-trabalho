"""
- Verifica se os modelos (vectorizer + classifier) existem em /model
- Se não existirem, chama train.train() para treinar e gerar os arquivos
- Em seguida importa o app Flask de app.py e inicia o servidor
"""

import os
import sys
import time

MODEL_DIR = "model"
VEC_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")
CLF_FILE = os.path.join(MODEL_DIR, "classifier.pkl")

def ensure_models():
    """
    Verifica presença dos arquivos do modelo
    Se não existirem, tenta treinar chamando train.train().
    """
    if os.path.exists(VEC_FILE) and os.path.exists(CLF_FILE):
        print(f"[ok] modelos encontrados: {VEC_FILE}, {CLF_FILE}")
        return True

    print("[info] modelos não encontrados. Vamos treinar um modelo agora (train.py).")
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        import train
    except Exception as e:
        print("[erro] não foi possível importar train.py:", e)
        return False
    try:
        if hasattr(train, "train"):
            print("[info] chamando train.train() ...")
            train.train()
        else:
            print("[info] train.py não tem função train(), executando como script...")
            import runpy
            runpy.run_path(os.path.join(project_root, "train.py"), run_name="__main__")
    except Exception as e:
        print("[erro] falha ao treinar o modelo:", e)
        return False
    time.sleep(0.5)
    if os.path.exists(VEC_FILE) and os.path.exists(CLF_FILE):
        print("[ok] modelos gerados com sucesso.")
        return True
    else:
        print("[erro] depois do treinamento os modelos ainda não foram encontrados.")
        return False

def run_app():
    """
    Importa a aplicação Flask definida em app.py e a executa.
    """
    try:
        from app import app
    except Exception as e:
        print("[erro] falha ao importar 'app' de app.py:", e)
        return
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "1") == "1"

    print(f"[info] iniciando servidor Flask em http://0.0.0.0:{port} (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)


if __name__ == "__main__":
    ok = ensure_models()
    if not ok:
        print("[fatal] não foi possível garantir os modelos. Corrija os erros acima e tente novamente.")
        sys.exit(1)
    run_app()