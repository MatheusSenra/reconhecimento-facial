import base64
import io
import os

import faiss
import face_recognition
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

EMBEDDINGS_DIR = "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cache
nomes_cadastrados = []
index_faiss = None

def carregar_embeddings():
    embeddings = {}
    for filename in os.listdir(EMBEDDINGS_DIR):
        if filename.endswith(".npy"):
            nome = filename.replace(".npy", "")
            caminho = os.path.join(EMBEDDINGS_DIR, filename)
            embeddings[nome] = np.load(caminho)
    return embeddings

def construir_indice(embeddings_dict):
    global nomes_cadastrados, index_faiss

    nomes_cadastrados = list(embeddings_dict.keys())
    vetores = np.stack(list(embeddings_dict.values())).astype("float32")

    index_faiss = faiss.IndexFlatL2(vetores.shape[1])
    index_faiss.add(vetores)

def adicionar_ao_indice(nome: str, vetor: np.ndarray):
    global nomes_cadastrados, index_faiss

    vetor = vetor.astype("float32").reshape(1, -1)
    index_faiss.add(vetor)
    nomes_cadastrados.append(nome)

# Inicializa FAISS ao iniciar o app
@app.on_event("startup")
def startup_event():
    embeddings = carregar_embeddings()
    if embeddings:
        construir_indice(embeddings)

@app.post("/reconhecer")
async def reconhecer(request: Request):
    try:
        data = await request.json()
        imagem_base64 = data.get("foto")

        if not imagem_base64:
            return JSONResponse(content={"erro": "Campo 'foto' ausente"}, status_code=400)

        if "," in imagem_base64:
            imagem_base64 = imagem_base64.split(",")[1]

        imagem_bytes = base64.b64decode(imagem_base64)
        imagem = face_recognition.load_image_file(io.BytesIO(imagem_bytes))
        codificacoes = face_recognition.face_encodings(imagem)

        if not codificacoes:
            return JSONResponse(content={"erro": "Nenhum rosto detectado"}, status_code=400)

        if index_faiss is None or not nomes_cadastrados:
            return JSONResponse(content={"erro": "Nenhum rosto cadastrado ainda"}, status_code=400)

        rosto_recebido = codificacoes[0].astype("float32").reshape(1, -1)
        D, I = index_faiss.search(rosto_recebido, k=1)
        distancia = float(D[0][0])
        nome = nomes_cadastrados[I[0][0]]

        if distancia < 0.6:
            return {"pessoa": nome, "distancia": distancia}
        else:
            return {"pessoa": "Desconhecido", "distancia": distancia}

    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)

@app.post("/cadastrar")
async def cadastrar(request: Request):
    try:
        data = await request.json()
        imagem_base64 = data.get("foto")
        nome = data.get("nome")

        if not imagem_base64 or not nome:
            return JSONResponse(content={"erro": "Campos 'foto' e 'nome' são obrigatórios"}, status_code=400)

        if "," in imagem_base64:
            imagem_base64 = imagem_base64.split(",")[1]

        imagem_bytes = base64.b64decode(imagem_base64)
        imagem = face_recognition.load_image_file(io.BytesIO(imagem_bytes))
        codificacoes = face_recognition.face_encodings(imagem)

        if not codificacoes:
            return JSONResponse(content={"erro": "Nenhum rosto detectado"}, status_code=400)

        rosto_codificado = codificacoes[0]
        caminho_saida = os.path.join(EMBEDDINGS_DIR, f"{nome}.npy")
        np.save(caminho_saida, rosto_codificado)

        # Adiciona ao índice FAISS em tempo real
        adicionar_ao_indice(nome, rosto_codificado)

        return {"mensagem": f"Rosto de '{nome}' cadastrado com sucesso."}

    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)