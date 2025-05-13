# 🧠 Reconhecimento Facial com FastAPI + FAISS

Este é um projeto de API para **reconhecimento facial** usando a biblioteca `face_recognition`, otimizado com **FAISS** para busca vetorial eficiente, e empacotado com **Docker** para fácil implantação.

---

## 🚀 Funcionalidades

- 📷 Recebe imagens base64 e identifica rostos.
- ➕ Permite cadastrar novos rostos com nome associado.
- ⚡ Alta performance com **FAISS** para comparação vetorial.
- 🔒 Armazena embeddings localmente (ou em banco externo).
- 🌐 API REST pronta para integrar com frontend ou apps mobile.

---

## 🧰 Tecnologias Utilizadas

- [FastAPI](https://fastapi.tiangolo.com/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [NumPy](https://numpy.org/)
- [Docker](https://www.docker.com/)
- [Uvicorn](https://www.uvicorn.org/)

---

## 📦 Instalação Local

### 1. Clone o repositório

```bash
git clone https://github.com/MatheusSenra/reconhecimento-facial.git
cd reconhecimento-facial
```

### 2. Crie um ambiente virtual (opcional, mas recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Execute o servidor

```bash
uvicorn main:app --reload
```

Acesse em: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Usando com Docker

### 1. Build da imagem

```bash
docker build -t face-api .
```

### 2. Executar o container

```bash
docker run -d -p 8000:8000 face-api
```

---

## 🛠 Endpoints Principais

### 🔍 `POST /reconhecer`

Recebe uma imagem em **base64** e retorna o nome da pessoa reconhecida.

**Body (JSON):**

```json
{
  "foto": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

---

### 📥 `POST /cadastrar`

Cadastra um novo rosto com nome, a partir da imagem base64.

**Body (JSON):**

```json
{
  "nome": "Matheus Senra",
  "foto": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
}
```

---

## 🗃 Estrutura de Pastas

```
📁 embeddings/         # Armazena os arquivos .npy com os vetores dos rostos
📄 main.py             # Código principal da API FastAPI
📄 requirements.txt    # Dependências Python
📄 Dockerfile          # Imagem Docker
📄 .gitignore
📄 README.md
```

---

## ⚠️ Observações

- Os embeddings são armazenados localmente em `embeddings/`, que está no `.gitignore` por segurança.
- Para produção, considere mover os vetores para um banco vetorial ou armazenamento seguro.

---

## 📄 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais informações.

---

## ✨ Autor

Desenvolvido com 💙 por [Matheus Senra](https://github.com/MatheusSenra)
