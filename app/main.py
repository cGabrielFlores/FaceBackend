import os
import io
import time
import uuid
import numpy as np
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import insightface
from insightface.app import FaceAnalysis
from supabase import create_client, Client
import cv2

# Carregar variáveis de ambiente
load_dotenv()

app = FastAPI(title="Face Recognition API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos
    allow_headers=["*"],  # Permitir todos os headers
)

# Inicializar Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("As variáveis de ambiente SUPABASE_URL e SUPABASE_KEY são obrigatórias")

supabase: Client = create_client(supabase_url, supabase_key)

# Inicializar o modelo do InsightFace
face_analyzer = None

@app.on_event("startup")
async def startup_event():
    global face_analyzer
    # Inicializar o modelo InsightFace com buffalo_l
    face_analyzer = FaceAnalysis(name="buffalo_l")
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

@app.get("/")
async def root():
    return {"message": "Face Recognition API está ativa", "version": "1.0.0"}

@app.post("/extract-embedding")
async def extract_embedding(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado")
    
    results = []
    
    for file in files:
        # Ler a imagem
        contents = await file.read()
        image_array = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if img is None:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Formato de arquivo não suportado"
            })
            continue
        
        # Detectar rosto
        faces = face_analyzer.get(img)
        
        if not faces:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Nenhum rosto detectado"
            })
            continue
        
        # Pegar apenas o primeiro rosto
        face = faces[0]
        
        # Extrair embedding (vetor facial)
        embedding = face.embedding.tolist()
        
        # Gerar um nome único para o arquivo
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Extrair o nome do arquivo (sem extensão)
        file_name_without_ext = os.path.splitext(file.filename)[0]
        
        # Fazer upload da imagem para o Supabase
        upload_path = f"faces/{unique_filename}"
        
        # Resetar o cursor do arquivo para o início
        file_stream = io.BytesIO(contents)
        
        # Upload para o bucket do Supabase
        storage_response = supabase.storage.from_("faces").upload(
            path=upload_path,
            file=file_stream,
            file_options={"content-type": file.content_type}
        )
        
        # Obter a URL pública da imagem
        public_url = supabase.storage.from_("faces").get_public_url(upload_path)
        
        # Salvar no banco de dados
        db_response = supabase.table("faces").insert({
            "image_url": public_url,
            "embedding": embedding,
            "name": file_name_without_ext  # Adicionando o nome extraído do arquivo
        }).execute()
        
        # Adicionar ao resultado
        results.append({
            "filename": file.filename,
            "status": "success",
            "public_url": public_url,
            "embedding_size": len(embedding),
            "name": file_name_without_ext
        })
    
    return {"processed_images": results}

@app.post("/compare")
async def compare_faces(file: UploadFile = File(...)):
    # Ler a imagem
    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado")
    
    # Detectar rosto
    faces = face_analyzer.get(img)
    
    if not faces:
        raise HTTPException(status_code=400, detail="Nenhum rosto detectado na imagem")
    
    # Pegar apenas o primeiro rosto
    face = faces[0]
    
    # Extrair embedding
    query_embedding = face.embedding
    
    # Buscar todos os embeddings do banco
    response = supabase.table("faces").select("*").execute()
    stored_faces = response.data
    
    if not stored_faces:
        return {"matches": [], "message": "Nenhuma face armazenada para comparação"}
    
    # Calcular distâncias euclidianas
    similarities = []
    
    for stored_face in stored_faces:
        stored_embedding = np.array(stored_face["embedding"])
        
        # Calcular distância euclidiana
        distance = np.linalg.norm(query_embedding - stored_embedding)
        
        # Converter para similaridade (quanto menor a distância, maior a similaridade)
        similarity = (1 - distance) * 100
        if similarity < 0:
            similarity = 0
        
        similarities.append({
            "image_url": stored_face["image_url"],
            "similarity": similarity,
            "id": stored_face.get("id"),
            "name": stored_face.get("name")  # Incluindo o nome na resposta
        })
    
    # Ordenar por similaridade (decrescente)
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Retornar as 5 mais similares
    top_matches = similarities[:5]
    
    return {
        "matches": top_matches
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=10000, reload=True)