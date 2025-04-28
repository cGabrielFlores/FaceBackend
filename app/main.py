import os
import io
import time
import uuid
import numpy as np
from typing import List
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis
from supabase import create_client, Client
import cv2

# Carregar variáveis de ambiente
load_dotenv()

app = FastAPI(title="Face Recognition API")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar Supabase
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("As variáveis de ambiente SUPABASE_URL e SUPABASE_KEY são obrigatórias")

supabase: Client = create_client(supabase_url, supabase_key)

# Inicializar o modelo de embedding (sem detector)
face_embedder = None

@app.on_event("startup")
async def startup_event():
    global face_analyzer
    # Inicializar o modelo InsightFace com buffalo_l
    try:
        face_analyzer = FaceAnalysis(name="buffalo_l")
        face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"Erro ao inicializar o modelo InsightFace: {e}")
        raise HTTPException(status_code=500, detail="Erro ao inicializar o modelo InsightFace")


@app.get("/")
async def root():
    return {"message": "Face Recognition API está ativa", "version": "1.0.0"}

def detect_face_opencv(img):
    """Detecta o rosto usando Haar Cascade do OpenCV (super leve)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face_crop = img[y:y+h, x:x+w]
    return face_crop

@app.post("/extract-embedding")
async def extract_embedding(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Nenhum arquivo enviado")
    
    results = []
    
    for file in files:
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
        
        # Detectar rosto usando OpenCV
        face_crop = detect_face_opencv(img)
        
        if face_crop is None:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": "Nenhum rosto detectado"
            })
            continue
        
        # Redimensionar o rosto para 112x112 (o InsightFace espera isso)
        face_crop = cv2.resize(face_crop, (112, 112))
        
        # Extrair embedding
        embedding = face_embedder.get(face_crop).tolist()
        
        # Gerar nome único
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_name_without_ext = os.path.splitext(file.filename)[0]
        
        # Upload da imagem para Supabase
        upload_path = f"faces/{unique_filename}"
        file_stream = io.BytesIO(contents)
        
        storage_response = supabase.storage.from_("faces").upload(
            path=upload_path,
            file=file_stream,
            file_options={"content-type": file.content_type}
        )
        
        # Obter a URL pública
        public_url = supabase.storage.from_("faces").get_public_url(upload_path)
        
        # Salvar no banco
        db_response = supabase.table("faces").insert({
            "image_url": public_url,
            "embedding": embedding,
            "name": file_name_without_ext
        }).execute()
        
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
    contents = await file.read()
    image_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Formato de arquivo não suportado")
    
    # Detectar rosto
    face_crop = detect_face_opencv(img)
    
    if face_crop is None:
        raise HTTPException(status_code=400, detail="Nenhum rosto detectado na imagem")
    
    # Redimensionar para 112x112
    face_crop = cv2.resize(face_crop, (112, 112))
    
    # Extrair embedding
    query_embedding = face_embedder.get(face_crop)
    
    # Buscar embeddings salvos
    response = supabase.table("faces").select("*").execute()
    stored_faces = response.data
    
    if not stored_faces:
        return {"matches": [], "message": "Nenhuma face armazenada para comparação"}
    
    similarities = []
    
    for stored_face in stored_faces:
        stored_embedding = np.array(stored_face["embedding"])
        distance = np.linalg.norm(query_embedding - stored_embedding)
        similarity = (1 - distance) * 100
        if similarity < 0:
            similarity = 0
        similarities.append({
            "image_url": stored_face["image_url"],
            "similarity": similarity,
            "id": stored_face.get("id"),
            "name": stored_face.get("name")
        })
    
    similarities.sort(key=lambda x: x["similarity"], reverse=True)
    
    top_matches = similarities[:5]
    
    return {
        "matches": top_matches
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=10000, reload=True)
