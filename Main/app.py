from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel, PeftConfig
import torch
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuration CORS pour autoriser les requêtes depuis ton frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Remplace * par l’URL de ton frontend si besoin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement du modèle LoRA et du tokenizer
peft_model_dir = r"C:\Users\DELL\Desktop\pfe\FakeNews_V3\Main\checkpoint-11172"

config = PeftConfig.from_pretrained(peft_model_dir)
base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, peft_model_dir)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Mise en mode évaluation
model.eval()

# 📥 Modèle de requête
class TextInput(BaseModel):
    text: str

# ✅ Route de test
@app.get("/")
async def home():
    return {"message": "✅ API opérationnelle"}

# 🧠 Route de prédiction
@app.post("/predict")
async def predict(input: TextInput):
    # Aucun nettoyage ici car le dataset est déjà nettoyé avant entraînement
    cleaned_text = input.text.strip()

    # Encodage du texte
    inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Prédiction sans gradient
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).item()
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][prediction].item()

    label = "Real" if prediction == 1 else "Fake"

    return {
        "original_text": input.text,
        "prediction": label,
        "confidence": round(confidence * 100, 2)  # pourcentage de confiance
    }
