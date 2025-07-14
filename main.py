
import pickle
import re
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel



__version__ = "0.1.0"
BASE_DIR = Path(__file__).resolve(strict=True).parent
with open(f"{BASE_DIR}/model/trained_pipeline-{__version__}.pkl", "rb") as f:
    model = pickle.load(f)

classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]

def predict_pipeline(text):
    text = re.sub(r'[!@#$(),\n"%^*?:;~`0-9]', " ", text)
    text = re.sub(r"[[]]", " ", text)
    text = text.lower()
    pred = model.predict([text])
    return classes[pred[0]]

model_version = __version__
app = FastAPI()



class TextIn(BaseModel):
    text: str



class PredictionOut(BaseModel):
    language: str



@app.get("/")
def home():
    return {"health_check": "OK", "model_version": model_version}



@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    language = predict_pipeline(payload.text)
    return {"language": language}