from fastapi import APIRouter, UploadFile, File
from app.models.yolox_model import load_model
from app.schemas.prediction import Prediction
from PIL import Image
import io

router = APIRouter()

# Load the YOLOX model
model = load_model()

@router.post("/", response_model=list[Prediction])
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Perform inference
    results = model(image)
    
    # Process results
    predictions = []
    for det in results.xyxy[0]:
        predictions.append(Prediction(
            label=results.names[int(det[5])],
            confidence=det[4].item(),
            box=det[:4].tolist()
        ))
    
    return predictions
