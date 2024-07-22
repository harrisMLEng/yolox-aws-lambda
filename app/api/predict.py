import io

from fastapi import APIRouter, File, UploadFile
from PIL import Image

from app.models.prediction import Prediction

router = APIRouter()

# Load the YOLOX model
model = load_model()


@router.post("/predict", response_model=list[Prediction])
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # Perform inference
    results = model(image)

    # Process results
    predictions = []
    for det in results.xyxy[0]:
        predictions.append(Prediction(label=results.names[int(det[5])], confidence=det[4].item(), box=det[:4].tolist()))

    return predictions
