from fastapi import FastAPI

from app.api.predict import predict

app = FastAPI()

# Include the prediction endpoint
app.include_router(predict.router, prefix="/predict", tags=["predict"])
