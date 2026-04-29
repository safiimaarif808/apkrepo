from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid

from core import process_image, process_image_side, estimate_weight

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/process-images")
async def process_images(
    front_image: UploadFile = File(...),
    side_image: UploadFile = File(...),
):
    try:
        front_path = os.path.join(UPLOAD_DIR, f"front_{uuid.uuid4()}.jpg")
        side_path = os.path.join(UPLOAD_DIR, f"side_{uuid.uuid4()}.jpg")

        with open(front_path, "wb") as buffer:
            shutil.copyfileobj(front_image.file, buffer)

        with open(side_path, "wb") as buffer:
            shutil.copyfileobj(side_image.file, buffer)

        front_result = process_image(front_path)
        if not front_result["success"]:
            return {
                "success": False,
                "message": front_result["message"],
            }

        side_result = process_image_side(side_path)
        if not side_result["success"]:
            return {
                "success": False,
                "message": side_result["message"],
            }

        tinggi_cm = front_result["tinggi_cm"]
        lebar_cm = front_result["lebar_cm"]
        tebal_cm = side_result["tebal_cm"]

        berat_kg = estimate_weight(
            tinggi_cm=tinggi_cm,
            lebar_cm=lebar_cm,
            tebal_cm=tebal_cm,
        )

        return {
            "success": True,
            "data": {
                "height_cm": tinggi_cm,
                "weight_kg": round(berat_kg, 2),
                "width_cm": lebar_cm,
                "thickness_cm": tebal_cm,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e),
        }