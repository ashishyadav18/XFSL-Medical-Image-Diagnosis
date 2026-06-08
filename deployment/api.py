from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import shutil
import os
import uuid

from engine import analyze_image

# ==========================================
# APP
# ==========================================

app = FastAPI(
    title="XFSL Medical Diagnosis API",
    version="1.0.0"
)

# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict later for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# DIRECTORIES
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs"
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

# ==========================================
# STATIC HEATMAPS
# ==========================================

app.mount(
    "/outputs",
    StaticFiles(directory=OUTPUT_DIR),
    name="outputs"
)

# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/")
def root():

    return {
        "status": "online",
        "service": "XFSL Medical Diagnosis API"
    }

# ==========================================
# PREDICT
# ==========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    file_path = None

    try:

        # --------------------------
        # Validate file type
        # --------------------------

        if not file.content_type:

            return JSONResponse(
                status_code=400,
                content={
                    "error":
                    "Unable to determine file type."
                }
            )

        if not file.content_type.startswith(
            "image/"
        ):

            return JSONResponse(
                status_code=400,
                content={
                    "error":
                    "Only image files are allowed."
                }
            )

        # --------------------------
        # Unique filename
        # --------------------------

        extension = os.path.splitext(
            file.filename
        )[1]

        unique_name = (
            f"{uuid.uuid4()}{extension}"
        )

        file_path = os.path.join(
            UPLOAD_DIR,
            unique_name
        )

        # --------------------------
        # Save upload
        # --------------------------

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        # --------------------------
        # Run analysis
        # --------------------------

        result = analyze_image(
            file_path
        )

        # --------------------------
        # Convert heatmap path
        # --------------------------

        if result.get("heatmap"):

            heatmap_name = os.path.basename(
                result["heatmap"]
            )

            result["heatmap"] = (
                f"/outputs/{heatmap_name}"
            )

        # --------------------------
        # Success
        # --------------------------

        return JSONResponse(
            content=result
        )

    except Exception as e:

        print(
            "\nXFSL API ERROR:"
        )

        print(str(e))

        return JSONResponse(
            status_code=500,
            content={
                "error":
                "Prediction failed.",
                "details":
                str(e)
            }
        )

    finally:

        # Close uploaded file

        try:
            await file.close()
        except:
            pass