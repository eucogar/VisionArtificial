from fastapi import FastAPI, File, HTTPException, UploadFile
from dotenv import load_dotenv

from vision_service import VisionService

load_dotenv()
app = FastAPI(title="API de Vision Artificial", version="1.0.0")
vision = VisionService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/detectar")
async def detectar(
    imagen: UploadFile = File(...),
) -> dict:
    contenido = await imagen.read()
    if not contenido:
        raise HTTPException(status_code=400, detail="La imagen está vacía")

    try:
        placa = vision.procesar_bytes(contenido)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {"placa": placa or None}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
