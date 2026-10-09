import os
import re
from pathlib import Path
import cv2
import easyocr
import numpy as np
from ultralytics import YOLO


class VisionService:
    def __init__(self, model_path: str | None = None):
        configured_model = model_path or os.getenv("VISION_MODEL_PATH")
        self.model_path = Path(configured_model) if configured_model else Path(__file__).parent / "yolov8n.pt"
        if not self.model_path.is_absolute():
            self.model_path = (Path(__file__).parent / self.model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"No se encontro el modelo de vision: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.reader = easyocr.Reader(["es", "en"])

    @staticmethod
    def limpiar_placa(texto: str) -> str:
        limpio = re.sub(r"[^A-Z0-9]", "", texto.upper())
        for inicio in range(max(0, len(limpio) - 5)):
            candidato = limpio[inicio:inicio + 6]
            if len(candidato) != 6 or not re.match(r"^[A-Z]{3}", candidato):
                continue
            candidato = candidato[:3] + candidato[3:5].translate(
                str.maketrans({"O": "0", "Q": "0", "D": "0"})
            ) + candidato[5]
            if re.fullmatch(r"[A-Z]{3}\d{2}[A-Z0-9]", candidato):
                return candidato
        return ""

    def procesar_bytes(self, contenido: bytes) -> str:
        imagen = cv2.imdecode(np.frombuffer(contenido, np.uint8), cv2.IMREAD_COLOR)
        if imagen is None:
            raise ValueError("El archivo no contiene una imagen válida")
        return self._procesar(imagen)

    def procesar_archivo(self, image_path: str) -> str:
        imagen = cv2.imread(image_path)
        if imagen is None:
            raise ValueError(f"No se pudo cargar la imagen: {image_path}")
        return self._procesar(imagen)

    def _procesar(self, imagen: np.ndarray) -> str:
        alto, ancho = imagen.shape[:2]
        regiones = [imagen]

        # If the weights detect the motorcycle (or a plate), OCR the detected
        # boxes first. This also works with a custom plate-trained YOLO model.
        for resultado in self.model(imagen, verbose=False):
            if resultado.boxes is None:
                continue
            cajas = sorted(resultado.boxes, key=lambda caja: float(caja.conf[0]), reverse=True)
            for caja in cajas[:8]:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                margen_x, margen_y = int((x2 - x1) * 0.08), int((y2 - y1) * 0.08)
                x1, y1 = max(0, x1 - margen_x), max(0, y1 - margen_y)
                x2, y2 = min(ancho, x2 + margen_x), min(alto, y2 + margen_y)
                if x2 > x1 and y2 > y1:
                    regiones.append(imagen[y1:y2, x1:x2])

        for region in regiones:
            if region.size == 0:
                continue
            escala = min(3.0, max(1.0, 1000 / max(region.shape[:2])))
            grande = cv2.resize(region, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
            gris = cv2.cvtColor(grande, cv2.COLOR_BGR2GRAY)
            contraste = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
            variantes = [grande, contraste, cv2.threshold(contraste, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
            for variante in variantes:
                texto = "".join(self.reader.readtext(variante, detail=0, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"))
                placa = self.limpiar_placa(texto)
                if placa:
                    return placa

        return ""
