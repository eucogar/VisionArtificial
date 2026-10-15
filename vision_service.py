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
        service_dir = Path(__file__).parent
        default_model = service_dir / "plate_detector.pt"
        if configured_model:
            self.model_path = Path(configured_model)
        elif default_model.is_file():
            self.model_path = default_model
        else:
            self.model_path = service_dir / "yolov8n.pt"
        if not self.model_path.is_absolute():
            self.model_path = (Path(__file__).parent / self.model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"No se encontro el modelo de vision: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.reader = easyocr.Reader(["es", "en"])

    @staticmethod
    def limpiar_placa(texto: str) -> str:
        limpio = re.sub(r"[^A-Z0-9]", "", texto.upper())
        # Colombian motorcycle plate: exactly 3 letters, 2 digits, 1 letter.
        # Do not slide over longer words such as COLOMBIA.
        if len(limpio) == 7 and re.fullmatch(r"[A-Z]{4}\d{2}[A-Z]", limpio):
            # EasyOCR can insert one false letter at the letter/digit boundary.
            limpio = limpio[:3] + limpio[4:]
        if len(limpio) != 6 or not re.fullmatch(r"[A-Z]{3}[A-Z0-9]{2}[A-Z]", limpio):
            return ""
        numeros = limpio[3:5].translate(str.maketrans({
            "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
            "Z": "2", "S": "5", "B": "8", "G": "6",
        }))
        candidato = limpio[:3] + numeros + limpio[5]
        return candidato if re.fullmatch(r"[A-Z]{3}\d{2}[A-Z]", candidato) else ""

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
        # Try plate-specific detector crops before running OCR over the full frame.
        regiones = []

        # If the weights detect the motorcycle (or a plate), OCR the detected
        # boxes first. This also works with a custom plate-trained YOLO model.
        for resultado in self.model(imagen, verbose=False):
            if resultado.boxes is None:
                continue
            cajas = sorted(resultado.boxes, key=lambda caja: float(caja.conf[0]), reverse=True)
            for caja in cajas[:8]:
                x1, y1, x2, y2 = map(int, caja.xyxy[0])
                # Keep OCR inside the detected plate; margins bring in body text.
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(ancho, x2), min(alto, y2)
                if x2 > x1 and y2 > y1:
                    regiones.append(imagen[y1:y2, x1:x2])

        # With a plate-specific model, no detection means no OCR candidate.
        # Reading the whole scene can turn the small word COLOMBIA into a plate.
        if not regiones and not any("plate" in str(name).lower() for name in self.model.names.values()):
            regiones.append(imagen)

        for region in regiones:
            if region.size == 0:
                continue
            escala = min(3.0, max(1.0, 1000 / max(region.shape[:2])))
            grande = cv2.resize(region, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
            gris = cv2.cvtColor(grande, cv2.COLOR_BGR2GRAY)
            contraste = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gris)
            variantes = [grande, contraste, cv2.threshold(contraste, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
            for variante in variantes:
                lecturas = self.reader.readtext(
                    variante, detail=1, paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", decoder="greedy",
                )
                alto_variante = variante.shape[0]
                candidatas = []
                for caja_texto, texto, confianza in lecturas:
                    centro_y = sum(punto[1] for punto in caja_texto) / len(caja_texto)
                    # "COLOMBIA" is printed in a separate, lower row on the plate.
                    if centro_y >= alto_variante * 0.72 or confianza < 0.20:
                        continue
                    placa = self.limpiar_placa(texto)
                    if placa:
                        candidatas.append((float(confianza), placa))
                if candidatas:
                    return max(candidatas, key=lambda item: item[0])[1]

        return ""
