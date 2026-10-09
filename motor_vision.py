from pathlib import Path

from vision_service import VisionService


def main() -> None:
    servicio = VisionService()
    carpeta = Path(__file__).parent / "img"
    for ruta in carpeta.iterdir():
        if ruta.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        placa = servicio.procesar_archivo(str(ruta))
        print(f"{ruta.name}: {placa or 'no detectada'}")


if __name__ == "__main__":
    main()