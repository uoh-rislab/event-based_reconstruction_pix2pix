#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageOps

# -------------------- CONFIG --------------------
# Directorio raíz de entrada (según tu estructura)
IN_ROOT = Path("../input/face")

# Archivo con las rutas relativas .jpg de la intersección
INTERSECTION_LIST = Path("common_jpg_files.txt")

# Directorio raíz de salida (nuevo)
OUT_ROOT = Path("../output/face_square256")

# Nombres de datasets
CK = "ck+_frames_process_30fps"  # fuente .png (dentro de 'cropped_frames')
ESETS = [
    "e-ck+_frames_process_30fps_t0_t",
    "e-ck+_frames_process_30fps_t0_t0",
    "e-ck+_frames_process_30fps_t_t",
    "e-ck+_frames_process_30fps_t_t0",
]

TARGET_SIZE = (256, 256)

# ------------------------------------------------

def make_square(img: Image.Image, fill_rgb=(0, 0, 0)):
    """Devuelve una imagen cuadrada con padding centrado."""
    w, h = img.size
    if w == h:
        return img

    # Determinar color de relleno según modo
    if img.mode == "RGBA":
        fill = (0, 0, 0, 0)  # mantiene transparencia en PNG si existiera
    elif img.mode == "L":
        fill = 0
    else:
        fill = fill_rgb  # RGB u otros → negro

    if w < h:
        pad = ( (h - w) // 2, 0, (h - w) - (h - w)//2, 0 )  # left, top, right, bottom
    else:  # w > h
        pad = ( 0, (w - h)//2, 0, (w - h) - (w - h)//2 )

    return ImageOps.expand(img, border=pad, fill=fill)

def load_image(path: Path, for_jpeg: bool):
    """
    Abre una imagen conservando/ajustando modos:
      - Para JPG, convierte a RGB si no lo está.
      - Para PNG, respeta RGBA/L/ RGB según fuente.
    """
    img = Image.open(path)
    if for_jpeg:
        # JPEG no soporta alpha/ paletas → a RGB
        if img.mode not in ("RGB",):
            img = img.convert("RGB")
    else:
        # Para PNG aceptamos L, RGB, RGBA. Si es P → convertir a RGBA para seguridad.
        if img.mode == "P":
            img = img.convert("RGBA")
    return img

def save_image(img: Image.Image, out_path: Path, is_png: bool):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if is_png:
        img.save(out_path, format="PNG", optimize=True)
    else:
        img.save(out_path, format="JPEG", quality=95, optimize=True, progressive=True)

def process_dataset(stems_jpg_rel: list[str]):
    """
    Recorre la lista de rutas relativas (.jpg) y genera versiones 256x256
    con padding en las 5 bases.
    - ck+ lee/escribe .png dentro de 'cropped_frames'
    - e-ck+ lee/escribe .jpg
    """
    missing = 0
    written = 0

    # ---- CK+ (.png) ----
    ck_in_base = IN_ROOT / CK / "cropped_frames"
    ck_out_base = OUT_ROOT / CK / "cropped_frames"
    for stem_jpg in stems_jpg_rel:
        stem = Path(stem_jpg).with_suffix("")    # quita .jpg → ruta relativa sin extensión
        src = ck_in_base / f"{stem}.png"
        dst = ck_out_base / f"{stem}.png"

        if not src.exists():
            missing += 1
            continue

        img = load_image(src, for_jpeg=False)
        img_sq = make_square(img)
        img_256 = img_sq.resize(TARGET_SIZE, Image.LANCZOS)
        save_image(img_256, dst, is_png=True)
        written += 1

    # ---- e-CK+ (.jpg) ----
    for ds in ESETS:
        in_base = IN_ROOT / ds
        out_base = OUT_ROOT / ds
        for stem_jpg in stems_jpg_rel:
            stem = Path(stem_jpg).with_suffix("")   # sin extensión
            src = in_base / f"{stem}.jpg"
            dst = out_base / f"{stem}.jpg"

            if not src.exists():
                missing += 1
                continue

            img = load_image(src, for_jpeg=True)
            img_sq = make_square(img)
            img_256 = img_sq.resize(TARGET_SIZE, Image.LANCZOS)
            save_image(img_256, dst, is_png=False)
            written += 1

    print(f"[DONE] Escrito: {written} archivos. Faltantes (no encontrados): {missing}")

def main():
    if not INTERSECTION_LIST.exists():
        raise SystemExit(f"No se encuentra el archivo de intersección: {INTERSECTION_LIST}")

    # Leer líneas (rutas relativas .jpg)
    with open(INTERSECTION_LIST, "r", encoding="utf-8") as f:
        stems_jpg_rel = [ln.strip() for ln in f if ln.strip()]

    if not stems_jpg_rel:
        raise SystemExit("El archivo de intersección está vacío.")

    # Verificaciones rápidas de existencia de raíces
    if not (IN_ROOT / CK / "cropped_frames").exists():
        raise SystemExit(f"No existe: {IN_ROOT / CK / 'cropped_frames'}")
    for ds in ESETS:
        if not (IN_ROOT / ds).exists():
            raise SystemExit(f"No existe: {IN_ROOT / ds}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    process_dataset(stems_jpg_rel)

if __name__ == "__main__":
    main()
