#!/usr/bin/env python3
import os
from pathlib import Path
import cv2

# ==================== CONFIG ====================
# Raíz de la estructura 256x256 con intersección ya filtrada:
FACE256_ROOT = Path("../input/face_square256")

# CK+ (target, X2): PNG dentro de 'cropped_frames'
CK_DIR = FACE256_ROOT / "ck+_frames_process_30fps" / "cropped_frames"

# Variantes de eventos (source, X1)
EVENT_VARIANTS = {
    "X1_t0_t":  "e-ck+_frames_process_30fps_t0_t",
    "X1_t0_t0": "e-ck+_frames_process_30fps_t0_t0",
    "X1_t_t":   "e-ck+_frames_process_30fps_t_t",
    "X1_t_t0":  "e-ck+_frames_process_30fps_t_t0",
}

# Archivo con rutas relativas de la intersección (rutas .jpg tipo Test_Set/…/frame_XXXX.jpg)
INTERSECTION_LIST = Path("common_jpg_files.txt")

# Carpeta de salida principal (⚠️ ACTUALIZADA)
OUTPUT_BASE_DIR = Path("../input/faces_events_ck")

# Tamaño esperado (ya cuadradas y escaladas en el paso previo)
TARGET_SIZE = (256, 256)
# =================================================


def ensure_dirs():
    # X2
    for split in ["train", "val"]:
        (OUTPUT_BASE_DIR / "X2" / split).mkdir(parents=True, exist_ok=True)
    # Cada X1_*
    for x1_name in EVENT_VARIANTS.keys():
        for split in ["train", "val"]:
            (OUTPUT_BASE_DIR / x1_name / split).mkdir(parents=True, exist_ok=True)


def flattened_name(rel_jpg: str) -> str:
    """
    Convierte 'Test_Set/1/S133_003/frame_0003.jpg' en
    'Test_Set__1__S133_003__frame_0003.jpg'
    """
    return rel_jpg.replace("/", "__")


def read_rel_list():
    if not INTERSECTION_LIST.exists():
        raise SystemExit(f"No existe el archivo de intersección: {INTERSECTION_LIST}")
    lines = [ln.strip() for ln in INTERSECTION_LIST.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("El archivo de intersección está vacío.")
    return lines


def split_of(rel_path: Path) -> str:
    """
    Determina el split a partir del primer componente:
    'Train_Set' → 'train', 'Test_Set' → 'val'
    """
    first = rel_path.parts[0]
    if first == "Train_Set":
        return "train"
    elif first == "Test_Set":
        return "val"
    else:
        raise ValueError(f"Split desconocido en ruta: {rel_path}")


def load_img_rgb(path: Path):
    """Lee imagen con OpenCV y devuelve RGB (convierte de BGR o de escala de grises)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_jpg_rgb(img_rgb, out_path: Path, quality: int = 95):
    """Guarda imagen RGB como JPG (OpenCV espera BGR)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def main():
    # Chequeos
    if not CK_DIR.exists():
        raise SystemExit(f"No existe el directorio CK+: {CK_DIR}")
    for name, ds in EVENT_VARIANTS.items():
        if not (FACE256_ROOT / ds).exists():
            raise SystemExit(f"No existe el directorio de eventos '{name}': {FACE256_ROOT / ds}")

    ensure_dirs()
    rel_list = read_rel_list()

    # Contadores
    processed_X2 = 0
    processed_X1 = {name: 0 for name in EVENT_VARIANTS.keys()}
    missing_any = 0
    size_mismatch_any = 0

    for rel in rel_list:
        rel_path = Path(rel)  # p.ej. Test_Set/1/S133_003/frame_0003.jpg
        split = split_of(rel_path)
        out_name = flattened_name(rel)  # nombre aplanado final

        # --------- X2 (ck+) ---------
        src_X2 = CK_DIR / rel_path.with_suffix(".png")
        if not src_X2.exists():
            missing_any += 1
            continue

        x2 = load_img_rgb(src_X2)
        if x2 is None:
            missing_any += 1
            continue

        h2, w2 = x2.shape[:2]
        if (w2, h2) != TARGET_SIZE:
            x2 = cv2.resize(x2, TARGET_SIZE, interpolation=cv2.INTER_AREA)
            size_mismatch_any += 1

        out_X2 = OUTPUT_BASE_DIR / "X2" / split / out_name
        save_jpg_rgb(x2, out_X2)
        processed_X2 += 1

        # --------- Cada X1_* (eventos) ---------
        for x1_name, ds in EVENT_VARIANTS.items():
            src_X1 = (FACE256_ROOT / ds / rel_path)  # .jpg
            if not src_X1.exists():
                missing_any += 1
                continue

            x1 = load_img_rgb(src_X1)
            if x1 is None:
                missing_any += 1
                continue

            h1, w1 = x1.shape[:2]
            if (w1, h1) != TARGET_SIZE:
                x1 = cv2.resize(x1, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                size_mismatch_any += 1

            out_X1 = OUTPUT_BASE_DIR / x1_name / split / out_name
            save_jpg_rgb(x1, out_X1)
            processed_X1[x1_name] += 1

    # Resumen
    print(f"[OK] X2 (ck+) procesadas: {processed_X2}")
    for k, v in processed_X1.items():
        print(f"[OK] {k} procesadas: {v}")
    print(f"[WARN] Faltantes totales (cualquier causa): {missing_any}")
    print(f"[INFO] Reescaladas por tamaño ≠256: {size_mismatch_any}")
    print(f"[OUT] Raíz de salida: {OUTPUT_BASE_DIR.resolve()}")


if __name__ == "__main__":
    main()
