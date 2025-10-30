#!/usr/bin/env python3
from pathlib import Path

# Ajusta si cambiaste la ubicación
ROOT = Path("../input/face")

# Nombres de las 5 bases
CK = "ck+_frames_process_30fps"  # .png
ESETS = [
    "e-ck+_frames_process_30fps_t0_t",
    "e-ck+_frames_process_30fps_t0_t0",
    "e-ck+_frames_process_30fps_t_t",
    "e-ck+_frames_process_30fps_t_t0",
]

OUT_TXT = "common_jpg_files.txt"

def collect_stems(base_dir: Path, exts: tuple[str, ...]) -> set[str]:
    """
    Recorre recursivamente base_dir y devuelve un set de rutas relativas sin extensión
    (p.ej. 'Test_Set/1/S133_003/frame_0003') para archivos con extensión en exts.
    """
    stems = set()
    for ext in exts:
        for p in base_dir.rglob(f"*{ext}"):
            # ruta relativa respecto al directorio de la base (quitar extensión)
            rel = p.relative_to(base_dir).with_suffix("")  # quita .jpg/.png
            stems.add(rel.as_posix())
    return stems

def main():
    ck_dir = ROOT / CK / "cropped_frames"  # según tu estructura
    # Verificación rápida
    if not ck_dir.exists():
        raise SystemExit(f"No existe: {ck_dir}")

    e_dirs = [ROOT / e for e in ESETS]
    for d in e_dirs:
        if not d.exists():
            raise SystemExit(f"No existe: {d}")

    # 1) stems presentes como .png en ck+
    ck_stems = collect_stems(ck_dir, (".png",))
    # 2) stems presentes como .jpg en cada e-ck+
    e_stem_sets = [collect_stems(d, (".jpg",)) for d in e_dirs]

    # Intersección: stems que están en ck+ (.png) y en TODAS las e-ck+ (.jpg)
    common_stems = ck_stems.copy()
    for s in e_stem_sets:
        common_stems &= s

    # Escribimos .jpg relativos (porque pediste archivos .jpg)
    # Ej.: 'Test_Set/1/S133_003/frame_0003.jpg'
    out_lines = sorted(f"{stem}.jpg" for stem in common_stems)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")

    print(f"[OK] Coincidencias encontradas: {len(out_lines)}")
    print(f"[OK] Escrito en: {OUT_TXT}")

if __name__ == "__main__":
    main()
