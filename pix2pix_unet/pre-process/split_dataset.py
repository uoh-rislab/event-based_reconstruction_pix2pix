import os
import cv2

# Directorios de entrada y salida
input_dirs = ["../data/maps/train", "../data/maps/val"]
output_base_dir = "../data/maps_processed"

# Crear directorios de salida para X1 y X2 en train y val
for split in ["train", "val"]:
    os.makedirs(os.path.join(output_base_dir, "X1", split), exist_ok=True)
    os.makedirs(os.path.join(output_base_dir, "X2", split), exist_ok=True)

# Procesar cada directorio (train/val)
for input_dir in input_dirs:
    split = os.path.basename(input_dir)  # Obtener "train" o "val"
    output_X1_dir = os.path.join(output_base_dir, "X1", split)
    output_X2_dir = os.path.join(output_base_dir, "X2", split)

    # Verificar si el directorio existe
    if not os.path.exists(input_dir):
        print(f"⚠️ El directorio {input_dir} no existe. Verifica la ruta.")
        continue

    # Listar todas las imágenes en el directorio actual
    for img_name in sorted(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name)

        # Leer la imagen con OpenCV
        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ No se pudo cargar la imagen: {img_path}")
            continue

        # Obtener dimensiones de la imagen
        h, w, _ = img.shape
        mid_w = w // 2

        # Dividir la imagen en entrada (izquierda) y objetivo (derecha)
        img_A = img[:, :mid_w, :]  # Izquierda
        img_B = img[:, mid_w:, :]  # Derecha

        # Guardar imágenes de entrada y objetivo en sus carpetas respectivas
        output_A_path = os.path.join(output_X1_dir, img_name)
        output_B_path = os.path.join(output_X2_dir, img_name)

        cv2.imwrite(output_A_path, img_A)
        cv2.imwrite(output_B_path, img_B)

    print(f"✅ Proceso completado para {split}: {len(os.listdir(input_dir))} imágenes procesadas.")
