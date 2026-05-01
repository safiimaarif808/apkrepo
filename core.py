import cv2
import numpy as np
import os

# =====================================
# KONFIGURASI GLOBAL
# =====================================
SCALE = 0.3

TOP_CROP    = 0.03
BOTTOM_CROP = 0.97
LEFT_CROP   = 0.05
RIGHT_CROP  = 0.95

# ── RASIO DIKALIBRASI dari ground truth Afdal ──
# Tinggi asli: 93.7 cm  |  Berat asli: 18.8 kg
# tinggi_pixel terdeteksi ≈ 803px  →  93.7 / 803 ≈ 0.1167
# Nilai ini berlaku selama:
#   • resolusi kamera sama
#   • jarak kamera ke subjek sama
#   • SCALE = 0.3 tidak berubah
RASIO_CM_PER_PIXEL = 0.1167

LOWER_GREEN = np.array([35, 60, 60])
UPPER_GREEN = np.array([85, 255, 255])


def process_image(input_path, save_output=False, output_folder="output_hsv"):
    if save_output:
        os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        return {
            "success": False,
            "message": f"Gambar depan tidak ditemukan: {input_path}"
        }

    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width * SCALE), int(height * SCALE)))

    h, w = img.shape[:2]
    top_crop    = int(h * TOP_CROP)
    bottom_crop = int(h * BOTTOM_CROP)
    left_crop   = int(w * LEFT_CROP)
    right_crop  = int(w * RIGHT_CROP)

    crop_img = img[top_crop:bottom_crop, left_crop:right_crop]

    blur = cv2.GaussianBlur(crop_img, (5, 5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    body_mask  = cv2.bitwise_not(green_mask)

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN,  kernel_open)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(
        body_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_mask = np.zeros_like(body_mask)

    if not contours:
        return {"success": False, "message": "Kontur tubuh depan tidak ditemukan."}

    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_smooth)
    clean_mask = cv2.medianBlur(clean_mask, 5)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask   = cv2.erode(clean_mask, kernel_erode, iterations=1)

    contours_final, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours_final:
        return {"success": False, "message": "Kontur final depan tidak ditemukan."}

    largest_final      = max(contours_final, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest_final)

    margin_top    = 10
    margin_bottom = 15
    margin_x      = 20

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_top)
    x2 = min(crop_img.shape[1], x + w_box + margin_x)
    y2 = min(crop_img.shape[0], y + h_box + margin_bottom)

    crop_img   = crop_img[y1:y2, x1:x2]
    clean_mask = clean_mask[y1:y2, x1:x2]

    contours_final_crop, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours_final_crop:
        return {"success": False, "message": "Kontur final depan setelah auto-crop tidak ditemukan."}

    ys, xs = np.where(clean_mask == 255)

    if len(xs) == 0 or len(ys) == 0:
        return {"success": False, "message": "Mask tubuh depan kosong."}

    top_y    = np.min(ys)
    bottom_y = np.max(ys)
    left_x   = np.min(xs)
    right_x  = np.max(xs)

    tinggi_pixel = bottom_y - top_y
    lebar_pixel  = right_x  - left_x

    tinggi_cm = tinggi_pixel * RASIO_CM_PER_PIXEL
    lebar_cm  = lebar_pixel  * RASIO_CM_PER_PIXEL

    if save_output:
        result = crop_img.copy()
        cv2.drawContours(result, [max(contours_final_crop, key=cv2.contourArea)], -1, (0, 255, 0), 2)
        cv2.rectangle(result, (left_x, top_y), (right_x, bottom_y), (255, 0, 0), 2)

        center_x = (left_x + right_x) // 2
        cv2.line(result, (center_x, top_y), (center_x, bottom_y), (0, 0, 255), 2)

        cv2.putText(result, f"Tinggi: {round(tinggi_cm, 2)} cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result, f"Lebar: {round(lebar_cm, 2)} cm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(output_folder, "1_resize_crop.jpg"),  crop_img)
        cv2.imwrite(os.path.join(output_folder, "2_blur.jpg"),         blur)
        cv2.imwrite(os.path.join(output_folder, "3_green_mask.jpg"),   green_mask)
        cv2.imwrite(os.path.join(output_folder, "4_body_mask.jpg"),    body_mask)
        cv2.imwrite(os.path.join(output_folder, "5_clean_mask.jpg"),   clean_mask)
        cv2.imwrite(os.path.join(output_folder, "6_result.jpg"),       result)

    return {
        "success": True,
        "message": "Proses foto depan berhasil",
        "tinggi_pixel": int(tinggi_pixel),
        "tinggi_cm":    round(float(tinggi_cm), 2),
        "lebar_pixel":  int(lebar_pixel),
        "lebar_cm":     round(float(lebar_cm), 2),
    }


def process_image_side(input_path, save_output=False, output_folder="output_side"):
    if save_output:
        os.makedirs(output_folder, exist_ok=True)

    img = cv2.imread(input_path)
    if img is None:
        return {
            "success": False,
            "message": f"Gambar samping tidak ditemukan: {input_path}"
        }

    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width * SCALE), int(height * SCALE)))

    h, w = img.shape[:2]
    top_crop    = int(h * 0.08)
    bottom_crop = int(h * 0.97)
    left_crop   = int(w * 0.15)
    right_crop  = int(w * 0.90)

    crop_img = img[top_crop:bottom_crop, left_crop:right_crop]

    blur = cv2.GaussianBlur(crop_img, (5, 5), 0)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
    body_mask  = cv2.bitwise_not(green_mask)

    kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN,  kernel_open)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(
        body_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    clean_mask = np.zeros_like(body_mask)

    if not contours:
        return {"success": False, "message": "Kontur tubuh samping tidak ditemukan."}

    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_smooth)
    clean_mask = cv2.medianBlur(clean_mask, 5)

    contours_final, _ = cv2.findContours(
        clean_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours_final:
        return {"success": False, "message": "Kontur final samping tidak ditemukan."}

    largest_final      = max(contours_final, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(largest_final)

    margin_top    = 10
    margin_bottom = 20
    margin_x      = 10

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_top)
    x2 = min(clean_mask.shape[1], x + w_box + margin_x)
    y2 = min(clean_mask.shape[0], y + h_box + margin_bottom)

    clean_mask     = clean_mask[y1:y2, x1:x2]
    focused_result = crop_img[y1:y2, x1:x2].copy()

    ys, xs = np.where(clean_mask == 255)

    if len(xs) == 0 or len(ys) == 0:
        return {"success": False, "message": "Mask tubuh samping kosong."}

    top_y    = np.min(ys)
    bottom_y = np.max(ys)
    left_x   = np.min(xs)
    right_x  = np.max(xs)

    tinggi_pixel = bottom_y - top_y
    tebal_pixel  = right_x  - left_x

    tinggi_cm = tinggi_pixel * RASIO_CM_PER_PIXEL
    tebal_cm  = tebal_pixel  * RASIO_CM_PER_PIXEL

    if save_output:
        cv2.rectangle(focused_result, (left_x, top_y), (right_x, bottom_y), (255, 0, 0), 2)

        center_x = (left_x + right_x) // 2
        cv2.line(focused_result, (center_x, top_y), (center_x, bottom_y), (0, 0, 255), 2)

        cv2.putText(focused_result, f"Tinggi: {round(tinggi_cm, 2)} cm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(focused_result, f"Tebal: {round(tebal_cm, 2)} cm", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imwrite(os.path.join(output_folder, "1_resize_crop.jpg"), crop_img)
        cv2.imwrite(os.path.join(output_folder, "2_blur.jpg"),        blur)
        cv2.imwrite(os.path.join(output_folder, "3_green_mask.jpg"),  green_mask)
        cv2.imwrite(os.path.join(output_folder, "4_body_mask.jpg"),   body_mask)
        cv2.imwrite(os.path.join(output_folder, "5_clean_mask.jpg"),  clean_mask)
        cv2.imwrite(os.path.join(output_folder, "6_result.jpg"),      focused_result)

    return {
        "success": True,
        "message": "Proses foto samping berhasil",
        "tinggi_pixel": int(tinggi_pixel),
        "tinggi_cm":    round(float(tinggi_cm), 2),
        "tebal_pixel":  int(tebal_pixel),
        "tebal_cm":     round(float(tebal_cm), 2),
    }


def estimate_weight(tinggi_cm, lebar_cm, tebal_cm):
    """
    Estimasi berat badan balita menggunakan pendekatan:
    - Formula dasar WHO untuk balita: berat ≈ (tinggi/100)^2 × 16.5
      (BMI rata-rata balita normal ≈ 15-17)
    - Koreksi proporsional dari lebar dan tebal tubuh
    - Dikalibrasi dari ground truth Afdal: tinggi=93.7cm, berat=18.8kg

    Kalibrasi:
      berat_base = (93.7/100)^2 × 16.5 = 14.48 kg  (terlalu rendah)
      faktor_koreksi = 18.8 / 14.48 = 1.298
      → pakai BMI_TARGET = 16.5 × 1.298 ≈ 21.4

    Koreksi proporsional lebar & tebal:
      lebar normal balita usia 3-5 th ≈ 22-26 cm
      tebal normal balita usia 3-5 th ≈ 16-20 cm
      faktor_lebar = lebar_cm / 24
      faktor_tebal = tebal_cm / 18
      faktor_proporsi = (faktor_lebar + faktor_tebal) / 2
    """
    tinggi_m = tinggi_cm / 100.0

    # BMI target dikalibrasi dari Afdal
    BMI_TARGET = 21.4

    # Berat dasar dari tinggi
    berat_base = (tinggi_m ** 2) * BMI_TARGET

    # Koreksi proporsional tubuh (lebar & tebal)
    LEBAR_NORMAL = 24.0
    TEBAL_NORMAL = 18.0

    faktor_lebar  = lebar_cm / LEBAR_NORMAL
    faktor_tebal  = tebal_cm / TEBAL_NORMAL
    faktor_proporsi = (faktor_lebar + faktor_tebal) / 2.0

    # Batasi faktor agar tidak terlalu ekstrem (0.7 - 1.3)
    faktor_proporsi = max(0.7, min(1.3, faktor_proporsi))

    berat_final = berat_base * faktor_proporsi

    return round(float(berat_final), 2)


def process_images(front_path, side_path):
    try:
        front_result = process_image(front_path, save_output=False)

        if not front_result["success"]:
            return {
                "success": False,
                "message": front_result["message"]
            }

        side_result = process_image_side(side_path, save_output=False)

        if not side_result["success"]:
            return {
                "success": False,
                "message": side_result["message"]
            }

        tinggi_cm = front_result["tinggi_cm"]
        lebar_cm  = front_result["lebar_cm"]
        tebal_cm  = side_result["tebal_cm"]

        berat_kg = estimate_weight(
            tinggi_cm=tinggi_cm,
            lebar_cm=lebar_cm,
            tebal_cm=tebal_cm
        )

        return {
            "success": True,
            "message": "Gambar berhasil diproses",
            "data": {
                "height_cm":     round(float(tinggi_cm), 2),
                "weight_kg":     round(float(berat_kg), 2),
                "width_cm":      round(float(lebar_cm), 2),
                "thickness_cm":  round(float(tebal_cm), 2)
            }
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }