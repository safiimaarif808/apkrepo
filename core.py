import cv2
import numpy as np
import os
from PIL import Image, ExifTags

# =====================================
# KONFIGURASI GLOBAL
# =====================================
TARGET_WIDTH = 480

TOP_CROP = 0.03
BOTTOM_CROP = 0.97
LEFT_CROP = 0.05
RIGHT_CROP = 0.95

RASIO_CM_PER_PIXEL = 0.22

LOWER_GREEN = np.array([35, 60, 60])
UPPER_GREEN = np.array([85, 255, 255])


def fix_exif_rotation(img_path):
    pil_img = Image.open(img_path)
    try:
        exif = pil_img._getexif()
        if exif:
            for tag, value in exif.items():
                if ExifTags.TAGS.get(tag) == 'Orientation':
                    if value == 3:
                        pil_img = pil_img.rotate(180, expand=True)
                    elif value == 6:
                        pil_img = pil_img.rotate(270, expand=True)
                    elif value == 8:
                        pil_img = pil_img.rotate(90, expand=True)
    except:
        pass
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def process_image(input_path, save_output=True, output_folder="output_hsv"):
    if save_output:
        os.makedirs(output_folder, exist_ok=True)

    img = fix_exif_rotation(input_path)
    if img is None:
        return {"success": False, "message": f"Gambar tidak ditemukan: {input_path}"}

    # STEP 1: RESIZE
    height, width = img.shape[:2]
    new_height = int(height * TARGET_WIDTH / width)
    img = cv2.resize(img, (TARGET_WIDTH, new_height))

    # STEP 2: CROP
    h, w = img.shape[:2]
    top_crop = int(h * TOP_CROP)
    bottom_crop = int(h * BOTTOM_CROP)
    left_crop = int(w * LEFT_CROP)
    right_crop = int(w * RIGHT_CROP)
    crop_img = img[top_crop:bottom_crop, left_crop:right_crop]

    # STEP 3: BLUR
    blur = cv2.GaussianBlur(crop_img, (5, 5), 0)

    # STEP 4: HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # STEP 5: MASK BACKGROUND HIJAU
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # STEP 6: BALIK JADI BODY MASK
    body_mask = cv2.bitwise_not(green_mask)

    # STEP 7: MORPHOLOGY
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel_open)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_close)

    # STEP 8: AMBIL KONTUR TERBESAR
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = crop_img.copy()
    clean_mask = np.zeros_like(body_mask)

    if not contours:
        return {"success": False, "message": "Kontur tubuh tidak ditemukan."}

    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_smooth)
    clean_mask = cv2.medianBlur(clean_mask, 5)
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean_mask = cv2.erode(clean_mask, kernel_erode, iterations=1)

    contours_final, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_final:
        return {"success": False, "message": "Kontur final tidak ditemukan."}

    largest_final = max(contours_final, key=cv2.contourArea)

    # AUTO-CROP
    x, y, w_box, h_box = cv2.boundingRect(largest_final)
    x1 = max(0, x - 20)
    y1 = max(0, y - 10)
    x2 = min(crop_img.shape[1], x + w_box + 20)
    y2 = min(crop_img.shape[0], y + h_box + 15)

    crop_img = crop_img[y1:y2, x1:x2]
    clean_mask = clean_mask[y1:y2, x1:x2]
    result = crop_img.copy()

    contours_final_crop, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_final_crop:
        return {"success": False, "message": "Kontur final setelah auto-crop tidak ditemukan."}

    largest_final = max(contours_final_crop, key=cv2.contourArea)
    cv2.drawContours(result, [largest_final], -1, (0, 255, 0), 2)

    # STEP 9: UKUR
    ys, xs = np.where(clean_mask == 255)
    top_y = np.min(ys)
    bottom_y = np.max(ys)
    left_x = np.min(xs)
    right_x = np.max(xs)

    tinggi_pixel = bottom_y - top_y
    lebar_pixel = right_x - left_x
    tinggi_cm = tinggi_pixel * RASIO_CM_PER_PIXEL
    lebar_cm = lebar_pixel * RASIO_CM_PER_PIXEL

    if save_output:
        cv2.imwrite(os.path.join(output_folder, "1_resize_crop.jpg"), crop_img)
        cv2.imwrite(os.path.join(output_folder, "2_blur.jpg"), blur)
        cv2.imwrite(os.path.join(output_folder, "3_green_mask.jpg"), green_mask)
        cv2.imwrite(os.path.join(output_folder, "4_body_mask.jpg"), body_mask)
        cv2.imwrite(os.path.join(output_folder, "5_clean_mask.jpg"), clean_mask)
        cv2.imwrite(os.path.join(output_folder, "6_result.jpg"), result)

    return {
        "success": True,
        "message": "Proses foto depan berhasil",
        "tinggi_pixel": int(tinggi_pixel),
        "tinggi_cm": round(tinggi_cm, 2),
        "lebar_pixel": int(lebar_pixel),
        "lebar_cm": round(lebar_cm, 2),
        "crop_img": crop_img,
        "green_mask": green_mask,
        "body_mask": body_mask,
        "clean_mask": clean_mask,
        "result_img": result
    }


def process_image_side(input_path, save_output=True, output_folder="output_side"):
    if save_output:
        os.makedirs(output_folder, exist_ok=True)

    img = fix_exif_rotation(input_path)
    if img is None:
        return {"success": False, "message": f"Gambar tidak ditemukan: {input_path}"}

    # STEP 1: RESIZE
    height, width = img.shape[:2]
    new_height = int(height * TARGET_WIDTH / width)
    img = cv2.resize(img, (TARGET_WIDTH, new_height))

    # STEP 2: CROP
    h, w = img.shape[:2]
    top_crop = int(h * 0.08)
    bottom_crop = int(h * 0.97)
    left_crop = int(w * 0.15)
    right_crop = int(w * 0.90)
    crop_img = img[top_crop:bottom_crop, left_crop:right_crop]

    # STEP 3: BLUR
    blur = cv2.GaussianBlur(crop_img, (5, 5), 0)

    # STEP 4: HSV
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # STEP 5: MASK BACKGROUND HIJAU
    green_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    # STEP 6: BALIK JADI BODY MASK
    body_mask = cv2.bitwise_not(green_mask)

    # STEP 7: MORPHOLOGY
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_OPEN, kernel_open)
    body_mask = cv2.morphologyEx(body_mask, cv2.MORPH_CLOSE, kernel_close)

    # STEP 8: AMBIL KONTUR TERBESAR
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = crop_img.copy()
    clean_mask = np.zeros_like(body_mask)

    if not contours:
        return {"success": False, "message": "Kontur tubuh samping tidak ditemukan."}

    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_close)
    kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel_smooth)
    clean_mask = cv2.medianBlur(clean_mask, 5)

    contours_final, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours_final:
        return {"success": False, "message": "Kontur final samping tidak ditemukan."}

    largest_final = max(contours_final, key=cv2.contourArea)
    cv2.drawContours(result, [largest_final], -1, (0, 255, 0), 2)

    # AUTO-CROP
    x, y, w_box, h_box = cv2.boundingRect(largest_final)
    x1 = max(0, x - 10)
    y1 = max(0, y - 10)
    x2 = min(clean_mask.shape[1], x + w_box + 10)
    y2 = min(clean_mask.shape[0], y + h_box + 20)

    focused_mask = clean_mask[y1:y2, x1:x2]
    focused_result = result[y1:y2, x1:x2]

    ys, xs = np.where(focused_mask == 255)
    top_y = np.min(ys)
    bottom_y = np.max(ys)
    left_x = np.min(xs)
    right_x = np.max(xs)

    tinggi_pixel = bottom_y - top_y
    tebal_pixel = right_x - left_x
    tinggi_cm = tinggi_pixel * RASIO_CM_PER_PIXEL
    tebal_cm = tebal_pixel * RASIO_CM_PER_PIXEL

    result = focused_result
    clean_mask = focused_mask

    if save_output:
        cv2.imwrite(os.path.join(output_folder, "1_resize_crop.jpg"), crop_img)
        cv2.imwrite(os.path.join(output_folder, "2_blur.jpg"), blur)
        cv2.imwrite(os.path.join(output_folder, "3_green_mask.jpg"), green_mask)
        cv2.imwrite(os.path.join(output_folder, "4_body_mask.jpg"), body_mask)
        cv2.imwrite(os.path.join(output_folder, "5_clean_mask.jpg"), clean_mask)
        cv2.imwrite(os.path.join(output_folder, "6_result.jpg"), result)

    return {
        "success": True,
        "message": "Proses foto samping berhasil",
        "tinggi_pixel": int(tinggi_pixel),
        "tinggi_cm": round(tinggi_cm, 2),
        "tebal_pixel": int(tebal_pixel),
        "tebal_cm": round(tebal_cm, 2),
        "crop_img": crop_img,
        "green_mask": green_mask,
        "body_mask": body_mask,
        "clean_mask": clean_mask,
        "result_img": result
    }


def estimate_weight(tinggi_cm, lebar_cm, tebal_cm):
    berat = 0.23 * tinggi_cm - 8.5
    return berat