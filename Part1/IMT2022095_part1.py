import cv2
import numpy as np
import os

def preprocess_image(image_path):
    """Preprocess the image: grayscale, resize, CLAHE, Gaussian blur, and Otsu thresholding."""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize image while maintaining aspect ratio
    new_width = 700
    aspect_ratio = new_width / img.shape[1]
    new_height = int(img.shape[0] * aspect_ratio)
    resized_gray = cv2.resize(gray, (new_width, new_height))

    # Apply CLAHE (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(resized_gray)

    # Apply Gaussian Blur
    blurred_img = cv2.GaussianBlur(enhanced_gray, (7, 7), 1)

    # Apply Otsu’s Binarization
    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply Morphological Closing
    kernel = np.ones((5, 5), np.uint8)
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    return binary_img, aspect_ratio, img

def detect_coins(binary_img, aspect_ratio):
    """Detect circular coin-like objects based on contour area, perimeter, circularity, and solidity."""
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coin_contours = []
    min_area = 400
    max_area = 0.1 * (binary_img.shape[0] * binary_img.shape[1])

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if min_area < area < max_area and perimeter > 30:
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0

            if len(cnt) >= 5:
                ellipse = cv2.fitEllipse(cnt)
                (center, axes, angle) = ellipse
                major_axis, minor_axis = max(axes), min(axes)
                axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
            else:
                axis_ratio = 0

            if (0.5 < circularity < 1.5 or 0.7 < axis_ratio < 1.3) and solidity > 0.9:
                coin_contours.append(cnt)

    return coin_contours

def draw_coin_contours(original_img, coin_contours, aspect_ratio):
    """Draw detected coin contours on the image."""
    img_with_contours = original_img.copy()
    scaled_contours = [np.array(cnt / aspect_ratio, dtype=np.int32) for cnt in coin_contours]
    cv2.drawContours(img_with_contours, scaled_contours, -1, (0, 255, 0), 3)
    return img_with_contours


def segment_coins(original_img, binary_img, coin_contours, aspect_ratio, output_folder, filename):
    """Segment each coin separately and save them as individual images."""
    binary_resized = cv2.resize(binary_img, (original_img.shape[1], original_img.shape[0]))
    mask = np.zeros_like(binary_resized)
    scaled_contours = [np.array(cnt / aspect_ratio, dtype=np.int32) for cnt in coin_contours]

    # Create and save images for each segmented coin
    for idx, cnt in enumerate(scaled_contours):
        single_mask = np.zeros_like(mask)
        cv2.drawContours(single_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        single_mask = cv2.cvtColor(single_mask, cv2.COLOR_GRAY2BGR)
        segmented_coin = cv2.bitwise_and(original_img, single_mask)

        # Crop the bounding box around the coin
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_coin = segmented_coin[y:y+h, x:x+w]

        # Save each coin separately
        coin_filename = os.path.join(output_folder, f"coin_{idx+1}_{filename}")
        cv2.imwrite(coin_filename, cropped_coin)

    # Create a combined mask for visualization
    cv2.drawContours(mask, scaled_contours, -1, 255, thickness=cv2.FILLED)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    segmented = cv2.bitwise_and(original_img, mask)
    bg = np.zeros_like(original_img)
    bg[mask[:, :, 0] == 255] = segmented[mask[:, :, 0] == 255]
    cv2.drawContours(bg, scaled_contours, -1, (255, 0, 0), 3)
    
    return bg


def process_folder(input_folder, output_folder):
    """Process all images in the input folder and save results in the output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            binary_img, aspect_ratio, original_img = preprocess_image(image_path)
            coin_contours = detect_coins(binary_img, aspect_ratio)
            outlined_coins = draw_coin_contours(original_img, coin_contours, aspect_ratio)
            segmented_coins = segment_coins(original_img, binary_img, coin_contours, aspect_ratio, output_folder, filename)


            # cv2.imwrite(os.path.join(output_folder, f'binary_{filename}'), binary_img)
            cv2.imwrite(os.path.join(output_folder, f'detected_{filename}'), outlined_coins)
            cv2.imwrite(os.path.join(output_folder, f'segmented_{filename}'), segmented_coins)
            print(f"Processed {filename}: {len(coin_contours)} coins detected.")

# Define input and output directories
input_folder = "./input_images"
output_folder = "./output_images"

process_folder(input_folder, output_folder)
