import os
import cv2
import numpy as np
import random

def extract_features(image):
    sift_detector = cv2.SIFT_create()
    key_pts, descriptors = sift_detector.detectAndCompute(image, None)
    return key_pts, descriptors

def find_correspondences(kp_a, kp_b, desc_a, desc_b, ratio_thresh=0.75, ransac_thresh=4.0):
    matcher = cv2.BFMatcher()
    potential_matches = matcher.knnMatch(desc_a, desc_b, k=2)
    filtered_matches = [m for m, n in potential_matches if m.distance < ratio_thresh * n.distance]
    
    if len(filtered_matches) > 4:
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in filtered_matches])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in filtered_matches])
        homography_matrix, mask = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransac_thresh)
        return filtered_matches, homography_matrix, mask
    return None

def visualize_matches(img_a, img_b, kp_a, kp_b, matches, step, max_display=80):
    if len(matches) > max_display:
        matches = random.sample(matches, max_display)
    
    match_visual = cv2.drawMatches(img_a, kp_a, img_b, kp_b, matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    output_filename = os.path.join(output_folder, f"match_step_{step}.jpg")
    cv2.imwrite(output_filename, match_visual)
    print(f"Saved keypoint match visualization: {output_filename}")

def merge_images(img_a, img_b, homography):
    h1, w1 = img_a.shape[:2]
    h2, w2 = img_b.shape[:2]
    
    corners = np.array([[0, 0], [w1, 0], [w1, h1], [0, h1]], dtype=np.float32)
    transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    
    min_x = min(0, transformed_corners[:, 0].min())
    min_y = min(0, transformed_corners[:, 1].min())
    max_x = max(w2, transformed_corners[:, 0].max())
    max_y = max(h2, transformed_corners[:, 1].max())
    
    canvas_width = int(max_x - min_x)
    canvas_height = int(max_y - min_y)
    
    translation_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
    homography = translation_matrix @ homography
    
    result_canvas = cv2.warpPerspective(img_a, homography, (canvas_width, canvas_height))
    result_canvas[-int(min_y): -int(min_y) + h2, -int(min_x): -int(min_x) + w2] = img_b
    
    return result_canvas

def combine_images(image_list):
    base_image = image_list[0]
    for index in range(1, len(image_list)):
        next_image = image_list[index]
        
        kp1, desc1 = extract_features(base_image)
        kp2, desc2 = extract_features(next_image)
        
        match_result = find_correspondences(kp1, kp2, desc1, desc2)
        if match_result is None:
            print(f"Insufficient matches between images {index - 1} and {index}.")
            continue
        
        match_points, homography, _ = match_result
        print(f"Step {index}: Found {len(match_points)} valid matches.")
        
        visualize_matches(base_image, next_image, kp1, kp2, match_points, step=index)
        base_image = merge_images(base_image, next_image, homography)
    
    return base_image

def main(input_folder, output_folder):
    img_files = sorted(os.listdir(input_folder))
    img_list = [cv2.imread(os.path.join(input_folder, file)) for file in img_files if file.endswith(('.jpg', '.png'))]
    
    if len(img_list) < 2:
        print("Need at least two images for stitching.")
        return
    
    panorama_result = combine_images(img_list)
    
    if panorama_result is not None:
        height, width = panorama_result.shape[:2]
        scale = 1000 / width  # Resize to a max width of 1000px
        resized_output = cv2.resize(panorama_result, (1000, int(height * scale)))
        
        output_path = os.path.join(output_folder, "stitched_panorama.jpg")
        cv2.imwrite(output_path, resized_output)
        print(f"Final panorama saved as '{output_path}'.")
        cv2.imshow("Panorama View", resized_output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

input_folder = "input"
output_folder = "output"
main(input_folder, output_folder)

