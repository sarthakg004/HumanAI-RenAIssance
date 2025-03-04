import pandas
import fitz
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import unsharp_mask
import os
import string
from docx import Document
import math
import os
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A    
import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
import random
import shutil
import random
import shutil
import re
from skimage import exposure, restoration

def pdf_to_images(pdf_path, output_dir):
    
    os.makedirs(output_dir, exist_ok=True)
    pdf_doc = fitz.open(pdf_path)
    image_counter = 1  # Start naming from 1

    for page_num in range(len(pdf_doc)):
        page = pdf_doc.load_page(page_num)
        pix = page.get_pixmap()  # Render page as image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # If the page is significantly wider than tall, assume it's a double-page
        if img.width > img.height * 1.2:
            left = img.crop((0, 0, img.width // 2, img.height))  # Left half
            right = img.crop((img.width // 2, 0, img.width, img.height))  # Right half

            left.save(os.path.join(output_dir, f"{image_counter}.png"), format="PNG")
            image_counter += 1
            right.save(os.path.join(output_dir, f"{image_counter}.png"), format="PNG")
        else:
            img.save(os.path.join(output_dir, f"{image_counter}.png"), format="PNG")

        image_counter += 1  # Increment for next image

    
    
##################Image Processing######################
def load_image(image_path):
    """Loads an image from a given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image

def convert_to_grayscale(image):
    """Converts an image to grayscale if it's not already."""
    if len(image.shape) == 3:  # Check if the image is in color (BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image  # Already grayscale

def resize_image(image, height=None, width=None):
    """Resizes image while maintaining aspect ratio if only one dimension is specified.
    Uses Lanczos interpolation to better preserve text quality."""
    if height is None and width is None:
        return image
    
    h, w = image.shape[:2]
    if height is None:
        aspect_ratio = width / w
        height = int(h * aspect_ratio)
    elif width is None:
        aspect_ratio = height / h
        width = int(w * aspect_ratio)
    
    # Use Lanczos interpolation for higher quality resizing when upscaling
    # Use INTER_AREA for downscaling
    if height > h or width > w:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    else:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def crop_image_percentage(image, image_path, crop_odd, crop_even):
    """Crops image based on percentages provided, different for odd/even pages."""
    height, width = image.shape[:2]

    # Determine if the page is odd or even based on filename
    filename = os.path.basename(image_path)
    page_number_match = re.search(r'\d+', filename)
    
    if page_number_match:
        page_number = int(page_number_match.group())
        is_odd = page_number % 2 == 1
    else:
        print(f"Warning: Could not determine page number for {filename}, using odd-page cropping by default.")
        is_odd = True

    # Select cropping values
    crop_values = crop_odd if is_odd else crop_even

    # Convert percentage to absolute pixels
    top = int(height * crop_values['top'] / 100)
    bottom = int(height * crop_values['bottom'] / 100)
    left = int(width * crop_values['left'] / 100)
    right = int(width * crop_values['right'] / 100)

    # Compute cropping bounds
    bottom = height - bottom if bottom > 0 else height
    right = width - right if right > 0 else width

    # Crop the image
    cropped = image[top:bottom, left:right]
    
    return cropped

def correct_illumination(image):
    """Corrects uneven illumination using background subtraction."""
    # Create a morphological kernel for background estimation
    kernel_size = max(3, min(image.shape) // 30)
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # Ensure odd kernel size
    
    # Estimate background
    background = cv2.morphologyEx(image, cv2.MORPH_DILATE, 
                                 np.ones((kernel_size, kernel_size), np.uint8))
    background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)
    
    # Subtract background and normalize
    diff_img = cv2.subtract(255, cv2.subtract(background, image))
    
    # Apply normalization to enhance contrast
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, 
                            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    return norm_img

def enhance_contrast(image, method="clahe"):
    """Enhances contrast using various methods."""
    if method == "clahe":
        # CLAHE is good for localized contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
    elif method == "adaptive_eq":
        # Adaptive histogram equalization
        result = exposure.equalize_adapthist(image, clip_limit=0.03) * 255
        return result.astype(np.uint8)
    elif method == "stretch":
        # Contrast stretching
        p2, p98 = np.percentile(image, (2, 98))
        result = exposure.rescale_intensity(image, in_range=(p2, p98), out_range=(0, 255))
        return result.astype(np.uint8)
    
    return image

def denoise_image(image, method="nlm"):
    """Applies different denoising methods."""
    if method == "gaussian":
        return cv2.GaussianBlur(image, (3, 3), 0)
    elif method == "bilateral":
        # Bilateral filter preserves edges better
        return cv2.bilateralFilter(image, 9, 75, 75)
    elif method == "nlm":
        # Non-local means denoising - good for preserving text details
        return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    elif method == "wiener":
        # Wiener filter for restoration
        result = restoration.wiener(image, psf=np.ones((3, 3)) / 9)
        return (result * 255).astype(np.uint8)
    elif method == "tv":
        # Total variation denoising, effective for preserving edges
        result = restoration.denoise_tv_chambolle(image, weight=0.1)
        return (result * 255).astype(np.uint8)
    
    return image

def sharpen_image(image, method="unsharp"):
    """Sharpens an image using various methods."""
    if method == "unsharp":
        # Unsharp masking
        result = unsharp_mask(image, radius=1.5, amount=1.8, preserve_range=True)
        return result.astype(np.uint8)
    elif method == "laplacian":
        # Laplacian sharpening
        laplacian = cv2.Laplacian(image, cv2.CV_8U)
        return cv2.add(image, laplacian)
    elif method == "custom":
        # Custom sharpening kernel
        kernel = np.array([[0,-2,0], 
                          [-2, 9,-2],
                          [0,-2,0]])
        return cv2.filter2D(image, -1, kernel)
    
    return image

def binarize_image(image, method="otsu"):
    """Binarizes the image using various methods."""
    if method == "otsu":
        # Otsu's method for global thresholding
        _, binary = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif method == "adaptive":
        # Adaptive thresholding - good for uneven illumination
        binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, 15, 8)
    return binary

def morphological_operations(image, operation, kernel_size=(2, 2), iterations=1):
    """Performs morphological operations with configurable kernel size and iterations."""
    kernel = np.ones(kernel_size, np.uint8)

    if operation == "open":
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == "close":
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == "dilate":
        return cv2.dilate(image, kernel, iterations=iterations)
    elif operation == "erode":
        return cv2.erode(image, kernel, iterations=iterations)
    elif operation == "tophat":
        # Enhances bright text on darker background
        return cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    elif operation == "blackhat":
        # Enhances dark text on lighter background
        return cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    
    return image

def correct_skew(image):
    """Corrects image skew by detecting and straightening text lines."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No text found, skipping skew correction.")
        return image  # Return original image with 0° skew angle

    # Find the largest contour (assumed to be text block)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area bounding box
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]

    # Normalize the angle to be between -45° and +45°
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    # Get image dimensions and center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Perform the rotation with border replication
    deskewed = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return deskewed

def add_fixed_padding(image, padding=(20, 20, 20, 20), padding_value=255):
    """Adds fixed padding to the image."""
    top, bottom, left, right = padding
    if len(image.shape) == 3:
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                          cv2.BORDER_CONSTANT, value=[padding_value]*3)
    else:
        padded_image = cv2.copyMakeBorder(image, top, bottom, left, right, 
                                          cv2.BORDER_CONSTANT, value=padding_value)
    return padded_image

def text_enhance_for_detection(image):
    """Special text enhancement for better text detection and OCR."""
    # Create a copy of the image to work with
    enhanced = image.copy()
    
    # Apply edge-preserving bilateral filter
    enhanced = cv2.bilateralFilter(enhanced, 5, 75, 75)
    
    # Apply local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    enhanced = clahe.apply(enhanced)
    
    # Apply unsharp masking to enhance edges (text)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 3)
    enhanced = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    
    return enhanced

def process_image_for_text_detection(image_path, output_path=None, resize_height=None, crop_odd=None, crop_even=None):
    results = {}
    
    # Step 1: Load image
    img = load_image(image_path)
    results['original'] = img.copy()
    
    # Step 2: Convert to grayscale first (for better processing)
    gray = convert_to_grayscale(img)
    results['grayscale'] = gray.copy()
    
    # Step 3: Crop unwanted regions (margins, binding, etc.)
    if crop_odd and crop_even:
        cropped = crop_image_percentage(gray, image_path, crop_odd, crop_even)
        results['cropped'] = cropped.copy()
    else:
        cropped = gray
    
    # Step 4: Process the image at original resolution first
    # Correct skew (straighten text lines)
    deskewed = correct_skew(cropped)
    results['deskewed'] = deskewed.copy()
    
    # Step 5: Correct uneven illumination (common in old books)
    illumination_corrected = correct_illumination(deskewed)
    results['illumination_corrected'] = illumination_corrected.copy()
    
    # Step 8: Continue with denoising'
    denoised = denoise_image(illumination_corrected, method="weiner")
    denoised = denoise_image(denoised, method="weiner")
    results['denoised'] = denoised.copy()
    
    morph = morphological_operations(denoised, "close", kernel_size=(1,1), iterations=2)
    results['morph_closed'] = morph.copy()

        # Step 6: Apply specific text enhancement for detection
    text_enhanced = enhance_contrast(morph)
    results['text_enhanced'] = text_enhanced.copy()
    
    
    denoised = denoise_image(text_enhanced, method="weiner")
    denoised = denoise_image(denoised, method="weiner")
    denoised = denoise_image(denoised, method="weiner")
    denoised = denoise_image(denoised, method="nlm")
    results['denoised2'] = denoised.copy()
    # Step 7: Now resize (if needed) after initial processing for better text preservation
    if resize_height:
        processed = resize_image(denoised, height=resize_height)
        results['resized'] = processed.copy()
    else:
        processed = denoised
        
    # binary = binarize_image(processed, method="otsu")
    # results['binary'] = binary.copy()


    # Step 13: Add padding for OCR safety margin
    final = add_fixed_padding(processed, padding=(2,2,2,2), padding_value=255)
    final = add_fixed_padding(final, padding=(3,3,3,3), padding_value=0)
    results['final'] = final
    
    # Save result if output path is provided
    if output_path:
        cv2.imwrite(output_path, final)
    
    return results
# Visualization function
def visualize_results(results):
    """Visualizes all the intermediate results."""
    rows = len(results) // 4 + (1 if len(results) % 4 != 0 else 0)
    fig, axes = plt.subplots(rows, min(4, len(results)), figsize=(20, 8 * rows))
    
    for i, (key, img) in enumerate(results.items()):
        row, col = i // 4, i % 4
        if rows == 1:
            ax = axes[col] if len(results) > 1 else axes
        else:
            ax = axes[row, col]
            
        if key == 'original' and len(img.shape) == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray')
            
        ax.set_title(key)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def process_and_save_images(input_dir, output_dir,crop_odd,crop_even):
    """Processes all images in the input directory and saves them to the output directory."""
    print("Transforming images for text detection...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for filename in image_files:
        input_path = os.path.join(input_dir, filename)
        save_path = os.path.join(output_dir, filename)

        # Process image
        processed_image = process_image_for_text_detection(input_path, save_path,resize_height=450,crop_odd=crop_odd,crop_even=crop_even)

        # Save final processed image
        cv2.imwrite(save_path, processed_image['final'])
    
def load_bounding_boxes(txt_path):
    """Reads the bounding box coordinates from CRAFT output .txt file."""
    boxes = []
    with open(txt_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line == "\n":
                continue
            coords = list(map(float, line.strip().split(',')))  # Convert to integers
            boxes.append(np.array(coords,dtype = int).reshape(4, 2))  # Convert to (4,2) shape
    return boxes

def plot_bounding_boxes(image_path, bboxes):

    image = cv2.imread(image_path)
    # Draw bounding boxes
    for bbox in bboxes:
        cv2.polylines(image, [bbox], isClosed=True, color=(0,0,255), thickness=1)

    # Convert BGR to RGB for correct color display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.figure(figsize=(6, 7))
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.show()


def remove_punctuation(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def extract_text_by_page(docx_file, output_dir):
    
    document = Document(docx_file)
    pages = []
    current_page = []
    start_reading = False  # Flag to track when to start reading

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if not text:  # Skip initial Note section
            continue

        if text == "****" and not start_reading:
            start_reading = True  # Start reading after the first "****"
            continue

        if not start_reading:
            continue  # Skip lines before the first "****"

        if text == "****":  # Page separator found
            if current_page:
                pages.append("\n".join(current_page))  # Save current page
                current_page = []  # Start new page
        else:
            text = remove_punctuation(text)  # Remove punctuation
            current_page.append(text)

    ## As last page contain unnecessary text
    # if current_page:  # Save last page
    #     pages.append("\n".join(current_page))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each page separately in its own text file
    for idx, page_text in enumerate(pages, start=1):
        page_file = os.path.join(output_dir, f"page_{idx}.txt")
        with open(page_file, "w", encoding="utf-8") as f:
            f.write(page_text)

    # print(f"Extracted {len(pages)} pages from transcript documents and saved them as txt file in '{output_dir}'.")

def load_bounding_boxes(file_path):
    bounding_boxes = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                coords = list(map(float or int, line.split(',')))
                if len(coords) == 8:  # Ensure exactly 8 values
                    bounding_boxes.append(coords)
                else:
                    print(f"Skipping invalid line (wrong number of coordinates): {line}")
            except ValueError:
                print(f"Skipping invalid line (non-numeric values): {line}")

    return np.array(bounding_boxes) if bounding_boxes else np.empty((0, 8), dtype=int)


def get_top_left_point(box):
    """Extracts the top-left corner (x_min, y_min) from a bounding box."""
    x_values = [box[0], box[2], box[4], box[6]]
    y_values = [box[1], box[3], box[5], box[7]]
    
    x_min = min(x_values)
    y_min = min(y_values)
    
    return x_min, y_min

def sort_bounding_boxes(bounding_boxes, y_tolerance=10):
    """Sort bounding boxes in reading order (top-to-bottom, left-to-right)."""
    
    # Convert bounding boxes to (x_min, y_min, full_box)
    box_with_refs = [(get_top_left_point(box)[0], get_top_left_point(box)[1], box) for box in bounding_boxes]
    
    # Sort by y_min (top-to-bottom), then by x_min (left-to-right within a line)
    sorted_boxes = sorted(box_with_refs, key=lambda b: (b[1], b[0]))

    return [box[2] for box in sorted_boxes]  # Return sorted bounding boxes

def save_sorted_boxes(sorted_boxes, output_file):
    """Save sorted bounding boxes to a text file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for box in sorted_boxes:
            f.write(",".join(map(str, box)) + "\n")

def process_bounding_boxes_folder(input_folder, output_folder):
    """Process all bounding box files in a folder and save sorted results."""
    os.makedirs(output_folder, exist_ok=True)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".txt"):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace(".txt", "_sorted.txt"))
            
            bounding_boxes = load_bounding_boxes(input_path)
            sorted_boxes = sort_bounding_boxes(bounding_boxes)
            save_sorted_boxes(sorted_boxes, output_path)

    # print(f"sorted bounding box saved in '{output_folder}'.")
    
    
def sanitize_filename(text, existing_filenames):
    """Sanitize text for a valid filename and ensure uniqueness."""
    filename = text.replace(" ", "_")  # Replace spaces with underscores
    filename = re.sub(r'[^a-zA-Z0-9_-]', '', filename)  # Remove special characters

    # Ensure uniqueness by appending a counter if filename already exists
    original_filename = filename
    counter = 1
    while filename in existing_filenames:
        filename = f"{original_filename}_{counter}"
        counter += 1

    existing_filenames.add(filename)
    return filename

def find_max_dimensions(image_dir):
    """Finds the maximum width and height of images in a directory and rounds them to the next multiple of 10."""
    max_width, max_height = 0, 0
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(image_dir, img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                h, w = img.shape
                max_width = max(max_width, w)
                max_height = max(max_height, h)
    
    # Round up to the next multiple of 10
    max_width = math.ceil(max_width / 10) * 10
    max_height = math.ceil(max_height / 10) * 10
    
    return max_width, max_height

def pad_image(img, target_width, target_height):
    """Pads an image to match target dimensions while keeping content centered."""
    h, w = img.shape
    pad_top = (target_height - h) // 2
    pad_bottom = target_height - h - pad_top
    pad_left = (target_width - w) // 2
    pad_right = target_width - w - pad_left

    padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, 
                                    cv2.BORDER_CONSTANT, value=255)  # White background
    return padded_img

def resize_and_pad_images(image_dir, output_dir):
    """Finds the max size, rounds to the next multiple of 10, and pads all images to that size."""
    os.makedirs(output_dir, exist_ok=True)

    max_width, max_height = find_max_dimensions(image_dir)
    print(f"Resizing all images to: Width={max_width}, Height={max_height}")

    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is not None:
                padded_img = pad_image(img, max_width, max_height)
                cv2.imwrite(os.path.join(output_dir, img_file), padded_img)

    print("All images have been padded and saved.")

def create_augmentation_pipeline():
    """Create an optimized augmentation pipeline for OCR training."""
    return A.Compose([
        # --- Spatial Augmentations ---
        A.OneOf([
            A.Affine(scale=(0.95, 1.05), translate_percent=0.02, rotate=(-3, 3), shear=(-1, 1), p=0.7),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=3, border_mode=0, p=0.5),
        ], p=0.8),

        # --- Brightness & Contrast (Simulate Ink Variations) ---
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.2, p=0.7),
            A.RandomGamma(gamma_limit=(90, 110), p=0.5),
        ], p=0.7),

        # --- Noise & Blur (Simulate Degraded Text) ---
        A.OneOf([
            A.GaussNoise(var_limit=(5, 30), p=0.6),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.4),  # JPEG compression artifacts
        ], p=0.5),

        # --- Ink Bleed, Paper Aging, & Edge Enhancements ---
        A.OneOf([
            A.Sharpen(alpha=(0.1, 0.3), p=0.5),
            A.Emboss(alpha=(0.1, 0.3), strength=(0.5, 1.0), p=0.4),
            A.InvertImg(p=0.1),  # Invert colors to simulate different print types
        ], p=0.4),

        # --- Document Background Texture & Warping (Simulate Old Paper) ---
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=20, p=0.3),
            A.CLAHE(clip_limit=2, tile_grid_size=(8, 8), p=0.5),  # Improve local contrast
        ], p=0.3),

        # --- Fading & Shadows (Worn-out Text Simulation) ---
        A.OneOf([
            A.RandomShadow(num_shadows=1, shadow_dimension=5, p=0.3),
            A.Solarize(threshold=128, p=0.2),  # Extreme contrast for historical docs
        ], p=0.3),
    ])


def augment_dataset(input_dir, df_path, output_dir, augmentations_per_image=3):

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the original dataset dataframe
    df = pd.read_csv(df_path)
    print(f"Original dataset: {len(df)} images")
    
    # First, copy all original images to the output directory
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Copying original images"):
        img_path = os.path.join(input_dir, row['image_name'])
        if os.path.exists(img_path):
            shutil.copy2(img_path, os.path.join(output_dir, row['image_name']))
        else:
            print(f"Warning: Original image not found: {img_path}")
    
    # Create augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # List to store new augmented image entries
    new_entries = []
    
    # Apply augmentations
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating augmentations"):
        image_path = os.path.join(input_dir, row['image_name'])
        text = row['text']
        
        if not os.path.exists(image_path):
            continue
        
        # Read the original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue
        
        # Generate augmented versions
        for aug_idx in range(augmentations_per_image):
            # Apply the augmentation
            aug_image = transform(image=image)['image']
            
            # Generate a new filename for the augmented image
            base_name, ext = os.path.splitext(row['image_name'])
            aug_name = f"{base_name}_aug{aug_idx+1}{ext}"
            
            # Save the augmented image
            aug_path = os.path.join(output_dir, aug_name)
            cv2.imwrite(aug_path, aug_image)
            
            # Add to new entries
            new_entries.append({
                'image_name': aug_name,
                'text': text,
                'augmented': True
            })
    
    # Add 'augmented' column to original DataFrame
    df['augmented'] = False
    
    # Create new DataFrame with augmented images
    aug_df = pd.DataFrame(new_entries)
    
    # Combine original and augmented DataFrames
    combined_df = pd.concat([df, aug_df], ignore_index=True)
    
    print(f" {len(combined_df)} total images")
    print(f"  - {len(df)} original images")
    print(f"  - {len(aug_df)} augmented images")
    
    return combined_df

def visualize_augmentations(input_dir, sample_image_name, num_examples=5):

    # Read the sample image
    sample_path = os.path.join(input_dir, sample_image_name)
    if not os.path.exists(sample_path):
        print(f"Sample image not found: {sample_path}")
        return
    
    original_image = cv2.imread(sample_path)
    if original_image is None:
        print(f"Could not read sample image: {sample_path}")
        return
    
    # Convert BGR to RGB for matplotlib
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Create the augmentation pipeline
    transform = create_augmentation_pipeline()
    
    # Create a figure to display the original and augmented images
    plt.figure(figsize=(12, 8))
    
    # Display the original image
    plt.subplot(2, 3, 1)
    plt.title("Original")
    plt.imshow(original_image_rgb)
    plt.axis('off')
    
    # Generate and display augmented versions
    for i in range(num_examples):
        augmented = transform(image=original_image)['image']
        augmented_rgb = cv2.cvtColor(augmented, cv2.COLOR_BGR2RGB)
        
        plt.subplot(2, 3, i+2)
        plt.title(f"Augmentation {i+1}")
        plt.imshow(augmented_rgb)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    plt.close()
