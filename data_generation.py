import cv2
import numpy as np
import os
import re
import shutil
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import warnings
from sklearn.model_selection import train_test_split
import paddle
import pytesseract
from difflib import SequenceMatcher
import pandas as pd
import glob
from tqdm import tqdm
from data_utils import (pdf_to_images, process_and_save_images, extract_text_by_page, 
                        process_bounding_boxes_folder,resize_and_pad_images ,augment_dataset)

warnings.filterwarnings("ignore")

class Generate_Dataset():
    def __init__(self, detector, align, craft_params=None):
        self.detector = detector
        self.align = align
        # Default CRAFT model parameters if none provided
        self.craft_params = {
            'trained_model': 'CRAFT-pytorch/weights/craft_mlt_25k.pth',
            'text_threshold': 0.8,
            'low_text': 0.33,
            'link_threshold': 0.5,
            'mag_ratio': 1.5,
            'canvas_size': 2200,
            'refiner_model': 'CRAFT-pytorch/weights/craft_refiner_CTW1500.pth',
            'cuda': 'True'
        }
        # Update with user provided parameters if any
        if craft_params:
            self.craft_params.update(craft_params)

    def save_bounding_boxes(self, input_root, output_root):
        """Detects bounding boxes in all book page folders inside input_root and saves results in output_root."""
        # Create the output root directory if it doesn't exist
        os.makedirs(output_root, exist_ok=True)
        
        if self.detector == "paddle":
            paddle.device.set_device("cpu")
            ocr_detector = PaddleOCR(
                        # det_model_dir="detection_model/en_PP-OCRv3_det_infer",  # For PP-OCRv2
                        det_model_dir="detection_model/det_r50_db++_icdar15_train",  # For det_r50_vd_db
                        use_angle_cls=True, 
                        lang="en", 
                        det=True, 
                        rec=False
                    )

            # Iterate over all book folders in input_root
            book_folders = sorted(os.listdir(input_root))
            for i, book_folder in enumerate(tqdm(book_folders, desc="Detecting bounding boxes (paddle)")):
                book_path = os.path.join(input_root, book_folder)
                
                if os.path.isdir(book_path):  # Ensure it's a directory
                    output_book_path = os.path.join(output_root, f"book{i+1}")
                    os.makedirs(output_book_path, exist_ok=True)

                    # Process each image in the book folder
                    image_files = [f for f in sorted(os.listdir(book_path)) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    for image_file in image_files:
                        image_path = os.path.join(book_path, image_file)
                        image_name = os.path.splitext(image_file)[0]
                        output_file = os.path.join(output_book_path, f"{image_name}.txt")
                        # Detect bounding boxes
                        results = ocr_detector.ocr(image_path, cls=True)

                        # Save bounding boxes to a text file
                        with open(output_file, 'w') as f:
                            if results:
                                bbox_count = 0
                                for line in results:
                                    for word in line:
                                        bbox = word[0]  # Bounding box coordinates
                                        f.write(f"{bbox[0][0]},{bbox[0][1]},{bbox[1][0]},{bbox[1][1]},"
                                                f"{bbox[2][0]},{bbox[2][1]},{bbox[3][0]},{bbox[3][1]}\n")
                                        bbox_count += 1

        elif self.detector == "craft":
            # Clone CRAFT repo if not already cloned
            if not os.path.exists("CRAFT-pytorch"):
                os.system("git clone https://github.com/sarthakg004/CRAFT-pytorch.git")

            os.makedirs(output_root, exist_ok=True)

            # Iterate over all book folders in input_root
            book_folders = sorted(os.listdir(input_root))
            for i, book_folder in enumerate(tqdm(book_folders, desc="Detecting bounding boxes (CRAFT)")):
                book_path = os.path.join(input_root, book_folder)
                
                if os.path.isdir(book_path):  # Ensure it's a directory
                    output_book_path = os.path.join(output_root, f"book{i+1}/")  # Fixed: Now includes book_folder name
                    os.makedirs(output_book_path, exist_ok=True)

                    # Run CRAFT text detection on the entire folder
                    craft_command = f"""
                        python CRAFT-pytorch/test.py \
                        --trained_model={self.craft_params['trained_model']} \
                        --result_folder={output_book_path} \
                        --test_folder={book_path} \
                        --text_threshold={self.craft_params['text_threshold']} \
                        --low_text={self.craft_params['low_text']} \
                        --link_threshold={self.craft_params['link_threshold']} \
                        --mag_ratio={self.craft_params['mag_ratio']} \
                        --canvas_size={self.craft_params['canvas_size']} \
                        --refiner_model={self.craft_params['refiner_model']} \
                        --cuda={self.craft_params['cuda']}
                    """
                    # Execute command
                    os.system(craft_command)

    def align_text(self, transcript_folder, bounding_box_folder, image_folder, output_folder, No_transcript_path,technique="word"):
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Find all books in the transcript folder
        books = [d for d in os.listdir(transcript_folder) if os.path.isdir(os.path.join(transcript_folder, d))]
        
        results = {}
        
        for book in tqdm(books, desc=f"Aligning text ({technique} technique)"):
            # Create output directory for this book
            book_output_dir = os.path.join(output_folder, book)
            os.makedirs(book_output_dir, exist_ok=True)
            
            # Determine paths for this book
            book_transcript_dir = os.path.join(transcript_folder, book)
            book_bbox_dir = os.path.join(bounding_box_folder, f"{book}_sorted")
            book_image_dir = os.path.join(image_folder, f"{book}_transformed")
            
            # Check if directories exist
            if not os.path.exists(book_transcript_dir) or not os.path.exists(book_bbox_dir) or not os.path.exists(book_image_dir):
                continue
            
            # Process based on technique
            if technique == "line":
                stats = self.process_line_alignment(book_transcript_dir, book_bbox_dir, book_output_dir)
            elif technique == "word":
                stats = self.process_word_alignment(book,book_transcript_dir, book_bbox_dir, book_image_dir, book_output_dir, No_transcript_path)
            else:
                raise ValueError(f"Unknown technique: {technique}. Use 'line' or 'word'.")
            
            results[book] = stats
        
        return results

    def process_line_alignment(self, transcript_dir, bbox_dir, output_dir):
        """Process line-by-line alignment for a single book."""
        stats = {
            'total_pages': 0,
            'total_lines': 0,
            'aligned_lines': 0,
            'skip_pages': []
        }
        
        # Find all transcript files
        transcript_files = [f for f in os.listdir(transcript_dir) if f.startswith("page_") and f.endswith(".txt")]
        
        for transcript_file in transcript_files:
            # Extract page number
            page_number = transcript_file.split("page_")[1].split(".")[0]
            
            # Construct file paths
            transcript_path = os.path.join(transcript_dir, transcript_file)
            bbox_path = os.path.join(bbox_dir, f"{page_number}_sorted.txt")
            output_path = os.path.join(output_dir, f"{page_number}.txt")
            
            # Skip if bounding box file doesn't exist
            if not os.path.exists(bbox_path):
                stats['skip_pages'].append(int(page_number))
                continue
            
            try:
                # Load text lines
                with open(transcript_path, "r", encoding="utf-8") as f:
                    text_lines = [line.strip() for line in f if line.strip()]
                
                # Load bounding boxes
                with open(bbox_path, "r", encoding="utf-8") as f:
                    bounding_boxes = [list(map(float, line.strip().split(','))) for line in f if line.strip()]

                # Ensure text and bounding boxes count match
                if len(text_lines) != len(bounding_boxes):
                    stats['skip_pages'].append(int(page_number))
                    continue
                
                # Align text with bounding boxes
                aligned_data = []
                for text, bbox in zip(text_lines, bounding_boxes):
                    aligned_data.append(f"{text}\t{','.join(map(str, bbox))}")
                
                # Save aligned data
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(aligned_data))
                
                stats['total_lines'] += len(text_lines)
                stats['aligned_lines'] += len(aligned_data)
                stats['total_pages'] += 1
                
            except Exception as e:
                stats['skip_pages'].append(int(page_number))
        
        return stats

    def similarity_score(self, a, b):
        """Calculate string similarity between two strings"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def process_word_alignment(self, book_folder,transcript_dir, bbox_dir, image_dir, output_dir,No_transcript_path, tesseract_model=1):
        """Process word-level alignment for a single book."""
        stats = {
            'total_pages': 0,
            'total_words': 0,
            'mapped_words': 0,
            'total_bboxes': 0,
            'NO_transcript_pages': 0
        }
        
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        for image in image_files:
            page_number = image.split(".")[0]
             
            transcript_path = os.path.join(transcript_dir,f"page_{page_number}.txt")
            bbox_path = os.path.join(bbox_dir, f"{page_number}_sorted.txt")
            image_path = os.path.join(image_dir, f"{page_number}.png")
            output_path = os.path.join(output_dir, f"{page_number}.txt") 
            
            if not os.path.exists(transcript_path):
                stats['NO_transcript_pages'] += 1
                os.makedirs(os.path.join(No_transcript_path, book_folder), exist_ok=True)
                shutil.copy(image_path, os.path.join(No_transcript_path, book_folder, image))
                continue
            try:
                actual_words, mapped_words, bb_count = self.map_bounding_boxes_to_transcript(
                    image_path, bbox_path, transcript_path, output_path, tesseract_model
                )
                
                stats['total_words'] += actual_words
                stats['mapped_words'] += mapped_words
                stats['total_bboxes'] += bb_count
                stats['total_pages'] += 1
                
                
            except Exception as e:
                continue
        
        return stats

    def map_bounding_boxes_to_transcript(self, image_path, bbox_file, transcript_file, output_file, tesseract_model=1):
        """Map individual bounding boxes to words in transcript."""
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Read bounding boxes
        with open(bbox_file, 'r') as f:
            bounding_boxes = [line.strip() for line in f.readlines() if line.strip()]
        
        # Read transcript
        with open(transcript_file, 'r') as f:
            transcript = f.read().strip()
        # Split transcript into words
        transcript_lines = transcript.split('\n')
        transcript_words = []
        for line in transcript_lines:
            transcript_words.extend(line.strip().split())
        
        # Result mappings
        mappings = []
        
        # Process each bounding box
        for i, bbox in enumerate(bounding_boxes):
                
            # Parse the bounding box coordinates
            coords = [float(c) for c in bbox.split(',')]
            x_coords = [coords[0], coords[2], coords[4], coords[6]]
            y_coords = [coords[1], coords[3], coords[5], coords[7]]
            
            # Get the rectangular region (min/max coordinates)
            x_min, y_min = int(min(x_coords)), int(min(y_coords))
            x_max, y_max = int(max(x_coords)), int(max(y_coords))
            
            # Ensure coordinates are within image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(image.shape[1], x_max)
            y_max = min(image.shape[0], y_max)
            
            # Skip if region is too small
            if x_max - x_min < 5 or y_max - y_min < 5:
                continue
            
            # Extract the region from the image
            roi = image[y_min:y_max, x_min:x_max]
            
            # Skip if ROI is empty
            if roi.size == 0:
                continue

            # Use pytesseract to get the text in this bounding box
            try:
                # 0: Legacy engine only 1: Neural nets LSTM engine only 2: Legacy + LSTM engines 3: Default
                detected_text = pytesseract.image_to_string(roi, 
                                                        config=f'--psm 7 --oem {tesseract_model} -l spa').strip()
                
                # Skip if no text detected
                if not detected_text:
                    continue
                    
                # Find the closest matching word in the transcript
                max_similarity = 0
                best_match = None
                
                for word in transcript_words:
                    sim = self.similarity_score(detected_text, word)
                    if sim > max_similarity:
                        max_similarity = sim
                        best_match = word
                
                # Only consider it a match if similarity is above threshold
                if max_similarity > 0.5 and best_match:
                    # Add to mappings and remove matched word from transcript to avoid duplicates
                    mappings.append((best_match, bbox, max_similarity))
                    transcript_words.remove(best_match)
                
            except Exception as e:
                continue
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Write results to output file
        with open(output_file, 'w') as f:
            for word, bbox, sim in mappings:
                f.write(f"{word}\t{bbox}\n")
        
        return len(transcript_words) + len(mappings), len(mappings), len(bounding_boxes)

    def extract_and_save_regions(self,image_path, aligned_text_file, output_dir, start_index):
        """Extract regions from image using bounding boxes and save them with sequential filenames."""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return [], start_index

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Read aligned text and bounding boxes
        with open(aligned_text_file, "r", encoding="utf-8") as f:
            lines = [line.strip().split("\t") for line in f if "\t" in line]

        results = []  # To store image names and corresponding text
        current_index = start_index

        for i, (text, bbox_str) in enumerate(lines):
            # Generate sequential filename using the global counter
            output_filename = f"image{current_index}.png"
            output_path = os.path.join(output_dir, output_filename)
            
            # Add to results list
            results.append((output_filename, text))
            
            bbox = list(map(float, bbox_str.split(',')))  # Convert bbox string to float list

            # Convert to NumPy array (4 corner points)
            pts = np.array(bbox, dtype=np.int32).reshape((4, 2))

            # Get bounding box limits
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)

            # Crop the region
            cropped_region = image[y_min:y_max, x_min:x_max]

            # Validate cropping
            if cropped_region.size == 0:
                continue

            # Save the cropped image
            cv2.imwrite(output_path, cropped_region)
            
            # Increment counter for next image
            current_index += 1
        
        return results, current_index

    def process_book_directories(self,images_dir, aligned_text_dir, output_dir, image_ext='.png', text_ext='.txt'):
        """Process all book directories and return a DataFrame with image names and text."""
        os.makedirs(output_dir, exist_ok=True)
        
        # For storing all results
        all_results = []
        
        # Global counter for image naming
        global_image_counter = 1
        
        # Find all book directories in the images directory
        book_dirs = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
        
        if not book_dirs:
            return pd.DataFrame(columns=['image_name', 'text'])
        
        # Process each book directory
        for i, book_dir in enumerate(tqdm(sorted(book_dirs), desc="Extracting text regions")):
            book_images_dir = os.path.join(images_dir, book_dir)
            book_text_dir = os.path.join(aligned_text_dir, f"book{i+1}")
            
            # Check if corresponding text directory exists
            if not os.path.isdir(book_text_dir):
                continue
            
            # Get all image files in this book directory
            image_files = glob.glob(os.path.join(book_images_dir, f"*{image_ext}"))
            
            # Process each image file
            for image_path in sorted(image_files):
                # Get base filename without extension
                image_basename = os.path.splitext(os.path.basename(image_path))[0]
                
                # Find corresponding text file
                text_path = os.path.join(book_text_dir, f"{image_basename}{text_ext}")
                
                # Check if text file exists
                if os.path.exists(text_path):
                    results, global_image_counter = self.extract_and_save_regions(
                        image_path, text_path, output_dir, global_image_counter
                    )
                    all_results.extend(results)
        
        # Create DataFrame from results
        df = pd.DataFrame(all_results, columns=['image_name', 'label'])
        
        return df

    def generate(self,
                book_path="data/raw/books",                                   ## Path to the folder containing the PDFs
                transcript_doc_path="data/raw/transcripts",                      ## Path to the folder containing the transcripts
                transcript_text_path="data/PreProcessed/transcripts_pages",          ## Path to the folder where the transcripts will be saved
                book_pages_path="data/PreProcessed/book_pages",           ## Path to the folder where the PDFs will be split into pages
                transformed_pages_path="data/PreProcessed/transformed",   ## Path to the folder where the transformed pages will be saved
                detected_boxes_path="data/PreProcessed/bounding_boxes/craft",  ## Path to the folder where the bounding boxes will be saved
                aligned_text_path="data/PreProcessed/aligned/craft",        ## Path to the folder where the aligned text will be saved
                pages_without_transcript_path = "data/PreProcessed/missing_transcripts",
                cropped_bbox_path="data/Processed/words",      ## Path to the folder where the cropped regions will be saved
                augmented_bbox_path = "data/Processed/augmented_words",
                df_path = "data/Processed/words.csv",
                augmented_df_path="data/Processed/augmented_words.csv",
                augmentations_per_image=3
                ):
        
        # Splitting PDFs into pages
        books = sorted(os.listdir(book_path))
        for i, book in enumerate(tqdm(books, desc="Step 1: Splitting PDFs into pages")):
            if book.lower().endswith(".pdf"):
                book_pdf_path = os.path.join(book_path, book)
                output_folder = os.path.join(book_pages_path, f"book{i+1}")
                
                os.makedirs(output_folder, exist_ok=True)
                
                pdf_to_images(book_pdf_path, output_folder)
                
        ### Applying transformations to the pages
        books = sorted(os.listdir(book_pages_path))
        for i, book in enumerate(tqdm(books, desc="Step 2: Applying transformations to pages")):  
            book_folder = os.path.join(book_pages_path,f"book{i+1}" )
            transformed_folder = os.path.join(transformed_pages_path, f"book{i+1}_transformed")
            os.makedirs(transformed_folder, exist_ok=True)
            process_and_save_images(book_folder, transformed_folder)

        ### Detecting bounding boxes and saving them
        self.save_bounding_boxes(transformed_pages_path, detected_boxes_path)

        ### Mapping the bounding boxes to the original images
        ## creating text file from transcript
        books = sorted(os.listdir(transcript_doc_path))
        for i, book in enumerate(tqdm(books, desc="Step 4: Creating text files from transcripts")):
            docx_file = os.path.join(transcript_doc_path, book)
            output_directory = os.path.join(transcript_text_path, f"book{i+1}")
            os.makedirs(output_directory, exist_ok=True)
            extract_text_by_page(docx_file, output_directory)  
            
        ## sort the bounding boxes
        detected_boxes = [b for b in sorted(os.listdir(detected_boxes_path)) if not b.endswith("_sorted")]
        for i, book in enumerate(tqdm(detected_boxes, desc="Step 5: Sorting bounding boxes")):
            detected_boxes_folder = os.path.join(detected_boxes_path, book)
            sorted_boxes_folder = os.path.join(detected_boxes_path, f"{book}_sorted")
            
            process_bounding_boxes_folder(detected_boxes_folder, sorted_boxes_folder)
            
        # align the text
        self.align_text(
            transcript_folder=transcript_text_path,
            bounding_box_folder=detected_boxes_path,
            image_folder=transformed_pages_path,
            output_folder=aligned_text_path,
            No_transcript_path = pages_without_transcript_path,
            technique="word"
        )

        df = self.process_book_directories(transformed_pages_path, aligned_text_path, cropped_bbox_path)
        df.to_csv(df_path, index=False)
        
        # Perform augmentation
        with tqdm(total=4, desc="Step 8: Augmenting dataset") as pbar:
            augmented_df = augment_dataset(cropped_bbox_path, df_path, augmented_bbox_path, augmentations_per_image=augmentations_per_image)
            pbar.update(2)
            
            augmented_df.to_csv(augmented_df_path, index=False)
            pbar.update(1)
            
            resize_and_pad_images(augmented_bbox_path,augmented_bbox_path)
            pbar.update(1)
    
    def get_train_val_data(self,
                           path_to_dataframe="data/Processed/augmented_words.csv",
                           output_dir = "data/Processed/",
                            val_size = 0.2):
        
        augment_df = pd.read_csv(path_to_dataframe)
        train, val = train_test_split(augment_df, test_size=val_size, random_state=42)
        train.to_csv(os.path.join(output_dir,"train.csv"), index = False)
        val.to_csv(os.path.join(output_dir,"val.csv"), index = False)