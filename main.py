from data_generation import Generate_Dataset

# Generate data
if __name__ == "__main__":
    generator = Generate_Dataset(detector="craft", align="word")
    
    # Generate the dataset
    generator.generate(book_path="data/raw/books",                             
                transcript_doc_path="data/raw/transcripts",                    
                transcript_text_path="data/PreProcessed/transcripts_pages",    
                book_pages_path="data/PreProcessed/book_pages",                
                transformed_pages_path="data/PreProcessed/transformed",        
                detected_boxes_path="data/PreProcessed/bounding_boxes/craft",  
                aligned_text_path="data/PreProcessed/aligned/craft",     
                pages_without_transcript_path="data/PreProcessed/missing_transcripts",      
                cropped_bbox_path="data/Processed/words",                      
                augmented_bbox_path = "data/Processed/augmented_words",
                df_path = "data/Processed/words.csv",
                augmented_df_path="data/Processed/augmented_words.csv",
                augmentations_per_image=4
    )
    
    generator.get_train_val_data(path_to_dataframe="data/Processed/augmented_words.csv",
                           output_dir = "data/Processed/",
                            val_size = 0.2)