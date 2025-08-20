# get test, train and validation data from database
import os
import kagglehub
import pandas as pd

# make it a class
class DataLoader:
    def __init__(self):
        self.path = dowload_data()
        
        # Test data paths (all categories if available)
        self.test_normal_path = os.path.join(self.path, "test", "0_normal")
        self.test_normal_files = os.listdir(self.test_normal_path) if os.path.exists(self.test_normal_path) else []
        
        self.test_ulcerative_colitis_path = os.path.join(self.path, "test", "1_ulcerative_colitis")
        self.test_ulcerative_colitis_files = os.listdir(self.test_ulcerative_colitis_path) if os.path.exists(self.test_ulcerative_colitis_path) else []

        self.test_polyps_path = os.path.join(self.path, "test", "2_polyps")
        self.test_polyps_files = os.listdir(self.test_polyps_path) if os.path.exists(self.test_polyps_path) else []

        self.test_esophagitis_path = os.path.join(self.path, "test", "3_esophagitis")
        self.test_esophagitis_files = os.listdir(self.test_esophagitis_path) if os.path.exists(self.test_esophagitis_path) else []
        
        # Train data paths for all categories
        self.train_normal_path = os.path.join(self.path, "train", "0_normal")
        self.train_normal_files = os.listdir(self.train_normal_path)
        
        self.train_ulcerative_colitis_path = os.path.join(self.path, "train", "1_ulcerative_colitis")
        self.train_ulcerative_colitis_files = os.listdir(self.train_ulcerative_colitis_path)
        
        self.train_polyps_path = os.path.join(self.path, "train", "2_polyps")
        self.train_polyps_files = os.listdir(self.train_polyps_path)
        
        self.train_esophagitis_path = os.path.join(self.path, "train", "3_esophagitis")
        self.train_esophagitis_files = os.listdir(self.train_esophagitis_path)

        # Validation data paths (using "val" folder name)
        self.validation_normal_path = os.path.join(self.path, "val", "0_normal")
        self.validation_normal_files = os.listdir(self.validation_normal_path)

        self.validation_ulcerative_colitis_path = os.path.join(self.path, "val", "1_ulcerative_colitis")
        self.validation_ulcerative_colitis_files = os.listdir(self.validation_ulcerative_colitis_path)

        self.validation_polyps_path = os.path.join(self.path, "val", "2_polyps")
        self.validation_polyps_files = os.listdir(self.validation_polyps_path)

        self.validation_esophagitis_path = os.path.join(self.path, "val", "3_esophagitis")
        self.validation_esophagitis_files = os.listdir(self.validation_esophagitis_path)

    def get_test_data(self):
        """Returns all test files from all categories combined with labels (if available)"""
        all_test_files = []
        all_test_files.extend([(f, 0) for f in self.test_normal_files])  # label 0: normal
        all_test_files.extend([(f, 1) for f in self.test_ulcerative_colitis_files])  # label 1: ulcerative_colitis
        all_test_files.extend([(f, 2) for f in self.test_polyps_files])  # label 2: polyps
        all_test_files.extend([(f, 3) for f in self.test_esophagitis_files])  # label 3: esophagitis
        return all_test_files
    
    def get_train_data(self):
        """Returns all train files from all categories combined"""
        all_train_files = []
        all_train_files.extend([(f, 0) for f in self.train_normal_files])  # label 0: normal
        all_train_files.extend([(f, 1) for f in self.train_ulcerative_colitis_files])  # label 1: ulcerative_colitis
        all_train_files.extend([(f, 2) for f in self.train_polyps_files])  # label 2: polyps  
        all_train_files.extend([(f, 3) for f in self.train_esophagitis_files])  # label 3: esophagitis
        return all_train_files
    
    def get_train_data_by_category(self):
        """Returns train data organized by category"""
        return {
            'normal': self.train_normal_files,
            'ulcerative_colitis': self.train_ulcerative_colitis_files,
            'polyps': self.train_polyps_files,
            'esophagitis': self.train_esophagitis_files
        }
    
    def get_category_counts(self):
        """Returns count of images in each category"""
        return {
            'test_normal': len(self.test_normal_files),
            'train_normal': len(self.train_normal_files),
            'train_ulcerative_colitis': len(self.train_ulcerative_colitis_files),
            'train_polyps': len(self.train_polyps_files),
            'train_esophagitis': len(self.train_esophagitis_files),
            'val_normal': len(self.validation_normal_files),
            'val_ulcerative_colitis': len(self.validation_ulcerative_colitis_files),
            'val_polyps': len(self.validation_polyps_files),
            'val_esophagitis': len(self.validation_esophagitis_files)
        }
    
    def get_validation_data(self):
        """Returns validation data organized by category"""
        return {
            'normal': self.validation_normal_files,
            'ulcerative_colitis': self.validation_ulcerative_colitis_files,
            'polyps': self.validation_polyps_files,
            'esophagitis': self.validation_esophagitis_files
        }
    
    def get_validation_data_with_labels(self):
        """Returns all validation files from all categories combined with labels"""
        all_val_files = []
        all_val_files.extend([(f, 0) for f in self.validation_normal_files])  # label 0: normal
        all_val_files.extend([(f, 1) for f in self.validation_ulcerative_colitis_files])  # label 1: ulcerative_colitis
        all_val_files.extend([(f, 2) for f in self.validation_polyps_files])  # label 2: polyps  
        all_val_files.extend([(f, 3) for f in self.validation_esophagitis_files])  # label 3: esophagitis
        return all_val_files
    

    
def dowload_data():
    import kagglehub
    # Download latest version (kagglehub will handle caching automatically)
    path = kagglehub.dataset_download("francismon/curated-colon-dataset-for-deep-learning")
    print(f"Dataset downloaded/cached at: {path}")
    return path
