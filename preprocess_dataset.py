# Script to preprocess dataset with cropping
from pathlib import Path
from PIL import Image
import shutil

def crop_image(img):
    """Crop image: remove 25% from bottom, 20% from left and right"""
    width, height = img.size
    # Calculate crop box: (left, top, right, bottom)
    left = int(width * 0.20)
    top = 0  # Keep top unchanged
    right = int(width * 0.80)
    bottom = int(height * 0.75)
    return img.crop((left, top, right, bottom))

def preprocess_dataset(source_dir="dataset", target_dir="dataset_crop"):
    """
    Preprocess all images in dataset:
    1. Crop (remove 25% bottom, 20% left/right)
    2. Keep original aspect ratio (no resizing)
    3. Save to new directory
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    # Create target directory if it doesn't exist
    if target_path.exists():
        print(f"Warning: {target_dir} already exists. Clearing it...")
        shutil.rmtree(target_path)
    target_path.mkdir(exist_ok=True)
    
    total_processed = 0
    
    # Process each class folder
    for class_folder in source_path.iterdir():
        if not class_folder.is_dir():
            continue
            
        class_name = class_folder.name
        print(f"\nProcessing class: {class_name}")
        
        # Create class folder in target directory
        target_class_path = target_path / class_name
        target_class_path.mkdir(exist_ok=True)
        
        # Process each image in the class folder
        count = 0
        for img_path in class_folder.glob("*"):
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
                
            try:
                # Load image
                img = Image.open(img_path)
                
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply cropping ONLY (no resize)
                img = crop_image(img)
                
                # Save with same filename
                target_img_path = target_class_path / img_path.name
                img.save(target_img_path, quality=95)
                
                count += 1
                
            except Exception as e:
                print(f"  Error processing {img_path.name}: {e}")
        
        print(f"  Processed {count} images")
        total_processed += count
    
    print(f"\n✅ Total images processed: {total_processed}")
    print(f"✅ Cropped dataset saved to: {target_dir}")
    print(f"   - Images are cropped (25% bottom, 20% left/right removed)")
    print(f"   - Original dimensions preserved (no resizing)")

if __name__ == "__main__":
    print("Starting dataset preprocessing...")
    print("=" * 60)
    preprocess_dataset(
        source_dir="dataset",
        target_dir="dataset_crop"
    )
    print("=" * 60)
    print("Done! You can now train your model with the cropped dataset.")
