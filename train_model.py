# pip install scikit-learn joblib pillow
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from joblib import dump
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
# from scipy import ndimage
# import matplotlib.pyplot as plt

# Advanced image preprocessing functions
# Note: Cropping is now done during dataset preprocessing (preprocess_dataset.py)
# Images in dataset_crop are already cropped and resized to 224x224

def enhance_image(img):
    """Apply advanced preprocessing to improve image quality"""
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.1)
    
    # Apply slight denoising filter
    img = img.filter(ImageFilter.MedianFilter(size=3))
    
    return img

def augment_image(img, augment=True):
    """Apply data augmentation techniques"""
    if not augment:
        return [img]
    
    augmented = [img]  # Original image
    
    # Add multiple rotations (helps with road/sheep confusion)
    rotated_small = img.rotate(3, fillcolor=(128, 128, 128))
    augmented.append(rotated_small)
    
    rotated_neg = img.rotate(-3, fillcolor=(128, 128, 128))
    augmented.append(rotated_neg)
    
    # Brightness variations (helps with lighting conditions)
    enhancer = ImageEnhance.Brightness(img)
    bright = enhancer.enhance(1.1)
    augmented.append(bright)
    
    dark = enhancer.enhance(0.9)
    augmented.append(dark)
    
    # Add slight blur (helps with focus variations)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=1))
    augmented.append(blurred)
    
    return augmented

def extract_features(img_array):
    """Extract additional features from image"""
    # Normalize pixel values - keep as float32 to save memory
    img_array = img_array.astype('float32') / 255.0
    
    # Calculate histograms with more bins for better color discrimination
    # This helps distinguish similar-looking signs (Road vs Sheep)
    hist_r = np.histogram(img_array[:, :, 0], bins=8, range=(0, 1))[0].astype('float32')
    hist_g = np.histogram(img_array[:, :, 1], bins=8, range=(0, 1))[0].astype('float32')
    hist_b = np.histogram(img_array[:, :, 2], bins=8, range=(0, 1))[0].astype('float32')
    
    # Flatten and combine with histogram features
    flattened = img_array.reshape(-1)
    features = np.concatenate([flattened, hist_r, hist_g, hist_b])
    
    return features

# 1) Load data from a folder structure: data/<class_name>/*.jpg
def load_dataset(root, image_size=None, use_augmentation=True):
    """
    Load dataset from pre-cropped images.
    Images are already cropped by preprocess_dataset.py.
    If image_size is None, uses the natural cropped size (no resize).
    """
    X, y = [], []
    # Get class names and ensure they're unique and sorted
    class_names = sorted([p.name for p in Path(root).iterdir() if p.is_dir()])
    class_names = list(dict.fromkeys(class_names))  # Remove duplicates while preserving order
    class_to_idx = {c:i for i,c in enumerate(class_names)}
    
    detected_size = None
    
    for c in class_names:
        for img_path in Path(root, c).glob("*"):
            try:
                # Load image (already cropped)
                img = Image.open(img_path)
                
                # Detect size from first image if not specified
                if image_size is None and detected_size is None:
                    detected_size = img.size
                    print(f"Detected cropped image size: {detected_size}")
                
                # Resize if specified, otherwise use natural size
                if image_size is not None:
                    img = img.resize(image_size)
                
                # Apply enhancement
                img = enhance_image(img)
                
                # Apply augmentation (only for training data)
                augmented_imgs = augment_image(img, augment=use_augmentation)
                
                for aug_img in augmented_imgs:
                    img_array = np.asarray(aug_img).astype(np.float32)
                    features = extract_features(img_array)
                    X.append(features)
                    y.append(class_to_idx[c])
                    
            except Exception:
                pass
    
    X, y = np.array(X, dtype='float32'), np.array(y)  # Force float32 to save memory
    final_size = image_size if image_size is not None else detected_size
    return X, y, class_names, final_size

# Load dataset with resize to small size while maintaining aspect ratio (135:168 = 26:32)
X, y, class_names, image_size = load_dataset("dataset_crop", image_size=(26, 32), use_augmentation=True)
print(f"Using image size: {image_size}")
X, y = shuffle(X, y, random_state=42)
print(f"Total samples after augmentation: {len(X)}")
print(f"Feature dimension: {X[0].shape if len(X) > 0 else 'No data'}")

# 2) Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3) Pipeline: scale -> MLP with original complexity maintained
# Even with cropped size (135x168 = 68,052 features), we use robust architecture
# for better accuracy in safety-critical traffic sign classification
pipe = Pipeline([
    ("scaler", StandardScaler()),  # important for MLP
    ("mlp", MLPClassifier(
        hidden_layer_sizes=(512, 256, 128), activation="relu", solver="adam",
        alpha=1e-5, batch_size=128, learning_rate_init=1e-3,
        max_iter=150, random_state=42, verbose=True,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15))
])

pipe.fit(X_train, y_train)

# 4) Evaluate
y_pred = pipe.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=class_names))

# Display confusion matrix in console
print("\n" + "="*60)
print("CONFUSION MATRIX")
print("="*60)
cm = confusion_matrix(y_test, y_pred)

# Print header
max_label_len = max(len(name) for name in class_names)
header = "Act \\ Pre"
print(f"{header:<{max_label_len+2}}", end="")
for name in class_names:
    print(f"{name:>8}", end="")
print()
print("-" * (max_label_len + 2 + 8 * len(class_names)))

# Print each row
for i, actual_class in enumerate(class_names):
    print(f"{actual_class:<{max_label_len+2}}", end="")
    for j in range(len(class_names)):
        print(f"{cm[i,j]:>8}", end="")
    print()
print("="*60)

# 5) Save in the required format
dump({"model": pipe, "class_names": class_names, "image_size": image_size}, "traffic_sign_mlp.joblib")
print(f"Saved to traffic_sign_mlp.joblib with image_size: {image_size}")
