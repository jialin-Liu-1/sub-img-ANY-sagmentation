import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
from tqdm import tqdm
import pydicom
from PIL import Image
import time
import cv2
import json
import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import traceback
from sklearn.metrics import roc_curve, auc
import gc

# Import your model
from multi.ves_U import EnhancedAttentionAwareUNet

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Add safe globals for PyTorch 2.6+
import torch.serialization

try:
    import numpy as np

    torch.serialization.add_safe_globals([np._core.multiarray.scalar])
    torch.serialization.add_safe_globals([np.ndarray])
except Exception as e:
    print(f"Warning when adding safe globals: {e}")


class MedicalRecordPositionLoader:
    """Medical record-based position information loader"""

    def __init__(self, excel_path="D:\\med_data\\ai\\classify.xlsx"):
        self.excel_path = excel_path
        self.position_dict = {}  # medical record number -> position
        self._load_by_medical_record()

    def _load_by_medical_record(self):
        """Load position information by medical record number"""
        try:
            # Read Excel using pandas
            df = pd.read_excel(self.excel_path, header=0)
            print(f"Excel columns: {df.columns.tolist()}")
            print(f"First 5 rows:\n{df.head()}")

            for idx, row in df.iterrows():
                try:
                    # First column: medical record number
                    record_str = str(row.iloc[0]).strip()

                    # Extract numeric medical record number
                    record_num = None
                    if record_str:
                        # Try to match numbers
                        num_match = re.search(r'(\d+)', record_str)
                        if num_match:
                            record_num = int(num_match.group(1))

                    if record_num is None:
                        continue

                    # Second column: position value
                    position_val = row.iloc[1]
                    if pd.isna(position_val):
                        position_num = 0
                    else:
                        try:
                            position_num = int(float(position_val))
                            position_num = max(0, min(7, position_num))
                        except:
                            position_num = 0

                    self.position_dict[record_num] = position_num
                    if idx < 5:  # Show first 5 records
                        print(f"  Medical record {record_num}: position {position_num}")

                except Exception as e:
                    continue

            print(f"Position information loaded: {len(self.position_dict)} medical records")
            print(f"First 10 medical records: {list(self.position_dict.keys())[:10]}")

        except Exception as e:
            print(f"Failed to load position information: {e}")
            traceback.print_exc()
            self.position_dict = {}

    def extract_medical_record_from_filename(self, filename):
        """Extract medical record number from filename"""
        try:
            # Filename format: "ANY_450_0" or "ANY_450_0.dcm"
            basename = os.path.splitext(filename)[0]  # Remove extension
            parts = basename.split('_')

            if len(parts) >= 2:
                # Format: ANY_450_0 -> take second part as medical record number
                try:
                    record_num = int(parts[1])
                    return record_num
                except:
                    pass

            # Try to extract numbers from string
            num_match = re.search(r'(\d+)', basename)
            if num_match:
                try:
                    return int(num_match.group(1))
                except:
                    pass

            return None

        except Exception as e:
            print(f"Failed to extract medical record from {filename}: {e}")
            return None

    def get_position_for_image(self, filename):
        """Get position information based on image filename"""
        try:
            # Extract medical record number
            record_num = self.extract_medical_record_from_filename(filename)

            if record_num is None:
                print(f"Warning: Cannot extract medical record from filename: {filename}")
                position_num = 0
                case_id = f"unknown_{filename}"
            else:
                case_id = f"record_{record_num}"

                # Find position information
                if record_num in self.position_dict:
                    position_num = self.position_dict[record_num]
                    print(f"Found medical record {record_num}: position {position_num}")
                else:
                    # Check if flipped case (medical record > 500)
                    if record_num > 500:
                        original_num = record_num - 500
                        if original_num in self.position_dict:
                            position_num = self.position_dict[original_num]
                            print(f"Flipped case {record_num} -> {original_num}: position {position_num}")
                        else:
                            position_num = 0
                            print(f"Warning: Medical record {record_num} not found in position information")
                    else:
                        position_num = 0
                        print(f"Warning: Medical record {record_num} not found in position information")

            # Create position tensor (one-hot encoding)
            position_tensor = torch.zeros(8, dtype=torch.float32)
            if 0 <= position_num < 8:
                position_tensor[position_num] = 1.0

            return position_tensor, case_id

        except Exception as e:
            print(f"Failed to get position information for {filename}: {e}")
            return torch.zeros(8, dtype=torch.float32), f"error_{filename}"


class DicomTestDataset(Dataset):
    """DICOM test dataset, supporting extensionless files"""

    def __init__(self, image_dir, mask_dir, position_loader=None, max_samples=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.position_loader = position_loader
        self.max_samples = max_samples

        # Get file list
        self.file_pairs = self._get_dicom_file_pairs()

        if self.file_pairs:
            if max_samples and len(self.file_pairs) > max_samples:
                self.file_pairs = self.file_pairs[:max_samples]

            print(f"Found {len(self.file_pairs)} DICOM test samples")
            for i, (img_file, mask_file) in enumerate(self.file_pairs[:5]):
                print(f"  Sample {i + 1}: {img_file} -> {mask_file}")
        else:
            print("Error: No DICOM file pairs found")
            # Show directory contents
            print(f"\nImage directory contents ({image_dir}):")
            img_files = os.listdir(image_dir)[:10]
            for f in img_files:
                print(f"  {f}")

            print(f"\nMask directory contents ({mask_dir}):")
            mask_files = os.listdir(mask_dir)[:10]
            for f in mask_files:
                print(f"  {f}")

    def _get_dicom_file_pairs(self):
        """Get DICOM image and mask file pairs"""
        file_pairs = []

        try:
            # Get all image files (support extensionless DICOM files)
            image_files = []
            for f in os.listdir(self.image_dir):
                file_path = os.path.join(self.image_dir, f)

                # Check if DICOM file
                if self._is_dicom_file(file_path):
                    image_files.append(f)
                # Or if filename looks like DICOM (no extension and has numbers)
                elif '_' in f and not '.' in f:
                    image_files.append(f)

            print(f"Found {len(image_files)} possible DICOM image files")

            # Get all mask files
            mask_files = []
            for f in os.listdir(self.mask_dir):
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.dcm')):
                    mask_files.append(f)
                # Also check extensionless files
                elif '_' in f and not '.' in f:
                    mask_files.append(f)

            print(f"Found {len(mask_files)} mask files")

            if not image_files or not mask_files:
                return file_pairs

            # Create base name mapping
            image_dict = {}
            for img_file in image_files:
                base_name = self._get_file_base_name(img_file)
                if base_name:
                    image_dict[base_name] = img_file
                    if len(image_dict) <= 5:  # Show first 5 mappings
                        print(f"  Image mapping: {img_file} -> {base_name}")

            # Match files
            matched_count = 0
            for mask_file in mask_files:
                mask_base = self._get_file_base_name(mask_file)
                if mask_base in image_dict:
                    file_pairs.append((image_dict[mask_base], mask_file))
                    matched_count += 1
                    if matched_count <= 5:
                        print(f"  Matched: {image_dict[mask_base]} <-> {mask_file}")

            print(f"Successfully matched {matched_count} file pairs")

        except Exception as e:
            print(f"Error getting file pairs: {e}")
            traceback.print_exc()

        return file_pairs

    def _get_file_base_name(self, filename):
        """Get file base name (for matching)"""
        try:
            # Remove extension
            basename = os.path.splitext(filename)[0]

            # Format: "ANY_450_0" -> take first two parts "ANY_450"
            parts = basename.split('_')
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"
            else:
                return basename

        except:
            return filename

    def _is_dicom_file(self, file_path):
        """Check if file is DICOM file"""
        try:
            # Try to read as DICOM
            pydicom.dcmread(file_path, force=True, stop_before_pixels=True)
            return True
        except:
            # Check file header
            try:
                with open(file_path, 'rb') as f:
                    header = f.read(132)  # DICOM file should have "DICM" at byte 128
                    if len(header) >= 132 and header[128:132] == b'DICM':
                        return True
            except:
                pass
            return False

    def _load_dicom_file(self, file_path):
        """Load DICOM file"""
        try:
            dicom_data = pydicom.dcmread(file_path, force=True)
            image = dicom_data.pixel_array.astype(np.float32)

            # Handle possible color channels
            if len(image.shape) == 3:
                # If RGB, convert to grayscale
                if image.shape[2] == 3:
                    image = np.mean(image, axis=2)
                elif image.shape[2] == 4:
                    image = np.mean(image[:, :, :3], axis=2)

            # Normalize to 0-1
            img_min = np.min(image)
            img_max = np.max(image)
            if img_max > img_min:
                image = (image - img_min) / (img_max - img_min + 1e-8)
            else:
                image = np.zeros_like(image)

            return image

        except Exception as e:
            print(f"Failed to load DICOM {file_path}: {e}")
            # Return default image
            return np.zeros((512, 512), dtype=np.float32)

    def _load_mask_file(self, file_path):
        """Load mask file"""
        try:
            if file_path.lower().endswith('.dcm'):
                # If DICOM format mask
                return self._load_dicom_file(file_path)
            else:
                # Regular image file
                img = Image.open(file_path)
                mask = np.array(img).astype(np.float32)

                # Convert to grayscale
                if len(mask.shape) == 3:
                    mask = np.mean(mask, axis=2)

                # Binarize
                mask = (mask > 0.5).astype(np.float32)

                return mask

        except Exception as e:
            print(f"Failed to load mask {file_path}: {e}")
            return np.zeros((512, 512), dtype=np.float32)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        image_file, mask_file = self.file_pairs[idx]

        try:
            # Load DICOM image
            image_path = os.path.join(self.image_dir, image_file)
            image = self._load_dicom_file(image_path)

            # Load mask
            mask_path = os.path.join(self.mask_dir, mask_file)
            mask = self._load_mask_file(mask_path)

            # Ensure dimensions match
            if image.shape != mask.shape:
                # Resize mask
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

            # Get position information
            if self.position_loader:
                position_tensor, case_id = self.position_loader.get_position_for_image(image_file)
            else:
                position_tensor = torch.zeros(8, dtype=torch.float32)
                case_id = image_file

            # Add channel dimension
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)

            # Convert to tensor
            image_tensor = torch.from_numpy(image).float()
            mask_tensor = torch.from_numpy(mask).float()

            # Clean memory
            del image, mask
            if idx % 10 == 0:
                gc.collect()

            return image_tensor, mask_tensor, position_tensor, case_id, image_file

        except Exception as e:
            print(f"Failed to process sample {image_file}: {e}")
            traceback.print_exc()
            # Return default tensor
            dummy_image = torch.zeros((1, 512, 512), dtype=torch.float32)
            dummy_mask = torch.zeros((1, 512, 512), dtype=torch.float32)
            dummy_position = torch.zeros(8, dtype=torch.float32)
            return dummy_image, dummy_mask, dummy_position, "error", image_file


# Helper functions remain unchanged
def calculate_dice_safe(preds, targets):
    try:
        preds_binary = (preds > 0.5).float()
        intersection = (preds_binary * targets).sum()
        union = preds_binary.sum() + targets.sum()
        dice = (2. * intersection) / (union + 1e-8)
        return dice.item()
    except:
        return 0.0


def calculate_iou_safe(preds, targets):
    try:
        preds_binary = (preds > 0.5).float()
        intersection = (preds_binary * targets).sum()
        union = preds_binary.sum() + targets.sum() - intersection
        iou = intersection / (union + 1e-8)
        return iou.item()
    except:
        return 0.0


def calculate_sensitivity_specificity_safe(preds, targets):
    try:
        preds_binary = (preds > 0.5).float()
        tp = (preds_binary * targets).sum()
        fp = (preds_binary * (1 - targets)).sum()
        tn = ((1 - preds_binary) * (1 - targets)).sum()
        fn = ((1 - preds_binary) * targets).sum()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        return sensitivity.item(), specificity.item()
    except:
        return 0.0, 0.0


def load_model_safely(model_path, device):
    print(f"Loading model: {os.path.basename(model_path)}")
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        print("✓ Model loaded successfully")

        model = EnhancedAttentionAwareUNet(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            dropout_rate=0.1,
            use_attention=True,
            attention_strength=0.3,
            num_position_classes=8
        )

        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.eval()

        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        traceback.print_exc()
        return None


class EnhancedTestResultSaver:
    """Enhanced test result saver with comprehensive visualizations"""

    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.save_dir / "predictions").mkdir(exist_ok=True)
        (self.save_dir / "attention_maps").mkdir(exist_ok=True)
        (self.save_dir / "comparisons").mkdir(exist_ok=True)
        (self.save_dir / "overlays").mkdir(exist_ok=True)  # For overlay images

        self.results = []

    def binarize_prediction(self, prediction, threshold=0.5):
        """Binarize prediction with threshold"""
        return (prediction > threshold).astype(np.float32)

    def create_comparison_figure(self, original_img, true_mask, pred_mask, attention_map,
                                 filename, case_id, position, dice, iou):
        """Create comprehensive comparison figure"""
        try:
            # Binarize prediction
            pred_binary = self.binarize_prediction(pred_mask)

            # Normalize attention map for visualization
            if attention_map.max() > attention_map.min():
                attention_normalized = (attention_map - attention_map.min()) / (
                            attention_map.max() - attention_map.min() + 1e-8)
            else:
                attention_normalized = np.zeros_like(attention_map)

            # Create figure with 2x3 layout
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1. Original DSA image
            axes[0, 0].imshow(original_img, cmap='gray')
            axes[0, 0].set_title('1. Original DSA Image', fontsize=12, fontweight='bold')
            axes[0, 0].axis('off')

            # 2. True mask
            axes[0, 1].imshow(true_mask, cmap='gray')
            axes[0, 1].set_title('2. Ground Truth Mask', fontsize=12, fontweight='bold')
            axes[0, 1].axis('off')

            # 3. Model-generated mask (binarized)
            axes[0, 2].imshow(pred_binary, cmap='gray')
            axes[0, 2].set_title(f'3. Model Prediction (Binary)\nDice: {dice:.3f}, IoU: {iou:.3f}',
                                 fontsize=12, fontweight='bold')
            axes[0, 2].axis('off')

            # 4. Position attention/weight map
            im4 = axes[1, 0].imshow(attention_normalized, cmap='hot')
            axes[1, 0].set_title('4. Position Attention Map', fontsize=12, fontweight='bold')
            axes[1, 0].axis('off')
            plt.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Attention Weight')

            # 5. Attention + True Mask + DSA overlay
            axes[1, 1].imshow(original_img, cmap='gray', alpha=0.7)
            axes[1, 1].imshow(attention_normalized, cmap='hot', alpha=0.5)
            # Add true mask contour
            axes[1, 1].contour(true_mask, colors='lime', linewidths=2, alpha=0.8)
            axes[1, 1].set_title('5. Attention + True Mask Overlay\n(Green: Ground Truth)',
                                 fontsize=12, fontweight='bold')
            axes[1, 1].axis('off')

            # 6. Model Prediction + DSA overlay
            axes[1, 2].imshow(original_img, cmap='gray', alpha=0.7)
            axes[1, 2].imshow(pred_binary, cmap='Reds', alpha=0.5)
            axes[1, 2].set_title('6. Prediction + DSA Overlay\n(Red: Model Prediction)',
                                 fontsize=12, fontweight='bold')
            axes[1, 2].axis('off')

            # Add overall title
            plt.suptitle(f'Medical Record: {case_id} | Position: {position} | File: {filename}',
                         fontsize=14, fontweight='bold', y=0.98)

            plt.tight_layout()

            # Save comparison figure
            safe_name = re.sub(r'[^\w\-_.]', '_', filename)[:50]
            comparison_path = self.save_dir / "comparisons" / f'{safe_name}_comparison.png'
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Create individual overlay images
            self._create_individual_overlays(original_img, true_mask, pred_binary,
                                             attention_normalized, safe_name, case_id)

            return str(comparison_path)

        except Exception as e:
            print(f"Failed to create comparison figure: {e}")
            traceback.print_exc()
            return None

    def _create_individual_overlays(self, original_img, true_mask, pred_binary,
                                    attention_normalized, safe_name, case_id):
        """Create individual overlay images"""
        try:
            # 1. Attention + True Mask + DSA overlay (detailed)
            fig1, ax1 = plt.subplots(figsize=(8, 8))
            ax1.imshow(original_img, cmap='gray', alpha=0.6)
            ax1.imshow(attention_normalized, cmap='hot', alpha=0.4)

            # Create colored mask overlay
            overlay_mask = np.zeros((*true_mask.shape, 3))
            overlay_mask[true_mask > 0.5, 1] = 0.7  # Green for true mask

            ax1.imshow(overlay_mask, alpha=0.6)
            ax1.set_title(f'Attention + True Mask Overlay\nMedical Record: {case_id}',
                          fontsize=12, fontweight='bold')
            ax1.axis('off')
            plt.tight_layout()
            plt.savefig(self.save_dir / "overlays" / f'{safe_name}_attention_true_overlay.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

            # 2. Model Prediction + DSA overlay (detailed)
            fig2, ax2 = plt.subplots(figsize=(8, 8))
            ax2.imshow(original_img, cmap='gray', alpha=0.6)

            # Create colored prediction overlay
            pred_overlay = np.zeros((*pred_binary.shape, 3))
            pred_overlay[pred_binary > 0.5, 0] = 0.7  # Red for prediction
            pred_overlay[pred_binary > 0.5, 2] = 0.3  # Add some blue

            ax2.imshow(pred_overlay, alpha=0.6)
            ax2.set_title(f'Model Prediction Overlay\nMedical Record: {case_id}',
                          fontsize=12, fontweight='bold')
            ax2.axis('off')
            plt.tight_layout()
            plt.savefig(self.save_dir / "overlays" / f'{safe_name}_prediction_overlay.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

            # 3. Side-by-side comparison
            fig3, (ax3, ax4, ax5) = plt.subplots(1, 3, figsize=(15, 5))

            # Original + attention
            ax3.imshow(original_img, cmap='gray')
            ax3.imshow(attention_normalized, cmap='hot', alpha=0.5)
            ax3.set_title('DSA + Attention')
            ax3.axis('off')

            # Original + true mask
            ax4.imshow(original_img, cmap='gray')
            ax4.imshow(true_mask, cmap='Greens', alpha=0.5)
            ax4.set_title('DSA + Ground Truth')
            ax4.axis('off')

            # Original + prediction
            ax5.imshow(original_img, cmap='gray')
            ax5.imshow(pred_binary, cmap='Reds', alpha=0.5)
            ax5.set_title('DSA + Prediction')
            ax5.axis('off')

            plt.suptitle(f'Medical Record: {case_id}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.save_dir / "overlays" / f'{safe_name}_side_by_side.png',
                        dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Failed to create individual overlays: {e}")

    def save_sample_result(self, filename, image, true_mask, pred_mask, attention_map,
                           dice, iou, sensitivity, specificity, case_id, position):
        """Save individual sample result"""
        try:
            # Generate safe filename
            safe_name = re.sub(r'[^\w\-_.]', '_', filename)
            safe_name = safe_name[:50]  # Limit length

            # Convert to numpy
            img_np = image.squeeze().cpu().numpy() if isinstance(image, torch.Tensor) else image.squeeze()
            true_np = true_mask.squeeze().cpu().numpy() if isinstance(true_mask, torch.Tensor) else true_mask.squeeze()
            pred_np = pred_mask.squeeze().cpu().numpy() if isinstance(pred_mask, torch.Tensor) else pred_mask.squeeze()
            att_np = attention_map.squeeze().cpu().numpy() if isinstance(attention_map,
                                                                         torch.Tensor) else attention_map.squeeze()

            # 1. Save binarized prediction
            pred_binary = self.binarize_prediction(pred_np)

            plt.figure(figsize=(8, 6))
            plt.imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title(
                f'Binarized Prediction\nMedical Record: {case_id}, Position: {position}\nDice: {dice:.3f}, IoU: {iou:.3f}')
            plt.savefig(self.save_dir / "predictions" / f'{safe_name}_pred_binary.png',
                        dpi=120, bbox_inches='tight')
            plt.close()

            # Also save raw prediction (probability)
            plt.figure(figsize=(8, 6))
            plt.imshow(pred_np, cmap='gray', vmin=0, vmax=1)
            plt.axis('off')
            plt.title(f'Raw Prediction Probability\nMedical Record: {case_id}')
            plt.colorbar(label='Probability')
            plt.savefig(self.save_dir / "predictions" / f'{safe_name}_pred_raw.png',
                        dpi=120, bbox_inches='tight')
            plt.close()

            # 2. Save attention map
            plt.figure(figsize=(8, 6))
            plt.imshow(att_np, cmap='hot')
            plt.axis('off')
            plt.title(f'Position Attention Map\nMedical Record: {case_id}')
            plt.colorbar(label='Attention Weight')
            plt.savefig(self.save_dir / "attention_maps" / f'{safe_name}_attention.png',
                        dpi=120, bbox_inches='tight')
            plt.close()

            # 3. Create and save comprehensive comparison figure
            comparison_path = self.create_comparison_figure(
                original_img=img_np,
                true_mask=true_np,
                pred_mask=pred_np,
                attention_map=att_np,
                filename=filename,
                case_id=case_id,
                position=position,
                dice=dice,
                iou=iou
            )

            # 4. Save individual DSA image
            plt.figure(figsize=(8, 6))
            plt.imshow(img_np, cmap='gray')
            plt.axis('off')
            plt.title(f'Original DSA Image\nMedical Record: {case_id}')
            plt.savefig(self.save_dir / f'{safe_name}_dsa.png',
                        dpi=120, bbox_inches='tight')
            plt.close()

            # 5. Save true mask
            plt.figure(figsize=(8, 6))
            plt.imshow(true_np, cmap='gray')
            plt.axis('off')
            plt.title(f'Ground Truth Mask\nMedical Record: {case_id}')
            plt.savefig(self.save_dir / f'{safe_name}_truth.png',
                        dpi=120, bbox_inches='tight')
            plt.close()

            # Record results
            self.results.append({
                'filename': filename,
                'medical_record': case_id,
                'position': position,
                'dice': float(dice),
                'iou': float(iou),
                'sensitivity': float(sensitivity),
                'specificity': float(specificity),
                'prediction_binary': f'{safe_name}_pred_binary.png',
                'prediction_raw': f'{safe_name}_pred_raw.png',
                'attention_map': f'{safe_name}_attention.png',
                'comparison_figure': f'{safe_name}_comparison.png',
                'dsa_image': f'{safe_name}_dsa.png',
                'truth_mask': f'{safe_name}_truth.png',
                'attention_overlay': f'{safe_name}_attention_true_overlay.png',
                'prediction_overlay': f'{safe_name}_prediction_overlay.png',
                'side_by_side': f'{safe_name}_side_by_side.png'
            })

            print(f"  Saved: {filename}")
            print(f"    Dice: {dice:.3f}, IoU: {iou:.3f}")

            # Clean memory
            del img_np, true_np, pred_np, att_np, pred_binary
            gc.collect()

        except Exception as e:
            print(f"Failed to save sample {filename} result: {e}")
            traceback.print_exc()

    def save_final_report(self, avg_dice, avg_iou, avg_sens, avg_spec, auc_score):
        """Save final report"""
        try:
            # Save CSV results
            if self.results:
                df = pd.DataFrame(self.results)
                csv_path = self.save_dir / "test_results.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"✓ CSV results saved to: {csv_path}")

            # Save text summary
            summary_path = self.save_dir / "test_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("Brain Aneurysm Segmentation Model Test Results\n")
                f.write("=" * 70 + "\n\n")

                f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Test Samples: {len(self.results)}\n\n")

                f.write("Overall Performance Metrics:\n")
                f.write("-" * 40 + "\n")
                f.write(f"  Average Dice Coefficient: {avg_dice:.4f}\n")
                f.write(f"  Average IoU Coefficient:  {avg_iou:.4f}\n")
                f.write(f"  Average Sensitivity:      {avg_sens:.4f}\n")
                f.write(f"  Average Specificity:      {avg_spec:.4f}\n")
                f.write(f"  AUC:                     {auc_score:.4f}\n\n")

                f.write("File Structure Explanation:\n")
                f.write("-" * 40 + "\n")
                f.write("  predictions/           : Model prediction results\n")
                f.write("    *_pred_binary.png    : Binarized predictions (threshold=0.5)\n")
                f.write("    *_pred_raw.png       : Raw probability predictions\n")
                f.write("  attention_maps/        : Position attention/weight maps\n")
                f.write("  comparisons/           : Comprehensive 6-panel comparison figures\n")
                f.write("  overlays/              : Individual overlay images\n")
                f.write("    *_attention_true_overlay.png : Attention + True Mask overlay\n")
                f.write("    *_prediction_overlay.png     : Prediction overlay\n")
                f.write("    *_side_by_side.png           : Side-by-side comparison\n\n")

                if self.results:
                    f.write("Top 10 Sample Results:\n")
                    f.write("-" * 40 + "\n")
                    sorted_results = sorted(self.results, key=lambda x: x['dice'], reverse=True)
                    for i, result in enumerate(sorted_results[:10]):
                        f.write(f"\n{i + 1}. File: {result['filename']}\n")
                        f.write(f"   Medical Record: {result['medical_record']}\n")
                        f.write(f"   Position:       {result['position']}\n")
                        f.write(f"   Dice:           {result['dice']:.4f}\n")
                        f.write(f"   IoU:            {result['iou']:.4f}\n")

                    if len(self.results) > 10:
                        f.write(f"\n... and {len(self.results) - 10} more samples, see CSV for details\n")

            print(f"✓ Text summary saved to: {summary_path}")

            # Create performance plots
            if self.results:
                self._create_performance_plots(avg_dice, avg_iou, auc_score)

        except Exception as e:
            print(f"Failed to save final report: {e}")
            traceback.print_exc()

    def _create_performance_plots(self, avg_dice, avg_iou, auc_score):
        """Create performance distribution plots"""
        try:
            df = pd.DataFrame(self.results)

            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Dice distribution
            axes[0, 0].hist(df['dice'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            axes[0, 0].axvline(avg_dice, color='red', linestyle='--', linewidth=2,
                               label=f'Average: {avg_dice:.3f}')
            axes[0, 0].set_xlabel('Dice Coefficient')
            axes[0, 0].set_ylabel('Number of Samples')
            axes[0, 0].set_title('Dice Coefficient Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. IoU distribution
            axes[0, 1].hist(df['iou'], bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
            axes[0, 1].axvline(avg_iou, color='red', linestyle='--', linewidth=2,
                               label=f'Average: {avg_iou:.3f}')
            axes[0, 1].set_xlabel('IoU Coefficient')
            axes[0, 1].set_ylabel('Number of Samples')
            axes[0, 1].set_title('IoU Coefficient Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Dice vs IoU scatter plot
            axes[1, 0].scatter(df['dice'], df['iou'], alpha=0.6, c=df['position'],
                               cmap='tab10', s=50)
            axes[1, 0].set_xlabel('Dice Coefficient')
            axes[1, 0].set_ylabel('IoU Coefficient')
            axes[1, 0].set_title('Dice vs IoU (Color: Position)')
            axes[1, 0].grid(True, alpha=0.3)

            # Add color bar
            scatter = axes[1, 0].collections[0]
            cbar = plt.colorbar(scatter, ax=axes[1, 0])
            cbar.set_label('Position Number')

            # 4. Performance summary
            axes[1, 1].axis('off')
            summary_text = (
                f'Test Results Summary\n\n'
                f'Total Samples: {len(df)}\n'
                f'Average Dice: {avg_dice:.4f}\n'
                f'Average IoU: {avg_iou:.4f}\n'
                f'AUC: {auc_score:.4f}\n\n'
                f'Best Dice: {df["dice"].max():.4f}\n'
                f'Worst Dice: {df["dice"].min():.4f}\n'
                f'Standard Deviation: {df["dice"].std():.4f}'
            )
            axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12,
                            verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.suptitle('Test Performance Analysis', fontsize=16, y=0.98)
            plt.tight_layout()
            plt.savefig(self.save_dir / "performance_analysis.png",
                        dpi=150, bbox_inches='tight')
            plt.close()

            print(f"✓ Performance analysis plot saved to: {self.save_dir}/performance_analysis.png")

        except Exception as e:
            print(f"Failed to create performance plots: {e}")


def test_dicom_model(model_path, test_image_dir, test_mask_dir,
                     position_excel_path, batch_size=4,
                     result_base_dir="D:/med_data/ai/result"):
    """Test DICOM model with comprehensive visualizations"""
    print("=" * 80)
    print("DICOM Model Testing with Comprehensive Visualizations")
    print("=" * 80)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Clean GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Extract result folder name from model path
    try:
        model_path_obj = Path(model_path)
        experiment_name = model_path_obj.parent.parent.parent.name
        fold_name = model_path_obj.parent.parent.name
        result_folder = f"{experiment_name}_{fold_name}"
        result_dir = Path(result_base_dir) / result_folder

        print(f"Model: {model_path_obj.name}")
        print(f"Experiment: {experiment_name}")
        print(f"Fold: {fold_name}")
        print(f"Result Directory: {result_dir}")

    except Exception as e:
        print(f"Failed to parse path: {e}")
        result_dir = Path(result_base_dir) / "dicom_test_results"

    # 2. Load model
    print("\n1. Loading model...")
    model = load_model_safely(model_path, device)
    if model is None:
        print("Error: Cannot load model")
        return

    # 3. Load position information
    print("\n2. Loading position information...")
    position_loader = MedicalRecordPositionLoader(position_excel_path)

    # 4. Create DICOM dataset
    print("\n3. Creating DICOM test dataset...")
    try:
        dataset = DicomTestDataset(
            image_dir=test_image_dir,
            mask_dir=test_mask_dir,
            position_loader=position_loader,
            max_samples=None  # Use all samples
        )

        if len(dataset) == 0:
            print("Error: DICOM dataset is empty")
            return

        print(f"Ready to test {len(dataset)} DICOM samples")

    except Exception as e:
        print(f"Failed to create DICOM dataset: {e}")
        traceback.print_exc()
        return

    # 5. Create data loader
    test_loader = DataLoader(
        dataset,
        batch_size=min(batch_size, 4),  # Limit batch size
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    print(f"Batch Size: {test_loader.batch_size}")
    print(f"Total Batches: {len(test_loader)}")

    # 6. Create result saver
    result_saver = EnhancedTestResultSaver(result_dir)

    # 7. Start testing
    print("\n4. Starting test...")
    total_dice = 0.0
    total_iou = 0.0
    total_sensitivity = 0.0
    total_specificity = 0.0

    all_preds = []
    all_targets = []

    model.eval()

    try:
        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Test Progress", unit="batch")
            for batch_idx, batch_data in enumerate(pbar):
                try:
                    images, masks, positions, case_ids, filenames = batch_data

                    # Move to device
                    images = images.to(device)
                    masks = masks.to(device)
                    positions = positions.to(device)

                    # Forward pass
                    outputs, attention_maps = model(images, positions)

                    # Calculate batch metrics
                    batch_dices = []
                    batch_ious = []
                    batch_sensitivities = []
                    batch_specificities = []

                    for i in range(len(images)):
                        output_i = outputs[i:i + 1]
                        mask_i = masks[i:i + 1]

                        dice = calculate_dice_safe(output_i, mask_i)
                        iou = calculate_iou_safe(output_i, mask_i)
                        sensitivity, specificity = calculate_sensitivity_specificity_safe(output_i, mask_i)

                        batch_dices.append(dice)
                        batch_ious.append(iou)
                        batch_sensitivities.append(sensitivity)
                        batch_specificities.append(specificity)

                        total_dice += dice
                        total_iou += iou
                        total_sensitivity += sensitivity
                        total_specificity += specificity

                        # Collect predictions
                        all_preds.extend(output_i.cpu().numpy().flatten())
                        all_targets.extend(mask_i.cpu().numpy().flatten())

                    # Save batch results
                    for i in range(len(images)):
                        result_saver.save_sample_result(
                            filename=filenames[i],
                            image=images[i],
                            true_mask=masks[i],
                            pred_mask=outputs[i],
                            attention_map=attention_maps[i] if attention_maps is not None else torch.zeros_like(
                                images[i]),
                            dice=batch_dices[i],
                            iou=batch_ious[i],
                            sensitivity=batch_sensitivities[i],
                            specificity=batch_specificities[i],
                            case_id=case_ids[i],
                            position=torch.argmax(positions[i]).item() if positions[i].dim() > 0 else positions[
                                i].item()
                        )

                    # Update progress bar
                    current_avg_dice = total_dice / ((batch_idx + 1) * len(images))
                    current_avg_iou = total_iou / ((batch_idx + 1) * len(images))
                    pbar.set_postfix({
                        'Avg Dice': f'{current_avg_dice:.3f}',
                        'Avg IoU': f'{current_avg_iou:.3f}'
                    })

                    # Clean memory
                    del images, masks, positions, outputs, attention_maps
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if batch_idx % 5 == 0:
                        gc.collect()

                except Exception as e:
                    print(f"\nBatch {batch_idx} processing failed: {e}")
                    traceback.print_exc()
                    continue

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest process error: {e}")
        traceback.print_exc()

    # 8. Calculate final metrics
    num_samples = len(dataset)
    if num_samples > 0:
        avg_dice = total_dice / num_samples
        avg_iou = total_iou / num_samples
        avg_sensitivity = total_sensitivity / num_samples
        avg_specificity = total_specificity / num_samples
    else:
        avg_dice = avg_iou = avg_sensitivity = avg_specificity = 0.0

    # 9. Calculate AUC
    auc_score = 0.0
    if len(all_preds) > 0 and len(np.unique(all_targets)) > 1:
        try:
            fpr, tpr, _ = roc_curve(all_targets, all_preds)
            auc_score = auc(fpr, tpr)

            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc_score:.3f}')
            plt.plot([0, 1], [0, 1], 'k--', lw=1)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig(result_dir / "roc_curve.png", dpi=120, bbox_inches='tight')
            plt.close()
            print(f"✓ ROC curve saved to: {result_dir}/roc_curve.png")

        except Exception as e:
            print(f"Failed to calculate AUC: {e}")

    # 10. Save final report
    print("\n5. Saving test results...")
    result_saver.save_final_report(
        avg_dice=avg_dice,
        avg_iou=avg_iou,
        avg_sens=avg_sensitivity,
        avg_spec=avg_specificity,
        auc_score=auc_score
    )

    # 11. Print summary
    print("\n" + "=" * 80)
    print("Test Completed!")
    print("=" * 80)
    print(f"Test Samples: {num_samples}")
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average IoU Coefficient:  {avg_iou:.4f}")
    print(f"Average Sensitivity:      {avg_sensitivity:.4f}")
    print(f"Average Specificity:      {avg_specificity:.4f}")
    print(f"AUC:                     {auc_score:.4f}")
    print(f"\nResult Directory: {result_dir}")

    # Show folder structure
    if result_dir.exists():
        print("\nGenerated File Structure:")
        folders = {
            "predictions": "Model prediction results",
            "attention_maps": "Position attention maps",
            "comparisons": "6-panel comparison figures",
            "overlays": "Individual overlay images"
        }

        for folder_name, description in folders.items():
            folder_path = result_dir / folder_name
            if folder_path.exists():
                file_count = len(list(folder_path.glob("*")))
                print(f"  📁 {folder_name}/ - {description} ({file_count} files)")

        # Show important files
        important_files = [
            "test_summary.txt", "test_results.csv",
            "performance_analysis.png", "roc_curve.png"
        ]

        print(f"\nImportant Files:")
        for file_name in important_files:
            file_path = result_dir / file_name
            if file_path.exists():
                size_kb = file_path.stat().st_size / 1024
                print(f"  📄 {file_name} ({size_kb:.1f} KB)")


def main():
    """Main function"""
    print("DICOM Brain Aneurysm Segmentation Model Test Program")
    print("-" * 60)

    # Configuration
    config = {
        'model_path': "D:/med_data/ai/model/20260201_795/folds/fold3/models/model_fold_3.pth",
        'test_image_dir': "D:/med_data/ai/test1",
        'test_mask_dir': "D:/med_data/ai/test2",
        'position_excel_path': "D:/med_data/ai/classify.xlsx",
        'batch_size': 4,
        'result_base_dir': "D:/med_data/ai/result"
    }

    # Display configuration
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 60)

    # Run test
    try:
        test_dicom_model(**config)
    except Exception as e:
        print(f"Test process error: {e}")
        traceback.print_exc()

    print("\nProgram Ended")


if __name__ == "__main__":
    # Add memory cleanup
    import gc

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Run main function
    main()