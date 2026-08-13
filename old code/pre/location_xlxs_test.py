import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import warnings
from enum import Enum

warnings.filterwarnings('ignore')


class SizeCategory(Enum):
    """Size classification enumeration"""
    EXTRA_SMALL = "Extra Small"
    SMALL = "Small"
    MEDIUM = "Medium"
    LARGE = "Large"


class MaskLocationExtractor:
    """Extract aneurysm location information from mask images (using minimum enclosing circle to ensure complete coverage)"""

    def __init__(self, mask_dir="D:\\med_data\\ai\\test2",
                 output_excel="D:\\med_data\\ai\\location4.xlsx",
                 visualization_dir="D:\\med_data\\ai\\mask4"):
        self.mask_dir = Path(mask_dir)
        self.output_excel = Path(output_excel)
        self.visualization_dir = Path(visualization_dir)

        # Create output directory
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        # Store results
        self.location_data = []

        # Radius classification thresholds for 4 categories (configurable)
        self.threshold_1 = 0.04  # Extra Small upper limit
        self.threshold_2 = 0.1  # Small upper limit
        self.threshold_3 = 0.3  # Medium upper limit
        self.threshold_4 = 0.4
        # Note: Large radius > threshold_3

        # Enable classification flag
        self.enable_size_classification = True

    def set_size_classification_thresholds(self, thresh1, thresh2, thresh3):
        """Set thresholds for size classification (4 categories)

        Args:
            thresh1: Upper limit for extra small radius
            thresh2: Upper limit for small radius
            thresh3: Upper limit for medium radius (large radius > thresh3)
        """
        self.threshold_1 = thresh1
        self.threshold_2 = thresh2
        self.threshold_3 = thresh3
        print(f"Size classification thresholds set: "
              f"Extra Small ≤ {thresh1}, "
              f"Small ≤ {thresh2}, "
              f"Medium ≤ {thresh3}, "
              f"Large > {thresh3}")

    def classify_size_by_radius(self, radius_ratio):
        """Classify size based on radius ratio (4 categories)

        Args:
            radius_ratio: Radius ratio (value between 0-1)

        Returns:
            SizeCategory: Size classification
        """
        if not self.enable_size_classification:
            return None

        if radius_ratio <= self.threshold_1:
            return SizeCategory.EXTRA_SMALL
        elif radius_ratio <= self.threshold_2:
            return SizeCategory.SMALL
        elif radius_ratio <= self.threshold_3:
            return SizeCategory.MEDIUM
        else:
            return SizeCategory.LARGE

    def get_size_range(self, size_category):
        """Get the size range string for a given size category

        Args:
            size_category: SizeCategory enum value

        Returns:
            str: Size range description
        """
        if size_category == SizeCategory.EXTRA_SMALL:
            return f" {self.threshold_1}"
        elif size_category == SizeCategory.SMALL:
            return f"{self.threshold_2}"
        elif size_category == SizeCategory.MEDIUM:
            return f"{self.threshold_3}"
        elif size_category == SizeCategory.LARGE:
            return f"{self.threshold_4}"
        else:
            return "N/A"

    def find_mask_files(self):
        """Find all TIF format mask files"""
        mask_files = list(self.mask_dir.glob("*.tif"))
        print(f"Found {len(mask_files)} TIF format mask files")

        # Sort by filename
        mask_files.sort()

        # Show first 10 files
        print("First 10 files:")
        for i, file in enumerate(mask_files[:10]):
            print(f"  {i + 1}. {file.name}")

        return mask_files

    def load_mask(self, mask_path):
        """Load mask image and preprocess"""
        try:
            # Read TIF file using cv2
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if mask is None:
                print(f"Warning: Cannot read {mask_path.name}")
                return None

            # Convert to 0-255 range
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)

            # Binarize (ensure 0 and 255)
            _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Convert to 0-1
            mask_normalized = (mask_binary > 127).astype(np.uint8)

            return mask_normalized

        except Exception as e:
            print(f"Failed to load mask {mask_path.name}: {e}")
            return None

    def find_main_aneurysm_region(self, mask):
        """Find the main aneurysm region"""
        try:
            # Label connected regions
            labeled_mask, num_labels = label(mask, connectivity=2, return_num=True)

            if num_labels == 0:
                return None

            # Get all region properties
            regions = regionprops(labeled_mask)

            if not regions:
                return None

            # Select the largest region (assuming the largest region is the aneurysm)
            largest_region = max(regions, key=lambda r: r.area)

            # Filter out regions that are too small
            if largest_region.area < 10:
                return None

            return largest_region

        except Exception as e:
            print(f"Failed to find main aneurysm region: {e}")
            return None

    def calculate_minimum_enclosing_circle(self, region):
        """Calculate minimum enclosing circle (ensuring complete coverage of aneurysm)"""
        try:
            # Get all pixel coordinates of the aneurysm
            coords = region.coords  # [y, x] format

            if len(coords) < 3:
                # If too few points, use bounding box calculation
                min_row, min_col, max_row, max_col = region.bbox
                center_x = (min_col + max_col) / 2
                center_y = (min_row + max_row) / 2
                radius = max((max_row - min_row), (max_col - min_col)) / 2
                return center_y, center_x, radius

            # Use OpenCV's minEnclosingCircle to calculate minimum enclosing circle
            # Note: OpenCV's minEnclosingCircle requires [x, y] format
            points = coords[:, [1, 0]].astype(np.float32)  # Convert to [x, y] format

            (center_x, center_y), radius = cv2.minEnclosingCircle(points)

            # Ensure radius is large enough (at least cover the main area)
            # Add 10% margin to ensure complete coverage
            radius = radius * 1.1

            return center_y, center_x, radius

        except Exception as e:
            print(f"Failed to calculate minimum enclosing circle: {e}")
            # Backup plan: use bounding box calculation
            min_row, min_col, max_row, max_col = region.bbox
            center_x = (min_col + max_col) / 2
            center_y = (min_row + max_row) / 2
            radius = max((max_row - min_row), (max_col - min_col)) / 2 * 1.2
            return center_y, center_x, radius

    def create_circle_mask(self, image_shape, center_y, center_x, radius):
        """Create circular mask"""
        h, w = image_shape
        y_coords, x_coords = np.ogrid[:h, :w]

        # Calculate distance
        dist = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

        # Create circular mask
        circle_mask = (dist <= radius).astype(np.uint8)

        return circle_mask

    def check_coverage_and_adjust(self, original_mask, center_y, center_x, radius):
        """Check coverage and adjust radius to ensure complete coverage"""
        # Create initial circle
        h, w = original_mask.shape
        initial_circle = self.create_circle_mask((h, w), center_y, center_x, radius)

        # Check coverage
        aneurysm_area = np.sum(original_mask > 0)
        overlap_area = np.sum(np.logical_and(original_mask > 0, initial_circle > 0))
        initial_coverage = overlap_area / aneurysm_area if aneurysm_area > 0 else 0

        # If coverage is already high (>95%), return directly
        if initial_coverage > 0.95:
            return center_y, center_x, radius, initial_coverage

        # If coverage is insufficient, gradually increase radius
        max_radius = min(h, w) / 2  # Maximum radius is half the image dimension
        adjusted_radius = radius

        for i in range(20):  # Try up to 20 times
            if adjusted_radius >= max_radius:
                break

            # Slightly increase radius
            adjusted_radius = adjusted_radius * 1.05

            # Create new circle
            new_circle = self.create_circle_mask((h, w), center_y, center_x, adjusted_radius)

            # Calculate new coverage
            new_overlap = np.sum(np.logical_and(original_mask > 0, new_circle > 0))
            new_coverage = new_overlap / aneurysm_area if aneurysm_area > 0 else 0

            # If satisfactory coverage reached or no further improvement, stop
            if new_coverage > 0.99 or (i > 5 and new_coverage - initial_coverage < 0.01):
                return center_y, center_x, adjusted_radius, new_coverage

        return center_y, center_x, adjusted_radius, new_coverage

    def visualize_results(self, original_mask, circle_mask, filename,
                          center_y, center_x, radius, region_bbox, save_figures=True):
        """Visualize results and save"""
        if not save_figures:
            return 0.0, 0

        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1. Original mask
            axes[0, 0].imshow(original_mask, cmap='gray')
            axes[0, 0].set_title(f'Original Mask: {filename}')
            axes[0, 0].axis('off')

            # Mark bounding box
            min_row, min_col, max_row, max_col = region_bbox
            rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                 fill=False, edgecolor='red', linewidth=2)
            axes[0, 0].add_patch(rect)

            # 2. Minimum enclosing circle analysis
            axes[0, 1].imshow(original_mask, cmap='gray')
            axes[0, 1].set_title('Minimum Enclosing Circle Analysis')
            axes[0, 1].axis('off')

            # Draw circle center
            axes[0, 1].plot(center_x, center_y, 'go', markersize=8, linewidth=2, label='Center')

            # Draw circle boundary
            theta = np.linspace(0, 2 * np.pi, 100)
            circle_x = center_x + radius * np.cos(theta)
            circle_y = center_y + radius * np.sin(theta)
            axes[0, 1].plot(circle_x, circle_y, 'g-', linewidth=2, label=f'Radius={radius:.1f}')

            # Draw bounding box
            rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                 fill=False, edgecolor='red', linewidth=2, linestyle='--',
                                 label='Bounding Box')
            axes[0, 1].add_patch(rect)

            axes[0, 1].legend(loc='upper right')

            # 3. Generated circle
            axes[0, 2].imshow(circle_mask, cmap='gray')
            axes[0, 2].set_title(f'Minimum Enclosing Circle (Radius={radius:.1f} pixels)')
            axes[0, 2].axis('off')
            axes[0, 2].plot(center_x, center_y, 'go', markersize=8, linewidth=2)

            # 4. Overlay image (color)
            overlap = np.zeros((*original_mask.shape, 3), dtype=np.uint8)

            # Aneurysm area: green
            aneurysm_mask = original_mask > 0
            overlap[aneurysm_mask, 1] = 255  # Green

            # Circle area: red (semi-transparent)
            circle_area = circle_mask > 0
            overlap[circle_area, 0] = 128  # Red

            # Overlap area: yellow
            overlap_area = np.logical_and(aneurysm_mask, circle_area)
            overlap[overlap_area, 0] = 255  # Red
            overlap[overlap_area, 1] = 255  # Green
            overlap[overlap_area, 2] = 0  # No blue

            axes[1, 0].imshow(overlap)
            axes[1, 0].set_title('Overlay Image (Green:Aneurysm, Red:Circle, Yellow:Overlap)')
            axes[1, 0].axis('off')

            # 5. Coverage analysis
            axes[1, 1].imshow(original_mask, cmap='gray')

            # Mark uncovered areas
            uncovered = np.logical_and(aneurysm_mask, ~circle_area)
            if np.any(uncovered):
                # Find contours of uncovered areas
                uncovered_uint8 = uncovered.astype(np.uint8) * 255
                contours, _ = cv2.findContours(uncovered_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 5:  # Only show areas larger than 5 pixels
                        contour = contour.squeeze()
                        axes[1, 1].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.7)

            axes[1, 1].set_title('Uncovered Area Analysis (Red: Uncovered)')
            axes[1, 1].axis('off')

            # 6. Statistics information
            axes[1, 2].axis('off')

            # Calculate statistics
            aneurysm_area = np.sum(aneurysm_mask)
            circle_area_sum = np.sum(circle_area)
            overlap_area_sum = np.sum(overlap_area)

            coverage = overlap_area_sum / aneurysm_area if aneurysm_area > 0 else 0
            uncovered_area = aneurysm_area - overlap_area_sum

            bbox_width = max_col - min_col
            bbox_height = max_row - min_row

            info_text = f"File: {filename}\n"
            info_text += "=" * 40 + "\n"
            info_text += f"Center: ({center_x:.1f}, {center_y:.1f})\n"
            info_text += f"Circle Radius: {radius:.1f} pixels\n"
            info_text += f"Bounding Box: {bbox_width:.1f}×{bbox_height:.1f}\n"
            info_text += f"Aneurysm Area: {aneurysm_area} pixels\n"
            info_text += f"Circle Area: {circle_area_sum} pixels\n"
            info_text += f"Overlap Area: {overlap_area_sum} pixels\n"
            info_text += f"Uncovered Area: {uncovered_area} pixels\n"
            info_text += f"Coverage: {coverage:.1%}\n"

            if uncovered_area > 0:
                info_text += f"Uncovered Rate: {(1 - coverage):.1%}"

            axes[1, 2].text(0.1, 0.5, info_text, transform=axes[1, 2].transAxes,
                            fontsize=10, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.suptitle(f'Aneurysm Location Extraction - Minimum Enclosing Circle Method: {filename}', fontsize=16,
                         y=1.02)
            plt.tight_layout()

            # Save visualization results
            output_name = filename.replace('.tif', '_analysis.png')
            output_path = self.visualization_dir / output_name
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Save overlay image separately
            overlap_name = filename.replace('.tif', '_overlap.png')
            overlap_path = self.visualization_dir / overlap_name
            cv2.imwrite(str(overlap_path), overlap)

            return coverage, uncovered_area

        except Exception as e:
            print(f"Visualization saving failed: {e}")
            return 0.0, 0

    def process_single_mask(self, mask_path, save_figures=True):
        """Process single mask file - using minimum enclosing circle method"""
        filename = mask_path.name
        print(f"\nProcessing: {filename}")

        # 1. Load mask
        mask = self.load_mask(mask_path)
        if mask is None:
            return None

        # Get image dimensions
        h, w = mask.shape

        # 2. Find main aneurysm region
        region = self.find_main_aneurysm_region(mask)

        if region is None:
            print(f"  {filename}: No valid aneurysm region found")
            return None

        # 3. Calculate minimum enclosing circle
        center_y, center_x, radius = self.calculate_minimum_enclosing_circle(region)

        print(f"  Initial center: ({center_x:.1f}, {center_y:.1f})")
        print(f"  Initial radius: {radius:.1f} pixels")

        # 4. Check and adjust radius to ensure complete coverage
        center_y, center_x, final_radius, coverage = self.check_coverage_and_adjust(
            mask, center_y, center_x, radius
        )

        print(f"  Adjusted center: ({center_x:.1f}, {center_y:.1f})")
        print(f"  Adjusted radius: {final_radius:.1f} pixels")
        print(f"  Estimated coverage: {coverage:.1%}")

        # 5. Create circular mask
        circle_mask = self.create_circle_mask((h, w), center_y, center_x, final_radius)

        # 6. Calculate normalized parameters
        # Height ratio: position from top to bottom, 0=top, 1=bottom
        height_ratio = center_y / h

        # Radius ratio: relative to image dimensions
        max_dim = max(h, w)
        radius_ratio = final_radius / (max_dim / 2)  # Maximum radius is half the image dimension

        # Set minimum radius ratio
        MIN_RADIUS_RATIO = 0.01
        if radius_ratio < MIN_RADIUS_RATIO:
            print(
                f"  Warning: Calculated radius ratio {radius_ratio:.3f} is less than minimum {MIN_RADIUS_RATIO}, setting to {MIN_RADIUS_RATIO}")
            radius_ratio = MIN_RADIUS_RATIO

        # Limit to 0-1 range
        height_ratio = max(0.0, min(1.0, height_ratio))
        radius_ratio = max(0.0, min(1.0, radius_ratio))

        # 6.5 Size classification - Using 4 categories with thresholds 0.04, 0.1, 0.3
        size_category = self.classify_size_by_radius(radius_ratio)
        size_range = None

        if size_category:
            size_range = self.get_size_range(size_category)
            print(f"  Size category: {size_category.value} (Radius ratio: {radius_ratio:.3f}, Range: {size_range})")

        # 7. Visualize and save
        bbox = region.bbox
        final_coverage, uncovered_area = self.visualize_results(
            mask, circle_mask, filename,
            center_y, center_x, final_radius, bbox, save_figures
        )

        # 8. Record results
        result = {
            'filename': filename.replace('.tif', ''),  # Remove extension
            'height_ratio': height_ratio,
            'radius_ratio': radius_ratio,
            'center_x': center_x,
            'center_y': center_y,
            'pixel_radius': final_radius,
            'initial_radius': radius,
            'coverage': final_coverage,
            'uncovered_area': uncovered_area,
            'aneurysm_area': region.area,
            'image_width': w,
            'image_height': h,
            'bbox_width': bbox[3] - bbox[1],
            'bbox_height': bbox[2] - bbox[0]
        }

        # Add size classification information
        if size_category:
            result['size_category'] = size_category.value  # "Extra Small", "Small", "Medium", or "Large"
            result['size_category_code'] = size_category.name  # "EXTRA_SMALL", "SMALL", "MEDIUM", or "LARGE"
            result['size_range'] = size_range  # e.g., "≤ 0.04", "≤ 0.1", "≤ 0.3", or "> 0.3"

        print(f"  Height ratio: {height_ratio:.3f}")
        print(f"  Radius ratio: {radius_ratio:.3f}")
        print(f"  Actual coverage: {final_coverage:.1%}")
        print(f"  Uncovered area: {uncovered_area} pixels")

        if final_coverage < 0.95:
            print(f"  ⚠️ Warning: Low coverage ({final_coverage:.1%})")

        return result

    def process_all_masks(self, save_figures=True):
        """Process all mask files"""
        print("Starting aneurysm location extraction (minimum enclosing circle method)...")
        print("=" * 60)

        # Find all mask files
        mask_files = self.find_mask_files()

        if not mask_files:
            print("Error: No mask files found")
            return False

        # Process each file
        successful_count = 0
        failed_count = 0
        low_coverage_count = 0

        for mask_path in mask_files:
            result = self.process_single_mask(mask_path, save_figures)

            if result:
                self.location_data.append(result)
                successful_count += 1

                # Count low coverage cases
                if result['coverage'] < 0.95:
                    low_coverage_count += 1
            else:
                failed_count += 1

        print(f"\nProcessing complete!")
        print(f"Successfully processed: {successful_count} files")
        print(f"Failed: {failed_count} files")
        if successful_count > 0:
            print(f"Coverage <95%: {low_coverage_count} ({low_coverage_count / successful_count * 100:.1f}%)")

        return successful_count > 0

    def save_to_excel(self):
        """Save results to Excel file with all information in English"""
        if not self.location_data:
            print("Error: No data to save")
            return False

        try:
            # Create DataFrame
            df = pd.DataFrame(self.location_data)

            # Select columns to save based on classification setting
            if self.enable_size_classification and 'size_category' in df.columns:
                # Include size category and size range columns
                output_df = df[['filename', 'height_ratio', 'radius_ratio',
                                'size_category', 'size_range']].copy()

                # Rename columns to English
                output_df.columns = ['Filename', 'Height Ratio', 'Radius Ratio',
                                     'Size Category', 'Size Range']
            else:
                output_df = df[['filename', 'height_ratio', 'radius_ratio']].copy()
                output_df.columns = ['Filename', 'Height Ratio', 'Radius Ratio']

            # Save to Excel
            output_df.to_excel(self.output_excel, index=False)

            print(f"\nLocation information saved to: {self.output_excel}")
            print(f"Total records: {len(output_df)}")

            # Show first few rows
            print("\nFirst 10 records:")
            print(output_df.head(10))

            # Save complete data to CSV (includes more information)
            full_csv_path = self.output_excel.with_suffix('.csv')

            # Rename all columns to English for CSV as well
            csv_df = df.copy()
            column_mapping = {
                'filename': 'Filename',
                'height_ratio': 'Height Ratio',
                'radius_ratio': 'Radius Ratio',
                'center_x': 'Center X',
                'center_y': 'Center Y',
                'pixel_radius': 'Pixel Radius',
                'initial_radius': 'Initial Radius',
                'coverage': 'Coverage',
                'uncovered_area': 'Uncovered Area',
                'aneurysm_area': 'Aneurysm Area',
                'image_width': 'Image Width',
                'image_height': 'Image Height',
                'bbox_width': 'Bounding Box Width',
                'bbox_height': 'Bounding Box Height',
                'size_category': 'Size Category',
                'size_category_code': 'Size Category Code',
                'size_range': 'Size Range'
            }

            # Only rename columns that exist
            rename_dict = {k: v for k, v in column_mapping.items() if k in csv_df.columns}
            csv_df = csv_df.rename(columns=rename_dict)

            csv_df.to_csv(full_csv_path, index=False, encoding='utf-8-sig')
            print(f"Complete data saved to: {full_csv_path}")

            return True

        except Exception as e:
            print(f"Failed to save Excel file: {e}")
            return False

    def generate_size_statistics_chart(self):
        """Generate size statistics chart showing size distribution for 4 categories"""
        if not self.location_data or not self.enable_size_classification:
            print("Insufficient data or classification not enabled, cannot generate size statistics chart")
            return

        df = pd.DataFrame(self.location_data)

        if 'size_category' not in df.columns:
            print("No size classification information in data")
            return

        # Create size statistics chart
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))

        # Define colors for 4 categories
        colors = {
            'Extra Small': '#4CAF50',  # Green
            'Small': '#2196F3',  # Blue
            'Medium': '#FFC107',  # Yellow/Orange
            'Large': '#F44336'  # Red
        }

        # 1. Size classification pie chart
        size_counts = df['size_category'].value_counts()
        # Ensure all categories are included even if count is 0
        for cat in ['Extra Small', 'Small', 'Medium', 'Large']:
            if cat not in size_counts.index:
                size_counts[cat] = 0
        size_counts = size_counts.sort_index()

        pie_colors = [colors.get(cat, '#9E9E9E') for cat in size_counts.index if size_counts[cat] > 0]
        non_zero_counts = size_counts[size_counts > 0]

        if len(non_zero_counts) > 0:
            axes[0, 0].pie(non_zero_counts.values, labels=non_zero_counts.index, autopct='%1.1f%%',
                           colors=pie_colors, startangle=90, explode=[0.05] * len(non_zero_counts))
        axes[0, 0].set_title('Aneurysm Size Classification Distribution (4 Categories)', fontsize=14, fontweight='bold')

        # 2. Size classification bar chart
        categories = ['Extra Small', 'Small', 'Medium', 'Large']
        counts = [size_counts.get(cat, 0) for cat in categories]
        bar_colors = [colors.get(cat, '#9E9E9E') for cat in categories]

        bars = axes[0, 1].bar(categories, counts, color=bar_colors)
        axes[0, 1].set_xlabel('Size Category')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Count by Size Category (4 Categories)', fontsize=14, fontweight='bold')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                axes[0, 1].text(bar.get_x() + bar.get_width() / 2., height,
                                f'{int(height)}', ha='center', va='bottom')

        # 3. Radius ratio distribution histogram (colored by category)
        for category, color in colors.items():
            cat_data = df[df['size_category'] == category]['radius_ratio']
            if len(cat_data) > 0:
                axes[1, 0].hist(cat_data, bins=15, alpha=0.7, label=f'{category}',
                                color=color, edgecolor='black')

        # Mark all three thresholds
        axes[1, 0].axvline(self.threshold_1, color='blue', linestyle='--', linewidth=1.5,
                           label=f'Threshold 1: {self.threshold_1}')
        axes[1, 0].axvline(self.threshold_2, color='green', linestyle='--', linewidth=1.5,
                           label=f'Threshold 2: {self.threshold_2}')
        axes[1, 0].axvline(self.threshold_3, color='red', linestyle='--', linewidth=1.5,
                           label=f'Threshold 3: {self.threshold_3}')

        axes[1, 0].set_xlabel('Radius Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Radius Ratio Distribution (Colored by Category - 4 Categories)',
                             fontsize=14, fontweight='bold')
        axes[1, 0].legend(loc='upper right', fontsize=9)
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Category statistics table
        axes[1, 1].axis('off')

        # Calculate statistics for each category
        stats_data = []
        for category in ['Extra Small', 'Small', 'Medium', 'Large']:
            cat_df = df[df['size_category'] == category]
            if len(cat_df) > 0:
                # Get the range for this category
                if category == 'Extra Small':
                    range_str = f"≤ {self.threshold_1}"
                elif category == 'Small':
                    range_str = f"≤ {self.threshold_2}"
                elif category == 'Medium':
                    range_str = f"≤ {self.threshold_3}"
                else:  # Large
                    range_str = f"> {self.threshold_3}"

                stats_data.append([
                    category,
                    range_str,
                    len(cat_df),
                    f"{cat_df['radius_ratio'].mean():.3f} ± {cat_df['radius_ratio'].std():.3f}",
                    f"{cat_df['height_ratio'].mean():.3f} ± {cat_df['height_ratio'].std():.3f}",
                    f"{cat_df['coverage'].mean():.1%}"
                ])
            else:
                stats_data.append([category, '-', 0, '-', '-', '-'])

        # Create table
        columns = ['Category', 'Range', 'Count', 'Mean Radius Ratio', 'Mean Height Ratio', 'Mean Coverage']
        table = axes[1, 1].table(cellText=stats_data, colLabels=columns,
                                 cellLoc='center', loc='center',
                                 colColours=['#f5f5f5'] * 6)
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Add color to table
        for i, category in enumerate(['Extra Small', 'Small', 'Medium', 'Large']):
            if i < len(stats_data):
                table[(i + 1, 0)].set_facecolor(colors.get(category, '#ffffff'))

        axes[1, 1].set_title('Category Statistics (4 Categories)', fontsize=14, fontweight='bold')

        plt.suptitle('Aneurysm Size Classification Statistics Report (4 Categories)', fontsize=16, y=1.02)
        plt.tight_layout()

        # Save chart
        size_stats_path = self.visualization_dir / 'size_classification_statistics_4cat.png'
        plt.savefig(size_stats_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nSize statistics chart (4 categories) saved to: {size_stats_path}")

        # Print category statistics
        print("\n" + "=" * 70)
        print("Size Classification Statistics (4 Categories)")
        print("=" * 70)
        for category in ['Extra Small', 'Small', 'Medium', 'Large']:
            cat_df = df[df['size_category'] == category]
            if len(cat_df) > 0:
                if category == 'Extra Small':
                    range_str = f"≤ {self.threshold_1}"
                elif category == 'Small':
                    range_str = f"≤ {self.threshold_2}"
                elif category == 'Medium':
                    range_str = f"≤ {self.threshold_3}"
                else:
                    range_str = f"> {self.threshold_3}"

                print(f"{category} (Range: {range_str}): {len(cat_df)} ({len(cat_df) / len(df) * 100:.1f}%)")
                print(f"  Radius Ratio: {cat_df['radius_ratio'].mean():.3f} ± {cat_df['radius_ratio'].std():.3f}")
                print(f"  Height Ratio: {cat_df['height_ratio'].mean():.3f} ± {cat_df['height_ratio'].std():.3f}")
                print(f"  Mean Coverage: {cat_df['coverage'].mean():.1%}")

    def generate_summary_report(self):
        """Generate summary report"""
        if not self.location_data:
            print("No data to generate report")
            return

        df = pd.DataFrame(self.location_data)

        print("\n" + "=" * 60)
        print("Location Information Summary Report (Minimum Enclosing Circle Method)")
        print("=" * 60)

        print(f"Total samples: {len(df)}")
        print(f"Mean height ratio: {df['height_ratio'].mean():.3f} (±{df['height_ratio'].std():.3f})")
        print(f"Mean radius ratio: {df['radius_ratio'].mean():.3f} (±{df['radius_ratio'].std():.3f})")
        print(f"Mean coverage: {df['coverage'].mean():.1%} (±{df['coverage'].std():.1%})")

        # Coverage distribution statistics
        coverage_stats = {
            'Excellent (≥95%)': len(df[df['coverage'] >= 0.95]),
            'Good (85-95%)': len(df[(df['coverage'] >= 0.85) & (df['coverage'] < 0.95)]),
            'Fair (75-85%)': len(df[(df['coverage'] >= 0.75) & (df['coverage'] < 0.85)]),
            'Poor (<75%)': len(df[df['coverage'] < 0.75])
        }

        print("\nCoverage Distribution:")
        for category, count in coverage_stats.items():
            percentage = count / len(df) * 100
            print(f"  {category}: {count} ({percentage:.1f}%)")

        # Generate statistics plots
        self.create_statistics_plots(df)

        # If classification is enabled, generate size statistics chart
        if self.enable_size_classification:
            self.generate_size_statistics_chart()

    def create_statistics_plots(self, df):
        """Create statistics plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 1. Height ratio distribution
            axes[0, 0].hist(df['height_ratio'], bins=20, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(df['height_ratio'].mean(), color='red', linestyle='--',
                               label=f'Mean: {df["height_ratio"].mean():.3f}')
            axes[0, 0].set_xlabel('Height Ratio (0=Top, 1=Bottom)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Height Ratio Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Radius ratio distribution
            axes[0, 1].hist(df['radius_ratio'], bins=20, edgecolor='black', alpha=0.7, color='orange')
            axes[0, 1].axvline(df['radius_ratio'].mean(), color='red', linestyle='--',
                               label=f'Mean: {df["radius_ratio"].mean():.3f}')

            # If classification is enabled, mark all three thresholds on the plot
            if self.enable_size_classification:
                axes[0, 1].axvline(self.threshold_1, color='blue', linestyle=':', linewidth=1.5,
                                   label=f'Threshold 1: {self.threshold_1}')
                axes[0, 1].axvline(self.threshold_2, color='green', linestyle=':', linewidth=1.5,
                                   label=f'Threshold 2: {self.threshold_2}')
                axes[0, 1].axvline(self.threshold_3, color='red', linestyle=':', linewidth=1.5,
                                   label=f'Threshold 3: {self.threshold_3}')

            axes[0, 1].set_xlabel('Radius Ratio')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Radius Ratio Distribution')
            axes[0, 1].legend(loc='upper right', fontsize=9)
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Coverage distribution
            axes[1, 0].hist(df['coverage'], bins=20, edgecolor='black', alpha=0.7, color='green')
            axes[1, 0].axvline(df['coverage'].mean(), color='red', linestyle='--',
                               label=f'Mean: {df["coverage"].mean():.1%}')
            axes[1, 0].axvline(0.95, color='blue', linestyle=':', label='95% Threshold')
            axes[1, 0].set_xlabel('Coverage')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Coverage Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 4. Height vs Radius scatter plot (color represents coverage)
            scatter = axes[1, 1].scatter(df['height_ratio'], df['radius_ratio'],
                                         c=df['coverage'], cmap='RdYlGn', alpha=0.7, s=50,
                                         vmin=0.7, vmax=1.0)
            axes[1, 1].set_xlabel('Height Ratio')
            axes[1, 1].set_ylabel('Radius Ratio')
            axes[1, 1].set_title('Height vs Radius (Color=Coverage)')
            cbar = plt.colorbar(scatter, ax=axes[1, 1])
            cbar.set_label('Coverage')
            axes[1, 1].grid(True, alpha=0.3)

            plt.suptitle('Aneurysm Location Statistics - Minimum Enclosing Circle Method', fontsize=16, y=1.02)
            plt.tight_layout()

            # Save chart
            stats_path = self.visualization_dir / 'statistics_summary.png'
            plt.savefig(stats_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"\nStatistics chart saved to: {stats_path}")

        except Exception as e:
            print(f"Failed to generate statistics chart: {e}")


def main():
    """Main function"""
    print(
        "Aneurysm Location Information Extraction Program - Minimum Enclosing Circle Method (Ensuring Complete Coverage)")
    print("=" * 60)

    # Configuration parameters
    mask_dir = "D:\\med_data\\ai\\translate\\contrast_mask"
    output_excel = "D:\\med_data\\ai\\translate\\location_contrast_size.xlsx"
    visualization_dir = "D:\\med_data\\ai\\translate\\mask"    # ===== User Configuration Options =====
    # 1. Whether to save comparison figures (visualization results)
    save_comparison_figures = True # True: Save figures, False: Generate table only

    # 2. Whether to enable size classification
    enable_size_classification = True  # True: Enable classification, False: Disable

    # 3. Size classification thresholds for 4 categories
    threshold_1 = 0.04  # Extra Small upper limit
    threshold_2 = 0.08  # Small upper limit
    threshold_3 = 0.19  # Medium upper limit (Large > threshold_3)
    # =======================

    # Create extractor
    extractor = MaskLocationExtractor(mask_dir, output_excel, visualization_dir)

    # Set classification options
    if enable_size_classification:
        extractor.enable_size_classification = True
        extractor.set_size_classification_thresholds(threshold_1, threshold_2, threshold_3)
    else:
        extractor.enable_size_classification = False
        print("Size classification disabled")

    try:
        # Process all mask files
        success = extractor.process_all_masks(save_figures=save_comparison_figures)

        if success:
            # Save to Excel
            extractor.save_to_excel()

            # Generate summary report
            extractor.generate_summary_report()

            print("\n" + "=" * 60)
            print("Processing Complete!")
            print("=" * 60)
            print(f"Location table: {output_excel}")
            if save_comparison_figures:
                print(f"Visualization results: {visualization_dir}")
            else:
                print("Comparison figures not saved (as per configuration)")

            # Display key statistics
            df = pd.DataFrame(extractor.location_data)
            if len(df) > 0:
                high_coverage = len(df[df['coverage'] >= 0.95])
                print(f"High coverage (≥95%) samples: {high_coverage}/{len(df)} ({high_coverage / len(df) * 100:.1f}%)")

                # Display low coverage samples
                low_coverage = df[df['coverage'] < 0.85]
                if len(low_coverage) > 0:
                    print(f"\nLow coverage (<85%) samples ({len(low_coverage)}):")
                    for idx, row in low_coverage.head(5).iterrows():
                        print(f"  {row['filename']}: Coverage={row['coverage']:.1%}")
                    if len(low_coverage) > 5:
                        print(f"  ... and {len(low_coverage) - 5} more")

        else:
            print("Processing failed, please check input files")

    except Exception as e:
        print(f"Program execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()