import os
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
import warnings

warnings.filterwarnings('ignore')


class MaskLocationExtractor:
    """Extract aneurysm location information from mask images (using minimum enclosing circle for complete coverage)"""

    def __init__(self, mask_dir="D:\\med_data\\ai\\translate\\all_mask",
                 output_excel="D:\\med_data\\ai\\translate\\location_all.xlsx",
                 visualization_dir="D:\\med_data\\ai\\translate\\all_mask_PNG"):
        self.mask_dir = Path(mask_dir)
        self.output_excel = Path(output_excel)
        self.visualization_dir = Path(visualization_dir)

        # Create output directory
        self.visualization_dir.mkdir(parents=True, exist_ok=True)

        # Store results
        self.location_data = []

    def find_mask_files(self):
        """Find all TIF format mask files"""
        mask_files = list(self.mask_dir.glob("*.tif"))
        print(f"Found {len(mask_files)} TIF format mask files")

        # Sort by filename
        mask_files.sort()

        # Display first 10 files
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

            # Select the largest area region (assuming largest region is the aneurysm)
            largest_region = max(regions, key=lambda r: r.area)

            # Filter out too small regions
            if largest_region.area < 10:
                return None

            return largest_region

        except Exception as e:
            print(f"Failed to find main aneurysm region: {e}")
            return None

    def calculate_minimum_enclosing_circle(self, region):
        """Calculate minimum enclosing circle (ensures complete coverage of aneurysm)"""
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
            # Backup: use bounding box calculation
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
        max_radius = min(h, w) / 2  # Maximum radius is half of the image size
        adjusted_radius = radius

        for i in range(20):  # Try at most 20 times
            if adjusted_radius >= max_radius:
                break

            # Slightly increase radius
            adjusted_radius = adjusted_radius * 1.05

            # Create new circle
            new_circle = self.create_circle_mask((h, w), center_y, center_x, adjusted_radius)

            # Calculate new coverage
            new_overlap = np.sum(np.logical_and(original_mask > 0, new_circle > 0))
            new_coverage = new_overlap / aneurysm_area if aneurysm_area > 0 else 0

            # If satisfactory coverage achieved or no further improvement, stop
            if new_coverage > 0.99 or (i > 5 and new_coverage - initial_coverage < 0.01):
                return center_y, center_x, adjusted_radius, new_coverage

        return center_y, center_x, adjusted_radius, new_coverage

    def visualize_results(self, original_mask, circle_mask, filename,
                          center_y, center_x, radius, region_bbox):
        """Visualize results and save"""
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
            axes[0, 1].plot(circle_x, circle_y, 'g-', linewidth=2, label=f'Radius={radius:.1f}px')

            # Draw bounding box
            rect = plt.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                 fill=False, edgecolor='red', linewidth=2, linestyle='--',
                                 label='Bounding Box')
            axes[0, 1].add_patch(rect)

            axes[0, 1].legend(loc='upper right')

            # 3. Generated circle
            axes[0, 2].imshow(circle_mask, cmap='gray')
            axes[0, 2].set_title(f'Minimum Enclosing Circle (Radius={radius:.1f}px)')
            axes[0, 2].axis('off')
            axes[0, 2].plot(center_x, center_y, 'go', markersize=8, linewidth=2)

            # 4. Overlap image (color)
            overlap = np.zeros((*original_mask.shape, 3), dtype=np.uint8)

            # Aneurysm region: Green
            aneurysm_mask = original_mask > 0
            overlap[aneurysm_mask, 1] = 255  # Green

            # Circle region: Red (semi-transparent)
            circle_area = circle_mask > 0
            overlap[circle_area, 0] = 128  # Red

            # Overlap region: Yellow
            overlap_area = np.logical_and(aneurysm_mask, circle_area)
            overlap[overlap_area, 0] = 255  # Red
            overlap[overlap_area, 1] = 255  # Green
            overlap[overlap_area, 2] = 0  # No blue

            axes[1, 0].imshow(overlap)
            axes[1, 0].set_title('Overlap (Green:Aneurysm, Red:Circle, Yellow:Overlap)')
            axes[1, 0].axis('off')

            # 5. Coverage analysis
            axes[1, 1].imshow(original_mask, cmap='gray')

            # Mark uncovered regions
            uncovered = np.logical_and(aneurysm_mask, ~circle_area)
            if np.any(uncovered):
                # Find contours of uncovered regions
                uncovered_uint8 = uncovered.astype(np.uint8) * 255
                contours, _ = cv2.findContours(uncovered_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) > 5:  # Only show areas larger than 5 pixels
                        contour = contour.squeeze()
                        axes[1, 1].plot(contour[:, 0], contour[:, 1], 'r-', linewidth=2, alpha=0.7)

            axes[1, 1].set_title('Uncovered Region Analysis (Red:Uncovered)')
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
            info_text += f"Bounding Box: {bbox_width:.1f} x {bbox_height:.1f}\n"
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

            plt.suptitle(f'Aneurysm Location Extraction - Minimum Enclosing Circle: {filename}', fontsize=16, y=1.02)
            plt.tight_layout()

            # Save visualization result
            output_name = filename.replace('.tif', '_analysis.png')
            output_path = self.visualization_dir / output_name
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            # Save overlap image as separate file
            overlap_name = filename.replace('.tif', '_overlap.png')
            overlap_path = self.visualization_dir / overlap_name
            cv2.imwrite(str(overlap_path), overlap)

            return coverage, uncovered_area

        except Exception as e:
            print(f"Failed to save visualization: {e}")
            return 0.0, 0

    def process_single_mask(self, mask_path):
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

        # 5. Create circle mask
        circle_mask = self.create_circle_mask((h, w), center_y, center_x, final_radius)

        # 6. Calculate normalized parameters (real ratios, no threshold constraints)
        # Y-axis ratio: top to bottom position, 0=top, 1=bottom
        height_ratio = center_y / h

        # X-axis ratio: left to right position, 0=left, 1=right
        width_ratio = center_x / w

        # Radius ratio: circle radius relative to image diagonal (real ratio, no min constraint)
        # Using image diagonal as reference for better size representation
        image_diagonal = np.sqrt(h ** 2 + w ** 2)
        radius_ratio = final_radius / image_diagonal

        # Limit to [0, 1] range (though radius_ratio should naturally be < 1)
        height_ratio = max(0.0, min(1.0, height_ratio))
        width_ratio = max(0.0, min(1.0, width_ratio))
        radius_ratio = max(0.0, min(1.0, radius_ratio))

        # 7. Visualize and save
        bbox = region.bbox
        final_coverage, uncovered_area = self.visualize_results(
            mask, circle_mask, filename,
            center_y, center_x, final_radius, bbox
        )

        # 8. Record results
        result = {
            'filename': filename.replace('.tif', ''),  # Remove extension
            'height_ratio': height_ratio,  # Y-axis ratio (0=top, 1=bottom)
            'width_ratio': width_ratio,  # X-axis ratio (0=left, 1=right)
            'radius_ratio': radius_ratio,  # Radius ratio relative to image diagonal
            'center_x': center_x,  # Actual center X coordinate (pixels)
            'center_y': center_y,  # Actual center Y coordinate (pixels)
            'pixel_radius': final_radius,  # Actual radius in pixels
            'image_width': w,  # Image width in pixels
            'image_height': h,  # Image height in pixels
            'initial_radius': radius,
            'coverage': final_coverage,
            'uncovered_area': uncovered_area,
            'aneurysm_area': region.area,
            'bbox_width': bbox[3] - bbox[1],
            'bbox_height': bbox[2] - bbox[0]
        }

        # Print all information in English
        print(f"  Aneurysm Center Coordinates: ({center_x:.1f}, {center_y:.1f}) pixels")
        print(f"  Actual Radius: {final_radius:.1f} pixels")
        print(f"  Image Size: {w} x {h} pixels")
        print(f"  X-axis ratio (0=left, 1=right): {width_ratio:.4f}")
        print(f"  Y-axis ratio (0=top, 1=bottom): {height_ratio:.4f}")
        print(f"  Radius ratio (radius / image_diagonal): {radius_ratio:.4f}")
        print(f"  Image Diagonal: {image_diagonal:.1f} pixels")
        print(f"  Actual coverage: {final_coverage:.1%}")
        print(f"  Uncovered area: {uncovered_area} pixels")

        if final_coverage < 0.95:
            print(f"  ⚠️ Warning: Low coverage ({final_coverage:.1%})")

        return result

    def process_all_masks(self):
        """Process all mask files"""
        print("Starting aneurysm location extraction (Minimum Enclosing Circle Method)...")
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
            result = self.process_single_mask(mask_path)

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
            print(f"Coverage < 95%: {low_coverage_count} ({low_coverage_count / successful_count * 100:.1f}%)")

        return successful_count > 0

    def save_to_excel(self):
        """Save results to Excel file"""
        if not self.location_data:
            print("Error: No data to save")
            return False

        try:
            # Create DataFrame
            df = pd.DataFrame(self.location_data)

            # Reorder columns, include width_ratio, height_ratio, and actual values
            output_df = df[['filename', 'width_ratio', 'height_ratio', 'radius_ratio',
                            'center_x', 'center_y', 'pixel_radius',
                            'image_width', 'image_height', 'coverage']].copy()

            # Rename columns for clarity
            output_df.columns = ['filename', 'x_ratio(0=left,1=right)', 'y_ratio(0=top,1=bottom)',
                                 'radius_ratio', 'center_x(pixels)', 'center_y(pixels)',
                                 'radius(pixels)', 'image_width', 'image_height', 'coverage']

            # Save to Excel
            output_df.to_excel(self.output_excel, index=False)

            print(f"\nLocation information saved to: {self.output_excel}")
            print(f"Total records: {len(output_df)}")

            # Display first few rows
            print("\nFirst 10 records:")
            print(output_df.head(10))

            # Save complete data to CSV (including more information)
            full_csv_path = self.output_excel.with_suffix('.csv')
            df.to_csv(full_csv_path, index=False, encoding='utf-8-sig')
            print(f"Complete data saved to: {full_csv_path}")

            return True

        except Exception as e:
            print(f"Failed to save Excel file: {e}")
            return False

    def generate_summary_report(self):
        """Generate summary report"""
        if not self.location_data:
            print("No data to generate report")
            return

        df = pd.DataFrame(self.location_data)

        print("\n" + "=" * 60)
        print("Location Information Statistics Report (Minimum Enclosing Circle Method)")
        print("=" * 60)

        print(f"Total samples: {len(df)}")
        print(
            f"Average X-axis ratio (0=left, 1=right): {df['width_ratio'].mean():.4f} (±{df['width_ratio'].std():.4f})")
        print(
            f"Average Y-axis ratio (0=top, 1=bottom): {df['height_ratio'].mean():.4f} (±{df['height_ratio'].std():.4f})")
        print(f"Average radius ratio: {df['radius_ratio'].mean():.4f} (±{df['radius_ratio'].std():.4f})")
        print(f"Average coverage: {df['coverage'].mean():.1%} (±{df['coverage'].std():.1%})")
        print(f"Average center position: ({df['center_x'].mean():.1f}, {df['center_y'].mean():.1f}) pixels")
        print(f"Average radius: {df['pixel_radius'].mean():.1f} pixels")

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

        # Generate visualization charts
        self.create_statistics_plots(df)

    def create_statistics_plots(self, df):
        """Create statistics charts"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))

            # 1. X-axis ratio distribution (left to right)
            axes[0, 0].hist(df['width_ratio'], bins=20, edgecolor='black', alpha=0.7, color='blue')
            axes[0, 0].axvline(df['width_ratio'].mean(), color='red', linestyle='--',
                               label=f'Mean: {df["width_ratio"].mean():.4f}')
            axes[0, 0].set_xlabel('X-axis Ratio (0=left, 1=right)')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('X-axis Ratio Distribution')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 2. Y-axis ratio distribution (top to bottom)
            axes[0, 1].hist(df['height_ratio'], bins=20, edgecolor='black', alpha=0.7, color='green')
            axes[0, 1].axvline(df['height_ratio'].mean(), color='red', linestyle='--',
                               label=f'Mean: {df["height_ratio"].mean():.4f}')
            axes[0, 1].set_xlabel('Y-axis Ratio (0=top, 1=bottom)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Y-axis Ratio Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 3. Radius ratio distribution (real values, no threshold)
            axes[0, 2].hist(df['radius_ratio'], bins=20, edgecolor='black', alpha=0.7, color='orange')
            axes[0, 2].axvline(df['radius_ratio'].mean(), color='red', linestyle='--',
                               label=f'Mean: {df["radius_ratio"].mean():.4f}')
            axes[0, 2].set_xlabel('Radius Ratio (radius / image_diagonal)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Radius Ratio Distribution (Real Values)')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

            # 4. Coverage distribution
            axes[1, 0].hist(df['coverage'], bins=20, edgecolor='black', alpha=0.7, color='green')
            axes[1, 0].axvline(df['coverage'].mean(), color='red', linestyle='--',
                               label=f'Mean: {df["coverage"].mean():.1%}')
            axes[1, 0].axvline(0.95, color='blue', linestyle=':', label='95% Threshold')
            axes[1, 0].set_xlabel('Coverage')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Coverage Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # 5. X-axis vs Y-axis scatter plot (color represents coverage)
            scatter1 = axes[1, 1].scatter(df['width_ratio'], df['height_ratio'],
                                          c=df['coverage'], cmap='RdYlGn', alpha=0.7, s=50,
                                          vmin=0.7, vmax=1.0)
            axes[1, 1].set_xlabel('X-axis Ratio (0=left, 1=right)')
            axes[1, 1].set_ylabel('Y-axis Ratio (0=top, 1=bottom)')
            axes[1, 1].set_title('X-axis vs Y-axis (Color=Coverage)')
            cbar1 = plt.colorbar(scatter1, ax=axes[1, 1])
            cbar1.set_label('Coverage')
            axes[1, 1].grid(True, alpha=0.3)

            # 6. Radius ratio vs Coverage scatter plot
            scatter2 = axes[1, 2].scatter(df['radius_ratio'], df['coverage'],
                                          c=df['height_ratio'], cmap='viridis', alpha=0.7, s=50)
            axes[1, 2].set_xlabel('Radius Ratio')
            axes[1, 2].set_ylabel('Coverage')
            axes[1, 2].set_title('Radius Ratio vs Coverage (Color=Y-axis Ratio)')
            cbar2 = plt.colorbar(scatter2, ax=axes[1, 2])
            cbar2.set_label('Y-axis Ratio')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axhline(0.95, color='blue', linestyle=':', alpha=0.5)

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
    print("Aneurysm Location Information Extraction - Minimum Enclosing Circle Method (Ensuring Complete Coverage)")
    print("=" * 60)

    # Configuration parameters
    mask_dir = "D:\\med_data\\multi\\test2_tif"
    output_excel = "D:\\med_data\\multi\\location_test.xlsx"
    visualization_dir = "D:\\med_data\\multi\\all_mask_PNG"

    # Create extractor
    extractor = MaskLocationExtractor(mask_dir, output_excel, visualization_dir)

    try:
        # Process all mask files
        success = extractor.process_all_masks()

        if success:
            # Save to Excel
            extractor.save_to_excel()

            # Generate summary report
            extractor.generate_summary_report()

            print("\n" + "=" * 60)
            print("Processing Complete!")
            print("=" * 60)
            print(f"Location table: {output_excel}")
            print(f"Visualization results: {visualization_dir}")

            # Display key statistics
            df = pd.DataFrame(extractor.location_data)
            if len(df) > 0:
                high_coverage = len(df[df['coverage'] >= 0.95])
                print(f"High coverage (≥95%) samples: {high_coverage}/{len(df)} ({high_coverage / len(df) * 100:.1f}%)")

                # Display low coverage samples
                low_coverage = df[df['coverage'] < 0.85]
                if len(low_coverage) > 0:
                    print(f"\nLow coverage (<85%) samples ({len(low_coverage)} total):")
                    for idx, row in low_coverage.head(5).iterrows():
                        print(f"  {row['filename']}: coverage={row['coverage']:.1%}")
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