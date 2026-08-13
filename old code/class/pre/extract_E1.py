import os
import pydicom
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import json


def read_dicom_file(filepath):
    """Read DICOM file without extension"""
    try:
        dicom_data = pydicom.dcmread(filepath)
        image_array = dicom_data.pixel_array
        return image_array
    except Exception as e:
        print(f"Failed to read DICOM file {filepath}: {e}")
        return None


def calculate_distributions():
    # Input paths
    dicom_dir = r"D:\med_data\ai\siri\train_0"  # DICOM medical image directory
    mask_dir = r"D:\med_data\ai\siri\train_1"  # TIF mask image directory

    # Output path
    output_dir = r"D:\med_data\ai\siri\lo"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize statistical variables for accumulation
    all_pixel_values = []  # All DICOM image pixel values for histogram
    aneurysm_area_ratios = []  # Aneurysm area ratios
    aneurysm_centers = []  # Aneurysm center coordinates
    aneurysm_mean_values = []  # Mean pixel values for each aneurysm
    aneurysm_individual_pixels = []  # Individual aneurysm pixel values

    # Variables for sampling to avoid memory issues
    max_pixel_samples = 1000000  # Maximum number of pixel samples for histogram
    max_individual_samples = 500000  # Maximum number of individual aneurysm pixel samples

    # Get DICOM file list (files without extension)
    dicom_files = [f for f in os.listdir(dicom_dir)
                   if os.path.isfile(os.path.join(dicom_dir, f))]

    print(f"Found {len(dicom_files)} DICOM files")
    print("Starting distribution calculation...")
    print("Processing files (no intermediate images will be generated)...")

    processed_count = 0
    files_with_aneurysm = 0

    for dicom_file in dicom_files:
        try:
            # Build file paths
            dicom_path = os.path.join(dicom_dir, dicom_file)
            mask_path = os.path.join(mask_dir, dicom_file + ".tif")

            # Check if mask file exists
            if not os.path.exists(mask_path):
                print(f"Warning: Corresponding mask file not found {mask_path}")
                continue

            # Read DICOM file
            dicom_image = read_dicom_file(dicom_path)
            if dicom_image is None:
                continue

            # Read mask file
            mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_image is None:
                print(f"Failed to read mask file: {mask_path}")
                continue

            # Ensure mask is binary image
            mask_binary = (mask_image > 0).astype(np.uint8)

            # 1. Collect DICOM image pixel values (with sampling to avoid memory issues)
            if len(all_pixel_values) < max_pixel_samples:
                flat_pixels = dicom_image.flatten()
                if len(all_pixel_values) + len(flat_pixels) > max_pixel_samples:
                    # Sample remaining pixels
                    remaining = max_pixel_samples - len(all_pixel_values)
                    if remaining > 0:
                        sampled_pixels = np.random.choice(flat_pixels, remaining, replace=False)
                        all_pixel_values.extend(sampled_pixels)
                else:
                    all_pixel_values.extend(flat_pixels)

            # 2. Calculate aneurysm area ratio
            total_pixels = mask_binary.size
            aneurysm_pixels = np.sum(mask_binary)
            aneurysm_ratio = aneurysm_pixels / total_pixels
            aneurysm_area_ratios.append(aneurysm_ratio)

            # 3. Process aneurysm data if present
            if aneurysm_pixels > 0:
                files_with_aneurysm += 1

                # Calculate aneurysm center point
                y_coords, x_coords = np.where(mask_binary > 0)
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                aneurysm_centers.append((center_x, center_y))

                # Calculate mean pixel value for this specific aneurysm
                aneurysm_region = dicom_image * mask_binary
                aneurysm_pixels_values = aneurysm_region[aneurysm_region > 0]

                if len(aneurysm_pixels_values) > 0:
                    # Store mean value for this aneurysm
                    aneurysm_mean = np.mean(aneurysm_pixels_values)
                    aneurysm_mean_values.append(aneurysm_mean)

                    # Collect individual pixel values with sampling
                    if len(aneurysm_individual_pixels) < max_individual_samples:
                        if len(aneurysm_pixels_values) > 100:
                            # Sample up to 100 pixels per aneurysm
                            sampled_pixels = np.random.choice(aneurysm_pixels_values,
                                                              min(100, len(aneurysm_pixels_values)),
                                                              replace=False)
                            aneurysm_individual_pixels.extend(sampled_pixels)
                        else:
                            aneurysm_individual_pixels.extend(aneurysm_pixels_values)

            processed_count += 1

            # Progress update every 50 files
            if processed_count % 50 == 0:
                print(f"Processed {processed_count}/{len(dicom_files)} files... "
                      f"(Aneurysm cases: {files_with_aneurysm})")

        except Exception as e:
            print(f"Error processing file {dicom_file}: {e}")
            continue

    print(f"\nProcessing completed!")
    print(f"Successfully processed {processed_count} image sets")
    print(f"Images with aneurysms: {files_with_aneurysm}")
    print(f"Total aneurysm centers found: {len(aneurysm_centers)}")
    print(f"Generating final distribution plots...")

    # Generate final distribution plots
    generate_final_distribution_plots(all_pixel_values, aneurysm_area_ratios,
                                      aneurysm_centers, aneurysm_mean_values,
                                      aneurysm_individual_pixels, output_dir)

    # Save statistical information to text files
    save_statistics(all_pixel_values, aneurysm_area_ratios, aneurysm_centers,
                    aneurysm_mean_values, aneurysm_individual_pixels,
                    output_dir, processed_count, files_with_aneurysm)


def generate_final_distribution_plots(pixel_values, aneurysm_areas, aneurysm_centers,
                                      aneurysm_mean_values, individual_pixels, output_dir):
    """Generate and save final distribution plots"""

    print("Creating distribution plots...")

    # Set font for better visualization
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. DICOM image pixel value distribution plot
    if pixel_values:
        plt.figure(figsize=(12, 8))
        plt.hist(pixel_values, bins=100, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title(f'DICOM Image Pixel Value Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pixel_value_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: pixel_value_distribution.png")

    # 2. Aneurysm area ratio distribution plot
    if aneurysm_areas:
        plt.figure(figsize=(12, 8))
        plt.hist(aneurysm_areas, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Aneurysm Area Ratio')
        plt.ylabel('Frequency')
        plt.title(f'Aneurysm Area Ratio Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aneurysm_area_ratio_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: aneurysm_area_ratio_distribution.png")

    # 3. Aneurysm location distribution plot (center points overlay)
    if aneurysm_centers:
        # Create heatmap with fixed axis range 0-512
        fig, ax = plt.subplots(figsize=(10, 10))

        x_coords = [center[0] for center in aneurysm_centers]
        y_coords = [center[1] for center in aneurysm_centers]

        # Create 2D histogram (heatmap) with fixed range
        hb = ax.hist2d(x_coords, y_coords, bins=50, cmap='hot', range=[[0, 512], [0, 512]])
        plt.colorbar(hb[3], ax=ax, label='Frequency')

        ax.set_xlabel('X Coordinate (pixels)')
        ax.set_ylabel('Y Coordinate (pixels)')
        ax.set_title(f'Aneurysm Location Heatmap')
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aneurysm_location_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: aneurysm_location_distribution.png")

        # Scatter plot version
        plt.figure(figsize=(10, 10))
        plt.scatter(x_coords, y_coords, alpha=0.6, s=20, c='red')
        plt.xlabel('X Coordinate (pixels)')
        plt.ylabel('Y Coordinate (pixels)')
        plt.title(f'Aneurysm Center Points Distribution')
        plt.xlim(0, 512)
        plt.ylim(0, 512)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aneurysm_center_scatter.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: aneurysm_center_scatter.png")

    # 4. Aneurysm mean pixel value distribution plot
    if aneurysm_mean_values:
        plt.figure(figsize=(12, 8))
        plt.hist(aneurysm_mean_values, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Mean Pixel Value per Aneurysm')
        plt.ylabel('Frequency')
        plt.title(f'Aneurysm Mean Pixel Value Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aneurysm_mean_pixel_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: aneurysm_mean_pixel_distribution.png")

    # 5. Individual aneurysm pixel value distribution plot
    if individual_pixels:
        plt.figure(figsize=(12, 8))
        plt.hist(individual_pixels, bins=100, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title(f'Individual Aneurysm Pixel Value Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'aneurysm_individual_pixel_distribution.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: aneurysm_individual_pixel_distribution.png")

    print("All distribution plots have been generated successfully!")


def save_statistics(pixel_values, aneurysm_areas, aneurysm_centers,
                    aneurysm_mean_values, individual_pixels, output_dir,
                    processed_count, files_with_aneurysm):
    """Save statistical information to text files"""

    print("\nGenerating statistical reports...")

    stats = {
        "processing_summary": {
            "processed_images": processed_count,
            "images_with_aneurysm": files_with_aneurysm,
            "aneurysm_centers_found": len(aneurysm_centers),
            "aneurysms_with_pixel_data": len(aneurysm_mean_values),
            "total_pixel_samples": len(pixel_values),
            "individual_aneurysm_pixel_samples": len(individual_pixels)
        }
    }

    if pixel_values:
        stats["dicom_pixel_stats"] = {
            "mean": float(np.mean(pixel_values)),
            "std": float(np.std(pixel_values)),
            "min": float(np.min(pixel_values)),
            "max": float(np.max(pixel_values)),
            "median": float(np.median(pixel_values)),
            "q1": float(np.percentile(pixel_values, 25)),
            "q3": float(np.percentile(pixel_values, 75))
        }

    if aneurysm_areas:
        stats["aneurysm_area_stats"] = {
            "mean_ratio": float(np.mean(aneurysm_areas)),
            "std_ratio": float(np.std(aneurysm_areas)),
            "min_ratio": float(np.min(aneurysm_areas)),
            "max_ratio": float(np.max(aneurysm_areas)),
            "median_ratio": float(np.median(aneurysm_areas))
        }

    if aneurysm_mean_values:
        stats["aneurysm_mean_pixel_stats"] = {
            "mean": float(np.mean(aneurysm_mean_values)),
            "std": float(np.std(aneurysm_mean_values)),
            "min": float(np.min(aneurysm_mean_values)),
            "max": float(np.max(aneurysm_mean_values)),
            "median": float(np.median(aneurysm_mean_values))
        }

    if individual_pixels:
        stats["aneurysm_individual_pixel_stats"] = {
            "mean": float(np.mean(individual_pixels)),
            "std": float(np.std(individual_pixels)),
            "min": float(np.min(individual_pixels)),
            "max": float(np.max(individual_pixels)),
            "median": float(np.median(individual_pixels))
        }

    # Save as JSON file
    with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print("✓ Saved: statistics.json")

    # Save as text file
    with open(os.path.join(output_dir, 'statistics.txt'), 'w', encoding='utf-8') as f:
        f.write("DICOM IMAGE STATISTICAL ANALYSIS REPORT\n")
        f.write("=" * 55 + "\n\n")

        summary = stats["processing_summary"]
        f.write("PROCESSING SUMMARY:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Processed Images:          {summary['processed_images']:>8}\n")
        f.write(f"Images with Aneurysm:      {summary['images_with_aneurysm']:>8}\n")
        f.write(f"Aneurysm Centers Found:    {summary['aneurysm_centers_found']:>8}\n")
        f.write(f"Aneurysms with Pixel Data: {summary['aneurysms_with_pixel_data']:>8}\n")
        f.write(f"Total Pixel Samples:       {summary['total_pixel_samples']:>8,}\n")
        f.write(f"Individual Aneurysm Pixels:{summary['individual_aneurysm_pixel_samples']:>8,}\n\n")

        # Add all statistical sections
        sections = [
            ("DICOM IMAGE PIXEL STATISTICS", "dicom_pixel_stats"),
            ("ANEURYSM AREA RATIO STATISTICS", "aneurysm_area_stats"),
            ("ANEURYSM MEAN PIXEL STATISTICS", "aneurysm_mean_pixel_stats"),
            ("INDIVIDUAL ANEURYSM PIXEL STATISTICS", "aneurysm_individual_pixel_stats")
        ]

        for section_name, section_key in sections:
            if section_key in stats:
                f.write(f"{section_name}:\n")
                f.write("-" * 40 + "\n")
                for key, value in stats[section_key].items():
                    if 'ratio' in key:
                        f.write(f"  {key:>12}: {value:>12.6f}\n")
                    else:
                        f.write(f"  {key:>12}: {value:>12.2f}\n")
                f.write("\n")

    print("✓ Saved: statistics.txt")
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    calculate_distributions()