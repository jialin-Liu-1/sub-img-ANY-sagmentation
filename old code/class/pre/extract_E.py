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


def normalize_image(image):
    """Normalize image to 0-255 range"""
    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255
        image = image.astype(np.uint8)
    return image


def calculate_distributions():
    # Input paths
    dicom_dir = r"D:\ai1\processed_data\D"  # DICOM medical image directory
    mask_dir = r"D:\ai1\mask2"  # TIF mask image directory

    # Output path
    output_dir = r"D:\ai\Distribution"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize statistical variables
    pixel_values = []  # All DICOM image pixel values
    aneurysm_areas = []  # Aneurysm area ratio
    aneurysm_centers = []  # Aneurysm center coordinates
    aneurysm_mean_pixel_values = []  # Mean pixel values for each aneurysm
    individual_aneurysm_pixels = []  # Individual aneurysm pixel values for distribution

    # Get DICOM file list (files without extension)
    dicom_files = [f for f in os.listdir(dicom_dir)
                   if os.path.isfile(os.path.join(dicom_dir, f))]

    print(f"Found {len(dicom_files)} DICOM files")
    print("Starting distribution calculation...")

    processed_count = 0

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

            # 1. Collect DICOM image pixel values
            pixel_values.extend(dicom_image.flatten())

            # 2. Calculate aneurysm area ratio
            total_pixels = mask_binary.size
            aneurysm_pixels = np.sum(mask_binary)
            aneurysm_ratio = aneurysm_pixels / total_pixels
            aneurysm_areas.append(aneurysm_ratio)

            # 3. Calculate aneurysm center point and mean pixel value
            if aneurysm_pixels > 0:
                # Find coordinates of all non-zero pixels
                y_coords, x_coords = np.where(mask_binary > 0)
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                aneurysm_centers.append((center_x, center_y))

                # 4. Calculate mean pixel value for this specific aneurysm
                # Multiply DICOM image with mask to extract aneurysm region
                aneurysm_region = dicom_image * mask_binary
                # Only take non-zero values (aneurysm region)
                aneurysm_pixels_values = aneurysm_region[aneurysm_region > 0]

                if len(aneurysm_pixels_values) > 0:
                    # Calculate mean pixel value for this aneurysm
                    aneurysm_mean = np.mean(aneurysm_pixels_values)
                    aneurysm_mean_pixel_values.append(aneurysm_mean)

                    # Also collect individual pixel values for distribution plot
                    # Sample up to 100 pixels per aneurysm to avoid memory issues
                    if len(aneurysm_pixels_values) > 100:
                        sampled_pixels = np.random.choice(aneurysm_pixels_values, 100, replace=False)
                        individual_aneurysm_pixels.extend(sampled_pixels)
                    else:
                        individual_aneurysm_pixels.extend(aneurysm_pixels_values)

            processed_count += 1
            print(f"Processed {processed_count}/{len(dicom_files)}: {dicom_file} "
                  f"(Aneurysm ratio: {aneurysm_ratio:.4f}, "
                  f"Mean pixel: {aneurysm_mean_pixel_values[-1] if aneurysm_mean_pixel_values else 'N/A':.2f})")

            # Generate intermediate results every 100 files (optional)
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files, generating intermediate distribution plots...")
                generate_distribution_plots(pixel_values, aneurysm_areas,
                                            aneurysm_centers, aneurysm_mean_pixel_values,
                                            individual_aneurysm_pixels,
                                            output_dir, suffix=f"_intermediate_{processed_count}")

        except Exception as e:
            print(f"Error processing file {dicom_file}: {e}")
            continue

    print(f"\nProcessing completed! Successfully processed {processed_count} image sets")

    # Generate final distribution plots
    generate_distribution_plots(pixel_values, aneurysm_areas,
                                aneurysm_centers, aneurysm_mean_pixel_values,
                                individual_aneurysm_pixels,
                                output_dir)

    # Save statistical information to text files
    save_statistics(pixel_values, aneurysm_areas, aneurysm_centers,
                    aneurysm_mean_pixel_values, individual_aneurysm_pixels,
                    output_dir, processed_count)


def generate_distribution_plots(pixel_values, aneurysm_areas, aneurysm_centers,
                                aneurysm_mean_pixel_values, individual_aneurysm_pixels,
                                output_dir, suffix=""):
    """Generate and save all distribution plots"""

    # Set font for better visualization
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['axes.unicode_minus'] = False

    # 1. DICOM image pixel value distribution plot
    if pixel_values:
        plt.figure(figsize=(12, 8))
        plt.hist(pixel_values, bins=100, alpha=0.7, color='blue', edgecolor='black')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('DICOM Image Pixel Value Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'pixel_value_distribution{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: pixel_value_distribution{suffix}.png")

    # 2. Aneurysm area ratio distribution plot
    if aneurysm_areas:
        plt.figure(figsize=(12, 8))
        plt.hist(aneurysm_areas, bins=50, alpha=0.7, color='green', edgecolor='black')
        plt.xlabel('Aneurysm Area Ratio')
        plt.ylabel('Frequency')
        plt.title('Aneurysm Area Ratio Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'aneurysm_area_ratio_distribution{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: aneurysm_area_ratio_distribution{suffix}.png")

    # 3. Aneurysm location distribution plot (center points overlay)
    if aneurysm_centers:
        # Create heatmap with fixed axis range 0-512
        fig, ax = plt.subplots(figsize=(10, 10))

        # Extract x and y coordinates of all center points
        x_coords = [center[0] for center in aneurysm_centers]
        y_coords = [center[1] for center in aneurysm_centers]

        # Create 2D histogram (heatmap) with fixed range
        hb = ax.hist2d(x_coords, y_coords, bins=50, cmap='hot', range=[[0, 512], [0, 512]])
        plt.colorbar(hb[3], ax=ax, label='Frequency')

        ax.set_xlabel('X Coordinate (pixels)')
        ax.set_ylabel('Y Coordinate (pixels)')
        ax.set_title('Aneurysm Location Heatmap')
        ax.set_xlim(0, 512)
        ax.set_ylim(0, 512)
        ax.grid(True, alpha=0.3)

        plt.savefig(os.path.join(output_dir, f'aneurysm_location_distribution{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: aneurysm_location_distribution{suffix}.png")

        # Also save a scatter plot version with fixed range
        plt.figure(figsize=(10, 10))
        plt.scatter(x_coords, y_coords, alpha=0.6, s=20, c='red')
        plt.xlabel('X Coordinate (pixels)')
        plt.ylabel('Y Coordinate (pixels)')
        plt.title('Aneurysm Center Points Distribution')
        plt.xlim(0, 512)
        plt.ylim(0, 512)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'aneurysm_center_scatter{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: aneurysm_center_scatter{suffix}.png")

    # 4. Aneurysm mean pixel value distribution plot
    if aneurysm_mean_pixel_values:
        plt.figure(figsize=(12, 8))
        plt.hist(aneurysm_mean_pixel_values, bins=50, alpha=0.7, color='red', edgecolor='black')
        plt.xlabel('Mean Pixel Value per Aneurysm')
        plt.ylabel('Frequency')
        plt.title('Aneurysm Mean Pixel Value Distribution (Per Aneurysm)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'aneurysm_mean_pixel_distribution{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: aneurysm_mean_pixel_distribution{suffix}.png")

    # 5. Individual aneurysm pixel value distribution plot
    if individual_aneurysm_pixels:
        plt.figure(figsize=(12, 8))
        plt.hist(individual_aneurysm_pixels, bins=100, alpha=0.7, color='purple', edgecolor='black')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.title('Individual Aneurysm Pixel Value Distribution')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, f'aneurysm_individual_pixel_distribution{suffix}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: aneurysm_individual_pixel_distribution{suffix}.png")


def save_statistics(pixel_values, aneurysm_areas, aneurysm_centers,
                    aneurysm_mean_pixel_values, individual_aneurysm_pixels,
                    output_dir, processed_count):
    """Save statistical information to text files"""

    stats = {
        "processed_images": processed_count,
        "total_pixels_analyzed": len(pixel_values),
        "images_with_aneurysm": len(aneurysm_areas),
        "aneurysm_centers_found": len(aneurysm_centers),
        "aneurysms_with_pixel_data": len(aneurysm_mean_pixel_values),
        "individual_aneurysm_pixels_analyzed": len(individual_aneurysm_pixels)
    }

    if pixel_values:
        stats["dicom_pixel_stats"] = {
            "mean": float(np.mean(pixel_values)),
            "std": float(np.std(pixel_values)),
            "min": float(np.min(pixel_values)),
            "max": float(np.max(pixel_values)),
            "median": float(np.median(pixel_values))
        }

    if aneurysm_areas:
        stats["aneurysm_area_stats"] = {
            "mean_ratio": float(np.mean(aneurysm_areas)),
            "std_ratio": float(np.std(aneurysm_areas)),
            "min_ratio": float(np.min(aneurysm_areas)),
            "max_ratio": float(np.max(aneurysm_areas)),
            "median_ratio": float(np.median(aneurysm_areas))
        }

    if aneurysm_mean_pixel_values:
        stats["aneurysm_mean_pixel_stats"] = {
            "mean": float(np.mean(aneurysm_mean_pixel_values)),
            "std": float(np.std(aneurysm_mean_pixel_values)),
            "min": float(np.min(aneurysm_mean_pixel_values)),
            "max": float(np.max(aneurysm_mean_pixel_values)),
            "median": float(np.median(aneurysm_mean_pixel_values))
        }

    if individual_aneurysm_pixels:
        stats["aneurysm_individual_pixel_stats"] = {
            "mean": float(np.mean(individual_aneurysm_pixels)),
            "std": float(np.std(individual_aneurysm_pixels)),
            "min": float(np.min(individual_aneurysm_pixels)),
            "max": float(np.max(individual_aneurysm_pixels)),
            "median": float(np.median(individual_aneurysm_pixels))
        }

    # Save as JSON file
    with open(os.path.join(output_dir, 'statistics.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)
    print("Saved: statistics.json")

    # Save as text file
    with open(os.path.join(output_dir, 'statistics.txt'), 'w', encoding='utf-8') as f:
        f.write("DICOM Image Statistical Analysis Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Processed Images: {stats['processed_images']}\n")
        f.write(f"Total Pixels Analyzed: {stats['total_pixels_analyzed']}\n")
        f.write(f"Images with Aneurysm: {stats['images_with_aneurysm']}\n")
        f.write(f"Aneurysm Centers Found: {stats['aneurysm_centers_found']}\n")
        f.write(f"Aneurysms with Pixel Data: {stats['aneurysms_with_pixel_data']}\n")
        f.write(f"Individual Aneurysm Pixels Analyzed: {stats['individual_aneurysm_pixels_analyzed']}\n\n")

        if 'dicom_pixel_stats' in stats:
            f.write("DICOM Image Pixel Statistics:\n")
            for key, value in stats['dicom_pixel_stats'].items():
                f.write(f"  {key}: {value:.2f}\n")
            f.write("\n")

        if 'aneurysm_area_stats' in stats:
            f.write("Aneurysm Area Ratio Statistics:\n")
            for key, value in stats['aneurysm_area_stats'].items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")

        if 'aneurysm_mean_pixel_stats' in stats:
            f.write("Aneurysm Mean Pixel Value Statistics (per aneurysm):\n")
            for key, value in stats['aneurysm_mean_pixel_stats'].items():
                f.write(f"  {key}: {value:.2f}\n")
            f.write("\n")

        if 'aneurysm_individual_pixel_stats' in stats:
            f.write("Individual Aneurysm Pixel Statistics:\n")
            for key, value in stats['aneurysm_individual_pixel_stats'].items():
                f.write(f"  {key}: {value:.2f}\n")
    print("Saved: statistics.txt")


if __name__ == "__main__":
    calculate_distributions()