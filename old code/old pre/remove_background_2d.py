import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt


def process_dicom_remove_background(input_dir, output_dir_D, output_dir_M, show_comparison=True):
    """
    Batch process DICOM images to remove background

    Parameters:
    input_dir: Input directory
    output_dir_D: DICOM output directory
    output_dir_M: PNG output directory
    show_comparison: Whether to display comparison images
    """

    # Internal function: Remove background from a single DICOM image
    def remove_single_image_background(image_data, max_search_distance=100, extend_pixels=3):
        """Remove background from a single image"""
        height, width = image_data.shape
        min_val = np.min(image_data)

        # Initialize mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Process each column
        for j in range(width):
            col = image_data[:, j]

            # Search from top to bottom
            for i in range(min(max_search_distance, height)):
                if col[i] != min_val:
                    # Found non-minimum value, stop and set mask, extend background
                    extend_position = min(i + extend_pixels, height - 1)
                    mask[:extend_position, j] = 1
                    break
            else:
                # If the entire column is the minimum value, set the whole column
                mask[:, j] = 1

            # Search from bottom to top
            for i in range(min(max_search_distance, height)):
                if col[height - 1 - i] != min_val:
                    # Found non-minimum value, stop and set mask, extend background
                    extend_position = max(height - i - extend_pixels, 0)
                    mask[extend_position:, j] = 1
                    break
            else:
                # If the entire column is the minimum value, set the whole column
                mask[:, j] = 1

        # Process image: Set background regions to maximum value
        processed_image = image_data.copy()
        max_val = np.max(processed_image)
        processed_image[mask == 1] = max_val

        return processed_image, mask

    os.makedirs(output_dir_D, exist_ok=True)
    os.makedirs(output_dir_M, exist_ok=True)

    # Get all DICOM files
    dicom_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))]

    processed_count = 0

    for filename in dicom_files:
        try:
            file_path = os.path.join(input_dir, filename)

            # Read DICOM file
            ds = pydicom.dcmread(file_path)

            # Get pixel data
            image_data = ds.pixel_array

            print(f"Processing image: {filename}, Shape: {image_data.shape}")

            # Use internal function to remove background
            processed_image, mask = remove_single_image_background(image_data)

            # Save processed DICOM file (keep original filename)
            output_dicom_path = os.path.join(output_dir_D, filename)

            # Update DICOM file's pixel data
            ds.PixelData = processed_image.tobytes()
            ds.Rows, ds.Columns = processed_image.shape

            # Save DICOM file
            pydicom.dcmwrite(output_dicom_path, ds)

            # Save PNG file
            output_png_path = os.path.join(output_dir_M, f"{filename}.png")

            # Normalize image data for display
            if np.max(processed_image) > np.min(processed_image):
                normalized_image = (processed_image - np.min(processed_image)) / (
                        np.max(processed_image) - np.min(processed_image))
                normalized_image = (normalized_image * 255).astype(np.uint8)
            else:
                normalized_image = processed_image.astype(np.uint8)

            # Use PIL to save PNG
            png_image = Image.fromarray(normalized_image)
            png_image.save(output_png_path)

            # Display before-and-after comparison
            if show_comparison and processed_count < 3:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))

                # Original image
                axes[0].imshow(image_data, cmap='gray')
                axes[0].set_title('Original Image')
                axes[0].axis('off')

                # Mask
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Background Mask')
                axes[1].axis('off')

                # Processed image
                axes[2].imshow(processed_image, cmap='gray')
                axes[2].set_title('Processed Image')
                axes[2].axis('off')

                plt.tight_layout()
                plt.show()

            processed_count += 1
            print(f"Processed: {filename} ({processed_count}/{len(dicom_files)})")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue


def main():
    input_directory = r"D:\med_data\multi\preprocess\MEAN\DSA_resized"
    output_D = r"D:\med_data\multi\preprocess\MEAN\DICOM_back"
    output_M = r"D:\med_data\multi\preprocess\MEAN\DICOM_back\PNG"

    print("Starting DICOM image processing...")
    print(f"Input directory: {input_directory}")
    print(f"Output DICOM directory: {output_D}")
    print(f"Output PNG directory: {output_M}")

    process_dicom_remove_background(input_directory, output_D, output_M, show_comparison=True)

    print("Processing completed!")


if __name__ == "__main__":
    main()