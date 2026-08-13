import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt


def process_dicom_remove_background(input_dir, output_dir_D, output_dir_M, output_dir_Mask, output_dir_Original,
                                    show_comparison=True):
    """
    Batch process DICOM images to remove background

    Parameters:
    input_dir: Input directory
    output_dir_D: DICOM output directory
    output_dir_M: PNG output directory for processed images
    output_dir_Mask: PNG output directory for masks
    output_dir_Original: PNG output directory for original images
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

    # Internal function: Normalize and save image as PNG
    def save_image_as_png(image_data, output_path, normalize=True):
        """Save image data as PNG file"""
        if normalize and np.max(image_data) > np.min(image_data):
            # Normalize image to 0-255 range
            normalized_image = (image_data - np.min(image_data)) / (
                    np.max(image_data) - np.min(image_data))
            normalized_image = (normalized_image * 255).astype(np.uint8)
        else:
            normalized_image = image_data.astype(np.uint8)

        # Use PIL to save PNG
        png_image = Image.fromarray(normalized_image)
        png_image.save(output_path)

    # Create all output directories
    os.makedirs(output_dir_D, exist_ok=True)
    os.makedirs(output_dir_M, exist_ok=True)
    os.makedirs(output_dir_Mask, exist_ok=True)
    os.makedirs(output_dir_Original, exist_ok=True)

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

            # Save original image as PNG
            original_png_path = os.path.join(output_dir_Original, f"{filename}.png")
            save_image_as_png(image_data, original_png_path, normalize=True)

            # Use internal function to remove background
            processed_image, mask = remove_single_image_background(image_data)

            # Save processed DICOM file (keep original filename)
            output_dicom_path = os.path.join(output_dir_D, filename)

            # Update DICOM file's pixel data
            ds.PixelData = processed_image.tobytes()
            ds.Rows, ds.Columns = processed_image.shape

            # Save DICOM file
            pydicom.dcmwrite(output_dicom_path, ds)

            # Save processed image as PNG
            processed_png_path = os.path.join(output_dir_M, f"{filename}.png")
            save_image_as_png(processed_image, processed_png_path, normalize=True)

            # Save mask as PNG
            mask_png_path = os.path.join(output_dir_Mask, f"{filename}_mask.png")

            # Convert mask to 8-bit image (0 and 255) and save
            mask_image = (mask * 255).astype(np.uint8)
            mask_png = Image.fromarray(mask_image)
            mask_png.save(mask_png_path)

            # Display before-and-after comparison
            if show_comparison and processed_count < 3:
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))

                # Original image
                axes[0, 0].imshow(image_data, cmap='gray')
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')

                # Mask
                axes[0, 1].imshow(mask, cmap='gray')
                axes[0, 1].set_title('Background Mask')
                axes[0, 1].axis('off')

                # Processed image
                axes[0, 2].imshow(processed_image, cmap='gray')
                axes[0, 2].set_title('Processed Image')
                axes[0, 2].axis('off')

                # Original with mask overlay
                axes[1, 0].imshow(image_data, cmap='gray')
                axes[1, 0].imshow(mask, cmap='Reds', alpha=0.3)
                axes[1, 0].set_title('Original with Mask Overlay')
                axes[1, 0].axis('off')

                # Histogram of original image
                axes[1, 1].hist(image_data.flatten(), bins=50, color='blue', alpha=0.7)
                axes[1, 1].axvline(x=np.min(image_data), color='red', linestyle='--',
                                   label=f'Min: {np.min(image_data):.2f}')
                axes[1, 1].axvline(x=np.max(image_data), color='green', linestyle='--',
                                   label=f'Max: {np.max(image_data):.2f}')
                axes[1, 1].set_title('Original Image Histogram')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)

                # Histogram of processed image
                axes[1, 2].hist(processed_image.flatten(), bins=50, color='orange', alpha=0.7)
                axes[1, 2].axvline(x=np.min(processed_image), color='red', linestyle='--',
                                   label=f'Min: {np.min(processed_image):.2f}')
                axes[1, 2].axvline(x=np.max(processed_image), color='green', linestyle='--',
                                   label=f'Max: {np.max(processed_image):.2f}')
                axes[1, 2].set_title('Processed Image Histogram')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

            processed_count += 1
            print(f"Processed: {filename} ({processed_count}/{len(dicom_files)})")
            print(f"  ✓ Original PNG: {original_png_path}")
            print(f"  ✓ Processed DICOM: {output_dicom_path}")
            print(f"  ✓ Processed PNG: {processed_png_path}")
            print(f"  ✓ Mask PNG: {mask_png_path}")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            continue


def main():
    input_directory = r"D:/med_data/ANY/data4"
    output_D = r"D:/med_data/ANY/pro4/D"
    output_M = r"D:/med_data/ANY/pro4/PNG"
    output_Mask = r"D:/med_data/ANY/pro4/Mask"
    output_Original = r"D:/med_data/ANY/pro4/Original"

    print("Starting DICOM image processing...")
    print(f"Input directory: {input_directory}")
    print(f"Output DICOM directory: {output_D}")
    print(f"Output PNG directory (processed): {output_M}")
    print(f"Output PNG directory (original): {output_Original}")
    print(f"Output PNG directory (mask): {output_Mask}")

    process_dicom_remove_background(
        input_dir=input_directory,
        output_dir_D=output_D,
        output_dir_M=output_M,
        output_dir_Mask=output_Mask,
        output_dir_Original=output_Original,
        show_comparison=True
    )

    print("Processing completed!")


if __name__ == "__main__":
    main()