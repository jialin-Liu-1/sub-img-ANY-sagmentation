import os
import numpy as np
import pydicom
from PIL import Image
import matplotlib.pyplot as plt


def process_3d_dsa_remove_background(input_dir, output_dir_D, output_dir_M, output_dir_Mask, output_dir_Original,
                                     show_comparison=True):
    """
    Batch process 3D DSA DICOM images to remove background

    Parameters:
    input_dir: Input directory
    output_dir_D: DICOM output directory
    output_dir_M: PNG output directory for processed images
    output_dir_Mask: PNG output directory for masks
    output_dir_Original: PNG output directory for original images
    show_comparison: Whether to display comparison images
    """

    # Internal function: Remove background from a single 2D image slice
    def remove_single_image_background(image_data, max_search_distance=100, extend_pixels=3):
        """Remove background from a single 2D image"""
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

            # Get pixel data - 3D DSA: 2D + time dimension
            image_data = ds.pixel_array

            print(f"Processing 3D DSA image: {filename}, Shape: {image_data.shape}")

            # Check if it's a 3D image (dimensions > 2)
            if len(image_data.shape) <= 2:
                print(f"  警告: {filename} 不是3D图像，跳过处理")
                continue

            # For 3D DSA images: shape is (time, height, width) or (height, width, time)
            # We need to identify which dimension is time
            if len(image_data.shape) == 3:
                # Find the smallest dimension as time (usually)
                time_dim = np.argmin(image_data.shape)
                height_dim = 0 if time_dim != 0 else 1
                width_dim = 2 if time_dim != 2 else 1

                # Rearrange dimensions to (time, height, width)
                if time_dim == 0:
                    # Already in (time, height, width)
                    pass
                elif time_dim == 1:
                    # (height, time, width) -> (time, height, width)
                    image_data = np.transpose(image_data, (1, 0, 2))
                elif time_dim == 2:
                    # (height, width, time) -> (time, height, width)
                    image_data = np.transpose(image_data, (2, 0, 1))

                n_timepoints, height, width = image_data.shape
                print(f"  3D DSA图像: {n_timepoints}个时间点, {height}x{width}")

                # Select middle timepoint for mask generation
                middle_timepoint = n_timepoints // 2
                print(f"  使用中间时间点 {middle_timepoint} 生成mask")

                # Extract middle slice for mask generation
                middle_slice = image_data[middle_timepoint]

                # Save original middle slice as PNG
                original_png_path = os.path.join(output_dir_Original, f"{filename}.png")
                save_image_as_png(middle_slice, original_png_path, normalize=True)

                # Generate mask from middle slice
                processed_slice, mask = remove_single_image_background(middle_slice)

                # Apply the same mask to ALL timepoints
                print(f"  将mask应用到所有{len(image_data)}个时间点...")
                processed_3d_data = image_data.copy()
                max_val = np.max(processed_3d_data)

                # Apply mask to each timepoint
                for t in range(n_timepoints):
                    processed_3d_data[t][mask == 1] = max_val

                # Save processed middle slice as PNG
                processed_png_path = os.path.join(output_dir_M, f"{filename}.png")
                save_image_as_png(processed_slice, processed_png_path, normalize=True)

                # Save mask as PNG
                mask_png_path = os.path.join(output_dir_Mask, f"{filename}_mask.png")
                mask_image = (mask * 255).astype(np.uint8)
                mask_png = Image.fromarray(mask_image)
                mask_png.save(mask_png_path)

                # Save processed DICOM file (keep original filename)
                output_dicom_path = os.path.join(output_dir_D, filename)

                # Update DICOM file's pixel data
                # Transpose back to original orientation if needed
                # But first, flatten the time dimension for saving
                if ds.NumberOfFrames is not None:
                    # DICOM expects pixel array with NumberOfFrames dimension
                    ds.PixelData = processed_3d_data.tobytes()
                    ds.Rows = height
                    ds.Columns = width
                    ds.NumberOfFrames = n_timepoints
                else:
                    # Some DICOM files don't have NumberOfFrames
                    # We need to handle them differently
                    ds.PixelData = processed_3d_data.tobytes()
                    ds.Rows = height
                    ds.Columns = width
                    # Add NumberOfFrames tag if not present
                    if (0x0028, 0x0008) not in ds:
                        ds.NumberOfFrames = n_timepoints

                # Save DICOM file
                pydicom.dcmwrite(output_dicom_path, ds)

                # Display before-and-after comparison for middle timepoint
                if show_comparison and processed_count < 3:
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    fig.suptitle(f'3D DSA Image: {filename} (Timepoint {middle_timepoint})', fontsize=16)

                    # Original middle slice
                    axes[0, 0].imshow(middle_slice, cmap='gray')
                    axes[0, 0].set_title(f'Original (Timepoint {middle_timepoint})')
                    axes[0, 0].axis('off')

                    # Mask
                    axes[0, 1].imshow(mask, cmap='gray')
                    axes[0, 1].set_title('Background Mask')
                    axes[0, 1].axis('off')

                    # Processed middle slice
                    axes[0, 2].imshow(processed_slice, cmap='gray')
                    axes[0, 2].set_title(f'Processed (Timepoint {middle_timepoint})')
                    axes[0, 2].axis('off')

                    # Original with mask overlay
                    axes[1, 0].imshow(middle_slice, cmap='gray')
                    axes[1, 0].imshow(mask, cmap='Reds', alpha=0.3)
                    axes[1, 0].set_title('Original with Mask Overlay')
                    axes[1, 0].axis('off')

                    # Histogram of original middle slice
                    axes[1, 1].hist(middle_slice.flatten(), bins=50, color='blue', alpha=0.7)
                    axes[1, 1].axvline(x=np.min(middle_slice), color='red', linestyle='--',
                                       label=f'Min: {np.min(middle_slice):.2f}')
                    axes[1, 1].axvline(x=np.max(middle_slice), color='green', linestyle='--',
                                       label=f'Max: {np.max(middle_slice):.2f}')
                    axes[1, 1].set_title('Original Histogram')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)

                    # Histogram of processed middle slice
                    axes[1, 2].hist(processed_slice.flatten(), bins=50, color='orange', alpha=0.7)
                    axes[1, 2].axvline(x=np.min(processed_slice), color='red', linestyle='--',
                                       label=f'Min: {np.min(processed_slice):.2f}')
                    axes[1, 2].axvline(x=np.max(processed_slice), color='green', linestyle='--',
                                       label=f'Max: {np.max(processed_slice):.2f}')
                    axes[1, 2].set_title('Processed Histogram')
                    axes[1, 2].legend()
                    axes[1, 2].grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.show()

                processed_count += 1
                print(f"Processed 3D DSA: {filename} ({processed_count}/{len(dicom_files)})")
                print(f"  ✓ Original PNG (middle timepoint): {original_png_path}")
                print(f"  ✓ Processed DICOM (all timepoints): {output_dicom_path}")
                print(f"  ✓ Processed PNG (middle timepoint): {processed_png_path}")
                print(f"  ✓ Mask PNG: {mask_png_path}")
                print(f"  ✓ Mask applied to {n_timepoints} timepoints")

                # Display timepoint comparison
                if show_comparison and processed_count <= 1:
                    # Show different timepoints for the first processed file
                    selected_timepoints = [0, middle_timepoint, n_timepoints - 1]
                    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
                    fig2.suptitle(f'Different Timepoints of {filename}', fontsize=14)

                    for idx, t in enumerate(selected_timepoints):
                        axes2[idx].imshow(processed_3d_data[t], cmap='gray')
                        axes2[idx].set_title(f'Timepoint {t}')
                        axes2[idx].axis('off')

                    plt.tight_layout()
                    plt.show()
            else:
                print(f"  警告: {filename} 的维度 {image_data.shape} 不支持，跳过")

        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue


def main():
    input_directory = r"D:\med_data\ANY\0"
    output_D = r"D:\med_data\ANY\processed_3d_dsa\D"
    output_M = r"D:\med_data\ANY\processed_3d_dsa\PNG"
    output_Mask = r"D:\med_data\ANY\processed_3d_dsa\Mask"
    output_Original = r"D:\med_data\ANY\processed_3d_dsa\Original"

    print("Starting 3D DSA DICOM image processing...")
    print(f"Input directory: {input_directory}")
    print(f"Output DICOM directory: {output_D}")
    print(f"Output PNG directory (processed): {output_M}")
    print(f"Output PNG directory (original): {output_Original}")
    print(f"Output PNG directory (mask): {output_Mask}")

    process_3d_dsa_remove_background(
        input_dir=input_directory,
        output_dir_D=output_D,
        output_dir_M=output_M,
        output_dir_Mask=output_Mask,
        output_dir_Original=output_Original,
        show_comparison=True
    )

    print("3D DSA processing completed!")


if __name__ == "__main__":
    main()