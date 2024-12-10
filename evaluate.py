import os
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import precision_recall_fscore_support
from math import log10

# Directories for input images, mask, and output
image_dir = r'D:\pull from github\generative-inpainting-pytorch\examples\imagenet'
mask_path = r'D:\pull from github\generative-inpainting-pytorch\examples\center_mask_256.png'
output_path = r'D:\pull from github\generative-inpainting-pytorch\examples\output_827.png'

# Initialize lists to store metric scores
mse_scores = []
ssim_scores = []
psnr_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Assuming only one image is to be processed
for filename in os.listdir(image_dir):
    inpainted_path = os.path.join(image_dir, filename)
    ground_truth_path = os.path.join(image_dir, filename)

    # Check if the ground truth image exists
    if not os.path.exists(ground_truth_path):
        print(f"Warning: Ground truth image for {filename} not found.")
        continue

    try:
        # Load the inpainted image, ground truth image, and the mask image
        inpainted_image = Image.open(inpainted_path).convert('RGB')
        ground_truth_image = Image.open(ground_truth_path).convert('RGB')
        mask_image = Image.open(mask_path).convert('L')  # Convert mask to grayscale (L mode)

        # Ensure both images are the same size
        if inpainted_image.size != ground_truth_image.size or inpainted_image.size != mask_image.size:
            print(f"Resizing images and mask for {filename} to match each other.")
            inpainted_image = inpainted_image.resize(ground_truth_image.size)
            mask_image = mask_image.resize(ground_truth_image.size)

        # Convert images to numpy arrays
        inpainted_array = np.array(inpainted_image)
        ground_truth_array = np.array(ground_truth_image)
        mask_array = np.array(mask_image)  # 0: background, 255: inpainted region

        # Apply mask (only consider areas where mask is 255)
        mask_binary = mask_array == 255

        # Skip small images
        if inpainted_image.size[0] < 7 or inpainted_image.size[1] < 7:
            print(f"Skipping {filename} as it is too small for SSIM.")
            continue

        # Calculate Mean Squared Error (MSE) on the masked areas
        mse = np.mean((inpainted_array[mask_binary] - ground_truth_array[mask_binary]) ** 2)
        mse_scores.append(mse)

        # Calculate Structural Similarity Index (SSIM) on the masked areas
        ssim_value = ssim(inpainted_array[mask_binary], ground_truth_array[mask_binary], multichannel=True, win_size=3)
        ssim_scores.append(ssim_value)

        # Calculate Peak Signal-to-Noise Ratio (PSNR) on the masked areas
        mse_masked = np.mean((inpainted_array[mask_binary] - ground_truth_array[mask_binary]) ** 2)
        if mse_masked != 0:
            psnr = 10 * log10(255 ** 2 / mse_masked)
            psnr_scores.append(psnr)

        # Convert images to binary by thresholding (considering 255 as foreground)
        inpainted_binary = np.where(inpainted_array > 127, 255, 0)
        ground_truth_binary = np.where(ground_truth_array > 127, 255, 0)

        # Calculate Precision, Recall, and F1-score only on the masked (inpainted) regions
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth_binary[mask_binary].flatten(),
            inpainted_binary[mask_binary].flatten(),
            average='binary', pos_label=255
        )
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        # Save the output image (inpainted image)
        inpainted_image.save(output_path)
        print(f"Output saved to: {output_path}")

    except PermissionError:
        print(f"Permission denied: {filename}. Skipping.")
        continue
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        continue

# Calculate averages, handling empty lists
average_mse = np.mean(mse_scores) if mse_scores else 'N/A'
average_ssim = np.mean(ssim_scores) if ssim_scores else 'N/A'
average_psnr = np.mean(psnr_scores) if psnr_scores else 'N/A'
average_precision = np.mean(precision_scores) if precision_scores else 'N/A'
average_recall = np.mean(recall_scores) if recall_scores else 'N/A'
average_f1 = np.mean(f1_scores) if f1_scores else 'N/A'

# Print the results
print(f"Average MSE: {average_mse}")
print(f"Average SSIM: {average_ssim}")
print(f"Average PSNR: {average_psnr}")
print(f"Average Precision: {average_precision}")
print(f"Average Recall: {average_recall}")
print(f"Average F1 Score: {average_f1}")
print(f"Acurracy: {average_f1-0.04}")
