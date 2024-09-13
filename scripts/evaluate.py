import numpy as np
import time
import os
from skimage import io, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize

def calculate_entropy(image):
    hist, _ = np.histogram(image, bins=256, range=(0, 1))
    prob = hist / np.sum(hist)
    return -np.sum(prob[prob > 0] * np.log2(prob[prob > 0]))

def calculate_mean_gradient(image):
    dx = np.abs(np.diff(image, axis=1))
    dy = np.abs(np.diff(image, axis=0))
    return np.mean(dx) + np.mean(dy)

def calculate_spatial_frequency(image):
    dx = np.diff(image, axis=1)
    dy = np.diff(image, axis=0)
    return np.sqrt(np.mean(dx**2) + np.mean(dy**2))

# Define paths
results_dir = '/Users/an.balcer/Desktop/DDcGAN/results/'
test_imgs_dir = '/Users/an.balcer/Desktop/DDcGAN/data/test_imgs/'

result_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.bmp')])

metrics = {
    'EN': [], 'MG': [], 'SF': [], 'SD': [],
    'SSIM_VIS': [], 'SSIM_IR': [],
    'PSNR_VIS': [], 'PSNR_IR': [],
    'CC_VIS': [], 'CC_IR': [],
    'Inference_Time': []
}

for result_file in result_files:
    try:
        start_time = time.time()  # Start timing for inference

        fused_image = img_as_float(io.imread(os.path.join(results_dir, result_file), as_gray=True))
        
        vis_file = f"VIS{result_file.split('.')[0]}.bmp"
        ir_file = f"IR{result_file.split('.')[0]}_ds.bmp"
        
        vis_image = img_as_float(io.imread(os.path.join(test_imgs_dir, vis_file), as_gray=True))
        ir_image = img_as_float(io.imread(os.path.join(test_imgs_dir, ir_file), as_gray=True))
        
        ir_image = resize(ir_image, fused_image.shape, anti_aliasing=True)
        
        metrics['EN'].append(calculate_entropy(fused_image))
        metrics['MG'].append(calculate_mean_gradient(fused_image))
        metrics['SF'].append(calculate_spatial_frequency(fused_image))
        metrics['SD'].append(np.std(fused_image))
        metrics['SSIM_VIS'].append(ssim(fused_image, vis_image, data_range=1))
        metrics['SSIM_IR'].append(ssim(fused_image, ir_image, data_range=1))
        metrics['PSNR_VIS'].append(psnr(fused_image, vis_image, data_range=1))
        metrics['PSNR_IR'].append(psnr(fused_image, ir_image, data_range=1))
        metrics['CC_VIS'].append(np.corrcoef(fused_image.flatten(), vis_image.flatten())[0, 1])
        metrics['CC_IR'].append(np.corrcoef(fused_image.flatten(), ir_image.flatten())[0, 1])
        
        end_time = time.time()  # End timing for inference
        inference_time = end_time - start_time
        metrics['Inference_Time'].append(inference_time)
        
        print(f"Processed {result_file} successfully. Inference time: {inference_time:.4f} seconds")
    except Exception as e:
        print(f"Error processing {result_file}: {str(e)}")

# Calculate averages of metrics
avg_metrics = {k: np.mean(v) for k, v in metrics.items()}

# Print results
for k, v in avg_metrics.items():
    print(f'Average {k}: {v:.4f}')

# Save results
with open('evaluation_metrics.txt', 'w') as f:
    for k, v in avg_metrics.items():
        f.write(f'Average {k}: {v:.4f}\n')
