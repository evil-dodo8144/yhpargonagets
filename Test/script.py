import argparse
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import os
import Levenshtein as lev

def evaluate_image_quality(original, stego):
    """Calculate image quality metrics between original and stego images"""
    # Convert images to grayscale for SSIM calculation
    if len(original.shape) == 3:
        original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        stego_gray = cv2.cvtColor(stego, cv2.COLOR_BGR2GRAY)
    else:
        original_gray = original
        stego_gray = stego
    
    # Calculate MSE (Mean Squared Error)
    mse = mean_squared_error(original, stego)
    
    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    max_pixel = 255.0
    psnr = cv2.PSNR(original, stego, max_pixel)
    
    # Calculate SSIM (Structural Similarity Index)
    ssim_score = ssim(original_gray, stego_gray, 
                      data_range=stego_gray.max() - stego_gray.min())
    
    return {
        'MSE': mse,
        'PSNR': psnr,
        'SSIM': ssim_score
    }

def evaluate_message_accuracy(original_msg, extracted_msg):
    """Calculate message recovery accuracy metrics"""
    # Convert messages to binary for bitwise comparison
    original_bin = ''.join(format(ord(c), '08b') for c in original_msg)
    extracted_bin = ''.join(format(ord(c), '08b') for c in extracted_msg)
    
    # Bitwise accuracy
    min_length = min(len(original_bin), len(extracted_bin))
    if min_length == 0:
        bit_acc = 0.0
    else:
        matches = sum(1 for a, b in zip(original_bin[:min_length], 
                                       extracted_bin[:min_length]) if a == b)
        bit_acc = matches / min_length
    
    # Character-wise accuracy
    min_len_chars = min(len(original_msg), len(extracted_msg))
    if min_len_chars == 0:
        char_acc = 0.0
    else:
        char_matches = sum(1 for a, b in zip(original_msg[:min_len_chars], 
                                           extracted_msg[:min_len_chars]) if a == b)
        char_acc = char_matches / min_len_chars
    
    # Levenshtein distance (edit distance)
    lev_distance = lev.distance(original_msg, extracted_msg)
    
    # Hamming distance (only for equal length strings)
    if len(original_msg) == len(extracted_msg):
        hamming_dist = lev.hamming(original_msg, extracted_msg)
    else:
        hamming_dist = "N/A (lengths differ)"
    
    return {
        'Bit Accuracy': bit_acc,
        'Character Accuracy': char_acc,
        'Levenshtein Distance': lev_distance,
        'Hamming Distance': hamming_dist
    }

def extract_message(stego_image_path):
    """PLACEHOLDER: Replace with actual message extraction logic"""
    # In a real implementation, this would contain your steganography extraction code
    print(f"⚠️ Using placeholder message extractor. Replace this function with your actual implementation!")
    
    # Dummy extracted message - replace with actual extraction
    return "This is a test message extracted from the stego image."

def generate_report(metrics, output_file="evaluation_report.txt"):
    """Generate evaluation report with suggestions for improvement"""
    report = "Steganography Quality Evaluation Report\n"
    report += "=" * 50 + "\n\n"
    
    # Image quality section
    report += "IMAGE QUALITY METRICS:\n"
    report += f"- MSE: {metrics['image']['MSE']:.4f}\n"
    report += f"- PSNR: {metrics['image']['PSNR']:.2f} dB\n"
    report += f"- SSIM: {metrics['image']['SSIM']:.4f}\n\n"
    
    # Image quality suggestions
    report += "QUALITY ASSESSMENT:\n"
    if metrics['image']['PSNR'] < 30:
        report += "❌ PSNR < 30 dB indicates significant quality degradation\n"
        report += "   Suggestion: Reduce embedding strength or use adaptive embedding\n"
    elif metrics['image']['PSNR'] < 40:
        report += "⚠️ PSNR between 30-40 dB shows moderate quality degradation\n"
        report += "   Suggestion: Optimize embedding regions to preserve critical areas\n"
    else:
        report += "✅ Excellent image quality (PSNR > 40 dB)\n"
    
    if metrics['image']['SSIM'] < 0.9:
        report += "❌ SSIM < 0.9 indicates noticeable structural differences\n"
        report += "   Suggestion: Use perceptual masking to preserve structural elements\n"
    else:
        report += "✅ Structural integrity well preserved (SSIM > 0.9)\n"
    report += "\n"
    
    # Message recovery section
    report += "MESSAGE RECOVERY METRICS:\n"
    for key, value in metrics['message'].items():
        report += f"- {key}: {value}\n"
    
    report += "\nMESSAGE INTEGRITY ASSESSMENT:\n"
    if metrics['message']['Bit Accuracy'] == 1.0:
        report += "✅ Perfect bit recovery achieved\n"
    elif metrics['message']['Bit Accuracy'] > 0.95:
        report += "⚠️ Good recovery with minor errors\n"
        report += "   Suggestion: Add error correction coding to the payload\n"
    else:
        report += "❌ Significant bit errors detected\n"
        report += "   Suggestion: Check embedding algorithm and extraction logic\n"
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Evaluation report saved to {output_file}")
    return report

def main():
    parser = argparse.ArgumentParser(description='Steganography Quality Evaluator')
    parser.add_argument('--original_image', type=str, required=True,
                        help='Path to original cover image')
    parser.add_argument('--stego_image', type=str, required=True,
                        help='Path to stego image with embedded message')
    parser.add_argument('--original_message', type=str, default=None,
                        help='Path to original message file (optional)')
    args = parser.parse_args()

    # Validate paths
    for path in [args.original_image, args.stego_image]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
    
    # Load images
    original_img = cv2.imread(args.original_image)
    stego_img = cv2.imread(args.stego_image)
    
    if original_img is None or stego_img is None:
        raise ValueError("Error loading images. Check file formats and paths")
    
    # Check image dimensions match
    if original_img.shape != stego_img.shape:
        raise ValueError("Image dimensions do not match!")

    # Load or create original message
    if args.original_message:
        if not os.path.exists(args.original_message):
            print(f"⚠️ Message file not found: {args.original_message}")
            original_msg = "Test message for steganography evaluation"
        else:
            with open(args.original_message, 'r') as f:
                original_msg = f.read()
    else:
        original_msg = "Test message for steganography evaluation"
    
    # Extract message from stego image
    extracted_msg = extract_message(args.stego_image)
    
    # Calculate metrics
    results = {
        'image': evaluate_image_quality(original_img, stego_img),
        'message': evaluate_message_accuracy(original_msg, extracted_msg)
    }
    
    # Display results
    print("\n" + "="*50)
    print("IMAGE QUALITY METRICS:")
    for metric, value in results['image'].items():
        print(f"{metric}: {value}")
    
    print("\nMESSAGE RECOVERY METRICS:")
    for metric, value in results['message'].items():
        print(f"{metric}: {value}")
    
    # Generate full report
    report = generate_report(results)
    print("\nFINAL ASSESSMENT:")
    print(report.split("QUALITY ASSESSMENT:")[-1])

if __name__ == "__main__":
    main()