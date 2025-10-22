import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os

def mse(imageA, imageB):
    # Mean Squared Error
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1] * imageA.shape[2])
    return err

def psnr(imageA, imageB):
    mse_val = np.mean((imageA.astype(np.float64) - imageB.astype(np.float64)) ** 2)
    if mse_val == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_val))

def load_gif_frames(path):
    # Load GIF frames as RGB arrays
    gif = Image.open(path)
    frames = []
    try:
        while True:
            frame = gif.convert('RGB')
            frames.append(np.array(frame))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    return frames

def compare_gifs(input_path, output_path, message):
    print("🔍 Comparing GIFs...\n")
    print(f"Original GIF: {input_path}")
    print(f"Stego GIF: {output_path}")
    print(f"Encoded message: \"{message}\"\n")

    input_frames = load_gif_frames(input_path)
    output_frames = load_gif_frames(output_path)

    if len(input_frames) != len(output_frames):
        print("⚠️ Frame count mismatch!")
        print(f"Original: {len(input_frames)} frames | Stego: {len(output_frames)} frames")
        return

    total_mse, total_psnr, total_ssim = 0, 0, 0
    for i, (imgA, imgB) in enumerate(zip(input_frames, output_frames)):
        imgA = cv2.resize(imgA, (min(imgA.shape[1], imgB.shape[1]), min(imgA.shape[0], imgB.shape[0])))
        imgB = cv2.resize(imgB, (imgA.shape[1], imgA.shape[0]))

        m = mse(imgA, imgB)
        p = psnr(imgA, imgB)

        # ✅ FIX: pass win_size=3 and channel_axis=-1 for latest version
        s = ssim(imgA, imgB, win_size=3, channel_axis=-1)

        total_mse += m
        total_psnr += p
        total_ssim += s

        print(f"🖼 Frame {i+1}: MSE={m:.4f}, PSNR={p:.2f} dB, SSIM={s:.4f}")

    n = len(input_frames)
    print("\n📊 Average Metrics Across All Frames:")
    print(f"➡️ MSE  : {total_mse/n:.4f}")
    print(f"➡️ PSNR : {total_psnr/n:.2f} dB")
    print(f"➡️ SSIM : {total_ssim/n:.4f}")

if __name__ == "__main__":
    input_path = r"D:\Debayan\yhpargonagets\model\input.gif"
    output_path = r"D:\Debayan\yhpargonagets\model\output.gif"
    message = "hi how are you"
    compare_gifs(input_path, output_path, message)
