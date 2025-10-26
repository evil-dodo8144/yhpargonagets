import os
import sys
from PIL import Image

# ==============================================================================
# --- CONFIGURATION (STUDENT ADJUSTMENT AREA) ---
# ==============================================================================

# Input 1: The visible image (container). This is where the secret will be hidden.
COVER_IMAGE_INPUT = '/home/suboptimal/Steganography/yhpargonagets/Image_in_Image/kp50Ri.jpg'      

# Input 2: The secret image to hide. Its data will be embedded into the LSB of the cover.
PAYLOAD_IMAGE_INPUT = '/home/suboptimal/Steganography/yhpargonagets/Image_in_Image/OIP.webp'  

# Output: The resulting image (stego-image). It will look visually identical to the cover.
STEGO_IMAGE_OUTPUT = 'stego_output.png' 

# Define a unique end marker to signal where the hidden data stops.
# This marker is embedded along with the payload data.
END_OF_DATA_MARKER = b'\x00\x00\x03'

# ==============================================================================
# --- CORE LOGIC: BINARY CONVERSION AND LSB MANIPULATION ---
# ==============================================================================

def bytes_to_bits(data: bytes) -> list:
    """
    Converts a sequence of bytes into a flat list of 0s and 1s (bits).
    This transforms the image data into the smallest units we can hide.
    """
    bits = []
    for byte in data:
        # For each byte (8 bits), extract the bits from most significant (7) to least significant (0).
        for i in range(7, -1, -1):
            # Use bitwise right shift (>> i) and AND (& 1) to get the bit value.
            bits.append((byte >> i) & 1)
    return bits

def get_image_data(path: str) -> bytes:
    """
    Reads an image, temporarily saves it as a PNG byte stream, and returns the raw data.
    Saving as PNG ensures consistent, lossless pixel data for embedding.
    """
    try:
        img = Image.open(path)
        
        # Use a temporary file path for the byte stream
        byte_stream_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_payload.png")
        
        # Convert and save the image's raw data as a PNG file
        img.save(byte_stream_path, format='PNG')

        with open(byte_stream_path, 'rb') as f:
            data = f.read()

        # Clean up the temporary file
        os.remove(byte_stream_path)
        return data
        
    except FileNotFoundError:
        print(f"Error: Image file '{path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        sys.exit(1)

# ==============================================================================
# --- HIDING FUNCTION ---
# ==============================================================================

def hide(cover_path: str, payload_path: str, output_path: str):
    """
    The main function to embed the payload image's data into the LSB 
    of the cover image.
    """
    
    # 1. Prepare Payload Data
    payload_data = get_image_data(payload_path)
    
    # Append the End Marker to the payload. This tells the extractor where to stop.
    secret_data = payload_data + END_OF_DATA_MARKER
    secret_bits = bytes_to_bits(secret_data)
    
    print(f"Payload Size (Bytes): {len(secret_data)}")
    print(f"Payload Bits to Hide: {len(secret_bits)}")

    # 2. Open Cover Image and Validate Space
    # We convert to 'RGB' mode to ensure exactly three 8-bit channels per pixel.
    try:
        cover_img = Image.open(cover_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Cover file '{cover_path}' not found.")
        sys.exit(1)
        
    cover_width, cover_height = cover_img.size
    
    # Validation check: We need 1 cover pixel for every 3 hidden bits.
    # Total pixels required must be less than total pixels available.
    required_pixels = (len(secret_bits) + 2) // 3 # Ensure proper rounding up
    total_pixels = cover_width * cover_height
    
    if required_pixels > total_pixels:
        print("\nERROR: Cover image is too small to hide the payload!")
        print(f"Required pixels: {required_pixels} | Available pixels: {total_pixels}")
        sys.exit(1)

    print(f"Space Check: {required_pixels} pixels required. Hiding...")
    
    # 3. LSB Embedding Process
    
    pixels = cover_img.load()
    bit_index = 0
    
    # Iterate through every pixel (x, y) in the cover image
    for y in range(cover_height):
        for x in range(cover_width):
            if bit_index >= len(secret_bits):
                # Stop iterating once all secret bits and the marker are hidden
                break
            
            # Get the current pixel's color values (R, G, B)
            r, g, b = pixels[x, y]
            
            # Use a list to hold the channel values for modification
            new_rgb = list(pixels[x, y])

            # For each color channel (R, G, B), embed one secret bit
            for i in range(3): # i=0 (Red), i=1 (Green), i=2 (Blue)
                if bit_index < len(secret_bits):
                    secret_bit = secret_bits[bit_index]
                    
                    # LSB Logic:
                    # 1. Clear the LSB of the channel value (e.g., 253 -> 252)
                    new_val = new_rgb[i] & 0xFE 
                    
                    # 2. Set the LSB to the secret bit (e.g., 252 | 1 = 253)
                    new_rgb[i] = new_val | secret_bit
                    
                    bit_index += 1
                else:
                    break # Ran out of bits for this pixel

            # Set the modified pixel back into the image
            pixels[x, y] = tuple(new_rgb)
        
        if bit_index >= len(secret_bits):
            break

    # 4. Save the Stego Image
    cover_img.save(output_path)
    print(f"\n--- HIDE SUCCESS ---")
    print(f"Payload hidden successfully in: {output_path}")

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

if __name__ == '__main__':
    # Determine the script directory for consistent pathing
    script_dir = os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else '.'

    print("--- LSB STEGANOGRAPHY: HIDING ONLY MODE ---")
    cover_path = os.path.join(script_dir, COVER_IMAGE_INPUT)
    payload_path = os.path.join(script_dir, PAYLOAD_IMAGE_INPUT)
    output_path = os.path.join(script_dir, STEGO_IMAGE_OUTPUT)
    hide(cover_path, payload_path, output_path)
