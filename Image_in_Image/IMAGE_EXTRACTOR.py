import os
import sys
from PIL import Image

# ==============================================================================
# --- CONFIGURATION (STUDENT ADJUSTMENT AREA) ---
# NOTE: These settings MUST match the HIDER script configuration.
# ==============================================================================

# Input: The image containing the hidden data (output from the HIDER script).
STEGO_IMAGE_INPUT = '/home/suboptimal/Steganography/yhpargonagets/Image_in_Image/stego_output.png' 

# Output: The path where the recovered secret image will be saved.
EXTRACTED_IMAGE_OUTPUT = 'recovered_payload.png' 

# Define the unique end marker. This MUST be the exact marker used by the hider.
END_OF_DATA_MARKER = b'\x00\x00\x03'

# ==============================================================================
# --- CORE LOGIC: BINARY CONVERSION AND LSB MANIPULATION ---
# ==============================================================================

def bits_to_bytes(bits: list) -> bytes:
    """
    Converts a flat list of 0s and 1s back into bytes.
    This reassembles the individual hidden bits into the original file's bytes.
    """
    bytes_list = []
    # Process bits in chunks of 8
    for i in range(0, len(bits), 8):
        byte = 0
        # Iterate over the 8 bits in the chunk
        for j in range(8):
            bit = bits[i + j]
            # Use bitwise OR to set the bit position
            byte |= (bit << (7 - j))
        bytes_list.append(byte)
    return bytes(bytes_list)

# ==============================================================================
# --- EXTRACTION FUNCTION ---
# ==============================================================================

def extract(stego_path: str, output_path: str):
    """
    Extracts the hidden payload bits from the LSB of the stego image and 
    saves the recovered data as an image file.
    """
    
    try:
        # Open the stego image and convert it to 'RGB' for consistent channel access.
        stego_img = Image.open(stego_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Stego file '{stego_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error opening image {stego_path}: {e}")
        sys.exit(1)

    width, height = stego_img.size
    pixels = stego_img.load()
    
    extracted_bits = []
    
    print("Starting LSB extraction (searching for end marker)...")

    # Iterate through every pixel
    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]
            
            # For each color channel (R, G, B), extract one bit (the LSB)
            for channel_val in [r, g, b]:
                # Core LSB Extraction Logic:
                # Use bitwise AND with 1 (& 1) to isolate the LSB (the hidden bit).
                extracted_bit = channel_val & 1 
                extracted_bits.append(extracted_bit)
                
                # Check for the End Marker every 8 bits (i.e., every full byte)
                if len(extracted_bits) >= (len(END_OF_DATA_MARKER) * 8) and len(extracted_bits) % 8 == 0:
                    current_bytes = bits_to_bytes(extracted_bits)
                    
                    # Check if the recovered data ends with the marker
                    if current_bytes.endswith(END_OF_DATA_MARKER):
                        # Marker found! Trim it off to get the clean payload data.
                        payload_data = current_bytes[:-len(END_OF_DATA_MARKER)]
                        
                        # Save the recovered bytes as an image file (PNG format)
                        with open(output_path, 'wb') as f:
                            f.write(payload_data)

                        print(f"\n--- EXTRACT SUCCESS ---")
                        print(f"Hidden file successfully recovered to: {output_path}")
                        print(f"Recovered size: {len(payload_data)} bytes.")
                        return # Exit the function on success

    print("\nError: End marker not found.")
    print("Extraction failed. The file may be corrupted, or the cover image was too small to fully hide the payload.")

# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

if __name__ == '__main__':
    # Determine the script directory for consistent pathing
    script_dir = os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else '.'

    print("--- LSB STEGANOGRAPHY: EXTRACTION ONLY MODE ---")
    stego_path = os.path.join(script_dir, STEGO_IMAGE_INPUT)
    output_path = os.path.join(script_dir, EXTRACTED_IMAGE_OUTPUT)
    
    extract(stego_path, output_path)
