import os
import sys

# ==============================================================================
# 1. --- USER DEFINED FILE PATHS ---
#   *** CHANGE THESE NAMES TO YOUR ACTUAL FILE NAMES ***
# ==============================================================================

# Input: The file containing the hidden data (this is the output from the 'hide' step).
STEGO_GIF_INPUT = '/home/suboptimal/Steganography/yhpargonagets/Gif_In_Gif/stego_hidden_result.gif' 

# Output: The file name for the recovered hidden GIF.
EXTRACTED_GIF_OUTPUT = 'recovered_payload.gif'

# ==============================================================================
# 2. --- CORE STEGANOGRAPHY LOGIC (Most Robust Header Search and Trailer Trimming) ---
# ==============================================================================

# Constants used for identifying the GIF file structure
GIF_TRAILER = b'\x3B'
# We search for the 4-byte header prefix, allowing for any version (e.g., 87a, 89a, or corrupted).
GIF_HEADER_PREFIX = b'GIF8' 

def extract_gif(stego_path: str, output_path: str):
    """
    Extracts the hidden payload by searching for the simpler 4-byte GIF header prefix 
    ('GIF8') and then explicitly trimming the data to the payload's own trailer.
    This offers the highest chance of successful recovery.
    """
    try:
        with open(stego_path, 'rb') as f:
            stego_data = f.read()
    except FileNotFoundError as e:
        print(f"Error: Stego file not found. Ensure '{e.filename}' exists in the correct folder.")
        sys.exit(1)

    if not stego_data.startswith(GIF_HEADER_PREFIX):
        print("Error: Input file does not start with GIF header.")
        sys.exit(1)

    print(f"Searching for the payload's GIF header prefix ('GIF8') in {stego_path}...")

    # The cover GIF's full header is 6 bytes long. Start searching for the *second* GIF header after that.
    search_start_index = 6 
    
    # Find the position of the payload's GIF header prefix. This is more flexible than searching for 'GIF89a'.
    payload_start_index = stego_data.find(GIF_HEADER_PREFIX, search_start_index)

    if payload_start_index == -1:
        print(f"\nError: Could not find any hidden GIF payload header prefix ('GIF8') after the cover GIF.")
        print("Extraction failed. The hidden file may not exist or the hiding mechanism was too destructive.")
        sys.exit(1)

    print(f"Found payload header starting at byte index {payload_start_index}...")

    # Start the payload from its validated header prefix
    potential_payload = bytearray(stego_data[payload_start_index:])
    
    # --- CRITICAL FIX: Find the payload's own trailer and trim the file ---
    # We must trim the payload to end exactly at its own trailer (0x3B).
    trailer_index = potential_payload.rfind(GIF_TRAILER)
    
    if trailer_index == -1:
        # If the trailer is not found, we append it for file validity.
        print("Warning: Payload trailer (0x3B) not found. Appending trailer for file validity.")
        payload = potential_payload
        payload.extend(GIF_TRAILER)
    else:
        # If the trailer is found, the payload ends one byte AFTER the trailer.
        # This prevents including extraneous data from the cover GIF.
        payload = potential_payload[:trailer_index + 1]
        print(f"Payload successfully trimmed to its internal trailer at index {trailer_index}.")

    # Write the extracted data
    with open(output_path, 'wb') as f:
        f.write(payload)

    print(f"\n--- EXTRACT SUCCESS ---")
    print(f"Payload extracted: {output_path}")
    print(f"Extracted size: {len(payload)} bytes.")
    print("The recovered file should now be a complete and loadable GIF.")


def main():
    """
    Main execution function using the fixed paths.
    """
    # Get the directory of the script for relative pathing
    script_dir = os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else '.'

    # Construct paths for EXTRACTION
    stego_path = os.path.join(script_dir, STEGO_GIF_INPUT)
    output_path = os.path.join(script_dir, EXTRACTED_GIF_OUTPUT)

    print("--- GIF Steganography (EXTRACT MODE - Most Robust) ---")
    print(f"1. Stego (Input): {STEGO_GIF_INPUT}")
    print(f"2. Output (Payload): {EXTRACTED_GIF_OUTPUT} (saved in the same directory)")

    extract_gif(stego_path, output_path)

if __name__ == '__main__':
    main()
