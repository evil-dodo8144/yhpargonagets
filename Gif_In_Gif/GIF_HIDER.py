import os
import sys

# ==============================================================================
# 1. --- USER DEFINED FILE PATHS ---
#   *** CHANGE THESE NAMES TO YOUR ACTUAL FILE NAMES ***
# ==============================================================================

# Input 1: The GIF that will be visible (the container).
COVER_GIF_INPUT = '/home/suboptimal/Steganography/yhpargonagets/Gif_In_Gif/yes-9.gif'

# Input 2: The GIF file to be hidden (the secret payload).
PAYLOAD_GIF_INPUT = '/home/suboptimal/Steganography/yhpargonagets/model/output.gif'

# Output: The resulting GIF containing the hidden data (stego-GIF).
STEGO_GIF_OUTPUT = 'stego_hidden_result.gif'

# ==============================================================================
# 2. --- CORE STEGANOGRAPHY LOGIC (Robust Hiding) ---
# ==============================================================================

# Constants used for identifying the GIF file structure
GIF_TRAILER = b'\x3B'

def hide_gif(cover_path: str, payload_path: str, output_path: str):
    """
    Hides the payload GIF inside the cover GIF's data stream, ensuring the 
    cover GIF is correctly terminated before insertion.
    """
    # --- Load Files ---
    try:
        with open(cover_path, 'rb') as f:
            cover_data = f.read()
    except FileNotFoundError:
        print(f"Error: Cover file '{cover_path}' not found.")
        sys.exit(1)

    try:
        with open(payload_path, 'rb') as f:
            payload_data = f.read()
    except FileNotFoundError:
        print(f"Error: Payload file '{payload_path}' not found.")
        sys.exit(1)

    # --- Validation and Repair of Cover GIF Trailer ---
    
    # 1. Find the official GIF trailer (0x3B) by searching backwards from the end
    trailer_index = cover_data.rfind(GIF_TRAILER)
    
    if trailer_index == -1:
        print(f"Error: Cover file '{cover_path}' does not contain a GIF Trailer (0x3B) anywhere.")
        sys.exit(1)

    # 2. Check if the trailer is the LAST byte
    if trailer_index != len(cover_data) - 1:
        # If the trailer is NOT the last byte, it means there is extraneous data.
        print(f"Warning: Cover GIF data contains {len(cover_data) - 1 - trailer_index} bytes of junk data after the trailer. Trimming...")
        
        # Trim the cover data to end exactly at the trailer byte
        clean_cover_data = cover_data[:trailer_index + 1]
    else:
        # If the trailer is the last byte, it's clean.
        clean_cover_data = cover_data
        print("Cover GIF validated: Ends correctly with Trailer (0x3B).")

    # --- Hiding Logic ---

    # 1. Remove the Trailer (0x3B) from the cover GIF's end
    # We must remove it because the payload and a new trailer will be appended.
    cover_data_without_trailer = clean_cover_data[:-1]

    # 2. Append the Payload Data
    # The payload is appended directly in the data stream.
    stego_data = cover_data_without_trailer + payload_data

    # 3. Append the final Trailer (0x3B) to mark the end of the entire file
    stego_data += GIF_TRAILER

    # --- Write Output ---
    with open(output_path, 'wb') as f:
        f.write(stego_data)

    print(f"\n--- HIDE SUCCESS ---")
    print(f"Payload hidden: {payload_path}")
    print(f"Stego-GIF created: {output_path}")
    print(f"Final size: {len(stego_data)} bytes.")


def main():
    """
    Main execution function using the fixed paths.
    """
    # Get the directory of the script for relative pathing
    script_dir = os.path.dirname(os.path.abspath(__file__)) if os.path.dirname(os.path.abspath(__file__)) else '.'

    # Construct paths
    cover_path = os.path.join(script_dir, COVER_GIF_INPUT)
    payload_path = os.path.join(script_dir, PAYLOAD_GIF_INPUT)
    output_path = os.path.join(script_dir, STEGO_GIF_OUTPUT)

    print("--- GIF Steganography (ROBUST HIDING MODE) ---")
    print(f"1. Cover (Input): {cover_path}")
    print(f"2. Payload (Input): {payload_path}")
    print(f"3. Output (Stego-GIF): {output_path}")

    hide_gif(cover_path, payload_path, output_path)

if __name__ == '__main__':
    main()
