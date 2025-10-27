import os
import sys
import io
import math
from PIL import Image, ImageChops
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Try to import numpy, which is needed for quality comparison
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

# ==============================================================================
# --- CONFIGURATION ---
# This marker is used by both the hider and extractor.
# ==============================================================================
END_OF_DATA_MARKER = b'\x00\x00\x03'

# ==============================================================================
# --- CORE LOGIC: BINARY CONVERSION (From your scripts) ---
# ==============================================================================

def bytes_to_bits(data: bytes) -> list:
    """
    Converts a sequence of bytes into a flat list of 0s and 1s (bits).
    """
    bits = []
    for byte in data:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return bits

def bits_to_bytes(bits: list) -> bytes:
    """
    Converts a flat list of 0s and 1s back into bytes.
    """
    bytes_list = []
    for i in range(0, len(bits), 8):
        byte = 0
        chunk = bits[i:i+8]
        
        # Ensure we have a full byte
        if len(chunk) < 8:
            continue 
            
        for j in range(8):
            bit = chunk[j]
            byte |= (bit << (7 - j))
        bytes_list.append(byte)
    return bytes(bytes_list)

def get_image_data_in_memory(path: str) -> bytes:
    """
    Reads an image and returns its raw PNG byte data from memory.
    Saving as PNG ensures consistent, lossless pixel data for embedding.
    This is the key to our steganography method.
    """
    try:
        img = Image.open(path)
        # Use an in-memory byte stream
        with io.BytesIO() as byte_stream:
            # We MUST save as PNG, as it's lossless. This is the data we hide.
            img.save(byte_stream, format='PNG')
            data = byte_stream.getvalue()
        return data
    except FileNotFoundError:
        raise Exception(f"Error: Image file '{path}' not found.")
    except Exception as e:
        raise Exception(f"Error processing image {path}: {e}")

# ==============================================================================
# --- CORE LOGIC: HIDING (Refactored for GUI) ---
# ==============================================================================

def hide(cover_path: str, payload_path: str, output_path: str, log_callback):
    """
    Embeds the payload image into the cover image.
    Uses log_callback to send status messages to the GUI.
    Returns True on success, False on failure.
    """
    try:
        # 1. Prepare Payload Data
        log_callback("Loading payload image and converting to PNG bytes...")
        # This function converts any image format (JPG, WEBP) into
        # a standard PNG byte stream. This is what we hide.
        payload_data = get_image_data_in_memory(payload_path)
        
        secret_data = payload_data + END_OF_DATA_MARKER
        secret_bits = bytes_to_bits(secret_data)
        
        log_callback(f"Payload Size (Bytes): {len(secret_data)}")
        log_callback(f"Payload Bits to Hide: {len(secret_bits)}")

        # 2. Open Cover Image and Validate Space
        log_callback("Loading cover image...")
        try:
            cover_img = Image.open(cover_path).convert('RGB')
        except FileNotFoundError:
            log_callback(f"Error: Cover file '{cover_path}' not found.")
            return False
            
        cover_width, cover_height = cover_img.size
        
        required_pixels = (len(secret_bits) + 2) // 3
        total_pixels = cover_width * cover_height
        
        if required_pixels > total_pixels:
            log_callback("\nERROR: Cover image is too small!")
            log_callback(f"Required pixels: {required_pixels} | Available: {total_pixels}")
            return False

        log_callback(f"Space Check: OK. {required_pixels} pixels required.")
        log_callback("Starting LSB embedding...")
        
        # 3. LSB Embedding Process
        pixels = cover_img.load()
        bit_index = 0
        
        for y in range(cover_height):
            for x in range(cover_width):
                if bit_index >= len(secret_bits):
                    break
                
                new_rgb = list(pixels[x, y])

                for i in range(3): # R, G, B
                    if bit_index < len(secret_bits):
                        secret_bit = secret_bits[bit_index]
                        new_val = new_rgb[i] & 0xFE # Clear LSB
                        new_rgb[i] = new_val | secret_bit # Set LSB
                        bit_index += 1
                
                pixels[x, y] = tuple(new_rgb)
            
            if bit_index >= len(secret_bits):
                break

        # 4. Save the Stego Image (must be PNG to be lossless)
        cover_img.save(output_path, "PNG")
        log_callback(f"\n--- HIDE SUCCESS ---")
        log_callback(f"Payload hidden successfully in: {output_path}")
        return True

    except Exception as e:
        log_callback(f"\nAn unexpected error occurred during hiding: {e}")
        return False

# ==============================================================================
# --- CORE LOGIC: EXTRACTION (Refactored for GUI) ---
# ==============================================================================

def extract(stego_path: str, output_path: str, log_callback):
    """
    Extracts the hidden payload from the stego image.
    Uses log_callback to send status messages to the GUI.
    Returns True on success, False on failure.
    """
    try:
        # Open the stego image
        try:
            stego_img = Image.open(stego_path).convert('RGB')
        except FileNotFoundError:
            log_callback(f"Error: Stego file '{stego_path}' not found.")
            return False
        except Exception as e:
            log_callback(f"Error opening image {stego_path}: {e}")
            return False

        width, height = stego_img.size
        pixels = stego_img.load()
        extracted_bits = []
        
        log_callback("Starting LSB extraction (searching for end marker)...")
        marker_len_bits = len(END_OF_DATA_MARKER) * 8

        for y in range(height):
            for x in range(width):
                r, g, b = pixels[x, y]
                
                for channel_val in [r, g, b]:
                    extracted_bit = channel_val & 1 
                    extracted_bits.append(extracted_bit)
                    
                    # Check for the End Marker every 8 bits
                    if len(extracted_bits) >= marker_len_bits and len(extracted_bits) % 8 == 0:
                        
                        potential_marker_bits = extracted_bits[-marker_len_bits:]
                        current_bytes = bits_to_bytes(potential_marker_bits)
                        
                        if current_bytes == END_OF_DATA_MARKER:
                            log_callback("End marker found!")
                            
                            # Get all bits *before* the marker
                            payload_bits = extracted_bits[:-marker_len_bits]
                            payload_data = bits_to_bytes(payload_bits)
                            
                            # Save the recovered bytes (which are a PNG file)
                            with open(output_path, 'wb') as f:
                                f.write(payload_data)

                            log_callback(f"\n--- EXTRACT SUCCESS ---")
                            log_callback(f"Hidden file recovered to: {output_path}")
                            log_callback(f"Recovered size: {len(payload_data)} bytes.")
                            return True # Exit the function on success

        log_callback("\nError: End marker not found.")
        log_callback("Extraction failed. File may be corrupt or not a stego-image.")
        return False

    except Exception as e:
        log_callback(f"\nAn unexpected error occurred during extraction: {e}")
        return False

# ==============================================================================
# --- CORE LOGIC: QUALITY COMPARISON ---
# ==============================================================================

def compare_images(original_path: str, extracted_path: str, log_callback):
    """
    Compares the original payload with the extracted one and logs quality metrics.
    """
    if not NUMPY_AVAILABLE:
        log_callback("Error: NumPy is not installed. Quality comparison is disabled.")
        log_callback("Please install it by running: pip install numpy")
        return

    try:
        log_callback("\n--- Starting Quality Comparison ---")
        
        # 1. Load extracted image (this is already a PNG)
        extracted_img = Image.open(extracted_path).convert('RGB')
        log_callback(f"Loaded extracted image: {extracted_path}")

        # 2. Load original image and convert it to PNG bytes *in memory*
        # This simulates the exact process it went through before hiding.
        # This is the ONLY way to get an accurate, fair comparison.
        log_callback(f"Loading original image and converting to PNG for comparison...")
        original_img = Image.open(original_path)
        with io.BytesIO() as byte_stream:
            original_img.save(byte_stream, format='PNG')
            byte_stream.seek(0)
            original_as_png = Image.open(byte_stream).convert('RGB')
        
        log_callback("Images loaded and formatted for comparison.")

        # 3. Check dimensions
        if original_as_png.size != extracted_img.size:
            log_callback("Error: Image dimensions do not match.")
            log_callback(f"Original (as PNG): {original_as_png.size}")
            log_callback(f"Extracted: {extracted_img.size}")
            return

        # 4. Calculate Difference
        diff = ImageChops.difference(original_as_png, extracted_img)
        np_diff = np.array(diff)
        
        # 5. Calculate MSE (Mean Squared Error)
        mse = np.mean(np_diff**2)
        
        if mse == 0:
            log_callback("--- Quality Result ---")
            log_callback("Images are PIXEL-PERFECT IDENTICAL.")
            log_callback("MSE: 0.0")
            log_callback("PSNR: Infinite (Perfect)")
        else:
            # 6. Calculate PSNR (Peak Signal-to-Noise Ratio)
            max_pixel = 255.0
            psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
            
            log_callback("--- Quality Result ---")
            log_callback("Images are DIFFERENT.")
            log_callback(f"MSE: {mse:.4f}")
            log_callback(f"PSNR: {psnr:.4f} dB")
            log_callback("(Higher PSNR is better. Infinite is perfect.)")

    except Exception as e:
        log_callback(f"\nAn error occurred during comparison: {e}")

# ==============================================================================
# --- GUI APPLICATION CLASS ---
# ==============================================================================

class SteganographyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("LSB Steganography Tool")
        self.root.geometry("600x700") # Increased height for compare section

        # --- Get script directory for portable paths ---
        # This makes file dialogs default to the app's folder
        try:
            # For bundled app (PyInstaller)
            self.script_dir = sys._MEIPASS
        except AttributeError:
            # For .py script
            self.script_dir = os.path.dirname(os.path.abspath(__file__))


        # --- Create Tabs ---
        self.notebook = ttk.Notebook(root)
        self.hide_frame = ttk.Frame(self.notebook, padding=10)
        self.extract_frame = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.hide_frame, text='Hide Image')
        self.notebook.add(self.extract_frame, text='Extract Image')
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        # --- Variables to store file paths ---
        self.cover_path = tk.StringVar()
        self.payload_path = tk.StringVar()
        self.hide_output_path = tk.StringVar()
        
        self.stego_input_path = tk.StringVar()
        self.extract_output_path = tk.StringVar()
        self.original_payload_compare_path = tk.StringVar()

        # --- Populate Hide Tab ---
        self.create_hide_widgets()

        # --- Populate Extract Tab ---
        self.create_extract_widgets()
        
        # --- Create Shared Log Area ---
        log_frame = ttk.LabelFrame(root, text="Log", padding=10)
        log_frame.pack(padx=10, pady=(0, 10), fill='x')
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state='disabled', wrap=tk.WORD)
        self.log_text.pack(fill='x', expand=True)
        
        clear_log_btn = ttk.Button(log_frame, text="Clear Log", command=self.clear_log)
        clear_log_btn.pack(pady=5)

    def log(self, message):
        """Appends a message to the log text box."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END) # Auto-scroll
        self.log_text.config(state='disabled')
        self.root.update_idletasks() # Force GUI to update

    def clear_log(self):
        """Clears the log text box."""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')

    def create_hide_widgets(self):
        """Builds the UI elements for the 'Hide' tab."""
        frame = self.hide_frame
        
        # --- Cover Image ---
        cover_frame = ttk.LabelFrame(frame, text="1. Cover Image (Visible)", padding=10)
        cover_frame.pack(fill='x', expand=True, pady=5)
        
        cover_entry = ttk.Entry(cover_frame, textvariable=self.cover_path, width=60)
        cover_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        
        cover_btn = ttk.Button(cover_frame, text="Browse...", command=self.select_cover)
        cover_btn.pack(side=tk.LEFT)

        # --- Payload Image ---
        payload_frame = ttk.LabelFrame(frame, text="2. Payload Image (Secret)", padding=10)
        payload_frame.pack(fill='x', expand=True, pady=5)
        
        payload_entry = ttk.Entry(payload_frame, textvariable=self.payload_path, width=60)
        payload_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        
        payload_btn = ttk.Button(payload_frame, text="Browse...", command=self.select_payload)
        payload_btn.pack(side=tk.LEFT)

        # --- Output Image ---
        output_frame = ttk.LabelFrame(frame, text="3. Output Stego Image (Save as .png)", padding=10)
        output_frame.pack(fill='x', expand=True, pady=5)
        
        output_entry = ttk.Entry(output_frame, textvariable=self.hide_output_path, width=60)
        output_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        
        output_btn = ttk.Button(output_frame, text="Save As...", command=self.select_hide_output)
        output_btn.pack(side=tk.LEFT)
        
        # --- Action Button ---
        self.hide_button = ttk.Button(frame, text="Hide Image", command=self.run_hide_process, style='Accent.TButton')
        self.hide_button.pack(pady=20, ipady=5, fill='x')

    def create_extract_widgets(self):
        """Builds the UI elements for the 'Extract' tab."""
        frame = self.extract_frame
        
        # --- Stego Image Input ---
        stego_in_frame = ttk.LabelFrame(frame, text="1. Stego Image (Contains Secret)", padding=10)
        stego_in_frame.pack(fill='x', expand=True, pady=5)
        
        stego_in_entry = ttk.Entry(stego_in_frame, textvariable=self.stego_input_path, width=60)
        stego_in_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        
        stego_in_btn = ttk.Button(stego_in_frame, text="Browse...", command=self.select_stego_input)
        stego_in_btn.pack(side=tk.LEFT)

        # --- Recovered Image Output ---
        extract_out_frame = ttk.LabelFrame(frame, text="2. Output for Recovered Image (Save as .png)", padding=10)
        extract_out_frame.pack(fill='x', expand=True, pady=5)
        
        extract_out_entry = ttk.Entry(extract_out_frame, textvariable=self.extract_output_path, width=60)
        extract_out_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        
        extract_out_btn = ttk.Button(extract_out_frame, text="Save As...", command=self.select_extract_output)
        extract_out_btn.pack(side=tk.LEFT)

        # --- Action Button ---
        self.extract_button = ttk.Button(frame, text="Extract Image", command=self.run_extract_process, style='Accent.TButton')
        self.extract_button.pack(pady=15, ipady=5, fill='x')

        # --- Separator ---
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=10)

        # --- Compare Section ---
        compare_frame = ttk.LabelFrame(frame, text="3. Quality Comparison (Optional)", padding=10)
        compare_frame.pack(fill='x', expand=True, pady=5)

        # --- Original Payload Input (for comparison) ---
        orig_payload_frame = ttk.Frame(compare_frame)
        orig_payload_frame.pack(fill='x', expand=True, pady=5)
        
        orig_payload_label = ttk.Label(orig_payload_frame, text="Original Payload:")
        orig_payload_label.pack(side=tk.LEFT, padx=(0, 5))
        
        orig_payload_entry = ttk.Entry(orig_payload_frame, textvariable=self.original_payload_compare_path, width=45)
        orig_payload_entry.pack(side=tk.LEFT, fill='x', expand=True, padx=(0, 5))
        
        orig_payload_btn = ttk.Button(orig_payload_frame, text="Browse...", command=self.select_original_for_compare)
        orig_payload_btn.pack(side=tk.LEFT)
        
        # --- Compare Button ---
        self.compare_button = ttk.Button(compare_frame, text="Compare Images", command=self.run_compare_process)
        self.compare_button.pack(pady=10, ipady=5, fill='x')
        if not NUMPY_AVAILABLE:
            self.compare_button.config(state='disabled', text="Compare (Requires NumPy)")


    # --- File Dialog Functions ---
    
    def select_cover(self):
        path = filedialog.askopenfilename(title="Select Cover Image",
                                          initialdir=self.script_dir,
                                          filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")])
        if path:
            self.cover_path.set(path)

    def select_payload(self):
        path = filedialog.askopenfilename(title="Select Payload Image",
                                          initialdir=self.script_dir,
                                          filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All Files", "*.*")])
        if path:
            self.payload_path.set(path)

    def select_hide_output(self):
        path = filedialog.asksaveasfilename(title="Save Stego Image As",
                                            initialdir=self.script_dir,
                                            defaultextension=".png",
                                            filetypes=[("PNG Image", "*.png")])
        if path:
            # Ensure it ends with .png
            if not path.lower().endswith('.png'):
                path += '.png'
            self.hide_output_path.set(path)

    def select_stego_input(self):
        path = filedialog.askopenfilename(title="Select Stego Image",
                                          initialdir=self.script_dir,
                                          filetypes=[("PNG Image", "*.png"), ("All Files", "*.*")])
        if path:
            self.stego_input_path.set(path)

    def select_extract_output(self):
        path = filedialog.asksaveasfilename(title="Save Recovered Image As",
                                            initialdir=self.script_dir,
                                            defaultextension=".png",
                                            filetypes=[("PNG Image", "*.png")])
        if path:
            # Ensure it ends with .png
            if not path.lower().endswith('.png'):
                path += '.png'
            self.extract_output_path.set(path)
            
    def select_original_for_compare(self):
        path = filedialog.askopenfilename(title="Select Original Payload (for comparison)",
                                          initialdir=self.script_dir,
                                          filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All Files", "*.*")])
        if path:
            self.original_payload_compare_path.set(path)

    # --- Process Functions ---

    def run_hide_process(self):
        cover = self.cover_path.get()
        payload = self.payload_path.get()
        output = self.hide_output_path.get()

        if not cover or not payload or not output:
            messagebox.showerror("Error", "All fields are required for hiding.")
            return
            
        self.log("--- Starting Hide Process ---")
        self.hide_button.config(state='disabled', text="Hiding...")
        
        # Run the core logic
        success = hide(cover, payload, output, self.log)
        
        self.hide_button.config(state='normal', text="Hide Image")
        
        if success:
            messagebox.showinfo("Success", "Image hidden successfully!")
        else:
            messagebox.showerror("Failure", "Failed to hide image. Check log for details.")

    def run_extract_process(self):
        stego = self.stego_input_path.get()
        output = self.extract_output_path.get()

        if not stego or not output:
            messagebox.showerror("Error", "All fields are required for extracting.")
            return

        self.log("--- Starting Extract Process ---")
        self.extract_button.config(state='disabled', text="Extracting...")
        
        # Run the core logic
        success = extract(stego, output, self.log)
        
        self.extract_button.config(state='normal', text="Extract Image")

        if success:
            messagebox.showinfo("Success", "Image extracted successfully!")
            # Auto-fill the "extracted" path in the compare section
            self.extract_output_path.set(output)
        else:
            messagebox.showerror("Failure", "Failed to extract image. Check log for details.")
            
    def run_compare_process(self):
        original = self.original_payload_compare_path.get()
        extracted = self.extract_output_path.get()

        if not original or not extracted:
            messagebox.showerror("Error", "Both 'Original Payload' and 'Extracted Output' paths are needed for comparison.")
            return

        self.compare_button.config(state='disabled', text="Comparing...")
        
        # Run the core logic
        compare_images(original, extracted, self.log)
        
        self.compare_button.config(state='normal', text="Compare Images")


# ==============================================================================
# --- MAIN EXECUTION ---
# ==============================================================================

if __name__ == '__main__':
    root = tk.Tk()
    
    # Add a simple style for the action buttons
    style = ttk.Style(root)
    style.configure('Accent.TButton', font=('Arial', 10, 'bold'), foreground='blue')

    app = SteganographyApp(root)
    root.mainloop()
