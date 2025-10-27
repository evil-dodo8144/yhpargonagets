import os
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
from PIL import Image, ImageChops, ImageStat, ImageTk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# --- Check for Optional Dependencies ---

try:
    from skimage.metrics import structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import imagehash
    IMAGEHASH_AVAILABLE = True
except ImportError:
    IMAGEHASH_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    # TensorFlow is a heavy dependency
    import tensorflow as tf
    from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
    from tensorflow.keras.preprocessing import image as tf_image
    from scipy.spatial.distance import cosine
    TENSORFLOW_AVAILABLE = True
    VGG_MODEL = None # We will load this once, globally
except ImportError:
    TENSORFLOW_AVAILABLE = False

# --- CORE ANALYSIS FUNCTIONS (Existing) ---

def calculate_metrics(im1_arr, im2_arr):
    """Calculates Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)."""
    # Convert to float64 to prevent overflow errors during calculation
    im1_arr = im1_arr.astype(np.float64)
    im2_arr = im2_arr.astype(np.float64)
    
    mse = np.mean((im1_arr - im2_arr) ** 2)
    
    if mse == 0:
        # Images are identical, PSNR is infinite
        psnr = float('inf')
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        
    return mse, psnr

def calculate_ssim(im1_arr, im2_arr):
    """Calculates the Structural Similarity Index (SSIM)."""
    if not SKIMAGE_AVAILABLE:
        return "N/A"
        
    try:
        # Use channel_axis=-1 for (H, W, C) shape (color images)
        # data_range is 255 for 8-bit images
        ssim_score = structural_similarity(im1_arr, im2_arr, channel_axis=-1, data_range=255)
        return ssim_score
    except ValueError:
        # This can happen if one image is grayscale and the other is color
        # For simplicity, we'll just return N/A
        return "N/A (Likely mode mismatch)"
    except Exception as e:
        return f"Error ({e})"

# --- NEW: Plotting functions now take an 'ax' argument to draw on ---

def plot_histograms_on_ax(ax, im, title):
    """Helper function to plot 3-channel (R, G, B) histograms on a given ax."""
    try:
        colors = ('r', 'g', 'b')
        channels = im.split()
        
        for i, color in enumerate(colors):
            hist_values, bin_edges = np.histogram(
                np.array(channels[i]).ravel(),
                bins=256,
                range=[0, 256]
            )
            ax.plot(bin_edges[0:-1], hist_values, color=color, alpha=0.7, label=f'{color.upper()} Channel')
        
        ax.set_title(title, fontsize=10)
        ax.set_xlim([0, 256])
        ax.set_ylabel("Pixel Count", fontsize=8)
        ax.set_xlabel("Pixel Value", fontsize=8)
        ax.legend(loc='upper right', prop={'size': 6})
        ax.tick_params(axis='both', which='major', labelsize=8)
    except Exception as e:
        ax.text(0.5, 0.5, f"Could not plot histogram:\n{e}", ha='center', va='center')

# --- NEW ANALYSIS FUNCTIONS ---

def calculate_perceptual_hash(im1, im2):
    """Calculates and compares perceptual hashes (pHash)."""
    if not IMAGEHASH_AVAILABLE:
        return "N/A", "N/A"
    
    try:
        hash1 = imagehash.phash(im1)
        hash2 = imagehash.phash(im2)
        hash_diff = hash1 - hash2 # Hamming distance
        return str(hash1), hash_diff
    except Exception as e:
        return f"Error ({e})", "Error"

def get_keypoint_match_image(im1_rgb, im2_rgb, log_callback):
    """Finds, matches, and returns an image of ORB keypoints."""
    if not CV2_AVAILABLE:
        log_callback("  OpenCV (cv2) not installed. Skipping keypoint matching.")
        return None, 0, 0
    
    try:
        im1_bgr = cv2.cvtColor(np.array(im1_rgb), cv2.COLOR_RGB2BGR)
        im2_bgr = cv2.cvtColor(np.array(im2_rgb), cv2.COLOR_RGB2BGR)
        im1_gray = cv2.cvtColor(im1_bgr, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(im2_bgr, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(im1_gray, None)
        kp2, des2 = orb.detectAndCompute(im2_gray, None)

        if des1 is None or des2 is None:
            log_callback("  Could not find descriptors for one or both images.")
            return None, 0, 0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good_matches = matches[:50]
        
        img_matches = cv2.drawMatches(im1_rgb, kp1, im2_rgb, kp2, good_matches, None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Convert OpenCV (Numpy) image back to PIL Image
        match_image_pil = Image.fromarray(img_matches)

        match_percent = (len(good_matches) / min(len(kp1), len(kp2))) * 100 if min(len(kp1), len(kp2)) > 0 else 0
        return match_image_pil, len(good_matches), match_percent

    except Exception as e:
        log_callback(f"  ERROR during keypoint matching: {e}")
        return None, 0, 0

def get_vgg_model(log_callback):
    """Loads the VGG16 model once and returns it."""
    global VGG_MODEL
    if VGG_MODEL is None and TENSORFLOW_AVAILABLE:
        try:
            log_callback("  Loading VGG16 AI model (this may take a moment)...")
            VGG_MODEL = VGG16(weights='imagenet', include_top=False, pooling='avg')
            log_callback("  VGG16 model loaded successfully.")
        except Exception as e:
            log_callback(f"  ERROR: Could not load VGG16 model: {e}")
            return None
    return VGG_MODEL

def get_image_embedding(model, img_pil):
    """Pre-processes an image and gets its VGG16 embedding."""
    img = img_pil.resize((224, 224))
    x = tf_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    embedding = model.predict(x, verbose=0)
    return embedding.flatten()

def calculate_semantic_similarity(im1, im2, log_callback):
    """Calculates semantic similarity using VGG16 and Cosine Similarity."""
    if not TENSORFLOW_AVAILABLE:
        return "N/A (TensorFlow not installed)"
        
    model = get_vgg_model(log_callback)
    if model is None:
        return "N/A (Model failed to load)"
        
    try:
        log_callback("  Calculating image embeddings...")
        emb1 = get_image_embedding(model, im1)
        emb2 = get_image_embedding(model, im2)
        
        cosine_dist = cosine(emb1, emb2)
        cosine_sim = 1 - cosine_dist
        
        return f"{cosine_sim:.4f} (1.0 is identical)"
    except Exception as e:
        log_callback(f"  ERROR during embedding calculation: {e}")
        return f"Error ({e})"

# --- MAIN ANALYSIS FUNCTION (Refactored to return data) ---

def analyze_images(path1, path2, log_callback):
    """
    Performs all comparisons and returns a dictionary of results.
    This function no longer creates any plots.
    """
    
    results = {
        "metadata": {}, "stats": {}, "metrics": {}, "hash": {}, 
        "semantic": {}, "keypoints": {}, "images": {}, "identity": {}
    }
    
    log_callback(f"--- Image Comparison Analysis ---")
    log_callback(f"Image 1: {os.path.basename(path1)}")
    log_callback(f"Image 2: {os.path.basename(path2)}")
    log_callback("-" * 30)

    try:
        im1 = Image.open(path1)
        im2 = Image.open(path2)
        results["images"]["original_im1"] = im1
        results["images"]["original_im2"] = im2
    except Exception as e:
        log_callback(f"ERROR: Could not open images. {e}")
        return None

    # --- 1. Metadata Analysis ---
    log_callback("[1. METADATA ANALYSIS]")
    meta1 = f"Format: {im1.format}, Size: {im1.size}, Mode: {im1.mode}"
    meta2 = f"Format: {im2.format}, Size: {im2.size}, Mode: {im2.mode}"
    log_callback(f"  Img 1: {meta1}")
    log_callback(f"  Img 2: {meta2}")
    results["metadata"]["im1"] = meta1
    results["metadata"]["im2"] = meta2

    # --- 2. Pre-processing for Comparison ---
    if im1.size != im2.size:
        log_callback("  WARNING: Sizes differ. Resizing Image 2 to match Image 1 for metric analysis.")
        im2_resized = im2.resize(im1.size, Image.LANCZOS)
    else:
        im2_resized = im2
        
    im1_rgb = im1.convert('RGB')
    im2_rgb = im2_resized.convert('RGB')
    
    results["images"]["im1_rgb"] = im1_rgb
    results["images"]["im2_rgb"] = im2_rgb
    
    # --- 3. Pixel Identity Check ---
    log_callback("\n[2. PIXEL IDENTITY CHECK (on processed images)]")
    diff = ImageChops.difference(im1_rgb, im2_rgb)
    diff_enhanced = ImageChops.multiply(diff, Image.new('RGB', diff.size, (50, 50, 50)))
    results["images"]["diff_enhanced"] = diff_enhanced
    
    if not diff.getbbox():
        log_callback("  RESULT: Images are PIXEL-PERFECT IDENTICAL.")
        results["identity"]["identical"] = True
    else:
        log_callback("  RESULT: Images are DIFFERENT.")
        results["identity"]["identical"] = False

    # --- 4. Statistical & Numerical Analysis ---
    log_callback("\n[3. STATISTICAL & METRIC ANALYSIS]")
    arr1 = np.array(im1_rgb)
    arr2 = np.array(im2_rgb)
    stat1 = ImageStat.Stat(im1_rgb)
    stat2 = ImageStat.Stat(im2_rgb)
    
    results["stats"]["im1"] = {
        "Min/Max": stat1.extrema,
        "Mean": [round(m, 2) for m in stat1.mean],
        "Median": [round(m, 2) for m in stat1.median],
        "StdDev": [round(s, 2) for s in stat1.stddev],
        "Variance": [round(v, 2) for v in stat1.var],
        "RMS": [round(r, 2) for r in stat1.rms]
    }
    results["stats"]["im2"] = {
        "Min/Max": stat2.extrema,
        "Mean": [round(m, 2) for m in stat2.mean],
        "Median": [round(m, 2) for m in stat2.median],
        "StdDev": [round(s, 2) for s in stat2.stddev],
        "Variance": [round(v, 2) for v in stat2.var],
        "RMS": [round(r, 2) for r in stat2.rms]
    }

    # MSE & PSNR
    mse, psnr = calculate_metrics(arr1, arr2)
    results["metrics"]["mse"] = f"{mse:.4f}"
    results["metrics"]["psnr"] = f"{psnr:.4f} dB"
    log_callback(f"    Mean Squared Error (MSE): {results['metrics']['mse']}")
    log_callback(f"    Peak Signal-to-Noise Ratio (PSNR): {results['metrics']['psnr']}")

    # SSIM
    ssim_score = calculate_ssim(arr1, arr2)
    results["metrics"]["ssim"] = str(ssim_score)
    log_callback(f"    Structural Similarity (SSIM): {ssim_score}")
    
    # --- 5. Perceptual Hash Analysis ---
    log_callback("\n[4. PERCEPTUAL HASH ANALYSIS]")
    hash1, hash_diff = calculate_perceptual_hash(im1_rgb, im2_rgb)
    results["hash"]["hash1"] = hash1
    results["hash"]["hash_diff"] = str(hash_diff)
    log_callback(f"  Img 1 pHash: {hash1}")
    log_callback(f"  Hash Distance: {hash_diff} (0-1 is near-identical)")

    # --- 6. Semantic Similarity Analysis ---
    log_callback("\n[5. SEMANTIC (AI) SIMILARITY]")
    semantic_sim = calculate_semantic_similarity(im1, im2, log_callback)
    results["semantic"]["similarity"] = semantic_sim
    log_callback(f"  VGG16 Cosine Similarity: {semantic_sim}")
    
    # --- 7. Keypoint Feature Analysis ---
    log_callback("\n[6. KEYPOINT FEATURE ANALYSIS]")
    # Use original, un-resized images for keypoint matching
    match_image, num_matches, match_pct = get_keypoint_match_image(
        im1.convert('RGB'), im2.convert('RGB'), log_callback
    )
    results["images"]["keypoint_match"] = match_image
    results["keypoints"]["matches"] = num_matches
    results["keypoints"]["match_pct"] = f"{match_pct:.2f}%"
    log_callback(f"  Good Matches Found: {num_matches}")
    log_callback(f"  Match Percentage: {match_pct:.2f}%")
    
    log_callback("\n--- Analysis Complete ---")
    return results


# --- GUI APPLICATION CLASS (Redesigned) ---

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Comparison Analyzer (Full Suite)")
        self.root.geometry("900x750") # Larger window

        self.path1 = tk.StringVar()
        self.path2 = tk.StringVar()
        
        # Store PhotoImage objects to prevent garbage collection
        self.image_references = []

        # --- Top Frame for Controls ---
        top_frame = ttk.Frame(root, padding=10)
        top_frame.pack(fill='x', side='top')

        frame1 = self.create_file_frame(top_frame, "Image 1 (Original)", self.path1)
        frame1.pack(fill='x', pady=5)
        
        frame2 = self.create_file_frame(top_frame, "Image 2 (Modified/Extracted)", self.path2)
        frame2.pack(fill='x', pady=5)

        self.run_button = ttk.Button(top_frame, text="Run Full Analysis", command=self.run_analysis)
        self.run_button.pack(fill='x', ipady=5, pady=10)

        # --- Bottom Paned Window for Results and Log ---
        main_paned_window = ttk.PanedWindow(root, orient='vertical')
        main_paned_window.pack(fill='both', expand=True, padx=10, pady=(0, 10))

        # --- Results Notebook ---
        self.results_notebook = ttk.Notebook(main_paned_window)
        main_paned_window.add(self.results_notebook, weight=3) # Give more space to results

        # Tab 1: Statistics
        self.stats_frame = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.stats_frame, text="Statistics (Side-by-Side)")
        self.create_stats_tab()

        # Tab 2: Histograms
        self.histogram_frame = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.histogram_frame, text="Histograms (Side-by-Side)")
        self.create_histogram_tab()

        # Tab 3: Visuals
        self.visual_frame = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.visual_frame, text="Visual Analysis")
        self.create_visual_tab()

        # --- Log Area ---
        log_frame = ttk.LabelFrame(main_paned_window, text="Log & Dependency Status", padding=10)
        main_paned_window.add(log_frame, weight=1) # Less space for log
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=10, state='disabled', wrap=tk.WORD)
        self.log_text.pack(fill='both', expand=True)
        
        self.check_dependencies() # Log status on startup

    def create_file_frame(self, parent, label_text, string_var):
        """Helper to create a file selection row."""
        frame = ttk.LabelFrame(parent, text=label_text)
        
        entry = ttk.Entry(frame, textvariable=string_var)
        entry.pack(side=tk.LEFT, fill='x', expand=True, padx=5, pady=5)
        
        button = ttk.Button(frame, text="Browse...", 
                            command=lambda: self.select_file(string_var, label_text))
        button.pack(side=tk.LEFT, padx=5, pady=5)
        
        return frame

    def create_stats_tab(self):
        """Creates the Treeview widget for side-by-side number comparison."""
        cols = ('metric', 'image1', 'image2')
        self.stats_tree = ttk.Treeview(self.stats_frame, columns=cols, show='headings')
        self.stats_tree.heading('metric', text='Metric')
        self.stats_tree.heading('image1', text='Image 1')
        self.stats_tree.heading('image2', text='Image 2')
        
        self.stats_tree.column('metric', width=150, anchor='w')
        self.stats_tree.column('image1', anchor='w')
        self.stats_tree.column('image2', anchor='w')
        
        self.stats_tree.pack(fill='both', expand=True)

    def create_histogram_tab(self):
        """Creates the embedded Matplotlib canvas for side-by-side graphs."""
        # We create the Figure and Axes *once* and store them
        self.hist_fig, (self.hist_ax1, self.hist_ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.hist_canvas = FigureCanvasTkAgg(self.hist_fig, master=self.histogram_frame)
        self.hist_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.hist_fig.tight_layout()

    def create_visual_tab(self):
        """Creates the labels that will hold the comparison images."""
        self.visual_frame.rowconfigure(0, weight=1)
        self.visual_frame.rowconfigure(1, weight=1)
        self.visual_frame.columnconfigure(0, weight=1)
        self.visual_frame.columnconfigure(1, weight=1)

        self.img1_label = self.create_image_label(self.visual_frame, "Image 1")
        self.img1_label.grid(row=0, column=0, padx=5, pady=5, sticky='nsew')
        
        self.img2_label = self.create_image_label(self.visual_frame, "Image 2 (Processed)")
        self.img2_label.grid(row=0, column=1, padx=5, pady=5, sticky='nsew')
        
        self.diff_label = self.create_image_label(self.visual_frame, "Enhanced Difference")
        self.diff_label.grid(row=1, column=0, padx=5, pady=5, sticky='nsew')
        
        self.keypoint_label = self.create_image_label(self.visual_frame, "Keypoint Matches")
        self.keypoint_label.grid(row=1, column=1, padx=5, pady=5, sticky='nsew')

    def create_image_label(self, parent, text):
        """Helper to create a bordered label for images."""
        frame = ttk.Frame(parent, borderwidth=1, relief='solid')
        label = ttk.Label(frame, text=text, anchor='center')
        label.pack(fill='both', expand=True)
        return label

    def select_file(self, string_var, title):
        """Opens a file dialog and sets the path."""
        filetypes = [("Image Files", "*.png *.jpg *.jpeg *.bmp *.webp *.tiff"), ("All Files", "*.*")]
        path = filedialog.askopenfilename(title=f"Select {title}", filetypes=filetypes)
        if path:
            string_var.set(path)

    def log(self, message):
        """Appends a message to the log text box."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + '\n')
        self.log_text.see(tk.END) # Auto-scroll
        self.log_text.config(state='disabled')
        self.root.update_idletasks() # Force GUI to update

    def check_dependencies(self):
        """Checks for optional libraries and logs their status."""
        self.log("--- Dependency Check ---")
        self.log(f"Pillow (PIL): FOUND (Required)")
        self.log(f"NumPy: FOUND (Required)")
        self.log(f"Matplotlib: FOUND (Required)")
        
        if SKIMAGE_AVAILABLE: self.log("scikit-image: FOUND (SSIM enabled)")
        else: self.log("WARNING: scikit-image not found. SSIM will be N/A. (pip install scikit-image)")
            
        if IMAGEHASH_AVAILABLE: self.log("imagehash: FOUND (Perceptual Hash enabled)")
        else: self.log("WARNING: imagehash not found. Hashing will be N/A. (pip install imagehash)")
        
        if CV2_AVAILABLE: self.log("OpenCV (cv2): FOUND (Keypoint Matching enabled)")
        else: self.log("WARNING: opencv-python not found. Keypoints will be N/A. (pip install opencv-python)")
            
        if TENSORFLOW_AVAILABLE: self.log("TensorFlow: FOUND (Semantic AI Similarity enabled)")
        else: self.log("WARNING: tensorflow not found. AI similarity will be N/A. (pip install tensorflow)")
        
        self.log("-" * 30)

    def run_analysis(self):
        """Validates paths and starts the analysis."""
        p1 = self.path1.get()
        p2 = self.path2.get()

        if not p1 or not p2:
            tk.messagebox.showerror("Error", "Please select both images first.")
            return
            
        # Clear log and old results
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        self.check_dependencies()
        self.clear_results()
        
        self.run_button.config(text="Analyzing... Please wait.", state="disabled")
        
        try:
            # Run analysis
            results = analyze_images(p1, p2, self.log)
            
            # Populate GUI with results
            if results:
                self.populate_stats_tab(results)
                self.populate_histogram_tab(results)
                self.populate_visual_tab(results)
                tk.messagebox.showinfo("Success", "Analysis complete. Check the tabs for results.")
                self.results_notebook.select(0) # Switch to first tab
            else:
                tk.messagebox.showerror("Error", "Analysis failed. Check the log for details.")
                
        except Exception as e:
            self.log(f"FATAL ERROR: An unexpected error occurred: {e}")
            tk.messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}")
        finally:
            self.run_button.config(text="Run Full Analysis", state="normal")

    def clear_results(self):
        """Clears all old data from the results tabs."""
        # Clear stats tree
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
            
        # Clear histograms
        self.hist_ax1.clear()
        self.hist_ax2.clear()
        self.hist_ax1.set_title("Image 1 Histogram")
        self.hist_ax2.set_title("Image 2 Histogram")
        self.hist_canvas.draw()
        
        # Clear visual labels
        self.image_references.clear() # Clear old PhotoImage objects
        self.img1_label.config(image='', text='Image 1')
        self.img2_label.config(image='', text='Image 2 (Processed)')
        self.diff_label.config(image='', text='Enhanced Difference')
        self.keypoint_label.config(image='', text='Keypoint Matches')

    def populate_stats_tab(self, results):
        """Fills the stats Treeview with side-by-side data."""
        # --- Metadata ---
        self.stats_tree.insert('', 'end', values=('Metadata', results['metadata']['im1'], results['metadata']['im2']))
        
        # --- Identity ---
        identity_str = "PIXEL-PERFECT IDENTICAL" if results['identity']['identical'] else "Images are Different"
        self.stats_tree.insert('', 'end', values=('Pixel Identity', identity_str, ''))
        
        # --- Comparative Metrics ---
        self.stats_tree.insert('', 'end', values=('-COMPARATIVE METRICS-', '', ''))
        self.stats_tree.insert('', 'end', values=('MSE', results['metrics']['mse'], ''))
        self.stats_tree.insert('', 'end', values=('PSNR', results['metrics']['psnr'], ''))
        self.stats_tree.insert('', 'end', values=('SSIM', results['metrics']['ssim'], ''))
        
        # --- Hash ---
        self.stats_tree.insert('', 'end', values=('-PERCEPTUAL HASH-', '', ''))
        self.stats_tree.insert('', 'end', values=('pHash', results['hash']['hash1'], ''))
        self.stats_tree.insert('', 'end', values=('Hash Distance', results['hash']['hash_diff'], '(0-1 is near-identical)'))
        
        # --- Keypoints ---
        self.stats_tree.insert('', 'end', values=('-KEYPOINT FEATURES-', '', ''))
        self.stats_tree.insert('', 'end', values=('Good Matches', results['keypoints']['matches'], ''))
        self.stats_tree.insert('', 'end', values=('Match %', results['keypoints']['match_pct'], ''))
        
        # --- Semantic ---
        self.stats_tree.insert('', 'end', values=('-SEMANTIC (AI) SIMILARITY-', '', ''))
        self.stats_tree.insert('', 'end', values=('VGG16 Similarity', results['semantic']['similarity'], ''))

        # --- Statistics ---
        self.stats_tree.insert('', 'end', values=('-STATISTICS (per-channel)-', '', ''))
        for metric in results['stats']['im1']:
            val1 = str(results['stats']['im1'][metric])
            val2 = str(results['stats']['im2'][metric])
            self.stats_tree.insert('', 'end', values=(metric, val1, val2))

    def populate_histogram_tab(self, results):
        """Draws the new histograms on the embedded canvas."""
        # Clear old plots
        self.hist_ax1.clear()
        self.hist_ax2.clear()
        
        # Plot new data
        plot_histograms_on_ax(self.hist_ax1, results['images']['im1_rgb'], "Image 1 Histogram")
        plot_histograms_on_ax(self.hist_ax2, results['images']['im2_rgb'], "Image 2 (Processed) Histogram")
        
        # Redraw the canvas
        self.hist_fig.tight_layout()
        self.hist_canvas.draw()
        
    def populate_visual_tab(self, results):
        """Displays the four analysis images."""
        self.image_references.clear() # Clear old PhotoImage objects
        
        # Update Image 1
        self.update_image_on_label(self.img1_label, results['images']['im1_rgb'])
        
        # Update Image 2
        self.update_image_on_label(self.img2_label, results['images']['im2_rgb'])

        # Update Diff Image
        self.update_image_on_label(self.diff_label, results['images']['diff_enhanced'])
        
        # Update Keypoint Image
        if results['images']['keypoint_match']:
            self.update_image_on_label(self.keypoint_label, results['images']['keypoint_match'])
        else:
            self.keypoint_label.config(image='', text='Keypoint Matches (N/A)')

    def update_image_on_label(self, label, pil_image):
        """Resizes and displays a PIL image on a Tkinter Label."""
        # Get label size
        label.update_idletasks() # Ensure size is calculated
        w = label.winfo_width() - 10 # Subtract padding
        h = label.winfo_height() - 10
        
        if w <= 1 or h <= 1:
             w, h = 300, 300 # Default fallback
        
        # Resize image to fit label
        img_copy = pil_image.copy()
        img_copy.thumbnail((w, h), Image.LANCZOS)
        
        # Convert to PhotoImage and display
        photo = ImageTk.PhotoImage(img_copy)
        label.config(image=photo, text='')
        
        # Store reference
        self.image_references.append(photo)


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
