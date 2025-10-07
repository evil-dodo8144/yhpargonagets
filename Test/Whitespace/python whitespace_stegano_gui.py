import tkinter as tk
from tkinter import messagebox, simpledialog, scrolledtext
import json
import os

LOG_FILE = "stego_log2.json"

# --- Encoding/Decoding Functions ---
def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_whitespace(binary):
    return ''.join(' ' if bit == '0' else '\t' for bit in binary)

def whitespace_to_binary(whitespace):
    return ''.join('0' if ch == ' ' else '1' for ch in whitespace)

def binary_to_text(binary):
    return ''.join(chr(int(binary[i:i+8], 2)) for i in range(0, len(binary), 8))

def encode_message(cover, secret):
    binary = text_to_binary(secret)
    hidden = binary_to_whitespace(binary)
    return cover + hidden

def decode_message(stego_text):
    # Handle escaped tabs if pasted from repr()
    if "\\t" in stego_text or "\\n" in stego_text:
        stego_text = stego_text.encode().decode("unicode_escape")
    hidden_part = ''.join(ch for ch in stego_text if ch in (' ', '\t'))
    binary = whitespace_to_binary(hidden_part)
    return binary_to_text(binary)

# --- Log Utilities ---
def load_log():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, 'r') as f:
        return json.load(f)

def save_log(logs):
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=4)

def add_to_log(cover, secret, stego):
    logs = load_log()
    logs.append({"cover": cover, "secret": secret, "stego": stego})
    save_log(logs)

def get_hidden_by_cover(cover):
    logs = load_log()
    for entry in logs:
        if entry["cover"] == cover:
            return entry["secret"]
    return None

# --- GUI Setup ---
def encode_gui():
    cover = simpledialog.askstring("Cover Text", "Enter visible cover message:")
    secret = simpledialog.askstring("Secret Message", "Enter secret message to hide:")

    if cover and secret:
        stego = encode_message(cover, secret)
        add_to_log(cover, secret, stego)
        result_box.delete("1.0", tk.END)
        result_box.insert(tk.END, stego)
        messagebox.showinfo("Success", "Message encoded and saved to log!")

def decode_gui():
    stego = text_input.get("1.0", tk.END).strip()
    if not stego:
        messagebox.showwarning("Missing", "Paste or type the stego message!")
        return
    try:
        secret = decode_message(stego)
        result_box.delete("1.0", tk.END)
        result_box.insert(tk.END, secret)
    except:
        messagebox.showerror("Error", "Failed to decode! Make sure whitespace is valid.")

def retrieve_gui():
    cover = simpledialog.askstring("Retrieve", "Enter original cover message:")
    if cover:
        hidden = get_hidden_by_cover(cover)
        result_box.delete("1.0", tk.END)
        if hidden:
            result_box.insert(tk.END, hidden)
        else:
            result_box.insert(tk.END, "❌ No hidden message found for that cover.")

# --- Main Window ---
window = tk.Tk()
window.title("Whitespace Steganography 🔐")
window.geometry("700x500")

frame = tk.Frame(window)
frame.pack(pady=10)

tk.Button(frame, text="Encode Message", command=encode_gui, width=20).grid(row=0, column=0, padx=10)
tk.Button(frame, text="Decode Message", command=decode_gui, width=20).grid(row=0, column=1, padx=10)
tk.Button(frame, text="Retrieve from Log", command=retrieve_gui, width=20).grid(row=0, column=2, padx=10)

tk.Label(window, text="📥 Input / Paste Stego Message Below (use ␣ and ↹):").pack()
text_input = scrolledtext.ScrolledText(window, height=6, width=80)
text_input.pack(pady=5)

tk.Label(window, text="📤 Output (Result / Decoded Message):").pack()
result_box = scrolledtext.ScrolledText(window, height=8, width=80)
result_box.pack(pady=5)

window.mainloop()
