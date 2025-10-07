import tkinter as tk
from tkinter import scrolledtext, messagebox

# --- Steganography core functions ---
def text_to_binary(text):
    return ''.join(format(ord(c), '08b') for c in text)

def binary_to_whitespace(binary):
    return ''.join(' ' if bit == '0' else '\t' for bit in binary)

def whitespace_to_binary(whitespace):
    return ''.join('0' if ch == ' ' else '1' for ch in whitespace)

def binary_to_text(binary):
    chars = [binary[i:i+8] for i in range(0, len(binary), 8)]
    return ''.join(chr(int(char, 2)) for char in chars)

def encode_message(cover, secret):
    binary = text_to_binary(secret)
    hidden = binary_to_whitespace(binary)
    return cover + "\n[hidden]\n" + hidden + "\n[end]"

def decode_message(stego_text):
    if "[hidden]" not in stego_text or "[end]" not in stego_text:
        return "❌ Missing [hidden] or [end] marker."

    try:
        hidden_part = stego_text.split("[hidden]", 1)[1].split("[end]", 1)[0]
        whitespace_only = ''.join(ch for ch in hidden_part if ch in (' ', '\t'))

        if not whitespace_only:
            return "⚠️ No space/tab hidden message found."

        binary = whitespace_to_binary(whitespace_only)
        if len(binary) % 8 != 0:
            return f"⚠️ Corrupted message: binary length not multiple of 8."

        return binary_to_text(binary)
    except Exception as e:
        return f"❌ Error during decoding: {e}"

# --- GUI Logic ---
def show_main_menu():
    clear_window()

    tk.Label(root, text="Whitespace Steganography", font=("Arial", 18, "bold")).pack(pady=20)
    tk.Button(root, text="Encode Message", width=30, height=2, command=show_encode_screen).pack(pady=10)
    tk.Button(root, text="Decode Message", width=30, height=2, command=show_decode_screen).pack(pady=10)

def show_encode_screen():
    clear_window()

    tk.Label(root, text="Encode Message", font=("Arial", 16, "bold")).pack(pady=10)

    tk.Label(root, text="Cover Text:").pack()
    global cover_entry
    cover_entry = tk.Entry(root, width=80)
    cover_entry.pack()

    tk.Label(root, text="Secret Message:").pack(pady=(10, 0))
    global secret_entry
    secret_entry = tk.Entry(root, width=80)
    secret_entry.pack()

    tk.Button(root, text="Encode", command=perform_encoding).pack(pady=10)

    global output_text
    output_text = scrolledtext.ScrolledText(root, width=72, height=10, wrap=tk.WORD)
    output_text.pack()

    tk.Button(root, text="Copy Encoded Message", command=copy_to_clipboard).pack(pady=5)
    tk.Button(root, text="⬅ Back", command=show_main_menu).pack(pady=(10, 0))

def show_decode_screen():
    clear_window()

    tk.Label(root, text="Decode Message", font=("Arial", 16, "bold")).pack(pady=10)
    tk.Label(root, text="Paste the encoded message (with [hidden] ... [end])").pack()

    global output_text
    output_text = scrolledtext.ScrolledText(root, width=72, height=12, wrap=tk.WORD)
    output_text.pack()

    tk.Button(root, text="Decode", command=perform_decoding).pack(pady=10)
    tk.Button(root, text="⬅ Back", command=show_main_menu).pack()

def perform_encoding():
    cover = cover_entry.get()
    secret = secret_entry.get()
    if not cover or not secret:
        messagebox.showwarning("Missing Input", "Enter both cover and secret messages.")
        return
    stego = encode_message(cover, secret)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, stego)

def perform_decoding():
    stego = output_text.get(1.0, tk.END)
    result = decode_message(stego)
    messagebox.showinfo("Decoded Message", result)

def copy_to_clipboard():
    text = output_text.get(1.0, tk.END)
    if not text.strip():
        messagebox.showwarning("Empty", "Nothing to copy.")
        return
    root.clipboard_clear()
    root.clipboard_append(text)
    messagebox.showinfo("Copied", "Encoded message copied to clipboard.")

def clear_window():
    for widget in root.winfo_children():
        widget.destroy()

# --- GUI Setup ---
root = tk.Tk()
root.title("Whitespace Steganography")
root.geometry("650x500")
root.resizable(False, False)

show_main_menu()
root.mainloop()
