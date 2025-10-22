import numpy as np
from PIL import Image, ImageSequence
import os

class GIFSteganography:
    def __init__(self):
        self.magic_header = b"GIFSTEG"  # 7-byte magic header
        
    def encode_message(self, gif_path, message, output_path):
        """Encode a message into a GIF file"""
        try:
            # Open the GIF and process each frame
            gif = Image.open(gif_path)
            frames = []
            for frame in ImageSequence.Iterator(gif):
                frames.append(frame.copy())
            
            # Convert message to binary with header and termination
            binary_msg = self._message_to_binary(message)
            
            # Check if message fits in the GIF
            total_pixels = sum(frame.size[0] * frame.size[1] for frame in frames)
            if len(binary_msg) > total_pixels * 3:  # 3 channels (RGB)
                raise ValueError("Message is too long for this GIF")
            
            # Encode the message in the frames
            encoded_frames = self._encode_in_frames(frames, binary_msg)
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save the new GIF
            encoded_frames[0].save(
                output_path,
                save_all=True,
                append_images=encoded_frames[1:],
                loop=0,  # Infinite loop
                duration=gif.info.get('duration', 100),
                disposal=2  # Restore to background
            )
            
            print(f"✅ Message successfully encoded in: {output_path}")
            
        except Exception as e:
            print(f"❌ Error encoding message: {e}")
    
    def decode_message(self, gif_path):
        """Decode a message from a GIF file"""
        try:
            # Open the GIF and process each frame
            gif = Image.open(gif_path)
            frames = []
            for frame in ImageSequence.Iterator(gif):
                frames.append(frame.copy())
            
            # Extract the binary message from the frames
            binary_msg = self._decode_from_frames(frames)
            
            # Convert binary to string
            message = self._binary_to_message(binary_msg)
            
            return message
            
        except Exception as e:
            print(f"❌ Error decoding message: {e}")
            return None
    
    def _message_to_binary(self, message):
        """Convert a message to binary with header and termination"""
        # Add magic header and message length
        length = len(message)
        header = self.magic_header + length.to_bytes(4, byteorder='big')
        
        # Convert to binary string
        binary_str = ''.join(format(byte, '08b') for byte in header)
        binary_str += ''.join(format(ord(char), '08b') for char in message)
        
        return binary_str
    
    def _binary_to_message(self, binary_str):
        """Convert binary string back to message"""
        # Extract magic header and verify
        header_bytes = []
        for i in range(0, 7*8, 8):
            byte = binary_str[i:i+8]
            header_bytes.append(int(byte, 2))
        
        if bytes(header_bytes) != self.magic_header:
            raise ValueError("No hidden message found or corrupted data")
        
        # Extract message length
        length_bytes = []
        for i in range(7*8, 11*8, 8):
            byte = binary_str[i:i+8]
            length_bytes.append(int(byte, 2))
        
        length = int.from_bytes(bytes(length_bytes), byteorder='big')
        
        # Extract message
        message = ""
        start_index = 11*8
        for i in range(start_index, start_index + length*8, 8):
            byte = binary_str[i:i+8]
            message += chr(int(byte, 2))
        
        return message
    
    def _encode_in_frames(self, frames, binary_msg):
        """Encode the binary message into the frames"""
        encoded_frames = []
        msg_index = 0
        msg_length = len(binary_msg)
        
        for frame in frames:
            # Convert frame to RGB if needed
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            
            img_array = np.array(frame)
            height, width, _ = img_array.shape
            
            # Encode message in this frame
            for y in range(height):
                for x in range(width):
                    if msg_index < msg_length:
                        # Modify the least significant bit of each color channel
                        for c in range(3):  # R, G, B channels
                            if msg_index < msg_length:
                                img_array[y, x, c] = (img_array[y, x, c] & 0xFE) | int(binary_msg[msg_index])
                                msg_index += 1
                    else:
                        break
                if msg_index >= msg_length:
                    break
            
            encoded_frames.append(Image.fromarray(img_array))
        
        return encoded_frames
    
    def _decode_from_frames(self, frames):
        """Extract the binary message from the frames"""
        binary_msg = ""
        found_end = False
        
        for frame in frames:
            if frame.mode != 'RGB':
                frame = frame.convert('RGB')
            
            img_array = np.array(frame)
            height, width, _ = img_array.shape
            
            for y in range(height):
                for x in range(width):
                    for c in range(3):
                        binary_msg += str(img_array[y, x, c] & 1)
                        
                        if len(binary_msg) >= 88:  # 11 bytes * 8 bits
                            try:
                                self._binary_to_message(binary_msg)
                                found_end = True
                            except:
                                continue
                    
                    if found_end:
                        break
                if found_end:
                    break
            if found_end:
                break
        
        if not found_end:
            raise ValueError("No valid message found in the GIF")
        
        return binary_msg


def main():
    steg = GIFSteganography()

    # ✅ Fixed paths for input and output
    input_path = r"D:\Debayan\yhpargonagets\model\input.gif"
    output_path = r"D:\Debayan\yhpargonagets\model\output.gif"

    # 👇 USER INPUT MESSAGE LINE
    message = input("Enter the message to encode inside GIF: ")

    steg.encode_message(input_path, message, output_path)
    
    # Optional decode check
    print("\nNow decoding to verify...")
    decoded_msg = steg.decode_message(output_path)
    if decoded_msg:
        print(f"🔍 Decoded message: {decoded_msg}")


if __name__ == "__main__":
    main()
