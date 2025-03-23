import os
import time
import numpy as np
import cv2
import fcntl
import errno

class ORBSLAMInterface:
    def __init__(self):
        self.control_pipe_path = "/tmp/orbslam_control_pipe"
        self.data_pipe_path = "/tmp/orbslam_data_pipe"
        self.control_pipe = None
        self.data_pipe = None
        self.initialized = False
        
    def initialize(self):
        """Initialize the pipes for communication with ORB-SLAM"""
        # Create pipes if they don't exist
        try:
            os.mkfifo(self.control_pipe_path, 0o666)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(f"Failed to create control pipe: {e}")
                return False
                
        try:
            os.mkfifo(self.data_pipe_path, 0o666)
        except OSError as e:
            if e.errno != errno.EEXIST:
                print(f"Failed to create data pipe: {e}")
                return False
        
        # Open pipes for writing (non-blocking for control)
        try:
            self.control_pipe = os.open(self.control_pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            # Data pipe is opened when needed to avoid blocking
        except OSError as e:
            print(f"Failed to open pipes: {e}")
            self.cleanup()
            return False
            
        self.initialized = True
        # return True
        retry_count = 0
        max_retries = 10
        while retry_count < max_retries:
            try:
                self.control_pipe = os.open(self.control_pipe_path, os.O_WRONLY | os.O_NONBLOCK)
                print("Successfully opened control pipe")
                
                # Try to open data pipe too (non-blocking initially to check availability)
                self.data_pipe = os.open(self.data_pipe_path, os.O_WRONLY | os.O_NONBLOCK)
                print("Successfully opened data pipe")
                
                # If we got here, both pipes are open
                self.initialized = True
                return True
            except OSError as e:
                print(f"Attempt {retry_count+1}: Failed to open pipes: {e}")
                retry_count += 1
                time.sleep(1)  # Wait before retrying
                
        print("Failed to connect to ORB-SLAM after multiple attempts")
        return False
        
    # def send_frame(self, image, timestamp):
    #     """Send a frame to ORB-SLAM through the pipes"""
    #     if not self.initialized and not self.initialize():
    #         return False
            
    #     # Convert image to grayscale if needed
    #     if len(image.shape) == 3:
    #         gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     else:
    #         gray_image = image
            
    #     # Prepare control message with image dimensions and timestamp
    #     height, width = gray_image.shape
    #     type_value = 0  # CV_8UC1
    #     control_message = f"{width} {height} {type_value} {timestamp}\n"
        
    #     try:
    #         # Send control message
    #         os.write(self.control_pipe, control_message.encode())
            
    #         # Open data pipe for writing (this might block until reader is ready)
    #         if self.data_pipe is None:
    #             self.data_pipe = os.open(self.data_pipe_path, os.O_WRONLY)
                
    #         # Send image data
    #         os.write(self.data_pipe, gray_image.tobytes())
    #         return True
    #     except OSError as e:
    #         if e.errno == errno.EPIPE:
    #             # Reader has closed the pipe
    #             print("ORB-SLAM has closed the connection")
    #             self.cleanup()
    #             return False
    #         elif e.errno == errno.EAGAIN:
    #             # No reader available yet
    #             return False
    #         else:
    #             print(f"Error sending frame: {e}")
    #             return False

    # def send_frame(self, image, timestamp):
    #     print("trying to send frame")
    #     if not self.initialized and not self.initialize():
    #         return False
            
    #     # Convert image to grayscale if needed
    #     if len(image.shape) == 3:
    #         gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     else:
    #         gray_image = image
            
    #     # Prepare control message with image dimensions and timestamp
    #     height, width = gray_image.shape
    #     type_value = 0  # CV_8UC1 for grayscale
    #     control_message = f"{width} {height} {type_value} {timestamp}"
        
    #     try:
    #         # Send control message first
    #         bytes_written = os.write(self.control_pipe, control_message.encode())
    #         print(f"Wrote {bytes_written} bytes to control pipe: {control_message}")
            
    #         # Brief pause to allow the reader to process the control message
    #         time.sleep(0.01)
            
    #         # Send image data
    #         img_bytes = gray_image.tobytes()
    #         bytes_written = os.write(self.data_pipe, img_bytes)
    #         print(f"Wrote {bytes_written} bytes of image data")
    #         return True
            
    #     except OSError as e:
    #         print(f"Error sending frame: {e}")
    #         if e.errno == errno.EPIPE:
    #             print("ORB-SLAM has closed the connection")
    #             self.cleanup()
    #         return False

    def test_send_frame(self, timestamp):
        if not self.initialized and not self.initialize():
            return False
        try:
            # Send a fixed message for testing
            os.write(self.control_pipe, b"100 100 0 123456.789\0")
            # Send a small dummy image (10000 bytes of zeros)
            os.write(self.data_pipe, bytes(10000))
            # Wait to ensure the control message is processed
            time.sleep(0.1)
            print("sent frame test at ", timestamp)
            return True
        except OSError as e:
            print(f"Error sending frame: {e}")
            return False
        
    def send_frame(self, image, timestamp):
        if not self.initialized and not self.initialize():
            return False
            
        # Convert image to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
            
        # Prepare control message with image dimensions and timestamp
        height, width = gray_image.shape
        type_value = 0  # CV_8UC1 for grayscale
        
        # Format exactly as the C++ code expects it
        control_message = f"{width} {height} {type_value} {timestamp}\n"
        
        print(f"Image dimensions: {width}x{height}, expected bytes: {width*height}")
        print(f"Sending control message: '{control_message}'")
        
        try:
            # Send control message
            bytes_written = os.write(self.control_pipe, control_message.encode())
            print(f"Wrote {bytes_written} bytes to control pipe")
            
            # Get the entire image data
            img_bytes = gray_image.tobytes()
            total_bytes = len(img_bytes)
            
            # Send the image data in chunks
            bytes_sent = 0
            chunk_size = 65536  # Maximum size that worked previously
            
            while bytes_sent < total_bytes:
                remaining = total_bytes - bytes_sent
                current_chunk_size = min(chunk_size, remaining)
                chunk = img_bytes[bytes_sent:bytes_sent + current_chunk_size]
                
                bytes_written = os.write(self.data_pipe, chunk)
                bytes_sent += bytes_written
                
                print(f"Sent chunk: {bytes_written} bytes, total sent: {bytes_sent}/{total_bytes}")
                
                # Small delay to prevent pipe overflow
                time.sleep(0.001)
            
            return True
        except OSError as e:
            print(f"Error sending frame: {e}")
            return False
    # def send_frame(self, image, timestamp):
    #     if not self.initialized and not self.initialize():
    #         return False
            
    #     # Convert image to grayscale if needed
    #     if len(image.shape) == 3:
    #         gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #     else:
    #         gray_image = image
            
    #     # Prepare control message with image dimensions and timestamp
    #     height, width = gray_image.shape
    #     type_value = 0  # CV_8UC1 for grayscale
        
    #     # Format exactly as the C++ code expects it, including null terminator
    #     control_message = f"{width} {height} {type_value} {timestamp}\n"
        
    #     print(f"Sending control message: '{control_message}'")
        
    #     try:
    #         # Send control message
    #         bytes_written = os.write(self.control_pipe, control_message.encode())
    #         print(f"Wrote {bytes_written} bytes to control pipe")
            
    #         # Wait to ensure the control message is processed
    #         time.sleep(0.1)
            
    #         # Send image data
    #         img_bytes = gray_image.tobytes()
    #         bytes_written = os.write(self.data_pipe, img_bytes)
    #         print(f"Wrote {bytes_written} bytes of image data")
            
    #         return True
    #     except OSError as e:
    #         print(f"Error sending frame: {e}")
    #         return False
        
    def cleanup(self):
        """Clean up resources"""
        if self.control_pipe is not None:
            try:
                os.close(self.control_pipe)
            except:
                pass
            self.control_pipe = None
            
        if self.data_pipe is not None:
            try:
                os.close(self.data_pipe)
            except:
                pass
            self.data_pipe = None
            
        self.initialized = False