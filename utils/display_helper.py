"""
Display helper for showing emulator screen in real-time
Uses cv2 (OpenCV) to create a window
"""

import cv2
import numpy as np
from PIL import Image
import threading
import queue
import os

class GameDisplay:
    """Display game frames in a window (non-blocking, thread-safe)"""
    
    def __init__(self, title="Pokemon Emerald - RL Training", scale=3):
        self.title = title
        self.scale = scale
        self.window_created = False
        self.frame_queue = queue.Queue(maxsize=2)  # Keep only latest frame
        self.running = True
        self.display_thread = None
        
        # Handle Wayland/X11 display issues
        # OpenCV on Wayland can be problematic, try multiple approaches
        session_type = os.environ.get('XDG_SESSION_TYPE', '')
        if session_type == 'wayland':
            # Try multiple backends for OpenCV on Wayland
            # Option 1: Force X11 if available
            if 'DISPLAY' in os.environ:
                os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Force X11 for Qt-based backends
                print("🎬 Wayland detected, attempting to use X11 backend via DISPLAY")
            else:
                # Option 2: Try Wayland-native
                os.environ['QT_QPA_PLATFORM'] = 'wayland'
                print("🎬 Wayland detected, using Wayland-native backend")
        
        # Start display thread
        self._start_display_thread()
        
    def _start_display_thread(self):
        """Start the display thread that handles all OpenCV operations"""
        def display_loop():
            """Display loop running in separate thread"""
            while self.running:
                try:
                    # Get frame from queue (non-blocking, timeout after 0.1s)
                    try:
                        frame = self.frame_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue  # No frame available, continue loop
                    
                    # Process frame
                    if frame is None:  # Shutdown signal
                        break
                    
                    # Convert PIL Image to numpy if needed
                    if isinstance(frame, Image.Image):
                        frame = np.array(frame)
                    
                    # Check if frame has content
                    if frame is None or frame.size == 0:
                        continue
                    
                    # Ensure frame is 3-channel RGB
                    if len(frame.shape) == 2:
                        # Grayscale, convert to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    
                    # Convert RGB to BGR for OpenCV
                    if len(frame.shape) == 3 and frame.shape[2] == 3:
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = frame
                    
                    # Debug: Print frame info on first display
                    if not self.window_created:
                        print(f"🎬 Frame info: shape={frame_bgr.shape}, dtype={frame_bgr.dtype}, "
                              f"min={frame_bgr.min()}, max={frame_bgr.max()}")
                    
                    # Scale up for better visibility
                    if self.scale > 1:
                        height, width = frame_bgr.shape[:2]
                        new_width = width * self.scale
                        new_height = height * self.scale
                        frame_bgr = cv2.resize(frame_bgr, (new_width, new_height), 
                                              interpolation=cv2.INTER_NEAREST)
                    
                    # Create window on first call
                    if not self.window_created:
                        try:
                            # Try to create window with different flags for Wayland compatibility
                            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)
                            self.window_created = True
                            print(f"✅ Window created: {self.title}")
                        except Exception as e:
                            print(f"⚠️  Could not create window: {e}")
                            # Continue anyway, might work on next frame
                            continue
                    
                    # Display the frame (this is now in a separate thread, won't block main loop)
                    try:
                        cv2.imshow(self.title, frame_bgr)
                        # Use waitKey with very short timeout (non-blocking)
                        # On Wayland, this might still cause issues, but it's in a separate thread
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):  # Allow 'q' to quit
                            self.running = False
                            break
                    except Exception as e:
                        # If display fails, just log and continue (don't crash)
                        if "not responding" not in str(e).lower():
                            print(f"⚠️  Display update error: {e}")
                        continue
                    
                except Exception as e:
                    # Don't crash on display errors, just log and continue
                    print(f"⚠️  Display thread error: {e}")
                    continue
        
        self.display_thread = threading.Thread(target=display_loop, daemon=True)
        self.display_thread.start()
        print(f"🎬 Display thread started (non-blocking)")
        # Give thread a moment to start
        import time
        time.sleep(0.1)
    
    def show_frame(self, frame):
        """
        Display a frame in the window (non-blocking, thread-safe)
        
        Args:
            frame: PIL Image or numpy array (H, W, 3) RGB
        """
        if not self.running:
            return
        
        try:
            # Put frame in queue (non-blocking, drop old frames if queue is full)
            try:
                # Clear queue if it's full (keep only latest frame)
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                # Add new frame
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Queue full, skip this frame (non-critical)
        except Exception as e:
            # Don't crash on queue errors, but log first few for debugging
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            if self._error_count < 3:
                print(f"⚠️  Display queue error (will suppress): {e}")
                self._error_count += 1
    
    def close(self):
        """Close the display window and stop the display thread"""
        self.running = False
        try:
            # Signal thread to stop
            self.frame_queue.put_nowait(None)
        except:
            pass
        
        # Wait for thread to finish (with timeout)
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=1.0)
        
        try:
            if self.window_created:
                cv2.destroyWindow(self.title)
                cv2.destroyAllWindows()  # Clean up all windows
                self.window_created = False
        except:
            pass

# Global display instance
_display = None

def get_display(title="Pokemon Emerald - RL Training", scale=3):
    """Get or create the global display instance"""
    global _display
    if _display is None:
        _display = GameDisplay(title, scale)
    return _display

def show_frame(frame):
    """Convenience function to show a frame"""
    display = get_display()
    display.show_frame(frame)

def close_display():
    """Close the display"""
    global _display
    if _display is not None:
        _display.close()
        _display = None

