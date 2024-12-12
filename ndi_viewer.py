import os
import sys
import tkinter as tk
from tkinter import messagebox
import ctypes
from ctypes import *
from enum import IntEnum
import logging
import traceback
import numpy as np
from PIL import Image, ImageTk
import cv2
import threading
import queue
from threading import Lock, Event
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Constants
FRAME_QUEUE_SIZE = 1  # Reduced queue size
CAPTURE_TIMEOUT_MS = 500
DISPLAY_INTERVAL_MS = 100  # 10 FPS default
TARGET_FPS = 10  # Target FPS for capture
MAX_RECONNECT_ATTEMPTS = 3
MAX_FRAME_WIDTH = 640  # Reduced default resolution
MAX_FRAME_HEIGHT = 480
MIN_FRAME_INTERVAL = 1.0 / TARGET_FPS  # Minimum time between frames

# Use CPU processing only
USE_CUDA = False
logger.info("Using CPU processing only")

def resize_frame(frame, max_width=MAX_FRAME_WIDTH, max_height=MAX_FRAME_HEIGHT):
    """Resize frame if it exceeds maximum dimensions"""
    try:
        height, width = frame.shape[:2]
        if width > max_width or height > max_height:
            ratio = min(max_width/width, max_height/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return frame
    except Exception as e:
        logger.error(f"Error in resize_frame: {e}")
        return frame

def process_frame(frame):
    """Process frame with basic optimizations"""
    try:
        if frame is None or frame.size == 0:
            logger.warning("Received empty frame")
            return None
            
        if len(frame.shape) < 2:
            logger.warning(f"Invalid frame shape: {frame.shape}")
            return None
            
        if len(frame.shape) == 3 and frame.shape[2] == 4:  # RGBA
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame
    except Exception as e:
        logger.error(f"Error in process_frame: {e}")
        return None

def process_frame_gpu(frame):
    """Process frame using GPU acceleration"""
    if not USE_CUDA:
        return frame
        
    try:
        # Upload to GPU
        gpu_frame = cv2.cuda_GpuMat()
        gpu_frame.upload(frame)
        
        # Color conversion on GPU
        if frame.shape[2] == 4:  # RGBA
            gpu_bgr = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_RGBA2BGR)
        else:  # Already BGR
            gpu_bgr = gpu_frame
            
        # Additional GPU processing can be added here
        
        # Download result
        return gpu_bgr.download()
    except Exception as e:
        logger.error(f"GPU processing error: {e}")
        return frame

def load_ndi_library():
    """Load NDI library with proper error handling"""
    try:
        if sys.platform == 'darwin':
            # First try the local directory
            local_path = os.path.join(os.path.dirname(__file__), 'libndi_advanced.dylib')
            if os.path.exists(local_path):
                logger.info(f"Found NDI library at: {local_path}")
                return CDLL(local_path)
                
            # Try other common locations as fallback
            ndi_paths = [
                '/usr/local/lib/libndi_advanced.dylib',
                '/usr/local/lib/libndi.4.dylib',
                '/usr/local/lib/libndi.dylib',
                '/Library/NDI SDK for Apple/lib/x64/libndi_advanced.dylib',
                '/Applications/NDI Video Monitor.app/Contents/Frameworks/libndi_advanced.dylib'
            ]
            
            for path in ndi_paths:
                try:
                    if os.path.exists(path):
                        logger.info(f"Found NDI library at: {path}")
                        return CDLL(path)
                except Exception as e:
                    logger.warning(f"Failed to load NDI from {path}: {e}")
                    continue
                    
            error_msg = """
            Could not find NDI library. Please ensure the library file exists in:
            - The same directory as this script
            - /usr/local/lib/
            - /Library/NDI SDK for Apple/lib/x64/
            - /Applications/NDI Video Monitor.app/Contents/Frameworks/
            """
            logger.error(error_msg)
            messagebox.showerror("NDI Library Not Found", error_msg)
            sys.exit(1)
        else:
            error_msg = "Only macOS is supported for now"
            logger.error(error_msg)
            messagebox.showerror("Unsupported Platform", error_msg)
            sys.exit(1)
            
    except Exception as e:
        error_msg = f"Error loading NDI library: {e}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        messagebox.showerror("NDI Error", error_msg)
        sys.exit(1)

# NDI structures
class NDIlib_source_t(Structure):
    _fields_ = [
        ("p_ndi_name", c_char_p),
        ("p_url_address", c_char_p)
    ]

class NDIlib_find_create_t(Structure):
    _fields_ = [
        ("show_local_sources", c_bool),
        ("p_groups", c_char_p),
        ("p_extra_ips", c_char_p)
    ]

class NDIlib_recv_color_format_e(IntEnum):
    NDIlib_recv_color_format_BGRX_BGRA = 0
    NDIlib_recv_color_format_UYVY_BGRA = 1
    NDIlib_recv_color_format_RGBX_RGBA = 2
    NDIlib_recv_color_format_UYVY_RGBA = 3
    NDIlib_recv_color_format_fastest = 100
    NDIlib_recv_color_format_best = 101

class NDIlib_frame_type_e(IntEnum):
    NDIlib_frame_type_none = 0
    NDIlib_frame_type_video = 1
    NDIlib_frame_type_audio = 2
    NDIlib_frame_type_metadata = 3
    NDIlib_frame_type_error = 4
    NDIlib_frame_type_status_change = 100

class NDIlib_recv_bandwidth_e(IntEnum):
    NDIlib_recv_bandwidth_metadata_only = -10
    NDIlib_recv_bandwidth_audio_only = 10
    NDIlib_recv_bandwidth_lowest = 0
    NDIlib_recv_bandwidth_highest = 100

class NDIlib_recv_create_v3_t(Structure):
    _fields_ = [
        ("source_to_connect_to", NDIlib_source_t),
        ("color_format", c_int),
        ("bandwidth", c_int),
        ("allow_video_fields", c_bool),
        ("p_ndi_recv_name", c_char_p)
    ]

class NDIlib_video_frame_v2_t(Structure):
    _fields_ = [
        ("xres", c_int),
        ("yres", c_int),
        ("FourCC", c_int),
        ("frame_rate_N", c_int),
        ("frame_rate_D", c_int),
        ("picture_aspect_ratio", c_float),
        ("frame_format_type", c_int),
        ("timecode", c_longlong),
        ("p_data", POINTER(c_ubyte)),
        ("line_stride_in_bytes", c_int),
        ("metadata", c_char_p),
        ("timestamp", c_longlong)
    ]

class NDIlib_tally_t(Structure):
    _fields_ = [
        ("on_program", c_bool),
        ("on_preview", c_bool)
    ]

def setup_ndi_functions(ndi_lib):
    """Setup NDI function definitions"""
    ndi_lib.NDIlib_initialize.restype = c_bool
    
    ndi_lib.NDIlib_find_create_v2.argtypes = [POINTER(NDIlib_find_create_t)]
    ndi_lib.NDIlib_find_create_v2.restype = c_void_p
    
    ndi_lib.NDIlib_find_wait_for_sources.argtypes = [c_void_p, c_uint32]
    ndi_lib.NDIlib_find_wait_for_sources.restype = c_bool
    
    ndi_lib.NDIlib_find_get_current_sources.argtypes = [c_void_p, POINTER(c_uint32)]
    ndi_lib.NDIlib_find_get_current_sources.restype = POINTER(NDIlib_source_t)
    
    ndi_lib.NDIlib_recv_create_v3.argtypes = [POINTER(NDIlib_recv_create_v3_t)]
    ndi_lib.NDIlib_recv_create_v3.restype = c_void_p
    
    ndi_lib.NDIlib_recv_connect.argtypes = [c_void_p, POINTER(NDIlib_source_t)]
    ndi_lib.NDIlib_recv_connect.restype = c_bool
    
    ndi_lib.NDIlib_recv_capture_v2.argtypes = [c_void_p, POINTER(NDIlib_video_frame_v2_t), c_void_p, c_void_p, c_uint32]
    ndi_lib.NDIlib_recv_capture_v2.restype = c_bool
    
    ndi_lib.NDIlib_recv_free_video_v2.argtypes = [c_void_p, POINTER(NDIlib_video_frame_v2_t)]
    ndi_lib.NDIlib_recv_destroy.argtypes = [c_void_p]
    ndi_lib.NDIlib_find_destroy.argtypes = [c_void_p]
    ndi_lib.NDIlib_destroy.restype = None
    ndi_lib.NDIlib_recv_set_tally.argtypes = [c_void_p, POINTER(NDIlib_tally_t)]

def create_receiver(source):
    """Create an NDI receiver for a source"""
    try:
        create_settings = NDIlib_recv_create_v3_t()
        create_settings.source_to_connect_to = source
        create_settings.color_format = NDIlib_recv_color_format_e.NDIlib_recv_color_format_BGRX_BGRA
        create_settings.bandwidth = NDIlib_recv_bandwidth_e.NDIlib_recv_bandwidth_highest
        create_settings.allow_video_fields = False
        create_settings.p_ndi_recv_name = "NDI Video Viewer".encode('utf-8')
        
        receiver = ndi_lib.NDIlib_recv_create_v3(create_settings)
        if not receiver:
            raise RuntimeError("Failed to create receiver")
            
        return receiver
    except Exception as e:
        logger.error(f"Error creating receiver: {e}\n{traceback.format_exc()}")
        raise

def safe_frame_capture(receiver):
    """Safely capture a frame with timeout"""
    if not receiver:
        return None
        
    frame_type = None
    try:
        frame_type = NDIlib_video_frame_v2_t()
        success = ndi_lib.NDIlib_recv_capture_v2(receiver, byref(frame_type), None, None, CAPTURE_TIMEOUT_MS)
        
        if success == NDIlib_frame_type_e.NDIlib_frame_type_video:
            try:
                # Get frame data
                buffer_size = frame_type.xres * frame_type.yres * 4
                frame_buffer = (c_ubyte * buffer_size)()
                frame_buffer_ptr = cast(frame_buffer, POINTER(c_ubyte))
                
                # Copy frame data
                memmove(frame_buffer_ptr, frame_type.p_data, buffer_size)
                
                # Convert to numpy array
                frame = np.frombuffer(frame_buffer, dtype=np.uint8).reshape(frame_type.yres, frame_type.xres, 4)
                frame = np.ascontiguousarray(frame.copy())  # Make a copy to ensure memory ownership
                
                # Process frame
                frame = process_frame(frame)
                if frame is not None:
                    # Resize frame if needed
                    frame = resize_frame(frame)
                
                return frame
                
            except Exception as e:
                logger.error(f"Error processing captured frame: {e}")
                return None
                
        return None
        
    except Exception as e:
        logger.error(f"Error capturing frame: {e}")
        return None
        
    finally:
        # Always free NDI frame memory
        if frame_type is not None:
            try:
                ndi_lib.NDIlib_recv_free_video_v2(receiver, byref(frame_type))
            except Exception as e:
                logger.error(f"Error freeing frame memory: {e}")

class NDIViewer:
    def __init__(self, root):
        """Initialize the NDI viewer application"""
        try:
            logger.info("Initializing NDI viewer...")
            self.root = root
            self.root.title("NDI Video Viewer")
            
            # Initialize variables
            self.NDI_find = None
            self.NDI_sources = []
            self.current_source = None
            self.receiver = None
            self.current_frame = None
            self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
            self.running = Event()
            self.capture_thread = None
            self.lock = Lock()
            self.last_frame_time = 0
            self.connection_healthy = True
            self.reconnect_attempts = 0
            self.error_count = 0
            self.max_errors = 5
            self.last_error_time = 0
            self.error_window = 60  # Reset error count after 60 seconds
            
            # Add cleanup handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Initialize NDI
            logger.info("Initializing NDI...")
            if not ndi_lib.NDIlib_initialize():
                raise RuntimeError("Failed to initialize NDI")
            logger.info("NDI initialized successfully")
            
            # Create GUI
            self.create_gui()
            
            # Start NDI source discovery
            self.start_ndi_finder()
            
            # Start display loop
            self.display_frame()
            
        except Exception as e:
            error_msg = f"Error initializing application: {e}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            self.cleanup()
            raise RuntimeError(error_msg)

    def create_gui(self):
        """Create the GUI elements"""
        # Create main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Create control frame
        control_frame = tk.Frame(main_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Create source selection
        source_frame = tk.Frame(control_frame)
        source_frame.pack(side=tk.LEFT, padx=5)
        
        tk.Label(source_frame, text="NDI Source:").pack(side=tk.LEFT)
        self.source_var = tk.StringVar(value="")
        self.source_menu = tk.OptionMenu(source_frame, self.source_var, "")
        self.source_menu.pack(side=tk.LEFT, padx=5)
        
        # Create button frame
        button_frame = tk.Frame(control_frame)
        button_frame.pack(side=tk.LEFT, padx=5)
        
        self.connect_button = tk.Button(button_frame, text="Connect", command=self.connect_to_source)
        self.connect_button.pack(side=tk.LEFT, padx=2)
        
        self.refresh_button = tk.Button(button_frame, text="Refresh Sources", command=self.refresh_sources)
        self.refresh_button.pack(side=tk.LEFT, padx=2)
        
        # Create status frame
        status_frame = tk.Frame(main_frame)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = tk.Label(status_frame, text="Ready", fg="black")
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        self.fps_label = tk.Label(status_frame, text="FPS: --", fg="black")
        self.fps_label.pack(side=tk.RIGHT, padx=5)
        
        # Create video frame
        self.video_label = tk.Label(main_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

    def update_status(self, message, color="black"):
        """Update status message"""
        self.status_label.config(text=message, fg=color)

    def capture_frames(self):
        """Capture frames in a separate thread"""
        frame_count = 0
        last_frame_time = time.time()
        error_count = 0
        last_error_time = 0
        
        while self.running.is_set():
            try:
                current_time = time.time()
                # Enforce frame rate limit
                if current_time - last_frame_time < MIN_FRAME_INTERVAL:
                    time.sleep(MIN_FRAME_INTERVAL - (current_time - last_frame_time))
                    continue
                    
                if not self.receiver:
                    logger.error("Receiver is None in capture thread")
                    break
                    
                # Capture frame
                frame = safe_frame_capture(self.receiver)
                
                if frame is not None:
                    # Skip frame if queue is full
                    if self.frame_queue.full():
                        continue
                        
                    # Add new frame
                    try:
                        self.frame_queue.put(frame, timeout=0.1)
                        frame_count += 1
                        last_frame_time = time.time()
                        
                        # Reset error tracking on successful frame
                        error_count = 0
                        last_error_time = 0
                        
                        # Update connection status
                        if not self.connection_healthy:
                            self.connection_healthy = True
                            self.update_status("Connected", "green")
                            
                    except queue.Full:
                        continue
                        
                else:
                    # Handle frame capture failure
                    current_time = time.time()
                    if current_time - last_error_time > self.error_window:
                        error_count = 0
                        last_error_time = current_time
                        
                    error_count += 1
                    if error_count >= self.max_errors:
                        logger.error("Too many frame capture errors, attempting reconnect")
                        self.reconnect_to_source()
                        error_count = 0
                    
            except Exception as e:
                logger.error(f"Error in capture thread: {e}")
                current_time = time.time()
                if current_time - last_error_time > self.error_window:
                    error_count = 0
                error_count += 1
                if error_count >= self.max_errors:
                    self.reconnect_to_source()
                    error_count = 0
                last_error_time = current_time
                time.sleep(0.1)  # Prevent tight error loop
                
        logger.info("Capture thread stopped")

    def reconnect_to_source(self):
        """Attempt to reconnect to the current source"""
        try:
            with self.lock:
                if self.reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
                    logger.error("Max reconnection attempts reached")
                    self.update_status("Connection failed - Please reconnect manually", "red")
                    self.stop_capture()
                    return
                    
                self.reconnect_attempts += 1
                self.update_status(f"Reconnecting (Attempt {self.reconnect_attempts})...", "orange")
                
                # Close existing receiver
                if self.receiver:
                    try:
                        ndi_lib.NDIlib_recv_destroy(self.receiver)
                    except Exception as e:
                        logger.error(f"Error destroying receiver: {e}")
                    self.receiver = None
                
                # Clear frame queue
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Create new receiver
                if self.current_source:
                    try:
                        self.receiver = create_receiver(self.current_source)
                        logger.info("Successfully reconnected")
                        self.update_status("Reconnected", "green")
                        self.reconnect_attempts = 0
                    except Exception as e:
                        logger.error(f"Failed to create new receiver: {e}")
                        self.update_status("Reconnection failed", "red")
                
        except Exception as e:
            logger.error(f"Error during reconnection: {e}")
            self.update_status("Reconnection error", "red")

    def cleanup(self):
        """Clean up resources"""
        try:
            # Stop capture thread first
            self.stop_capture()
            
            # Clean up NDI resources
            with self.lock:
                if self.receiver:
                    try:
                        ndi_lib.NDIlib_recv_destroy(self.receiver)
                    except Exception as e:
                        logger.error(f"Error destroying receiver: {e}")
                    self.receiver = None
                    
                if self.NDI_find:
                    try:
                        ndi_lib.NDIlib_find_destroy(self.NDI_find)
                    except Exception as e:
                        logger.error(f"Error destroying finder: {e}")
                    self.NDI_find = None
            
            # Final NDI cleanup
            try:
                ndi_lib.NDIlib_destroy()
            except Exception as e:
                logger.error(f"Error in NDI cleanup: {e}")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def on_closing(self):
        """Handle window closing"""
        self.cleanup()
        self.root.destroy()

    def display_frame(self):
        """Display the current frame"""
        try:
            if not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    frame = None
                    
                if frame is not None and frame.size > 0:
                    try:
                        # Convert frame for display
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Create PIL image
                        image = Image.fromarray(frame_rgb)
                        photo = ImageTk.PhotoImage(image=image)
                        
                        # Update canvas
                        if not hasattr(self, 'canvas'):
                            self.canvas = tk.Canvas(self.root, width=frame.shape[1], height=frame.shape[0])
                            self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                            self.photo = None
                            
                        self.canvas.config(width=frame.shape[1], height=frame.shape[0])
                        if self.photo is None:
                            self.photo = self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                        else:
                            self.canvas.itemconfig(self.photo, image=photo)
                            
                        # Keep a reference to avoid garbage collection
                        self.current_frame = photo
                        
                        # Update FPS
                        current_time = time.time()
                        if hasattr(self, 'last_frame_time') and self.last_frame_time:
                            fps = 1.0 / (current_time - self.last_frame_time)
                            self.fps_label.config(text=f"FPS: {fps:.1f}")
                        self.last_frame_time = current_time
                        
                        # Update status
                        self.update_status("Connected - Receiving frames", "green")
                    except Exception as e:
                        logger.error(f"Error processing frame for display: {e}")
                        self.update_status("Error processing frame", "red")
                else:
                    self.update_status("No valid frame received", "orange")
            
        except Exception as e:
            logger.error(f"Error displaying frame: {e}")
            self.update_status("Display error", "red")
            
        finally:
            # Always schedule next update, even if there was an error
            if not self.root.winfo_exists():
                return
            self.root.after(DISPLAY_INTERVAL_MS, self.display_frame)

    def connect_to_source(self):
        """Connect to the selected NDI source"""
        try:
            # Update UI state
            self.connect_button.config(state=tk.DISABLED)
            self.refresh_button.config(state=tk.DISABLED)
            self.update_status("Connecting...", "orange")
            
            # Stop any existing capture thread
            self.stop_capture()
            
            # Get selected source
            source_name = self.source_var.get()
            if not source_name:
                logger.warning("No source selected")
                self.update_status("No source selected", "red")
                self.connect_button.config(state=tk.NORMAL)
                self.refresh_button.config(state=tk.NORMAL)
                return
            
            logger.info(f"Connecting to source: {source_name}")
            
            # Find source in list
            source = next((s for n, s in self.NDI_sources if n == source_name), None)
            if not source:
                logger.warning("Source not found in list")
                self.update_status("Source not found", "red")
                self.connect_button.config(state=tk.NORMAL)
                self.refresh_button.config(state=tk.NORMAL)
                return
            
            # Create new receiver
            if self.receiver:
                logger.info("Destroying old receiver")
                try:
                    ndi_lib.NDIlib_recv_destroy(self.receiver)
                except Exception as e:
                    logger.error(f"Error destroying old receiver: {e}")
            
            logger.info("Creating new receiver")
            try:
                self.receiver = create_receiver(source)
                if not self.receiver:
                    raise RuntimeError("Failed to create receiver")
            except Exception as e:
                logger.error(f"Error creating receiver: {e}")
                self.update_status("Failed to create receiver", "red")
                self.connect_button.config(state=tk.NORMAL)
                self.refresh_button.config(state=tk.NORMAL)
                return
            
            self.current_source = source
            
            # Reset connection state
            self.connection_healthy = True
            self.reconnect_attempts = 0
            self.error_count = 0
            
            # Start capture thread
            self.start_capture()
            
            # Update UI state
            self.connect_button.config(state=tk.NORMAL)
            self.refresh_button.config(state=tk.NORMAL)
            self.update_status("Connected", "green")
            
        except Exception as e:
            logger.error(f"Error connecting to source: {e}\n{traceback.format_exc()}")
            self.update_status(f"Connection error: {str(e)}", "red")
            self.connect_button.config(state=tk.NORMAL)
            self.refresh_button.config(state=tk.NORMAL)
            messagebox.showerror("Error", f"Failed to connect to source: {e}")

    def refresh_sources(self):
        """Refresh the list of NDI sources"""
        try:
            logger.info("Refreshing NDI sources...")
            self.update_status("Refreshing sources...", "orange")
            
            # Wait for sources
            if ndi_lib.NDIlib_find_wait_for_sources(self.NDI_find, 1000):
                # Get current sources
                num_sources = c_uint32()
                sources = ndi_lib.NDIlib_find_get_current_sources(self.NDI_find, byref(num_sources))
                
                if num_sources.value > 0:
                    # Update sources list
                    self.NDI_sources = [(sources[i].p_ndi_name.decode('utf-8'), sources[i]) for i in range(num_sources.value)]
                    
                    # Update source menu
                    menu = self.source_menu["menu"]
                    menu.delete(0, tk.END)
                    for name, _ in self.NDI_sources:
                        menu.add_command(label=name, command=lambda n=name: self.source_var.set(n))
                    
                    if self.NDI_sources:
                        self.source_var.set(self.NDI_sources[0][0])
                        self.update_status("Sources found", "green")
                    else:
                        self.update_status("No sources found", "orange")
                else:
                    logger.info("No NDI sources found")
                    self.source_var.set("")
                    self.update_status("No sources found", "orange")
            else:
                logger.warning("Timeout waiting for sources")
                self.update_status("Timeout waiting for sources", "orange")
                
        except Exception as e:
            logger.error(f"Error refreshing sources: {e}")
            self.update_status("Error refreshing sources", "red")

    def start_ndi_finder(self):
        """Initialize NDI source discovery"""
        try:
            logger.info("Starting NDI finder...")
            find_create = NDIlib_find_create_t()
            find_create.show_local_sources = True
            find_create.p_groups = None
            find_create.p_extra_ips = None
            
            self.NDI_find = ndi_lib.NDIlib_find_create_v2(byref(find_create))
            if not self.NDI_find:
                raise RuntimeError("Failed to create NDI finder")
                
            # Initial source refresh
            self.refresh_sources()
            
        except Exception as e:
            logger.error(f"Error starting NDI finder: {e}")
            raise

    def start_capture(self):
        """Start the frame capture thread"""
        try:
            self.running.set()
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.daemon = True
            self.capture_thread.start()
            logger.info("Capture thread started")
        except Exception as e:
            logger.error(f"Error starting capture thread: {e}")
            self.running.clear()
            raise

    def stop_capture(self):
        """Stop the frame capture thread"""
        try:
            self.running.clear()
            if self.capture_thread and self.capture_thread.is_alive():
                self.capture_thread.join(timeout=2.0)
                if self.capture_thread.is_alive():
                    logger.warning("Capture thread did not stop cleanly")
                self.capture_thread = None
            
            # Clear the frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
                    
        except Exception as e:
            logger.error(f"Error stopping capture thread: {e}")

def main():
    try:
        # Load NDI library
        global ndi_lib
        ndi_lib = load_ndi_library()
        
        # Setup NDI functions
        setup_ndi_functions(ndi_lib)
        
        # Create and run application
        root = tk.Tk()
        app = NDIViewer(root)
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Application error: {e}\n{traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()
