import serial
import struct
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage  # Required for array interpolation

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
COM_PORT = '/dev/ttyACM0'       # Change to your ESP32 port
BAUD_RATE = 921600              # Must match your ESP32 settings
MODE = "16x16"                  # Options: "8x8", "16x16", "32x32", "48x32"

VISUALIZATION_TYPE = "POINTCLOUD"  # Options: "HEATMAP" or "POINTCLOUD"
INTERPOLATE_2X = False             # Set to True to mathematically double the heatmap resolution!

#Minimum confidence required to display a point in the 3D Point Cloud (0-255)
CONFIDENCE_THRESHOLD = 10

# ─── Constants Based on Mode ──────────────────────────────────────────────────
MODE_MAP = {
    "8x8":   (8, 8),
    "16x16": (16, 16),
    "32x32": (32, 32),
    "48x32": (32, 48)  # Height x Width
}

if MODE not in MODE_MAP:
    raise ValueError(f"Invalid mode. Choose from {list(MODE_MAP.keys())}")

HEIGHT, WIDTH = MODE_MAP[MODE]
NUM_PIXELS = HEIGHT * WIDTH
EXPECTED_PAYLOAD_SIZE = NUM_PIXELS * 3 # 3 bytes per pixel (2 dist + 1 conf)

# Protocol Markers
SYNC_WORD = b'\xAA\x55'
END_WORD = b'\xEF\xBE'

# ─── Serial Setup ─────────────────────────────────────────────────────────────
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
    print(f"✅ Connected to {COM_PORT} at {BAUD_RATE} baud.")
    print(f"🎯 Mode: {MODE} ({WIDTH}x{HEIGHT} pixels)")
    print(f"📊 Visualization: {VISUALIZATION_TYPE} {'(2x Interpolated)' if INTERPOLATE_2X and VISUALIZATION_TYPE == 'HEATMAP' else ''}")
except Exception as e:
    print(f"❌ Could not open serial port: {e}")
    exit()

# ─── Visualization Setup ──────────────────────────────────────────────────────
plt.ion() # Interactive mode on

if VISUALIZATION_TYPE == "HEATMAP":
    # Setup 1x2 Subplots for side-by-side view
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"TMF8829 Live Heatmap ({MODE})", fontsize=14)
    
    # Adjust display grid size based on interpolation flag
    disp_h = HEIGHT * 2 if INTERPOLATE_2X else HEIGHT
    disp_w = WIDTH * 2 if INTERPOLATE_2X else WIDTH
    data_grid = np.zeros((disp_h, disp_w))
    
    # Use bicubic display smoothing if interpolated, otherwise raw nearest pixels
    interp_style = 'bilinear' if INTERPOLATE_2X else 'nearest'
    
    # Subplot 1: Distance Heatmap
    cmap_dist = plt.get_cmap('magma_r') 
    img_dist = ax1.imshow(data_grid, cmap=cmap_dist, vmin=0, vmax=300, interpolation=interp_style)
    plt.colorbar(img_dist, ax=ax1, label='Distance (mm)')
    ax1.set_title(f"Distance {'[2x Interpolated]' if INTERPOLATE_2X else ''}")
    
    # Subplot 2: Confidence Heatmap
    cmap_conf = plt.get_cmap('viridis') # Distinct colormap for confidences
    img_conf = ax2.imshow(data_grid, cmap=cmap_conf, vmin=0, vmax=255, interpolation=interp_style)
    plt.colorbar(img_conf, ax=ax2, label='Confidence')
    ax2.set_title(f"Confidence {'[2x Interpolated]' if INTERPOLATE_2X else ''}")

    center_text = ax1.text(0, -1, "Waiting for data...", color='black', fontsize=12)

elif VISUALIZATION_TYPE == "POINTCLOUD":
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    X_idx, Y_idx = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    xs_flat = X_idx.flatten()
    ys_flat = Y_idx.flatten()
    zs_flat = np.zeros(NUM_PIXELS)
    
    scatter = ax.scatter(xs_flat, ys_flat, zs_flat, c=zs_flat, cmap='magma', s=15, vmin=0, vmax=1000)
    
    ax.set_title(f"TMF8829 Live 3D Point Cloud ({MODE}) - Filtered > {CONFIDENCE_THRESHOLD}")
    ax.set_zlim(0, 1500) 
    ax.set_xlim(-800, 800) 
    ax.set_ylim(-800, 800) 
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('True Z Depth (mm)')
    
    ax.invert_zaxis()
    ax.invert_yaxis()

# Pre-allocate buffer for faster reading
frame_buffer = bytearray()

def read_frame():
    global frame_buffer
    target_len = 2 + 2 + EXPECTED_PAYLOAD_SIZE + 2
    
    while True:
        if ser.in_waiting > 0:
            frame_buffer.extend(ser.read(ser.in_waiting))
        
        sync_idx = frame_buffer.find(SYNC_WORD)
        if sync_idx == -1:
            if len(frame_buffer) > 1:
                frame_buffer = frame_buffer[-1:]
            continue
            
        frame_buffer = frame_buffer[sync_idx:]
        
        if len(frame_buffer) < target_len:
            plt.pause(0.001) 
            continue
            
        num_pixels = struct.unpack('<H', frame_buffer[2:4])[0]
        if num_pixels != NUM_PIXELS:
            frame_buffer = frame_buffer[2:] 
            continue

        payload_start = 4
        payload_end = payload_start + EXPECTED_PAYLOAD_SIZE
        
        pixel_data = frame_buffer[payload_start:payload_end]
        end_marker = frame_buffer[payload_end:payload_end+2]
        
        if end_marker != END_WORD:
            frame_buffer = frame_buffer[2:] 
            continue
            
        frame_buffer = frame_buffer[payload_end+2:]
        return pixel_data

# ─── Main Loop ────────────────────────────────────────────────────────────────
try:
    while True:
        raw_pixels = read_frame()
        if raw_pixels is None: continue

        arr = np.frombuffer(raw_pixels, dtype=np.uint8)
        pixels = arr.reshape((NUM_PIXELS, 3))
        
        distances = pixels[:, 0].astype(np.uint16) | (pixels[:, 1].astype(np.uint16) << 8)
        confidences = pixels[:, 2]

        if VISUALIZATION_TYPE == "HEATMAP":
            # Reshape both arrays into grids
            grid_dist = distances.reshape((HEIGHT, WIDTH))
            grid_conf = confidences.reshape((HEIGHT, WIDTH))
            
            # Mathematically double the array resolution if requested
            if INTERPOLATE_2X:
                # order=1 applies bilinear interpolation
                display_dist = scipy.ndimage.zoom(grid_dist, 2, order=1)
                display_conf = scipy.ndimage.zoom(grid_conf, 2, order=1)
            else:
                display_dist = grid_dist
                display_conf = grid_conf
                
            # Update both heatmaps
            img_dist.set_data(display_dist)
            img_conf.set_data(display_conf)
            
            # Keep the text reading the exact original center pixel data
            cy, cx = HEIGHT // 2, WIDTH // 2
            d = grid_dist[cy, cx]
            c = grid_conf[cy, cx]
            center_text.set_text(f"Center [{cx},{cy}]: {d}mm (Conf: {c})")

        elif VISUALIZATION_TYPE == "POINTCLOUD":
            # 1. Optical calculations
            FOV_X_DEG = 67.9
            FOV_Y_DEG = 52.8
            
            tan_half_fov_x = np.tan(np.radians(FOV_X_DEG) / 2.0)
            tan_half_fov_y = np.tan(np.radians(FOV_Y_DEG) / 2.0)
            
            fx = (WIDTH / 2.0) / tan_half_fov_x
            fy = (HEIGHT / 2.0) / tan_half_fov_y
            
            cx_pixel = (WIDTH - 1) / 2.0
            cy_pixel = (HEIGHT - 1) / 2.0
            
            dx = (xs_flat - cx_pixel) / fx
            dy = (ys_flat - cy_pixel) / fy
            
            Z_depth = distances / np.sqrt(dx**2 + dy**2 + 1.0)
            real_x = Z_depth * dx
            real_y = Z_depth * dy
            
            # 2.  Filter by Confidence 
            valid_mask = confidences > CONFIDENCE_THRESHOLD
            
            # Extract only the coordinates where confidence is high enough
            filtered_x = real_x[valid_mask]
            filtered_y = real_y[valid_mask]
            filtered_z = Z_depth[valid_mask]
            
            # 3. Update the plot with only the valid points
            scatter._offsets3d = (filtered_x, filtered_y, filtered_z)
            scatter.set_array(filtered_z) 

        fig.canvas.flush_events()
        
except KeyboardInterrupt:
    print("\n🛑 Stopped by user.")
    ser.close()