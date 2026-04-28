import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField # <-- Added PointField
from std_msgs.msg import Header
import sensor_msgs_py.point_cloud2 as pc2

import serial
import struct
import numpy as np
import threading

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
COM_PORT = '/dev/ttyACM0'       # Change to your ESP32 port
BAUD_RATE = 921600              # Must match your ESP32 settings
MODE = "16x16"                  # 32x32 Mode for high-res mapping
FRAME_ID = "lidar_link"         # The TF frame attached to the robot's wrist
CONFIDENCE_THRESHOLD = 40       # <-- NEW: Minimum confidence score (0-255) to keep a point

MODE_MAP = {
    "8x8":   (8, 8),
    "16x16": (16, 16),
    "32x32": (32, 32),
    "48x32": (32, 48)
}

HEIGHT, WIDTH = MODE_MAP[MODE]
NUM_PIXELS = HEIGHT * WIDTH
EXPECTED_PAYLOAD_SIZE = NUM_PIXELS * 3  # 2 dist bytes + 1 conf byte

SYNC_WORD = b'\xAA\x55'
END_WORD = b'\xEF\xBE'

class TofPublisher(Node):
    def __init__(self):
        super().__init__('tof_pointcloud_publisher')
        
        # Create the ROS 2 Publisher
        self.publisher_ = self.create_publisher(PointCloud2, '/tof_pointcloud', 10)
        
        # Serial Setup
        try:
            self.ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
            self.get_logger().info(f"✅ Connected to {COM_PORT} at {BAUD_RATE} baud.")
            self.get_logger().info(f"🎯 Mode: {MODE} ({WIDTH}x{HEIGHT} pixels)")
        except Exception as e:
            self.get_logger().error(f"❌ Could not open serial port: {e}")
            exit()

        self.frame_buffer = bytearray()
        
        # ─── PRE-COMPUTE UNIT RAYS (The Bowing Fix) ───────────────────────────
        X_idx, Y_idx = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
        xs_flat = X_idx.flatten()
        ys_flat = Y_idx.flatten()
        
        # FOV Math constants for TMF882x / VL53 series
        FOV_X_DEG = 67.9
        FOV_Y_DEG = 52.8
        tan_half_x = np.tan(np.radians(FOV_X_DEG) / 2.0)
        tan_half_y = np.tan(np.radians(FOV_Y_DEG) / 2.0)

        # Normalize pixel coordinates from -1.0 to 1.0
        nx = (xs_flat - (WIDTH - 1) / 2.0) / (WIDTH / 2.0)
        ny = (ys_flat - (HEIGHT - 1) / 2.0) / (HEIGHT / 2.0)

        # Create direction rays
        ray_x = -nx * tan_half_x
        ray_y = ny * tan_half_y
        ray_z = np.ones_like(ray_x)

        # Normalize the rays into Unit Vectors (length of exactly 1)
        ray_norms = np.sqrt(ray_x**2 + ray_y**2 + ray_z**2)
        self.unit_ray_x = ray_x / ray_norms
        self.unit_ray_y = ray_y / ray_norms
        self.unit_ray_z = ray_z / ray_norms

        # Define custom fields for PointCloud2 (x, y, z, intensity)
        self.custom_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]

        # Start a background thread to read the serial port continuously
        self.running = True
        self.read_thread = threading.Thread(target=self.serial_read_loop)
        self.read_thread.start()

    def read_frame(self):
        target_len = 2 + 2 + EXPECTED_PAYLOAD_SIZE + 2
        
        while self.running and rclpy.ok():
            if self.ser.in_waiting > 0:
                self.frame_buffer.extend(self.ser.read(self.ser.in_waiting))
            
            sync_idx = self.frame_buffer.find(SYNC_WORD)
            if sync_idx == -1:
                if len(self.frame_buffer) > 1:
                    self.frame_buffer = self.frame_buffer[-1:]
                continue
                
            self.frame_buffer = self.frame_buffer[sync_idx:]
            
            if len(self.frame_buffer) < target_len:
                continue
                
            num_pixels = struct.unpack('<H', self.frame_buffer[2:4])[0]
            if num_pixels != NUM_PIXELS:
                self.frame_buffer = self.frame_buffer[2:] 
                continue

            payload_start = 4
            payload_end = payload_start + EXPECTED_PAYLOAD_SIZE
            
            pixel_data = self.frame_buffer[payload_start:payload_end]
            end_marker = self.frame_buffer[payload_end:payload_end+2]
            
            if end_marker != END_WORD:
                self.frame_buffer = self.frame_buffer[2:] 
                continue
                
            self.frame_buffer = self.frame_buffer[payload_end+2:]
            return pixel_data
        return None

    def serial_read_loop(self):
        while self.running and rclpy.ok():
            raw_pixels = self.read_frame()
            if raw_pixels is None: continue

            arr = np.frombuffer(raw_pixels, dtype=np.uint8)
            pixels = arr.reshape((NUM_PIXELS, 3))
            
            # Extract distance and confidence
            distances_mm = pixels[:, 0].astype(np.uint16) | (pixels[:, 1].astype(np.uint16) << 8)
            confidence = pixels[:, 2].astype(np.uint8)
            
            # Remove failed readings AND low confidence points
            valid_mask = (distances_mm > 0) & (confidence >= CONFIDENCE_THRESHOLD)
            
            if not np.any(valid_mask):
                continue

            # Convert to meters immediately
            valid_dist_m = distances_mm[valid_mask] / 1000.0
            
            # Get valid confidence values (cast to float32 for RViz compatibility)
            valid_conf = confidence[valid_mask].astype(np.float32)

            # Get the pre-computed unit rays for the valid pixels
            ux = self.unit_ray_x[valid_mask]
            uy = self.unit_ray_y[valid_mask]
            uz = self.unit_ray_z[valid_mask]

            # Multiply radial distance by unit vectors to get flat Cartesian XYZ
            x_m = valid_dist_m * ux
            y_m = valid_dist_m * uy
            z_m = valid_dist_m * uz

            # Stack into a list of [x, y, z, intensity] points
            points_4d = np.vstack((x_m, y_m, z_m, valid_conf)).T

            # Create standard ROS Header
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = FRAME_ID
            
            # Generate and publish PointCloud2 message using our custom fields
            pc_msg = pc2.create_cloud(header, self.custom_fields, points_4d.tolist())
            self.publisher_.publish(pc_msg)

    def destroy_node(self):
        self.running = False
        self.read_thread.join()
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = TofPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("🛑 Stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
