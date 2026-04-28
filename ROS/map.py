import rclpy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_msgs.msg import TFMessage
import tf2_ros
import open3d as o3d
import numpy as np
import rosbag2_py

def main():
    rclpy.init()
    
    # --- CONFIGURATION ---
    bag_path = 'kinova_pointcloud_data_confidence_40'
    
    tf_buffer = tf2_ros.Buffer(rclpy.duration.Duration(seconds=10000))
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions('', '')

    # ==========================================
    # PASS 1: MEMORIZE ALL TRANSFORMS
    # ==========================================
    print("Pass 1: Loading all TF data into memory...")
    reader_tf = rosbag2_py.SequentialReader()
    reader_tf.open(storage_options, converter_options)

    while reader_tf.has_next():
        (topic, data, timestamp) = reader_tf.read_next()
        if topic == '/tf' or topic == '/tf_static':
            msg = deserialize_message(data, TFMessage)
            for transform in msg.transforms:
                if topic == '/tf_static':
                    tf_buffer.set_transform_static(transform, 'bag')
                else:
                    tf_buffer.set_transform(transform, 'bag')

    # ==========================================
    # PASS 2: STITCH THE POINTCLOUDS
    # ==========================================
    print("Pass 2: Processing pointclouds...")
    reader_pc = rosbag2_py.SequentialReader()
    reader_pc.open(storage_options, converter_options)

    global_map = o3d.geometry.PointCloud()

    while reader_pc.has_next():
        (topic, data, timestamp) = reader_pc.read_next()

        if topic == '/tof_pointcloud':
            msg = deserialize_message(data, PointCloud2)
            
            try:
                trans = tf_buffer.lookup_transform('base_link', 
                                                   msg.header.frame_id, 
                                                   msg.header.stamp,
                                                   rclpy.duration.Duration(seconds=0.05))
            except Exception:
                continue

            pc_data = point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            points = np.column_stack((pc_data['x'], pc_data['y'], pc_data['z'])).astype(np.float64)
            
            if points.shape[0] == 0: continue

            current_cloud = o3d.geometry.PointCloud()
            current_cloud.points = o3d.utility.Vector3dVector(points)

            # Filter gripper
            bbox = o3d.geometry.AxisAlignedBoundingBox(np.array([-0.1, -0.1, 0.0]), np.array([0.1, 0.1, 0.2]))
            cropped = current_cloud.crop(bbox)
            if len(cropped.points) > 0:
                distances = np.asarray(current_cloud.compute_point_cloud_distance(cropped))
                current_cloud = current_cloud.select_by_index(np.where(distances > 0.001)[0])

            # Transform
            tx, rx = trans.transform.translation, trans.transform.rotation
            x, y, z, w = rx.x, rx.y, rx.z, rx.w
            matrix = np.array([
                [1 - 2*(y**2 + z**2), 2*(x*y - z*w),       2*(x*z + y*w),       tx.x],
                [2*(x*y + z*w),       1 - 2*(x**2 + z**2), 2*(y*z - x*w),       tx.y],
                [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x**2 + y**2), tx.z],
                [0.0,                 0.0,                 0.0,                 1.0 ]
            ])
            current_cloud.transform(matrix)
            global_map += current_cloud.voxel_down_sample(voxel_size=0.01)

    global_map = global_map.voxel_down_sample(voxel_size=0.005)
    
    # ==========================================
    # PASS 3: RANSAC & PLANE GENERATION
    # ==========================================
    print("Extracting table and generating visual plane...")
    plane_model, inliers = global_map.segment_plane(distance_threshold=0.025, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane_model
    if c < 0: a, b, c, d = -a, -b, -c, -d 
        
    table_cloud = global_map.select_by_index(inliers)
    
    # Mathematical Plane for Vis 2
    table_bbox = table_cloud.get_axis_aligned_bounding_box()
    extents = table_bbox.get_max_bound() - table_bbox.get_min_bound()
    
    plane_mesh = o3d.geometry.TriangleMesh.create_box(width=extents[0], height=extents[1], depth=0.002)
    plane_mesh.translate(-plane_mesh.get_center()) 
    
    normal = np.array([a, b, c])
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, normal)
    s = np.linalg.norm(v)
    c_angle = np.dot(z_axis, normal)
    
    if s != 0:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c_angle) / (s**2))
        plane_mesh.rotate(R, center=(0,0,0))
    
    center_pt = table_cloud.get_center()
    dist_to_plane = np.dot(center_pt, normal) + d
    plane_mesh.translate(center_pt - (dist_to_plane * normal))
    
    plane_mesh.paint_uniform_color([0.8, 0.8, 0.8]) # Light gray table
    plane_mesh.compute_vertex_normals()

    # ==========================================
    # PASS 4: EXTRACT & FILTER OBJECTS
    # ==========================================
    print("Filtering and evaluating objects...")
    outlier_cloud = global_map.select_by_index(inliers, invert=True)
    outlier_points = np.asarray(outlier_cloud.points)
    
    signed_distances = (a * outlier_points[:, 0] + b * outlier_points[:, 1] + c * outlier_points[:, 2] + d)
    above_table_idx = np.where((signed_distances > 0.01) & (signed_distances < 0.50))[0]
    objects_above_table = outlier_cloud.select_by_index(above_table_idx)

    # For Visualization 1: We need a copy of these raw points colored Red & Blue
    vis1_table = o3d.geometry.PointCloud(table_cloud)
    vis1_table.paint_uniform_color([1.0, 0.0, 0.0]) # Red Table Points
    
    vis1_objects = o3d.geometry.PointCloud(objects_above_table)
    vis1_objects.paint_uniform_color([0.0, 0.0, 1.0]) # Blue Object Points

    # Cluster for Visualization 2
    labels = np.array(objects_above_table.cluster_dbscan(eps=0.019, min_points=40, print_progress=False))
    
    valid_hulls = []
    height_lines = []
    
    if len(labels) > 0:
        max_label = labels.max()
        object_counter = 1
        
        print("\n--- OBJECT MEASUREMENTS ---")
        
        for i in range(max_label + 1):
            cluster_idx = np.where(labels == i)[0]
            
            if len(cluster_idx) < 150: 
                continue
                
            cluster_cloud = objects_above_table.select_by_index(cluster_idx)
            cluster_points = np.asarray(cluster_cloud.points)
            
            obj_bbox = cluster_cloud.get_axis_aligned_bounding_box()
            obj_extents = obj_bbox.get_max_bound() - obj_bbox.get_min_bound()
            
            if max(obj_extents) < 0.02:
                continue
            
            cluster_color = np.random.uniform(0.2, 1.0, size=3)

            # --- CALCULATE MAX HEIGHT ---
            # Distance of every point in this object to the table plane
            obj_distances = (a * cluster_points[:, 0] + b * cluster_points[:, 1] + c * cluster_points[:, 2] + d)
            max_height_idx = np.argmax(obj_distances)
            max_height = obj_distances[max_height_idx]
            highest_point = cluster_points[max_height_idx]
            
            # Project highest point straight down to the table to draw the line
            projected_point = highest_point - (max_height * normal)
            
            print(f"Object {object_counter}: Max Height = {max_height:.3f} meters")
            object_counter += 1

            # Create the line geometry (visual ruler)
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector([highest_point, projected_point])
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.paint_uniform_color([1.0, 0.0, 0.0]) # Red line to stand out
            height_lines.append(line_set)

            # Compute Convex Hull
            try:
                hull_mesh, _ = cluster_cloud.compute_convex_hull()
                hull_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(hull_mesh)
                hull_wireframe.paint_uniform_color(cluster_color)
                valid_hulls.append(hull_wireframe)
            except Exception as e:
                pass

        print("---------------------------\n")

    # ==========================================
    # PASS 5: SEQUENTIAL VISUALIZATIONS
    # ==========================================
    
    # VISUALIZATION 1
    print("Opening Window 1: Red Table Points + Blue Object Points.")
    print(">>> CLOSE WINDOW 1 TO PROCEED TO WINDOW 2 <<<")
    o3d.visualization.draw_geometries([vis1_table, vis1_objects], window_name="Pass 1: Raw Points")

    # VISUALIZATION 2
    print("Opening Window 2: Plane Mesh + Object Wireframes + Height Markers.")
    geometries_vis2 = [plane_mesh] + valid_hulls + height_lines
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3, origin=[0, 0, 0])
    geometries_vis2.append(coord_frame)

    o3d.visualization.draw_geometries(geometries_vis2, mesh_show_back_face=True, window_name="Pass 2: Hulls and Heights")

    rclpy.shutdown()

if __name__ == '__main__':
    main()