"""
Copyright (C) 2026 Geoffrey Daniels. https://gpdaniels.com/

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License only.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import ctypes
import math
import sys

import zeroslam


def require(passed, expression):
    if passed:
        return
    print("ERROR: Requirement '{}' failed.".format(expression), file=sys.stderr)
    sys.exit(1)


def render_corridor(width, height, camera, camera_x):
    wall_depth = 6.0
    mask = 0xFFFFFFFFFFFFFFFF
    pixels = bytearray(width * height)
    for row in range(height):
        ray_y = ((row + 0.5) - camera.centre_y) / camera.focal_y
        wall = (abs(ray_y) * wall_depth) <= 1.0
        depth = wall_depth if wall else (1.0 / abs(ray_y))
        surface = 1 if wall else (2 if ray_y > 0.0 else 3)
        cell_v = math.floor(4.0 * ((ray_y * depth) if wall else depth))
        row_hash = ((cell_v * 19349663) ^ (surface * 83492791)) & mask
        for column in range(width):
            ray_x = ((column + 0.5) - camera.centre_x) / camera.focal_x
            cell_u = math.floor(4.0 * (camera_x + (ray_x * depth)))
            value = ((cell_u * 73856093) & mask) ^ row_hash
            value ^= value >> 13
            value = (value * 0x5BD1E995) & mask
            value ^= value >> 15
            pixels[(row * width) + column] = 40 + (value % 176)
    return pixels


def print_pose(label, pose):
    print("{}: timestamp {}, centre ({:.3f}, {:.3f}, {:.3f}), rotation ({:.3f}, {:.3f}, {:.3f}, {:.3f})".format(label, pose.timestamp, *pose.pose))


def main():
    success = zeroslam.zeroslam_return_enum.zeroslam_return_success

    with zeroslam.system() as system:
        print("create: {}".format("success" if system.is_valid() else "failure"))
        require(system.is_valid(), "system.is_valid()")

        length = ctypes.c_int(0)
        result = system.get_configuration(None, length)
        require(result == zeroslam.zeroslam_return_enum.zeroslam_return_failure_insufficient_data_length, "get_configuration(None) reports the length")
        require(length.value > 0, "length > 0")
        configuration = ctypes.create_string_buffer(length.value)
        result = system.get_configuration(configuration, length)
        print("get_configuration: {} ({} characters)".format(result.name, length.value))
        require(result == success, "get_configuration")
        print(configuration.value.decode("ascii"), end="")

        result = system.set_configuration(b"verbosity=1\nlines=on\n")
        print("set_configuration: {}".format(result.name))
        require(result == success, "set_configuration")

        result = system.set_configuration(b"warp_speed=9\n")
        print("set_configuration (unknown key): {}".format(result.name))
        require(result == zeroslam.zeroslam_return_enum.zeroslam_return_failure_invalid_configuration, "set_configuration rejects unknown keys")

        width = 320
        height = 240
        sensor_id = 1
        camera_parameters = zeroslam.zeroslam_sensor_parameters_camera_struct(width, height, 262.5, 262.5, 160.0, 120.0, (ctypes.c_double * 8)())
        rig = zeroslam.zeroslam_sensor_rig_struct()
        rig.type = zeroslam.zeroslam_sensor_enum.zeroslam_sensor_camera
        rig.sensor_id = sensor_id
        rig.parameters_length = ctypes.sizeof(camera_parameters)
        rig.parameters_data = ctypes.cast(ctypes.pointer(camera_parameters), ctypes.c_void_p)
        result = system.set_sensor_rig(rig, 1)
        print("set_sensor_rig: {} (camera, id {}, {}x{})".format(result.name, sensor_id, width, height))
        require(result == success, "set_sensor_rig")

        # Where each sensor sits on the rig's body, sensor to body, and whether it is refined: a single camera is the body.
        extrinsic = zeroslam.zeroslam_sensor_extrinsics_struct()
        extrinsic.sensor_id = sensor_id
        extrinsic.refinement = zeroslam.zeroslam_extrinsic_enum.zeroslam_extrinsic_fixed
        extrinsic.pose[6] = 1.0
        result = system.set_sensor_extrinsics(extrinsic, 1)
        print("set_sensor_extrinsics: {} (id {} at the body's origin, fixed)".format(result.name, sensor_id))
        require(result == success, "set_sensor_extrinsics")

        frame_count = 20
        frame_interval = 33333333
        for frame_index in range(frame_count):
            pixels = render_corridor(width, height, camera_parameters, 0.05 * frame_index)
            buffer = (ctypes.c_ubyte * len(pixels)).from_buffer(pixels)
            data = zeroslam.zeroslam_sensor_data_struct()
            data.timestamp = 1000000000 + (frame_interval * frame_index)
            data.sensor_id = sensor_id
            data.measurement_length = width * height
            data.measurement_data = ctypes.cast(buffer, ctypes.c_void_p)
            result = system.set_sensor_data(data, 1)
            print("set_sensor_data: {} (timestamp {})".format(result.name, data.timestamp))
            require(result == success, "set_sensor_data")

        result = system.finalise()
        print("finalise: {}".format(result.name))
        require(result == success, "finalise")

        timestamp = ctypes.c_int64(0)
        result = system.get_timestamp(timestamp)
        print("get_timestamp: {} ({})".format(result.name, timestamp.value))
        require(result == success, "get_timestamp")
        require(timestamp.value == 1000000000 + (frame_interval * (frame_count - 1)), "timestamp is the newest frame")

        # The camera's calibration as the system holds it: the rig's, as the calibration key is off.
        calibration = zeroslam.zeroslam_sensor_parameters_camera_model_struct()
        result = system.get_camera_calibration(sensor_id, calibration)
        print("get_camera_calibration: {} (fx {:.3f}, fy {:.3f}, cx {:.3f}, cy {:.3f})".format(result.name, calibration.parameters[0], calibration.parameters[1], calibration.parameters[2], calibration.parameters[3]))
        require(result == success, "get_camera_calibration")
        require(calibration.parameters_length == 12, "the calibration is of the rig's model")
        require(calibration.parameters[0] == camera_parameters.focal_x, "the calibration is the rig's")
        require(system.get_camera_calibration(sensor_id + 1, calibration) == zeroslam.zeroslam_return_enum.zeroslam_return_failure_invalid_argument, "get_camera_calibration rejects an unknown sensor")

        pose = zeroslam.zeroslam_pose_struct()
        result = system.get_pose(pose)
        print("get_pose: {}".format(result.name))
        require(result == success, "get_pose")
        print_pose("newest pose", pose)

        pose = zeroslam.zeroslam_pose_struct()
        result = system.get_pose_at_timestamp(pose, 1000000000)
        print("get_pose_at_timestamp: {}".format(result.name))
        require(result == success, "get_pose_at_timestamp")
        print_pose("first pose", pose)
        require(pose.pose[0] == 0.0 and pose.pose[6] == 1.0, "first pose is the identity")

        result = system.get_pose_at_timestamp(pose, 12345)
        print("get_pose_at_timestamp (unknown): {}".format(result.name))
        require(result == zeroslam.zeroslam_return_enum.zeroslam_return_failure_invalid_argument, "unknown timestamp is invalid")

        chunk = zeroslam.zeroslam_map_chunk_struct()
        result = system.get_map_chunk(0.0, 0.0, 0.0, chunk)
        print("get_map_chunk: {} ({} points)".format(result.name, chunk.points_length))
        require(result == zeroslam.zeroslam_return_enum.zeroslam_return_failure_insufficient_data_length, "get_map_chunk reports the length")
        require(chunk.points_length > 0, "points_length > 0")
        points = (zeroslam.zeroslam_point_struct * chunk.points_length)()
        chunk.points = ctypes.cast(points, ctypes.POINTER(zeroslam.zeroslam_point_struct))
        result = system.get_map_chunk(0.0, 0.0, 0.0, chunk)
        print("get_map_chunk (sized): {} ({} points)".format(result.name, chunk.points_length))
        require(result == success, "get_map_chunk (sized)")
        for index in range(min(chunk.points_length, 5)):
            point = points[index]
            print("  point {}: ({:.3f}, {:.3f}, {:.3f}) confidence {:.3f}".format(index, point.x, point.y, point.z, point.confidence))

        lines = zeroslam.zeroslam_map_lines_struct()
        result = system.get_map_lines(lines)
        print("get_map_lines: {} ({} lines)".format(result.name, lines.lines_length))
        require(result == zeroslam.zeroslam_return_enum.zeroslam_return_failure_insufficient_data_length, "get_map_lines reports the length")
        require(lines.lines_length > 0, "lines_length > 0")
        line_buffer = (zeroslam.zeroslam_line_struct * lines.lines_length)()
        lines.lines = ctypes.cast(line_buffer, ctypes.POINTER(zeroslam.zeroslam_line_struct))
        require(system.get_map_lines(lines) == success, "get_map_lines (sized)")
        edges = zeroslam.zeroslam_map_edges_struct()
        result = system.get_map_edges(edges)
        print("get_map_edges: {} ({} edges)".format(result.name, edges.edges_length))
        require(result == zeroslam.zeroslam_return_enum.zeroslam_return_failure_insufficient_data_length, "get_map_edges reports the length")
        require(edges.edges_length > 0, "edges_length > 0")
        edge_buffer = (zeroslam.zeroslam_edge_struct * edges.edges_length)()
        edges.edges = ctypes.cast(edge_buffer, ctypes.POINTER(zeroslam.zeroslam_edge_struct))
        require(system.get_map_edges(edges) == success, "get_map_edges (sized)")
        for index in range(min(edges.edges_length, 5)):
            edge = edge_buffer[index]
            print("  edge {}: {} - {} type {} weight {}".format(index, edge.timestamp_a, edge.timestamp_b, edge.type, edge.weight))
        voxels = zeroslam.zeroslam_map_voxels_struct()
        result = system.get_map_voxels(voxels)
        print("get_map_voxels: {} ({} voxels)".format(result.name, voxels.voxels_length))
        require(result == zeroslam.zeroslam_return_enum.zeroslam_return_failure_insufficient_data_length, "get_map_voxels reports the length")
        require(voxels.voxels_length > 0, "voxels_length > 0")
        voxel_buffer = (zeroslam.zeroslam_voxel_struct * voxels.voxels_length)()
        voxels.voxels = ctypes.cast(voxel_buffer, ctypes.POINTER(zeroslam.zeroslam_voxel_struct))
        require(system.get_map_voxels(voxels) == success, "get_map_voxels (sized)")
        for index in range(min(voxels.voxels_length, 5)):
            voxel = voxel_buffer[index]
            print("  voxel {}: corner ({:.3f}, {:.3f}, {:.3f}) size {:.3f}, {} points".format(index, voxel.x, voxel.y, voxel.z, voxels.voxel_size, voxel.points))

    require(not system.is_valid(), "closed system is invalid")
    require(system.get_timestamp(timestamp) == zeroslam.zeroslam_return_enum.zeroslam_return_failure_invalid_system, "closed system rejects calls")
    print("destroy: on with block exit")

    # A depth camera registered to the camera makes the map metric: the depth frames go with the camera frames.
    with zeroslam.system() as rgbd:
        require(rgbd.is_valid(), "create (rgb-d)")
        depth_sensor_id = 2
        depth_parameters = zeroslam.registered_depth(sensor_id, zeroslam.zeroslam_depth_encoding_enum.zeroslam_depth_encoding_float_32, 1.0)
        sensors = (zeroslam.zeroslam_sensor_rig_struct * 2)()
        sensors[0] = rig
        sensors[1].type = zeroslam.zeroslam_sensor_enum.zeroslam_sensor_depth
        sensors[1].sensor_id = depth_sensor_id
        sensors[1].parameters_length = ctypes.sizeof(depth_parameters)
        sensors[1].parameters_data = ctypes.cast(ctypes.pointer(depth_parameters), ctypes.c_void_p)
        result = rgbd.set_sensor_rig(sensors, 2)
        print("set_sensor_rig (rgb-d): {}".format(result.name))
        require(result == success, "set_sensor_rig (rgb-d)")
        metres = (ctypes.c_float * (width * height))()
        for row in range(height):
            ray_y = ((row + 0.5) - camera_parameters.centre_y) / camera_parameters.focal_y
            depth = 6.0 if (abs(ray_y) * 6.0) <= 1.0 else (1.0 / abs(ray_y))
            for column in range(width):
                metres[(row * width) + column] = depth
        for frame_index in range(frame_count):
            pixels = render_corridor(width, height, camera_parameters, 0.05 * frame_index)
            buffer = (ctypes.c_ubyte * len(pixels)).from_buffer(pixels)
            data = (zeroslam.zeroslam_sensor_data_struct * 2)()
            data[0].timestamp = 1000000000 + (frame_interval * frame_index)
            data[0].sensor_id = depth_sensor_id
            data[0].measurement_length = width * height * ctypes.sizeof(ctypes.c_float)
            data[0].measurement_data = ctypes.cast(metres, ctypes.c_void_p)
            data[1].timestamp = data[0].timestamp
            data[1].sensor_id = sensor_id
            data[1].measurement_length = width * height
            data[1].measurement_data = ctypes.cast(buffer, ctypes.c_void_p)
            require(rgbd.set_sensor_data(data, 2) == success, "set_sensor_data (rgb-d)")
        pose = zeroslam.zeroslam_pose_struct()
        result = rgbd.get_pose(pose)
        print("get_pose (rgb-d): {}".format(result.name))
        require(result == success, "get_pose (rgb-d)")
        print_pose("newest pose, metres", pose)
        require(0.85 < pose.pose[0] < 1.05, "the rgb-d map is metric")

    # A camera with an imu: the imu is a sensor of the rig with its noise model, posed on the body like a camera, and its samples go in
    # with the frames in time order. This camera glides at a steady speed, so the imu measures no turn and only gravity's pull (the
    # camera's y axis points down): it is taken in, and the system initialises it once the motion is excited enough.
    with zeroslam.system() as inertial:
        imu_id = 2
        imu_parameters = zeroslam.zeroslam_sensor_parameters_imu_struct(1.6968e-04, 2.0e-03, 1.9393e-05, 3.0e-03, 200.0, 0.0, 0.0)
        sensors = (zeroslam.zeroslam_sensor_rig_struct * 2)()
        sensors[0] = rig
        sensors[1].type = zeroslam.zeroslam_sensor_enum.zeroslam_sensor_imu
        sensors[1].sensor_id = imu_id
        sensors[1].parameters_length = ctypes.sizeof(imu_parameters)
        sensors[1].parameters_data = ctypes.cast(ctypes.pointer(imu_parameters), ctypes.c_void_p)
        result = inertial.set_sensor_rig(sensors[0], 2)
        print("set_sensor_rig (camera and imu): {} (imu, id {})".format(result.name, imu_id))
        require(result == success, "set_sensor_rig (camera and imu)")
        placements = (zeroslam.zeroslam_sensor_extrinsics_struct * 2)()
        placements[0] = extrinsic
        placements[1].sensor_id = imu_id
        placements[1].refinement = zeroslam.zeroslam_extrinsic_enum.zeroslam_extrinsic_fixed
        placements[1].pose[0] = 0.05
        placements[1].pose[6] = 1.0
        result = inertial.set_sensor_extrinsics(placements[0], 2)
        print("set_sensor_extrinsics (camera and imu): {} (imu 5 cm along the camera's x axis)".format(result.name))
        require(result == success, "set_sensor_extrinsics (camera and imu)")
        still = zeroslam.zeroslam_sensor_measurement_imu_struct()
        still.linear_acceleration[1] = -9.81
        imu_time = 1000000000 - 5000000
        samples = 0
        for frame_index in range(frame_count):
            frame_time = 1000000000 + (frame_interval * frame_index)
            entries = []
            while imu_time + 5000000 <= frame_time:
                imu_time += 5000000
                sample = zeroslam.zeroslam_sensor_data_struct()
                sample.timestamp = imu_time
                sample.sensor_id = imu_id
                sample.measurement_length = ctypes.sizeof(still)
                sample.measurement_data = ctypes.cast(ctypes.pointer(still), ctypes.c_void_p)
                entries.append(sample)
                samples += 1
            pixels = render_corridor(width, height, camera_parameters, 0.05 * frame_index)
            buffer = (ctypes.c_ubyte * len(pixels)).from_buffer(pixels)
            image = zeroslam.zeroslam_sensor_data_struct()
            image.timestamp = frame_time
            image.sensor_id = sensor_id
            image.measurement_length = width * height
            image.measurement_data = ctypes.cast(buffer, ctypes.c_void_p)
            entries.append(image)
            batch = (zeroslam.zeroslam_sensor_data_struct * len(entries))(*entries)
            result = inertial.set_sensor_data(batch[0], len(entries))
            require(result == success, "set_sensor_data (camera and imu)")
        print("set_sensor_data (camera and imu): {} ({} frames, {} imu samples between them)".format(result.name, frame_count, samples))
        pose = zeroslam.zeroslam_pose_struct()
        require(inertial.get_pose(pose) == success, "get_pose (camera and imu)")
        print("newest pose (camera and imu): timestamp {}, centre ({:.3f}, {:.3f}, {:.3f})".format(pose.timestamp, pose.pose[0], pose.pose[1], pose.pose[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
