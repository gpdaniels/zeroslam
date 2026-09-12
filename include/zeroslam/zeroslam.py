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

import ctypes as _ctypes
import ctypes.util as _ctypes_util
import enum as _enum
import os as _os
import sys as _sys

################################################################################

class zeroslam_system(_ctypes.Structure):
    pass

class zeroslam_return_enum(_enum.IntEnum):
    zeroslam_return_success = 0x00000000
    zeroslam_return_failure_insufficient_data_length = 0x00000001
    zeroslam_return_failure_invalid_system = 0x00000010
    zeroslam_return_failure_invalid_argument = 0x00000100
    zeroslam_return_failure_invalid_configuration = 0x00001000
    zeroslam_return_failure_invalid_rig_sensor = 0x00010000
    zeroslam_return_failure_invalid_sensor_data = 0x00100000
    zeroslam_return_invalid = -1

class zeroslam_sensor_enum(_enum.IntEnum):
    zeroslam_sensor_empty = 0x00000000
    zeroslam_sensor_map_chunk = 0x00000001
    zeroslam_sensor_local_scale = 0x00000010
    zeroslam_sensor_local_linear = 0x00000020
    zeroslam_sensor_local_angular = 0x00000040
    zeroslam_sensor_remote_range = 0x00000100
    zeroslam_sensor_remote_bearing = 0x00000200
    zeroslam_sensor_remote_description = 0x00000400
    zeroslam_sensor_camera = 0x00000800
    zeroslam_sensor_invalid = -1


class zeroslam_sensor_parameters_camera_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("width", _ctypes.c_int),
        ("height", _ctypes.c_int),
        ("focal_x", _ctypes.c_double),
        ("focal_y", _ctypes.c_double),
        ("centre_x", _ctypes.c_double),
        ("centre_y", _ctypes.c_double),
        ("distortion", _ctypes.c_double * 8)
    ]

class zeroslam_sensor_rig_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("type", _ctypes.c_int),
        ("sensor_id", _ctypes.c_int),
        ("parameters_length", _ctypes.c_int),
        ("parameters_data", _ctypes.c_void_p)
    ]

class zeroslam_sensor_data_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp", _ctypes.c_int64),
        ("sensor_id", _ctypes.c_int),
        ("measurement_length", _ctypes.c_int),
        ("measurement_data", _ctypes.c_void_p)
    ]

class zeroslam_pose_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp", _ctypes.c_int64),
        ("pose", _ctypes.c_double * 7),
        ("covariance", _ctypes.c_double * (7 * 7))
    ]

class zeroslam_point_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("x", _ctypes.c_float),
        ("y", _ctypes.c_float),
        ("z", _ctypes.c_float),
        ("confidence", _ctypes.c_float),
        ("r", _ctypes.c_float),
        ("g", _ctypes.c_float),
        ("b", _ctypes.c_float),
        ("a", _ctypes.c_float)
    ]

class zeroslam_map_chunk_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp", _ctypes.c_int64),
        ("min_x", _ctypes.c_float),
        ("min_y", _ctypes.c_float),
        ("min_z", _ctypes.c_float),
        ("max_x", _ctypes.c_float),
        ("max_y", _ctypes.c_float),
        ("max_z", _ctypes.c_float),
        ("points_length", _ctypes.c_int),
        ("points", _ctypes.POINTER(zeroslam_point_struct))
    ]

class zeroslam_line_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("x1", _ctypes.c_float),
        ("y1", _ctypes.c_float),
        ("z1", _ctypes.c_float),
        ("x2", _ctypes.c_float),
        ("y2", _ctypes.c_float),
        ("z2", _ctypes.c_float),
        ("confidence", _ctypes.c_float),
        ("r", _ctypes.c_float),
        ("g", _ctypes.c_float),
        ("b", _ctypes.c_float),
        ("a", _ctypes.c_float)
    ]

class zeroslam_map_lines_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp", _ctypes.c_int64),
        ("lines_length", _ctypes.c_int),
        ("reserved", _ctypes.c_int),
        ("lines", _ctypes.POINTER(zeroslam_line_struct))
    ]

class zeroslam_edge_enum(_enum.IntEnum):
    covisibility = 0
    loop = 1

class zeroslam_edge_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp_a", _ctypes.c_int64),
        ("timestamp_b", _ctypes.c_int64),
        ("type", _ctypes.c_int),
        ("weight", _ctypes.c_int)
    ]

class zeroslam_map_keyframes_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp", _ctypes.c_int64),
        ("keyframes_length", _ctypes.c_int),
        ("reserved", _ctypes.c_int),
        ("keyframes", _ctypes.POINTER(_ctypes.c_int64))
    ]

class zeroslam_map_edges_struct(_ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("timestamp", _ctypes.c_int64),
        ("edges_length", _ctypes.c_int),
        ("reserved", _ctypes.c_int),
        ("edges", _ctypes.POINTER(zeroslam_edge_struct))
    ]

assert _ctypes.sizeof(zeroslam_sensor_parameters_camera_struct) == 104
assert _ctypes.sizeof(zeroslam_sensor_rig_struct) == 12 + _ctypes.sizeof(_ctypes.c_void_p)
assert _ctypes.sizeof(zeroslam_sensor_data_struct) == 16 + _ctypes.sizeof(_ctypes.c_void_p)
assert _ctypes.sizeof(zeroslam_pose_struct) == 456
assert _ctypes.sizeof(zeroslam_point_struct) == 32
assert _ctypes.sizeof(zeroslam_map_chunk_struct) == 36 + _ctypes.sizeof(_ctypes.c_void_p)
assert _ctypes.sizeof(zeroslam_line_struct) == 44
assert _ctypes.sizeof(zeroslam_map_lines_struct) == 16 + _ctypes.sizeof(_ctypes.c_void_p)
assert _ctypes.sizeof(zeroslam_edge_struct) == 24
assert _ctypes.sizeof(zeroslam_map_edges_struct) == 16 + _ctypes.sizeof(_ctypes.c_void_p)
assert _ctypes.sizeof(zeroslam_map_keyframes_struct) == 16 + _ctypes.sizeof(_ctypes.c_void_p)

################################################################################

def _library_file_name():
    if _sys.platform.startswith("win"):
        return "zeroslam.dll"
    if _sys.platform == "darwin":
        return "libzeroslam.dylib"
    return "libzeroslam.so"

def _locate_library():
    candidates = []
    environment = _os.environ.get("ZEROSLAM_API_LIBRARY")
    if environment:
        candidates.append(environment)
    file_name = _library_file_name()
    candidates.append(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), file_name))
    candidates.append(_os.path.join(_os.getcwd(), file_name))
    for candidate in candidates:
        if _os.path.isfile(candidate):
            return candidate
    found = _ctypes_util.find_library("zeroslam")
    if found:
        return found
    raise OSError("Could not locate the zeroslam shared library; set ZEROSLAM_API_LIBRARY to its path.")

if _sys.platform.startswith("win"):
    libapi = _ctypes.WinDLL(_locate_library())
else:
    libapi = _ctypes.CDLL(_locate_library())

def _wrap(name, argtypes):
    function = getattr(libapi, name)
    function.restype = _ctypes.c_int
    function.argtypes = argtypes
    return function

zeroslam_create = _wrap("zeroslam_create", [
    _ctypes.POINTER(_ctypes.POINTER(zeroslam_system))
])

zeroslam_destroy = _wrap("zeroslam_destroy", [
    _ctypes.POINTER(_ctypes.POINTER(zeroslam_system))
])

zeroslam_get_timestamp = _wrap("zeroslam_get_timestamp", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.POINTER(_ctypes.c_int64)
])

zeroslam_get_configuration = _wrap("zeroslam_get_configuration", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.c_char_p,
    _ctypes.POINTER(_ctypes.c_int)
])

zeroslam_set_configuration = _wrap("zeroslam_set_configuration", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.c_char_p,
    _ctypes.c_int
])

zeroslam_set_sensor_rig = _wrap("zeroslam_set_sensor_rig", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.POINTER(zeroslam_sensor_rig_struct),
    _ctypes.c_int
])

zeroslam_set_sensor_data = _wrap("zeroslam_set_sensor_data", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.POINTER(zeroslam_sensor_data_struct),
    _ctypes.c_int
])

zeroslam_finalise = _wrap("zeroslam_finalise", [
    _ctypes.POINTER(zeroslam_system)
])

zeroslam_get_pose = _wrap("zeroslam_get_pose", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.POINTER(zeroslam_pose_struct)
])

zeroslam_get_pose_at_timestamp = _wrap("zeroslam_get_pose_at_timestamp", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.POINTER(zeroslam_pose_struct),
    _ctypes.c_int64
])

zeroslam_get_map_chunk = _wrap("zeroslam_get_map_chunk", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.c_float,
    _ctypes.c_float,
    _ctypes.c_float,
    _ctypes.POINTER(zeroslam_map_chunk_struct)
])

zeroslam_get_map_lines = _wrap("zeroslam_get_map_lines", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.POINTER(zeroslam_map_lines_struct)
])

zeroslam_get_map_edges = _wrap("zeroslam_get_map_edges", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.POINTER(zeroslam_map_edges_struct)
])

zeroslam_get_map_keyframes = _wrap("zeroslam_get_map_keyframes", [
    _ctypes.POINTER(zeroslam_system),
    _ctypes.POINTER(zeroslam_map_keyframes_struct)
])

################################################################################

class system:
    def __init__(self):
        self._handle = _ctypes.POINTER(zeroslam_system)()
        if zeroslam_create(_ctypes.byref(self._handle)) != zeroslam_return_enum.zeroslam_return_success:
            self._handle = _ctypes.POINTER(zeroslam_system)()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        return False

    def close(self):
        if self._handle:
            zeroslam_destroy(_ctypes.byref(self._handle))
            self._handle = _ctypes.POINTER(zeroslam_system)()

    def is_valid(self):
        return bool(self._handle)

    def get_timestamp(self, timestamp):
        return zeroslam_return_enum(zeroslam_get_timestamp(self._handle, _ctypes.byref(timestamp)))

    def get_configuration(self, configuration, length):
        return zeroslam_return_enum(zeroslam_get_configuration(self._handle, configuration, _ctypes.byref(length)))

    def set_configuration(self, configuration):
        return zeroslam_return_enum(zeroslam_set_configuration(self._handle, configuration, len(configuration)))

    def set_sensor_rig(self, rig, length):
        return zeroslam_return_enum(zeroslam_set_sensor_rig(self._handle, rig, length))

    def set_sensor_data(self, data, length):
        return zeroslam_return_enum(zeroslam_set_sensor_data(self._handle, data, length))

    def finalise(self):
        return zeroslam_return_enum(zeroslam_finalise(self._handle))

    def get_pose(self, pose):
        return zeroslam_return_enum(zeroslam_get_pose(self._handle, _ctypes.byref(pose)))

    def get_pose_at_timestamp(self, pose, timestamp):
        return zeroslam_return_enum(zeroslam_get_pose_at_timestamp(self._handle, _ctypes.byref(pose), timestamp))

    def get_map_chunk(self, x, y, z, chunk):
        return zeroslam_return_enum(zeroslam_get_map_chunk(self._handle, x, y, z, _ctypes.byref(chunk)))

    def get_map_lines(self, lines):
        return zeroslam_return_enum(zeroslam_get_map_lines(self._handle, _ctypes.byref(lines)))

    def get_map_edges(self, edges):
        return zeroslam_return_enum(zeroslam_get_map_edges(self._handle, _ctypes.byref(edges)))

    def get_map_keyframes(self, keyframes):
        return zeroslam_return_enum(zeroslam_get_map_keyframes(self._handle, _ctypes.byref(keyframes)))

__all__ = [
    "zeroslam_system",
    "zeroslam_return_enum",
    "zeroslam_sensor_enum",
    "zeroslam_sensor_parameters_camera_struct",
    "zeroslam_sensor_rig_struct",
    "zeroslam_sensor_data_struct",
    "zeroslam_pose_struct",
    "zeroslam_point_struct",
    "zeroslam_line_struct",
    "zeroslam_map_lines_struct",
    "zeroslam_edge_enum",
    "zeroslam_edge_struct",
    "zeroslam_map_edges_struct",
    "zeroslam_map_keyframes_struct",
    "zeroslam_map_chunk_struct",
    "zeroslam_create",
    "zeroslam_destroy",
    "zeroslam_get_timestamp",
    "zeroslam_get_configuration",
    "zeroslam_set_configuration",
    "zeroslam_set_sensor_rig",
    "zeroslam_set_sensor_data",
    "zeroslam_finalise",
    "zeroslam_get_pose",
    "zeroslam_get_pose_at_timestamp",
    "zeroslam_get_map_chunk",
    "zeroslam_get_map_lines",
    "zeroslam_get_map_edges",
    "zeroslam_get_map_keyframes",
    "system"
]
