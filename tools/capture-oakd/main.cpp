/*
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
*/

#include "cdr.hpp"
#include "dataset.hpp"
#include "directory.hpp"
#include "file.hpp"
#include "mcap.hpp"
#include "paths.hpp"
#include "rotation.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <depthai/depthai.hpp>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    std::atomic<bool> stop_requested(false);

    void handle_stop_signal(int) {
        stop_requested = true;
    }

    struct options {
        std::string output_path;
        unsigned int width = 1280;
        unsigned int height = 800;
        float fps = 30.0f;
        double seconds = 60.0;
        unsigned long long frames = 0;
        unsigned long long skip = 20;
        unsigned int imu_rate = 200;
        bool raw = false;
        bool depth = true;
        bool laser = true;
        bool imu = true;
    };

    using rigid = rotation::rigid;

    bool to_rigid(const std::vector<std::vector<float>>& matrix, rigid& result) {
        if ((matrix.size() < 3) || (matrix[0].size() < 3)) {
            return false;
        }
        const bool homogeneous = (matrix[0].size() >= 4);
        for (int row = 0; row < 3; ++row) {
            if (matrix[static_cast<std::size_t>(row)].size() < (homogeneous ? 4u : 3u)) {
                return false;
            }
            for (int column = 0; column < 3; ++column) {
                result.rotation[(3 * row) + column] = static_cast<double>(matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(column)]);
            }
            result.translation[row] = homogeneous ? static_cast<double>(matrix[static_cast<std::size_t>(row)][3]) : 0.0;
        }
        return true;
    }

    double rotation_angle(const rigid& transform) {
        const double trace = transform.rotation[0] + transform.rotation[4] + transform.rotation[8];
        return std::acos(std::max(-1.0, std::min(1.0, (trace - 1.0) * 0.5)));
    }

    long long to_epoch_nanoseconds(const std::chrono::time_point<std::chrono::steady_clock, std::chrono::steady_clock::duration>& timestamp, const long long epoch_offset_nanoseconds) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(timestamp.time_since_epoch()).count() + epoch_offset_nanoseconds;
    }

    struct camera_stream {
        std::string name;
        unsigned short image_channel = 0;
        unsigned short info_channel = 0;
        unsigned int sequence = 0;
        long long last_timestamp_nanoseconds = -1;
        double extrinsic_translation[3] = {};
        double extrinsic_rotation_xyzw[4] = { 0.0, 0.0, 0.0, 1.0 };
    };

    void write_frame(
        mcap::writer& writer,
        const unsigned short tf_channel,
        camera_stream& stream,
        const long long timestamp_nanoseconds,
        const unsigned int width,
        const unsigned int height,
        const char* const encoding,
        const unsigned int bytes_per_pixel,
        const unsigned char* const pixels,
        const double intrinsics[4],
        const std::vector<double>& distortion,
        const char* const distortion_model
    ) {
        const cdr::time stamp = cdr::time::from_nanoseconds(timestamp_nanoseconds);
        const unsigned long long log_time = static_cast<unsigned long long>(timestamp_nanoseconds);
        const std::string frame_id = "sensor/" + stream.name;

        cdr::image image;
        image.frame_header.stamp = stamp;
        image.frame_header.frame_id = frame_id;
        image.width = width;
        image.height = height;
        image.encoding = encoding;
        image.is_bigendian = 0;
        image.step = width * bytes_per_pixel;
        image.data.assign(pixels, pixels + (static_cast<std::size_t>(width) * height * bytes_per_pixel));
        const std::vector<unsigned char> image_payload = cdr::write_image(image);
        writer.add_message(stream.image_channel, stream.sequence, log_time, log_time, image_payload.data(), image_payload.size());

        cdr::camera_info information;
        information.frame_header.stamp = stamp;
        information.frame_header.frame_id = frame_id;
        information.width = width;
        information.height = height;
        information.distortion_model = distortion_model;
        information.d = distortion;
        information.k[0] = intrinsics[0];
        information.k[4] = intrinsics[1];
        information.k[2] = intrinsics[2];
        information.k[5] = intrinsics[3];
        information.k[8] = 1.0;
        information.r[0] = 1.0;
        information.r[4] = 1.0;
        information.r[8] = 1.0;
        information.p[0] = intrinsics[0];
        information.p[2] = intrinsics[2];
        information.p[5] = intrinsics[1];
        information.p[6] = intrinsics[3];
        information.p[10] = 1.0;
        const std::vector<unsigned char> info_payload = cdr::write_camera_info(information);
        writer.add_message(stream.info_channel, stream.sequence, log_time, log_time, info_payload.data(), info_payload.size());

        const std::vector<unsigned char> transform = cdr::write_tf_message(stamp, "ego", frame_id, &stream.extrinsic_translation[0], &stream.extrinsic_rotation_xyzw[0]);
        writer.add_message(tf_channel, stream.sequence, log_time, log_time, transform.data(), transform.size());

        ++stream.sequence;
        stream.last_timestamp_nanoseconds = timestamp_nanoseconds;
    }

    bool packed_rows(const dai::ImgFrame& frame, const std::size_t bytes_per_pixel, std::vector<unsigned char>& packed, std::string& error) {
        const dai::span<const std::uint8_t> data = frame.getData();
        const std::size_t width = frame.getWidth();
        const std::size_t height = frame.getHeight();
        const std::size_t row_bytes = width * bytes_per_pixel;
        const std::size_t stride = frame.getStride();
        const std::size_t offset = frame.fb.p1Offset;
        if (stride < row_bytes) {
            error = "a frame has a row stride of " + std::to_string(stride) + " bytes for " + std::to_string(row_bytes) + " bytes of pixels";
            return false;
        }
        if ((height > 0) && (data.size() < (offset + (stride * (height - 1)) + row_bytes))) {
            error = "a frame holds " + std::to_string(data.size()) + " bytes for " + std::to_string(height) + " rows with a stride of " + std::to_string(stride) + " bytes";
            return false;
        }
        packed.resize(row_bytes * height);
        for (std::size_t row = 0; row < height; ++row) {
            std::memcpy(packed.data() + (row * row_bytes), data.data() + offset + (row * stride), row_bytes);
        }
        return true;
    }

    bool mono_pixels(const dai::ImgFrame& frame, std::vector<unsigned char>& pixels, std::string& error) {
        if ((frame.getType() != dai::ImgFrame::Type::GRAY8) && (frame.getType() != dai::ImgFrame::Type::RAW8)) {
            error = "the camera returned a frame that is neither GRAY8 nor RAW8";
            return false;
        }
        return packed_rows(frame, 1, pixels, error);
    }

    void frame_intrinsics(const dai::ImgFrame& frame, double intrinsics[4]) {
        const std::array<std::array<float, 3>, 3> matrix = frame.transformation.getIntrinsicMatrix();
        intrinsics[0] = static_cast<double>(matrix[0][0]);
        intrinsics[1] = static_cast<double>(matrix[1][1]);
        intrinsics[2] = static_cast<double>(matrix[0][2]) + 0.5;
        intrinsics[3] = static_cast<double>(matrix[1][2]) + 0.5;
    }

    void print_usage(const char* argv0) {
        std::printf("Usage %s [output.mcap] [options...]\n", argv0);
        std::printf("    output.mcap - Where to write the captured scene.\n");
        std::printf("    options:\n");
        std::printf("        --width [n]     - Mono camera output width (default: 1280).\n");
        std::printf("        --height [n]    - Mono camera output height (default: 800).\n");
        std::printf("        --fps [n]       - Camera frame rate (default: 30).\n");
        std::printf("        --seconds [n]   - Stop after this many seconds (default: 60, 0 for no limit).\n");
        std::printf("        --frames [n]    - Also stop after this many stereo frames (default: no limit).\n");
        std::printf("        --skip [n]      - Drop the first n stereo frames while the auto exposure settles (default: 20).\n");
        std::printf("        --imu-rate [n]  - Inertial report rate in hz (default: 200).\n");
        std::printf("        --raw           - Write the sensors' own unrectified frames with their\n");
        std::printf("                          distortion coefficients, and no depth.\n");
        std::printf("        --no-depth      - Do not record the depth camera.\n");
        std::printf("        --no-laser      - Do not record the depth camera using the laser.\n");
        std::printf("        --no-imu        - Do not record the inertial sensor.\n");
        std::printf("Ctrl-C ends the capture and writes out what was recorded. The ego frame is the left\n");
        std::printf("camera (x right, y down, z forward): image_01 is the left camera, image_02 the right,\n");
        std::printf("depth_01 the depth in metres aligned to image_01, imu_01 the accelerometer and\n");
        std::printf("gyroscope, and magnetometer_01 the magnetic field in tesla (BNO086 devices only).\n");
    }
}

int main(int argc, char* argv[]) {
    options settings;

    for (int i = 1; i < argc; ++i) {
        const auto matches = [&](const char* name) {
            return std::strcmp(argv[i], name) == 0;
        };
        const auto value_of = [&](const char* name, double& destination) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: %s\n", name);
                return false;
            }
            destination = std::strtod(argv[++i], nullptr);
            return true;
        };
        double value = 0.0;
        if (matches("--help") || matches("-h")) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (matches("--width")) {
            if (!value_of("--width", value)) {
                return EXIT_FAILURE;
            }
            settings.width = static_cast<unsigned int>(value);
        }
        else if (matches("--height")) {
            if (!value_of("--height", value)) {
                return EXIT_FAILURE;
            }
            settings.height = static_cast<unsigned int>(value);
        }
        else if (matches("--fps")) {
            if (!value_of("--fps", value)) {
                return EXIT_FAILURE;
            }
            settings.fps = static_cast<float>(value);
        }
        else if (matches("--seconds")) {
            if (!value_of("--seconds", value)) {
                return EXIT_FAILURE;
            }
            settings.seconds = value;
        }
        else if (matches("--frames")) {
            if (!value_of("--frames", value)) {
                return EXIT_FAILURE;
            }
            settings.frames = static_cast<unsigned long long>(value);
        }
        else if (matches("--skip")) {
            if (!value_of("--skip", value)) {
                return EXIT_FAILURE;
            }
            settings.skip = static_cast<unsigned long long>(value);
        }
        else if (matches("--imu-rate")) {
            if (!value_of("--imu-rate", value)) {
                return EXIT_FAILURE;
            }
            settings.imu_rate = static_cast<unsigned int>(value);
        }
        else if (matches("--raw")) {
            settings.raw = true;
            settings.depth = false;
            settings.laser = false;
        }
        else if (matches("--no-depth")) {
            settings.depth = false;
            settings.laser = false;
        }
        else if (matches("--no-laser")) {
            settings.laser = false;
        }
        else if (matches("--no-imu")) {
            settings.imu = false;
        }
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        else if (settings.output_path.empty()) {
            settings.output_path = argv[i];
        }
        else {
            std::fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }
    if (settings.output_path.empty()) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if ((settings.width < 2) || (settings.height < 2) || (settings.fps <= 0.0f)) {
        std::fprintf(stderr, "The camera resolution and frame rate must all be positive.\n");
        return EXIT_FAILURE;
    }

    std::signal(SIGINT, handle_stop_signal);
    std::signal(SIGTERM, handle_stop_signal);

    try {
        std::printf("Opening the device...\n");
        std::shared_ptr<dai::Device> device = std::make_shared<dai::Device>();
        const std::string imu_name = device->getConnectedIMU();
        const dai::UsbSpeed usb_speed = device->getUsbSpeed();
        const char* usb_speed_name = "unknown";
        switch (usb_speed) {
            case dai::UsbSpeed::LOW:
                usb_speed_name = "usb 1 low speed";
                break;
            case dai::UsbSpeed::FULL:
                usb_speed_name = "usb 1 full speed";
                break;
            case dai::UsbSpeed::HIGH:
                usb_speed_name = "usb 2 high speed";
                break;
            case dai::UsbSpeed::SUPER:
                usb_speed_name = "usb 3 super speed";
                break;
            case dai::UsbSpeed::SUPER_PLUS:
                usb_speed_name = "usb 3 super speed plus";
                break;
            case dai::UsbSpeed::UNKNOWN:
                break;
        }
        std::printf("    %s, %s, imu %s, %s\n", device->getDeviceName().c_str(), device->getDeviceId().c_str(), imu_name.c_str(), usb_speed_name);
        const bool has_magnetometer = settings.imu && (imu_name.find("BNO") != std::string::npos);
        if ((usb_speed != dai::UsbSpeed::UNKNOWN) && (usb_speed < dai::UsbSpeed::SUPER)) {
            std::fprintf(stderr, "Warning: the device is on a usb 2 link (480 Mbit/s); expect well under the requested frame rate at the default resolution.\n");
            std::fprintf(stderr, "         Use a usb 3 port and cable, or lower --width, --height or --fps.\n");
        }

        if (settings.laser) {
            device->setIrLaserDotProjectorIntensity(1.0f);
            device->setIrFloodLightIntensity(0.0f);
        }

        dai::Pipeline pipeline(device);
        std::shared_ptr<dai::node::Camera> left = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_B, std::nullopt, settings.fps);
        std::shared_ptr<dai::node::Camera> right = pipeline.create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_C, std::nullopt, settings.fps);

        const std::pair<std::uint32_t, std::uint32_t> size(settings.width, settings.height);
        dai::Node::Output* const left_output = left->requestOutput(size, dai::ImgFrame::Type::GRAY8, dai::ImgResizeMode::CROP, settings.fps);
        dai::Node::Output* const right_output = right->requestOutput(size, dai::ImgFrame::Type::GRAY8, dai::ImgResizeMode::CROP, settings.fps);
        if ((left_output == nullptr) || (right_output == nullptr)) {
            std::fprintf(stderr, "The cameras cannot deliver %ux%u at %.1f fps.\n", settings.width, settings.height, static_cast<double>(settings.fps));
            return EXIT_FAILURE;
        }

        std::shared_ptr<dai::node::Sync> sync = pipeline.create<dai::node::Sync>();
        sync->setRunOnHost(true);
        if (settings.raw) {
            left_output->link(sync->inputs["left"]);
            right_output->link(sync->inputs["right"]);
        }
        else {
            std::shared_ptr<dai::node::StereoDepth> stereo = pipeline.create<dai::node::StereoDepth>();
            left_output->link(stereo->left);
            right_output->link(stereo->right);
            stereo->setRectification(true);
            stereo->enableDistortionCorrection(true);
            stereo->setLeftRightCheck(true);
            stereo->setSubpixel(true);
            stereo->setDepthAlign(dai::StereoDepthProperties::DepthAlign::RECTIFIED_LEFT);
            stereo->initialConfig->setDepthUnit(dai::StereoDepthConfig::AlgorithmControl::DepthUnit::MILLIMETER);
            stereo->rectifiedLeft.link(sync->inputs["left"]);
            stereo->rectifiedRight.link(sync->inputs["right"]);
            if (settings.depth) {
                stereo->depth.link(sync->inputs["depth"]);
            }
        }
        std::shared_ptr<dai::MessageQueue> frame_queue = sync->out.createOutputQueue(8, false);

        std::shared_ptr<dai::MessageQueue> imu_queue;
        if (settings.imu) {
            std::shared_ptr<dai::node::IMU> imu = pipeline.create<dai::node::IMU>();
            imu->enableIMUSensor({ dai::IMUSensor::ACCELEROMETER_RAW, dai::IMUSensor::GYROSCOPE_RAW }, settings.imu_rate);
            if (has_magnetometer) {
                imu->enableIMUSensor(dai::IMUSensor::MAGNETOMETER_RAW, std::min(settings.imu_rate, 100u));
            }
            imu->setBatchReportThreshold(1);
            imu->setMaxBatchReports(10);
            imu_queue = imu->out.createOutputQueue(64, false);
        }

        std::printf("Reading the calibration...\n");
        dai::CalibrationHandler calibration = device->readCalibration();
        rigid left_from_right;
        rigid left_from_imu;
        if (!to_rigid(calibration.getCameraExtrinsics(dai::CameraBoardSocket::CAM_C, dai::CameraBoardSocket::CAM_B, false, dai::LengthUnit::METER), left_from_right)) {
            std::fprintf(stderr, "The device calibration holds no left to right camera extrinsic.\n");
            return EXIT_FAILURE;
        }
        const bool has_imu_extrinsic = settings.imu && to_rigid(calibration.getImuToCameraExtrinsics(dai::CameraBoardSocket::CAM_B, false, dai::LengthUnit::METER), left_from_imu);
        if (settings.imu && !has_imu_extrinsic) {
            std::fprintf(stderr, "The device calibration holds no imu to left camera extrinsic.\n");
            return EXIT_FAILURE;
        }
        if (!settings.raw) {
            rigid rectified_from_left;
            rigid rectified_from_right;
            if (!to_rigid(calibration.getStereoLeftRectificationRotation(), rectified_from_left) || !to_rigid(calibration.getStereoRightRectificationRotation(), rectified_from_right)) {
                std::fprintf(stderr, "The device calibration holds no stereo rectification rotations.\n");
                return EXIT_FAILURE;
            }
            left_from_right = rectified_from_left * left_from_right * rectified_from_right.inverse();
            left_from_imu = rectified_from_left * left_from_imu;
            const double residual_degrees = rotation_angle(left_from_right) * 180.0 / 3.14159265358979323846;
            const double baseline = std::sqrt((left_from_right.translation[0] * left_from_right.translation[0]) + (left_from_right.translation[1] * left_from_right.translation[1]) + (left_from_right.translation[2] * left_from_right.translation[2]));
            const double off_axis = std::sqrt((left_from_right.translation[1] * left_from_right.translation[1]) + (left_from_right.translation[2] * left_from_right.translation[2]));
            if ((residual_degrees > 0.5) || ((baseline > 0.0) && ((off_axis / baseline) > 0.01))) {
                std::fprintf(stderr, "Warning: the rectified stereo pair is not row aligned (%.3f degrees of residual rotation, %.2f mm off the baseline axis).\n", residual_degrees, off_axis * 1000.0);
                std::fprintf(stderr, "         The frames are still correct, but image_02's extrinsic may be. Capture with --raw to sidestep the rectification.\n");
            }
        }
        std::printf("    baseline %.2f mm, imu at (%.4f, %.4f, %.4f) m.\n", left_from_right.translation[0] * 1000.0, left_from_imu.translation[0], left_from_imu.translation[1], left_from_imu.translation[2]);

        {
            const std::string parent = gtl::paths::path_parent_directory(settings.output_path);
            if (!parent.empty()) {
                gtl::directory::make_directories(parent);
            }
        }
        mcap::writer writer;
        if (!writer.begin(settings.output_path, "ros2", "zeroslam-capture-oakd", "lz4")) {
            std::fprintf(stderr, "Failed to create: %s\n", settings.output_path.c_str());
            return EXIT_FAILURE;
        }
        const unsigned short image_schema = writer.add_schema("sensor_msgs/msg/Image", "ros2msg", cdr::image_schema());
        const unsigned short info_schema = writer.add_schema("sensor_msgs/msg/CameraInfo", "ros2msg", cdr::camera_info_schema());
        const unsigned short tf_schema = writer.add_schema("tf2_msgs/msg/TFMessage", "ros2msg", cdr::tf_message_schema());
        const unsigned short imu_schema = settings.imu ? writer.add_schema("sensor_msgs/msg/Imu", "ros2msg", cdr::imu_schema()) : static_cast<unsigned short>(0);
        const unsigned short magnetic_field_schema = has_magnetometer ? writer.add_schema("sensor_msgs/msg/MagneticField", "ros2msg", cdr::magnetic_field_schema()) : static_cast<unsigned short>(0);
        const unsigned short tf_channel = writer.add_channel(tf_schema, "/tf", "cdr");

        std::vector<camera_stream> streams;
        {
            const char* const names[3] = { "image_01", "image_02", "depth_01" };
            const std::size_t count = settings.depth ? 3u : 2u;
            for (std::size_t index = 0; index < count; ++index) {
                camera_stream stream;
                stream.name = names[index];
                stream.image_channel = writer.add_channel(image_schema, "/sensor/" + stream.name, "cdr");
                stream.info_channel = writer.add_channel(info_schema, "/sensor/" + stream.name + "/camera_info", "cdr");
                streams.push_back(stream);
            }
            left_from_right.to_pose(&streams[1].extrinsic_translation[0], &streams[1].extrinsic_rotation_xyzw[0]);
        }
        unsigned short imu_channel = 0;
        unsigned short magnetometer_channel = 0;
        double imu_translation[3] = {};
        double imu_rotation_xyzw[4] = { 0.0, 0.0, 0.0, 1.0 };
        if (settings.imu) {
            imu_channel = writer.add_channel(imu_schema, "/sensor/imu_01", "cdr");
            left_from_imu.to_pose(&imu_translation[0], &imu_rotation_xyzw[0]);
        }
        if (has_magnetometer) {
            magnetometer_channel = writer.add_channel(magnetic_field_schema, "/sensor/magnetometer_01", "cdr");
        }

        const long long epoch_offset_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();

        unsigned int magnetometer_sequence = 0;

        std::printf("Capturing (ctrl-c to stop)...\n");
        pipeline.start();

        std::vector<unsigned char> left_pixels;
        std::vector<unsigned char> right_pixels;
        std::vector<unsigned char> depth_pixels;
        std::vector<float> depth_metres;
        bool warned_about_distortion = false;
        long long last_magnetometer_timestamp_nanoseconds = -1;
        unsigned int imu_sequence = 0;
        long long last_imu_timestamp_nanoseconds = -1;
        unsigned long long captured_frames = 0;
        unsigned long long skipped_frames = 0;
        long long first_kept_timestamp_nanoseconds = -1;
        std::size_t dropped_frames = 0;
        std::size_t dropped_samples = 0;
        std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point last_report = started;

        while (pipeline.isRunning() && !stop_requested) {
            if ((settings.seconds > 0.0) && (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >= settings.seconds)) {
                break;
            }
            if ((settings.frames > 0) && (captured_frames >= settings.frames)) {
                break;
            }

            bool timed_out = false;
            const std::shared_ptr<dai::MessageGroup> group = frame_queue->get<dai::MessageGroup>(std::chrono::milliseconds(20), timed_out);
            if (!timed_out && (group != nullptr)) {
                const std::shared_ptr<dai::ImgFrame> left_frame = group->get<dai::ImgFrame>("left");
                const std::shared_ptr<dai::ImgFrame> right_frame = group->get<dai::ImgFrame>("right");
                const std::shared_ptr<dai::ImgFrame> depth_frame = settings.depth ? group->get<dai::ImgFrame>("depth") : nullptr;
                if ((left_frame == nullptr) || (right_frame == nullptr) || (settings.depth && (depth_frame == nullptr))) {
                    ++dropped_frames;
                    continue;
                }
                const long long timestamp = to_epoch_nanoseconds(left_frame->getTimestamp(), epoch_offset_nanoseconds);
                if (timestamp <= streams[0].last_timestamp_nanoseconds) {
                    ++dropped_frames;
                    continue;
                }
                if (skipped_frames < settings.skip) {
                    ++skipped_frames;
                    streams[0].last_timestamp_nanoseconds = timestamp;
                    continue;
                }
                if (captured_frames == 0) {
                    started = std::chrono::steady_clock::now();
                    first_kept_timestamp_nanoseconds = timestamp;
                }

                std::string error;
                if (!mono_pixels(*left_frame, left_pixels, error) || !mono_pixels(*right_frame, right_pixels, error)) {
                    std::fprintf(stderr, "%s.\n", error.c_str());
                    return EXIT_FAILURE;
                }

                std::vector<double> left_distortion;
                std::vector<double> right_distortion;
                const char* distortion_model = "plumb_bob";
                if (settings.raw) {
                    distortion_model = "rational_polynomial";
                    const std::vector<float> coefficients[2] = { left_frame->transformation.getDistortionCoefficients(), right_frame->transformation.getDistortionCoefficients() };
                    std::vector<double>* const destinations[2] = { &left_distortion, &right_distortion };
                    for (std::size_t camera = 0; camera < 2; ++camera) {
                        for (std::size_t index = 0; (index < coefficients[camera].size()) && (index < 8); ++index) {
                            destinations[camera]->push_back(static_cast<double>(coefficients[camera][index]));
                        }
                        for (std::size_t index = 8; (index < coefficients[camera].size()) && !warned_about_distortion; ++index) {
                            if (std::fabs(static_cast<double>(coefficients[camera][index])) > 1e-9) {
                                std::fprintf(stderr, "Warning: camera %zu has thin prism or tilt distortion the scene's camera model cannot hold; it is dropped.\n", camera + 1);
                                warned_about_distortion = true;
                            }
                        }
                    }
                }

                double intrinsics[4] = {};
                frame_intrinsics(*left_frame, &intrinsics[0]);
                write_frame(writer, tf_channel, streams[0], timestamp, left_frame->getWidth(), left_frame->getHeight(), "mono8", 1, left_pixels.data(), &intrinsics[0], left_distortion, distortion_model);
                frame_intrinsics(*right_frame, &intrinsics[0]);
                write_frame(writer, tf_channel, streams[1], timestamp, right_frame->getWidth(), right_frame->getHeight(), "mono8", 1, right_pixels.data(), &intrinsics[0], right_distortion, distortion_model);

                if (settings.depth) {
                    if (depth_frame->getType() != dai::ImgFrame::Type::RAW16) {
                        std::fprintf(stderr, "The stereo node returned a depth frame that is not RAW16.\n");
                        return EXIT_FAILURE;
                    }
                    if (!packed_rows(*depth_frame, 2, depth_pixels, error)) {
                        std::fprintf(stderr, "%s.\n", error.c_str());
                        return EXIT_FAILURE;
                    }
                    const std::size_t pixels = depth_pixels.size() / 2;
                    depth_metres.resize(pixels);
                    for (std::size_t index = 0; index < pixels; ++index) {
                        unsigned short millimetres = 0;
                        std::memcpy(&millimetres, depth_pixels.data() + (index * 2), sizeof(millimetres));
                        depth_metres[index] = static_cast<float>(static_cast<double>(millimetres) / 1000.0);
                    }
                    frame_intrinsics(*depth_frame, &intrinsics[0]);
                    write_frame(writer, tf_channel, streams[2], timestamp, depth_frame->getWidth(), depth_frame->getHeight(), "32FC1", 4, reinterpret_cast<const unsigned char*>(depth_metres.data()), &intrinsics[0], std::vector<double>(), "plumb_bob");
                }
                ++captured_frames;
            }

            if (settings.imu) {
                const std::vector<std::shared_ptr<dai::IMUData>> inertial_batches = imu_queue->tryGetAll<dai::IMUData>();
                for (const std::shared_ptr<dai::IMUData>& inertial : inertial_batches) {
                    for (const dai::IMUPacket& packet : inertial->packets) {
                        const long long timestamp = to_epoch_nanoseconds(packet.acceleroMeter.getTimestamp(), epoch_offset_nanoseconds);
                        if ((first_kept_timestamp_nanoseconds < 0) || (timestamp < first_kept_timestamp_nanoseconds)) {
                            continue;
                        }
                        if (timestamp <= last_imu_timestamp_nanoseconds) {
                            ++dropped_samples;
                            continue;
                        }
                        const cdr::time stamp = cdr::time::from_nanoseconds(timestamp);
                        const unsigned long long log_time = static_cast<unsigned long long>(timestamp);
                        cdr::imu sample;
                        sample.frame_header.stamp = stamp;
                        sample.frame_header.frame_id = "sensor/imu_01";
                        sample.angular_velocity[0] = static_cast<double>(packet.gyroscope.x);
                        sample.angular_velocity[1] = static_cast<double>(packet.gyroscope.y);
                        sample.angular_velocity[2] = static_cast<double>(packet.gyroscope.z);
                        sample.linear_acceleration[0] = static_cast<double>(packet.acceleroMeter.x);
                        sample.linear_acceleration[1] = static_cast<double>(packet.acceleroMeter.y);
                        sample.linear_acceleration[2] = static_cast<double>(packet.acceleroMeter.z);
                        const std::vector<unsigned char> payload = cdr::write_imu(sample);
                        writer.add_message(imu_channel, imu_sequence, log_time, log_time, payload.data(), payload.size());
                        const std::vector<unsigned char> transform = cdr::write_tf_message(stamp, "ego", "sensor/imu_01", &imu_translation[0], &imu_rotation_xyzw[0]);
                        writer.add_message(tf_channel, imu_sequence, log_time, log_time, transform.data(), transform.size());
                        ++imu_sequence;
                        last_imu_timestamp_nanoseconds = timestamp;

                        const long long magnetometer_timestamp = has_magnetometer ? to_epoch_nanoseconds(packet.magneticField.getTimestamp(), epoch_offset_nanoseconds) : -1;
                        if (has_magnetometer && (magnetometer_timestamp > last_magnetometer_timestamp_nanoseconds)) {
                            const cdr::time magnetometer_stamp = cdr::time::from_nanoseconds(magnetometer_timestamp);
                            const unsigned long long magnetometer_log_time = static_cast<unsigned long long>(magnetometer_timestamp);
                            cdr::magnetic_field field;
                            field.frame_header.stamp = magnetometer_stamp;
                            field.frame_header.frame_id = "sensor/magnetometer_01";
                            field.field[0] = static_cast<double>(packet.magneticField.x) * 1e-6;
                            field.field[1] = static_cast<double>(packet.magneticField.y) * 1e-6;
                            field.field[2] = static_cast<double>(packet.magneticField.z) * 1e-6;
                            const std::vector<unsigned char> field_payload = cdr::write_magnetic_field(field);
                            writer.add_message(magnetometer_channel, magnetometer_sequence, magnetometer_log_time, magnetometer_log_time, field_payload.data(), field_payload.size());
                            const std::vector<unsigned char> magnetometer_transform = cdr::write_tf_message(magnetometer_stamp, "ego", "sensor/magnetometer_01", &imu_translation[0], &imu_rotation_xyzw[0]);
                            writer.add_message(tf_channel, magnetometer_sequence, magnetometer_log_time, magnetometer_log_time, magnetometer_transform.data(), magnetometer_transform.size());
                            ++magnetometer_sequence;
                            last_magnetometer_timestamp_nanoseconds = magnetometer_timestamp;
                        }
                    }
                }
            }

            const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - last_report).count() >= 1.0) {
                last_report = now;
                std::printf("    %.1f s, %llu frames, %u imu samples\n", std::chrono::duration<double>(now - started).count(), captured_frames, imu_sequence);
                std::fflush(stdout);
            }
        }

        pipeline.stop();
        pipeline.wait();

        if (captured_frames < 2) {
            std::fprintf(stderr, "Captured %llu frames, at least two are needed for a scene.\n", captured_frames);
            return EXIT_FAILURE;
        }
        if (skipped_frames > 0) {
            std::printf("    skipped the first %llu frames.\n", skipped_frames);
        }
        if ((dropped_frames > 0) || (dropped_samples > 0)) {
            std::printf("    dropped %zu incomplete frame groups and %zu out of order imu samples.\n", dropped_frames, dropped_samples);
        }

        std::printf("Finishing %s...\n", settings.output_path.c_str());
        if (!writer.finish()) {
            std::fprintf(stderr, "Failed to write: %s\n", settings.output_path.c_str());
            return EXIT_FAILURE;
        }
        std::printf("    %llu frames, %u imu samples, %u magnetometer samples, %llu bytes.\n", captured_frames, imu_sequence, magnetometer_sequence, writer.get_written());
    } catch (const std::exception& failure) {
        std::fprintf(stderr, "Capture failed: %s\n", failure.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
