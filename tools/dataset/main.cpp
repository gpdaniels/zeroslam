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

#include "dataset.hpp"
#include "directory.hpp"
#include "json.hpp"
#include "paths.hpp"
#include "process.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    // A file or directory in the hub repository listing.
    struct hub_entry {
        std::string path;
        bool directory = false;
        unsigned long long size = 0;
        std::string sha256; // The lfs content hash when present, for future verification.
    };

    // Hub paths are joined onto the datasets directory, restrict them to safe relative paths.
    bool is_safe_hub_path(const std::string& path) {
        if (path.empty() || (path.front() == '/')) {
            return false;
        }
        std::size_t component_start = 0;
        for (std::size_t index = 0; index <= path.size(); ++index) {
            if ((index == path.size()) || (path[index] == '/')) {
                const std::size_t component_length = index - component_start;
                if (component_length == 0) {
                    return false;
                }
                const std::string component = path.substr(component_start, component_length);
                if ((component == ".") || (component == "..")) {
                    return false;
                }
                component_start = index + 1;
                continue;
            }
            const char character = path[index];
            const bool valid = ((character >= 'a') && (character <= 'z')) || ((character >= 'A') && (character <= 'Z')) || ((character >= '0') && (character <= '9')) || (character == '_') || (character == '-') || (character == '.');
            if (!valid) {
                return false;
            }
        }
        return true;
    }

    // Names are "[dataset]" or "[dataset]/[scene]".
    bool is_valid_dataset_or_scene_name(const std::string& name) {
        if (!is_safe_hub_path(name)) {
            return false;
        }
        const std::size_t separator = name.find('/');
        return (separator == std::string::npos) || (name.find('/', separator + 1) == std::string::npos);
    }

    // Repository identifiers are "owner/name".
    bool is_valid_repository(const std::string& repository) {
        const std::size_t separator = repository.find('/');
        if ((separator == std::string::npos) || (repository.find('/', separator + 1) != std::string::npos)) {
            return false;
        }
        return is_safe_hub_path(repository);
    }

    std::string format_size(const unsigned long long bytes) {
        char formatted[32];
        if (bytes < 1000ull) {
            std::snprintf(&formatted[0], sizeof(formatted), "%llu B", bytes);
        }
        else if (bytes < 1000ull * 1000ull) {
            std::snprintf(&formatted[0], sizeof(formatted), "%.1f KB", static_cast<double>(bytes) / 1.0e3);
        }
        else if (bytes < 1000ull * 1000ull * 1000ull) {
            std::snprintf(&formatted[0], sizeof(formatted), "%.1f MB", static_cast<double>(bytes) / 1.0e6);
        }
        else {
            std::snprintf(&formatted[0], sizeof(formatted), "%.2f GB", static_cast<double>(bytes) / 1.0e9);
        }
        return &formatted[0];
    }

    // The base curl command shared by every transfer.
    std::string curl_command(const std::string& token) {
        std::string command = "curl --fail --silent --show-error --location --connect-timeout 30 --retry 3 --user-agent " + platform::quote_for_shell("zeroslam-dataset");
        if (!token.empty()) {
            command += " --header " + platform::quote_for_shell("Authorization: Bearer " + token);
        }
        return command;
    }

    // Fetch and parse the recursive file listing of the hub repository.
    // Note: the hub api paginates past ~1000 entries via a Link header, unhandled until needed.
    bool fetch_listing(const std::string& repository, const std::string& token, std::vector<hub_entry>& entries, std::string& error) {
        const std::string url = "https://huggingface.co/api/datasets/" + repository + "/tree/main?recursive=true";
        std::string output;
        int exit_code = -1;
        if (!platform::run_command(curl_command(token) + " " + platform::quote_for_shell(url) + " 2>&1", false, output, exit_code) || (exit_code != 0)) {
            error = platform::last_line(output);
            if (error.empty()) {
                error = "curl failed (exit code " + std::to_string(exit_code) + ")";
            }
            return false;
        }
        gtl::json document;
        if (!document.parse(output)) {
            error = "the hub listing is not valid json";
            return false;
        }
        const gtl::json::value& root = document.document();
        if (!root.is<gtl::json::value::array_type>()) {
            error = "the hub listing is not a json array";
            return false;
        }
        const auto find = [](const gtl::json::value& object, const char* key) -> const gtl::json::value* {
            const gtl::json::value::object_type& members = object.as<gtl::json::value::object_type>();
            const gtl::json::value::object_type::const_iterator iterator = members.find(key);
            return (iterator == members.end()) ? nullptr : &iterator->second;
        };
        for (const gtl::json::value& item : root.as<gtl::json::value::array_type>()) {
            if (!item.is<gtl::json::value::object_type>()) {
                continue;
            }
            const gtl::json::value* type = find(item, "type");
            const gtl::json::value* path = find(item, "path");
            const gtl::json::value* size = find(item, "size");
            if ((type == nullptr) || !type->is<gtl::json::value::string_type>() || (path == nullptr) || !path->is<gtl::json::value::string_type>()) {
                continue;
            }
            hub_entry entry;
            entry.path = path->as<gtl::json::value::string_type>();
            entry.directory = (type->as<gtl::json::value::string_type>() == "directory");
            if ((size != nullptr) && size->is<gtl::json::value::number_type>()) {
                entry.size = static_cast<unsigned long long>(size->as<gtl::json::value::number_type>());
            }
            const gtl::json::value* lfs = find(item, "lfs");
            if ((lfs != nullptr) && lfs->is<gtl::json::value::object_type>()) {
                const gtl::json::value* oid = find(*lfs, "oid");
                if ((oid != nullptr) && oid->is<gtl::json::value::string_type>()) {
                    entry.sha256 = oid->as<gtl::json::value::string_type>();
                }
            }
            entries.push_back(entry);
        }
        return true;
    }

    // Download one file to '[local].part' and rename it into place after a size check.
    // The first attempt resumes any partial download, a corrupt or stale part triggers
    // one fresh attempt from scratch.
    bool download_file(const std::string& base_command, const std::string& url, const std::string& local, const unsigned long long expected_size, std::string& error) {
        const std::string part = local + ".part";
        for (int attempt = 0; attempt < 2; ++attempt) {
            std::string command = base_command;
            if (attempt == 0) {
                command += " --continue-at -";
            }
            else {
                std::remove(part.c_str());
            }
            command += " --output " + platform::quote_for_shell(part) + " " + platform::quote_for_shell(url) + " 2>&1";
            std::string output;
            int exit_code = -1;
            if (!platform::run_command(command, false, output, exit_code) || (exit_code != 0)) {
                error = platform::last_line(output);
                if (error.empty()) {
                    error = "curl failed (exit code " + std::to_string(exit_code) + ")";
                }
                continue;
            }
            unsigned long long actual_size = 0;
            if (!gtl::paths::get_file_size(part, actual_size) || (actual_size != expected_size)) {
                error = "size mismatch, expected " + std::to_string(expected_size) + " bytes, downloaded " + std::to_string(actual_size);
                continue;
            }
            std::remove(local.c_str());
            if (std::rename(part.c_str(), local.c_str()) != 0) {
                error = "failed to move the downloaded file into place";
                break;
            }
            return true;
        }
        std::remove(part.c_str());
        return false;
    }

    bool load_scene(const std::string& path, mcap& reader, dataset::mcap_scene_information& scene, std::string& error, const bool decode_every_image = true) {
        if (!reader.open(path, error)) {
            error = "cannot open '" + path + "': " + error;
            return false;
        }
        return dataset::inspect_mcap_scene(reader, scene, error, decode_every_image);
    }

    // Whether a camera info holds exactly the pinhole projection of its own intrinsics, with
    // no rectification (R identity, P == [K|0]): the values may change frame to frame, but
    bool is_valid_pinhole_calibration(const cdr::camera_info& information, const unsigned int width, const unsigned int height) {
        double k[9] = {};
        k[0] = information.k[0];
        k[2] = information.k[2];
        k[4] = information.k[4];
        k[5] = information.k[5];
        k[8] = 1.0;
        double r[9] = {};
        r[0] = 1.0;
        r[4] = 1.0;
        r[8] = 1.0;
        double p[12] = {};
        p[0] = information.k[0];
        p[2] = information.k[2];
        p[5] = information.k[4];
        p[6] = information.k[5];
        p[10] = 1.0;
        bool calibrated = (information.k[0] > 0.0) && (information.k[4] > 0.0);
        calibrated = calibrated && (information.width == width) && (information.height == height);
        if (information.distortion_model == "plumb_bob") {
            calibrated = calibrated && (information.d.empty() || (information.d.size() == 4) || (information.d.size() == 5));
        }
        else if (information.distortion_model == "rational_polynomial") {
            calibrated = calibrated && (information.d.size() == 8);
        }
        else {
            calibrated = false;
        }
        for (int i = 0; i < 9; ++i) {
            calibrated = calibrated && (information.k[i] == k[i]) && (information.r[i] == r[i]);
        }
        for (int i = 0; i < 12; ++i) {
            calibrated = calibrated && (information.p[i] == p[i]);
        }
        calibrated = calibrated && (information.binning_x == 0) && (information.binning_y == 0);
        calibrated = calibrated && (information.roi_x_offset == 0) && (information.roi_y_offset == 0) && (information.roi_height == 0) && (information.roi_width == 0) && (information.roi_do_rectify == 0);
        return calibrated;
    }

    bool marks_absent(const double covariance[9]) {
        return covariance[0] == -1.0;
    }

    // Check that a scene mcap holds exactly what the directory form represents, so a round
    // trip through expand and collapse reproduces the file: every camera holds one
    // calibration per frame logged at the frame times and holding exactly the pinhole
    bool check_scene_round_trip(const dataset::mcap_scene_information& scene, const bool require_transforms, std::string& error) {
        if (scene.cameras.empty()) {
            error = "the scene has no camera";
            return false;
        }
        if (!scene.attachments.empty()) {
            error = "the scene holds attachments the directory form cannot represent:";
            for (const std::string& name : scene.attachments) {
                error += " " + name;
            }
            return false;
        }
        if (!scene.extra_topics.empty()) {
            error = "the scene holds topics the directory form cannot represent:";
            for (const std::string& topic : scene.extra_topics) {
                error += " " + topic;
            }
            return false;
        }
        if (!scene.dynamics_log_times_consistent) {
            error = "a /tf message log time does not match its own header timestamp";
            return false;
        }
        // 'sensor_edge_log_times' below.
        std::vector<std::string> sensor_names;
        for (const dataset::camera_information& camera : scene.cameras) {
            if (camera.camera_name.empty()) {
                error = "an image topic gives the camera no name";
                return false;
            }
            if (!dataset::is_valid_sensor_name(camera.camera_name, "image") && !dataset::is_valid_sensor_name(camera.camera_name, "depth")) {
                error = "the camera name '" + camera.camera_name + "' is not 'image_01' to 'image_99' or 'depth_01' to 'depth_99'";
                return false;
            }
            if (camera.image_topic != ("/sensor/" + camera.camera_name)) {
                error = "the image topic '" + camera.image_topic + "' is not '/sensor/" + camera.camera_name + "'";
                return false;
            }
            if (!camera.camera_info_topic.empty() && (camera.camera_info_topic != (camera.image_topic + "/camera_info"))) {
                error = "the camera info topic '" + camera.camera_info_topic + "' is not '" + camera.image_topic + "/camera_info'";
                return false;
            }
            if (camera.camera_infos != camera.frames) {
                error = "camera '" + camera.camera_name + "' has " + std::to_string(camera.camera_infos) + " camera info messages for " + std::to_string(camera.frames) + " frames, one per frame is required";
                return false;
            }
            if (camera.camera_info_log_times != camera.image_log_times) {
                error = "camera '" + camera.camera_name + "'s camera info messages are not logged at the frame times";
                return false;
            }
            if (!camera.frame_ids_consistent) {
                error = "a message frame id does not match sensor '" + camera.camera_name + "'";
                return false;
            }
            if (!camera.log_times_consistent) {
                error = "a '" + camera.camera_name + "' message log time does not match its own header timestamp";
                return false;
            }
            for (const cdr::camera_info& information : camera.camera_infos_data) {
                if (!is_valid_pinhole_calibration(information, camera.width, camera.height)) {
                    error = "a '" + camera.camera_name + "' camera info does not hold exactly the pinhole projection of its intrinsics";
                    return false;
                }
            }
            sensor_names.push_back(camera.camera_name);
        }
        for (const dataset::imu_information& imu : scene.imus) {
            if (imu.imu_name.empty()) {
                error = "an imu topic gives the sensor no name";
                return false;
            }
            const dataset::inertial_type type = dataset::inertial_type_of(imu.imu_name);
            if (type == dataset::inertial_type::unknown) {
                error = "the inertial sensor name '" + imu.imu_name + "' is not 'imu_01' to 'imu_99', 'accelerometer_01' to 'accelerometer_99', or 'gyroscope_01' to 'gyroscope_99'";
                return false;
            }
            if (imu.imu_topic != ("/sensor/" + imu.imu_name)) {
                error = "the imu topic '" + imu.imu_topic + "' is not '/sensor/" + imu.imu_name + "'";
                return false;
            }
            if (!imu.frame_ids_consistent) {
                error = "a message frame id does not match sensor '" + imu.imu_name + "'";
                return false;
            }
            if (!imu.log_times_consistent) {
                error = "an '" + imu.imu_name + "' message log time does not match its own header timestamp";
                return false;
            }
            for (const cdr::imu& sample : imu.imu_data) {
                if (marks_absent(sample.angular_velocity_covariance) == dataset::measures_angular_velocity(type)) {
                    error = "'" + imu.imu_name + "' " + (dataset::measures_angular_velocity(type) ? "lacks the angular velocity its kind measures" : "carries an angular velocity its kind does not measure");
                    return false;
                }
                if (marks_absent(sample.linear_acceleration_covariance) == dataset::measures_linear_acceleration(type)) {
                    error = "'" + imu.imu_name + "' " + (dataset::measures_linear_acceleration(type) ? "lacks the linear acceleration its kind measures" : "carries a linear acceleration its kind does not measure");
                    return false;
                }
            }
            sensor_names.push_back(imu.imu_name);
        }
        // The /tf transforms are the ground truth frame tree, root -> ego -> sensor/[name] for
        // every sensor: the root -> ego transforms carry the trajectory of the body, and the
        // per message ego -> sensor/[name] extrinsics pose each sensor on it (so they can
        // change over time).
        std::vector<std::vector<unsigned long long>> sensor_edge_log_times(sensor_names.size());
        for (std::size_t i = 0; i < scene.dynamics.size(); ++i) {
            const cdr::transform_stamped& transform = scene.dynamics[i];
            if ((transform.frame_header.frame_id == "root") && (transform.child_frame_id == "ego")) {
                continue;
            }
            bool matched = false;
            if (transform.frame_header.frame_id == "ego") {
                for (std::size_t sensor = 0; sensor < sensor_names.size(); ++sensor) {
                    if (transform.child_frame_id == ("sensor/" + sensor_names[sensor])) {
                        sensor_edge_log_times[sensor].push_back(scene.dynamic_log_times[i]);
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) {
                error = "the transform '" + transform.frame_header.frame_id + "' -> '" + transform.child_frame_id + "' is not part of the scene frame tree";
                return false;
            }
        }
        for (std::size_t index = 0; index < scene.cameras.size(); ++index) {
            const dataset::camera_information& camera = scene.cameras[index];
            const std::vector<unsigned long long>& edges = sensor_edge_log_times[index];
            if (require_transforms && (edges.size() != camera.frames)) {
                error = "there are " + std::to_string(edges.size()) + " ego -> sensor/" + camera.camera_name + " transforms for " + std::to_string(camera.frames) + " frames, one per frame is required";
                return false;
            }
            if (!edges.empty() && (edges != camera.image_log_times)) {
                error = "the ego -> sensor/" + camera.camera_name + " transforms are not logged at the frame times";
                return false;
            }
        }
        for (std::size_t index = 0; index < scene.imus.size(); ++index) {
            const dataset::imu_information& imu = scene.imus[index];
            const std::vector<unsigned long long>& edges = sensor_edge_log_times[scene.cameras.size() + index];
            if (require_transforms && (edges.size() != imu.samples)) {
                error = "there are " + std::to_string(edges.size()) + " ego -> sensor/" + imu.imu_name + " transforms for " + std::to_string(imu.samples) + " samples, one per sample is required";
                return false;
            }
            if (!edges.empty() && (edges != imu.imu_log_times)) {
                error = "the ego -> sensor/" + imu.imu_name + " transforms are not logged at the sample times";
                return false;
            }
        }
        return true;
    }

    // Whether any timestamp in a stream is earlier than its predecessor.
    bool timestamps_go_backwards(const std::vector<unsigned long long>& times) {
        for (std::size_t index = 1; index < times.size(); ++index) {
            if (times[index] < times[index - 1]) {
                return true;
            }
        }
        return false;
    }

    bool validate_scene(const std::string& path) {
        std::printf("Validating '%s'...\n", path.c_str());
        std::fflush(stdout);
        mcap reader;
        dataset::mcap_scene_information scene;
        std::string error;
        if (!load_scene(path, reader, scene, error)) {
            std::fprintf(stderr, "Invalid scene: %s: %s.\n", path.c_str(), error.c_str());
            return false;
        }
        for (const dataset::camera_information& camera : scene.cameras) {
            if (timestamps_go_backwards(camera.image_log_times)) {
                std::fprintf(stderr, "Invalid scene: camera '%s' image timestamps go backwards.\n", camera.camera_name.c_str());
                return false;
            }
            if (timestamps_go_backwards(camera.camera_info_log_times)) {
                std::fprintf(stderr, "Invalid scene: camera '%s' calibration timestamps go backwards.\n", camera.camera_name.c_str());
                return false;
            }
        }
        for (const dataset::imu_information& imu : scene.imus) {
            if (timestamps_go_backwards(imu.imu_log_times)) {
                std::fprintf(stderr, "Invalid scene: imu '%s' timestamps go backwards.\n", imu.imu_name.c_str());
                return false;
            }
        }
        // The /tf topic multiplexes independent edge streams ('root -> ego', 'ego ->
        // sensor/[name]') that need not interleave in time, so ordering is judged per edge.
        std::vector<std::pair<std::string, std::string>> edge_names;
        std::vector<long long> edge_last_timestamps;
        for (const cdr::transform_stamped& transform : scene.dynamics) {
            const long long stamp = transform.frame_header.stamp.nanoseconds();
            std::size_t edge = 0;
            while ((edge < edge_names.size()) && ((edge_names[edge].first != transform.frame_header.frame_id) || (edge_names[edge].second != transform.child_frame_id))) {
                ++edge;
            }
            if (edge == edge_names.size()) {
                edge_names.push_back({ transform.frame_header.frame_id, transform.child_frame_id });
                edge_last_timestamps.push_back(stamp);
                continue;
            }
            if (stamp < edge_last_timestamps[edge]) {
                std::fprintf(stderr, "Invalid scene: /tf timestamps go backwards on '%s' -> '%s'.\n", transform.frame_header.frame_id.c_str(), transform.child_frame_id.c_str());
                return false;
            }
            edge_last_timestamps[edge] = stamp;
        }
        const dataset::camera_information& primary = scene.primary();
        for (const dataset::camera_information& camera : scene.cameras) {
            if (camera.frames < 2) {
                std::fprintf(stderr, "Invalid scene: camera '%s' has fewer than two frames.\n", camera.camera_name.c_str());
                return false;
            }
            if (!camera.camera_info || (camera.fx <= 0.0) || (camera.fy <= 0.0)) {
                std::fprintf(stderr, "Invalid scene: camera '%s' has no usable intrinsics.\n", camera.camera_name.c_str());
                return false;
            }
            if (camera.frames != primary.frames) {
                std::fprintf(stderr, "Invalid scene: camera '%s' has %zu frames, camera '%s' has %zu; every camera must share the same frame count.\n", camera.camera_name.c_str(), camera.frames, primary.camera_name.c_str(), primary.frames);
                return false;
            }
        }
        if ((scene.poses != 0) && (scene.poses != primary.frames)) {
            std::fprintf(stderr, "Invalid scene: %zu root -> ego transforms for %zu frames, a ground truth must cover every frame or be absent.\n", scene.poses, primary.frames);
            return false;
        }
        {
            const std::string primary_child = "sensor/" + primary.camera_name;
            for (const cdr::transform_stamped& transform : scene.dynamics) {
                if ((transform.frame_header.frame_id != "ego") || (transform.child_frame_id != primary_child)) {
                    continue;
                }
                const bool identity =
                    (transform.translation[0] == 0.0) && (transform.translation[1] == 0.0) && (transform.translation[2] == 0.0) &&
                    (transform.rotation[0] == 0.0) && (transform.rotation[1] == 0.0) && (transform.rotation[2] == 0.0) && (transform.rotation[3] == 1.0);
                if (!identity) {
                    std::fprintf(stderr, "Invalid scene: the primary camera '%s' is not the ego frame, its ego -> %s extrinsic is not the identity; re-import the scene so the ground truth is the primary camera's pose.\n", primary.camera_name.c_str(), primary_child.c_str());
                    return false;
                }
            }
        }
        if (!check_scene_round_trip(scene, true, error)) {
            std::fprintf(stderr, "Invalid scene: %s.\n", error.c_str());
            return false;
        }
        for (const dataset::camera_information& camera : scene.cameras) {
            const bool distorted = !camera.camera_infos_data.empty() && !camera.camera_infos_data.front().d.empty();
            const char* model = camera.camera_infos_data.empty() ? "unknown" : camera.camera_infos_data.front().distortion_model.c_str();
            std::printf("    camera:       %s (%s [%.10g %.10g %.10g %.10g]", camera.camera_name.c_str(), model, camera.fx, camera.fy, camera.cx, camera.cy);
            if (distorted) {
                const std::vector<double>& d = camera.camera_infos_data.front().d;
                std::printf(", d [");
                for (std::size_t index = 0; index < d.size(); ++index) {
                    std::printf("%s%.10g", (index == 0) ? "" : " ", d[index]);
                }
                std::printf("]");
            }
            std::printf(")%s, %zu %s frames of %ux%u\n", distorted ? ", raw distorted frames" : "", camera.frames, camera.encoding.c_str(), camera.width, camera.height);
        }
        for (const dataset::imu_information& imu : scene.imus) {
            std::printf("    imu:          %s, %zu samples\n", imu.imu_name.c_str(), imu.samples);
        }
        std::printf("    frame tree:   root -> ego -> sensor/[name] (%zu transforms on /tf)\n", scene.dynamics.size());
        if (scene.poses == 0) {
            std::printf("    ground truth: none, this scene cannot be benchmarked against\n");
        }
        else {
            std::printf("    ground truth: %zu root -> ego transforms, matching the frames\n", scene.poses);
        }
        return true;
    }

    // Every 'ego -> sensor/[name]' transform, in message order.
    std::vector<const cdr::transform_stamped*> find_sensor_edges(const dataset::mcap_scene_information& scene, const std::string& sensor_name) {
        std::vector<const cdr::transform_stamped*> edges;
        for (const cdr::transform_stamped& transform : scene.dynamics) {
            if ((transform.frame_header.frame_id == "ego") && (transform.child_frame_id == ("sensor/" + sensor_name))) {
                edges.push_back(&transform);
            }
        }
        return edges;
    }

    int trajectory_scene(const std::string& input_path, const std::string& output_path) {
        mcap reader;
        dataset::mcap_scene_information scene;
        std::string error;
        if (!load_scene(input_path, reader, scene, error, false)) {
            std::fprintf(stderr, "Invalid scene: %s.\n", error.c_str());
            return EXIT_FAILURE;
        }
        std::size_t poses = 0;
        if (!dataset::write_trajectory(scene, output_path, poses)) {
            std::fprintf(stderr, "Failed to write the ground truth: %s\n", output_path.c_str());
            return EXIT_FAILURE;
        }
        if (poses == 0) {
            std::fprintf(stderr, "The scene has no root -> ego ground truth transforms.\n");
            return EXIT_FAILURE;
        }
        std::printf("Wrote %zu ground truth poses to '%s'.\n", poses, output_path.c_str());
        return EXIT_SUCCESS;
    }

    void sensor_extrinsic(const std::vector<const cdr::transform_stamped*>& edges, const std::size_t index, double extrinsic[7]) {
        const double identity[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        std::copy(&identity[0], &identity[0] + 7, extrinsic);
        if (!edges.empty()) {
            std::copy(&edges[index]->translation[0], &edges[index]->translation[0] + 3, extrinsic);
            std::copy(&edges[index]->rotation[0], &edges[index]->rotation[0] + 4, extrinsic + 3);
        }
    }

    int expand_scene(const std::string& input_path, const std::string& output_directory) {
        mcap reader;
        dataset::mcap_scene_information scene;
        std::string error;
        if (!load_scene(input_path, reader, scene, error)) {
            std::fprintf(stderr, "Invalid scene: %s.\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (!check_scene_round_trip(scene, false, error)) {
            std::fprintf(stderr, "Invalid scene: %s.\n", error.c_str());
            return EXIT_FAILURE;
        }
        if (gtl::paths::exists(output_directory)) {
            std::fprintf(stderr, "The output '%s' already exists; remove it or expand into a new directory.\n", output_directory.c_str());
            return EXIT_FAILURE;
        }
        if (!gtl::directory::make_directories(output_directory + "/sensor")) {
            std::fprintf(stderr, "Failed to create: %s\n", (output_directory + "/sensor").c_str());
            return EXIT_FAILURE;
        }
        // One calibration file per camera, one line per frame: the frame timestamp as
        std::size_t frames = 0;
        for (const dataset::camera_information& camera : scene.cameras) {
            const std::string frames_directory = output_directory + "/sensor/" + camera.camera_name;
            if (!gtl::directory::make_directories(frames_directory)) {
                std::fprintf(stderr, "Failed to create: %s\n", frames_directory.c_str());
                return EXIT_FAILURE;
            }
            const std::vector<const cdr::transform_stamped*> camera_edges = find_sensor_edges(scene, camera.camera_name);
            gtl::file handle((frames_directory + ".txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_only, gtl::file::cursor_type::start_of_truncated);
            if (!handle.is_open()) {
                std::fprintf(stderr, "Failed to create: %s.txt\n", frames_directory.c_str());
                return EXIT_FAILURE;
            }
            for (std::size_t i = 0; i < camera.frames; ++i) {
                const cdr::camera_info& information = camera.camera_infos_data[i];
                double extrinsic[7];
                sensor_extrinsic(camera_edges, i, extrinsic);
                const double intrinsics[4] = { information.k[0], information.k[4], information.k[2], information.k[5] };
                std::string line;
                bool formatted = dataset::format_sample_line(line, information.frame_header.stamp.nanoseconds(), &extrinsic[0], 7);
                line += " " + information.distortion_model;
                formatted = formatted && dataset::append_values(line, &intrinsics[0], 4) && dataset::append_values(line, information.d.data(), information.d.size());
                line += '\n';
                if (!formatted || !dataset::write_line(handle, line.c_str())) {
                    std::fprintf(stderr, "Failed to write the calibration for '%s'.\n", camera.camera_name.c_str());
                    return EXIT_FAILURE;
                }
            }
            const mcap::channel_type* image_channel = nullptr;
            for (const mcap::channel_type& channel : reader.get_channels()) {
                if (channel.topic == camera.image_topic) {
                    image_channel = &channel;
                    break;
                }
            }
            std::size_t camera_frame_index = 0;
            const std::vector<mcap::message_index_type>& message_index = reader.get_message_index();
            for (std::size_t index = 0; index < message_index.size(); ++index) {
                if ((image_channel == nullptr) || (message_index[index].channel_id != image_channel->id)) {
                    continue;
                }
                mcap::message_type message;
                cdr::image image;
                dataset::pnm_format format = dataset::pnm_format::pgm8;
                if (!reader.read_message(index, message) || !cdr::read_image(message.data, message.length, image) || !dataset::pnm_format_of_encoding(image.encoding, format)) {
                    std::fprintf(stderr, "Image %zu of '%s' does not decode.\n", camera_frame_index, camera.camera_name.c_str());
                    return EXIT_FAILURE;
                }
                const std::string path = frames_directory + "/" + dataset::frame_filename(camera.image_log_times[camera_frame_index]);
                if (!dataset::write_pnm(path, format, image.width, image.height, image.data.data())) {
                    std::fprintf(stderr, "Failed to write frame %zu of '%s'.\n", camera_frame_index, camera.camera_name.c_str());
                    return EXIT_FAILURE;
                }
                ++camera_frame_index;
                ++frames;
            }
        }
        std::size_t imu_samples = 0;
        for (const dataset::imu_information& imu : scene.imus) {
            const dataset::inertial_type type = dataset::inertial_type_of(imu.imu_name);
            const std::vector<const cdr::transform_stamped*> imu_edges = find_sensor_edges(scene, imu.imu_name);
            gtl::file handle((output_directory + "/sensor/" + imu.imu_name + ".txt").c_str(), gtl::file::access_type::write_only, gtl::file::creation_type::create_only, gtl::file::cursor_type::start_of_truncated);
            if (!handle.is_open()) {
                std::fprintf(stderr, "Failed to create: %s/sensor/%s.txt\n", output_directory.c_str(), imu.imu_name.c_str());
                return EXIT_FAILURE;
            }
            std::vector<double> values;
            for (std::size_t i = 0; i < imu.samples; ++i) {
                const cdr::imu& sample = imu.imu_data[i];
                double extrinsic[7];
                sensor_extrinsic(imu_edges, i, extrinsic);
                values.assign(&extrinsic[0], &extrinsic[0] + 7);
                if (dataset::measures_angular_velocity(type)) {
                    values.insert(values.end(), &sample.angular_velocity[0], &sample.angular_velocity[0] + 3);
                }
                if (dataset::measures_linear_acceleration(type)) {
                    values.insert(values.end(), &sample.linear_acceleration[0], &sample.linear_acceleration[0] + 3);
                }
                if (!dataset::write_sample_line(handle, sample.frame_header.stamp.nanoseconds(), values.data(), values.size())) {
                    std::fprintf(stderr, "Failed to write the data for '%s'.\n", imu.imu_name.c_str());
                    return EXIT_FAILURE;
                }
            }
            imu_samples += imu.samples;
        }
        std::size_t poses = 0;
        if ((scene.poses > 0) && !dataset::write_trajectory(scene, output_directory + "/trajectory.txt", poses)) {
            std::fprintf(stderr, "Failed to write the ground truth.\n");
            return EXIT_FAILURE;
        }
        std::printf("Expanded %zu frames, %zu imu samples, and %zu poses into %s.\n", frames, imu_samples, poses, output_directory.c_str());
        return EXIT_SUCCESS;
    }

    // One parsed line of a camera's calibration file: a seconds.subseconds timestamp, the
    struct calibration_line {
        long long timestamp_nanoseconds = 0;
        double extrinsic[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        double parameters[4] = {};
        std::vector<double> distortion;
        std::string model_name;
    };

    struct imu_line {
        long long timestamp_nanoseconds = 0;
        double extrinsic[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
        double angular_velocity[3] = {};
        double linear_acceleration[3] = {};
    };

    bool is_unit_quaternion(const double* const rotation_xyzw) {
        const double norm_squared = (rotation_xyzw[0] * rotation_xyzw[0]) + (rotation_xyzw[1] * rotation_xyzw[1]) + (rotation_xyzw[2] * rotation_xyzw[2]) + (rotation_xyzw[3] * rotation_xyzw[3]);
        return (norm_squared >= 1.0 - 1.0e-6) && (norm_squared <= 1.0 + 1.0e-6);
    }

    bool read_calibration_file(const std::string& path, std::vector<calibration_line>& lines, std::string& error) {
        std::string text;
        if (!dataset::read_text_file(path, text)) {
            error = "cannot read '" + path + "'";
            return false;
        }
        for (char* line = &text[0]; line != nullptr;) {
            char* end = std::strchr(line, '\n');
            if (end != nullptr) {
                *end = '\0';
            }
            calibration_line entry;
            char model[64] = {};
            int offset = 0;
            char* timestamp_end = line;
            while ((*timestamp_end != '\0') && (*timestamp_end != ' ') && (*timestamp_end != '\t')) {
                ++timestamp_end;
            }
            const char saved_character = *timestamp_end;
            *timestamp_end = '\0';
            const bool timestamp_ok = dataset::parse_timestamp_nanoseconds(line, entry.timestamp_nanoseconds);
            *timestamp_end = saved_character;
            const int matched = timestamp_ok ? std::sscanf(timestamp_end, "%lf %lf %lf %lf %lf %lf %lf %63s %lf %lf %lf %lf %n", &entry.extrinsic[0], &entry.extrinsic[1], &entry.extrinsic[2], &entry.extrinsic[3], &entry.extrinsic[4], &entry.extrinsic[5], &entry.extrinsic[6], &model[0], &entry.parameters[0], &entry.parameters[1], &entry.parameters[2], &entry.parameters[3], &offset) : 0;
            if (matched != 12) {
                line = (end != nullptr) ? (end + 1) : nullptr;
                continue;
            }
            entry.model_name = &model[0];
            const char* cursor = timestamp_end + offset;
            while (*cursor != '\0') {
                char* float_end = nullptr;
                const double value = std::strtod(cursor, &float_end);
                if (float_end == cursor) {
                    break;
                }
                entry.distortion.push_back(value);
                cursor = float_end;
            }
            if (!is_unit_quaternion(&entry.extrinsic[3])) {
                error = "an extrinsic quaternion in '" + path + "' is not unit length";
                return false;
            }
            lines.push_back(entry);
            line = (end != nullptr) ? (end + 1) : nullptr;
        }
        return true;
    }

    bool read_imu_file(const std::string& path, const dataset::inertial_type type, std::vector<imu_line>& lines, std::string& error) {
        std::string text;
        if (!dataset::read_text_file(path, text)) {
            error = "cannot read '" + path + "'";
            return false;
        }
        const std::size_t expected = 7u + (dataset::measures_angular_velocity(type) ? 3u : 0u) + (dataset::measures_linear_acceleration(type) ? 3u : 0u);
        std::vector<double> values;
        std::size_t line_number = 0;
        for (char* line = &text[0]; line != nullptr;) {
            ++line_number;
            char* end = std::strchr(line, '\n');
            if (end != nullptr) {
                *end = '\0';
            }
            imu_line entry;
            if (dataset::parse_sample_line(line, entry.timestamp_nanoseconds, values)) {
                if (values.size() != expected) {
                    error = "line " + std::to_string(line_number) + " of '" + path + "' has " + std::to_string(values.size()) + " values, its kind's form has " + std::to_string(expected);
                    return false;
                }
                std::copy(values.begin(), values.begin() + 7, &entry.extrinsic[0]);
                if (!is_unit_quaternion(&entry.extrinsic[3])) {
                    error = "an extrinsic quaternion in '" + path + "' is not unit length";
                    return false;
                }
                std::size_t next = 7;
                if (dataset::measures_angular_velocity(type)) {
                    std::copy(values.begin() + static_cast<std::ptrdiff_t>(next), values.begin() + static_cast<std::ptrdiff_t>(next + 3), &entry.angular_velocity[0]);
                    next += 3;
                }
                if (dataset::measures_linear_acceleration(type)) {
                    std::copy(values.begin() + static_cast<std::ptrdiff_t>(next), values.begin() + static_cast<std::ptrdiff_t>(next + 3), &entry.linear_acceleration[0]);
                }
                lines.push_back(entry);
            }
            line = (end != nullptr) ? (end + 1) : nullptr;
        }
        return true;
    }

    // one lz4 compressed scene mcap.
    int collapse_scene(const std::string& input_directory, const std::string& output_path) {
        // Discover every sensor: a sensor/[name]/ frame directory paired with a
        std::vector<std::string> camera_names;
        std::vector<std::string> imu_names;
        {
            std::vector<std::string> entries;
            gtl::directory::list_directory(input_directory + "/sensor", entries);
            for (const std::string& entry : entries) {
                if (!gtl::paths::is_regular_file(input_directory + "/sensor/" + entry) || (gtl::paths::path_extension(entry) != ".txt")) {
                    continue;
                }
                const std::string name = gtl::paths::path_stem(entry);
                if (gtl::paths::is_directory(input_directory + "/sensor/" + name)) {
                    camera_names.push_back(name);
                }
                else if (dataset::inertial_type_of(name) != dataset::inertial_type::unknown) {
                    imu_names.push_back(name);
                }
                else {
                    std::fprintf(stderr, "Invalid scene directory: 'sensor/%s.txt' has no paired frame directory and is not a valid 'imu_01' to 'imu_99', 'accelerometer_01' to 'accelerometer_99', or 'gyroscope_01' to 'gyroscope_99' name.\n", name.c_str());
                    return EXIT_FAILURE;
                }
            }
            if (camera_names.empty()) {
                std::fprintf(stderr, "Invalid scene directory: no sensor/[type]_[01-99] directory with a sensor/[type]_[01-99].txt calibration file was found.\n");
                return EXIT_FAILURE;
            }
            // Sorted so 'image_01' always precedes 'image_02', keeping the primary camera
            std::sort(camera_names.begin(), camera_names.end(), [](const std::string& left, const std::string& right) {
                const auto camera_priority = [](const std::string& name) -> int {
                    if (name.rfind("image_", 0) == 0) {
                        return 0;
                    }
                    if (name.rfind("depth_", 0) == 0) {
                        return 1;
                    }
                    return 2;
                };
                const int left_priority = camera_priority(left);
                const int right_priority = camera_priority(right);
                if (left_priority != right_priority) {
                    return left_priority < right_priority;
                }
                return left < right;
            });
            std::sort(imu_names.begin(), imu_names.end());
        }

        const std::string parent = gtl::paths::path_parent_directory(output_path);
        if (!parent.empty()) {
            gtl::directory::make_directories(parent);
        }
        mcap::writer writer;
        if (!writer.begin(output_path, "ros2", "zeroslam-dataset", "lz4")) {
            std::fprintf(stderr, "Failed to create: %s\n", output_path.c_str());
            return EXIT_FAILURE;
        }
        const unsigned short image_schema = writer.add_schema("sensor_msgs/msg/Image", "ros2msg", cdr::image_schema());
        const unsigned short info_schema = writer.add_schema("sensor_msgs/msg/CameraInfo", "ros2msg", cdr::camera_info_schema());
        const unsigned short tf_schema = writer.add_schema("tf2_msgs/msg/TFMessage", "ros2msg", cdr::tf_message_schema());
        const unsigned short imu_schema = imu_names.empty() ? 0 : writer.add_schema("sensor_msgs/msg/Imu", "ros2msg", cdr::imu_schema());
        const unsigned short tf_channel = writer.add_channel(tf_schema, "/tf", "cdr");

        // The ground truth is its own frame tree, root -> ego -> sensor/[name]: the
        // root -> ego transforms carry the trajectory of the body, and the per message
        // ego -> sensor/[name] extrinsics pose each sensor on it (so they can change over
        // time). The live frames of the ros convention (earth -> map -> ego) are left
        // free: estimating map -> ego is the job of the SLAM system.
        std::size_t total_frames = 0;
        for (const std::string& camera_name : camera_names) {
            const std::string frames_directory = input_directory + "/sensor/" + camera_name;
            const std::size_t frame_count = dataset::count_frame_directory(frames_directory);
            if (frame_count < 2) {
                std::fprintf(stderr, "Invalid scene directory: at least two frames are required in '%s'.\n", frames_directory.c_str());
                return EXIT_FAILURE;
            }
            std::vector<calibration_line> lines;
            std::string error;
            if (!read_calibration_file(input_directory + "/sensor/" + camera_name + ".txt", lines, error)) {
                std::fprintf(stderr, "Invalid scene directory: %s.\n", error.c_str());
                return EXIT_FAILURE;
            }
            if (lines.size() != frame_count) {
                std::fprintf(stderr, "Invalid scene directory: %zu calibration lines for %zu frames in '%s'.\n", lines.size(), frame_count, frames_directory.c_str());
                return EXIT_FAILURE;
            }
            const unsigned short image_channel = writer.add_channel(image_schema, "/sensor/" + camera_name, "cdr");
            const unsigned short info_channel = writer.add_channel(info_schema, "/sensor/" + camera_name + "/camera_info", "cdr");
            unsigned int width = 0;
            unsigned int height = 0;
            for (std::size_t i = 0; i < lines.size(); ++i) {
                const std::string frame_path = frames_directory + "/" + dataset::frame_filename(static_cast<unsigned long long>(lines[i].timestamp_nanoseconds));
                cdr::image image;
                dataset::pnm_format format = dataset::pnm_format::pgm8;
                if (!dataset::read_pnm(frame_path, format, image.width, image.height, image.data)) {
                    std::fprintf(stderr, "Invalid scene directory: '%s' is not a readable pnm (referenced by calibration line %zu).\n", frame_path.c_str(), i);
                    return EXIT_FAILURE;
                }
                if (i == 0) {
                    width = image.width;
                    height = image.height;
                }
                else if ((image.width != width) || (image.height != height)) {
                    std::fprintf(stderr, "Invalid scene directory: the frame dimensions change at frame %zu of '%s'.\n", i, camera_name.c_str());
                    return EXIT_FAILURE;
                }
                const cdr::time stamp = cdr::time::from_nanoseconds(lines[i].timestamp_nanoseconds);
                const unsigned long long log_time = static_cast<unsigned long long>(lines[i].timestamp_nanoseconds);
                image.frame_header.stamp = stamp;
                image.frame_header.frame_id = "sensor/" + camera_name;
                image.is_bigendian = 0;
                image.encoding = dataset::pnm_encoding(format);
                image.step = width * static_cast<unsigned int>(dataset::pnm_bytes_per_pixel(format));
                const std::vector<unsigned char> payload = cdr::write_image(image);
                writer.add_message(image_channel, static_cast<unsigned int>(i), log_time, log_time, payload.data(), payload.size());
                // One calibration message per frame, stamped alongside its image, so viewers
                // find a current camera info wherever they seek: an identity rectification
                // and a projection straight from the intrinsics, which viewers take their
                // focal length from. When the calibration line carried distortion
                cdr::camera_info information;
                information.frame_header.stamp = stamp;
                information.frame_header.frame_id = "sensor/" + camera_name;
                information.height = height;
                information.width = width;
                information.distortion_model = lines[i].model_name;
                information.d = lines[i].distortion;
                information.k[0] = lines[i].parameters[0];
                information.k[4] = lines[i].parameters[1];
                information.k[2] = lines[i].parameters[2];
                information.k[5] = lines[i].parameters[3];
                information.k[8] = 1.0;
                information.r[0] = 1.0;
                information.r[4] = 1.0;
                information.r[8] = 1.0;
                information.p[0] = lines[i].parameters[0];
                information.p[2] = lines[i].parameters[2];
                information.p[5] = lines[i].parameters[1];
                information.p[6] = lines[i].parameters[3];
                information.p[10] = 1.0;
                const std::vector<unsigned char> info_payload = cdr::write_camera_info(information);
                writer.add_message(info_channel, static_cast<unsigned int>(i), log_time, log_time, info_payload.data(), info_payload.size());
                // The extrinsic of this frame: the pose of the camera on the body.
                const std::vector<unsigned char> transform = cdr::write_tf_message(stamp, "ego", "sensor/" + camera_name, &lines[i].extrinsic[0], &lines[i].extrinsic[3]);
                writer.add_message(tf_channel, static_cast<unsigned int>(i), log_time, log_time, transform.data(), transform.size());
            }
            if (camera_name == camera_names.front()) {
                total_frames = lines.size();
            }
        }
        std::size_t total_imu_samples = 0;
        for (const std::string& imu_name : imu_names) {
            const dataset::inertial_type type = dataset::inertial_type_of(imu_name);
            std::vector<imu_line> lines;
            std::string error;
            if (!read_imu_file(input_directory + "/sensor/" + imu_name + ".txt", type, lines, error)) {
                std::fprintf(stderr, "Invalid scene directory: %s.\n", error.c_str());
                return EXIT_FAILURE;
            }
            if (lines.empty()) {
                std::fprintf(stderr, "Invalid scene directory: 'sensor/%s.txt' holds no samples; remove the file if the scene has no %s.\n", imu_name.c_str(), imu_name.c_str());
                return EXIT_FAILURE;
            }
            const unsigned short imu_channel = writer.add_channel(imu_schema, "/sensor/" + imu_name, "cdr");
            for (std::size_t i = 0; i < lines.size(); ++i) {
                const cdr::time stamp = cdr::time::from_nanoseconds(lines[i].timestamp_nanoseconds);
                const unsigned long long log_time = static_cast<unsigned long long>(lines[i].timestamp_nanoseconds);
                cdr::imu sample;
                sample.frame_header.stamp = stamp;
                sample.frame_header.frame_id = "sensor/" + imu_name;
                std::copy(&lines[i].angular_velocity[0], &lines[i].angular_velocity[0] + 3, &sample.angular_velocity[0]);
                std::copy(&lines[i].linear_acceleration[0], &lines[i].linear_acceleration[0] + 3, &sample.linear_acceleration[0]);
                if (!dataset::measures_angular_velocity(type)) {
                    sample.angular_velocity_covariance[0] = -1.0;
                }
                if (!dataset::measures_linear_acceleration(type)) {
                    sample.linear_acceleration_covariance[0] = -1.0;
                }
                const std::vector<unsigned char> payload = cdr::write_imu(sample);
                writer.add_message(imu_channel, static_cast<unsigned int>(i), log_time, log_time, payload.data(), payload.size());
                const std::vector<unsigned char> transform = cdr::write_tf_message(stamp, "ego", "sensor/" + imu_name, &lines[i].extrinsic[0], &lines[i].extrinsic[3]);
                writer.add_message(tf_channel, static_cast<unsigned int>(i), log_time, log_time, transform.data(), transform.size());
            }
            total_imu_samples += lines.size();
        }
        std::vector<dataset::trajectory_pose> poses;
        const std::string trajectory_path = input_directory + "/trajectory.txt";
        if (gtl::paths::is_regular_file(trajectory_path) && !dataset::load_trajectory(trajectory_path, poses)) {
            std::fprintf(stderr, "Invalid scene directory: cannot read the ground truth at '%s'; remove the file if the scene has none.\n", trajectory_path.c_str());
            return EXIT_FAILURE;
        }
        for (std::size_t i = 0; i < poses.size(); ++i) {
            const cdr::time stamp = cdr::time::from_nanoseconds(poses[i].timestamp_nanoseconds);
            const unsigned long long log_time = static_cast<unsigned long long>(poses[i].timestamp_nanoseconds);
            const double translation[3] = { poses[i].x_coordinate, poses[i].y_coordinate, poses[i].z_coordinate };
            const double rotation_xyzw[4] = { poses[i].quaternion_x, poses[i].quaternion_y, poses[i].quaternion_z, poses[i].quaternion_w };
            const std::vector<unsigned char> transform = cdr::write_tf_message(stamp, "root", "ego", &translation[0], &rotation_xyzw[0]);
            writer.add_message(tf_channel, static_cast<unsigned int>(i), log_time, log_time, transform.data(), transform.size());
        }
        if (!writer.finish()) {
            std::fprintf(stderr, "Failed to write: %s\n", output_path.c_str());
            return EXIT_FAILURE;
        }
        std::printf("Collapsed %zu frames, %zu imu samples, and %zu poses into %s (%llu bytes).\n", total_frames, total_imu_samples, poses.size(), output_path.c_str(), writer.get_written());
        if (poses.empty()) {
            std::printf("    No ground truth at '%s', so the scene carries none; it cannot be benchmarked against.\n", trajectory_path.c_str());
        }
        return validate_scene(output_path) ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    void print_usage(const char* argv0) {
        std::printf("Usage %s [command] [options...]\n", argv0);
        std::printf("    commands:\n");
        std::printf("        list                     - List the dataset scenes available on the hub.\n");
        std::printf("        get [name]               - Download a whole dataset ('[dataset]') or one scene\n");
        std::printf("                                   ('[dataset]/[scene]') into the datasets directory and\n");
        std::printf("                                   validate every fetched scene mcap.\n");
        std::printf("        validate [name]          - Validate a scene by name, every scene of a dataset, or a\n");
        std::printf("                                   scene mcap path directly: raw frames, a pinhole\n");
        std::printf("                                   calibration per frame, and the root -> ego -> sensor\n");
        std::printf("                                   ground truth frame tree on /tf.\n");
        std::printf("        expand [mcap] [dir]      - Expand a scene mcap into a new directory in the directory\n");
        std::printf("                                   form: pnm frames, the calibration, and the trajectory.\n");
        std::printf("        trajectory [mcap] [txt]  - Write just the scene's ground truth (the root -> ego\n");
        std::printf("                                   transforms) as a trajectory file for zeroslam-evaluate.\n");
        std::printf("                                   Unlike expand this does not require the scene to round\n");
        std::printf("                                   trip through the directory form.\n");
        std::printf("        collapse [dir] [mcap]    - Collapse a scene directory back into one lz4 compressed\n");
        std::printf("                                   scene mcap. A 'trajectory.txt' is optional: without one\n");
        std::printf("                                   the scene holds no ground truth, as a live capture does.\n");
        std::printf("    options:\n");
        std::printf("        --datasets [dir] - The datasets directory (default: the 'datasets' directory next\n");
        std::printf("                           to this tool, otherwise './datasets').\n");
        std::printf("        --repo [id]    - The hub repository (default: gpdaniels/slam-datasets).\n");
        std::printf("        --token [t]    - Hub access token for private repositories (default: the HF_TOKEN\n");
        std::printf("                         environment variable).\n");
        std::printf("        --force        - Redownload files even when the local copy matches.\n");
        std::printf("Downloads stream to a '.part' file renamed into place after a size check, so an\n");
        std::printf("interrupted download never leaves a plausible looking file, and rerunning a download\n");
        std::printf("completes or repairs it while keeping the files that already match. The transport is\n");
        std::printf("the system curl.\n");
    }
}

int main(int argc, char* argv[]) {
    std::string command;
    std::string dataset_name;
    std::string second_argument;
    std::string datasets_directory_override;
    std::string repository = "gpdaniels/slam-datasets";
    std::string token;
    bool force = false;

    if (const char* environment_token = std::getenv("HF_TOKEN")) {
        token = environment_token;
    }

    for (int i = 1; i < argc; ++i) {
        const auto matches = [&](const char* name) {
            return std::strcmp(argv[i], name) == 0;
        };
        const auto take_value = [&](std::string& destination) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: %s\n", argv[i]);
                return false;
            }
            destination = argv[++i];
            return true;
        };
        if (matches("--help") || matches("-h")) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        else if (matches("--datasets")) {
            if (!take_value(datasets_directory_override))
                return EXIT_FAILURE;
        }
        else if (matches("--repo")) {
            if (!take_value(repository))
                return EXIT_FAILURE;
        }
        else if (matches("--token")) {
            if (!take_value(token))
                return EXIT_FAILURE;
        }
        else if (matches("--force")) {
            force = true;
        }
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        else if (command.empty()) {
            command = argv[i];
        }
        else if (((command == "get") || (command == "validate") || (command == "expand") || (command == "collapse") || (command == "trajectory")) && dataset_name.empty()) {
            dataset_name = argv[i];
        }
        else if (((command == "expand") || (command == "collapse") || (command == "trajectory")) && second_argument.empty()) {
            second_argument = argv[i];
        }
        else {
            std::fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    if (command.empty()) {
        print_usage(argv[0]);
        return EXIT_SUCCESS;
    }
    if ((command != "list") && (command != "get") && (command != "validate") && (command != "expand") && (command != "collapse") && (command != "trajectory")) {
        std::fprintf(stderr, "Unknown command: '%s' (expected 'list', 'get', 'validate', 'expand', 'collapse', or 'trajectory').\n", command.c_str());
        return EXIT_FAILURE;
    }
    if (command == "trajectory") {
        if (dataset_name.empty() || second_argument.empty()) {
            std::fprintf(stderr, "The trajectory command needs a scene mcap and an output file argument.\n");
            return EXIT_FAILURE;
        }
        return trajectory_scene(dataset_name, second_argument);
    }
    if ((command == "expand") || (command == "collapse")) {
        if (dataset_name.empty() || second_argument.empty()) {
            std::fprintf(stderr, "The %s command needs an input and an output argument.\n", command.c_str());
            return EXIT_FAILURE;
        }
        return (command == "expand") ? expand_scene(dataset_name, second_argument) : collapse_scene(dataset_name, second_argument);
    }
    if (!is_valid_repository(repository)) {
        std::fprintf(stderr, "Invalid repository identifier: '%s' (expected 'owner/name').\n", repository.c_str());
        return EXIT_FAILURE;
    }

    // The default datasets directory is the 'datasets' directory next to this tool.
    std::string executable_directory;
    std::string default_datasets_directory = "datasets";
    if (gtl::paths::get_executable_directory(executable_directory)) {
        default_datasets_directory = executable_directory + "/datasets";
    }
    const std::string datasets_directory = datasets_directory_override.empty() ? default_datasets_directory : datasets_directory_override;

    if (command == "validate") {
        if (dataset_name.empty()) {
            std::fprintf(stderr, "Missing dataset name or scene mcap path.\n");
            return EXIT_FAILURE;
        }
        // A path to a scene mcap: validate it directly, without the datasets directory.
        if ((dataset_name.size() >= 5) && (dataset_name.compare(dataset_name.size() - 5, 5, ".mcap") == 0)) {
            if (!gtl::paths::is_regular_file(dataset_name)) {
                std::fprintf(stderr, "Invalid mcap: not found: %s\n", dataset_name.c_str());
                return EXIT_FAILURE;
            }
            return validate_scene(dataset_name) ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        if (!is_valid_dataset_or_scene_name(dataset_name)) {
            std::fprintf(stderr, "Invalid dataset name.\n");
            return EXIT_FAILURE;
        }
        const std::string target = datasets_directory + "/" + dataset_name;
        if (gtl::paths::is_regular_file(target + ".mcap")) {
            return validate_scene(target + ".mcap") ? EXIT_SUCCESS : EXIT_FAILURE;
        }
        // A dataset directory: validate every scene mcap inside it.
        std::vector<std::string> entries;
        if (!gtl::directory::list_directory(target, entries)) {
            std::fprintf(stderr, "Invalid dataset: not found: %s(.mcap)\n", target.c_str());
            return EXIT_FAILURE;
        }
        std::vector<std::string> scenes;
        for (const std::string& entry : entries) {
            if (gtl::paths::is_regular_file(target + "/" + entry) && (gtl::paths::path_extension(entry) == ".mcap")) {
                scenes.push_back(target + "/" + entry);
            }
        }
        std::sort(scenes.begin(), scenes.end());
        if (scenes.empty()) {
            std::fprintf(stderr, "No scene mcaps were found in '%s'.\n", target.c_str());
            return EXIT_FAILURE;
        }
        for (const std::string& scene : scenes) {
            if (!validate_scene(scene)) {
                return EXIT_FAILURE;
            }
        }
        return EXIT_SUCCESS;
    }

    std::printf("Fetching the file listing of %s...\n", repository.c_str());
    std::fflush(stdout);
    std::vector<hub_entry> entries;
    std::string error;
    if (!fetch_listing(repository, token, entries, error)) {
        std::fprintf(stderr, "Failed to list the hub repository: %s\n", error.c_str());
        return EXIT_FAILURE;
    }

    // The downloadable scenes are the "[dataset]/[scene].mcap" files.
    constexpr static const char mcap_extension[] = ".mcap";
    constexpr static const std::size_t mcap_extension_length = sizeof(mcap_extension) - 1;
    std::vector<std::string> names;
    for (const hub_entry& entry : entries) {
        if (entry.directory || (entry.path.size() <= mcap_extension_length) || (entry.path.compare(entry.path.size() - mcap_extension_length, mcap_extension_length, mcap_extension) != 0)) {
            continue;
        }
        const std::string name = entry.path.substr(0, entry.path.size() - mcap_extension_length);
        if (is_valid_dataset_or_scene_name(name) && (name.find('/') != std::string::npos)) {
            names.push_back(name);
        }
    }
    const auto scene_size = [&](const std::string& name) {
        for (const hub_entry& entry : entries) {
            if (!entry.directory && (entry.path == name + ".mcap")) {
                return entry.size;
            }
        }
        return 0ull;
    };

    if (command == "list") {
        if (names.empty()) {
            std::printf("No scenes found.\n");
            return EXIT_SUCCESS;
        }
        std::printf("Scenes:\n");
        for (const std::string& name : names) {
            std::printf("    %s (%s)\n", name.c_str(), format_size(scene_size(name)).c_str());
        }
        return EXIT_SUCCESS;
    }

    // command == "get"
    if (dataset_name.empty() || !is_valid_dataset_or_scene_name(dataset_name)) {
        std::fprintf(stderr, "Invalid or missing dataset name.\n");
        return EXIT_FAILURE;
    }
    std::vector<hub_entry> files;
    for (const hub_entry& entry : entries) {
        const bool scene_file = (entry.path == dataset_name + ".mcap");
        const bool dataset_file = (entry.path.compare(0, dataset_name.size() + 1, dataset_name + "/") == 0);
        if (!entry.directory && (scene_file || dataset_file)) {
            if (!is_safe_hub_path(entry.path)) {
                std::fprintf(stderr, "Refusing unsafe path in the hub listing: '%s'\n", entry.path.c_str());
                return EXIT_FAILURE;
            }
            files.push_back(entry);
        }
    }
    if (files.empty()) {
        std::fprintf(stderr, "Unknown dataset or scene: '%s'. Available scenes:\n", dataset_name.c_str());
        for (const std::string& name : names) {
            std::fprintf(stderr, "    %s\n", name.c_str());
        }
        return EXIT_FAILURE;
    }

    std::printf("Downloading '%s' into '%s'...\n", dataset_name.c_str(), datasets_directory.c_str());
    std::fflush(stdout);

    const std::string base_command = curl_command(token);
    std::size_t downloaded = 0;
    std::size_t kept = 0;
    unsigned long long total_bytes = 0;
    for (const hub_entry& file : files) {
        total_bytes += file.size;
        const std::string local = datasets_directory + "/" + file.path;
        unsigned long long actual_size = 0;
        if (!force && gtl::paths::get_file_size(local, actual_size) && (actual_size == file.size)) {
            std::printf("    kept       %s (%s)\n", file.path.c_str(), format_size(file.size).c_str());
            std::fflush(stdout);
            ++kept;
            continue;
        }
        gtl::directory::make_directories(gtl::paths::path_parent_directory(local));
        std::printf("    downloading %s (%s)...", file.path.c_str(), format_size(file.size).c_str());
        std::fflush(stdout);
        const std::string url = "https://huggingface.co/datasets/" + repository + "/resolve/main/" + file.path;
        const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        std::string download_error;
        if (!download_file(base_command, url, local, file.size, download_error)) {
            std::printf("\n");
            std::fprintf(stderr, "Failed to download '%s': %s\n", file.path.c_str(), download_error.c_str());
            return EXIT_FAILURE;
        }
        const std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;
        std::printf(" done (%.1f s)\n", duration.count());
        std::fflush(stdout);
        ++downloaded;
    }
    std::printf("Done: %zu files (%s), %zu downloaded, %zu kept.\n", files.size(), format_size(total_bytes).c_str(), downloaded, kept);
    std::vector<std::string> scenes;
    for (const hub_entry& file : files) {
        if ((file.path.size() > mcap_extension_length) && (file.path.compare(file.path.size() - mcap_extension_length, mcap_extension_length, mcap_extension) == 0)) {
            scenes.push_back(file.path);
        }
    }
    std::sort(scenes.begin(), scenes.end());
    if (scenes.empty()) {
        std::fprintf(stderr, "No scene mcaps were found under '%s'.\n", dataset_name.c_str());
        return EXIT_FAILURE;
    }
    for (const std::string& scene : scenes) {
        if (!validate_scene(datasets_directory + "/" + scene)) {
            return EXIT_FAILURE;
        }
    }
    for (const std::string& scene : scenes) {
        std::printf("Benchmark it with: zeroslam regression %s\n", (datasets_directory + "/" + scene).c_str());
    }
    return EXIT_SUCCESS;
}
