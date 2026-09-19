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

#include "graph.hpp"
#include "icon.hpp"
#include "image_texture.hpp"
#include "imgui.hpp"
#include "map_file.hpp"
#include "metrics.hpp"
#include "opengl.hpp"
#include "render_snapshot.hpp"
#include "rotation.hpp"
#include "scene.hpp"
#include "triple_buffer.hpp"
#include "ui.hpp"
#include "window.hpp"
#include "zeroslam/zeroslam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    constexpr static const char* const association_names[3] = { "klt", "match", "both" };

    std::atomic<bool> shutdown_requested{ false };

    void signal_handler(int) {
        std::signal(SIGINT, SIG_DFL);
        shutdown_requested = true;
    }

    void print_usage(const char* const program_name) {
        std::printf("Usage %s [scene] [options...]\n", program_name);
        std::printf("    scene - The path of a scene mcap file holding raw image, camera info,\n");
        std::printf("            imu and /tf messages. The same scene files the other tools read.\n");
        std::printf("    options:\n");
        std::printf("        --play                 - Start playing as soon as the scene is loaded.\n");
        std::printf("        --screenshot [file]    - Write the window to a binary ppm and exit.\n");
        std::printf("        --screenshot-after [n] - Render n frames before the screenshot (default 30).\n");
        std::printf("        --exit-after [n]       - Exit after n rendered frames.\n");
        std::printf("        --load-map [file]      - Display a saved map instead of running the slam system.\n");
        std::printf("        --save-map [file]      - Write the built map out when the run ends.\n");
        std::printf("        --config [key=value]   - A front end setting to start with: tracker=klt|extrema, lines=on|off, culling=on|off or association=klt|match|both, default both (repeatable).\n");
        std::printf("        --help                 - Show this message.\n");
    }

    bool write_screenshot(const std::string& path, const int width, const int height) {
        if ((width <= 0) || (height <= 0)) {
            return false;
        }
        const std::size_t row_bytes = static_cast<std::size_t>(width) * 3;
        std::vector<unsigned char> pixels(row_bytes * static_cast<std::size_t>(height));
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
        for (int y = 0; y < height / 2; ++y) {
            std::swap_ranges(pixels.begin() + static_cast<std::ptrdiff_t>(row_bytes * static_cast<std::size_t>(y)), pixels.begin() + static_cast<std::ptrdiff_t>(row_bytes * static_cast<std::size_t>(y + 1)), pixels.begin() + static_cast<std::ptrdiff_t>(row_bytes * static_cast<std::size_t>(height - 1 - y)));
        }
        return dataset::write_pnm(path, dataset::pnm_format::ppm, static_cast<unsigned int>(width), static_cast<unsigned int>(height), pixels.data());
    }

    void gl_load_perspective(float matrix[16], const float fovy_degrees, const float aspect, const float z_near, const float z_far) {
        std::memset(matrix, 0, 16 * sizeof(float));
        const float tan_half = std::tan(fovy_degrees * 3.14159265358979323846f / 360.0f);
        matrix[0] = 1.0f / (aspect * tan_half);
        matrix[5] = 1.0f / tan_half;
        matrix[10] = -(z_far + z_near) / (z_far - z_near);
        matrix[11] = -1.0f;
        matrix[14] = -(2.0f * z_far * z_near) / (z_far - z_near);
    }

    void gl_multiply_pose(const double rotation[3][3], const double translation[3]) {
        float matrix[16] = {};
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                matrix[(column * 4) + row] = static_cast<float>(rotation[row][column]);
            }
            matrix[12 + row] = static_cast<float>(translation[row]);
        }
        matrix[15] = 1.0f;
        glMultMatrixf(matrix);
    }

    void gl_multiply_similarity(const metrics::similarity& alignment) {
        double scaled[3][3];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                scaled[row][column] = alignment.rotation[row][column] * alignment.scale;
            }
        }
        gl_multiply_pose(scaled, alignment.translation);
    }

    void gl_multiply_camera_to_world(const gui::render_snapshot::frame& pose) {
        double camera_to_world[3][3];
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                camera_to_world[row][column] = pose.rotation[column][row];
            }
        }
        double centre[3];
        gui::render_snapshot::camera_centre(pose, centre);
        gl_multiply_pose(camera_to_world, centre);
    }

    void gl_draw_frustum(const gui::render_snapshot::frame& pose, const float depth, const float red, const float green, const float blue) {
        const float fx = static_cast<float>(pose.camera_parameters[0]);
        const float fy = static_cast<float>(pose.camera_parameters[1]);
        const float cx = static_cast<float>(pose.camera_parameters[2]);
        const float cy = static_cast<float>(pose.camera_parameters[3]);
        const float width = static_cast<float>(pose.image_width > 0 ? pose.image_width : 640);
        const float height = static_cast<float>(pose.image_height > 0 ? pose.image_height : 480);
        const float corners[4][2] = { { 0.0f, 0.0f }, { width, 0.0f }, { width, height }, { 0.0f, height } };
        float rays[4][3];
        for (int i = 0; i < 4; ++i) {
            rays[i][0] = (corners[i][0] - cx) / fx;
            rays[i][1] = (corners[i][1] - cy) / fy;
            rays[i][2] = 1.0f;
            const float length = std::sqrt((rays[i][0] * rays[i][0]) + (rays[i][1] * rays[i][1]) + 1.0f);
            for (int axis = 0; axis < 3; ++axis) {
                rays[i][axis] *= depth / length;
            }
        }
        glColor3f(red, green, blue);
        glBegin(GL_LINES);
        for (int i = 0; i < 4; ++i) {
            glVertex3f(0.0f, 0.0f, 0.0f);
            glVertex3fv(&rays[i][0]);
            glVertex3fv(&rays[i][0]);
            glVertex3fv(&rays[(i + 1) % 4][0]);
        }
        glEnd();
    }

    void gl_draw_axes(const float scale) {
        glBegin(GL_LINES);
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(scale, 0.0f, 0.0f);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, scale, 0.0f);
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, scale);
        glEnd();
    }

    void gl_draw_grid(const float size, const float step) {
        glColor3f(0.25f, 0.25f, 0.28f);
        glBegin(GL_LINES);
        for (float v = -size; v <= size + 0.001f; v += step) {
            glVertex3f(v, 0.0f, -size);
            glVertex3f(v, 0.0f, size);
            glVertex3f(-size, 0.0f, v);
            glVertex3f(size, 0.0f, v);
        }
        glEnd();
    }

    void gl_draw_arrays(const GLenum mode, const std::vector<float>& vertices, const std::vector<float>& colours) {
        if (vertices.empty()) {
            return;
        }
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, vertices.data());
        glColorPointer(3, GL_FLOAT, 0, colours.data());
        glDrawArrays(mode, 0, static_cast<GLsizei>(vertices.size() / 3));
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    }

    struct orbit_camera {
        constexpr static const float field_of_view_degrees = 45.0f;
        float target[3] = { 0.0f, 0.0f, 0.0f };
        float distance = 10.0f;
        float elevation = 0.3f;
        float azimuth = 0.5f;

        void rotate(const float delta_x, const float delta_y, const int viewport_width, const int viewport_height) {
            this->azimuth -= (delta_x / static_cast<float>(viewport_width)) * 3.0f;
            this->elevation += (delta_y / static_cast<float>(viewport_height)) * 3.0f;
            this->elevation = (this->elevation < -1.5f) ? -1.5f : ((this->elevation > 1.5f) ? 1.5f : this->elevation);
        }

        void pan(const float delta_x, const float delta_y, const int viewport_height) {
            float eye[3];
            float forward[3];
            float right[3];
            float up[3];
            this->basis(eye, forward, right, up);
            const float units_per_pixel = 2.0f * this->distance * std::tan(field_of_view_degrees * 3.14159265358979323846f / 360.0f) / static_cast<float>(viewport_height);
            for (int axis = 0; axis < 3; ++axis) {
                this->target[axis] += ((up[axis] * delta_y) - (right[axis] * delta_x)) * units_per_pixel;
            }
        }

        void zoom(const int steps) {
            this->distance -= static_cast<float>(steps) * this->distance * 0.1f;
            this->distance = (this->distance < 0.1f) ? 0.1f : ((this->distance > 1000.0f) ? 1000.0f : this->distance);
        }

        void load_view(float matrix[16]) const {
            float eye[3];
            float forward[3];
            float right[3];
            float up[3];
            this->basis(eye, forward, right, up);
            for (int axis = 0; axis < 3; ++axis) {
                matrix[(axis * 4) + 0] = right[axis];
                matrix[(axis * 4) + 1] = up[axis];
                matrix[(axis * 4) + 2] = -forward[axis];
                matrix[(axis * 4) + 3] = 0.0f;
            }
            matrix[12] = -((right[0] * eye[0]) + (right[1] * eye[1]) + (right[2] * eye[2]));
            matrix[13] = -((up[0] * eye[0]) + (up[1] * eye[1]) + (up[2] * eye[2]));
            matrix[14] = (forward[0] * eye[0]) + (forward[1] * eye[1]) + (forward[2] * eye[2]);
            matrix[15] = 1.0f;
        }

    private:
        void basis(float eye[3], float forward[3], float right[3], float up[3]) const {
            eye[0] = this->target[0] + (this->distance * std::cos(this->elevation) * std::sin(this->azimuth));
            eye[1] = this->target[1] + (this->distance * std::sin(this->elevation));
            eye[2] = this->target[2] + (this->distance * std::cos(this->elevation) * std::cos(this->azimuth));
            for (int axis = 0; axis < 3; ++axis) {
                forward[axis] = (this->target[axis] - eye[axis]) / this->distance;
            }
            right[0] = -forward[2];
            right[1] = 0.0f;
            right[2] = forward[0];
            const float right_length = std::sqrt((right[0] * right[0]) + (right[2] * right[2]));
            right[0] /= right_length;
            right[2] /= right_length;
            up[0] = (right[1] * forward[2]) - (right[2] * forward[1]);
            up[1] = (right[2] * forward[0]) - (right[0] * forward[2]);
            up[2] = (right[0] * forward[1]) - (right[1] * forward[0]);
        }
    };

    class overlay final {
    private:
        std::vector<unsigned char> pixels;
        int width = 0;
        int height = 0;
        GLuint texture = 0;

    public:
        gtl::imgui_software_renderer renderer;

        void begin(const int window_width, const int window_height) {
            this->width = window_width;
            this->height = window_height;
            this->pixels.assign(static_cast<std::size_t>(this->width) * static_cast<std::size_t>(this->height) * 4, 0);
            this->renderer.set_buffer(this->pixels.data(), this->width, this->height, this->width * 4);
        }

        void flush() {
            if (this->texture == 0) {
                glGenTextures(1, &this->texture);
            }
            glPushAttrib(GL_ENABLE_BIT | GL_VIEWPORT_BIT | GL_TRANSFORM_BIT | GL_TEXTURE_BIT | GL_CURRENT_BIT | GL_COLOR_BUFFER_BIT);
            glDisable(GL_DEPTH_TEST);
            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_TEXTURE_2D);
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glOrtho(0, this->width, this->height, 0, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();
            glViewport(0, 0, this->width, this->height);
            glBindTexture(GL_TEXTURE_2D, this->texture);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, this->width, this->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, this->pixels.data());
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f);
            glVertex2f(0.0f, 0.0f);
            glTexCoord2f(1.0f, 0.0f);
            glVertex2f(static_cast<float>(this->width), 0.0f);
            glTexCoord2f(1.0f, 1.0f);
            glVertex2f(static_cast<float>(this->width), static_cast<float>(this->height));
            glTexCoord2f(0.0f, 1.0f);
            glVertex2f(0.0f, static_cast<float>(this->height));
            glEnd();
            glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glPopAttrib();
        }

        void release() {
            if (this->texture != 0) {
                glDeleteTextures(1, &this->texture);
                this->texture = 0;
            }
        }
    };

    gui::rotation3 rotation_from_quaternion(const double x, const double y, const double z, const double w) {
        const double quaternion_xyzw[4] = { x, y, z, w };
        gui::rotation3 matrix;
        rotation::quaternion_to_matrix(&quaternion_xyzw[0], &matrix.rows[0][0]);
        return matrix;
    }

    struct anchor_pair {
        double estimated[3] = { 0.0, 0.0, 0.0 };
        double truth[3] = { 0.0, 0.0, 0.0 };
        gui::rotation3 estimated_rotation = { { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } } };
        gui::rotation3 truth_rotation = { { { 1.0, 0.0, 0.0 }, { 0.0, 1.0, 0.0 }, { 0.0, 0.0, 1.0 } } };
    };

    std::vector<dataset::trajectory_pose> camera_ground_truth(const scene::scene& loaded, const int primary_channel) {
        std::vector<dataset::trajectory_pose> poses = loaded.ground_truth();
        if ((primary_channel < 0) || (primary_channel >= static_cast<int>(loaded.image_channels().size()))) {
            return poses;
        }
        const scene::extrinsic& mounting = loaded.image_channels()[static_cast<std::size_t>(primary_channel)].mounting;
        const bool identity =
            (mounting.translation[0] == 0.0) && (mounting.translation[1] == 0.0) && (mounting.translation[2] == 0.0) &&
            (mounting.rotation[0] == 0.0) && (mounting.rotation[1] == 0.0) && (mounting.rotation[2] == 0.0) && (mounting.rotation[3] == 1.0);
        if (!mounting.valid || identity) {
            return poses;
        }
        const double bx = mounting.rotation[0];
        const double by = mounting.rotation[1];
        const double bz = mounting.rotation[2];
        const double bw = mounting.rotation[3];
        for (dataset::trajectory_pose& pose : poses) {
            const gui::rotation3 world_from_ego = rotation_from_quaternion(pose.quaternion_x, pose.quaternion_y, pose.quaternion_z, pose.quaternion_w);
            pose.x_coordinate += (world_from_ego[0][0] * mounting.translation[0]) + (world_from_ego[0][1] * mounting.translation[1]) + (world_from_ego[0][2] * mounting.translation[2]);
            pose.y_coordinate += (world_from_ego[1][0] * mounting.translation[0]) + (world_from_ego[1][1] * mounting.translation[1]) + (world_from_ego[1][2] * mounting.translation[2]);
            pose.z_coordinate += (world_from_ego[2][0] * mounting.translation[0]) + (world_from_ego[2][1] * mounting.translation[1]) + (world_from_ego[2][2] * mounting.translation[2]);
            const double ax = pose.quaternion_x;
            const double ay = pose.quaternion_y;
            const double az = pose.quaternion_z;
            const double aw = pose.quaternion_w;
            pose.quaternion_x = (aw * bx) + (ax * bw) + (ay * bz) - (az * by);
            pose.quaternion_y = (aw * by) - (ax * bz) + (ay * bw) + (az * bx);
            pose.quaternion_z = (aw * bz) + (ax * by) - (ay * bx) + (az * bw);
            pose.quaternion_w = (aw * bw) - (ax * bx) - (ay * by) - (az * bz);
        }
        return poses;
    }

    metrics::similarity map_placement(const std::vector<dataset::trajectory_pose>& truth_camera, const metrics::result& accuracy, const anchor_pair& paired, const bool align_to_truth, const bool scale_to_truth) {
        metrics::similarity placement;
        if (truth_camera.empty()) {
            return placement;
        }
        placement.valid = true;
        placement.scale = (scale_to_truth && accuracy.valid) ? accuracy.transform.scale : 1.0;

        anchor_pair pin = paired;
        if (!accuracy.valid) {
            const dataset::trajectory_pose& first = truth_camera.front();
            pin = anchor_pair();
            pin.truth[0] = first.x_coordinate;
            pin.truth[1] = first.y_coordinate;
            pin.truth[2] = first.z_coordinate;
            pin.truth_rotation = rotation_from_quaternion(first.quaternion_x, first.quaternion_y, first.quaternion_z, first.quaternion_w);
        }

        if (align_to_truth && accuracy.valid) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    placement.rotation[row][column] = accuracy.transform.rotation[row][column];
                }
            }
        }
        else {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    placement.rotation[row][column] = (pin.truth_rotation[row][0] * pin.estimated_rotation[column][0]) + (pin.truth_rotation[row][1] * pin.estimated_rotation[column][1]) + (pin.truth_rotation[row][2] * pin.estimated_rotation[column][2]);
                }
            }
        }

        for (int row = 0; row < 3; ++row) {
            const double rotated = (placement.rotation[row][0] * pin.estimated[0]) + (placement.rotation[row][1] * pin.estimated[1]) + (placement.rotation[row][2] * pin.estimated[2]);
            placement.translation[row] = pin.truth[row] - (placement.scale * rotated);
        }
        return placement;
    }

    struct estimated_trajectory {
        std::vector<dataset::trajectory_pose> poses;
        std::vector<const gui::render_snapshot::frame*> frames;

        explicit estimated_trajectory(const gui::render_snapshot& snapshot) {
            this->frames.reserve(snapshot.frames.size());
            for (const auto& [id, pose] : snapshot.frames) {
                static_cast<void>(id);
                if (pose.timestamp_nanoseconds >= 0) {
                    this->frames.push_back(&pose);
                }
            }
            std::sort(this->frames.begin(), this->frames.end(), [](const gui::render_snapshot::frame* const lhs, const gui::render_snapshot::frame* const rhs) {
                return lhs->timestamp_nanoseconds < rhs->timestamp_nanoseconds;
            });
            this->poses.reserve(this->frames.size());
            for (const gui::render_snapshot::frame* const pose : this->frames) {
                dataset::trajectory_pose entry;
                entry.timestamp_nanoseconds = pose->timestamp_nanoseconds;
                double centre[3];
                gui::render_snapshot::camera_centre(*pose, centre);
                entry.x_coordinate = centre[0];
                entry.y_coordinate = centre[1];
                entry.z_coordinate = centre[2];
                double world_from_camera[9];
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        world_from_camera[(3 * row) + column] = pose->rotation[column][row];
                    }
                }
                double quaternion_xyzw[4];
                rotation::matrix_to_quaternion(&world_from_camera[0], &quaternion_xyzw[0]);
                entry.quaternion_x = quaternion_xyzw[0];
                entry.quaternion_y = quaternion_xyzw[1];
                entry.quaternion_z = quaternion_xyzw[2];
                entry.quaternion_w = quaternion_xyzw[3];
                this->poses.push_back(entry);
            }
        }

        anchor_pair anchor(const metrics::result& accuracy, const std::vector<dataset::trajectory_pose>& truth_camera) const {
            anchor_pair pin;
            if (!accuracy.valid) {
                return pin;
            }
            const dataset::trajectory_pose& truth = truth_camera[accuracy.pairs.front().first];
            const dataset::trajectory_pose& estimated = this->poses[accuracy.pairs.front().second];
            const gui::render_snapshot::frame& frame = *this->frames[accuracy.pairs.front().second];
            pin.estimated[0] = estimated.x_coordinate;
            pin.estimated[1] = estimated.y_coordinate;
            pin.estimated[2] = estimated.z_coordinate;
            pin.truth[0] = truth.x_coordinate;
            pin.truth[1] = truth.y_coordinate;
            pin.truth[2] = truth.z_coordinate;
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    pin.estimated_rotation[row][column] = frame.rotation[column][row];
                }
            }
            pin.truth_rotation = rotation_from_quaternion(truth.quaternion_x, truth.quaternion_y, truth.quaternion_z, truth.quaternion_w);
            return pin;
        }
    };

    const gui::render_snapshot::frame* nearest_frame(const gui::render_snapshot& snapshot, const long long timestamp_nanoseconds, int& frame_id) {
        const gui::render_snapshot::frame* nearest = nullptr;
        long long best = -1;
        for (const auto& [id, pose] : snapshot.frames) {
            if (pose.timestamp_nanoseconds < 0) {
                continue;
            }
            const long long difference = std::llabs(pose.timestamp_nanoseconds - timestamp_nanoseconds);
            if ((best < 0) || (difference < best)) {
                best = difference;
                nearest = &pose;
                frame_id = id;
            }
        }
        return nearest;
    }

    int primary_visual_channel(const scene::scene& loaded) {
        for (std::size_t i = 0; i < loaded.image_channels().size(); ++i) {
            const scene::image_channel& channel = loaded.image_channels()[i];
            if (channel.selected_for_slam && !channel.is_depth() && channel.has_intrinsics) {
                return static_cast<int>(i);
            }
        }
        return -1;
    }

    void relative_mounting(const scene::extrinsic& from, const scene::extrinsic& to, double rotation[3][3], double translation[3]) {
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                rotation[row][column] = (row == column) ? 1.0 : 0.0;
            }
            translation[row] = 0.0;
        }
        if (!from.valid || !to.valid) {
            return;
        }
        const gui::rotation3 ego_from_source = rotation_from_quaternion(from.rotation[0], from.rotation[1], from.rotation[2], from.rotation[3]);
        const gui::rotation3 ego_from_target = rotation_from_quaternion(to.rotation[0], to.rotation[1], to.rotation[2], to.rotation[3]);
        for (int row = 0; row < 3; ++row) {
            for (int column = 0; column < 3; ++column) {
                rotation[row][column] = (ego_from_target[0][row] * ego_from_source[0][column]) + (ego_from_target[1][row] * ego_from_source[1][column]) + (ego_from_target[2][row] * ego_from_source[2][column]);
            }
            translation[row] = (ego_from_target[0][row] * (from.translation[0] - to.translation[0])) + (ego_from_target[1][row] * (from.translation[1] - to.translation[1])) + (ego_from_target[2][row] * (from.translation[2] - to.translation[2]));
        }
    }

    bool same_mounting(const scene::extrinsic& lhs, const scene::extrinsic& rhs) {
        if (!lhs.valid || !rhs.valid) {
            return false;
        }
        for (int i = 0; i < 3; ++i) {
            if (lhs.translation[i] != rhs.translation[i]) {
                return false;
            }
        }
        for (int i = 0; i < 4; ++i) {
            if (lhs.rotation[i] != rhs.rotation[i]) {
                return false;
            }
        }
        return true;
    }

    struct playback {
        enum class mode {
            stopped,
            paused,
            playing
        };
        mode state = mode::stopped;
        long long time_nanoseconds = 0;
        float speed = 1.0f;

        void seek_to_fraction(const scene::scene& loaded, const double fraction) {
            this->time_nanoseconds = loaded.begin_time() + static_cast<long long>(fraction * static_cast<double>(loaded.duration_nanoseconds()));
        }

        double fraction_of(const scene::scene& loaded) const {
            const long long duration = loaded.duration_nanoseconds();
            if (duration <= 0) {
                return 0.0;
            }
            const double fraction = static_cast<double>(this->time_nanoseconds - loaded.begin_time()) / static_cast<double>(duration);
            return (fraction < 0.0) ? 0.0 : ((fraction > 1.0) ? 1.0 : fraction);
        }
    };

    class slam_worker final {
    public:
        std::atomic<int> channel{ -1 };
        std::atomic<long long> playback_time_nanoseconds{ 0 };
        std::atomic<long long> pending_time_nanoseconds{ -1 };
        std::atomic<bool> reset{ false };
        std::atomic<int> tracker_choice{ 0 };
        std::atomic<bool> lines_enabled{ false };
        std::atomic<bool> culling_enabled{ true };
        std::atomic<int> association_choice{ 0 };

    private:
        std::thread thread;
        std::atomic<bool> stop_requested{ false };
        zeroslam::system system;
        constexpr static const int sensor_id = 1;
        std::vector<gui::render_snapshot::frame> fed_frames;
        std::vector<zeroslam_point_struct> points;
        std::vector<zeroslam_line_struct> lines;
        std::vector<zeroslam_edge_struct> edges;
        std::vector<long long> keyframes;
        int applied_tracker = -1;
        bool applied_lines = false;
        bool applied_culling = true;
        int applied_association = -1;
        gtl::triple_buffer<gui::render_snapshot> snapshots;

        void publish() {
            this->snapshots.update_write();
        }

        void apply_configuration() {
            const int tracker = this->tracker_choice.load();
            const bool lines_on = this->lines_enabled.load();
            const bool culling_on = this->culling_enabled.load();
            const int association = this->association_choice.load();
            if ((tracker == this->applied_tracker) && (lines_on == this->applied_lines) && (culling_on == this->applied_culling) && (association == this->applied_association)) {
                return;
            }
            char text[128];
            const int length = std::snprintf(&text[0], sizeof(text), "tracker=%s\nlines=%s\nculling=%s\nassociation=%s\n", (tracker == 1) ? "extrema" : "klt", lines_on ? "on" : "off", culling_on ? "on" : "off", association_names[static_cast<std::size_t>(((association >= 0) && (association < 3)) ? association : 0)]);
            if ((length > 0) && (this->system.set_configuration(&text[0], length) != zeroslam_return_success)) {
                std::fprintf(stderr, "The library rejected the front end configuration.\n");
            }
            this->applied_tracker = tracker;
            this->applied_lines = lines_on;
            this->applied_culling = culling_on;
            this->applied_association = association;
        }

        bool describe_rig(const scene::image_channel& channel_fed) {
            zeroslam_sensor_parameters_camera_struct camera{};
            camera.width = static_cast<int>(channel_fed.width);
            camera.height = static_cast<int>(channel_fed.height);
            camera.focal_x = channel_fed.camera_parameters[0];
            camera.focal_y = channel_fed.camera_parameters[1];
            camera.centre_x = channel_fed.camera_parameters[2];
            camera.centre_y = channel_fed.camera_parameters[3];
            for (std::size_t i = 0; i < 8; ++i) {
                camera.distortion[i] = channel_fed.camera_parameters[4 + i];
            }
            zeroslam_sensor_rig_struct rig{};
            rig.type = zeroslam_sensor_camera;
            rig.sensor_id = slam_worker::sensor_id;
            rig.parameters_length = static_cast<int>(sizeof(camera));
            rig.parameters_data = &camera;
            return this->system.set_sensor_rig(&rig, 1) == zeroslam_return_success;
        }

        void capture(const double process_seconds, const long long frame_timestamp_nanoseconds) {
            gui::render_snapshot& snapshot = this->snapshots.get_write();
            snapshot.frames.clear();
            snapshot.landmarks.clear();
            snapshot.lines.clear();
            snapshot.edges.clear();
            snapshot.keyframes.clear();
            snapshot.processed_frame_count = static_cast<int>(this->fed_frames.size());
            snapshot.last_process_seconds = process_seconds;
            snapshot.last_frame_timestamp_nanoseconds = frame_timestamp_nanoseconds;
            for (std::size_t id = 0; id < this->fed_frames.size(); ++id) {
                zeroslam_pose_struct reported{};
                if (this->system.get_pose_at_timestamp(&reported, this->fed_frames[id].timestamp_nanoseconds) != zeroslam_return_success) {
                    continue;
                }
                gui::render_snapshot::frame& pose = snapshot.frames[static_cast<int>(id)];
                pose = this->fed_frames[id];
                gui::rotation3 camera_to_world;
                rotation::quaternion_to_matrix(&reported.pose[3], &camera_to_world.rows[0][0]);
                for (int row = 0; row < 3; ++row) {
                    for (int column = 0; column < 3; ++column) {
                        pose.rotation[row][column] = camera_to_world[column][row];
                    }
                }
                for (int row = 0; row < 3; ++row) {
                    pose.translation[row] = -((pose.rotation[row][0] * reported.pose[0]) + (pose.rotation[row][1] * reported.pose[1]) + (pose.rotation[row][2] * reported.pose[2]));
                }
            }
            zeroslam_map_chunk_struct chunk{};
            zeroslam_return_enum result = this->system.get_map_chunk(0.0f, 0.0f, 0.0f, &chunk);
            if (result == zeroslam_return_failure_insufficient_data_length) {
                this->points.resize(static_cast<std::size_t>(chunk.points_length));
                chunk.points = this->points.data();
                result = this->system.get_map_chunk(0.0f, 0.0f, 0.0f, &chunk);
            }
            if (result == zeroslam_return_success) {
                for (int i = 0; i < chunk.points_length; ++i) {
                    const zeroslam_point_struct& point = this->points[static_cast<std::size_t>(i)];
                    gui::render_snapshot::landmark& landmark = snapshot.landmarks[i];
                    landmark.location[0] = static_cast<double>(point.x);
                    landmark.location[1] = static_cast<double>(point.y);
                    landmark.location[2] = static_cast<double>(point.z);
                    landmark.colour[0] = static_cast<double>(point.r);
                    landmark.colour[1] = static_cast<double>(point.g);
                    landmark.colour[2] = static_cast<double>(point.b);
                }
            }
            zeroslam_map_lines_struct line_chunk{};
            result = this->system.get_map_lines(&line_chunk);
            if (result == zeroslam_return_failure_insufficient_data_length) {
                this->lines.resize(static_cast<std::size_t>(line_chunk.lines_length));
                line_chunk.lines = this->lines.data();
                result = this->system.get_map_lines(&line_chunk);
            }
            if (result == zeroslam_return_success) {
                for (int i = 0; i < line_chunk.lines_length; ++i) {
                    const zeroslam_line_struct& segment = this->lines[static_cast<std::size_t>(i)];
                    gui::render_snapshot::line& line = snapshot.lines[i];
                    line.a[0] = static_cast<double>(segment.x1);
                    line.a[1] = static_cast<double>(segment.y1);
                    line.a[2] = static_cast<double>(segment.z1);
                    line.b[0] = static_cast<double>(segment.x2);
                    line.b[1] = static_cast<double>(segment.y2);
                    line.b[2] = static_cast<double>(segment.z2);
                }
            }
            zeroslam_map_edges_struct edge_chunk{};
            result = this->system.get_map_edges(&edge_chunk);
            if (result == zeroslam_return_failure_insufficient_data_length) {
                this->edges.resize(static_cast<std::size_t>(edge_chunk.edges_length));
                edge_chunk.edges = this->edges.data();
                result = this->system.get_map_edges(&edge_chunk);
            }
            if (result == zeroslam_return_success) {
                std::unordered_map<long long, int> frame_of_instant;
                frame_of_instant.reserve(this->fed_frames.size());
                for (std::size_t id = 0; id < this->fed_frames.size(); ++id) {
                    frame_of_instant[this->fed_frames[id].timestamp_nanoseconds] = static_cast<int>(id);
                }
                for (int i = 0; i < edge_chunk.edges_length; ++i) {
                    const zeroslam_edge_struct& reported = this->edges[static_cast<std::size_t>(i)];
                    const auto a = frame_of_instant.find(reported.timestamp_a);
                    const auto b = frame_of_instant.find(reported.timestamp_b);
                    if ((a == frame_of_instant.end()) || (b == frame_of_instant.end())) {
                        continue;
                    }
                    snapshot.edges.push_back(gui::render_snapshot::edge{ a->second, b->second, reported.type, reported.weight });
                }
                zeroslam_map_keyframes_struct keyframe_chunk{};
                result = this->system.get_map_keyframes(&keyframe_chunk);
                if (result == zeroslam_return_failure_insufficient_data_length) {
                    this->keyframes.resize(static_cast<std::size_t>(keyframe_chunk.keyframes_length));
                    keyframe_chunk.keyframes = this->keyframes.data();
                    result = this->system.get_map_keyframes(&keyframe_chunk);
                }
                if (result == zeroslam_return_success) {
                    for (int i = 0; i < keyframe_chunk.keyframes_length; ++i) {
                        const auto found = frame_of_instant.find(this->keyframes[static_cast<std::size_t>(i)]);
                        if (found != frame_of_instant.end()) {
                            snapshot.keyframes.insert(found->second);
                        }
                    }
                }
            }
        }

        void run(const scene::scene* const loaded) {
            int fed_messages = 0;
            int fed_channel = -1;
            bool rig_described = false;
            while (!this->stop_requested.load()) {
                const int channel_index = this->channel.load();
                if (this->reset.exchange(false) || (channel_index != fed_channel)) {
                    this->system = zeroslam::system();
                    this->applied_tracker = -1;
                    this->fed_frames.clear();
                    fed_messages = 0;
                    fed_channel = channel_index;
                    rig_described = false;
                    this->pending_time_nanoseconds.store(-1);
                    this->capture(0.0, -1);
                    this->publish();
                    continue;
                }
                if (!loaded->is_loaded() || (channel_index < 0) || (channel_index >= static_cast<int>(loaded->image_channels().size()))) {
                    this->pending_time_nanoseconds.store(-1);
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                    continue;
                }
                const scene::image_channel& channel_fed = loaded->image_channels()[static_cast<std::size_t>(channel_index)];
                if (!channel_fed.has_intrinsics || (fed_messages >= static_cast<int>(channel_fed.messages.size()))) {
                    this->pending_time_nanoseconds.store(-1);
                    std::this_thread::sleep_for(std::chrono::milliseconds(4));
                    continue;
                }
                if (!rig_described) {
                    rig_described = this->describe_rig(channel_fed);
                    if (!rig_described) {
                        std::fprintf(stderr, "The library rejected the camera of channel %d.\n", channel_index);
                        this->pending_time_nanoseconds.store(-1);
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                        continue;
                    }
                }
                const long long message_time = channel_fed.log_times[static_cast<std::size_t>(fed_messages)];
                this->pending_time_nanoseconds.store(message_time);
                if (message_time > this->playback_time_nanoseconds.load()) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    continue;
                }
                scene::decoded_image decoded;
                if (!loaded->decode_image(channel_fed, static_cast<std::size_t>(fed_messages), decoded)) {
                    ++fed_messages;
                    continue;
                }
                this->apply_configuration();
                zeroslam_sensor_data_struct measurement{};
                measurement.timestamp = decoded.timestamp_nanoseconds;
                measurement.sensor_id = slam_worker::sensor_id;
                measurement.measurement_length = static_cast<int>(decoded.grey.size());
                measurement.measurement_data = decoded.grey.data();
                const std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
                const zeroslam_return_enum fed = this->system.set_sensor_data(&measurement, 1);
                const double elapsed_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
                if (fed != zeroslam_return_success) {
                    std::fprintf(stderr, "The library rejected frame %d: %s.\n", fed_messages, zeroslam_return_enum_to_string(fed));
                }
                gui::render_snapshot::frame metadata;
                metadata.image_width = static_cast<int>(decoded.width);
                metadata.image_height = static_cast<int>(decoded.height);
                metadata.timestamp_nanoseconds = decoded.timestamp_nanoseconds;
                for (std::size_t i = 0; i < gui::render_snapshot::camera_parameter_count; ++i) {
                    metadata.camera_parameters[i] = channel_fed.camera_parameters[i];
                }
                this->fed_frames.push_back(metadata);
                ++fed_messages;
                this->capture(elapsed_seconds, decoded.timestamp_nanoseconds);
                this->publish();
            }
        }

    public:
        slam_worker() = default;
        slam_worker(const slam_worker&) = delete;
        slam_worker& operator=(const slam_worker&) = delete;

        ~slam_worker() {
            this->stop();
        }

        void start(const scene::scene& loaded) {
            this->stop();
            this->stop_requested.store(false);
            this->reset.store(true);
            this->thread = std::thread(&slam_worker::run, this, &loaded);
        }

        void stop() {
            this->stop_requested.store(true);
            if (this->thread.joinable()) {
                this->thread.join();
            }
        }

        const gui::render_snapshot& read() {
            this->snapshots.update_read();
            return this->snapshots.get_read();
        }

        void clear() {
            this->snapshots.get_write() = gui::render_snapshot();
            this->snapshots.update_write();
            this->snapshots.update_read();
        }
    };

    struct map_geometry {
        std::vector<float> landmark_vertices;
        std::vector<float> landmark_colours;
        std::vector<float> line_vertices;
        std::vector<float> line_colours;
        std::vector<float> covisibility_vertices;
        std::vector<float> covisibility_colours;
        std::vector<float> loop_vertices;
        std::vector<float> loop_colours;
        const gui::render_snapshot* built_from = nullptr;
        int built_count = -1;
        int built_covisibility_minimum = -1;

        void update(const gui::render_snapshot& snapshot, const int covisibility_minimum) {
            if ((&snapshot == this->built_from) && (snapshot.processed_frame_count == this->built_count) && (covisibility_minimum == this->built_covisibility_minimum)) {
                return;
            }
            this->built_from = &snapshot;
            this->built_count = snapshot.processed_frame_count;
            this->built_covisibility_minimum = covisibility_minimum;
            this->line_vertices.clear();
            this->line_colours.clear();
            for (const auto& [id, line] : snapshot.lines) {
                static_cast<void>(id);
                for (int axis = 0; axis < 3; ++axis) {
                    this->line_vertices.push_back(static_cast<float>(line.a[axis]));
                }
                for (int axis = 0; axis < 3; ++axis) {
                    this->line_vertices.push_back(static_cast<float>(line.b[axis]));
                }
                for (int end = 0; end < 2; ++end) {
                    this->line_colours.push_back(1.0f);
                    this->line_colours.push_back(0.6f);
                    this->line_colours.push_back(0.1f);
                }
            }
            this->covisibility_vertices.clear();
            this->covisibility_colours.clear();
            this->loop_vertices.clear();
            this->loop_colours.clear();
            for (const gui::render_snapshot::edge& edge : snapshot.edges) {
                const auto a = snapshot.frames.find(edge.frame_a);
                const auto b = snapshot.frames.find(edge.frame_b);
                if ((a == snapshot.frames.end()) || (b == snapshot.frames.end())) {
                    continue;
                }
                if ((edge.kind == 0) && (edge.weight < covisibility_minimum)) {
                    continue;
                }
                double centre_a[3];
                double centre_b[3];
                gui::render_snapshot::camera_centre(a->second, centre_a);
                gui::render_snapshot::camera_centre(b->second, centre_b);
                std::vector<float>& vertices = (edge.kind == 0) ? this->covisibility_vertices : this->loop_vertices;
                std::vector<float>& colours = (edge.kind == 0) ? this->covisibility_colours : this->loop_colours;
                for (int axis = 0; axis < 3; ++axis) {
                    vertices.push_back(static_cast<float>(centre_a[axis]));
                }
                for (int axis = 0; axis < 3; ++axis) {
                    vertices.push_back(static_cast<float>(centre_b[axis]));
                }
                const float strength = (edge.kind == 0) ? std::min(1.0f, 0.25f + (0.75f * static_cast<float>(edge.weight - covisibility_minimum) / static_cast<float>(std::max(1, 3 * covisibility_minimum)))) : 1.0f;
                for (int end = 0; end < 2; ++end) {
                    if (edge.kind == 0) {
                        colours.push_back(0.3f * strength);
                        colours.push_back(0.6f * strength);
                        colours.push_back(1.0f * strength);
                    }
                    else {
                        colours.push_back(1.0f);
                        colours.push_back(0.5f);
                        colours.push_back(0.0f);
                    }
                }
            }
            this->landmark_vertices.clear();
            this->landmark_colours.clear();
            this->landmark_vertices.reserve(snapshot.landmarks.size() * 3);
            this->landmark_colours.reserve(snapshot.landmarks.size() * 3);
            for (const auto& [id, landmark] : snapshot.landmarks) {
                static_cast<void>(id);
                for (int axis = 0; axis < 3; ++axis) {
                    this->landmark_vertices.push_back(static_cast<float>(landmark.location[axis]));
                    this->landmark_colours.push_back(static_cast<float>(landmark.colour[axis]));
                }
            }
        }
    };

    struct depth_cloud {
        float stride = 2.0f;
        float maximum_metres = 15.0f;

        std::vector<float> vertices;
        std::vector<float> colours;
        int colour_channel = -1;
        bool colour_registered = false;

        long long built_depth_image = -1;
        long long built_colour_image = -1;
        int built_colour_channel = -1;
        int built_frame = -1;
        float built_stride = -1.0f;
        float built_maximum_metres = -1.0f;
        double built_metres_to_map = -1.0;

        void invalidate() {
            this->vertices.clear();
            this->colours.clear();
            this->built_depth_image = -1;
            this->built_frame = -1;
        }

        void update(const scene::scene& loaded, const std::vector<scene::decoded_image>& channel_images, const gui::render_snapshot& snapshot, const int primary, const double metres_to_map) {
            const std::vector<scene::image_channel>& channels = loaded.image_channels();
            int depth_index = -1;
            for (std::size_t i = 0; i < channels.size(); ++i) {
                if (channels[i].is_depth() && channel_images[i].is_valid()) {
                    depth_index = static_cast<int>(i);
                    break;
                }
            }
            if ((depth_index < 0) || (primary < 0)) {
                this->invalidate();
                return;
            }
            const scene::image_channel& depth_channel = channels[static_cast<std::size_t>(depth_index)];
            const scene::decoded_image& depth_image = channel_images[static_cast<std::size_t>(depth_index)];

            int placement_frame = -1;
            const gui::render_snapshot::frame* const placement_pose = nearest_frame(snapshot, depth_image.timestamp_nanoseconds, placement_frame);
            if (placement_pose == nullptr) {
                return;
            }

            const auto can_project_into = [&](const int index) {
                if ((index < 0) || (index >= static_cast<int>(channels.size()))) {
                    return false;
                }
                const scene::image_channel& candidate = channels[static_cast<std::size_t>(index)];
                return !candidate.is_depth() && candidate.has_intrinsics && channel_images[static_cast<std::size_t>(index)].is_valid();
            };
            int colour_index = -1;
            bool registered = false;
            for (std::size_t i = 0; i < channels.size(); ++i) {
                const scene::image_channel& candidate = channels[i];
                if (!candidate.is_depth() && channel_images[i].is_valid() && same_mounting(candidate.mounting, depth_channel.mounting) && (channel_images[i].width == depth_image.width) && (channel_images[i].height == depth_image.height)) {
                    colour_index = static_cast<int>(i);
                    registered = true;
                    break;
                }
            }
            if ((colour_index < 0) && can_project_into(primary)) {
                colour_index = primary;
            }
            for (std::size_t i = 0; (colour_index < 0) && (i < channels.size()); ++i) {
                if (can_project_into(static_cast<int>(i))) {
                    colour_index = static_cast<int>(i);
                }
            }
            const scene::image_channel* const colour_channel_record = (colour_index >= 0) ? &channels[static_cast<std::size_t>(colour_index)] : nullptr;
            const scene::decoded_image* const colour_image = (colour_index >= 0) ? &channel_images[static_cast<std::size_t>(colour_index)] : nullptr;
            const long long colour_time = (colour_image != nullptr) ? colour_image->timestamp_nanoseconds : -1;

            const bool unchanged =
                (depth_image.timestamp_nanoseconds == this->built_depth_image) && (placement_frame == this->built_frame) &&
                (colour_time == this->built_colour_image) && (colour_index == this->built_colour_channel) &&
                (this->stride == this->built_stride) && (this->maximum_metres == this->built_maximum_metres) && (metres_to_map == this->built_metres_to_map);
            if (unchanged) {
                return;
            }
            this->built_depth_image = depth_image.timestamp_nanoseconds;
            this->built_frame = placement_frame;
            this->built_colour_image = colour_time;
            this->built_colour_channel = colour_index;
            this->built_stride = this->stride;
            this->built_maximum_metres = this->maximum_metres;
            this->built_metres_to_map = metres_to_map;
            this->colour_channel = colour_index;
            this->colour_registered = registered;
            this->vertices.clear();
            this->colours.clear();

            const double inverse_fx = (depth_channel.fx > 0.0) ? (1.0 / depth_channel.fx) : 0.0;
            const double inverse_fy = (depth_channel.fy > 0.0) ? (1.0 / depth_channel.fy) : 0.0;
            if ((inverse_fx == 0.0) || (inverse_fy == 0.0) || depth_image.metres.empty()) {
                return;
            }

            double depth_to_camera_rotation[3][3];
            double depth_to_camera_translation[3];
            relative_mounting(depth_channel.mounting, channels[static_cast<std::size_t>(primary)].mounting, depth_to_camera_rotation, depth_to_camera_translation);
            double depth_to_colour_rotation[3][3];
            double depth_to_colour_translation[3];
            if ((colour_channel_record != nullptr) && !registered) {
                relative_mounting(depth_channel.mounting, colour_channel_record->mounting, depth_to_colour_rotation, depth_to_colour_translation);
            }

            const auto& pose_rotation = placement_pose->rotation;
            double centre[3];
            gui::render_snapshot::camera_centre(*placement_pose, centre);

            const int step = (this->stride < 1.0f) ? 1 : static_cast<int>(this->stride);
            const int width = static_cast<int>(depth_image.width);
            const int height = static_cast<int>(depth_image.height);
            this->vertices.reserve(static_cast<std::size_t>((width / step) * (height / step) * 3));
            this->colours.reserve(static_cast<std::size_t>((width / step) * (height / step) * 3));
            const auto push_rgb = [&](const std::vector<unsigned char>& rgb, const std::size_t pixel) {
                for (int component = 0; component < 3; ++component) {
                    this->colours.push_back(static_cast<float>(rgb[(pixel * 3) + static_cast<std::size_t>(component)]) / 255.0f);
                }
            };
            for (int y = 0; y < height; y += step) {
                for (int x = 0; x < width; x += step) {
                    const std::size_t pixel = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width)) + static_cast<std::size_t>(x);
                    const float range = depth_image.metres[pixel];
                    if (!(range > 0.0f) || (range > this->maximum_metres)) {
                        continue;
                    }
                    const double sensor[3] = {
                        (static_cast<double>(x) + 0.5 - depth_channel.cx) * inverse_fx * static_cast<double>(range),
                        (static_cast<double>(y) + 0.5 - depth_channel.cy) * inverse_fy * static_cast<double>(range),
                        static_cast<double>(range)
                    };
                    double in_camera[3];
                    for (int row = 0; row < 3; ++row) {
                        in_camera[row] = ((depth_to_camera_rotation[row][0] * sensor[0]) + (depth_to_camera_rotation[row][1] * sensor[1]) + (depth_to_camera_rotation[row][2] * sensor[2]) + depth_to_camera_translation[row]) * metres_to_map;
                    }
                    for (int axis = 0; axis < 3; ++axis) {
                        this->vertices.push_back(static_cast<float>((pose_rotation[0][axis] * in_camera[0]) + (pose_rotation[1][axis] * in_camera[1]) + (pose_rotation[2][axis] * in_camera[2]) + centre[axis]));
                    }

                    if ((colour_image != nullptr) && registered) {
                        push_rgb(colour_image->rgb, pixel);
                        continue;
                    }
                    if (colour_image != nullptr) {
                        double in_colour[3];
                        for (int row = 0; row < 3; ++row) {
                            in_colour[row] = (depth_to_colour_rotation[row][0] * sensor[0]) + (depth_to_colour_rotation[row][1] * sensor[1]) + (depth_to_colour_rotation[row][2] * sensor[2]) + depth_to_colour_translation[row];
                        }
                        if (in_colour[2] > 1.0e-6) {
                            const long u = static_cast<long>(std::floor((colour_channel_record->camera_parameters[0] * (in_colour[0] / in_colour[2])) + colour_channel_record->camera_parameters[2]));
                            const long v = static_cast<long>(std::floor((colour_channel_record->camera_parameters[1] * (in_colour[1] / in_colour[2])) + colour_channel_record->camera_parameters[3]));
                            if ((u >= 0) && (v >= 0) && (u < static_cast<long>(colour_image->width)) && (v < static_cast<long>(colour_image->height))) {
                                push_rgb(colour_image->rgb, (static_cast<std::size_t>(v) * static_cast<std::size_t>(colour_image->width)) + static_cast<std::size_t>(u));
                                continue;
                            }
                        }
                    }
                    push_rgb(depth_image.rgb, pixel);
                }
            }
        }
    };
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    std::string scene_path;
    std::string screenshot_path;
    std::string load_map_path;
    std::string save_map_path;
    long long screenshot_after_frames = 30;
    long long exit_after_frames = -1;
    bool autoplay = false;
    int initial_tracker = 0;
    bool initial_lines = false;
    bool initial_culling = true;
    int initial_association = 2;
    for (int i = 1; i < argc; ++i) {
        const auto take_string = [&](const char* const name, std::string& destination) -> bool {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Missing value for option: %s\n", name);
                return false;
            }
            destination = argv[++i];
            return true;
        };
        const auto take_number = [&](const char* const name, long long& destination) -> bool {
            std::string text;
            if (!take_string(name, text)) {
                return false;
            }
            destination = std::strtoll(text.c_str(), nullptr, 10);
            return true;
        };
        if ((std::strcmp(argv[i], "--help") == 0) || (std::strcmp(argv[i], "-h") == 0)) {
            print_usage(argv[0]);
            return EXIT_SUCCESS;
        }
        if (std::strcmp(argv[i], "--play") == 0) {
            autoplay = true;
        }
        else if (std::strcmp(argv[i], "--load-map") == 0) {
            if (!take_string("--load-map", load_map_path)) {
                return EXIT_FAILURE;
            }
        }
        else if (std::strcmp(argv[i], "--save-map") == 0) {
            if (!take_string("--save-map", save_map_path)) {
                return EXIT_FAILURE;
            }
        }
        else if (std::strcmp(argv[i], "--screenshot") == 0) {
            if (!take_string("--screenshot", screenshot_path)) {
                return EXIT_FAILURE;
            }
        }
        else if (std::strcmp(argv[i], "--screenshot-after") == 0) {
            if (!take_number("--screenshot-after", screenshot_after_frames)) {
                return EXIT_FAILURE;
            }
        }
        else if (std::strcmp(argv[i], "--config") == 0) {
            std::string setting;
            if (!take_string("--config", setting)) {
                return EXIT_FAILURE;
            }
            if (setting == "tracker=klt") {
                initial_tracker = 0;
            }
            else if (setting == "tracker=extrema") {
                initial_tracker = 1;
            }
            else if (setting == "lines=on") {
                initial_lines = true;
            }
            else if (setting == "lines=off") {
                initial_lines = false;
            }
            else if (setting == "culling=on") {
                initial_culling = true;
            }
            else if (setting == "culling=off") {
                initial_culling = false;
            }
            else if (setting == "association=klt") {
                initial_association = 0;
            }
            else if (setting == "association=match") {
                initial_association = 1;
            }
            else if (setting == "association=both") {
                initial_association = 2;
            }
            else {
                std::fprintf(stderr, "Unknown --config setting: %s (tracker=klt|extrema, lines=on|off, culling=on|off, association=klt|match|both)\n", setting.c_str());
                return EXIT_FAILURE;
            }
        }
        else if (std::strcmp(argv[i], "--exit-after") == 0) {
            if (!take_number("--exit-after", exit_after_frames)) {
                return EXIT_FAILURE;
            }
        }
        else if (argv[i][0] == '-') {
            std::fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return EXIT_FAILURE;
        }
        else if (scene_path.empty()) {
            scene_path = argv[i];
        }
        else {
            std::fprintf(stderr, "Unexpected argument: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }
    if (!screenshot_path.empty() && (exit_after_frames < 0)) {
        exit_after_frames = screenshot_after_frames;
    }

    if (!gtl::window::valid_driver()) {
        std::fprintf(stderr, "No valid window driver found.\n");
        return EXIT_FAILURE;
    }
    gtl::window window;
    int window_width = 1400;
    int window_height = 900;
    if (!window.open({ 100, 100 }, { window_width, window_height }, "ZeroSLAM", true)) {
        std::fprintf(stderr, "Failed to open window.\n");
        return EXIT_FAILURE;
    }
    window.set_icon(64, 64, icon::make());
    window.set_visible(true);
    for (int i = 0; (i < 1000) && !window.is_visible(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    window.prepare();
    window.set_vsync(true);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_POINT_SMOOTH);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

    overlay panels;
    gtl::imgui editor;
    editor.set_renderer(&panels.renderer);
    const int control_panel_width = 300;
    const int graph_panel_width = 320;
    const int image_strip_height = 260;

    orbit_camera camera;
    int mouse_x = 0;
    int mouse_y = 0;
    bool rotating = false;
    bool panning = false;
    ui::input pointer;
    bool click_pending = false;
    gtl::rect map_viewport = { 0, 0, window_width, window_height };

    scene::scene loaded_scene;
    std::string scene_error;
    char scene_path_text[1024] = {};
    std::snprintf(&scene_path_text[0], sizeof(scene_path_text), "%s", scene_path.c_str());
    slam_worker worker;
    playback timeline;
    std::chrono::steady_clock::time_point last_repaint = std::chrono::steady_clock::now();

    char map_path_text[1024] = "map.zsmap";
    gui::render_snapshot loaded_map;
    bool showing_loaded_map = false;
    std::string map_message;

    int feature_tracker = 0;
    bool feature_lines = false;
    bool feature_culling = true;
    int feature_association = 2;

    bool show_image_strip = true;
    bool show_landmarks = true;
    bool show_lines = true;
    bool show_covisibility = false;
    bool show_loops = true;
    float covisibility_minimum = 15.0f;
    bool show_camera_poses = true;
    bool show_camera_trajectory = true;
    bool show_keyframes_only = false;
    bool show_ground_truth = true;
    bool align_to_truth = false;
    bool scale_to_truth = true;
    bool show_visible_landmarks = true;
    bool show_depth_cloud = true;
    bool show_grid = true;
    bool show_axes = true;
    bool show_graph_panel = true;
    bool show_metrics = true;
    float point_size = 3.0f;
    float line_size = 1.5f;
    float frustum_scale = 0.5f;
    float image_alpha = 0.6f;
    float graph_window_seconds = 20.0f;

    std::vector<scene::decoded_image> channel_images;
    std::vector<long long> channel_image_indices;
    std::vector<gui::image_texture> channel_textures;
    int displayed_channel = 0;

    metrics::result accuracy;
    anchor_pair anchor;
    int accuracy_computed_for = -1;
    std::vector<dataset::trajectory_pose> truth_camera;
    int truth_camera_for_channel = -2;
    graph::history ate_history(4096);
    graph::history rde_history(4096);
    graph::history process_time_history(4096);
    long long last_timed_frame = -1;
    std::vector<graph::point> imu_points[6];
    map_geometry geometry;
    depth_cloud cloud;

    const auto reset_evaluation = [&]() {
        accuracy = metrics::result();
        anchor = anchor_pair();
        accuracy_computed_for = -1;
        ate_history.clear();
        rde_history.clear();
        process_time_history.clear();
        last_timed_frame = -1;
        cloud.invalidate();
    };

    const auto load_scene = [&]() {
        worker.stop();
        worker.clear();
        timeline.state = playback::mode::stopped;
        timeline.time_nanoseconds = 0;
        channel_images.clear();
        channel_image_indices.clear();
        for (gui::image_texture& texture : channel_textures) {
            texture.release();
        }
        channel_textures.clear();
        displayed_channel = 0;
        reset_evaluation();
        truth_camera.clear();
        truth_camera_for_channel = -2;
        const bool ok = loaded_scene.load(&scene_path_text[0], scene_error);
        if (ok) {
            scene_error.clear();
            channel_images.resize(loaded_scene.image_channels().size());
            channel_image_indices.assign(loaded_scene.image_channels().size(), -1);
            channel_textures.resize(loaded_scene.image_channels().size());
            timeline.time_nanoseconds = loaded_scene.begin_time();
            timeline.state = playback::mode::paused;
            displayed_channel = (primary_visual_channel(loaded_scene) >= 0) ? primary_visual_channel(loaded_scene) : 0;
            std::printf("Loaded scene '%s': %zu image channels, %zu imu channels, %zu ground truth poses.\n", &scene_path_text[0], loaded_scene.image_channels().size(), loaded_scene.imu_channels().size(), loaded_scene.ground_truth().size());
        }
        else {
            std::fprintf(stderr, "Failed to load the scene '%s': %s.\n", &scene_path_text[0], scene_error.c_str());
        }
        worker.channel.store(primary_visual_channel(loaded_scene));
        worker.start(loaded_scene);
    };

    if (!load_map_path.empty()) {
        std::snprintf(&map_path_text[0], sizeof(map_path_text), "%s", load_map_path.c_str());
        std::string load_error;
        if (!map_file::load(load_map_path, loaded_map, load_error)) {
            std::fprintf(stderr, "Failed to load the map '%s': %s.\n", load_map_path.c_str(), load_error.c_str());
            return EXIT_FAILURE;
        }
        showing_loaded_map = true;
        map_message = "loaded " + std::to_string(loaded_map.frames.size()) + " poses";
        std::printf("Loaded map '%s': %zu poses, %zu landmarks, %zu lines, %zu edges.\n", load_map_path.c_str(), loaded_map.frames.size(), loaded_map.landmarks.size(), loaded_map.lines.size(), loaded_map.edges.size());
    }

    feature_tracker = initial_tracker;
    feature_lines = initial_lines;
    feature_culling = initial_culling;
    feature_association = initial_association;
    worker.tracker_choice.store(feature_tracker);
    worker.lines_enabled.store(feature_lines);
    worker.culling_enabled.store(feature_culling);
    worker.association_choice.store(feature_association);
    if (scene_path.empty()) {
        worker.start(loaded_scene);
    }
    else {
        load_scene();
        if (autoplay && loaded_scene.is_loaded()) {
            timeline.state = playback::mode::playing;
        }
    }

    long long rendered_frames = 0;
    while (!shutdown_requested && window.is_open()) {
        gtl::window::event_type event;
        while (window.process(event)) {
            switch (event.input) {
                case gtl::window::event_type::input_type::configuration:
                    window_width = event.data.configuration.width;
                    window_height = event.data.configuration.height;
                    break;
                case gtl::window::event_type::input_type::focus:
                    editor.update_focus(event.data.focus == gtl::window::event_type::focus_type::gained);
                    break;
                case gtl::window::event_type::input_type::key_press:
                    editor.update_key(event.data.key, true);
                    break;
                case gtl::window::event_type::input_type::key_release:
                    editor.update_key(event.data.key, false);
                    break;
                case gtl::window::event_type::input_type::mouse_press:
                case gtl::window::event_type::input_type::mouse_release: {
                    const bool press = (event.input == gtl::window::event_type::input_type::mouse_press);
                    const bool left = (event.data.mouse == gtl::window::event_type::mouse_type::left);
                    const bool right = (event.data.mouse == gtl::window::event_type::mouse_type::right);
                    if (!left && !right && (event.data.mouse != gtl::window::event_type::mouse_type::middle)) {
                        break;
                    }
                    editor.update_mouse_button(left ? 0 : (right ? 2 : 1), press);
                    if (press && map_viewport.contains(mouse_x, mouse_y)) {
                        rotating = rotating || left;
                        panning = panning || right;
                    }
                    else if (press && left) {
                        click_pending = true;
                        pointer.mouse_down = true;
                        pointer.press_x = mouse_x;
                        pointer.press_y = mouse_y;
                    }
                    if (!press && left) {
                        rotating = false;
                        pointer.mouse_down = false;
                    }
                    if (!press && right) {
                        panning = false;
                    }
                    break;
                }
                case gtl::window::event_type::input_type::mouse_move: {
                    const float delta_x = static_cast<float>(event.data.motion.x - mouse_x);
                    const float delta_y = static_cast<float>(event.data.motion.y - mouse_y);
                    mouse_x = event.data.motion.x;
                    mouse_y = event.data.motion.y;
                    editor.update_mouse_move(mouse_x, mouse_y);
                    if (rotating) {
                        camera.rotate(delta_x, delta_y, map_viewport.w, map_viewport.h);
                    }
                    if (panning) {
                        camera.pan(delta_x, delta_y, map_viewport.h);
                    }
                    break;
                }
                case gtl::window::event_type::input_type::mouse_scroll:
                    if (map_viewport.contains(mouse_x, mouse_y)) {
                        camera.zoom(event.data.scroll);
                    }
                    break;
            }
        }
        if (!window.is_open()) {
            break;
        }
        pointer.mouse_x = mouse_x;
        pointer.mouse_y = mouse_y;
        pointer.clicked = click_pending;
        click_pending = false;
        editor.prepare();

        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        const double elapsed_seconds = std::chrono::duration<double>(now - last_repaint).count();
        last_repaint = now;
        if ((timeline.state == playback::mode::playing) && loaded_scene.is_loaded()) {
            timeline.time_nanoseconds += static_cast<long long>(elapsed_seconds * static_cast<double>(timeline.speed) * 1.0e9);
            const long long pending = worker.pending_time_nanoseconds.load();
            if ((pending >= 0) && (timeline.time_nanoseconds > pending)) {
                timeline.time_nanoseconds = pending;
            }
            if (timeline.time_nanoseconds >= loaded_scene.end_time()) {
                timeline.time_nanoseconds = loaded_scene.end_time();
                timeline.state = playback::mode::paused;
            }
        }
        worker.playback_time_nanoseconds.store(timeline.time_nanoseconds);

        const gui::render_snapshot& live = worker.read();
        const gui::render_snapshot& snapshot = showing_loaded_map ? loaded_map : live;
        const int primary = worker.channel.load();

        const auto seek = [&](const long long time_nanoseconds) {
            timeline.time_nanoseconds = time_nanoseconds;
            timeline.state = playback::mode::paused;
            if (time_nanoseconds < live.last_frame_timestamp_nanoseconds) {
                worker.reset.store(true);
                reset_evaluation();
            }
        };

        const long long processed_time = snapshot.last_frame_timestamp_nanoseconds;
        const long long display_time_nanoseconds = ((processed_time >= 0) && (timeline.time_nanoseconds > processed_time)) ? processed_time : timeline.time_nanoseconds;
        const std::vector<scene::image_channel>& channels = loaded_scene.image_channels();
        for (std::size_t i = 0; i < channels.size(); ++i) {
            const long long index = scene::scene::message_index_at(channels[i], display_time_nanoseconds);
            if (index == channel_image_indices[i]) {
                continue;
            }
            channel_image_indices[i] = index;
            if ((index < 0) || !loaded_scene.decode_image(channels[i], static_cast<std::size_t>(index), channel_images[i])) {
                channel_images[i].clear();
            }
        }

        std::vector<int> strip_channels;
        for (std::size_t i = 0; i < channels.size(); ++i) {
            if (channels[i].selected_for_slam) {
                strip_channels.push_back(static_cast<int>(i));
            }
        }
        if (!strip_channels.empty() && (std::find(strip_channels.begin(), strip_channels.end(), displayed_channel) == strip_channels.end())) {
            displayed_channel = (std::find(strip_channels.begin(), strip_channels.end(), primary) != strip_channels.end()) ? primary : strip_channels.front();
        }

        if (loaded_scene.is_loaded() && (primary != truth_camera_for_channel)) {
            truth_camera_for_channel = primary;
            truth_camera = camera_ground_truth(loaded_scene, primary);
            accuracy_computed_for = -1;
        }
        if (loaded_scene.is_loaded() && (snapshot.processed_frame_count != accuracy_computed_for)) {
            accuracy_computed_for = snapshot.processed_frame_count;
            if (!truth_camera.empty() && (snapshot.frames.size() >= 3)) {
                const estimated_trajectory estimated(snapshot);
                accuracy = metrics::evaluate(truth_camera, estimated.poses, true);
                anchor = estimated.anchor(accuracy, truth_camera);
                if (accuracy.valid) {
                    const double when = static_cast<double>(estimated.poses[accuracy.pairs.back().second].timestamp_nanoseconds - loaded_scene.begin_time()) * 1.0e-9;
                    ate_history.push_unique(when, static_cast<float>(accuracy.errors.back()));
                    rde_history.push_unique(when, static_cast<float>(accuracy.relative[0].latest));
                }
            }
        }
        if (loaded_scene.is_loaded() && (snapshot.last_frame_timestamp_nanoseconds >= 0) && (snapshot.last_frame_timestamp_nanoseconds != last_timed_frame)) {
            last_timed_frame = snapshot.last_frame_timestamp_nanoseconds;
            const double when = static_cast<double>(snapshot.last_frame_timestamp_nanoseconds - loaded_scene.begin_time()) * 1.0e-9;
            process_time_history.push_unique(when, static_cast<float>(snapshot.last_process_seconds * 1000.0));
        }
        const metrics::similarity placement = map_placement(truth_camera, accuracy, anchor, align_to_truth, scale_to_truth);

        geometry.update(snapshot, static_cast<int>(covisibility_minimum));
        if (show_depth_cloud && loaded_scene.is_loaded()) {
            cloud.update(loaded_scene, channel_images, snapshot, primary, (placement.scale > 0.0) ? (1.0 / placement.scale) : 1.0);
        }

        const int side_width = show_graph_panel ? graph_panel_width : 0;
        map_viewport = { control_panel_width, 0, std::max(1, window_width - control_panel_width - side_width), std::max(1, window_height) };

        glViewport(map_viewport.x, map_viewport.y, map_viewport.w, map_viewport.h);
        glScissor(map_viewport.x, map_viewport.y, map_viewport.w, map_viewport.h);
        glEnable(GL_SCISSOR_TEST);
        glClearColor(0.12f, 0.12f, 0.15f, 1.0f);
        glEnable(GL_DEPTH_TEST);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glDisable(GL_SCISSOR_TEST);
        {
            float projection[16];
            gl_load_perspective(projection, orbit_camera::field_of_view_degrees, static_cast<float>(map_viewport.w) / static_cast<float>(map_viewport.h), 0.01f, 10000.0f);
            float view[16];
            camera.load_view(view);
            glMatrixMode(GL_PROJECTION);
            glLoadMatrixf(projection);
            glMatrixMode(GL_MODELVIEW);
            glLoadMatrixf(view);

            glLineWidth(1.0f);
            if (show_grid) {
                gl_draw_grid(50.0f, 1.0f);
            }
            if (show_axes) {
                glLineWidth(2.0f);
                gl_draw_axes(1.0f);
            }
            glPointSize(point_size);
            glLineWidth(line_size);

            glPushMatrix();
            {
                const double z_up_to_y_up[3][3] = { { 1.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0 }, { 0.0, -1.0, 0.0 } };
                const double half_turn_about_z[3][3] = { { -1.0, 0.0, 0.0 }, { 0.0, -1.0, 0.0 }, { 0.0, 0.0, 1.0 } };
                const double origin[3] = { 0.0, 0.0, 0.0 };
                gl_multiply_pose(placement.valid ? z_up_to_y_up : half_turn_about_z, origin);
            }

            glPushMatrix();
            if (placement.valid) {
                gl_multiply_similarity(placement);
            }
            if (show_landmarks) {
                gl_draw_arrays(GL_POINTS, geometry.landmark_vertices, geometry.landmark_colours);
            }
            if (show_lines) {
                glLineWidth(line_size * 1.5f);
                gl_draw_arrays(GL_LINES, geometry.line_vertices, geometry.line_colours);
                glLineWidth(line_size);
            }
            if (show_covisibility) {
                gl_draw_arrays(GL_LINES, geometry.covisibility_vertices, geometry.covisibility_colours);
            }
            if (show_loops) {
                glLineWidth(line_size * 3.0f);
                gl_draw_arrays(GL_LINES, geometry.loop_vertices, geometry.loop_colours);
                glLineWidth(line_size);
            }
            if (show_depth_cloud) {
                gl_draw_arrays(GL_POINTS, cloud.vertices, cloud.colours);
            }

            const std::map<int, const gui::render_snapshot::frame*> ordered_frames = [&]() {
                std::map<int, const gui::render_snapshot::frame*> ordered;
                for (const auto& [id, pose] : snapshot.frames) {
                    if (show_keyframes_only && (snapshot.keyframes.count(id) == 0)) {
                        continue;
                    }
                    ordered[id] = &pose;
                }
                return ordered;
            }();
            const int current_frame = snapshot.processed_frame_count - 1;
            if (show_camera_poses) {
                int index = 0;
                for (const auto& [id, pose] : ordered_frames) {
                    const float age = (ordered_frames.size() > 1) ? static_cast<float>(index) / static_cast<float>(ordered_frames.size() - 1) : 0.0f;
                    glPushMatrix();
                    gl_multiply_camera_to_world(*pose);
                    gl_draw_frustum(*pose, frustum_scale, age, 1.0f - (age * 0.7f), 0.2f);
                    if (id == current_frame) {
                        glLineWidth(2.0f);
                        gl_draw_axes(frustum_scale * 2.0f);
                        glLineWidth(line_size);
                    }
                    glPopMatrix();
                    ++index;
                }
            }
            if (show_camera_trajectory && (ordered_frames.size() > 1)) {
                glColor3f(0.0f, 1.0f, 1.0f);
                glLineWidth(line_size * 2.0f);
                glBegin(GL_LINE_STRIP);
                for (const auto& [id, pose] : ordered_frames) {
                    static_cast<void>(id);
                    double centre[3];
                    gui::render_snapshot::camera_centre(*pose, centre);
                    glVertex3d(centre[0], centre[1], centre[2]);
                }
                glEnd();
                glLineWidth(line_size);
            }
            const auto current = snapshot.frames.find(current_frame);
            if (current != snapshot.frames.end()) {
                glPushMatrix();
                gl_multiply_camera_to_world(current->second);
                glLineWidth(line_size * 3.0f);
                gl_draw_frustum(current->second, frustum_scale * 2.0f, 1.0f, 1.0f, 0.0f);
                glLineWidth(line_size);
                glPopMatrix();
            }
            glPopMatrix();

            if (show_ground_truth && (truth_camera.size() > 1)) {
                long long estimate_end = -1;
                for (const auto& [id, pose] : snapshot.frames) {
                    static_cast<void>(id);
                    estimate_end = std::max(estimate_end, pose.timestamp_nanoseconds);
                }
                glColor3f(1.0f, 0.35f, 0.95f);
                glLineWidth(line_size * 2.5f);
                glBegin(GL_LINE_STRIP);
                for (const dataset::trajectory_pose& pose : truth_camera) {
                    if (pose.timestamp_nanoseconds > estimate_end) {
                        break;
                    }
                    glVertex3d(pose.x_coordinate, pose.y_coordinate, pose.z_coordinate);
                }
                glEnd();
                glLineWidth(line_size);
            }
            glPopMatrix();
        }

        if (show_image_strip && !strip_channels.empty()) {
            const int strip_y = window_height - image_strip_height;
            glViewport(map_viewport.x, strip_y, map_viewport.w, image_strip_height);
            glDisable(GL_DEPTH_TEST);
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glOrtho(0, map_viewport.w, 0, image_strip_height, -1, 1);
            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();

            const int panel_width = map_viewport.w / static_cast<int>(strip_channels.size());
            for (std::size_t slot = 0; slot < strip_channels.size(); ++slot) {
                const int column = strip_channels[slot];
                const scene::decoded_image& image = channel_images[static_cast<std::size_t>(column)];
                if (!image.is_valid()) {
                    continue;
                }
                gui::image_texture& texture = channel_textures[static_cast<std::size_t>(column)];
                texture.update(image);
                const gui::image_texture::placement placed = texture.draw(static_cast<int>(slot) * panel_width, 0, panel_width, image_strip_height, image_alpha);
                if (!placed.valid) {
                    continue;
                }
                if (column == primary) {
                    glColor3f(0.4f, 0.75f, 1.0f);
                }
                else {
                    glColor3f(0.45f, 0.5f, 0.6f);
                }
                glLineWidth(2.0f);
                glBegin(GL_LINE_LOOP);
                glVertex2i(placed.x, placed.y);
                glVertex2i(placed.x + placed.width, placed.y);
                glVertex2i(placed.x + placed.width, placed.y + placed.height);
                glVertex2i(placed.x, placed.y + placed.height);
                glEnd();
                glLineWidth(line_size);
                if (column != displayed_channel) {
                    continue;
                }

                int display_frame = -1;
                if (nearest_frame(snapshot, image.timestamp_nanoseconds, display_frame) == nullptr) {
                    continue;
                }
                const auto to_panel = [&](const float x, const float y) {
                    glVertex2f(static_cast<float>(placed.x) + (x * placed.scale), static_cast<float>(placed.y + placed.height) - (y * placed.scale));
                };
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                if (show_visible_landmarks) {
                    const gui::render_snapshot::frame& pose = snapshot.frames.at(display_frame);
                    const double fx = pose.camera_parameters[0];
                    const double fy = pose.camera_parameters[1];
                    const double cx = pose.camera_parameters[2];
                    const double cy = pose.camera_parameters[3];
                    glPointSize(point_size + 1.0f);
                    glBegin(GL_POINTS);
                    for (const auto& [landmark_id, landmark] : snapshot.landmarks) {
                        static_cast<void>(landmark_id);
                        double in_camera[3];
                        for (int row = 0; row < 3; ++row) {
                            in_camera[row] = (pose.rotation[row][0] * landmark.location[0]) + (pose.rotation[row][1] * landmark.location[1]) + (pose.rotation[row][2] * landmark.location[2]) + pose.translation[row];
                        }
                        if (in_camera[2] <= 1.0e-6) {
                            continue;
                        }
                        const double u = (fx * (in_camera[0] / in_camera[2])) + cx;
                        const double v = (fy * (in_camera[1] / in_camera[2])) + cy;
                        if ((u < 0.0) || (v < 0.0) || (u >= static_cast<double>(image.width)) || (v >= static_cast<double>(image.height))) {
                            continue;
                        }
                        glColor3f((static_cast<float>(landmark.colour[0]) * 0.5f) + 0.5f, (static_cast<float>(landmark.colour[1]) * 0.5f) + 0.5f, (static_cast<float>(landmark.colour[2]) * 0.5f) + 0.5f);
                        to_panel(static_cast<float>(u), static_cast<float>(v));
                    }
                    glEnd();
                    glPointSize(point_size);
                }
                glDisable(GL_BLEND);
            }
        }

        panels.begin(window_width, window_height);
        {
            ui::panel controls(&panels.renderer, pointer, 0, 0, control_panel_width);
            controls.fill(window_height);
            controls.space(8);
            controls.label("ZEROSLAM VIEWER");

            controls.header("Scene");
            controls.text_input(editor, "mcap:", &scene_path_text[0], static_cast<int>(sizeof(scene_path_text)));
            if (controls.button("Load Scene", ui::panel::button_kind::positive)) {
                load_scene();
            }
            if (controls.button("Reset SLAM", ui::panel::button_kind::negative, loaded_scene.is_loaded())) {
                timeline.time_nanoseconds = loaded_scene.begin_time();
                timeline.state = playback::mode::paused;
                worker.reset.store(true);
                reset_evaluation();
            }
            if (!scene_error.empty()) {
                controls.label_dim(scene_error.c_str());
            }

            controls.header("Features");
            {
                const char* const trackers[2] = { "Points", "Curvature" };
                const int pressed = controls.button_row(&trackers[0], 2, feature_tracker);
                if ((pressed >= 0) && (pressed != feature_tracker)) {
                    feature_tracker = pressed;
                    worker.tracker_choice.store(feature_tracker);
                    timeline.time_nanoseconds = loaded_scene.begin_time();
                    timeline.state = playback::mode::paused;
                    worker.reset.store(true);
                    reset_evaluation();
                }
                if (controls.checkbox("Lines", &feature_lines)) {
                    worker.lines_enabled.store(feature_lines);
                }
                if (controls.checkbox("Cull Keyframes", &feature_culling)) {
                    worker.culling_enabled.store(feature_culling);
                }
                const char* const associations[3] = { "KLT", "Match", "Both" };
                const int association_pressed = controls.button_row(&associations[0], 3, feature_association, feature_tracker == 0);
                if ((association_pressed >= 0) && (association_pressed != feature_association)) {
                    feature_association = association_pressed;
                    worker.association_choice.store(feature_association);
                }
            }
            controls.text_input(editor, "map:", &map_path_text[0], static_cast<int>(sizeof(map_path_text)));
            {
                const char* const map_buttons[2] = { "Save Map", "Load Map" };
                const int pressed = controls.button_row(&map_buttons[0], 2);
                if (pressed == 0) {
                    std::string save_error;
                    map_message = map_file::save(&map_path_text[0], snapshot, save_error) ? ("saved " + std::to_string(snapshot.frames.size()) + " poses") : save_error;
                }
                else if (pressed == 1) {
                    std::string load_error;
                    if (map_file::load(&map_path_text[0], loaded_map, load_error)) {
                        geometry.built_from = nullptr;
                        cloud.invalidate();
                        showing_loaded_map = true;
                        map_message = "loaded " + std::to_string(loaded_map.frames.size()) + " poses";
                        accuracy_computed_for = -1;
                    }
                    else {
                        map_message = load_error;
                    }
                }
            }
            if (showing_loaded_map && controls.button("Back to Live Map")) {
                showing_loaded_map = false;
                map_message.clear();
                accuracy_computed_for = -1;
            }
            if (!map_message.empty()) {
                controls.label_dim(map_message.c_str());
            }
            if (loaded_scene.is_loaded()) {
                controls.value_number("Duration", static_cast<double>(loaded_scene.duration_nanoseconds()) * 1.0e-9, "%.2f s");
                controls.value_number("Ground truth", static_cast<double>(loaded_scene.ground_truth().size()), "%.0f poses");
            }

            controls.header("Sensors");
            if (!loaded_scene.is_loaded()) {
                controls.label_dim("no scene loaded");
            }
            for (std::size_t i = 0; i < loaded_scene.image_channels().size(); ++i) {
                scene::image_channel& channel = loaded_scene.image_channels()[i];
                char text[128] = {};
                std::snprintf(&text[0], sizeof(text), "%s %s %ux%u", channel.sensor_name.c_str(), channel.is_depth() ? "(depth)" : "(visual)", channel.width, channel.height);
                if (controls.checkbox(&text[0], &channel.selected_for_slam)) {
                    worker.channel.store(primary_visual_channel(loaded_scene));
                    if (channel.selected_for_slam) {
                        displayed_channel = static_cast<int>(i);
                    }
                }
                if (!channel.has_intrinsics) {
                    controls.label_dim("  no camera_info, cannot slam");
                }
                else if (static_cast<int>(i) == primary) {
                    controls.label_dim("  fed to the slam system");
                }
            }
            for (scene::imu_channel& channel : loaded_scene.imu_channels()) {
                char text[128] = {};
                std::snprintf(&text[0], sizeof(text), "%s (imu) %zu", channel.sensor_name.c_str(), channel.samples.size());
                controls.checkbox(&text[0], &channel.selected_for_slam);
            }

            controls.header("Playback");
            {
                const bool playing = (timeline.state == playback::mode::playing);
                const char* const transport[4] = { "|<", "[]", playing ? "||" : ">", ">|" };
                const int pressed = controls.button_row(&transport[0], 4, playing ? 2 : -1, loaded_scene.is_loaded());
                const auto step_by = [&](const long long direction) {
                    if ((displayed_channel < 0) || (displayed_channel >= static_cast<int>(channels.size())) || channels[static_cast<std::size_t>(displayed_channel)].messages.empty()) {
                        return;
                    }
                    const scene::image_channel& channel = channels[static_cast<std::size_t>(displayed_channel)];
                    const long long last = static_cast<long long>(channel.messages.size()) - 1;
                    const long long index = std::min(last, std::max(0LL, scene::scene::message_index_at(channel, timeline.time_nanoseconds) + direction));
                    seek(channel.log_times[static_cast<std::size_t>(index)]);
                };
                if (pressed == 0) {
                    step_by(-1);
                }
                else if (pressed == 1) {
                    timeline.state = playback::mode::stopped;
                    timeline.time_nanoseconds = loaded_scene.begin_time();
                    worker.reset.store(true);
                    reset_evaluation();
                }
                else if (pressed == 2) {
                    timeline.state = playing ? playback::mode::paused : playback::mode::playing;
                }
                else if (pressed == 3) {
                    step_by(1);
                }
                if (loaded_scene.is_loaded()) {
                    double fraction = timeline.fraction_of(loaded_scene);
                    char readout[96] = {};
                    std::snprintf(&readout[0], sizeof(readout), "t %+.3f s of %.3f s  |  fed %d", static_cast<double>(timeline.time_nanoseconds - loaded_scene.begin_time()) * 1.0e-9, static_cast<double>(loaded_scene.duration_nanoseconds()) * 1.0e-9, snapshot.processed_frame_count);
                    if (controls.scrubber(&fraction, &readout[0])) {
                        playback target = timeline;
                        target.seek_to_fraction(loaded_scene, fraction);
                        seek(target.time_nanoseconds);
                    }
                }
                controls.slider("Speed:", &timeline.speed, 0.1f, 8.0f, "%.1fx");
            }

            controls.header("Display");
            if (strip_channels.size() > 1) {
                char text[128] = {};
                std::snprintf(&text[0], sizeof(text), "Features on: %s", channels[static_cast<std::size_t>(displayed_channel)].sensor_name.c_str());
                if (controls.button(&text[0])) {
                    const auto shown = std::find(strip_channels.begin(), strip_channels.end(), displayed_channel);
                    displayed_channel = strip_channels[static_cast<std::size_t>((shown - strip_channels.begin()) + 1) % strip_channels.size()];
                }
            }
            controls.checkbox("Image Panel", &show_image_strip);
            controls.checkbox("Landmarks", &show_landmarks);
            controls.checkbox("Line Landmarks", &show_lines);
            controls.checkbox("Covisibility", &show_covisibility);
            controls.checkbox("Loop Edges", &show_loops);
            if (show_covisibility) {
                controls.slider("Covis Min:", &covisibility_minimum, 1.0f, 100.0f, "%.0f");
            }
            controls.checkbox("Camera Poses", &show_camera_poses);
            controls.checkbox("SLAM Trajectory", &show_camera_trajectory);
            controls.checkbox("Keyframes Only", &show_keyframes_only);
            controls.checkbox("Ground Truth", &show_ground_truth);
            controls.checkbox("Align To Truth", &align_to_truth);
            controls.checkbox("Scale To Truth", &scale_to_truth);
            controls.checkbox("Landmarks In View", &show_visible_landmarks);
            controls.checkbox("Depth Cloud", &show_depth_cloud);
            controls.checkbox("Grid", &show_grid);
            controls.checkbox("Axes", &show_axes);
            controls.space(4);
            controls.slider("Pt Size:", &point_size, 1.0f, 11.0f, "%.1f");
            controls.slider("Ln Size:", &line_size, 0.5f, 5.5f, "%.1f");
            controls.slider("Frustum:", &frustum_scale, 0.1f, 5.1f, "%.1f");
            controls.slider("Img Fade:", &image_alpha, 0.0f, 1.0f, "%.2f");
            if (show_depth_cloud) {
                controls.slider("Dpt Step:", &cloud.stride, 1.0f, 8.0f, "%.0f px");
                controls.slider("Dpt Max:", &cloud.maximum_metres, 0.5f, 50.0f, "%.1f m");
                for (scene::image_channel& channel : loaded_scene.image_channels()) {
                    if (!channel.is_depth() || (channel.encoding == "32FC1")) {
                        continue;
                    }
                    float units_per_metre = (channel.depth_scale > 0.0) ? static_cast<float>(1.0 / channel.depth_scale) : 1000.0f;
                    if (controls.slider("Dpt Unit:", &units_per_metre, 1.0f, 2000.0f, "%.0f /m")) {
                        channel.depth_scale = (units_per_metre > 0.0f) ? (1.0 / static_cast<double>(units_per_metre)) : 1.0;
                        channel_image_indices.assign(channel_image_indices.size(), -1);
                        for (gui::image_texture& texture : channel_textures) {
                            texture.release();
                        }
                        cloud.invalidate();
                    }
                }
                controls.value_number("Cloud points", static_cast<double>(cloud.vertices.size() / 3), "%.0f");
                if (!cloud.vertices.empty()) {
                    char text[128] = {};
                    if ((cloud.colour_channel >= 0) && (cloud.colour_channel < static_cast<int>(channels.size()))) {
                        std::snprintf(&text[0], sizeof(text), "Cloud colour: %s (%s)", channels[static_cast<std::size_t>(cloud.colour_channel)].sensor_name.c_str(), cloud.colour_registered ? "registered" : "projected");
                    }
                    else {
                        std::snprintf(&text[0], sizeof(text), "Cloud colour: depth ramp");
                    }
                    controls.label_dim(&text[0]);
                }
            }

            controls.header("Map");
            controls.value_number("Poses", static_cast<double>(snapshot.frames.size()), "%.0f");
            controls.value_number("Landmarks", static_cast<double>(snapshot.landmarks.size()), "%.0f");
            controls.value_number("Frame cost", snapshot.last_process_seconds * 1000.0, "%.1f ms");
            controls.space(4);
            controls.label_dim("cyan: slam, magenta: truth (camera)");
            controls.checkbox("Graph Panel", &show_graph_panel);
            controls.checkbox("Metrics", &show_metrics);
            controls.space(4);
            controls.label_dim("L/R drag: rotate/pan, scroll: zoom");
        }

        if (show_graph_panel) {
            ui::panel side(&panels.renderer, pointer, window_width - graph_panel_width, 0, graph_panel_width);
            side.fill(window_height);
            side.space(8);
            if (show_metrics) {
                side.header("Accuracy vs ground truth");
                if (!loaded_scene.is_loaded()) {
                    side.label_dim("no scene loaded");
                }
                else if (loaded_scene.ground_truth().empty()) {
                    side.label_dim("scene carries no ground truth");
                }
                else if (!accuracy.valid) {
                    side.label_dim("waiting for 3 aligned poses");
                }
                else {
                    side.value_number("ATE rmse", accuracy.ate_rmse, "%.4f m");
                    side.value_number("ATE mean", accuracy.ate_mean, "%.4f m");
                    side.value_number("ATE median", accuracy.ate_median, "%.4f m");
                    side.value_number("ATE max", accuracy.ate_maximum, "%.4f m");
                    side.value_number("RDE rmse (1)", accuracy.relative[0].rmse, "%.4f m");
                    side.value_number("RDE rmse (10)", accuracy.relative[1].rmse, "%.4f m");
                    side.value_number("Segment scale", accuracy.relative[1].scale_median, "%.3f");
                    side.value_number("Aligned pairs", static_cast<double>(accuracy.pairs.size()), "%.0f");
                    if (accuracy.alignment_degenerate) {
                        side.label_dim("straight path, roll not observable");
                    }
                }
            }

            side.header("Graphs");
            side.slider("Window:", &graph_window_seconds, 2.0f, 120.0f, "%.0f s");
            const int graph_x = side.content_x();
            const int graph_width = side.content_width();
            const int graph_height = 96;
            const double display_seconds = loaded_scene.is_loaded() ? (static_cast<double>(display_time_nanoseconds - loaded_scene.begin_time()) * 1.0e-9) : 0.0;
            const auto trailing_window = [&](const graph::series* const list, const std::size_t count, const char* const units) {
                graph::axes bounds;
                bounds.units = units;
                bounds.non_negative = true;
                bounds.time_maximum = graph::latest_time(list, count, display_seconds);
                bounds.time_minimum = bounds.time_maximum - static_cast<double>(graph_window_seconds);
                return bounds;
            };
            {
                const graph::series list[2] = { rde_history.as_series("rde(1)", gtl::color::rgb(255, 140, 90)), ate_history.as_series("ate", gtl::color::rgb(100, 170, 255)) };
                side.space(graph::draw(&panels.renderer, graph_x, side.cursor(), graph_width, graph_height, "Trajectory error", &list[0], 2, trailing_window(&list[0], 2, "m")) + 6);
            }
            {
                const graph::series list[1] = { process_time_history.as_series("frame", gtl::color::rgb(160, 230, 130)) };
                side.space(graph::draw(&panels.renderer, graph_x, side.cursor(), graph_width, graph_height, "Process time", &list[0], 1, trailing_window(&list[0], 1, "ms")) + 6);
            }

            graph::axes imu_window;
            imu_window.time_maximum = display_seconds;
            imu_window.time_minimum = display_seconds - static_cast<double>(graph_window_seconds);
            imu_window.mark_time = true;
            imu_window.marked_time_seconds = display_seconds;
            bool any_imu_drawn = false;
            for (const scene::imu_channel& imu : loaded_scene.imu_channels()) {
                if (!imu.selected_for_slam) {
                    continue;
                }
                any_imu_drawn = true;
                for (std::vector<graph::point>& axis : imu_points) {
                    axis.clear();
                }
                const long long last = scene::scene::sample_index_at(imu, display_time_nanoseconds);
                if (last >= 0) {
                    const long long window_begin = display_time_nanoseconds - static_cast<long long>(static_cast<double>(graph_window_seconds) * 1.0e9);
                    const long long first = std::min(last, std::max(0LL, scene::scene::sample_index_at(imu, window_begin) + 1));
                    const long long stride = std::max(1LL, (last - first) / static_cast<long long>(std::max(8, graph_width)));
                    for (long long i = first; i <= last; i += stride) {
                        const scene::imu_sample& sample = imu.samples[static_cast<std::size_t>(i)];
                        const double when = static_cast<double>(sample.timestamp_nanoseconds - loaded_scene.begin_time()) * 1.0e-9;
                        for (int axis = 0; axis < 3; ++axis) {
                            imu_points[axis].push_back({ when, static_cast<float>(sample.angular_velocity[axis]) });
                            imu_points[axis + 3].push_back({ when, static_cast<float>(sample.linear_acceleration[axis]) });
                        }
                    }
                }
                const gtl::color axis_colours[3] = { gtl::color::rgb(255, 110, 110), gtl::color::rgb(130, 230, 130), gtl::color::rgb(130, 170, 255) };
                const char* const axis_names[3] = { "x", "y", "z" };
                for (int sensor = 0; sensor < 2; ++sensor) {
                    graph::series list[3];
                    for (int axis = 0; axis < 3; ++axis) {
                        list[axis] = { axis_names[axis], axis_colours[axis], imu_points[(sensor * 3) + axis].data(), imu_points[(sensor * 3) + axis].size() };
                    }
                    graph::axes bounds = imu_window;
                    bounds.units = (sensor == 0) ? "rad/s" : "m/s2";
                    const std::string title = imu.sensor_name + ((sensor == 0) ? " gyroscope" : " accelerometer");
                    side.space(graph::draw(&panels.renderer, graph_x, side.cursor(), graph_width, graph_height, title.c_str(), &list[0], 3, bounds) + 6);
                }
            }
            if (!any_imu_drawn) {
                side.label_dim(loaded_scene.imu_channels().empty() ? "scene carries no inertial channel" : "no inertial channel selected");
            }
        }
        panels.flush();
        editor.present();

        ++rendered_frames;
        if (!screenshot_path.empty() && (rendered_frames == screenshot_after_frames)) {
            if (write_screenshot(screenshot_path, window_width, window_height)) {
                std::printf("Wrote screenshot: %s (%dx%d)\n", screenshot_path.c_str(), window_width, window_height);
            }
            else {
                std::fprintf(stderr, "Failed to write screenshot: %s\n", screenshot_path.c_str());
            }
        }
        window.present();
        if ((exit_after_frames >= 0) && (rendered_frames >= exit_after_frames)) {
            break;
        }
    }

    worker.stop();
    if (!save_map_path.empty()) {
        const gui::render_snapshot& to_save = showing_loaded_map ? loaded_map : worker.read();
        std::string save_error;
        if (map_file::save(save_map_path, to_save, save_error)) {
            std::printf("Saved map '%s': %zu poses, %zu landmarks.\n", save_map_path.c_str(), to_save.frames.size(), to_save.landmarks.size());
        }
        else {
            std::fprintf(stderr, "Failed to save the map '%s': %s.\n", save_map_path.c_str(), save_error.c_str());
        }
    }

    for (gui::image_texture& texture : channel_textures) {
        texture.release();
    }
    panels.release();
    window.close();
    return EXIT_SUCCESS;
}
