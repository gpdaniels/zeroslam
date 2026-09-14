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

#include "feature/detector/elsed.hpp"

#include "math/math.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace feature::detector {
    namespace {
        class line_fit final {
        public:
            double sum_x = 0.0;
            double sum_y = 0.0;
            double sum_xx = 0.0;
            double sum_yy = 0.0;
            double sum_xy = 0.0;
            int count = 0;

        public:
            void clear() {
                this->sum_x = 0.0;
                this->sum_y = 0.0;
                this->sum_xx = 0.0;
                this->sum_yy = 0.0;
                this->sum_xy = 0.0;
                this->count = 0;
            }

            void add(const double x, const double y) {
                this->sum_x += x;
                this->sum_y += y;
                this->sum_xx += x * x;
                this->sum_yy += y * y;
                this->sum_xy += x * y;
                ++this->count;
            }

            void remove(const double x, const double y) {
                this->sum_x -= x;
                this->sum_y -= y;
                this->sum_xx -= x * x;
                this->sum_yy -= y * y;
                this->sum_xy -= x * y;
                --this->count;
            }

            double rms_error() const {
                if (this->count < 2) {
                    return 0.0;
                }
                const double n = static_cast<double>(this->count);
                const double centroid_x = this->sum_x / n;
                const double centroid_y = this->sum_y / n;
                const double covariance_xx = (this->sum_xx / n) - (centroid_x * centroid_x);
                const double covariance_yy = (this->sum_yy / n) - (centroid_y * centroid_y);
                const double covariance_xy = (this->sum_xy / n) - (centroid_x * centroid_y);
                const double difference = covariance_xx - covariance_yy;
                const double discriminant = math::sqrt((difference * difference) + (4.0 * covariance_xy * covariance_xy));
                const double eigenvalue = 0.5 * (covariance_xx + covariance_yy - discriminant);
                return math::sqrt(math::max(eigenvalue, 0.0));
            }

            bool direction(double& centroid_x, double& centroid_y, double& direction_x, double& direction_y) const {
                if (this->count < 2) {
                    return false;
                }
                const double n = static_cast<double>(this->count);
                centroid_x = this->sum_x / n;
                centroid_y = this->sum_y / n;
                const double covariance_xx = (this->sum_xx / n) - (centroid_x * centroid_x);
                const double covariance_yy = (this->sum_yy / n) - (centroid_y * centroid_y);
                const double covariance_xy = (this->sum_xy / n) - (centroid_x * centroid_y);
                const double difference = covariance_xx - covariance_yy;
                const double discriminant = math::sqrt((difference * difference) + (4.0 * covariance_xy * covariance_xy));
                const double eigenvalue = 0.5 * (covariance_xx + covariance_yy + discriminant);
                double vx = covariance_xy;
                double vy = eigenvalue - covariance_xx;
                const double alternative_x = eigenvalue - covariance_yy;
                const double alternative_y = covariance_xy;
                if (((alternative_x * alternative_x) + (alternative_y * alternative_y)) > ((vx * vx) + (vy * vy))) {
                    vx = alternative_x;
                    vy = alternative_y;
                }
                const double length = math::sqrt((vx * vx) + (vy * vy));
                if (length < 1.0e-12) {
                    return false;
                }
                direction_x = vx / length;
                direction_y = vy / length;
                return true;
            }

            bool distance(const double x, const double y, double& result) const {
                double centroid_x = 0.0;
                double centroid_y = 0.0;
                double direction_x = 0.0;
                double direction_y = 0.0;
                if (!this->direction(centroid_x, centroid_y, direction_x, direction_y)) {
                    return false;
                }
                result = math::abs(((x - centroid_x) * direction_y) - ((y - centroid_y) * direction_x));
                return true;
            }
        };

        class anchor final {
        public:
            int x;
            int y;
            int magnitude;
        };

        class draw_state final {
        public:
            int x;
            int y;
            int last_dx;
            int last_dy;
        };
    }

    size_t elsed::detect(
        const unsigned char* __restrict const data,
        const int width,
        const int height,
        const int stride,
        const elsed::options& detection_options,
        elsed::segment* __restrict const segment_buffer,
        const size_t segment_buffer_size
    ) {
        if ((width < 8) || (height < 8) || (segment_buffer_size == 0)) {
            return 0;
        }
        const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);

        std::vector<unsigned char> smoothed(pixel_count);
        {
            std::vector<unsigned short> horizontal(pixel_count);
            const int kernel[5] = { 1, 4, 6, 4, 1 };
            for (int y = 0; y < height; ++y) {
                const unsigned char* const row = data + (static_cast<size_t>(y) * static_cast<size_t>(stride));
                unsigned short* const target = horizontal.data() + (static_cast<size_t>(y) * static_cast<size_t>(width));
                for (int x = 0; x < width; ++x) {
                    int sum = 0;
                    for (int k = -2; k <= 2; ++k) {
                        const int sample = math::max(0, math::min(width - 1, x + k));
                        sum += kernel[k + 2] * static_cast<int>(row[sample]);
                    }
                    target[x] = static_cast<unsigned short>(sum);
                }
            }
            for (int y = 0; y < height; ++y) {
                unsigned char* const target = smoothed.data() + (static_cast<size_t>(y) * static_cast<size_t>(width));
                for (int x = 0; x < width; ++x) {
                    int sum = 0;
                    for (int k = -2; k <= 2; ++k) {
                        const int sample = math::max(0, math::min(height - 1, y + k));
                        sum += kernel[k + 2] * static_cast<int>(horizontal[(static_cast<size_t>(sample) * static_cast<size_t>(width)) + static_cast<size_t>(x)]);
                    }
                    target[x] = static_cast<unsigned char>((sum + 128) >> 8);
                }
            }
        }

        std::vector<int> magnitude(pixel_count, 0);
        std::vector<unsigned char> vertical(pixel_count, 0);
        const auto smoothed_at = [&smoothed, width](const int x, const int y) {
            return static_cast<int>(smoothed[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)]);
        };
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                const int gx = (smoothed_at(x + 1, y - 1) + (2 * smoothed_at(x + 1, y)) + smoothed_at(x + 1, y + 1)) - (smoothed_at(x - 1, y - 1) + (2 * smoothed_at(x - 1, y)) + smoothed_at(x - 1, y + 1));
                const int gy = (smoothed_at(x - 1, y + 1) + (2 * smoothed_at(x, y + 1)) + smoothed_at(x + 1, y + 1)) - (smoothed_at(x - 1, y - 1) + (2 * smoothed_at(x, y - 1)) + smoothed_at(x + 1, y - 1));
                const size_t index = (static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x);
                magnitude[index] = math::abs(gx) + math::abs(gy);
                vertical[index] = (math::abs(gx) >= math::abs(gy)) ? 1 : 0;
            }
        }
        const auto magnitude_at = [&magnitude, width](const int x, const int y) {
            return magnitude[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)];
        };
        const auto vertical_at = [&vertical, width](const int x, const int y) {
            return vertical[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)] != 0;
        };

        std::vector<anchor> anchors;
        const int interval = math::max(1, detection_options.scan_interval);
        for (int y = 1; y < height - 1; y += interval) {
            for (int x = 1; x < width - 1; x += interval) {
                const int central = magnitude_at(x, y);
                if (central < detection_options.gradient_threshold) {
                    continue;
                }
                bool is_anchor;
                if (vertical_at(x, y)) {
                    const int left = magnitude_at(x - 1, y);
                    const int right = magnitude_at(x + 1, y);
                    is_anchor = (central >= left) && (central >= right) && (((central - left) >= detection_options.anchor_threshold) || ((central - right) >= detection_options.anchor_threshold));
                }
                else {
                    const int above = magnitude_at(x, y - 1);
                    const int below = magnitude_at(x, y + 1);
                    is_anchor = (central >= above) && (central >= below) && (((central - above) >= detection_options.anchor_threshold) || ((central - below) >= detection_options.anchor_threshold));
                }
                if (is_anchor) {
                    anchors.push_back(anchor{ x, y, central });
                }
            }
        }
        std::sort(anchors.begin(), anchors.end(), [](const anchor& lhs, const anchor& rhs) {
            if (lhs.magnitude != rhs.magnitude) {
                return lhs.magnitude > rhs.magnitude;
            }
            if (lhs.y != rhs.y) {
                return lhs.y < rhs.y;
            }
            return lhs.x < rhs.x;
        });

        std::vector<unsigned char> used(pixel_count, 0);
        const auto used_at = [&used, width](const int x, const int y) -> unsigned char& {
            return used[(static_cast<size_t>(y) * static_cast<size_t>(width)) + static_cast<size_t>(x)];
        };
        const auto in_bounds = [width, height](const int x, const int y) {
            return (x >= 1) && (x < width - 1) && (y >= 1) && (y < height - 1);
        };

        size_t emitted = 0;

        const auto subpixel = [&magnitude_at, &vertical_at](const int x, const int y, double& sub_x, double& sub_y) {
            sub_x = static_cast<double>(x);
            sub_y = static_cast<double>(y);
            if (vertical_at(x, y)) {
                const double lower = static_cast<double>(magnitude_at(x - 1, y));
                const double upper = static_cast<double>(magnitude_at(x + 1, y));
                const double central = static_cast<double>(magnitude_at(x, y));
                const double denominator = 2.0 * ((2.0 * central) - lower - upper);
                if (denominator > 1.0e-9) {
                    sub_x += math::max(-0.5, math::min(0.5, (upper - lower) / denominator));
                }
            }
            else {
                const double lower = static_cast<double>(magnitude_at(x, y - 1));
                const double upper = static_cast<double>(magnitude_at(x, y + 1));
                const double central = static_cast<double>(magnitude_at(x, y));
                const double denominator = 2.0 * ((2.0 * central) - lower - upper);
                if (denominator > 1.0e-9) {
                    sub_y += math::max(-0.5, math::min(0.5, (upper - lower) / denominator));
                }
            }
        };

        const auto emit_segment = [&](const line_fit& fit, const double first_x, const double first_y, const double last_x, const double last_y, const double magnitude_sum) {
            if (emitted >= segment_buffer_size) {
                return;
            }
            if (fit.count < detection_options.minimum_fit_pixels) {
                return;
            }
            double centroid_x = 0.0;
            double centroid_y = 0.0;
            double direction_x = 0.0;
            double direction_y = 0.0;
            if (!fit.direction(centroid_x, centroid_y, direction_x, direction_y)) {
                return;
            }
            const double first_projection = ((first_x - centroid_x) * direction_x) + ((first_y - centroid_y) * direction_y);
            const double last_projection = ((last_x - centroid_x) * direction_x) + ((last_y - centroid_y) * direction_y);
            const double length = math::abs(last_projection - first_projection);
            if (length < static_cast<double>(detection_options.minimum_length)) {
                return;
            }
            elsed::segment& result = segment_buffer[emitted];
            result.x1 = static_cast<float>(centroid_x + (first_projection * direction_x) + 0.5);
            result.y1 = static_cast<float>(centroid_y + (first_projection * direction_y) + 0.5);
            result.x2 = static_cast<float>(centroid_x + (last_projection * direction_x) + 0.5);
            result.y2 = static_cast<float>(centroid_y + (last_projection * direction_y) + 0.5);
            result.length = static_cast<float>(length);
            result.response = static_cast<float>(magnitude_sum / static_cast<double>(fit.count));
            result.support = fit.count;
            ++emitted;
        };

        std::vector<double> window_x(static_cast<size_t>(math::max(detection_options.minimum_fit_pixels, 2)));
        std::vector<double> window_y(window_x.size());
        const auto walk = [&](const int start_x, const int start_y, const int initial_dx, const int initial_dy) {
            draw_state state{ start_x, start_y, initial_dx, initial_dy };
            line_fit fit;
            double magnitude_sum = 0.0;
            double first_x = static_cast<double>(start_x);
            double first_y = static_cast<double>(start_y);
            double last_x = first_x;
            double last_y = first_y;
            int outliers = 0;
            bool have_pixel = false;
            bool established = false;
            size_t window_head = 0;
            size_t window_count = 0;

            const auto restart_fit = [&](const double x, const double y, const double pixel_magnitude) {
                fit.clear();
                magnitude_sum = pixel_magnitude;
                fit.add(x, y);
                first_x = x;
                first_y = y;
                last_x = x;
                last_y = y;
                outliers = 0;
                established = false;
                window_head = 0;
                window_count = 1;
                window_x[0] = x;
                window_y[0] = y;
            };

            for (int steps = 0; steps < (width * height); ++steps) {
                if (!in_bounds(state.x, state.y)) {
                    break;
                }
                if (magnitude_at(state.x, state.y) < detection_options.gradient_threshold) {
                    break;
                }
                if (used_at(state.x, state.y) != 0) {
                    break;
                }
                used_at(state.x, state.y) = 1;
                double pixel_x = 0.0;
                double pixel_y = 0.0;
                subpixel(state.x, state.y, pixel_x, pixel_y);
                const double pixel_magnitude = static_cast<double>(magnitude_at(state.x, state.y));

                if (!have_pixel) {
                    have_pixel = true;
                    restart_fit(pixel_x, pixel_y, pixel_magnitude);
                }
                else if (established) {
                    double deviation = 0.0;
                    if (fit.distance(pixel_x, pixel_y, deviation) && (deviation > static_cast<double>(detection_options.point_to_line_threshold))) {
                        ++outliers;
                        if (outliers > detection_options.maximum_outliers) {
                            emit_segment(fit, first_x, first_y, last_x, last_y, magnitude_sum);
                            double centroid_x = 0.0;
                            double centroid_y = 0.0;
                            double direction_x = 0.0;
                            double direction_y = 0.0;
                            bool resumed = false;
                            if (fit.direction(centroid_x, centroid_y, direction_x, direction_y)) {
                                const double forward = (((last_x - first_x) * direction_x) + ((last_y - first_y) * direction_y) >= 0.0) ? 1.0 : -1.0;
                                for (int jump = 2; jump <= detection_options.maximum_jump; ++jump) {
                                    const int jump_x = static_cast<int>(last_x + (forward * direction_x * static_cast<double>(jump)) + 0.5);
                                    const int jump_y = static_cast<int>(last_y + (forward * direction_y * static_cast<double>(jump)) + 0.5);
                                    if (!in_bounds(jump_x, jump_y) || (used_at(jump_x, jump_y) != 0)) {
                                        break;
                                    }
                                    if (magnitude_at(jump_x, jump_y) >= detection_options.gradient_threshold) {
                                        state.x = jump_x;
                                        state.y = jump_y;
                                        state.last_dx = (direction_x * forward > 0.25) ? 1 : ((direction_x * forward < -0.25) ? -1 : 0);
                                        state.last_dy = (direction_y * forward > 0.25) ? 1 : ((direction_y * forward < -0.25) ? -1 : 0);
                                        used_at(jump_x, jump_y) = 1;
                                        double landing_x = 0.0;
                                        double landing_y = 0.0;
                                        subpixel(jump_x, jump_y, landing_x, landing_y);
                                        restart_fit(landing_x, landing_y, static_cast<double>(magnitude_at(jump_x, jump_y)));
                                        resumed = true;
                                        break;
                                    }
                                }
                            }
                            if (!resumed) {
                                return;
                            }
                        }
                    }
                    else {
                        fit.add(pixel_x, pixel_y);
                        magnitude_sum += pixel_magnitude;
                        last_x = pixel_x;
                        last_y = pixel_y;
                        outliers = 0;
                    }
                }
                else {
                    fit.add(pixel_x, pixel_y);
                    magnitude_sum += pixel_magnitude;
                    last_x = pixel_x;
                    last_y = pixel_y;
                    if (window_count < window_x.size()) {
                        const size_t slot = (window_head + window_count) % window_x.size();
                        window_x[slot] = pixel_x;
                        window_y[slot] = pixel_y;
                        ++window_count;
                    }
                    if (static_cast<int>(window_count) >= detection_options.minimum_fit_pixels) {
                        if (fit.rms_error() <= static_cast<double>(detection_options.establishment_rms_threshold)) {
                            established = true;
                        }
                        else {
                            const double oldest_x = window_x[window_head];
                            const double oldest_y = window_y[window_head];
                            fit.remove(oldest_x, oldest_y);
                            magnitude_sum -= magnitude_sum / static_cast<double>(fit.count + 1);
                            window_head = (window_head + 1) % window_x.size();
                            --window_count;
                            first_x = window_x[window_head];
                            first_y = window_y[window_head];
                        }
                    }
                }

                int next_x;
                int next_y;
                if (vertical_at(state.x, state.y)) {
                    int dy = state.last_dy;
                    if (dy == 0) {
                        dy = (magnitude_at(state.x, state.y - 1) >= magnitude_at(state.x, state.y + 1)) ? -1 : 1;
                    }
                    const int candidates[3][2] = { { state.x - 1, state.y + dy }, { state.x, state.y + dy }, { state.x + 1, state.y + dy } };
                    int best = 0;
                    int best_magnitude = -1;
                    for (int i = 0; i < 3; ++i) {
                        if (!in_bounds(candidates[i][0], candidates[i][1])) {
                            continue;
                        }
                        const int candidate_magnitude = magnitude_at(candidates[i][0], candidates[i][1]);
                        if (candidate_magnitude > best_magnitude) {
                            best_magnitude = candidate_magnitude;
                            best = i;
                        }
                    }
                    if (best_magnitude < 0) {
                        break;
                    }
                    next_x = candidates[best][0];
                    next_y = candidates[best][1];
                    state.last_dx = next_x - state.x;
                    state.last_dy = dy;
                }
                else {
                    int dx = state.last_dx;
                    if (dx == 0) {
                        dx = (magnitude_at(state.x - 1, state.y) >= magnitude_at(state.x + 1, state.y)) ? -1 : 1;
                    }
                    const int candidates[3][2] = { { state.x + dx, state.y - 1 }, { state.x + dx, state.y }, { state.x + dx, state.y + 1 } };
                    int best = 0;
                    int best_magnitude = -1;
                    for (int i = 0; i < 3; ++i) {
                        if (!in_bounds(candidates[i][0], candidates[i][1])) {
                            continue;
                        }
                        const int candidate_magnitude = magnitude_at(candidates[i][0], candidates[i][1]);
                        if (candidate_magnitude > best_magnitude) {
                            best_magnitude = candidate_magnitude;
                            best = i;
                        }
                    }
                    if (best_magnitude < 0) {
                        break;
                    }
                    next_x = candidates[best][0];
                    next_y = candidates[best][1];
                    state.last_dx = dx;
                    state.last_dy = next_y - state.y;
                }
                state.x = next_x;
                state.y = next_y;
            }
            if (established) {
                emit_segment(fit, first_x, first_y, last_x, last_y, magnitude_sum);
            }
        };

        for (const anchor& anchor : anchors) {
            if (emitted >= segment_buffer_size) {
                break;
            }
            if (used_at(anchor.x, anchor.y) != 0) {
                continue;
            }
            if (vertical_at(anchor.x, anchor.y)) {
                walk(anchor.x, anchor.y, 0, -1);
                used_at(anchor.x, anchor.y) = 0;
                walk(anchor.x, anchor.y, 0, 1);
            }
            else {
                walk(anchor.x, anchor.y, -1, 0);
                used_at(anchor.x, anchor.y) = 0;
                walk(anchor.x, anchor.y, 1, 0);
            }
        }
        return emitted;
    }
}
