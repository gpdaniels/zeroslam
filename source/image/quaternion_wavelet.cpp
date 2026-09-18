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

#include "image/quaternion_wavelet.hpp"

#include "core/assert.hpp"
#include "core/thread_pool.hpp"
#include "math/math.hpp"

namespace image {
    namespace {
        constexpr static const double first_lowpass_tree_a[quaternion_wavelet::filter_length] = {
            0.0,
            -0.08838834764832,
            0.08838834764832,
            0.69587998903400,
            0.69587998903400,
            0.08838834764832,
            -0.08838834764832,
            0.01122679215254,
            0.01122679215254,
            0.0
        };
        constexpr static const double first_lowpass_tree_b[quaternion_wavelet::filter_length] = {
            0.01122679215254,
            0.01122679215254,
            -0.08838834764832,
            0.08838834764832,
            0.69587998903400,
            0.69587998903400,
            0.08838834764832,
            -0.08838834764832,
            0.0,
            0.0
        };
        constexpr static const double later_lowpass_tree_a[quaternion_wavelet::filter_length] = {
            0.03516384000000,
            0.0,
            -0.08832942000000,
            0.23389032000000,
            0.76027237000000,
            0.58751830000000,
            0.0,
            -0.11430184000000,
            0.0,
            0.0
        };
        constexpr static const double later_lowpass_tree_b[quaternion_wavelet::filter_length] = {
            0.0,
            0.0,
            -0.11430184000000,
            0.0,
            0.58751830000000,
            0.76027237000000,
            0.23389032000000,
            -0.08832942000000,
            0.0,
            0.03516384000000
        };

        constexpr static const int filter_centre = quaternion_wavelet::filter_length / 2;

        int reflect(int index, const int extent) {
            ASSERT(extent > 0, "Cannot reflect into an empty extent.");
            while ((index < 0) || (index >= extent)) {
                if (index < 0) {
                    index = -index - 1;
                }
                if (index >= extent) {
                    index = (2 * extent) - index - 1;
                }
            }
            return index;
        }

        void analyse_x(
            const float* __restrict const source,
            const int rows,
            const int cols,
            const double* __restrict const lowpass,
            const double* __restrict const highpass,
            const bool decimate,
            float* __restrict const low_out,
            float* __restrict const high_out
        ) {
            const int stride = decimate ? 2 : 1;
            const int out_cols = cols / stride;
            const int interior_first = (quaternion_wavelet::filter_length - 1 - filter_centre + stride - 1) / stride;
            const int interior_last = (cols - 1 - filter_centre) / stride;
            core::thread_pool::instance().parallel_for(static_cast<size_t>(rows), 8, [&](const size_t row) {
                const float* __restrict const line = source + (row * static_cast<size_t>(cols));
                float* __restrict const low_line = low_out + (row * static_cast<size_t>(out_cols));
                float* __restrict const high_line = (highpass != nullptr) ? (high_out + (row * static_cast<size_t>(out_cols))) : nullptr;
                for (int node = 0; node < out_cols; ++node) {
                    double low_sum = 0.0;
                    double high_sum = 0.0;
                    if ((node >= interior_first) && (node <= interior_last)) {
                        for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
                            const double sample = static_cast<double>(line[(stride * node) - tap + filter_centre]);
                            low_sum += lowpass[tap] * sample;
                            if (highpass != nullptr) {
                                high_sum += highpass[tap] * sample;
                            }
                        }
                    }
                    else {
                        for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
                            const double sample = static_cast<double>(line[reflect((stride * node) - tap + filter_centre, cols)]);
                            low_sum += lowpass[tap] * sample;
                            if (highpass != nullptr) {
                                high_sum += highpass[tap] * sample;
                            }
                        }
                    }
                    low_line[node] = static_cast<float>(low_sum);
                    if (high_line != nullptr) {
                        high_line[node] = static_cast<float>(high_sum);
                    }
                }
            });
        }

        void analyse_y(
            const float* __restrict const source,
            const int rows,
            const int cols,
            const double* __restrict const lowpass,
            const double* __restrict const highpass,
            const bool decimate,
            float* __restrict const low_out,
            float* __restrict const high_out
        ) {
            const int stride = decimate ? 2 : 1;
            const int out_rows = rows / stride;
            core::thread_pool::instance().parallel_for(static_cast<size_t>(out_rows), 4, [&](const size_t node) {
                int source_row[quaternion_wavelet::filter_length];
                for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
                    source_row[tap] = reflect((stride * static_cast<int>(node)) - tap + filter_centre, rows);
                }
                float* __restrict const low_line = low_out + (node * static_cast<size_t>(cols));
                float* __restrict const high_line = (highpass != nullptr) ? (high_out + (node * static_cast<size_t>(cols))) : nullptr;
                for (int column = 0; column < cols; ++column) {
                    low_line[column] = 0.0f;
                    if (high_line != nullptr) {
                        high_line[column] = 0.0f;
                    }
                }
                for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
                    const float* __restrict const line = source + (static_cast<size_t>(source_row[tap]) * static_cast<size_t>(cols));
                    const float low_tap = static_cast<float>(lowpass[tap]);
                    for (int column = 0; column < cols; ++column) {
                        low_line[column] += low_tap * line[column];
                    }
                    if (high_line != nullptr) {
                        const float high_tap = static_cast<float>(highpass[tap]);
                        for (int column = 0; column < cols; ++column) {
                            high_line[column] += high_tap * line[column];
                        }
                    }
                }
            });
        }
    }

    const double* quaternion_wavelet::first_lowpass_a() {
        return &first_lowpass_tree_a[0];
    }

    const double* quaternion_wavelet::first_lowpass_b() {
        return &first_lowpass_tree_b[0];
    }

    const double* quaternion_wavelet::later_lowpass_a() {
        return &later_lowpass_tree_a[0];
    }

    const double* quaternion_wavelet::later_lowpass_b() {
        return &later_lowpass_tree_b[0];
    }

    void quaternion_wavelet::quadrature_mirror(const double* const lowpass, double* const highpass) {
        for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
            const double sign = ((tap % 2) == 0) ? 1.0 : -1.0;
            highpass[tap] = sign * lowpass[quaternion_wavelet::filter_length - 1 - tap];
        }
    }

    namespace {
        struct centre_table final {
            double wavelet[quaternion_wavelet::maximum_levels];
            double scaling[quaternion_wavelet::maximum_levels];
        };

        void compute_spectral_centres(const size_t level_count, double* const wavelet_out, double* const scaling_out);

        const centre_table& cached_centres() {
            static const centre_table table = []() {
                centre_table built;
                compute_spectral_centres(quaternion_wavelet::maximum_levels, &built.wavelet[0], &built.scaling[0]);
                return built;
            }();
            return table;
        }
    }

    void quaternion_wavelet::spectral_centres(const size_t level_count, double* const wavelet_out, double* const scaling_out) {
        const centre_table& table = cached_centres();
        for (size_t level = 0; (level < level_count) && (level < quaternion_wavelet::maximum_levels); ++level) {
            wavelet_out[level] = table.wavelet[level];
            scaling_out[level] = table.scaling[level];
        }
    }

    namespace {
        void compute_spectral_centres(const size_t level_count, double* const wavelet_out, double* const scaling_out) {
            constexpr static const int samples = 1024;
            std::vector<double> cumulative_real[2];
            std::vector<double> cumulative_imaginary[2];
            for (int tree = 0; tree < 2; ++tree) {
                cumulative_real[tree].assign(samples, 1.0);
                cumulative_imaginary[tree].assign(samples, 0.0);
            }
            std::vector<double> frequency(samples, 0.0);
            for (int index = 0; index < samples; ++index) {
                frequency[static_cast<size_t>(index)] = 2.0 * math::pi<double>() * static_cast<double>(index - (samples / 2)) / static_cast<double>(samples);
            }
            for (size_t level = 1; level <= level_count; ++level) {
                double lowpass[2][quaternion_wavelet::filter_length];
                double highpass[2][quaternion_wavelet::filter_length];
                for (int tree = 0; tree < 2; ++tree) {
                    const double* const source = (level == 1) ? ((tree == 0) ? &first_lowpass_tree_a[0] : &first_lowpass_tree_b[0]) : ((tree == 0) ? &later_lowpass_tree_a[0] : &later_lowpass_tree_b[0]);
                    for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
                        lowpass[tree][tap] = source[tap];
                    }
                    quaternion_wavelet::quadrature_mirror(&lowpass[tree][0], &highpass[tree][0]);
                }
                const double stride = static_cast<double>(1 << (level - 1));
                double wavelet_moment = 0.0;
                double wavelet_power = 0.0;
                double scaling_moment = 0.0;
                double scaling_power = 0.0;
                std::vector<double> next_real[2];
                std::vector<double> next_imaginary[2];
                for (int tree = 0; tree < 2; ++tree) {
                    next_real[tree].assign(samples, 0.0);
                    next_imaginary[tree].assign(samples, 0.0);
                }
                for (int index = 0; index < samples; ++index) {
                    const double angle = frequency[static_cast<size_t>(index)] * stride;
                    double wavelet_stage_real[2] = { 0.0, 0.0 };
                    double wavelet_stage_imaginary[2] = { 0.0, 0.0 };
                    double scaling_stage_real[2] = { 0.0, 0.0 };
                    double scaling_stage_imaginary[2] = { 0.0, 0.0 };
                    for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
                        double sine = 0.0;
                        double cosine = 0.0;
                        math::sincos(angle * static_cast<double>(tap), sine, cosine);
                        for (int tree = 0; tree < 2; ++tree) {
                            wavelet_stage_real[tree] += highpass[tree][tap] * cosine;
                            wavelet_stage_imaginary[tree] -= highpass[tree][tap] * sine;
                            scaling_stage_real[tree] += lowpass[tree][tap] * cosine;
                            scaling_stage_imaginary[tree] -= lowpass[tree][tap] * sine;
                        }
                    }
                    double wavelet_real[2];
                    double wavelet_imaginary[2];
                    for (int tree = 0; tree < 2; ++tree) {
                        const double base_real = cumulative_real[tree][static_cast<size_t>(index)];
                        const double base_imaginary = cumulative_imaginary[tree][static_cast<size_t>(index)];
                        wavelet_real[tree] = (base_real * wavelet_stage_real[tree]) - (base_imaginary * wavelet_stage_imaginary[tree]);
                        wavelet_imaginary[tree] = (base_real * wavelet_stage_imaginary[tree]) + (base_imaginary * wavelet_stage_real[tree]);
                        next_real[tree][static_cast<size_t>(index)] = (base_real * scaling_stage_real[tree]) - (base_imaginary * scaling_stage_imaginary[tree]);
                        next_imaginary[tree][static_cast<size_t>(index)] = (base_real * scaling_stage_imaginary[tree]) + (base_imaginary * scaling_stage_real[tree]);
                    }
                    const double analytic_wavelet_real = wavelet_real[0] - wavelet_imaginary[1];
                    const double analytic_wavelet_imaginary = wavelet_imaginary[0] + wavelet_real[1];
                    const double analytic_scaling_real = next_real[0][static_cast<size_t>(index)] - next_imaginary[1][static_cast<size_t>(index)];
                    const double analytic_scaling_imaginary = next_imaginary[0][static_cast<size_t>(index)] + next_real[1][static_cast<size_t>(index)];
                    const double wavelet_magnitude = (analytic_wavelet_real * analytic_wavelet_real) + (analytic_wavelet_imaginary * analytic_wavelet_imaginary);
                    const double scaling_magnitude = (analytic_scaling_real * analytic_scaling_real) + (analytic_scaling_imaginary * analytic_scaling_imaginary);
                    wavelet_moment += frequency[static_cast<size_t>(index)] * wavelet_magnitude;
                    wavelet_power += wavelet_magnitude;
                    scaling_moment += frequency[static_cast<size_t>(index)] * scaling_magnitude;
                    scaling_power += scaling_magnitude;
                }
                const double scale = static_cast<double>(1 << level) / (2.0 * math::pi<double>());
                wavelet_out[level - 1] = (wavelet_power > 0.0) ? ((wavelet_moment / wavelet_power) * scale) : 0.0;
                scaling_out[level - 1] = (scaling_power > 0.0) ? ((scaling_moment / scaling_power) * scale) : 0.0;
                for (int tree = 0; tree < 2; ++tree) {
                    cumulative_real[tree] = next_real[tree];
                    cumulative_imaginary[tree] = next_imaginary[tree];
                }
            }
        }

    }

    quaternion_wavelet::quaternion_wavelet()
        : levels() {
    }

    quaternion_wavelet::quaternion_wavelet(const image& base, const size_t level_count, const size_t finest_band, const bool undecimated_finest) {
        const size_t rows = base.get_rows();
        const size_t cols = base.get_cols();
        if ((rows < quaternion_wavelet::minimum_dimension * 2) || (cols < quaternion_wavelet::minimum_dimension * 2)) {
            return;
        }
        size_t limit = math::min(level_count, quaternion_wavelet::maximum_levels);
        if (limit == 0) {
            limit = 1;
            size_t extent = math::min(rows, cols);
            while (((extent / 2) >= quaternion_wavelet::minimum_dimension) && (limit < quaternion_wavelet::maximum_levels)) {
                extent /= 2;
                ++limit;
            }
        }
        constexpr static const int tree_x[quaternion_wavelet::component_count] = { 0, 1, 0, 1 };
        constexpr static const int tree_y[quaternion_wavelet::component_count] = { 0, 0, 1, 1 };
        std::vector<float> scaling[quaternion_wavelet::component_count];
        for (int component = 0; component < quaternion_wavelet::component_count; ++component) {
            scaling[component].resize(rows * cols);
            for (size_t index = 0; index < rows * cols; ++index) {
                scaling[component][index] = static_cast<float>(base.get_data()[index]);
            }
        }
        this->levels.reserve(limit);
        double wavelet_centres[quaternion_wavelet::maximum_levels];
        double scaling_centres[quaternion_wavelet::maximum_levels];
        quaternion_wavelet::spectral_centres(limit, &wavelet_centres[0], &scaling_centres[0]);
        size_t level_rows = rows;
        size_t level_cols = cols;
        for (size_t index = 0; index < limit; ++index) {
            const size_t half_rows = level_rows / 2;
            const size_t half_cols = level_cols / 2;
            if ((half_rows < quaternion_wavelet::minimum_dimension) || (half_cols < quaternion_wavelet::minimum_dimension)) {
                break;
            }
            double lowpass[2][quaternion_wavelet::filter_length];
            double highpass[2][quaternion_wavelet::filter_length];
            for (int tree = 0; tree < 2; ++tree) {
                const double* const source = (index == 0) ? ((tree == 0) ? &first_lowpass_tree_a[0] : &first_lowpass_tree_b[0]) : ((tree == 0) ? &later_lowpass_tree_a[0] : &later_lowpass_tree_b[0]);
                for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
                    lowpass[tree][tap] = source[tap];
                }
                quaternion_wavelet::quadrature_mirror(&lowpass[tree][0], &highpass[tree][0]);
            }
            const bool keep = ((index + 1) >= finest_band);
            const bool dense = keep && undecimated_finest && ((index + 1) == finest_band);
            const size_t band_rows = dense ? level_rows : half_rows;
            const size_t band_cols = dense ? level_cols : half_cols;
            level built;
            built.rows = keep ? band_rows : 0;
            built.cols = keep ? band_cols : 0;
            built.spacing = static_cast<double>(1u << (index + 1)) / (dense ? 2.0 : 1.0);
            built.wavelet_centre = wavelet_centres[index] / (dense ? 2.0 : 1.0);
            built.scaling_centre = scaling_centres[index] / (dense ? 2.0 : 1.0);
            if (keep) {
                for (int which = 0; which < quaternion_wavelet::band_count; ++which) {
                    built.bands[which].assign(band_rows * band_cols * quaternion_wavelet::component_count, 0.0f);
                }
            }
            std::vector<float> column_low(level_rows * half_cols, 0.0f);
            std::vector<float> column_high(level_rows * half_cols, 0.0f);
            std::vector<float> corner[quaternion_wavelet::component_count];
            for (int component = 0; component < quaternion_wavelet::component_count; ++component) {
                const int x = tree_x[component];
                const int y = tree_y[component];
                std::vector<float> next_scaling(half_rows * half_cols, 0.0f);
                analyse_x(scaling[component].data(), static_cast<int>(level_rows), static_cast<int>(level_cols), &lowpass[x][0], keep && !dense ? &highpass[x][0] : nullptr, true, column_low.data(), column_high.data());
                if (!keep) {
                    analyse_y(column_low.data(), static_cast<int>(level_rows), static_cast<int>(half_cols), &lowpass[y][0], nullptr, true, next_scaling.data(), nullptr);
                    corner[component] = static_cast<std::vector<float>&&>(next_scaling);
                    continue;
                }
                std::vector<float> horizontal(band_rows * band_cols, 0.0f);
                std::vector<float> vertical(band_rows * band_cols, 0.0f);
                std::vector<float> diagonal(band_rows * band_cols, 0.0f);
                if (dense) {
                    analyse_y(column_low.data(), static_cast<int>(level_rows), static_cast<int>(half_cols), &lowpass[y][0], nullptr, true, next_scaling.data(), nullptr);
                    std::vector<float> dense_low(level_rows * level_cols, 0.0f);
                    std::vector<float> dense_high(level_rows * level_cols, 0.0f);
                    analyse_x(scaling[component].data(), static_cast<int>(level_rows), static_cast<int>(level_cols), &lowpass[x][0], &highpass[x][0], false, dense_low.data(), dense_high.data());
                    std::vector<float> unused(level_rows * level_cols, 0.0f);
                    analyse_y(dense_low.data(), static_cast<int>(level_rows), static_cast<int>(level_cols), &lowpass[y][0], &highpass[y][0], false, unused.data(), horizontal.data());
                    analyse_y(dense_high.data(), static_cast<int>(level_rows), static_cast<int>(level_cols), &lowpass[y][0], &highpass[y][0], false, vertical.data(), diagonal.data());
                }
                else {
                    analyse_y(column_low.data(), static_cast<int>(level_rows), static_cast<int>(half_cols), &lowpass[y][0], &highpass[y][0], true, next_scaling.data(), horizontal.data());
                    analyse_y(column_high.data(), static_cast<int>(level_rows), static_cast<int>(half_cols), &lowpass[y][0], &highpass[y][0], true, vertical.data(), diagonal.data());
                }
                for (size_t node = 0; node < band_rows * band_cols; ++node) {
                    const size_t slot = (node * quaternion_wavelet::component_count) + static_cast<size_t>(component);
                    built.bands[static_cast<int>(band::horizontal)][slot] = horizontal[node];
                    built.bands[static_cast<int>(band::vertical)][slot] = vertical[node];
                    built.bands[static_cast<int>(band::diagonal)][slot] = diagonal[node];
                }
                corner[component] = static_cast<std::vector<float>&&>(next_scaling);
            }
            for (int component = 0; component < quaternion_wavelet::component_count; ++component) {
                scaling[component] = static_cast<std::vector<float>&&>(corner[component]);
            }
            this->levels.push_back(static_cast<level&&>(built));
            level_rows = half_rows;
            level_cols = half_cols;
        }
    }

    size_t quaternion_wavelet::size() const {
        return this->levels.size();
    }

    const quaternion_wavelet::level& quaternion_wavelet::operator[](const size_t index) const {
        ASSERT((index >= 1) && (index <= this->levels.size()), "Quaternion wavelet level out of range.");
        return this->levels[index - 1];
    }

    const float* quaternion_wavelet::at(const size_t index, const band which, const size_t x, const size_t y) const {
        if ((index < 1) || (index > this->levels.size())) {
            return nullptr;
        }
        const level& data = this->levels[index - 1];
        if ((x >= data.cols) || (y >= data.rows)) {
            return nullptr;
        }
        return &data.bands[static_cast<int>(which)][(((y * data.cols) + x) * quaternion_wavelet::component_count)];
    }

    double quaternion_wavelet::centre_x(const level& data, const band which) {
        return (which == band::horizontal) ? data.scaling_centre : data.wavelet_centre;
    }

    double quaternion_wavelet::centre_y(const level& data, const band which) {
        return (which == band::vertical) ? data.scaling_centre : data.wavelet_centre;
    }

    void quaternion_wavelet::complex_pair(const float* const components, float& plus_real, float& plus_imaginary, float& minus_real, float& minus_imaginary) {
        plus_real = components[0] - components[3];
        plus_imaginary = components[1] + components[2];
        minus_real = components[0] + components[3];
        minus_imaginary = components[1] - components[2];
    }

    float quaternion_wavelet::modulus(const float* const components) {
        return math::sqrt((components[0] * components[0]) + (components[1] * components[1]) + (components[2] * components[2]) + (components[3] * components[3]));
    }

    bool quaternion_wavelet::phases(const float* const components, double& first, double& second, double& third) {
        first = 0.0;
        second = 0.0;
        third = 0.0;
        float plus_real = 0.0f;
        float plus_imaginary = 0.0f;
        float minus_real = 0.0f;
        float minus_imaginary = 0.0f;
        quaternion_wavelet::complex_pair(components, plus_real, plus_imaginary, minus_real, minus_imaginary);
        const double plus_power = (static_cast<double>(plus_real) * static_cast<double>(plus_real)) + (static_cast<double>(plus_imaginary) * static_cast<double>(plus_imaginary));
        const double minus_power = (static_cast<double>(minus_real) * static_cast<double>(minus_real)) + (static_cast<double>(minus_imaginary) * static_cast<double>(minus_imaginary));
        if ((plus_power <= 0.0) || (minus_power <= 0.0)) {
            return false;
        }
        const double plus_angle = math::atan2(static_cast<double>(plus_imaginary), static_cast<double>(plus_real));
        const double minus_angle = math::atan2(static_cast<double>(minus_imaginary), static_cast<double>(minus_real));
        first = 0.5 * (plus_angle + minus_angle);
        second = 0.5 * (plus_angle - minus_angle);
        const double sine = (minus_power - plus_power) / (plus_power + minus_power);
        third = 0.5 * math::asin(math::max(-1.0, math::min(1.0, sine)));
        return true;
    }
}
