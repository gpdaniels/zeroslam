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

        // One row of the x analysis, the low and optional high outputs of every stride-th node.
        template <typename sample_type>
        void analyse_x_row(
            const sample_type* __restrict const line,
            const int cols,
            const double* __restrict const lowpass,
            const double* __restrict const highpass,
            const int stride,
            float* __restrict const low_line,
            float* __restrict const high_line
        ) {
            const int out_cols = cols / stride;
            const int interior_first = (quaternion_wavelet::filter_length - 1 - filter_centre + stride - 1) / stride;
            const int interior_last = (cols - 1 - filter_centre) / stride;
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
                if (highpass != nullptr) {
                    high_line[node] = static_cast<float>(high_sum);
                }
            }
        }

        // One output row of the y analysis, either output optional, accumulated a block of columns at a time and stored step floats apart so it can go straight into the interleaved bands.
        void analyse_y_row(
            const float* __restrict const source,
            const int rows,
            const int cols,
            const double* __restrict const lowpass,
            const double* __restrict const highpass,
            const int stride,
            const int node,
            float* __restrict const low_out,
            const size_t low_step,
            float* __restrict const high_out,
            const size_t high_step
        ) {
            constexpr static const int block = 256;
            const float* lines[quaternion_wavelet::filter_length];
            for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
                lines[tap] = source + (static_cast<size_t>(reflect((stride * node) - tap + filter_centre, rows)) * static_cast<size_t>(cols));
            }
            const size_t row_start = static_cast<size_t>(node) * static_cast<size_t>(cols);
            float low_sums[block];
            float high_sums[block];
            for (int first = 0; first < cols; first += block) {
                const int count = math::min(block, cols - first);
                for (int column = 0; column < count; ++column) {
                    low_sums[column] = 0.0f;
                    high_sums[column] = 0.0f;
                }
                for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
                    const float* __restrict const line = lines[tap] + first;
                    if (low_out != nullptr) {
                        const float low_tap = static_cast<float>(lowpass[tap]);
                        for (int column = 0; column < count; ++column) {
                            low_sums[column] += low_tap * line[column];
                        }
                    }
                    if (high_out != nullptr) {
                        const float high_tap = static_cast<float>(highpass[tap]);
                        for (int column = 0; column < count; ++column) {
                            high_sums[column] += high_tap * line[column];
                        }
                    }
                }
                const size_t start = row_start + static_cast<size_t>(first);
                if (low_out != nullptr) {
                    for (int column = 0; column < count; ++column) {
                        low_out[(start + static_cast<size_t>(column)) * low_step] = low_sums[column];
                    }
                }
                if (high_out != nullptr) {
                    for (int column = 0; column < count; ++column) {
                        high_out[(start + static_cast<size_t>(column)) * high_step] = high_sums[column];
                    }
                }
            }
        }

        // A task needs about this many multiply adds to outweigh the thread pool's cost of handing it out.
        constexpr static const size_t minimum_task_work = static_cast<size_t>(1) << 17;

        size_t task_grain(const size_t work_per_item) {
            return math::max(static_cast<size_t>(1), minimum_task_work / math::max(static_cast<size_t>(1), work_per_item));
        }

        // An x analysis of a scaling plane, or of the base image when the plane is null, the high output optional.
        struct x_analysis final {
            const float* plane;
            int tree;
            int stride;
            float* low_out;
            float* high_out;
        };

        // A y analysis of an x analysis output, either output optional and stored step floats apart.
        struct y_analysis final {
            const float* source;
            int cols;
            int tree;
            int stride;
            float* low_out;
            size_t low_step;
            float* high_out;
            size_t high_step;
        };

        // Runs y analyses with the same output rows as one set of tasks, ordered by row so a task fills every band slot of its rows.
        void run_y_analyses(const y_analysis* const analyses, const size_t count, const size_t rows, const size_t out_rows, const double (&lowpass)[2][quaternion_wavelet::filter_length], const double (&highpass)[2][quaternion_wavelet::filter_length]) {
            if (count == 0) {
                return;
            }
            core::thread_pool::instance().parallel_for(count * out_rows, task_grain(static_cast<size_t>(analyses[0].cols) * static_cast<size_t>(quaternion_wavelet::filter_length)), [&](const size_t item) {
                const y_analysis& analysis = analyses[item % count];
                analyse_y_row(analysis.source, static_cast<int>(rows), analysis.cols, &lowpass[analysis.tree][0], &highpass[analysis.tree][0], analysis.stride, static_cast<int>(item / count), analysis.low_out, analysis.low_step, analysis.high_out, analysis.high_step);
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
        struct complex_value final {
            double real;
            double imaginary;
        };

        complex_value multiply(const complex_value& lhs, const complex_value& rhs) {
            return complex_value{ (lhs.real * rhs.real) - (lhs.imaginary * rhs.imaginary), (lhs.real * rhs.imaginary) + (lhs.imaginary * rhs.real) };
        }

        // Each level's spectrum is half as wide as the level above, so the grid doubles with the level to resolve it.
        size_t centre_samples(const size_t level) {
            return math::max(static_cast<size_t>(1024), static_cast<size_t>(16) << level);
        }

        // A filter's response at 2 pi j / period for j in [0, period), from a table of e^(-2 pi i m / size) whose size the power of two period divides.
        // The taps are real, so the second half of the period is the conjugate of the first.
        void filter_response(const double* const taps, const std::vector<complex_value>& twiddles, const size_t period, complex_value* const response) {
            const size_t step = twiddles.size() / period;
            for (size_t node = 0; node <= (period / 2); ++node) {
                complex_value sum{ 0.0, 0.0 };
                size_t phase = 0;
                for (int tap = 0; tap < quaternion_wavelet::filter_length; ++tap) {
                    const complex_value& twiddle = twiddles[phase * step];
                    sum.real += taps[tap] * twiddle.real;
                    sum.imaginary += taps[tap] * twiddle.imaginary;
                    phase = (phase + node) & (period - 1);
                }
                response[node] = sum;
                if ((node > 0) && (node < (period / 2))) {
                    response[period - node] = complex_value{ sum.real, -sum.imaginary };
                }
            }
        }

        void compute_spectral_centres(const size_t level_count, double* const wavelet_out, double* const scaling_out) {
            const double* const first_lowpass[2] = { &first_lowpass_tree_a[0], &first_lowpass_tree_b[0] };
            const double* const later_lowpass[2] = { &later_lowpass_tree_a[0], &later_lowpass_tree_b[0] };
            // The twiddles e^(-2 pi i m / size) of the finest grid, the first quadrant's sines and cosines rotated into the other three.
            const size_t size = centre_samples(level_count);
            const size_t quarter = size / 4;
            std::vector<complex_value> twiddles(size);
            for (size_t index = 0; index < quarter; ++index) {
                double sine = 0.0;
                double cosine = 0.0;
                math::sincos(2.0 * math::pi<double>() * static_cast<double>(index) / static_cast<double>(size), sine, cosine);
                twiddles[index] = complex_value{ cosine, -sine };
                twiddles[index + quarter] = complex_value{ -sine, -cosine };
                twiddles[index + (2 * quarter)] = complex_value{ -cosine, sine };
                twiddles[index + (3 * quarter)] = complex_value{ sine, cosine };
            }
            // Every stage of every level samples one of these, the first lowpass on the finest grid and the later lowpass, whose stages filter at least twice the frequency, on half of it.
            std::vector<complex_value> first_response[2];
            std::vector<complex_value> later_response[2];
            std::vector<complex_value> wavelet[2];
            std::vector<complex_value> scaling[2];
            for (int tree = 0; tree < 2; ++tree) {
                first_response[tree].resize(size);
                later_response[tree].resize(size / 2);
                wavelet[tree].resize(size);
                scaling[tree].resize(size);
                filter_response(first_lowpass[tree], twiddles, size, first_response[tree].data());
                filter_response(later_lowpass[tree], twiddles, size / 2, later_response[tree].data());
            }
            for (size_t level = 1; level <= level_count; ++level) {
                // Stage k filters 2^(k - 1) times the frequency, so its response repeats every samples / 2^(k - 1) nodes.
                // Starting from the last stage, each finer stage multiplies the product so far over its own period.
                const size_t samples = centre_samples(level);
                const size_t coarsest = samples >> (level - 1);
                for (int tree = 0; tree < 2; ++tree) {
                    const std::vector<complex_value>& response = (level == 1) ? first_response[tree] : later_response[tree];
                    double highpass[quaternion_wavelet::filter_length];
                    quaternion_wavelet::quadrature_mirror((level == 1) ? first_lowpass[tree] : later_lowpass[tree], &highpass[0]);
                    filter_response(&highpass[0], twiddles, coarsest, wavelet[tree].data());
                    for (size_t node = 0; node < coarsest; ++node) {
                        scaling[tree][node] = response[node * (response.size() / coarsest)];
                    }
                }
                for (size_t finer = level - 1; finer >= 1; --finer) {
                    const size_t period = samples >> (finer - 1);
                    const size_t half = period / 2;
                    for (int tree = 0; tree < 2; ++tree) {
                        const std::vector<complex_value>& stage = (finer == 1) ? first_response[tree] : later_response[tree];
                        const size_t step = stage.size() / period;
                        // The product so far repeats every half period, so the upper half reads the lower half before the lower half is updated in place.
                        for (size_t node = half; node < period; ++node) {
                            wavelet[tree][node] = multiply(stage[node * step], wavelet[tree][node - half]);
                            scaling[tree][node] = multiply(stage[node * step], scaling[tree][node - half]);
                        }
                        for (size_t node = 0; node < half; ++node) {
                            wavelet[tree][node] = multiply(stage[node * step], wavelet[tree][node]);
                            scaling[tree][node] = multiply(stage[node * step], scaling[tree][node]);
                        }
                    }
                }
                double wavelet_moment = 0.0;
                double wavelet_power = 0.0;
                double scaling_moment = 0.0;
                double scaling_power = 0.0;
                for (size_t index = 0; index < samples; ++index) {
                    // The Nyquist sample is at both -pi and pi, so it adds power but no moment.
                    const double frequency = (index == 0) ? 0.0 : (2.0 * math::pi<double>() * (static_cast<double>(index) - static_cast<double>(samples / 2)) / static_cast<double>(samples));
                    const size_t node = (index + (samples / 2)) & (samples - 1);
                    const double analytic_wavelet_real = wavelet[0][node].real - wavelet[1][node].imaginary;
                    const double analytic_wavelet_imaginary = wavelet[0][node].imaginary + wavelet[1][node].real;
                    const double analytic_scaling_real = scaling[0][node].real - scaling[1][node].imaginary;
                    const double analytic_scaling_imaginary = scaling[0][node].imaginary + scaling[1][node].real;
                    const double wavelet_magnitude = (analytic_wavelet_real * analytic_wavelet_real) + (analytic_wavelet_imaginary * analytic_wavelet_imaginary);
                    const double scaling_magnitude = (analytic_scaling_real * analytic_scaling_real) + (analytic_scaling_imaginary * analytic_scaling_imaginary);
                    wavelet_moment += frequency * wavelet_magnitude;
                    wavelet_power += wavelet_magnitude;
                    scaling_moment += frequency * scaling_magnitude;
                    scaling_power += scaling_magnitude;
                }
                const double scale = static_cast<double>(1u << level) / (2.0 * math::pi<double>());
                wavelet_out[level - 1] = (wavelet_power > 0.0) ? ((wavelet_moment / wavelet_power) * scale) : 0.0;
                scaling_out[level - 1] = (scaling_power > 0.0) ? ((scaling_moment / scaling_power) * scale) : 0.0;
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
        // The first level's four scaling planes would all be the base image, so it is read directly and analysed once per x tree.
        std::vector<float> scaling[quaternion_wavelet::component_count];
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
            // Every x analysis of the level runs as one set of tasks, then every y analysis, each output computed by one task in a fixed order so the results do not depend on the thread count.
            const size_t planes = static_cast<size_t>((index == 0) ? 2 : quaternion_wavelet::component_count);
            std::vector<float> column_low[quaternion_wavelet::component_count];
            std::vector<float> column_high[quaternion_wavelet::component_count];
            std::vector<float> dense_low[quaternion_wavelet::component_count];
            std::vector<float> dense_high[quaternion_wavelet::component_count];
            x_analysis x_analyses[2 * quaternion_wavelet::component_count];
            size_t x_count = 0;
            for (size_t plane = 0; plane < planes; ++plane) {
                const float* const source = (index == 0) ? nullptr : scaling[plane].data();
                const int x = (index == 0) ? static_cast<int>(plane) : tree_x[plane];
                column_low[plane].resize(level_rows * half_cols);
                column_high[plane].resize((keep && !dense) ? (level_rows * half_cols) : 0);
                x_analyses[x_count++] = x_analysis{ source, x, 2, column_low[plane].data(), (keep && !dense) ? column_high[plane].data() : nullptr };
                if (dense) {
                    dense_low[plane].resize(level_rows * level_cols);
                    dense_high[plane].resize(level_rows * level_cols);
                    x_analyses[x_count++] = x_analysis{ source, x, 1, dense_low[plane].data(), dense_high[plane].data() };
                }
            }
            core::thread_pool::instance().parallel_for(x_count * level_rows, task_grain(half_cols * static_cast<size_t>(quaternion_wavelet::filter_length)), [&](const size_t item) {
                const x_analysis& analysis = x_analyses[item / level_rows];
                const size_t row = item % level_rows;
                const size_t out_cols = level_cols / static_cast<size_t>(analysis.stride);
                float* const low_line = analysis.low_out + (row * out_cols);
                float* const high_line = (analysis.high_out != nullptr) ? (analysis.high_out + (row * out_cols)) : nullptr;
                const double* const high_filter = (analysis.high_out != nullptr) ? &highpass[analysis.tree][0] : nullptr;
                if (analysis.plane == nullptr) {
                    analyse_x_row(base.get_data() + (row * level_cols), static_cast<int>(level_cols), &lowpass[analysis.tree][0], high_filter, analysis.stride, low_line, high_line);
                }
                else {
                    analyse_x_row(analysis.plane + (row * level_cols), static_cast<int>(level_cols), &lowpass[analysis.tree][0], high_filter, analysis.stride, low_line, high_line);
                }
            });
            // The decimated lowpass of the x lowpass is the next level's scaling plane, the other outputs go straight into the components' band slots.
            std::vector<float> next_scaling[quaternion_wavelet::component_count];
            y_analysis decimated[2 * quaternion_wavelet::component_count];
            y_analysis undecimated[2 * quaternion_wavelet::component_count];
            size_t decimated_count = 0;
            size_t undecimated_count = 0;
            constexpr static const size_t slot_step = static_cast<size_t>(quaternion_wavelet::component_count);
            for (int component = 0; component < quaternion_wavelet::component_count; ++component) {
                const size_t plane = (index == 0) ? static_cast<size_t>(tree_x[component]) : static_cast<size_t>(component);
                const int y = tree_y[component];
                next_scaling[component].resize(half_rows * half_cols);
                float* horizontal = nullptr;
                float* vertical = nullptr;
                float* diagonal = nullptr;
                if (keep) {
                    horizontal = &built.bands[static_cast<int>(band::horizontal)][static_cast<size_t>(component)];
                    vertical = &built.bands[static_cast<int>(band::vertical)][static_cast<size_t>(component)];
                    diagonal = &built.bands[static_cast<int>(band::diagonal)][static_cast<size_t>(component)];
                }
                if (keep && !dense) {
                    decimated[decimated_count++] = y_analysis{ column_low[plane].data(), static_cast<int>(half_cols), y, 2, next_scaling[component].data(), 1, horizontal, slot_step };
                    decimated[decimated_count++] = y_analysis{ column_high[plane].data(), static_cast<int>(half_cols), y, 2, vertical, slot_step, diagonal, slot_step };
                }
                else {
                    decimated[decimated_count++] = y_analysis{ column_low[plane].data(), static_cast<int>(half_cols), y, 2, next_scaling[component].data(), 1, nullptr, 0 };
                }
                if (dense) {
                    undecimated[undecimated_count++] = y_analysis{ dense_low[plane].data(), static_cast<int>(level_cols), y, 1, nullptr, 0, horizontal, slot_step };
                    undecimated[undecimated_count++] = y_analysis{ dense_high[plane].data(), static_cast<int>(level_cols), y, 1, vertical, slot_step, diagonal, slot_step };
                }
            }
            run_y_analyses(&decimated[0], decimated_count, level_rows, half_rows, lowpass, highpass);
            run_y_analyses(&undecimated[0], undecimated_count, level_rows, level_rows, lowpass, highpass);
            for (int component = 0; component < quaternion_wavelet::component_count; ++component) {
                scaling[component] = static_cast<std::vector<float>&&>(next_scaling[component]);
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
