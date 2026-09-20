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

#if !defined(ZEROSLAM_API_EXPORT)
#define ZEROSLAM_API_EXPORT
#endif

#include "api.hpp"

#include "core/logger.hpp"
#include "feature/refiner/structure_tensor.hpp"
#include "feature/score/structure_tensor.hpp"
#include "image/image.hpp"
#include "math/lie.hpp"
#include "math/matrix.hpp"
#include "optimisation/factor_graph.hpp"
#include "sensor/camera.hpp"
#include "slam.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>
#include <unordered_map>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

struct zeroslam_system final {
    slam slam_instance;

    struct camera_entry final {
        int sensor_id;
        zeroslam_sensor_parameters_camera_struct parameters;
    };

    std::vector<camera_entry> cameras;

    struct frame_record final {
        long long int timestamp;
        int frame_id;
    };

    std::vector<frame_record> frames;
    long long int latest_timestamp = 0;
    bool any_data = false;
};

extern "C" {

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_create(zeroslam_system** system) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_argument;
    }
    *system = new (std::nothrow) zeroslam_system();
    if (*system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    return zeroslam_return_success;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_destroy(zeroslam_system** system) {
    if ((system == nullptr) || (*system == nullptr)) {
        return zeroslam_return_failure_invalid_system;
    }
    delete *system;
    *system = nullptr;
    return zeroslam_return_success;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_timestamp(zeroslam_system* system, long long int* timestamp) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if (timestamp == nullptr) {
        return zeroslam_return_failure_invalid_argument;
    }
    *timestamp = system->latest_timestamp;
    return zeroslam_return_success;
}
}

namespace {
    constexpr static const char configuration_key_verbosity[] = "verbosity";
    constexpr static const char configuration_key_detector[] = "detector";
    constexpr static const char configuration_key_detector_sigma[] = "detector_sigma";
    constexpr static const char configuration_key_refiner[] = "refiner";
    constexpr static const char configuration_key_refiner_sigma[] = "refiner_sigma";
    constexpr static const char configuration_key_tracker[] = "tracker";
    constexpr static const char configuration_key_association[] = "association";
    constexpr static const char configuration_key_descriptor[] = "descriptor";
    constexpr static const char configuration_key_affine[] = "affine";
    constexpr static const char configuration_key_blur[] = "blur";
    constexpr static const char configuration_key_lines[] = "lines";
    constexpr static const char configuration_key_line_pose[] = "line_pose";
    constexpr static const char configuration_key_line_angle[] = "line_angle";
    constexpr static const char configuration_key_solver[] = "solver";
    constexpr static const char configuration_key_solver_precision[] = "solver_precision";
    constexpr static const char configuration_key_culling[] = "culling";
    constexpr static const char configuration_key_global_adjustment[] = "global_adjustment";
    constexpr static const char configuration_key_depth[] = "depth";
    constexpr static const char configuration_key_budget[] = "budget";
    constexpr static const char configuration_key_damping[] = "damping";
    constexpr static const char configuration_key_collisions[] = "collisions";
    constexpr static const char configuration_key_outliers[] = "outliers";
    constexpr static const char configuration_key_anchor[] = "anchor";
    constexpr static const char configuration_key_anchor_refresh[] = "anchor_refresh";
    constexpr static const char configuration_key_flow[] = "flow";
    constexpr static const char configuration_key_wavelet_window[] = "wavelet_window";
    constexpr static const char configuration_key_wavelet_levels[] = "wavelet_levels";
    constexpr static const char configuration_key_wavelet_robust[] = "wavelet_robust";
    constexpr static const char configuration_key_wavelet_undecimated[] = "wavelet_undecimated";
    constexpr static const char configuration_key_wavelet_seed[] = "wavelet_seed";

    constexpr static const char* const anchor_names[4] = { "translation", "affine", "translation_illumination", "affine_illumination" };

    const char* anchor_name(const mapping::frame::settings& frontend) {
        return frontend.anchored_patches ? anchor_names[static_cast<int>(frontend.anchor_model)] : "off";
    }

    constexpr static const char* const measure_names[5] = { "klt", "forstner", "harris", "rohr", "kenney" };

    const char* measure_name(const feature::score::structure_tensor::measure kind) {
        return measure_names[static_cast<int>(kind)];
    }

    const char* detector_name(const mapping::frame::settings& frontend) {
        return (frontend.detector == mapping::frame::settings::detector_kind::fast) ? "fast" : ((frontend.detector == mapping::frame::settings::detector_kind::mser) ? "mser" : measure_name(frontend.detector_measure));
    }

    const char* refiner_name(const mapping::frame::settings& frontend) {
        switch (frontend.refiner) {
            case mapping::frame::settings::refiner_kind::none:
                return "none";
            case mapping::frame::settings::refiner_kind::subpixel:
                return "subpixel";
            case mapping::frame::settings::refiner_kind::structure_tensor:
                return measure_name(frontend.refiner_measure);
        }
        return "subpixel";
    }

    int print_configuration(char* const buffer, const size_t capacity, const mapping::frame::settings& frontend) {
        char collisions[16];
        if (frontend.track_collision_distance > 0.0f) {
            std::snprintf(&collisions[0], sizeof(collisions), "%.9g", static_cast<double>(frontend.track_collision_distance));
        }
        else {
            std::snprintf(&collisions[0], sizeof(collisions), "off");
        }
        char global_adjustment[16];
        if (frontend.global_adjustment_keyframes > 0) {
            std::snprintf(&global_adjustment[0], sizeof(global_adjustment), "%d", frontend.global_adjustment_keyframes);
        }
        else {
            std::snprintf(&global_adjustment[0], sizeof(global_adjustment), "off");
        }
        char line_angle[16];
        if (frontend.line_angle > 0.0) {
            std::snprintf(&line_angle[0], sizeof(line_angle), "%.9g", frontend.line_angle);
        }
        else {
            std::snprintf(&line_angle[0], sizeof(line_angle), "off");
        }
        char anchor_refresh[16];
        if (frontend.anchor_refresh_error > 0.0f) {
            std::snprintf(&anchor_refresh[0], sizeof(anchor_refresh), "%.9g", static_cast<double>(frontend.anchor_refresh_error));
        }
        else {
            std::snprintf(&anchor_refresh[0], sizeof(anchor_refresh), "off");
        }
        return std::snprintf(
            buffer,
            capacity,
            "%s=%d\n%s=%s\n%s=%.9g\n%s=%s\n%s=%.9g\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%d\n%s=%s\n%s=%s\n%s=%s\n%s=%d\n%s=%d\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n%s=%s\n",
            configuration_key_verbosity,
            core::logger::get_verbosity(),
            configuration_key_detector,
            detector_name(frontend),
            configuration_key_detector_sigma,
            static_cast<double>(frontend.detector_sigma),
            configuration_key_refiner,
            refiner_name(frontend),
            configuration_key_refiner_sigma,
            static_cast<double>(frontend.refiner_sigma),
            configuration_key_tracker,
            (frontend.tracker == mapping::frame::settings::tracker_kind::extrema) ? "extrema" : "klt",
            configuration_key_association,
            (frontend.association == mapping::frame::settings::association_kind::match) ? "match" : ((frontend.association == mapping::frame::settings::association_kind::both) ? "both" : "klt"),
            configuration_key_descriptor,
            (frontend.descriptor == mapping::frame::settings::descriptor_kind::teblid) ? "teblid" : ((frontend.descriptor == mapping::frame::settings::descriptor_kind::bsift) ? "bsift" : "orb"),
            configuration_key_affine,
            frontend.affine ? "on" : "off",
            configuration_key_blur,
            frontend.blur_weighting ? "on" : "off",
            configuration_key_lines,
            frontend.lines ? "on" : "off",
            configuration_key_culling,
            frontend.cull_keyframes ? "on" : "off",
            configuration_key_global_adjustment,
            &global_adjustment[0],
            configuration_key_depth,
            frontend.inverse_depth ? "inverse" : "xyz",
            configuration_key_budget,
            frontend.fixed_budget ? "fixed" : "free",
            configuration_key_damping,
            frontend.klt_damped_steps ? "on" : "off",
            configuration_key_collisions,
            &collisions[0],
            configuration_key_outliers,
            frontend.pose_outlier_limit,
            configuration_key_anchor,
            anchor_name(frontend),
            configuration_key_anchor_refresh,
            &anchor_refresh[0],
            configuration_key_flow,
            (frontend.flow == feature::tracker::tracker::flow_kind::wavelet) ? "wavelet" : "intensity",
            configuration_key_wavelet_window,
            frontend.wavelet_half_window,
            configuration_key_wavelet_levels,
            frontend.wavelet_levels,
            configuration_key_wavelet_robust,
            frontend.wavelet_robust ? "on" : "off",
            configuration_key_wavelet_undecimated,
            frontend.wavelet_undecimated ? "on" : "off",
            configuration_key_wavelet_seed,
            (frontend.wavelet_seed == feature::tracker::tracker::wavelet_seed_kind::klt) ? "klt" : ((frontend.wavelet_seed == feature::tracker::tracker::wavelet_seed_kind::klt_fallback) ? "klt_fallback" : "rest"),
            configuration_key_line_pose,
            frontend.line_pose ? "on" : "off",
            configuration_key_line_angle,
            &line_angle[0],
            configuration_key_solver,
            (frontend.solver == optimisation::factor_graph::strategy::square_root) ? "square_root" : ((frontend.solver == optimisation::factor_graph::strategy::automatic) ? "automatic" : "dense_schur"),
            configuration_key_solver_precision,
            (frontend.solver_precision == optimisation::factor_graph::precision::single_precision) ? "single" : "double"
        );
    }

    bool token_equals(const char* const token, const size_t token_length, const char* const name) {
        return (token_length == std::strlen(name)) && (std::strncmp(token, name, token_length) == 0);
    }

    bool parse_configuration_int(const char* const token, const size_t token_length, int& value) {
        if ((token_length == 0) || (token_length > 10)) {
            return false;
        }
        long long parsed = 0;
        for (size_t index = 0; index < token_length; ++index) {
            if ((token[index] < '0') || (token[index] > '9')) {
                return false;
            }
            parsed = (parsed * 10) + (token[index] - '0');
        }
        if (parsed > 2147483647LL) {
            return false;
        }
        value = static_cast<int>(parsed);
        return true;
    }

    bool parse_configuration_decimal(const char* const token, const size_t token_length, const float maximum, float& value) {
        if ((token_length == 0) || (token_length > 16)) {
            return false;
        }
        double parsed = 0.0;
        double scale = 1.0;
        bool fraction = false;
        bool digits = false;
        for (size_t index = 0; index < token_length; ++index) {
            const char character = token[index];
            if ((character == '.') && !fraction) {
                fraction = true;
                continue;
            }
            if (((character == 'e') || (character == 'E')) && digits) {
                size_t exponent_index = index + 1;
                const bool negative = (exponent_index < token_length) && (token[exponent_index] == '-');
                if ((exponent_index < token_length) && ((token[exponent_index] == '-') || (token[exponent_index] == '+'))) {
                    ++exponent_index;
                }
                if (exponent_index == token_length) {
                    return false;
                }
                int exponent = 0;
                for (; exponent_index < token_length; ++exponent_index) {
                    if ((token[exponent_index] < '0') || (token[exponent_index] > '9')) {
                        return false;
                    }
                    exponent = math::min(99, (exponent * 10) + (token[exponent_index] - '0'));
                }
                for (int step = 0; step < exponent; ++step) {
                    parsed = negative ? (parsed * 0.1) : (parsed * 10.0);
                }
                break;
            }
            if ((character < '0') || (character > '9')) {
                return false;
            }
            digits = true;
            if (fraction) {
                scale *= 0.1;
                parsed += static_cast<double>(character - '0') * scale;
            }
            else {
                parsed = (parsed * 10.0) + static_cast<double>(character - '0');
            }
        }
        if (!digits || (parsed <= 0.0) || (parsed > static_cast<double>(maximum))) {
            return false;
        }
        value = static_cast<float>(parsed);
        return true;
    }

    bool parse_measure(const char* const token, const size_t token_length, feature::score::structure_tensor::measure& kind) {
        for (int index = 0; index < 5; ++index) {
            if (token_equals(token, token_length, measure_names[index])) {
                kind = static_cast<feature::score::structure_tensor::measure>(index);
                return true;
            }
        }
        return false;
    }

    bool parse_configuration(const char* const text, const size_t length, int& verbosity, mapping::frame::settings& frontend) {
        size_t index = 0;
        while (index < length) {
            size_t line_end = index;
            while ((line_end < length) && (text[line_end] != '\n') && (text[line_end] != '\r')) {
                ++line_end;
            }
            const char* const line = text + index;
            const size_t line_length = line_end - index;
            index = line_end;
            while ((index < length) && ((text[index] == '\n') || (text[index] == '\r'))) {
                ++index;
            }
            if (line_length == 0) {
                continue;
            }
            const char* const equals = static_cast<const char*>(std::memchr(line, '=', line_length));
            if (equals == nullptr) {
                return false;
            }
            const size_t key_length = static_cast<size_t>(equals - line);
            const char* const value = equals + 1;
            const size_t value_length = line_length - key_length - 1;
            if (token_equals(line, key_length, configuration_key_verbosity)) {
                if (!parse_configuration_int(value, value_length, verbosity)) {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_detector)) {
                if (token_equals(value, value_length, "fast")) {
                    frontend.detector = mapping::frame::settings::detector_kind::fast;
                }
                else if (token_equals(value, value_length, "mser")) {
                    frontend.detector = mapping::frame::settings::detector_kind::mser;
                }
                else if (parse_measure(value, value_length, frontend.detector_measure)) {
                    frontend.detector = mapping::frame::settings::detector_kind::structure_tensor;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_detector_sigma)) {
                if (!parse_configuration_decimal(value, value_length, feature::score::structure_tensor::sigma_maximum, frontend.detector_sigma)) {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_refiner)) {
                if (token_equals(value, value_length, "none")) {
                    frontend.refiner = mapping::frame::settings::refiner_kind::none;
                }
                else if (token_equals(value, value_length, "subpixel")) {
                    frontend.refiner = mapping::frame::settings::refiner_kind::subpixel;
                }
                else if (parse_measure(value, value_length, frontend.refiner_measure)) {
                    frontend.refiner = mapping::frame::settings::refiner_kind::structure_tensor;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_refiner_sigma)) {
                if (!parse_configuration_decimal(value, value_length, feature::refiner::structure_tensor::sigma_maximum, frontend.refiner_sigma)) {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_tracker)) {
                if (token_equals(value, value_length, "klt")) {
                    frontend.tracker = mapping::frame::settings::tracker_kind::klt;
                }
                else if (token_equals(value, value_length, "extrema")) {
                    frontend.tracker = mapping::frame::settings::tracker_kind::extrema;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_association)) {
                if (token_equals(value, value_length, "klt")) {
                    frontend.association = mapping::frame::settings::association_kind::klt;
                }
                else if (token_equals(value, value_length, "match")) {
                    frontend.association = mapping::frame::settings::association_kind::match;
                }
                else if (token_equals(value, value_length, "both")) {
                    frontend.association = mapping::frame::settings::association_kind::both;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_descriptor)) {
                if (token_equals(value, value_length, "orb")) {
                    frontend.descriptor = mapping::frame::settings::descriptor_kind::orb;
                }
                else if (token_equals(value, value_length, "teblid")) {
                    frontend.descriptor = mapping::frame::settings::descriptor_kind::teblid;
                }
                else if (token_equals(value, value_length, "bsift")) {
                    frontend.descriptor = mapping::frame::settings::descriptor_kind::bsift;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_affine)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.affine = false;
                }
                else if (token_equals(value, value_length, "on")) {
                    frontend.affine = true;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_blur)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.blur_weighting = false;
                }
                else if (token_equals(value, value_length, "on")) {
                    frontend.blur_weighting = true;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_lines)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.lines = false;
                }
                else if (token_equals(value, value_length, "on")) {
                    frontend.lines = true;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_culling)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.cull_keyframes = false;
                }
                else if (token_equals(value, value_length, "on")) {
                    frontend.cull_keyframes = true;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_depth)) {
                if (token_equals(value, value_length, "xyz")) {
                    frontend.inverse_depth = false;
                }
                else if (token_equals(value, value_length, "inverse")) {
                    frontend.inverse_depth = true;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_budget)) {
                if (token_equals(value, value_length, "free")) {
                    frontend.fixed_budget = false;
                }
                else if (token_equals(value, value_length, "fixed")) {
                    frontend.fixed_budget = true;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_damping)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.klt_damped_steps = false;
                }
                else if (token_equals(value, value_length, "on")) {
                    frontend.klt_damped_steps = true;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_collisions)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.track_collision_distance = 0.0f;
                }
                else if (!parse_configuration_decimal(value, value_length, 64.0f, frontend.track_collision_distance)) {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_outliers)) {
                if (!parse_configuration_int(value, value_length, frontend.pose_outlier_limit)) {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_global_adjustment)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.global_adjustment_keyframes = 0;
                }
                else if (!parse_configuration_int(value, value_length, frontend.global_adjustment_keyframes) || (frontend.global_adjustment_keyframes < 1)) {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_anchor)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.anchored_patches = false;
                }
                else {
                    size_t named = 4;
                    for (size_t anchor_index = 0; anchor_index < 4; ++anchor_index) {
                        if (token_equals(value, value_length, anchor_names[anchor_index])) {
                            named = anchor_index;
                            break;
                        }
                    }
                    if (named >= 4) {
                        return false;
                    }
                    frontend.anchored_patches = true;
                    frontend.anchor_model = static_cast<feature::tracker::patch_flow::model_kind>(named);
                }
            }
            else if (token_equals(line, key_length, configuration_key_anchor_refresh)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.anchor_refresh_error = 0.0f;
                }
                else if (!parse_configuration_decimal(value, value_length, 255.0f, frontend.anchor_refresh_error)) {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_flow)) {
                if (token_equals(value, value_length, "intensity")) {
                    frontend.flow = feature::tracker::tracker::flow_kind::intensity;
                }
                else if (token_equals(value, value_length, "wavelet")) {
                    frontend.flow = feature::tracker::tracker::flow_kind::wavelet;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_wavelet_window)) {
                int window = 0;
                if (!parse_configuration_int(value, value_length, window) || (window < 1) || (window > feature::tracker::wavelet_flow::maximum_half_window)) {
                    return false;
                }
                frontend.wavelet_half_window = window;
            }
            else if (token_equals(line, key_length, configuration_key_wavelet_levels)) {
                int levels = 0;
                if (!parse_configuration_int(value, value_length, levels) || (levels < 2) || (levels > 8)) {
                    return false;
                }
                frontend.wavelet_levels = levels;
            }
            else if (token_equals(line, key_length, configuration_key_wavelet_robust)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.wavelet_robust = false;
                }
                else if (token_equals(value, value_length, "on")) {
                    frontend.wavelet_robust = true;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_wavelet_undecimated)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.wavelet_undecimated = false;
                }
                else if (token_equals(value, value_length, "on")) {
                    frontend.wavelet_undecimated = true;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_line_pose)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.line_pose = false;
                }
                else if (token_equals(value, value_length, "on")) {
                    frontend.line_pose = true;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_line_angle)) {
                if (token_equals(value, value_length, "off")) {
                    frontend.line_angle = 0.0;
                }
                else {
                    float degrees = 0.0f;
                    if (!parse_configuration_decimal(value, value_length, 89.0f, degrees)) {
                        return false;
                    }
                    frontend.line_angle = static_cast<double>(degrees);
                }
            }
            else if (token_equals(line, key_length, configuration_key_solver)) {
                if (token_equals(value, value_length, "dense_schur")) {
                    frontend.solver = optimisation::factor_graph::strategy::dense_schur;
                }
                else if (token_equals(value, value_length, "square_root")) {
                    frontend.solver = optimisation::factor_graph::strategy::square_root;
                }
                else if (token_equals(value, value_length, "automatic")) {
                    frontend.solver = optimisation::factor_graph::strategy::automatic;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_solver_precision)) {
                if (token_equals(value, value_length, "double")) {
                    frontend.solver_precision = optimisation::factor_graph::precision::double_precision;
                }
                else if (token_equals(value, value_length, "single")) {
                    frontend.solver_precision = optimisation::factor_graph::precision::single_precision;
                }
                else {
                    return false;
                }
            }
            else if (token_equals(line, key_length, configuration_key_wavelet_seed)) {
                if (token_equals(value, value_length, "rest")) {
                    frontend.wavelet_seed = feature::tracker::tracker::wavelet_seed_kind::rest;
                }
                else if (token_equals(value, value_length, "klt")) {
                    frontend.wavelet_seed = feature::tracker::tracker::wavelet_seed_kind::klt;
                }
                else if (token_equals(value, value_length, "klt_fallback")) {
                    frontend.wavelet_seed = feature::tracker::tracker::wavelet_seed_kind::klt_fallback;
                }
                else {
                    return false;
                }
            }
            else {
                return false;
            }
        }
        return true;
    }
}

extern "C" {

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_configuration(zeroslam_system* system, char* configuration, int* length) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if (length == nullptr) {
        return zeroslam_return_failure_invalid_argument;
    }
    const int required = print_configuration(nullptr, 0, system->slam_instance.frontend);
    if ((configuration == nullptr) || (*length < required + 1)) {
        *length = required + 1;
        return zeroslam_return_failure_insufficient_data_length;
    }
    *length = print_configuration(configuration, static_cast<size_t>(required + 1), system->slam_instance.frontend);
    return zeroslam_return_success;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_set_configuration(zeroslam_system* system, const char* configuration, int length) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if ((configuration == nullptr) || (length < 0)) {
        return zeroslam_return_failure_invalid_argument;
    }
    int verbosity = core::logger::get_verbosity();
    mapping::frame::settings frontend = system->slam_instance.frontend;
    if (!parse_configuration(configuration, static_cast<size_t>(length), verbosity, frontend)) {
        return zeroslam_return_failure_invalid_configuration;
    }
    core::logger::set_verbosity(verbosity);
    system->slam_instance.frontend = frontend;
    return zeroslam_return_success;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_set_sensor_rig(zeroslam_system* system, const zeroslam_sensor_rig_struct* rig, int length) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if ((rig == nullptr) || (length <= 0)) {
        return zeroslam_return_failure_invalid_argument;
    }
    if (system->any_data) {
        return zeroslam_return_failure_invalid_rig_sensor;
    }
    std::vector<zeroslam_system::camera_entry> cameras;
    for (int i = 0; i < length; ++i) {
        const zeroslam_sensor_rig_struct& entry = rig[i];
        if (entry.sensor_id == 0) {
            return zeroslam_return_failure_invalid_rig_sensor;
        }
        if (entry.type != zeroslam_sensor_camera) {
            return zeroslam_return_failure_invalid_rig_sensor;
        }
        if ((entry.parameters_length != static_cast<int>(sizeof(zeroslam_sensor_parameters_camera_struct))) || (entry.parameters_data == nullptr)) {
            return zeroslam_return_failure_invalid_rig_sensor;
        }
        zeroslam_sensor_parameters_camera_struct parameters;
        std::memcpy(&parameters, entry.parameters_data, sizeof(parameters));
        if ((parameters.width <= 0) || (parameters.height <= 0) || !(parameters.focal_x > 0.0) || !(parameters.focal_y > 0.0)) {
            return zeroslam_return_failure_invalid_rig_sensor;
        }
        bool finite = math::isfinite(parameters.focal_x) && math::isfinite(parameters.focal_y) && math::isfinite(parameters.centre_x) && math::isfinite(parameters.centre_y);
        for (const double coefficient : parameters.distortion) {
            finite = finite && math::isfinite(coefficient);
        }
        if (!finite) {
            return zeroslam_return_failure_invalid_rig_sensor;
        }
        for (const zeroslam_system::camera_entry& existing : cameras) {
            if (existing.sensor_id == entry.sensor_id) {
                return zeroslam_return_failure_invalid_rig_sensor;
            }
        }
        cameras.push_back({ entry.sensor_id, parameters });
    }
    if (cameras.size() != 1) {
        return zeroslam_return_failure_invalid_rig_sensor;
    }
    system->cameras.swap(cameras);
    return zeroslam_return_success;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_set_sensor_data(zeroslam_system* system, const zeroslam_sensor_data_struct* data, int length) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if ((data == nullptr) || (length <= 0)) {
        return zeroslam_return_failure_invalid_argument;
    }
    for (int i = 0; i < length; ++i) {
        const zeroslam_sensor_data_struct& entry = data[i];
        if (entry.sensor_id == 0) {
            return zeroslam_return_failure_invalid_sensor_data;
        }
        if (system->any_data && (entry.timestamp <= system->latest_timestamp)) {
            return zeroslam_return_failure_invalid_sensor_data;
        }
        const zeroslam_system::camera_entry* camera = nullptr;
        for (const zeroslam_system::camera_entry& existing : system->cameras) {
            if (existing.sensor_id == entry.sensor_id) {
                camera = &existing;
                break;
            }
        }
        if (camera == nullptr) {
            return zeroslam_return_failure_invalid_sensor_data;
        }
        const size_t expected_length = static_cast<size_t>(camera->parameters.width) * static_cast<size_t>(camera->parameters.height);
        if ((entry.measurement_data == nullptr) || (static_cast<size_t>(entry.measurement_length) != expected_length)) {
            return zeroslam_return_failure_invalid_sensor_data;
        }
        const image::image frame_image(static_cast<size_t>(camera->parameters.height), static_cast<size_t>(camera->parameters.width), static_cast<unsigned char*>(entry.measurement_data));
        const double camera_parameters[sensor::model::parameter_count] = {
            camera->parameters.focal_x,
            camera->parameters.focal_y,
            camera->parameters.centre_x,
            camera->parameters.centre_y,
            camera->parameters.distortion[0],
            camera->parameters.distortion[1],
            camera->parameters.distortion[2],
            camera->parameters.distortion[3],
            camera->parameters.distortion[4],
            camera->parameters.distortion[5],
            camera->parameters.distortion[6],
            camera->parameters.distortion[7]
        };
        const sensor::model intrinsics(&camera_parameters[0], sensor::model::parameter_count);
        system->slam_instance.process_frame(intrinsics, frame_image);
        system->frames.push_back({ entry.timestamp, system->slam_instance.reconstruction.next_frame_id - 1 });
        system->latest_timestamp = entry.timestamp;
        system->any_data = true;
    }
    return zeroslam_return_success;
}
}

namespace {
    inline void fill_pose(const mapping::frame& frame, const long long int timestamp, zeroslam_pose_struct* pose) {
        const math::matrix<double, 3, 3> camera_to_world = math::transpose(frame.rotation);
        const math::matrix<double, 3, 1> centre = -camera_to_world * frame.translation;
        const math::matrix<double, 4, 1> quaternion = math::so3<double>(camera_to_world).get_quaternion();
        pose->timestamp = timestamp;
        pose->pose[0] = (centre[0]);
        pose->pose[1] = (centre[1]);
        pose->pose[2] = (centre[2]);
        pose->pose[3] = (quaternion[1]);
        pose->pose[4] = (quaternion[2]);
        pose->pose[5] = (quaternion[3]);
        pose->pose[6] = (quaternion[0]);
        for (int i = 0; i < 7 * 7; ++i) {
            pose->covariance[i] = 0.0;
        }
    }
}

extern "C" {

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_finalise(zeroslam_system* system) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    system->slam_instance.finalise();
    return zeroslam_return_success;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_pose(zeroslam_system* system, zeroslam_pose_struct* pose) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if (pose == nullptr) {
        return zeroslam_return_failure_invalid_argument;
    }
    for (size_t i = system->frames.size(); i > 0; --i) {
        const zeroslam_system::frame_record& record = system->frames[i - 1];
        const std::unordered_map<int, mapping::frame>::const_iterator found = system->slam_instance.reconstruction.frames.find(record.frame_id);
        if (found != system->slam_instance.reconstruction.frames.end()) {
            fill_pose(found->second, record.timestamp, pose);
            return zeroslam_return_success;
        }
    }
    return zeroslam_return_failure_invalid_argument;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_pose_at_timestamp(zeroslam_system* system, zeroslam_pose_struct* pose, long long int timestamp) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if (pose == nullptr) {
        return zeroslam_return_failure_invalid_argument;
    }
    for (const zeroslam_system::frame_record& record : system->frames) {
        if (record.timestamp != timestamp) {
            continue;
        }
        const std::unordered_map<int, mapping::frame>::const_iterator found = system->slam_instance.reconstruction.frames.find(record.frame_id);
        if (found == system->slam_instance.reconstruction.frames.end()) {
            return zeroslam_return_failure_invalid_argument;
        }
        fill_pose(found->second, record.timestamp, pose);
        return zeroslam_return_success;
    }
    return zeroslam_return_failure_invalid_argument;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_chunk(zeroslam_system* system, float x, float y, float z, zeroslam_map_chunk_struct* chunk) {
    static_cast<void>(x);
    static_cast<void>(y);
    static_cast<void>(z);
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if (chunk == nullptr) {
        return zeroslam_return_failure_invalid_argument;
    }
    const std::unordered_map<int, mapping::point>& landmarks = system->slam_instance.reconstruction.landmarks;
    const int required = static_cast<int>(landmarks.size());
    if ((required > 0) && ((chunk->points == nullptr) || (chunk->points_length < required))) {
        chunk->points_length = required;
        return zeroslam_return_failure_insufficient_data_length;
    }
    std::vector<int> ids;
    ids.reserve(landmarks.size());
    for (const auto& [id, landmark] : landmarks) {
        static_cast<void>(landmark);
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    chunk->timestamp = system->latest_timestamp;
    chunk->min_x = chunk->min_y = chunk->min_z = 0.0f;
    chunk->max_x = chunk->max_y = chunk->max_z = 0.0f;
    int written = 0;
    for (const int id : ids) {
        const mapping::point& landmark = landmarks.at(id);
        zeroslam_point_struct& point = chunk->points[written];
        point.x = static_cast<float>(landmark.location[0]);
        point.y = static_cast<float>(landmark.location[1]);
        point.z = static_cast<float>(landmark.location[2]);
        point.confidence = 1.0f;
        point.r = static_cast<float>(landmark.colour[0]);
        point.g = static_cast<float>(landmark.colour[1]);
        point.b = static_cast<float>(landmark.colour[2]);
        point.a = 1.0f;
        if (written == 0) {
            chunk->min_x = chunk->max_x = point.x;
            chunk->min_y = chunk->max_y = point.y;
            chunk->min_z = chunk->max_z = point.z;
        }
        else {
            chunk->min_x = math::min(chunk->min_x, point.x);
            chunk->min_y = math::min(chunk->min_y, point.y);
            chunk->min_z = math::min(chunk->min_z, point.z);
            chunk->max_x = math::max(chunk->max_x, point.x);
            chunk->max_y = math::max(chunk->max_y, point.y);
            chunk->max_z = math::max(chunk->max_z, point.z);
        }
        ++written;
    }
    chunk->points_length = written;
    return zeroslam_return_success;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_lines(zeroslam_system* system, zeroslam_map_lines_struct* lines) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if (lines == nullptr) {
        return zeroslam_return_failure_invalid_argument;
    }
    const std::unordered_map<int, mapping::line>& landmarks = system->slam_instance.reconstruction.line_landmarks;
    const int required = static_cast<int>(landmarks.size());
    if ((required > 0) && ((lines->lines == nullptr) || (lines->lines_length < required))) {
        lines->lines_length = required;
        return zeroslam_return_failure_insufficient_data_length;
    }
    std::vector<int> ids;
    ids.reserve(landmarks.size());
    for (const auto& [id, landmark] : landmarks) {
        static_cast<void>(landmark);
        ids.push_back(id);
    }
    std::sort(ids.begin(), ids.end());
    lines->timestamp = system->latest_timestamp;
    lines->reserved = 0;
    int written = 0;
    for (const int id : ids) {
        const mapping::line& landmark = landmarks.at(id);
        zeroslam_line_struct& line = lines->lines[written];
        line.x1 = static_cast<float>(landmark.locations[0][0]);
        line.y1 = static_cast<float>(landmark.locations[0][1]);
        line.z1 = static_cast<float>(landmark.locations[0][2]);
        line.x2 = static_cast<float>(landmark.locations[1][0]);
        line.y2 = static_cast<float>(landmark.locations[1][1]);
        line.z2 = static_cast<float>(landmark.locations[1][2]);
        line.confidence = 1.0f;
        line.r = 1.0f;
        line.g = 1.0f;
        line.b = 1.0f;
        line.a = 1.0f;
        ++written;
    }
    lines->lines_length = written;
    return zeroslam_return_success;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_edges(zeroslam_system* system, zeroslam_map_edges_struct* edges) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if (edges == nullptr) {
        return zeroslam_return_failure_invalid_argument;
    }
    std::unordered_map<int, long long int> timestamps;
    timestamps.reserve(system->frames.size());
    for (const zeroslam_system::frame_record& record : system->frames) {
        timestamps[record.frame_id] = record.timestamp;
    }
    std::vector<zeroslam_edge_struct> found;
    for (const mapping::covisibility::edge& edge : system->slam_instance.covisibility().edges()) {
        const std::unordered_map<int, long long int>::const_iterator a = timestamps.find(edge.frame_a);
        const std::unordered_map<int, long long int>::const_iterator b = timestamps.find(edge.frame_b);
        if ((a == timestamps.end()) || (b == timestamps.end())) {
            continue;
        }
        found.push_back(zeroslam_edge_struct{ a->second, b->second, static_cast<int>(zeroslam_edge_covisibility), edge.weight });
    }
    for (const slam::verified_loop& loop : system->slam_instance.verified_loops) {
        const std::unordered_map<int, long long int>::const_iterator a = timestamps.find(loop.loop.keyframe_id);
        const std::unordered_map<int, long long int>::const_iterator b = timestamps.find(loop.keyframe_id);
        if ((a == timestamps.end()) || (b == timestamps.end())) {
            continue;
        }
        found.push_back(zeroslam_edge_struct{ a->second, b->second, static_cast<int>(zeroslam_edge_loop), static_cast<int>(loop.loop.inliers) });
    }
    const int required = static_cast<int>(found.size());
    if ((required > 0) && ((edges->edges == nullptr) || (edges->edges_length < required))) {
        edges->edges_length = required;
        return zeroslam_return_failure_insufficient_data_length;
    }
    edges->timestamp = system->latest_timestamp;
    edges->reserved = 0;
    for (int i = 0; i < required; ++i) {
        edges->edges[i] = found[static_cast<size_t>(i)];
    }
    edges->edges_length = required;
    return zeroslam_return_success;
}

zeroslam_return_enum ZEROSLAM_API_CALL zeroslam_get_map_keyframes(zeroslam_system* system, zeroslam_map_keyframes_struct* keyframes) {
    if (system == nullptr) {
        return zeroslam_return_failure_invalid_system;
    }
    if (keyframes == nullptr) {
        return zeroslam_return_failure_invalid_argument;
    }
    std::vector<long long int> found;
    for (const zeroslam_system::frame_record& record : system->frames) {
        if ((system->slam_instance.keyframe_ids().count(record.frame_id) != 0) && (system->slam_instance.reconstruction.frames.count(record.frame_id) != 0)) {
            found.push_back(record.timestamp);
        }
    }
    std::sort(found.begin(), found.end());
    const int required = static_cast<int>(found.size());
    if ((required > 0) && ((keyframes->keyframes == nullptr) || (keyframes->keyframes_length < required))) {
        keyframes->keyframes_length = required;
        return zeroslam_return_failure_insufficient_data_length;
    }
    keyframes->timestamp = system->latest_timestamp;
    keyframes->reserved = 0;
    for (int i = 0; i < required; ++i) {
        keyframes->keyframes[i] = found[static_cast<size_t>(i)];
    }
    keyframes->keyframes_length = required;
    return zeroslam_return_success;
}
}
