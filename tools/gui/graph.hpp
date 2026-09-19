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

#pragma once
#ifndef ZEROSLAM_TOOLS_GUI_GRAPH_HPP
#define ZEROSLAM_TOOLS_GUI_GRAPH_HPP

#include "imgui.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstddef>
#include <cstdio>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace graph {

    struct point {
        double time_seconds = 0.0;
        float value = 0.0f;
    };

    struct series {
        const char* name = nullptr;
        gtl::color colour = gtl::color::rgb(100, 150, 255);
        const point* data = nullptr;
        std::size_t count = 0;
    };

    struct axes {
        double time_minimum = 0.0;
        double time_maximum = 1.0;
        float value_minimum = 0.0f;
        float value_maximum = 0.0f;
        const char* units = nullptr;
        bool mark_zero = true;
        bool non_negative = false;
        bool mark_time = false;
        double marked_time_seconds = 0.0;
    };

    class history final {
    private:
        std::vector<point> points;
        std::size_t capacity_limit = 2048;

    public:
        explicit history(const std::size_t capacity = 2048)
            : capacity_limit(capacity < 2 ? 2 : capacity) {
            this->points.reserve(this->capacity_limit);
        }

        void clear() {
            this->points.clear();
        }

        void push(const double time_seconds, const float value) {
            if (this->points.size() >= this->capacity_limit) {
                this->points.erase(this->points.begin(), this->points.begin() + static_cast<std::ptrdiff_t>(this->points.size() / 2));
            }
            this->points.push_back({ time_seconds, value });
        }

        void push_unique(const double time_seconds, const float value) {
            if ((!this->points.empty()) && (this->points.back().time_seconds >= time_seconds)) {
                this->points.back().value = value;
                return;
            }
            this->push(time_seconds, value);
        }

        series as_series(const char* const name, const gtl::color colour) const {
            series result;
            result.name = name;
            result.colour = colour;
            result.data = this->points.data();
            result.count = this->points.size();
            return result;
        }
    };

    inline double latest_time(const series* const list, const std::size_t series_count, const double fallback) {
        double latest = 0.0;
        bool any = false;
        for (std::size_t s = 0; s < series_count; ++s) {
            if (list[s].count == 0) {
                continue;
            }
            const double last = list[s].data[list[s].count - 1].time_seconds;
            if ((!any) || (last > latest)) {
                latest = last;
                any = true;
            }
        }
        return any ? latest : fallback;
    }

    inline int draw(
        gtl::imgui_renderer* const renderer,
        const int x,
        const int y,
        const int width,
        const int height,
        const char* const title,
        const series* const list,
        const std::size_t series_count,
        const axes& bounds
    ) {
        if ((renderer == nullptr) || (width < 24) || (height < 24)) {
            return 0;
        }
        const gtl::color frame_colour = gtl::color::rgb(70, 70, 82);
        const gtl::color panel_colour = gtl::color::rgb(18, 18, 22);
        const gtl::color text_colour = gtl::color::rgb(190, 190, 200);
        const gtl::color faint_colour = gtl::color::rgb(70, 70, 85);

        renderer->draw_rect({ x, y, width, height }, panel_colour);
        renderer->draw_frame({ x, y, width, height }, frame_colour);

        float minimum = bounds.value_minimum;
        float maximum = bounds.value_maximum;
        if (!(maximum > minimum)) {
            bool any = false;
            for (std::size_t s = 0; s < series_count; ++s) {
                for (std::size_t i = 0; i < list[s].count; ++i) {
                    const float value = list[s].data[i].value;
                    if (!any) {
                        minimum = value;
                        maximum = value;
                        any = true;
                        continue;
                    }
                    minimum = (value < minimum) ? value : minimum;
                    maximum = (value > maximum) ? value : maximum;
                }
            }
            if (!any) {
                minimum = 0.0f;
                maximum = 1.0f;
            }
            if (bounds.mark_zero) {
                minimum = (minimum > 0.0f) ? 0.0f : minimum;
                maximum = (maximum < 0.0f) ? 0.0f : maximum;
            }
            const float span = maximum - minimum;
            const float padding = (span > 0.0f) ? (span * 0.1f) : 1.0f;
            minimum -= padding;
            maximum += padding;
            if (bounds.non_negative && (minimum < 0.0f)) {
                minimum = 0.0f;
            }
        }

        const int title_height = 14;
        const int plot_x = x + 4;
        const int plot_y = y + title_height + 2;
        const int plot_width = width - 8;
        const int plot_height = height - title_height - 8;
        if ((plot_width < 8) || (plot_height < 8)) {
            return height;
        }

        {
            char text[160] = {};
            if (bounds.units != nullptr) {
                std::snprintf(&text[0], sizeof(text), "%s  [%.3g, %.3g] %s", title, static_cast<double>(minimum), static_cast<double>(maximum), bounds.units);
            }
            else {
                std::snprintf(&text[0], sizeof(text), "%s  [%.3g, %.3g]", title, static_cast<double>(minimum), static_cast<double>(maximum));
            }
            renderer->draw_text(&text[0], x + 4, y + 2, text_colour);
        }

        const double time_span = (bounds.time_maximum > bounds.time_minimum) ? (bounds.time_maximum - bounds.time_minimum) : 1.0;
        const float value_span = (maximum > minimum) ? (maximum - minimum) : 1.0f;
        const auto to_pixel_x = [&](const double time_seconds) {
            double fraction = (time_seconds - bounds.time_minimum) / time_span;
            fraction = (fraction < 0.0) ? 0.0 : ((fraction > 1.0) ? 1.0 : fraction);
            return plot_x + static_cast<int>(fraction * static_cast<double>(plot_width - 1));
        };
        const auto to_pixel_y = [&](const float value) {
            float fraction = (value - minimum) / value_span;
            fraction = (fraction < 0.0f) ? 0.0f : ((fraction > 1.0f) ? 1.0f : fraction);
            return plot_y + plot_height - 1 - static_cast<int>(fraction * static_cast<float>(plot_height - 1));
        };

        if (bounds.mark_zero && (minimum < 0.0f) && (maximum > 0.0f)) {
            const int zero_y = to_pixel_y(0.0f);
            renderer->draw_rect({ plot_x, zero_y, plot_width, 1 }, faint_colour);
        }
        if (bounds.mark_time) {
            const int time_x = to_pixel_x(bounds.marked_time_seconds);
            renderer->draw_rect({ time_x, plot_y, 1, plot_height }, gtl::color::rgb(120, 100, 60));
        }

        {
            char span[64] = {};
            std::snprintf(&span[0], sizeof(span), "%.1f-%.1fs", bounds.time_minimum, bounds.time_maximum);
            renderer->draw_text(&span[0], x + width - 4 - renderer->text_width(&span[0]), y + height - 12, gtl::color::rgb(120, 120, 135));
        }

        int legend_x = plot_x;
        for (std::size_t s = 0; s < series_count; ++s) {
            const series& current = list[s];
            for (std::size_t i = 1; i < current.count; ++i) {
                const point& previous = current.data[i - 1];
                const point& next = current.data[i];
                if ((next.time_seconds < bounds.time_minimum) || (previous.time_seconds > bounds.time_maximum)) {
                    continue;
                }
                renderer->draw_line(to_pixel_x(previous.time_seconds), to_pixel_y(previous.value), to_pixel_x(next.time_seconds), to_pixel_y(next.value), current.colour);
            }
            if ((current.name != nullptr) && (current.count > 0)) {
                renderer->draw_rect({ legend_x, y + height - 6, 8, 3 }, current.colour);
                renderer->draw_text(current.name, legend_x + 10, y + height - 12, current.colour);
                legend_x += 10 + renderer->text_width(current.name) + 8;
            }
        }
        return height;
    }
}

#endif // ZEROSLAM_TOOLS_GUI_GRAPH_HPP
