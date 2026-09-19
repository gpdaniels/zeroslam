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
#ifndef ZEROSLAM_TOOLS_GUI_UI_HPP
#define ZEROSLAM_TOOLS_GUI_UI_HPP

#include "imgui.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdio>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace ui {

    struct input {
        int mouse_x = 0;
        int mouse_y = 0;
        bool mouse_down = false;
        bool clicked = false;
        int press_x = 0;
        int press_y = 0;
    };

    struct theme {
        gtl::color background = gtl::color::rgb(25, 25, 30);
        gtl::color separator = gtl::color::rgb(60, 60, 70);
        gtl::color header = gtl::color::rgb(160, 160, 180);
        gtl::color text = gtl::color::rgb(200, 200, 200);
        gtl::color text_dim = gtl::color::rgb(150, 150, 170);
        gtl::color text_faint = gtl::color::rgb(110, 110, 130);
        gtl::color control = gtl::color::rgb(45, 45, 55);
        gtl::color control_hover = gtl::color::rgb(60, 60, 70);
        gtl::color control_edge = gtl::color::rgb(80, 80, 90);
        gtl::color track = gtl::color::rgb(35, 35, 45);
        gtl::color track_fill = gtl::color::rgb(45, 70, 110);
        gtl::color accent = gtl::color::rgb(100, 150, 255);
        gtl::color positive = gtl::color::rgb(50, 90, 50);
        gtl::color positive_hover = gtl::color::rgb(70, 120, 70);
        gtl::color negative = gtl::color::rgb(90, 40, 40);
        gtl::color negative_hover = gtl::color::rgb(120, 50, 50);
        int row_height = 18;
        int text_height = 14;
        int control_height = 20;
        int slider_height = 14;
        int gap = 4;
    };

    class panel final {
    private:
        gtl::imgui_renderer* renderer = nullptr;
        input pointer;
        theme colours;
        int origin_x = 0;
        int origin_y = 0;
        int panel_width = 0;
        int cursor_y = 0;
        int inset = 8;

        bool hovered(const gtl::rect& area) const {
            return area.contains(this->pointer.mouse_x, this->pointer.mouse_y);
        }

        bool pressed(const gtl::rect& area) const {
            return this->pointer.clicked && area.contains(this->pointer.press_x, this->pointer.press_y);
        }

        bool dragging(const gtl::rect& area) const {
            return this->pointer.mouse_down && this->hovered(area);
        }

        void draw_button(const gtl::rect& area, const gtl::color& fill, const char* const text, const bool enabled) {
            this->renderer->draw_rect(area, fill);
            this->renderer->draw_frame(area, this->colours.control_edge);
            const int text_x = area.x + ((area.w - this->renderer->text_width(text)) / 2);
            this->renderer->draw_text(text, text_x, area.y + 4, enabled ? this->colours.text : this->colours.text_faint);
        }

    public:
        panel(gtl::imgui_renderer* const target, const input& pointer_state, const int x, const int y, const int width)
            : renderer(target)
            , pointer(pointer_state)
            , origin_x(x)
            , origin_y(y)
            , panel_width(width)
            , cursor_y(y) {
        }

        int content_x() const {
            return this->origin_x + this->inset;
        }

        int content_width() const {
            return this->panel_width - (this->inset * 2);
        }

        int cursor() const {
            return this->cursor_y;
        }

        void space(const int pixels) {
            this->cursor_y += pixels;
        }

        void fill(const int height) {
            this->renderer->draw_rect({ this->origin_x, this->origin_y, this->panel_width, height }, this->colours.background);
        }

        void header(const char* const title) {
            this->renderer->draw_rect({ this->content_x() - 2, this->cursor_y, this->content_width() + 4, 1 }, this->colours.separator);
            this->cursor_y += this->colours.gap;
            this->renderer->draw_text(title, this->content_x(), this->cursor_y, this->colours.header);
            this->cursor_y += this->colours.row_height;
        }

        void label(const char* const text) {
            this->renderer->draw_text(text, this->content_x(), this->cursor_y, this->colours.text);
            this->cursor_y += this->colours.text_height;
        }

        void label_dim(const char* const text) {
            this->renderer->draw_text(text, this->content_x(), this->cursor_y, this->colours.text_faint);
            this->cursor_y += this->colours.text_height;
        }

        void value(const char* const name, const char* const text) {
            this->renderer->draw_text(name, this->content_x(), this->cursor_y, this->colours.text_dim);
            const int text_x = this->content_x() + this->content_width() - this->renderer->text_width(text);
            this->renderer->draw_text(text, text_x, this->cursor_y, this->colours.text);
            this->cursor_y += this->colours.text_height;
        }

        void value_number(const char* const name, const double number, const char* const format) {
            char text[64] = {};
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
            std::snprintf(&text[0], sizeof(text), format, number);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
            this->value(name, &text[0]);
        }

        enum class button_kind {
            neutral,
            positive,
            negative
        };

        bool button(const char* const text, const button_kind kind = button_kind::neutral, const bool enabled = true) {
            const gtl::rect area = { this->content_x(), this->cursor_y, this->content_width(), this->colours.control_height };
            const bool is_hovered = enabled && this->hovered(area);
            gtl::color fill = is_hovered ? this->colours.control_hover : this->colours.control;
            if (kind == button_kind::positive) {
                fill = is_hovered ? this->colours.positive_hover : this->colours.positive;
            }
            else if (kind == button_kind::negative) {
                fill = is_hovered ? this->colours.negative_hover : this->colours.negative;
            }
            this->draw_button(area, fill, text, enabled);
            this->cursor_y += area.h + this->colours.gap;
            return enabled && this->pressed(area);
        }

        int button_row(const char* const* const texts, const int count, const int active_index = -1, const bool enabled = true) {
            const int spacing = this->colours.gap;
            const int width = (this->content_width() - (spacing * (count - 1))) / count;
            int result = -1;
            for (int i = 0; i < count; ++i) {
                const gtl::rect area = { this->content_x() + (i * (width + spacing)), this->cursor_y, width, this->colours.control_height };
                const bool is_hovered = enabled && this->hovered(area);
                const bool active = (i == active_index);
                const gtl::color fill = is_hovered ? (active ? this->colours.positive_hover : this->colours.control_hover) : (active ? this->colours.positive : this->colours.control);
                this->draw_button(area, fill, texts[i], enabled);
                if (enabled && this->pressed(area)) {
                    result = i;
                }
            }
            this->cursor_y += this->colours.control_height + this->colours.gap;
            return result;
        }

        bool checkbox(const char* const text, bool* const value, const bool enabled = true) {
            const int box = 12;
            const gtl::rect row = { this->content_x(), this->cursor_y, this->content_width(), box };
            const gtl::rect square = { row.x, row.y, box, box };
            const bool is_hovered = enabled && this->hovered(row);
            this->renderer->draw_rect(square, is_hovered ? this->colours.control_hover : this->colours.track);
            this->renderer->draw_frame(square, this->colours.control_edge);
            if (*value) {
                this->renderer->draw_rect({ row.x + 2, row.y + 2, box - 4, box - 4 }, enabled ? this->colours.accent : this->colours.text_faint);
            }
            this->renderer->draw_text(text, row.x + box + 4, row.y + 1, enabled ? this->colours.text : this->colours.text_faint);
            this->cursor_y += box + this->colours.gap;
            if (enabled && this->pressed(row)) {
                *value = !*value;
                return true;
            }
            return false;
        }

        bool slider(const char* const name, float* const value, const float minimum, const float maximum, const char* const format, const int name_width = 56) {
            const int x = this->content_x();
            const int y = this->cursor_y;
            char readout[32] = {};
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
            std::snprintf(&readout[0], sizeof(readout), format, static_cast<double>(*value));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
            const int readout_width = this->renderer->text_width(&readout[0]) + 6;
            int track_width = this->content_width() - name_width - readout_width;
            if (track_width < 16) {
                track_width = 16;
            }
            const gtl::rect track = { x + name_width, y, track_width, this->colours.slider_height };
            this->renderer->draw_text(name, x, y + 2, this->colours.text_dim);
            this->renderer->draw_rect(track, this->colours.track);
            this->renderer->draw_frame(track, this->colours.control_edge);
            const float span = (maximum > minimum) ? (maximum - minimum) : 1.0f;
            float fraction = (*value - minimum) / span;
            fraction = (fraction < 0.0f) ? 0.0f : ((fraction > 1.0f) ? 1.0f : fraction);
            const int knob_x = track.x + 1 + static_cast<int>(fraction * static_cast<float>(track.w - 8));
            this->renderer->draw_rect({ knob_x, y + 1, 6, track.h - 2 }, this->colours.accent);
            this->renderer->draw_text(&readout[0], track.x + track.w + 4, y + 2, this->colours.text);
            this->cursor_y += track.h + this->colours.gap;
            if (this->dragging(track)) {
                float target = static_cast<float>(this->pointer.mouse_x - track.x) / static_cast<float>(track.w);
                target = (target < 0.0f) ? 0.0f : ((target > 1.0f) ? 1.0f : target);
                const float updated = minimum + (target * span);
                if (updated != *value) {
                    *value = updated;
                    return true;
                }
            }
            return false;
        }

        bool scrubber(double* const fraction, const char* const readout) {
            const gtl::rect track = { this->content_x(), this->cursor_y, this->content_width(), this->colours.slider_height };
            this->renderer->draw_rect(track, this->colours.track);
            this->renderer->draw_frame(track, this->colours.control_edge);
            double clamped = *fraction;
            clamped = (clamped < 0.0) ? 0.0 : ((clamped > 1.0) ? 1.0 : clamped);
            const int filled = static_cast<int>(clamped * static_cast<double>(track.w - 2));
            this->renderer->draw_rect({ track.x + 1, track.y + 1, filled, track.h - 2 }, this->colours.track_fill);
            const int knob_x = track.x + 1 + static_cast<int>(clamped * static_cast<double>(track.w - 8));
            this->renderer->draw_rect({ knob_x, track.y + 1, 6, track.h - 2 }, this->colours.accent);
            this->cursor_y += track.h + 2;
            if (readout != nullptr) {
                this->renderer->draw_text(readout, track.x, this->cursor_y, this->colours.text_dim);
                this->cursor_y += this->colours.text_height;
            }
            if (this->dragging(track)) {
                double target = static_cast<double>(this->pointer.mouse_x - track.x) / static_cast<double>(track.w);
                target = (target < 0.0) ? 0.0 : ((target > 1.0) ? 1.0 : target);
                if (target != *fraction) {
                    *fraction = target;
                    return true;
                }
            }
            return false;
        }

        bool text_input(gtl::imgui& editor, const char* const label, char* const buffer, const int capacity) {
            const gtl::style& style = editor.get_style();
            const int height = style.font_height + style.padding;
            editor.update_configuration(this->content_x(), this->cursor_y, this->content_width() + (style.padding * 2), height);
            editor.reset_layout();
            const bool changed = editor.widget_input_text(label, buffer, capacity);
            this->cursor_y += height + this->colours.gap;
            return changed;
        }
    };
}

#endif // ZEROSLAM_TOOLS_GUI_UI_HPP
