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
#ifndef ZEROSLAM_TOOLS_GUI_IMGUI_HPP
#define ZEROSLAM_TOOLS_GUI_IMGUI_HPP

#include "clipboard.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace gtl {

    struct color {
        uint8_t r, g, b, a;

        static color white() {
            return { 255, 255, 255, 255 };
        }

        static color black() {
            return { 0, 0, 0, 255 };
        }

        static color gray(uint8_t v) {
            return { v, v, v, 255 };
        }

        static color rgb(uint8_t r, uint8_t g, uint8_t b) {
            return { r, g, b, 255 };
        }

        static color rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
            return { r, g, b, a };
        }
    };

    struct rect {
        int x, y, w, h;

        bool contains(int px, int py) const {
            return px >= x && px < x + w && py >= y && py < y + h;
        }
    };

    struct style {
        color background = color::rgb(32, 32, 32);
        color foreground = color::white();
        color border = color::rgb(100, 100, 100);
        color hover = color::rgb(60, 60, 60);
        color active = color::rgb(80, 80, 80);
        color text = color::white();
        color text_disabled = color::gray(100);
        color button = color::rgb(50, 50, 50);
        color button_hover = color::rgb(70, 70, 70);
        color button_active = color::rgb(90, 90, 90);
        color slider_knob = color::rgb(100, 150, 255);
        color tab = color::rgb(40, 40, 40);
        color tab_active = color::rgb(60, 60, 80);
        int padding = 8;
        int spacing = 8;
        int indent = 16;
        int font_height = 16;
    };

    class imgui_renderer {
    public:
        virtual ~imgui_renderer() = default;
        virtual void draw_rect(const rect& r, const color& c) = 0;
        virtual void draw_text(const char* text, int x, int y, const color& c) = 0;
        virtual void draw_line(int x1, int y1, int x2, int y2, const color& c) = 0;
        virtual int text_width(const char* text) = 0;
        virtual int text_height() = 0;

        void draw_frame(const rect& r, const color& c) {
            this->draw_rect({ r.x, r.y, r.w, 1 }, c);
            this->draw_rect({ r.x, r.y + r.h - 1, r.w, 1 }, c);
            this->draw_rect({ r.x, r.y, 1, r.h }, c);
            this->draw_rect({ r.x + r.w - 1, r.y, 1, r.h }, c);
        }
    };

    namespace imgui_detail {
        inline int utf8_decode(const char*& s) {
            unsigned char c = static_cast<unsigned char>(*s);
            if (c < 0x80) {
                ++s;
                return c;
            }
            int codepoint = 0;
            int bytes = 0;
            if ((c & 0xE0) == 0xC0) {
                bytes = 2;
                codepoint = c & 0x1F;
            }
            else if ((c & 0xF0) == 0xE0) {
                bytes = 3;
                codepoint = c & 0x0F;
            }
            else if ((c & 0xF8) == 0xF0) {
                bytes = 4;
                codepoint = c & 0x07;
            }
            else {
                ++s;
                return c;
            }
            for (int i = 1; i < bytes; ++i) {
                if ((static_cast<unsigned char>(s[i]) & 0xC0) != 0x80) {
                    s += i;
                    return 0xFFFD;
                }
                codepoint = (codepoint << 6) | (static_cast<unsigned char>(s[i]) & 0x3F);
            }
            s += bytes;
            return codepoint;
        }

        inline constexpr unsigned char font_glyphs[128][12] = {
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x7E, 0xC3, 0x81, 0xA5, 0x81, 0xBD, 0x99, 0xC3, 0x7E, 0x00, 0x00 },
            { 0x00, 0x7E, 0xFF, 0xFF, 0xDB, 0xFF, 0xC3, 0xE7, 0xFF, 0x7E, 0x00, 0x00 },
            { 0x00, 0x00, 0x44, 0xEE, 0xFE, 0xFE, 0xFE, 0x7C, 0x38, 0x10, 0x00, 0x00 },
            { 0x00, 0x10, 0x38, 0x7C, 0xFE, 0xFE, 0x7C, 0x38, 0x10, 0x00, 0x00, 0x00 },
            { 0x00, 0x18, 0x3C, 0x3C, 0xFF, 0xE7, 0xE7, 0x18, 0x18, 0x7E, 0x00, 0x00 },
            { 0x00, 0x18, 0x3C, 0x7E, 0xFF, 0xFF, 0x7E, 0x18, 0x18, 0x7E, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x3E, 0x0E, 0x3A, 0x72, 0xF8, 0xCC, 0xCC, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0x3C, 0x66, 0x66, 0x66, 0x3C, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x7F, 0x63, 0x7F, 0x63, 0x63, 0x63, 0x67, 0xE7, 0xE6, 0xC0, 0x00 },
            { 0x00, 0x00, 0x18, 0xDB, 0x7E, 0xE7, 0xE7, 0x7E, 0xDB, 0x18, 0x00, 0x00 },
            { 0x00, 0x80, 0xC0, 0xE0, 0xF8, 0xFE, 0xF8, 0xE0, 0xC0, 0x80, 0x00, 0x00 },
            { 0x00, 0x02, 0x06, 0x0E, 0x3E, 0xFE, 0x3E, 0x0E, 0x06, 0x02, 0x00, 0x00 },
            { 0x00, 0x18, 0x3C, 0x7E, 0x18, 0x18, 0x18, 0x7E, 0x3C, 0x18, 0x00, 0x00 },
            { 0x00, 0x66, 0x66, 0x66, 0x66, 0x66, 0x00, 0x00, 0x66, 0x66, 0x00, 0x00 },
            { 0x00, 0x7F, 0xDB, 0xDB, 0xDB, 0x7B, 0x1B, 0x1B, 0x1B, 0x1B, 0x00, 0x00 },
            { 0x00, 0x7E, 0x63, 0x30, 0x3C, 0x66, 0x66, 0x3C, 0x0C, 0xC6, 0x7E, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFE, 0xFE, 0xFE, 0x00, 0x00 },
            { 0x00, 0x18, 0x3C, 0x7E, 0x18, 0x18, 0x18, 0x7E, 0x3C, 0x18, 0x7E, 0x00 },
            { 0x00, 0x18, 0x3C, 0x7E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00, 0x00 },
            { 0x00, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x3C, 0x18, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x18, 0x0C, 0xFE, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x30, 0x60, 0xFE, 0x60, 0x30, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0xC0, 0xC0, 0xC0, 0xFE, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x24, 0x66, 0xFF, 0x66, 0x24, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x10, 0x10, 0x38, 0x38, 0x7C, 0x7C, 0xFE, 0xFE, 0x00, 0x00 },
            { 0x00, 0x00, 0xFE, 0xFE, 0x7C, 0x7C, 0x38, 0x38, 0x10, 0x10, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x30, 0x78, 0x78, 0x78, 0x30, 0x30, 0x00, 0x30, 0x30, 0x00, 0x00 },
            { 0x00, 0x66, 0x66, 0x66, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x6C, 0x6C, 0xFE, 0x6C, 0x6C, 0x6C, 0xFE, 0x6C, 0x6C, 0x00, 0x00 },
            { 0x30, 0x30, 0x7C, 0xC0, 0xC0, 0x78, 0x0C, 0x0C, 0xF8, 0x30, 0x30, 0x00 },
            { 0x00, 0x00, 0x00, 0xC4, 0xCC, 0x18, 0x30, 0x60, 0xCC, 0x8C, 0x00, 0x00 },
            { 0x00, 0x70, 0xD8, 0xD8, 0x70, 0xFA, 0xDE, 0xCC, 0xDC, 0x76, 0x00, 0x00 },
            { 0x00, 0x30, 0x30, 0x30, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x0C, 0x18, 0x30, 0x60, 0x60, 0x60, 0x30, 0x18, 0x0C, 0x00, 0x00 },
            { 0x00, 0x60, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x60, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x18, 0x18, 0x7E, 0x18, 0x18, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0x38, 0x60, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0xFE, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x38, 0x38, 0x00, 0x00 },
            { 0x00, 0x00, 0x02, 0x06, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x80, 0x00, 0x00 },
            { 0x00, 0x7C, 0xC6, 0xCE, 0xDE, 0xD6, 0xF6, 0xE6, 0xC6, 0x7C, 0x00, 0x00 },
            { 0x00, 0x10, 0x30, 0xF0, 0x30, 0x30, 0x30, 0x30, 0x30, 0xFC, 0x00, 0x00 },
            { 0x00, 0x78, 0xCC, 0xCC, 0x0C, 0x18, 0x30, 0x60, 0xCC, 0xFC, 0x00, 0x00 },
            { 0x00, 0x78, 0xCC, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0x0C, 0x1C, 0x3C, 0x6C, 0xCC, 0xFE, 0x0C, 0x0C, 0x1E, 0x00, 0x00 },
            { 0x00, 0xFC, 0xC0, 0xC0, 0xC0, 0xF8, 0x0C, 0x0C, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0x38, 0x60, 0xC0, 0xC0, 0xF8, 0xCC, 0xCC, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0xFE, 0xC6, 0xC6, 0x06, 0x0C, 0x18, 0x30, 0x30, 0x30, 0x00, 0x00 },
            { 0x00, 0x78, 0xCC, 0xCC, 0xCC, 0x78, 0xCC, 0xCC, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0x78, 0xCC, 0xCC, 0xCC, 0x7C, 0x18, 0x18, 0x30, 0x70, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x38, 0x38, 0x00, 0x00, 0x38, 0x38, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x38, 0x38, 0x00, 0x00, 0x38, 0x38, 0x18, 0x30, 0x00 },
            { 0x00, 0x0C, 0x18, 0x30, 0x60, 0xC0, 0x60, 0x30, 0x18, 0x0C, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x7E, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x00, 0x00 },
            { 0x00, 0x78, 0xCC, 0x0C, 0x18, 0x30, 0x30, 0x00, 0x30, 0x30, 0x00, 0x00 },
            { 0x00, 0x7C, 0xC6, 0xC6, 0xDE, 0xDE, 0xDE, 0xC0, 0xC0, 0x7C, 0x00, 0x00 },
            { 0x00, 0x30, 0x78, 0xCC, 0xCC, 0xCC, 0xFC, 0xCC, 0xCC, 0xCC, 0x00, 0x00 },
            { 0x00, 0xFC, 0x66, 0x66, 0x66, 0x7C, 0x66, 0x66, 0x66, 0xFC, 0x00, 0x00 },
            { 0x00, 0x3C, 0x66, 0xC6, 0xC0, 0xC0, 0xC0, 0xC6, 0x66, 0x3C, 0x00, 0x00 },
            { 0x00, 0xF8, 0x6C, 0x66, 0x66, 0x66, 0x66, 0x66, 0x6C, 0xF8, 0x00, 0x00 },
            { 0x00, 0xFE, 0x62, 0x60, 0x64, 0x7C, 0x64, 0x60, 0x62, 0xFE, 0x00, 0x00 },
            { 0x00, 0xFE, 0x66, 0x62, 0x64, 0x7C, 0x64, 0x60, 0x60, 0xF0, 0x00, 0x00 },
            { 0x00, 0x3C, 0x66, 0xC6, 0xC0, 0xC0, 0xCE, 0xC6, 0x66, 0x3E, 0x00, 0x00 },
            { 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0xFC, 0xCC, 0xCC, 0xCC, 0xCC, 0x00, 0x00 },
            { 0x00, 0x78, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x78, 0x00, 0x00 },
            { 0x00, 0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0xE6, 0x66, 0x6C, 0x6C, 0x78, 0x6C, 0x6C, 0x66, 0xE6, 0x00, 0x00 },
            { 0x00, 0xF0, 0x60, 0x60, 0x60, 0x60, 0x62, 0x66, 0x66, 0xFE, 0x00, 0x00 },
            { 0x00, 0xC6, 0xEE, 0xFE, 0xFE, 0xD6, 0xC6, 0xC6, 0xC6, 0xC6, 0x00, 0x00 },
            { 0x00, 0xC6, 0xC6, 0xE6, 0xF6, 0xFE, 0xDE, 0xCE, 0xC6, 0xC6, 0x00, 0x00 },
            { 0x00, 0x38, 0x6C, 0xC6, 0xC6, 0xC6, 0xC6, 0xC6, 0x6C, 0x38, 0x00, 0x00 },
            { 0x00, 0xFC, 0x66, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0xF0, 0x00, 0x00 },
            { 0x00, 0x38, 0x6C, 0xC6, 0xC6, 0xC6, 0xCE, 0xDE, 0x7C, 0x0C, 0x1E, 0x00 },
            { 0x00, 0xFC, 0x66, 0x66, 0x66, 0x7C, 0x6C, 0x66, 0x66, 0xE6, 0x00, 0x00 },
            { 0x00, 0x78, 0xCC, 0xCC, 0xC0, 0x70, 0x18, 0xCC, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0xFC, 0xB4, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x78, 0x00, 0x00 },
            { 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0x78, 0x30, 0x00, 0x00 },
            { 0x00, 0xC6, 0xC6, 0xC6, 0xC6, 0xD6, 0xD6, 0x6C, 0x6C, 0x6C, 0x00, 0x00 },
            { 0x00, 0xCC, 0xCC, 0xCC, 0x78, 0x30, 0x78, 0xCC, 0xCC, 0xCC, 0x00, 0x00 },
            { 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0x78, 0x30, 0x30, 0x30, 0x78, 0x00, 0x00 },
            { 0x00, 0xFE, 0xCE, 0x98, 0x18, 0x30, 0x60, 0x62, 0xC6, 0xFE, 0x00, 0x00 },
            { 0x00, 0x3C, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x30, 0x3C, 0x00, 0x00 },
            { 0x00, 0x00, 0x80, 0xC0, 0x60, 0x30, 0x18, 0x0C, 0x06, 0x02, 0x00, 0x00 },
            { 0x00, 0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x3C, 0x00, 0x00 },
            { 0x10, 0x38, 0x6C, 0xC6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF, 0x00 },
            { 0x30, 0x30, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x78, 0x0C, 0x7C, 0xCC, 0xCC, 0x76, 0x00, 0x00 },
            { 0x00, 0xE0, 0x60, 0x60, 0x7C, 0x66, 0x66, 0x66, 0x66, 0xDC, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x78, 0xCC, 0xC0, 0xC0, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0x1C, 0x0C, 0x0C, 0x7C, 0xCC, 0xCC, 0xCC, 0xCC, 0x76, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x78, 0xCC, 0xFC, 0xC0, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0x38, 0x6C, 0x60, 0x60, 0xF8, 0x60, 0x60, 0x60, 0xF0, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x76, 0xCC, 0xCC, 0xCC, 0x7C, 0x0C, 0xCC, 0x78 },
            { 0x00, 0xE0, 0x60, 0x60, 0x6C, 0x76, 0x66, 0x66, 0x66, 0xE6, 0x00, 0x00 },
            { 0x00, 0x18, 0x18, 0x00, 0x78, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00, 0x00 },
            { 0x00, 0x0C, 0x0C, 0x00, 0x3C, 0x0C, 0x0C, 0x0C, 0x0C, 0xCC, 0xCC, 0x78 },
            { 0x00, 0xE0, 0x60, 0x60, 0x66, 0x6C, 0x78, 0x6C, 0x66, 0xE6, 0x00, 0x00 },
            { 0x00, 0x78, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0xFC, 0xD6, 0xD6, 0xD6, 0xD6, 0xC6, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0xF8, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x78, 0xCC, 0xCC, 0xCC, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0xDC, 0x66, 0x66, 0x66, 0x66, 0x7C, 0x60, 0xF0 },
            { 0x00, 0x00, 0x00, 0x00, 0x76, 0xCC, 0xCC, 0xCC, 0xCC, 0x7C, 0x0C, 0x1E },
            { 0x00, 0x00, 0x00, 0x00, 0xEC, 0x6E, 0x76, 0x60, 0x60, 0xF0, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x78, 0xCC, 0x60, 0x18, 0xCC, 0x78, 0x00, 0x00 },
            { 0x00, 0x00, 0x20, 0x60, 0xFC, 0x60, 0x60, 0x60, 0x6C, 0x38, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0x76, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0xCC, 0xCC, 0xCC, 0xCC, 0x78, 0x30, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0xC6, 0xC6, 0xD6, 0xD6, 0x6C, 0x6C, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0xC6, 0x6C, 0x38, 0x38, 0x6C, 0xC6, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x00, 0x66, 0x66, 0x66, 0x66, 0x3C, 0x0C, 0x18, 0xF0 },
            { 0x00, 0x00, 0x00, 0x00, 0xFC, 0x8C, 0x18, 0x60, 0xC4, 0xFC, 0x00, 0x00 },
            { 0x00, 0x1C, 0x30, 0x30, 0x60, 0xC0, 0x60, 0x30, 0x30, 0x1C, 0x00, 0x00 },
            { 0x00, 0x18, 0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x18, 0x00, 0x00 },
            { 0x00, 0xE0, 0x30, 0x30, 0x18, 0x0C, 0x18, 0x30, 0x30, 0xE0, 0x00, 0x00 },
            { 0x00, 0x73, 0xDA, 0xCE, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 },
            { 0x00, 0x00, 0x00, 0x10, 0x38, 0x6C, 0xC6, 0xC6, 0xFE, 0x00, 0x00, 0x00 },
        };
    }

    class imgui final {
    public:
        style& get_style() {
            return style_;
        }

        const style& get_style() const {
            return style_;
        }

        void set_renderer(imgui_renderer* renderer) {
            renderer_ = renderer;
        }

        imgui_renderer* get_renderer() const {
            return renderer_;
        }

        void update_configuration(int x, int y, int width, int height) {
            origin_x_ = x;
            origin_y_ = y;
            parent_width_ = width;
            parent_height_ = height;
            window_width_ = width;
            window_height_ = height;
        }

        bool was_resized() {
            if (window_width_ != prev_window_width_ || window_height_ != prev_window_height_) {
                prev_window_width_ = window_width_;
                prev_window_height_ = window_height_;
                return true;
            }
            return false;
        }

        void reset_layout() {
            cursor_x_ = origin_x_;
            cursor_y_ = origin_y_;
        }

        void update_focus(bool gained) {
            if (!gained) {
                ctrl_held_ = false;
                shift_held_ = false;
                mouse_down_[0] = mouse_down_[1] = mouse_down_[2] = false;
            }
        }

        void update_key(char32_t value, bool press) {
            if ((value == key_control_left) || (value == key_control_right))
                ctrl_held_ = press;
            if ((value == key_shift_left) || (value == key_shift_right))
                shift_held_ = press;

            if (press && num_keys_pressed_ < 16) {
                keys_pressed_[num_keys_pressed_++] = value;
            }
        }

        void update_mouse_button(int index, bool press) {
            if (index >= 0 && index < 3) {
                if (press && !mouse_down_[index]) {
                    mouse_pressed_[index] = true;
                }
                mouse_down_[index] = press;
            }
        }

        void update_mouse_scroll(int delta) {
            (void)delta;
        }

        void update_mouse_move(int x, int y) {
            mouse_x_ = x;
            mouse_y_ = y;
        }

        void prepare() {
            tab_index_ = 0;
        }

        void present() {
            mouse_pressed_[0] = mouse_pressed_[1] = mouse_pressed_[2] = false;
            num_keys_pressed_ = 0;
        }

        struct cursor_pos {
            int x, y;
        };

        cursor_pos get_cursor_pos() const {
            return cursor_pos{ cursor_x_, cursor_y_ };
        }

        void set_cursor_pos(int x, int y) {
            cursor_x_ = x;
            cursor_y_ = y;
        }

        void new_line() {
            cursor_x_ = origin_x_;
            cursor_y_ += style_.font_height + style_.spacing;
        }

        void indent() {
            cursor_x_ += style_.indent;
        }

        void unindent() {
            cursor_x_ -= style_.indent;
        }

        void widget_label(const char* text) {
            if (!renderer_)
                return;
            renderer_->draw_text(text, cursor_x_, cursor_y_, style_.text);
            int w = renderer_->text_width(text);
            cursor_x_ += w + style_.spacing;
        }

        bool widget_button(const char* label) {
            if (!renderer_)
                return false;
            int width = renderer_->text_width(label) + style_.padding * 2;
            int height = style_.font_height + style_.padding;

            rect btn_rect{ cursor_x_, cursor_y_, width, height };
            bool hovered = btn_rect.contains(mouse_x_, mouse_y_);
            color btn_color = hovered ? style_.button_hover : style_.button;

            renderer_->draw_rect(btn_rect, btn_color);
            renderer_->draw_frame(btn_rect, style_.border);

            renderer_->draw_text(label, cursor_x_ + style_.padding, cursor_y_ + style_.padding, style_.text);

            bool clicked = hovered && mouse_pressed_[0];
            cursor_x_ += width + style_.spacing;
            return clicked;
        }

        bool widget_checkbox(const char* label, bool* checked) {
            if (!renderer_)
                return false;
            int size = style_.font_height;
            rect box_rect{ cursor_x_, cursor_y_, size, size };

            bool hovered = box_rect.contains(mouse_x_, mouse_y_);
            renderer_->draw_rect(box_rect, hovered ? style_.hover : style_.button);
            renderer_->draw_frame(box_rect, style_.border);

            if (*checked) {
                renderer_->draw_rect({ cursor_x_ + 3, cursor_y_ + 3, size - 6, size - 6 }, style_.slider_knob);
            }

            cursor_x_ += size + style_.spacing;

            bool changed = false;
            if (hovered && mouse_pressed_[0]) {
                *checked = !*checked;
                changed = true;
            }

            widget_label(label);
            return changed;
        }

        bool widget_slider_int(const char* label, int* value, int min_val, int max_val) {
            float working = static_cast<float>(*value);
            const bool changed = widget_slider(label, &working, static_cast<float>(min_val), static_cast<float>(max_val), "%.0f");
            if (changed) {
                *value = static_cast<int>(working + ((working < 0.0f) ? -0.5f : 0.5f));
            }
            return changed;
        }

        bool widget_slider_float(const char* label, float* value, float min_val, float max_val) {
            return widget_slider(label, value, min_val, max_val, "%.2f");
        }

        bool widget_slider(const char* label, float* value, float min_val, float max_val, const char* format) {
            if (!renderer_)
                return false;

            widget_label(label);

            int slider_width = parent_width_ > 0 ? parent_width_ - style_.padding * 2 - renderer_->text_width(label) - style_.spacing : 100;
            if (slider_width < 50)
                slider_width = 50;
            int height = style_.font_height;
            rect slider_rect{ cursor_x_, cursor_y_, slider_width, height };

            bool hovered = slider_rect.contains(mouse_x_, mouse_y_);

            renderer_->draw_rect(slider_rect, style_.button);
            renderer_->draw_frame(slider_rect, style_.border);

            const float span = (max_val > min_val) ? (max_val - min_val) : 1.0f;
            float ratio = (*value - min_val) / span;
            ratio = (ratio < 0.0f) ? 0.0f : (ratio > 1.0f) ? 1.0f
                                                           : ratio;
            int knob_x = cursor_x_ + 2 + static_cast<int>(ratio * static_cast<float>(slider_width - 4 - height));
            renderer_->draw_rect({ knob_x, cursor_y_ + 2, height - 4, height - 4 }, style_.slider_knob);

            char value_str[32];
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wformat-nonliteral"
#endif
            std::snprintf(value_str, sizeof(value_str), format, static_cast<double>(*value));
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
            renderer_->draw_text(value_str, cursor_x_ + slider_width + style_.spacing, cursor_y_, style_.text);

            bool changed = false;
            if (hovered && mouse_down_[0]) {
                int local_x = mouse_x_ - cursor_x_;
                ratio = static_cast<float>(local_x) / static_cast<float>(slider_width);
                ratio = (ratio < 0.0f) ? 0.0f : (ratio > 1.0f) ? 1.0f
                                                               : ratio;
                const float updated = min_val + ratio * span;
                changed = (updated != *value);
                *value = updated;
            }

            cursor_x_ += slider_width + style_.spacing + renderer_->text_width(value_str) + style_.spacing;
            return changed;
        }

        bool widget_input_text(const char* label, char* value, int max_length) {
            if (!renderer_)
                return false;

            widget_label(label);

            int input_width = parent_width_ > 0 ? parent_width_ - style_.padding * 2 - renderer_->text_width(label) - style_.spacing : 100;
            if (input_width < 50)
                input_width = 50;
            int height = style_.font_height + style_.padding;

            rect input_rect{ cursor_x_, cursor_y_, input_width, height };
            bool hovered = input_rect.contains(mouse_x_, mouse_y_);
            int char_width = renderer_->text_width(" ");
            if (char_width <= 0)
                char_width = 1;
            int visible = (input_width - style_.padding * 2) / char_width;
            if (visible < 1)
                visible = 1;
            const int length = static_cast<int>(std::strlen(value));
            const bool was_focused = (focused_input_ == value);
            int first = (length > visible) ? (length - visible) : 0;
            if (was_focused)
                first = scroll_chars_;

            const auto position_at = [&](int x) {
                int position = first + (x - cursor_x_ - style_.padding + char_width / 2) / char_width;
                return (position < 0) ? 0 : (position > length) ? length
                                                                : position;
            };
            const auto move_caret = [&](int position, bool extend) {
                caret_pos_ = position;
                selection_end_ = position;
                if (!extend)
                    selection_start_ = position;
            };
            const auto selection_span = [&](int& from, int& to) {
                from = (selection_start_ < selection_end_) ? selection_start_ : selection_end_;
                to = (selection_start_ < selection_end_) ? selection_end_ : selection_start_;
            };

            bool changed = false;

            if (hovered && mouse_pressed_[0]) {
                const long long now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
                if (click_target_ != value) {
                    click_target_ = value;
                    click_times_[0] = click_times_[1] = click_times_[2] = 0;
                }
                click_times_[2] = click_times_[1];
                click_times_[1] = click_times_[0];
                click_times_[0] = now;
                const bool double_click = (click_times_[0] - click_times_[1]) < 500;
                const bool triple_click = double_click && ((click_times_[1] - click_times_[2]) < 500);

                focused_input_ = value;
                const int click_position = position_at(mouse_x_);
                drag_start_ = click_position;
                if (triple_click) {
                    selection_start_ = 0;
                    move_caret(length, true);
                }
                else if (double_click) {
                    int word_start = click_position;
                    int word_end = click_position;
                    while (word_start > 0 && value[word_start - 1] != ' ')
                        word_start--;
                    while (value[word_end] && value[word_end] != ' ')
                        word_end++;
                    selection_start_ = word_start;
                    move_caret(word_end, true);
                }
                else {
                    move_caret(click_position, false);
                }
            }

            if ((focused_input_ == value) && mouse_down_[0]) {
                const int drag_position = position_at(mouse_x_);
                if (drag_position != drag_start_) {
                    move_caret(drag_position, true);
                }
            }

            if (mouse_pressed_[0] && !hovered && (focused_input_ == value)) {
                focused_input_ = nullptr;
            }

            const bool focused = (focused_input_ == value);

            if (focused) {
                caret_pos_ = (caret_pos_ > length) ? length : (caret_pos_ < 0) ? 0
                                                                               : caret_pos_;
                selection_start_ = (selection_start_ > length) ? length : (selection_start_ < 0) ? 0
                                                                                                 : selection_start_;
                selection_end_ = caret_pos_;

                const auto erase_selection = [&]() {
                    int from = 0;
                    int to = 0;
                    selection_span(from, to);
                    if (from == to)
                        return false;
                    std::memmove(&value[from], &value[to], static_cast<std::size_t>(std::strlen(value) - static_cast<std::size_t>(to)) + 1);
                    move_caret(from, false);
                    return true;
                };
                const auto insert_character = [&](char character) {
                    const int current = static_cast<int>(std::strlen(value));
                    if (current >= max_length - 1)
                        return false;
                    std::memmove(&value[caret_pos_ + 1], &value[caret_pos_], static_cast<std::size_t>(current - caret_pos_) + 1);
                    value[caret_pos_] = character;
                    move_caret(caret_pos_ + 1, false);
                    return true;
                };
                const auto word_left = [&](int position) {
                    while (position > 0 && value[position - 1] == ' ')
                        position--;
                    while (position > 0 && value[position - 1] != ' ')
                        position--;
                    return position;
                };
                const auto word_right = [&](int position) {
                    while (value[position] == ' ')
                        position++;
                    while (value[position] && value[position] != ' ')
                        position++;
                    return position;
                };

                for (int k = 0; k < num_keys_pressed_; k++) {
                    const char32_t key = keys_pressed_[k];
                    const int current = static_cast<int>(std::strlen(value));
                    if ((key == key_control_left) || (key == key_control_right) || (key == key_shift_left) || (key == key_shift_right)) {
                        continue;
                    }
                    if (key == key_escape) {
                        move_caret(caret_pos_, false);
                    }
                    else if (ctrl_held_ && (key == 'a' || key == 'A')) {
                        selection_start_ = 0;
                        move_caret(current, true);
                    }
                    else if (ctrl_held_ && (key == 'c' || key == 'C' || key == 'x' || key == 'X')) {
                        int from = 0;
                        int to = 0;
                        selection_span(from, to);
                        if (from != to) {
                            clipboard::write(std::string(&value[from], static_cast<std::size_t>(to - from)));
                            if (key == 'x' || key == 'X') {
                                changed = erase_selection() || changed;
                            }
                        }
                    }
                    else if (ctrl_held_ && (key == 'v' || key == 'V')) {
                        std::string pasted;
                        if (clipboard::read(pasted)) {
                            changed = erase_selection() || changed;
                            for (const char character : pasted) {
                                if ((character < 32) || (character >= 127)) {
                                    continue;
                                }
                                if (!insert_character(character)) {
                                    break;
                                }
                                changed = true;
                            }
                        }
                    }
                    else if (key == key_right) {
                        move_caret(ctrl_held_ ? word_right(caret_pos_) : ((caret_pos_ < current) ? caret_pos_ + 1 : current), shift_held_);
                    }
                    else if (key == key_left) {
                        move_caret(ctrl_held_ ? word_left(caret_pos_) : ((caret_pos_ > 0) ? caret_pos_ - 1 : 0), shift_held_);
                    }
                    else if (key == key_home) {
                        move_caret(0, shift_held_);
                    }
                    else if (key == key_end) {
                        move_caret(current, shift_held_);
                    }
                    else if (key == key_backspace || key == 8) {
                        if (!erase_selection() && (caret_pos_ > 0)) {
                            selection_start_ = caret_pos_ - 1;
                            erase_selection();
                        }
                        changed = true;
                    }
                    else if (key == key_delete || key == 127) {
                        if (!erase_selection() && (caret_pos_ < current)) {
                            selection_start_ = caret_pos_ + 1;
                            erase_selection();
                        }
                        changed = true;
                    }
                    else if (key >= 32 && key < 127) {
                        changed = erase_selection() || changed;
                        changed = insert_character(static_cast<char>(key)) || changed;
                    }
                }

                if (caret_pos_ < scroll_chars_)
                    scroll_chars_ = caret_pos_;
                if (caret_pos_ >= scroll_chars_ + visible)
                    scroll_chars_ = caret_pos_ - visible + 1;
                const int shown = static_cast<int>(std::strlen(value));
                if (scroll_chars_ > ((shown > visible) ? (shown - visible) : 0))
                    scroll_chars_ = (shown > visible) ? (shown - visible) : 0;
                if (scroll_chars_ < 0)
                    scroll_chars_ = 0;
                first = scroll_chars_;
            }
            else if (was_focused) {
                scroll_chars_ = 0;
                first = (length > visible) ? (length - visible) : 0;
            }

            renderer_->draw_rect(input_rect, focused ? style_.active : (hovered ? style_.hover : style_.button));
            renderer_->draw_frame(input_rect, style_.border);

            const int shown_length = static_cast<int>(std::strlen(value));
            const int text_x = cursor_x_ + style_.padding;
            const auto column_of = [&](int position) {
                position -= first;
                return (position < 0) ? 0 : (position > visible) ? visible
                                                                 : position;
            };
            if (focused) {
                int from = 0;
                int to = 0;
                selection_span(from, to);
                if (from != to) {
                    renderer_->draw_rect({ text_x + column_of(from) * char_width, cursor_y_ + 2, (column_of(to) - column_of(from)) * char_width, height - 4 }, color::rgb(50, 100, 200));
                }
            }
            const int slice = ((shown_length - first) < visible) ? (shown_length - first) : visible;
            if (slice > 0) {
                renderer_->draw_text(std::string(&value[first], static_cast<std::size_t>(slice)).c_str(), text_x, cursor_y_ + style_.padding / 2 + (style_.font_height - renderer_->text_height()) / 2, style_.text);
            }
            if (focused) {
                renderer_->draw_rect({ text_x + column_of(caret_pos_) * char_width, cursor_y_ + 2, 2, height - 4 }, style_.text);
            }

            cursor_x_ += input_width + style_.spacing;
            return changed;
        }

        class scope_window {
        public:
            scope_window(imgui& imgui, const char* title, int x, int y, int w, int h)
                : imgui_(imgui)
                , saved_origin_x_(imgui.origin_x_)
                , saved_origin_y_(imgui.origin_y_)
                , saved_parent_w_(imgui.parent_width_)
                , saved_parent_h_(imgui.parent_height_)
                , below_y_(y + h + imgui.style_.spacing) {
                if (!imgui_.renderer_)
                    return;
                imgui_.parent_width_ = w;
                imgui_.parent_height_ = h;
                imgui_.renderer_->draw_rect({ x, y, w, h }, imgui_.style_.background);
                imgui_.renderer_->draw_frame({ x, y, w, h }, imgui_.style_.border);
                imgui_.renderer_->draw_text(title, x + 4, y + 4, imgui_.style_.text);
                imgui_.origin_x_ = x + 4;
                imgui_.origin_y_ = y + 20;
                imgui_.cursor_x_ = imgui_.origin_x_;
                imgui_.cursor_y_ = imgui_.origin_y_;
            }

            ~scope_window() {
                imgui_.origin_x_ = saved_origin_x_;
                imgui_.origin_y_ = saved_origin_y_;
                imgui_.parent_width_ = saved_parent_w_;
                imgui_.parent_height_ = saved_parent_h_;
                imgui_.cursor_x_ = imgui_.origin_x_;
                imgui_.cursor_y_ = below_y_;
            }

        private:
            imgui& imgui_;
            int saved_origin_x_;
            int saved_origin_y_;
            int saved_parent_w_;
            int saved_parent_h_;
            int below_y_;
        };

        class scope_group {
        public:
            scope_group(imgui& imgui, const char* name, bool* expanded = nullptr)
                : imgui_(imgui)
                , expanded_(expanded)
                , is_expanded_(expanded ? *expanded : true) {
                if (!imgui_.renderer_)
                    return;

                int toggle_size = 12;
                rect toggle_rect{ imgui_.cursor_x_, imgui_.cursor_y_, toggle_size, toggle_size };
                bool hovered = toggle_rect.contains(imgui_.mouse_x_, imgui_.mouse_y_);
                if (hovered && imgui_.mouse_pressed_[0]) {
                    if (expanded_) {
                        *expanded_ = !*expanded_;
                        is_expanded_ = *expanded_;
                    }
                }
                imgui_.renderer_->draw_rect(toggle_rect, imgui_.style_.button);
                if (is_expanded_) {
                    imgui_.renderer_->draw_line(toggle_rect.x + 2, toggle_rect.y + 6, toggle_rect.x + 10, toggle_rect.y + 6, imgui_.style_.text);
                }
                else {
                    imgui_.renderer_->draw_line(toggle_rect.x + 6, toggle_rect.y + 2, toggle_rect.x + 6, toggle_rect.y + 10, imgui_.style_.text);
                }
                imgui_.cursor_x_ += toggle_size + 4;
                imgui_.renderer_->draw_text(name, imgui_.cursor_x_, imgui_.cursor_y_, imgui_.style_.text);
                imgui_.cursor_x_ += imgui_.renderer_->text_width(name) + 4;

                imgui_.new_line();
            }

            ~scope_group() {
                imgui_.cursor_x_ = imgui_.origin_x_;
            }

            bool expanded() const {
                return is_expanded_;
            }

        private:
            imgui& imgui_;
            bool* expanded_;
            bool is_expanded_;
        };

        class scope_tab {
        public:
            scope_tab(imgui& imgui, const char* name, int* selected)
                : imgui_(imgui)
                , selected_(selected) {
                if (!imgui_.renderer_)
                    return;
                int tab_width = imgui_.renderer_->text_width(name) + 24;
                int height = imgui_.style_.font_height + 8;
                rect tab_rect{ imgui_.cursor_x_, imgui_.cursor_y_, tab_width, height };
                bool hovered = tab_rect.contains(imgui_.mouse_x_, imgui_.mouse_y_);
                bool is_selected = (*selected_ == imgui_.tab_index_);
                color tab_color = is_selected ? imgui_.style_.tab_active : (hovered ? imgui_.style_.hover : imgui_.style_.tab);
                imgui_.renderer_->draw_rect(tab_rect, tab_color);
                imgui_.renderer_->draw_rect({ tab_rect.x, tab_rect.y + height - 2, tab_width, 2 }, is_selected ? imgui_.style_.slider_knob : imgui_.style_.border);
                imgui_.renderer_->draw_text(name, imgui_.cursor_x_ + 12, imgui_.cursor_y_ + 6, imgui_.style_.text);
                if (hovered && imgui_.mouse_pressed_[0]) {
                    *selected_ = imgui_.tab_index_;
                }
                imgui_.tab_index_++;
                imgui_.cursor_x_ += tab_width + 4;
            }

        private:
            imgui& imgui_;
            int* selected_;
        };

        class scope_horizontal {
        public:
            scope_horizontal(imgui& imgui, int cell_width = 0, int cell_height = 0)
                : imgui_(imgui)
                , saved_x_(imgui.cursor_x_)
                , saved_y_(imgui.cursor_y_)
                , cell_width_(cell_width)
                , cell_height_(cell_height)
                , num_cells_(0) {
            }

            ~scope_horizontal() {
                int effective_height = cell_height_;
                if (effective_height == -1) {
                    effective_height = imgui_.parent_height_ - (saved_y_ - imgui_.origin_y_) - imgui_.style_.padding;
                }
                imgui_.cursor_x_ = saved_x_;
                if (effective_height > 0) {
                    imgui_.cursor_y_ += effective_height;
                }
            }

            void set_cell_size(int w, int h) {
                cell_width_ = w;
                cell_height_ = h;
            }

            int cell_width() const {
                if (cell_width_ == -1 && imgui_.parent_width_ > 0 && num_cells_ > 0) {
                    return (imgui_.parent_width_ - imgui_.style_.padding * 2) / num_cells_ - imgui_.style_.spacing;
                }
                return cell_width_;
            }

            void set_num_cells(int n) {
                num_cells_ = n;
            }

        private:
            imgui& imgui_;
            int saved_x_;
            int saved_y_;
            int cell_width_;
            int cell_height_;
            int num_cells_;
        };

        class scope_vertical {
        public:
            scope_vertical(imgui& imgui, int cell_width = 0, int cell_height = 0)
                : imgui_(imgui)
                , saved_y_(imgui.cursor_y_)
                , cell_width_(cell_width)
                , cell_height_(cell_height)
                , num_cells_(0) {
            }

            ~scope_vertical() {
                int effective_width = cell_width_;
                if (effective_width == -1 && imgui_.parent_width_ > 0 && num_cells_ > 0) {
                    effective_width = (imgui_.parent_width_ - imgui_.style_.padding * 2) / num_cells_ - imgui_.style_.spacing;
                }
                imgui_.cursor_y_ = saved_y_;
                if (effective_width > 0) {
                    imgui_.cursor_x_ += effective_width;
                }
            }

            void set_cell_size(int w, int h) {
                cell_width_ = w;
                cell_height_ = h;
            }

            void set_num_cells(int n) {
                num_cells_ = n;
            }

        private:
            imgui& imgui_;
            int saved_y_;
            int cell_width_;
            int cell_height_;
            int num_cells_;
        };

        void widget_spacer(int width) {
            cursor_x_ += width;
        }

        void widget_spacer_h(int height) {
            cursor_y_ += height;
        }

        class scope_grid {
        public:
            scope_grid(imgui& imgui, int columns, int cell_width = 0, int cell_height = 0)
                : imgui_(imgui)
                , columns_(columns)
                , saved_x_(imgui.cursor_x_)
                , saved_y_(imgui.cursor_y_)
                , item_index_(0)
                , cell_width_(cell_width)
                , cell_height_(cell_height) {
            }

            ~scope_grid() {
                imgui_.cursor_x_ = saved_x_;
                imgui_.cursor_y_ = saved_y_;
            }

            void next() {
                int effective_cell_width = cell_width_;
                int effective_cell_height = cell_height_;

                if (cell_width_ == -1) {
                    effective_cell_width = (imgui_.parent_width_ - imgui_.style_.padding * 2) / columns_ - imgui_.style_.spacing;
                }
                if (cell_height_ == -1) {
                    effective_cell_height = imgui_.style_.font_height + imgui_.style_.padding;
                }

                if (effective_cell_width > 0 && effective_cell_height > 0) {
                    item_index_++;
                    if (item_index_ >= columns_) {
                        item_index_ = 0;
                        imgui_.cursor_x_ = saved_x_;
                        imgui_.cursor_y_ += effective_cell_height;
                    }
                    else {
                        imgui_.cursor_x_ += effective_cell_width;
                    }
                }
                else {
                    if (++item_index_ >= columns_) {
                        item_index_ = 0;
                        imgui_.cursor_x_ = saved_x_;
                        imgui_.cursor_y_ += imgui_.style_.font_height + imgui_.style_.spacing + imgui_.style_.padding;
                    }
                }
            }

            void set_cell_size(int w, int h) {
                cell_width_ = w;
                cell_height_ = h;
            }

        private:
            imgui& imgui_;
            int columns_;
            int saved_x_;
            int saved_y_;
            int item_index_;
            int cell_width_;
            int cell_height_;
        };

        int get_window_width() const {
            return window_width_;
        }

        int get_window_height() const {
            return window_height_;
        }

    private:
        constexpr static const char32_t key_control_left = 65507;
        constexpr static const char32_t key_control_right = 65508;
        constexpr static const char32_t key_shift_left = 65505;
        constexpr static const char32_t key_shift_right = 65506;
        constexpr static const char32_t key_escape = 65307;
        constexpr static const char32_t key_left = 65361;
        constexpr static const char32_t key_right = 65363;
        constexpr static const char32_t key_home = 65360;
        constexpr static const char32_t key_end = 65367;
        constexpr static const char32_t key_backspace = 65288;
        constexpr static const char32_t key_delete = 65535;

        int tab_index_ = 0;
        int window_width_ = 640;
        int window_height_ = 480;
        int prev_window_width_ = 640;
        int prev_window_height_ = 480;
        int cursor_x_ = 0;
        int cursor_y_ = 0;
        int origin_x_ = 0;
        int origin_y_ = 0;
        int parent_width_ = 640;
        int parent_height_ = 480;
        int mouse_x_ = 0;
        int mouse_y_ = 0;
        bool mouse_down_[3] = { false, false, false };
        bool mouse_pressed_[3] = { false, false, false };
        const void* focused_input_ = nullptr;
        int caret_pos_ = 0;
        int selection_start_ = 0;
        int selection_end_ = 0;
        int drag_start_ = 0;
        int scroll_chars_ = 0;
        const void* click_target_ = nullptr;
        long long click_times_[3] = { 0, 0, 0 };
        bool ctrl_held_ = false;
        bool shift_held_ = false;
        char32_t keys_pressed_[16] = {};
        int num_keys_pressed_ = 0;
        style style_;
        imgui_renderer* renderer_ = nullptr;
    };

    class imgui_graph {
    public:
        void add_value(float value) {
            values_[value_index_] = value;
            value_index_ = (value_index_ + 1) % max_values_;
            if (num_values_ < max_values_)
                num_values_++;
        }

        void clear() {
            num_values_ = 0;
            value_index_ = 0;
        }

        void set_max_values(int max) {
            max_values_ = (max < 2) ? 2 : (max > 256) ? 256
                                                      : max;
            num_values_ = 0;
            value_index_ = 0;
        }

        void render(imgui_renderer* renderer, int x, int y, int w, int h, const char* label) {
            if (!renderer)
                return;

            renderer->draw_rect({ x, y, w, h }, color::rgb(20, 20, 20));
            renderer->draw_frame({ x, y, w, h }, color::rgb(100, 100, 100));

            if (label) {
                renderer->draw_text(label, x + 4, y + 4, color::white());
            }

            if (num_values_ < 2)
                return;

            const auto sample = [&](int i) {
                return values_[(num_values_ < max_values_) ? i : (value_index_ + i) % max_values_];
            };
            float min_val = sample(0);
            float max_val = sample(0);
            for (int i = 1; i < num_values_; i++) {
                min_val = (sample(i) < min_val) ? sample(i) : min_val;
                max_val = (sample(i) > max_val) ? sample(i) : max_val;
            }
            min_val = (min_val > 0.0f) ? 0.0f : min_val;
            const float span = (max_val > min_val) ? (max_val - min_val) : 1.0f;

            int graph_x = x + 4;
            int graph_y = y + 20;
            int graph_w = w - 8;
            int graph_h = h - 24;

            int num_points = num_values_;

            float x_step = static_cast<float>(graph_w) / static_cast<float>(num_points - 1);

            for (int i = 0; i < num_points - 1; ++i) {
                float v1 = (sample(i) - min_val) / span;
                float v2 = (sample(i + 1) - min_val) / span;
                int px1 = graph_x + static_cast<int>(static_cast<float>(i) * x_step);
                int px2 = graph_x + static_cast<int>(static_cast<float>(i + 1) * x_step);
                int py1 = graph_y + graph_h - static_cast<int>(v1 * static_cast<float>(graph_h));
                int py2 = graph_y + graph_h - static_cast<int>(v2 * static_cast<float>(graph_h));
                renderer->draw_line(px1, py1, px2, py2, color::rgb(100, 150, 255));
            }
        }

    private:
        float values_[256] = {};
        int max_values_ = 100;
        int num_values_ = 0;
        int value_index_ = 0;
    };

    class imgui_viewport {
    public:
        void set_render_callback(void (*callback)(int, int)) {
            render_callback_ = callback;
        }

        void render(int x, int y, int w, int h) {
            (void)x;
            (void)y;
            if (render_callback_) {
                render_callback_(w, h);
            }
        }

    private:
        void (*render_callback_)(int, int) = nullptr;
    };

    class imgui_software_renderer final : public imgui_renderer {
    public:
        void set_buffer(uint8_t* buffer, int width, int height, int stride) {
            buffer_ = buffer;
            width_ = width;
            height_ = height;
            stride_ = stride;
        }

        void draw_rect(const rect& r, const color& c) override {
            if (!buffer_)
                return;
            for (int py = r.y; py < r.y + r.h && py < height_; ++py) {
                if (py < 0)
                    continue;
                uint8_t* line = buffer_ + py * stride_;
                for (int px = r.x; px < r.x + r.w && px < width_; ++px) {
                    if (px < 0)
                        continue;
                    uint8_t* pixel = line + px * 4;
                    pixel[0] = c.r;
                    pixel[1] = c.g;
                    pixel[2] = c.b;
                    pixel[3] = c.a;
                }
            }
        }

        void draw_text(const char* text, int x, int y, const color& c) override {
            if (!buffer_ || !text)
                return;
            const char* p = text;
            int draw_x = x;
            while (*p) {
                const int codepoint = imgui_detail::utf8_decode(p);
                const unsigned char* glyph = ((codepoint >= 0) && (codepoint < 128)) ? &imgui_detail::font_glyphs[codepoint][0] : &imgui_detail::font_glyphs[0][0];
                for (int gy = 0; gy < 12 && (y + gy) < height_; ++gy) {
                    if (y + gy < 0)
                        continue;
                    uint8_t* line = buffer_ + (y + gy) * stride_;
                    for (int gx = 0; gx < 8 && (draw_x + gx) < width_; ++gx) {
                        if (draw_x + gx < 0)
                            continue;
                        if (glyph[gy] & (0x80 >> gx)) {
                            uint8_t* pixel = line + (draw_x + gx) * 4;
                            pixel[0] = c.r;
                            pixel[1] = c.g;
                            pixel[2] = c.b;
                            pixel[3] = c.a;
                        }
                    }
                }
                draw_x += 8;
            }
        }

        void draw_line(int x1, int y1, int x2, int y2, const color& c) override {
            int dx = (x2 > x1) ? x2 - x1 : x1 - x2;
            int dy = (y2 > y1) ? y2 - y1 : y1 - y2;
            int steps = (dx > dy) ? dx : dy;
            if (steps == 0)
                steps = 1;
            float x_inc = static_cast<float>(x2 - x1) / static_cast<float>(steps);
            float y_inc = static_cast<float>(y2 - y1) / static_cast<float>(steps);
            float x = static_cast<float>(x1);
            float y = static_cast<float>(y1);
            for (int i = 0; i <= steps; ++i) {
                int px = static_cast<int>(x);
                int py = static_cast<int>(y);
                if (px >= 0 && px < width_ && py >= 0 && py < height_) {
                    uint8_t* pixel = buffer_ + py * stride_ + px * 4;
                    pixel[0] = c.r;
                    pixel[1] = c.g;
                    pixel[2] = c.b;
                    pixel[3] = c.a;
                }
                x += x_inc;
                y += y_inc;
            }
        }

        int text_width(const char* text) override {
            if (!text)
                return 0;
            int len = 0;
            while (*text) {
                imgui_detail::utf8_decode(text);
                len += 8;
            }
            return len;
        }

        int text_height() override {
            return 12;
        }

    private:
        uint8_t* buffer_ = nullptr;
        int width_ = 0;
        int height_ = 0;
        int stride_ = 0;
    };

}

#endif // ZEROSLAM_TOOLS_GUI_IMGUI_HPP
