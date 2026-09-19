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
#ifndef ZEROSLAM_TOOLS_GUI_WINDOW_HPP
#define ZEROSLAM_TOOLS_GUI_WINDOW_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace gtl {
    class window final {
    public:
        class position_type final {
        public:
            int x;
            int y;

        public:
            bool operator==(const position_type& other) {
                return ((this->x == other.x) && (this->y == other.y));
            }

            bool operator!=(const position_type& other) {
                return ((this->x != other.x) || (this->y != other.y));
            }
        };

    public:
        class size_type final {
        public:
            int width;
            int height;

        public:
            bool operator==(const size_type& other) {
                return ((this->width == other.width) && (this->height == other.height));
            }

            bool operator!=(const size_type& other) {
                return ((this->width != other.width) || (this->height != other.height));
            }
        };

    public:
        class event_type final {
        public:
            enum class input_type {
                configuration,
                focus,
                key_press,
                key_release,
                mouse_press,
                mouse_release,
                mouse_scroll,
                mouse_move
            };

        public:
            struct configuration_type {
                int x, y, width, height;
            };

            enum class focus_type {
                gained,
                lost
            };

            using key_type = char32_t;

            enum class mouse_type {
                unknown,
                left,
                middle,
                right,
                scroll_forward,
                scroll_backward
            };

            using scroll_type = int;

            struct motion_type {
                int x, y;
            };

        public:
            union data_type {
                configuration_type configuration;
                focus_type focus;
                key_type key;
                mouse_type mouse;
                scroll_type scroll;
                motion_type motion;
            };

        public:
            input_type input;
            data_type data;
        };

    public:
        class driver_type {
        public:
            using position_type = window::position_type;
            using size_type = window::size_type;
            using event_type = window::event_type;

        public:
            virtual ~driver_type() = default;
            driver_type() = default;
            driver_type(const driver_type&) = delete;
            driver_type(driver_type&&) = delete;
            driver_type& operator=(const driver_type&) = delete;
            driver_type& operator=(driver_type&&) = delete;

        public:
            virtual bool is_open() const = 0;
            virtual bool open(const position_type& position, const size_type& size, const std::string& title, const bool hardware_accelerated) = 0;
            virtual void close() = 0;
            virtual bool is_hardware_accelerated() const = 0;
            virtual bool set_vsync(bool enable) const = 0;
            virtual bool is_iconised() const = 0;
            virtual void set_iconised(bool iconised) = 0;
            virtual bool is_visible() const = 0;
            virtual void set_visible(bool visible) = 0;
            virtual bool is_maximised() const = 0;
            virtual void set_maximised(bool maximised) = 0;
            virtual bool is_borderless() const = 0;
            virtual void set_borderless(bool borderless) = 0;
            virtual bool is_fullscreen() const = 0;
            virtual void set_fullscreen(bool fullscreen) = 0;
            virtual bool is_focused() const = 0;
            virtual void set_focused(bool focused) = 0;
            virtual bool is_cursor_hidden() const = 0;
            virtual void set_cursor_hidden(bool cursor_hidden) = 0;
            virtual bool is_grab_input() const = 0;
            virtual void set_grab_input(bool grab_input) = 0;
            virtual position_type get_position() const = 0;
            virtual void set_position(const position_type& position) = 0;
            virtual size_type get_size() const = 0;
            virtual void set_size(const size_type& size) = 0;
            virtual std::string get_title() const = 0;
            virtual void set_title(const std::string& new_title) = 0;
            virtual void get_icon(int& width, int& height, std::vector<unsigned int>& data) const = 0;
            virtual void set_icon(int width, int height, const std::vector<unsigned int>& data) = 0;
            virtual bool process(event_type& event) = 0;
            virtual void prepare() = 0;
            virtual void draw(const unsigned char* __restrict const rgb, const int width, const int height, const int step, const int stride) = 0;
            virtual void present() = 0;
            virtual bool read(unsigned char* __restrict const rgb, const int width, const int height, const int step, const int stride) = 0;
        };

    private:
        driver_type* driver = nullptr;

    private:
        static driver_type* create_driver();
        static driver_type* create_driver_linux_x11();
        static driver_type* create_driver_macos_cocoa();
        static driver_type* create_driver_windows_win32();

    public:
        static bool valid_driver();

    public:
        ~window();
        window();
        window(const window&) = delete;
        window(window&& other);
        window& operator=(const window&) = delete;
        window& operator=(window&& other);

    public:
        bool is_open() const;

    public:
        bool open(const position_type& position, const size_type& size, const std::string& title, const bool hardware_accelerated);
        void close();

    public:
        bool is_hardware_accelerated() const;
        bool set_vsync(bool enable) const;

        bool is_iconised() const;
        void set_iconised(bool iconised);

        bool is_visible() const;
        void set_visible(bool visible);

        bool is_maximised() const;
        void set_maximised(bool maximised);

        bool is_borderless() const;
        void set_borderless(bool borderless);

        bool is_fullscreen() const;
        void set_fullscreen(bool fullscreen);

        bool is_focused() const;
        void set_focused(bool focused);

        bool is_cursor_hidden() const;
        void set_cursor_hidden(bool cursor_hidden);

        bool is_grab_input() const;
        void set_grab_input(bool grab_input);

    public:
        position_type get_position() const;
        void set_position(const position_type& position);

    public:
        size_type get_size() const;
        void set_size(const size_type& size);

    public:
        std::string get_title() const;
        void set_title(const std::string& new_title);

    public:
        void get_icon(int& width, int& height, std::vector<unsigned int>& data) const;
        void set_icon(int width, int height, const std::vector<unsigned int>& data);

    public:
        bool process(window::event_type& event);
        void prepare();
        void draw(const unsigned char* __restrict const rgb, const int width, const int height, const int step, const int stride);
        void present();
        bool read(unsigned char* __restrict const rgb, const int width, const int height, const int step, const int stride);
    };
}

#endif // ZEROSLAM_TOOLS_GUI_WINDOW_HPP
