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

#include "window.hpp"

#include <string>
#include <vector>

namespace gtl {
    window::driver_type* window::create_driver() {
        if (driver_type* candidate = window::create_driver_windows_win32()) {
            return candidate;
        }
        if (driver_type* candidate = window::create_driver_macos_cocoa()) {
            return candidate;
        }
        if (driver_type* candidate = window::create_driver_linux_x11()) {
            return candidate;
        }
        return nullptr;
    }

    bool window::valid_driver() {
        driver_type* candidate = window::create_driver();
        const bool valid = (candidate != nullptr);
        delete candidate;
        return valid;
    }

    window::~window() {
        if (this->driver) {
            this->driver->close();
            delete this->driver;
        }
    }

    window::window()
        : driver(window::create_driver()) {
    }

    window::window(window&& other)
        : driver(other.driver) {
        other.driver = nullptr;
    }

    window& window::operator=(window&& other) {
        if (this != &other) {
            driver_type* temp = this->driver;
            this->driver = other.driver;
            other.driver = temp;
        }
        return *this;
    }

    bool window::is_open() const {
        return this->driver && this->driver->is_open();
    }

    bool window::open(const window::position_type& position, const window::size_type& size, const std::string& title, const bool hardware_accelerated) {
        return this->driver && this->driver->open(position, size, title, hardware_accelerated);
    }

    void window::close() {
        if (this->driver) {
            this->driver->close();
        }
    }

    bool window::is_hardware_accelerated() const {
        return this->driver && this->driver->is_hardware_accelerated();
    }

    bool window::set_vsync(bool enable) const {
        return this->driver && this->driver->set_vsync(enable);
    }

    bool window::is_iconised() const {
        return this->driver && this->driver->is_iconised();
    }

    void window::set_iconised(bool iconised) {
        if (this->driver) {
            this->driver->set_iconised(iconised);
        }
    }

    bool window::is_visible() const {
        return this->driver && this->driver->is_visible();
    }

    void window::set_visible(bool visible) {
        if (this->driver) {
            this->driver->set_visible(visible);
        }
    }

    bool window::is_maximised() const {
        return this->driver && this->driver->is_maximised();
    }

    void window::set_maximised(bool maximised) {
        if (this->driver) {
            this->driver->set_maximised(maximised);
        }
    }

    bool window::is_borderless() const {
        return this->driver && this->driver->is_borderless();
    }

    void window::set_borderless(bool borderless) {
        if (this->driver) {
            this->driver->set_borderless(borderless);
        }
    }

    bool window::is_fullscreen() const {
        return this->driver && this->driver->is_fullscreen();
    }

    void window::set_fullscreen(bool fullscreen) {
        if (this->driver) {
            this->driver->set_fullscreen(fullscreen);
        }
    }

    bool window::is_focused() const {
        return this->driver && this->driver->is_focused();
    }

    void window::set_focused(bool focused) {
        if (this->driver) {
            this->driver->set_focused(focused);
        }
    }

    bool window::is_cursor_hidden() const {
        return this->driver && this->driver->is_cursor_hidden();
    }

    void window::set_cursor_hidden(bool cursor_hidden) {
        if (this->driver) {
            this->driver->set_cursor_hidden(cursor_hidden);
        }
    }

    bool window::is_grab_input() const {
        return this->driver && this->driver->is_grab_input();
    }

    void window::set_grab_input(bool grab_input) {
        if (this->driver) {
            this->driver->set_grab_input(grab_input);
        }
    }

    window::position_type window::get_position() const {
        return this->driver ? this->driver->get_position() : window::position_type{ 0, 0 };
    }

    void window::set_position(const window::position_type& position) {
        if (this->driver) {
            this->driver->set_position(position);
        }
    }

    window::size_type window::get_size() const {
        return this->driver ? this->driver->get_size() : window::size_type{ 0, 0 };
    }

    void window::set_size(const window::size_type& size) {
        if (this->driver) {
            this->driver->set_size(size);
        }
    }

    std::string window::get_title() const {
        return this->driver ? this->driver->get_title() : std::string();
    }

    void window::set_title(const std::string& new_title) {
        if (this->driver) {
            this->driver->set_title(new_title);
        }
    }

    void window::get_icon(int& width, int& height, std::vector<unsigned int>& data) const {
        if (this->driver) {
            this->driver->get_icon(width, height, data);
        }
    }

    void window::set_icon(int width, int height, const std::vector<unsigned int>& data) {
        if (this->driver) {
            this->driver->set_icon(width, height, data);
        }
    }

    bool window::process(window::event_type& event) {
        return this->driver && this->driver->process(event);
    }

    void window::prepare() {
        if (this->driver) {
            this->driver->prepare();
        }
    }

    void window::draw(const unsigned char* __restrict const rgb, const int width, const int height, const int step, const int stride) {
        if (this->driver) {
            this->driver->draw(rgb, width, height, step, stride);
        }
    }

    void window::present() {
        if (this->driver) {
            this->driver->present();
        }
    }

    bool window::read(unsigned char* __restrict const rgb, const int width, const int height, const int step, const int stride) {
        return this->driver && this->driver->read(rgb, width, height, step, stride);
    }
}
