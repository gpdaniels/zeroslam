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

#if defined(_WIN32) || defined(_WIN64)
#include "opengl.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <deque>
#include <string>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    constexpr DWORD style_popup = static_cast<DWORD>(WS_POPUP);
    constexpr DWORD style_overlapped = static_cast<DWORD>(WS_OVERLAPPEDWINDOW);

    constexpr char32_t keysym_backspace = 65288;
    constexpr char32_t keysym_tab = 65289;
    constexpr char32_t keysym_return = 65293;
    constexpr char32_t keysym_pause = 65299;
    constexpr char32_t keysym_escape = 65307;
    constexpr char32_t keysym_home = 65360;
    constexpr char32_t keysym_left = 65361;
    constexpr char32_t keysym_up = 65362;
    constexpr char32_t keysym_right = 65363;
    constexpr char32_t keysym_down = 65364;
    constexpr char32_t keysym_page_up = 65365;
    constexpr char32_t keysym_page_down = 65366;
    constexpr char32_t keysym_end = 65367;
    constexpr char32_t keysym_insert = 65379;
    constexpr char32_t keysym_f1 = 65470;
    constexpr char32_t keysym_shift_left = 65505;
    constexpr char32_t keysym_control_left = 65507;
    constexpr char32_t keysym_caps_lock = 65509;
    constexpr char32_t keysym_alt_left = 65513;
    constexpr char32_t keysym_super_left = 65515;
    constexpr char32_t keysym_delete = 65535;
    constexpr char32_t keysym_unicode = 0x01000000;

    char32_t translate_virtual_key(WPARAM virtual_key) {
        switch (virtual_key) {
            case VK_BACK:
                return keysym_backspace;
            case VK_TAB:
                return keysym_tab;
            case VK_RETURN:
                return keysym_return;
            case VK_PAUSE:
                return keysym_pause;
            case VK_ESCAPE:
                return keysym_escape;
            case VK_HOME:
                return keysym_home;
            case VK_LEFT:
                return keysym_left;
            case VK_UP:
                return keysym_up;
            case VK_RIGHT:
                return keysym_right;
            case VK_DOWN:
                return keysym_down;
            case VK_PRIOR:
                return keysym_page_up;
            case VK_NEXT:
                return keysym_page_down;
            case VK_END:
                return keysym_end;
            case VK_INSERT:
                return keysym_insert;
            case VK_DELETE:
                return keysym_delete;
            case VK_SHIFT:
            case VK_LSHIFT:
            case VK_RSHIFT:
                return keysym_shift_left;
            case VK_CONTROL:
            case VK_LCONTROL:
            case VK_RCONTROL:
                return keysym_control_left;
            case VK_MENU:
            case VK_LMENU:
            case VK_RMENU:
                return keysym_alt_left;
            case VK_CAPITAL:
                return keysym_caps_lock;
            case VK_LWIN:
            case VK_RWIN:
                return keysym_super_left;
            default:
                break;
        }
        if ((virtual_key >= VK_F1) && (virtual_key <= VK_F12)) {
            return keysym_f1 + static_cast<char32_t>(virtual_key - VK_F1);
        }
        return 0;
    }

    char32_t translate_character_key(WPARAM virtual_key) {
        const UINT character = MapVirtualKeyA(static_cast<UINT>(virtual_key), MAPVK_VK_TO_CHAR) & 0xFFFF;
        if ((character >= 'A') && (character <= 'Z')) {
            return static_cast<char32_t>(character - 'A' + 'a');
        }
        if ((character < 0x20) || (character == 0x7F)) {
            return 0;
        }
        return static_cast<char32_t>(character);
    }

    char32_t translate_character(WPARAM character) {
        if ((character < 0x20) || (character == 0x7F)) {
            return 0;
        }
        if (character < 0x100) {
            return static_cast<char32_t>(character);
        }
        return keysym_unicode | static_cast<char32_t>(character);
    }

    RECT outer_rectangle(int width, int height, DWORD style_flags, DWORD extended_style_flags) {
        RECT rectangle = { 0, 0, width, height };
        AdjustWindowRectEx(&rectangle, style_flags, FALSE, extended_style_flags);
        return rectangle;
    }
}

namespace {
    class win32_driver final : public gtl::window::driver_type {
    public:
        HWND window_handle = nullptr;
        HDC device_context = nullptr;
        HGLRC opengl_context = nullptr;
        HICON icon_handle = nullptr;
        bool closed = false;
        int icon_width = 0;
        int icon_height = 0;
        std::vector<unsigned int> icon_data;

        struct message_type {
            UINT message;
            WPARAM word_param;
            LPARAM long_param;
        };

        std::deque<message_type> messages;

        static LRESULT CALLBACK callback(HWND handle, UINT message, WPARAM word_param, LPARAM long_param) {
            if (message == WM_NCCREATE) {
                const CREATESTRUCTA* create_data = reinterpret_cast<const CREATESTRUCTA*>(long_param);
                SetWindowLongPtrA(handle, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(create_data->lpCreateParams));
                return DefWindowProcA(handle, message, word_param, long_param);
            }

            win32_driver* platform = reinterpret_cast<win32_driver*>(GetWindowLongPtrA(handle, GWLP_USERDATA));
            if (platform != nullptr) {
                platform->messages.push_back(message_type{ message, word_param, long_param });
            }

            switch (message) {
                case WM_CLOSE:
                    return 0;
                case WM_MENUCHAR:
                    return MAKELRESULT(0, MNC_CLOSE);
                case WM_ERASEBKGND:
                    return TRUE;
                case WM_QUERYENDSESSION:
                    return TRUE;
                case WM_CANCELMODE:
                    ReleaseCapture();
                    return FALSE;
                default:
                    break;
            }

            return DefWindowProcA(handle, message, word_param, long_param);
        }

        bool is_open() const override {
            return ((this->window_handle != nullptr));
        }

        bool open(const position_type& position, const size_type& size, const std::string& title, const bool hardware_accelerated) override {
            if (this->is_open())
                return false;

            constexpr static const char* class_name = "gtl::window";
            constexpr static const DWORD style_flags = style_overlapped | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
            constexpr static const DWORD extended_style_flags = WS_EX_APPWINDOW;

            HINSTANCE module_handle = GetModuleHandleA(nullptr);
            WNDCLASSEXA window_class = {};

            if (GetClassInfoExA(module_handle, class_name, &window_class) == 0) {
                window_class.cbSize = sizeof(WNDCLASSEXA);
                window_class.style = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
                window_class.cbClsExtra = 0;
                window_class.cbWndExtra = 0;
                window_class.hInstance = module_handle;
                window_class.hIcon = nullptr;
                window_class.hCursor = LoadCursorA(nullptr, IDC_ARROW);
                window_class.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
                window_class.lpszMenuName = nullptr;
                window_class.lpszClassName = class_name;
                window_class.hIconSm = nullptr;
                window_class.lpfnWndProc = win32_driver::callback;

                if (RegisterClassExA(&window_class) == 0) {
                    return false;
                }
            }

            this->closed = false;
            this->messages.clear();

            const RECT outer = outer_rectangle(size.width, size.height, style_flags, extended_style_flags);

            this->window_handle = CreateWindowExA(
                extended_style_flags,
                class_name,
                title.c_str(),
                style_flags,
                position.x,
                position.y,
                outer.right - outer.left,
                outer.bottom - outer.top,
                nullptr,
                nullptr,
                module_handle,
                this
            );

            if (this->window_handle == nullptr) {
                return false;
            }

            this->device_context = GetDC(this->window_handle);
            if (!this->device_context) {
                this->close();
                return false;
            }

            PIXELFORMATDESCRIPTOR pixel_format_descriptor = {};
            pixel_format_descriptor.nSize = sizeof(PIXELFORMATDESCRIPTOR);
            pixel_format_descriptor.nVersion = 1;
            pixel_format_descriptor.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
            pixel_format_descriptor.iPixelType = PFD_TYPE_RGBA;
            pixel_format_descriptor.cColorBits = 32;
            pixel_format_descriptor.cDepthBits = 24;
            pixel_format_descriptor.cStencilBits = 8;
            pixel_format_descriptor.iLayerType = PFD_MAIN_PLANE;

            const int pixel_format_index = ChoosePixelFormat(this->device_context, &pixel_format_descriptor);
            if (pixel_format_index == 0) {
                this->close();
                return false;
            }

            if (!SetPixelFormat(this->device_context, pixel_format_index, &pixel_format_descriptor)) {
                this->close();
                return false;
            }

            if (hardware_accelerated) {
                this->opengl_context = wglCreateContext(this->device_context);
                if (!this->opengl_context) {
                    this->close();
                    return false;
                }
                wglMakeCurrent(this->device_context, this->opengl_context);
            }

            this->set_title(title);

            return true;
        }

        void close() override {
            if (!this->is_open())
                return;

            if (this->opengl_context) {
                wglMakeCurrent(nullptr, nullptr);
                wglDeleteContext(this->opengl_context);
                this->opengl_context = nullptr;
            }

            if (this->device_context) {
                ReleaseDC(this->window_handle, this->device_context);
                this->device_context = nullptr;
            }

            SetWindowLongPtrA(this->window_handle, GWLP_USERDATA, 0);
            if (!this->closed) {
                DestroyWindow(this->window_handle);
            }
            this->window_handle = nullptr;
            this->closed = false;
            this->messages.clear();

            if (this->icon_handle) {
                DestroyIcon(this->icon_handle);
                this->icon_handle = nullptr;
            }
        }

        bool is_hardware_accelerated() const override {
            if (!this->is_open())
                return false;
            return this->opengl_context != nullptr;
        }

        bool set_vsync(bool enable) const override {
            if (!this->is_open())
                return false;
            if (!this->is_hardware_accelerated())
                return false;

            using swap_interval_type = BOOL(WINAPI*)(int);
            swap_interval_type swap_interval = reinterpret_cast<swap_interval_type>(reinterpret_cast<void*>(wglGetProcAddress("wglSwapIntervalEXT")));
            if (!swap_interval) {
                return false;
            }
            return swap_interval(enable ? 1 : 0) != FALSE;
        }

        bool is_iconised() const override {
            if (!this->is_open())
                return false;
            return (IsIconic(this->window_handle) != FALSE);
        }

        void set_iconised(bool iconised) override {
            if (!this->is_open())
                return;
            ShowWindow(this->window_handle, iconised ? SW_MINIMIZE : SW_RESTORE);
        }

        bool is_visible() const override {
            if (!this->is_open())
                return false;
            return (IsWindowVisible(this->window_handle) != FALSE);
        }

        void set_visible(bool visible) override {
            if (!this->is_open())
                return;
            ShowWindow(this->window_handle, visible ? SW_SHOW : SW_HIDE);
        }

        bool is_maximised() const override {
            if (!this->is_open())
                return false;
            return (IsZoomed(this->window_handle) != FALSE);
        }

        void set_maximised(bool maximised) override {
            if (!this->is_open())
                return;
            ShowWindow(this->window_handle, maximised ? SW_MAXIMIZE : SW_RESTORE);
        }

        bool is_borderless() const override {
            if (!this->is_open())
                return false;
            const DWORD style_flags = static_cast<DWORD>(GetWindowLongA(this->window_handle, GWL_STYLE));
            return (style_flags & style_popup) != 0;
        }

        void set_borderless(bool borderless) override {
            if (!this->is_open())
                return;
            const DWORD style_flags = static_cast<DWORD>(GetWindowLongA(this->window_handle, GWL_STYLE));
            const DWORD new_style_flags = borderless ? ((style_flags | style_popup) & ~style_overlapped) : ((style_flags | style_overlapped) & ~style_popup);
            SetWindowLongA(this->window_handle, GWL_STYLE, static_cast<LONG>(new_style_flags));
            SetWindowPos(this->window_handle, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
        }

        bool is_fullscreen() const override {
            if (!this->is_open())
                return false;
            const DWORD style_flags = static_cast<DWORD>(GetWindowLongA(this->window_handle, GWL_STYLE));

            MONITORINFO monitor_info = {};
            monitor_info.cbSize = sizeof(monitor_info);
            GetMonitorInfoA(MonitorFromWindow(this->window_handle, MONITOR_DEFAULTTONEAREST), &monitor_info);

            RECT window_rect = {};
            GetWindowRect(this->window_handle, &window_rect);

            return ((style_flags & style_popup) != 0) && (monitor_info.rcMonitor.left == window_rect.left) && (monitor_info.rcMonitor.top == window_rect.top) && (monitor_info.rcMonitor.right == window_rect.right) && (monitor_info.rcMonitor.bottom == window_rect.bottom);
        }

        void set_fullscreen(bool fullscreen) override {
            if (!this->is_open())
                return;

            const DWORD style_flags = static_cast<DWORD>(GetWindowLongA(this->window_handle, GWL_STYLE));

            if (fullscreen) {
                SetWindowLongA(this->window_handle, GWL_STYLE, static_cast<LONG>((style_flags | style_popup) & ~style_overlapped));

                MONITORINFO monitor_info = {};
                monitor_info.cbSize = sizeof(monitor_info);
                GetMonitorInfoA(MonitorFromWindow(this->window_handle, MONITOR_DEFAULTTONEAREST), &monitor_info);
                SetWindowPos(
                    this->window_handle,
                    nullptr,
                    monitor_info.rcMonitor.left,
                    monitor_info.rcMonitor.top,
                    monitor_info.rcMonitor.right - monitor_info.rcMonitor.left,
                    monitor_info.rcMonitor.bottom - monitor_info.rcMonitor.top,
                    SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED
                );
            }
            else {
                SetWindowLongA(this->window_handle, GWL_STYLE, static_cast<LONG>((style_flags | style_overlapped) & ~style_popup));
                SetWindowPos(this->window_handle, nullptr, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED);
            }
        }

        bool is_focused() const override {
            if (!this->is_open())
                return false;
            return (GetFocus() == this->window_handle);
        }

        void set_focused(bool focused) override {
            if (!this->is_open())
                return;
            if (focused) {
                SetForegroundWindow(this->window_handle);
                SetFocus(this->window_handle);
            }
        }

        bool is_cursor_hidden() const override {
            if (!this->is_open())
                return false;
            const int visible = ShowCursor(TRUE);
            ShowCursor(FALSE);
            return (visible <= 0);
        }

        void set_cursor_hidden(bool cursor_hidden) override {
            if (!this->is_open())
                return;
            if (cursor_hidden) {
                while (ShowCursor(FALSE) >= 0) {
                }
            }
            else {
                while (ShowCursor(TRUE) < 0) {
                }
            }
        }

        bool is_grab_input() const override {
            if (!this->is_open())
                return false;
            return (this->window_handle == GetCapture());
        }

        void set_grab_input(bool grab_input) override {
            if (!this->is_open())
                return;
            if (grab_input) {
                SetCapture(this->window_handle);
            }
            else {
                ReleaseCapture();
            }
        }

        position_type get_position() const override {
            if (!this->is_open())
                return { 0, 0 };
            RECT window_rect = {};
            GetWindowRect(this->window_handle, &window_rect);
            return { static_cast<int>(window_rect.left), static_cast<int>(window_rect.top) };
        }

        void set_position(const position_type& position) override {
            if (!this->is_open())
                return;
            SetWindowPos(this->window_handle, nullptr, position.x, position.y, 0, 0, SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE);
        }

        size_type get_size() const override {
            if (!this->is_open())
                return { 0, 0 };
            RECT client_rect = {};
            GetClientRect(this->window_handle, &client_rect);
            return { static_cast<int>(client_rect.right - client_rect.left), static_cast<int>(client_rect.bottom - client_rect.top) };
        }

        void set_size(const size_type& size) override {
            if (!this->is_open())
                return;
            const DWORD style_flags = static_cast<DWORD>(GetWindowLongA(this->window_handle, GWL_STYLE));
            const DWORD extended_style_flags = static_cast<DWORD>(GetWindowLongA(this->window_handle, GWL_EXSTYLE));
            const RECT outer = outer_rectangle(size.width, size.height, style_flags, extended_style_flags);
            SetWindowPos(this->window_handle, nullptr, 0, 0, outer.right - outer.left, outer.bottom - outer.top, SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE);
        }

        std::string get_title() const override {
            if (!this->is_open())
                return {};
            const int title_length = GetWindowTextLengthA(this->window_handle);
            if (title_length <= 0) {
                return {};
            }
            std::string buffer(static_cast<size_t>(title_length) + 1, '\0');
            const int copied = GetWindowTextA(this->window_handle, buffer.data(), title_length + 1);
            buffer.resize(static_cast<size_t>((copied > 0) ? copied : 0));
            return buffer;
        }

        void set_title(const std::string& new_title) override {
            if (!this->is_open())
                return;
            SetWindowTextA(this->window_handle, new_title.c_str());
        }

        void get_icon(int& width, int& height, std::vector<unsigned int>& data) const override {
            width = this->icon_width;
            height = this->icon_height;
            data = this->icon_data;
        }

        void set_icon(int width, int height, const std::vector<unsigned int>& data) override {
            if (!this->is_open())
                return;
            if ((width <= 0) || (height <= 0)) {
                return;
            }
            if (static_cast<size_t>(width) * static_cast<size_t>(height) != data.size()) {
                return;
            }

            this->icon_width = width;
            this->icon_height = height;
            this->icon_data = data;

            HBITMAP colour_bitmap = CreateBitmap(width, height, 1, 32, data.data());
            HBITMAP mask_bitmap = CreateBitmap(width, height, 1, 1, nullptr);
            if (!colour_bitmap || !mask_bitmap) {
                if (colour_bitmap) {
                    DeleteObject(colour_bitmap);
                }
                if (mask_bitmap) {
                    DeleteObject(mask_bitmap);
                }
                return;
            }

            ICONINFO icon_info = {};
            icon_info.fIcon = TRUE;
            icon_info.hbmMask = mask_bitmap;
            icon_info.hbmColor = colour_bitmap;
            HICON new_icon_handle = CreateIconIndirect(&icon_info);
            DeleteObject(colour_bitmap);
            DeleteObject(mask_bitmap);
            if (!new_icon_handle) {
                return;
            }

            SendMessageA(this->window_handle, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(new_icon_handle));
            SendMessageA(this->window_handle, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(new_icon_handle));

            if (this->icon_handle) {
                DestroyIcon(this->icon_handle);
            }
            this->icon_handle = new_icon_handle;
        }

        bool process(event_type& event) override {
            if (!this->is_open())
                return false;

            MSG windows_event = {};
            while (PeekMessageA(&windows_event, this->window_handle, 0, 0, PM_REMOVE) != FALSE) {
                TranslateMessage(&windows_event);
                DispatchMessageA(&windows_event);
            }

            while (!this->messages.empty()) {
                const win32_driver::message_type message = this->messages.front();
                this->messages.pop_front();

                switch (message.message) {
                    case WM_NCDESTROY:
                    case WM_DESTROY: {
                        this->closed = true;
                        this->close();
                        return false;
                    }

                    case WM_ENDSESSION:
                    case WM_CLOSE: {
                        this->close();
                        return false;
                    }

                    case WM_SIZE:
                    case WM_MOVE:
                    case WM_EXITSIZEMOVE: {
                        RECT window_rect = {};
                        GetWindowRect(this->window_handle, &window_rect);
                        const size_type size = this->get_size();
                        event.input = event_type::input_type::configuration;
                        event.data.configuration = event_type::configuration_type{ static_cast<int>(window_rect.left), static_cast<int>(window_rect.top), size.width, size.height };
                        return true;
                    }

                    case WM_SETFOCUS: {
                        event.input = event_type::input_type::focus;
                        event.data.focus = event_type::focus_type::gained;
                        return true;
                    }

                    case WM_KILLFOCUS: {
                        event.input = event_type::input_type::focus;
                        event.data.focus = event_type::focus_type::lost;
                        return true;
                    }

                    case WM_SYSKEYDOWN:
                    case WM_KEYDOWN: {
                        const char32_t special_key = translate_virtual_key(message.word_param);
                        if (special_key != 0) {
                            event.input = event_type::input_type::key_press;
                            event.data.key = special_key;
                            return true;
                        }
                        const bool control_held = (GetKeyState(VK_CONTROL) & 0x8000) != 0;
                        const bool alt_held = (message.message == WM_SYSKEYDOWN);
                        if (control_held || alt_held) {
                            const char32_t character_key = translate_character_key(message.word_param);
                            if (character_key != 0) {
                                event.input = event_type::input_type::key_press;
                                event.data.key = character_key;
                                return true;
                            }
                        }
                        break;
                    }

                    case WM_CHAR: {
                        const char32_t character = translate_character(message.word_param);
                        if (character != 0) {
                            event.input = event_type::input_type::key_press;
                            event.data.key = character;
                            return true;
                        }
                        break;
                    }

                    case WM_SYSKEYUP:
                    case WM_KEYUP: {
                        const char32_t special_key = translate_virtual_key(message.word_param);
                        const char32_t key = (special_key != 0) ? special_key : translate_character_key(message.word_param);
                        if (key != 0) {
                            event.input = event_type::input_type::key_release;
                            event.data.key = key;
                            return true;
                        }
                        break;
                    }

                    case WM_LBUTTONDOWN:
                    case WM_LBUTTONDBLCLK:
                    case WM_LBUTTONUP: {
                        event.input = (message.message == WM_LBUTTONUP) ? event_type::input_type::mouse_release : event_type::input_type::mouse_press;
                        event.data.mouse = event_type::mouse_type::left;
                        return true;
                    }

                    case WM_MBUTTONDOWN:
                    case WM_MBUTTONDBLCLK:
                    case WM_MBUTTONUP: {
                        event.input = (message.message == WM_MBUTTONUP) ? event_type::input_type::mouse_release : event_type::input_type::mouse_press;
                        event.data.mouse = event_type::mouse_type::middle;
                        return true;
                    }

                    case WM_RBUTTONDOWN:
                    case WM_RBUTTONDBLCLK:
                    case WM_RBUTTONUP: {
                        event.input = (message.message == WM_RBUTTONUP) ? event_type::input_type::mouse_release : event_type::input_type::mouse_press;
                        event.data.mouse = event_type::mouse_type::right;
                        return true;
                    }

                    case WM_MOUSEWHEEL: {
                        const int delta = static_cast<int>(GET_WHEEL_DELTA_WPARAM(message.word_param));
                        if (delta == 0) {
                            break;
                        }
                        event.input = event_type::input_type::mouse_scroll;
                        event.data.scroll = (delta > 0) ? 1 : -1;
                        return true;
                    }

                    case WM_MOUSEMOVE: {
                        event.input = event_type::input_type::mouse_move;
                        event.data.motion.x = static_cast<int>(static_cast<short>(LOWORD(message.long_param)));
                        event.data.motion.y = static_cast<int>(static_cast<short>(HIWORD(message.long_param)));
                        return true;
                    }

                    default:
                        break;
                }
            }

            return false;
        }

        void prepare() override {
            if (!this->is_open())
                return;
            if (this->is_hardware_accelerated()) {
                wglMakeCurrent(this->device_context, this->opengl_context);
            }
        }

        void draw(
            const unsigned char* __restrict const data,
            const int width,
            const int height,
            const int step,
            const int stride
        ) override {
            if (!this->is_open())
                return;
            if (this->is_hardware_accelerated())
                return;

            const size_type size = this->get_size();
            if ((size.width <= 0) || (size.height <= 0)) {
                return;
            }

            BITMAPINFO bitmap_info = {};
            bitmap_info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bitmap_info.bmiHeader.biWidth = size.width;
            bitmap_info.bmiHeader.biHeight = -size.height;
            bitmap_info.bmiHeader.biPlanes = 1;
            bitmap_info.bmiHeader.biBitCount = 32;
            bitmap_info.bmiHeader.biCompression = BI_RGB;

            std::vector<unsigned int> buffer(static_cast<size_t>(size.width) * static_cast<size_t>(size.height), 0u);

            const int draw_width = (width < size.width) ? width : size.width;
            const int draw_height = (height < size.height) ? height : size.height;
            const int data_index_r = 0;
            const int data_index_g = (step == 1) ? 0 : 1;
            const int data_index_b = (step == 1) ? 0 : 2;

            for (int y = 0; y < draw_height; ++y) {
                const unsigned char* __restrict data_line = data + y * stride;
                unsigned int* __restrict buffer_line = buffer.data() + static_cast<size_t>(y) * static_cast<size_t>(size.width);
                for (int x = 0; x < draw_width; ++x, data_line += step) {
                    buffer_line[x] = ((static_cast<unsigned int>(data_line[data_index_r]) << 16) | (static_cast<unsigned int>(data_line[data_index_g]) << 8) | (static_cast<unsigned int>(data_line[data_index_b]) << 0));
                }
            }

            SetDIBitsToDevice(
                this->device_context,
                0,
                0,
                static_cast<DWORD>(size.width),
                static_cast<DWORD>(size.height),
                0,
                0,
                0,
                static_cast<UINT>(size.height),
                buffer.data(),
                &bitmap_info,
                DIB_RGB_COLORS
            );
        }

        void present() override {
            if (!this->is_open())
                return;
            if (this->is_hardware_accelerated()) {
                SwapBuffers(this->device_context);
            }
            else {
                GdiFlush();
            }
        }

        bool read(unsigned char* __restrict const rgb, const int width, const int height, const int step, const int stride) override {
            if (!this->is_open())
                return false;
            if (this->is_hardware_accelerated())
                return false;

            const size_type size = this->get_size();
            if ((size.width <= 0) || (size.height <= 0)) {
                return false;
            }

            HDC memory_context = CreateCompatibleDC(this->device_context);
            HBITMAP bitmap = CreateCompatibleBitmap(this->device_context, size.width, size.height);
            if (!memory_context || !bitmap) {
                if (bitmap) {
                    DeleteObject(bitmap);
                }
                if (memory_context) {
                    DeleteDC(memory_context);
                }
                return false;
            }
            HGDIOBJ old_bitmap = SelectObject(memory_context, bitmap);

            BitBlt(memory_context, 0, 0, size.width, size.height, this->device_context, 0, 0, SRCCOPY);

            BITMAPINFO bitmap_info = {};
            bitmap_info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
            bitmap_info.bmiHeader.biWidth = size.width;
            bitmap_info.bmiHeader.biHeight = -size.height;
            bitmap_info.bmiHeader.biPlanes = 1;
            bitmap_info.bmiHeader.biBitCount = 32;
            bitmap_info.bmiHeader.biCompression = BI_RGB;

            std::vector<unsigned char> buffer(static_cast<size_t>(size.width) * static_cast<size_t>(size.height) * 4);
            GetDIBits(memory_context, bitmap, 0, static_cast<UINT>(size.height), buffer.data(), &bitmap_info, DIB_RGB_COLORS);

            const int read_width = (width < size.width) ? width : size.width;
            const int read_height = (height < size.height) ? height : size.height;

            for (int y = 0; y < read_height; ++y) {
                unsigned char* __restrict rgb_line = rgb + y * stride;
                const unsigned char* __restrict buffer_line = buffer.data() + static_cast<size_t>(y) * static_cast<size_t>(size.width) * 4;
                for (int x = 0; x < read_width; ++x, rgb_line += step, buffer_line += 4) {
                    rgb_line[0] = buffer_line[2];
                    if (step > 1) {
                        rgb_line[1] = buffer_line[1];
                        rgb_line[2] = buffer_line[0];
                    }
                }
            }

            SelectObject(memory_context, old_bitmap);
            DeleteObject(bitmap);
            DeleteDC(memory_context);

            return true;
        }
    };
}

namespace gtl {
    window::driver_type* window::create_driver_windows_win32() {
        return new win32_driver();
    }
}
#else
namespace gtl {
    window::driver_type* window::create_driver_windows_win32() {
        return nullptr;
    }
}
#endif
