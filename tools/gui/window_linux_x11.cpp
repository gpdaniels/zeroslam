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

#if (defined(linux) || defined(__linux) || defined(__linux__)) && __has_include(<X11/Xlib.h>)
#include <GL/glx.h>
#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {
    class x11_driver final : public gtl::window::driver_type {
    public:
        Display* display_handle = nullptr;

        Window window_handle = 0;

        Atom close_handler = 0;

        GLXContext opengl_context = nullptr;

        bool hidden_cursor = false;

        bool grabbed_input = false;

        bool is_open() const override {
            return ((this->window_handle != 0));
        }

        bool open(const position_type& position, const size_type& size, const std::string& title, const bool hardware_accelerated) override {
            if (this->is_open())
                return false;

            this->display_handle = XOpenDisplay(nullptr);
            if (this->display_handle == nullptr) {
                return false;
            }

            XSetWindowAttributes window_attributes;

            window_attributes.event_mask =
                KeyPressMask | KeyReleaseMask |
                ButtonPressMask | ButtonReleaseMask |
                EnterWindowMask | LeaveWindowMask |
                PointerMotionMask |
                Button1MotionMask | Button2MotionMask | Button3MotionMask | Button4MotionMask | Button5MotionMask | ButtonMotionMask |
                KeymapStateMask |
                ExposureMask |
                VisibilityChangeMask |
                StructureNotifyMask |
                SubstructureNotifyMask |
                FocusChangeMask |
                PropertyChangeMask |
                ColormapChangeMask |
                OwnerGrabButtonMask;

            window_attributes.override_redirect = False;

            window_attributes.background_pixmap = None;

            window_attributes.border_pixel = 0;

            window_attributes.colormap = XCreateColormap(
                this->display_handle,
                (&(reinterpret_cast<_XPrivDisplay>(this->display_handle))->screens[((reinterpret_cast<_XPrivDisplay>(this->display_handle))->default_screen)])->root,
                DefaultVisual(this->display_handle, DefaultScreen(this->display_handle)),
                AllocNone
            );

            this->window_handle = XCreateWindow(
                this->display_handle,
                RootWindow(this->display_handle, DefaultScreen(this->display_handle)),
                position.x,
                position.y,
                static_cast<unsigned int>(size.width),
                static_cast<unsigned int>(size.height),
                0,
                DefaultDepth(this->display_handle, DefaultScreen(this->display_handle)),
                InputOutput,
                DefaultVisual(this->display_handle, DefaultScreen(this->display_handle)),
                CWBorderPixel | CWBackPixmap | CWColormap | CWEventMask | CWOverrideRedirect,
                &window_attributes
            );

            XSizeHints window_size{};
            window_size.flags = PPosition | PSize;
            window_size.x = position.x;
            window_size.y = position.y;
            window_size.width = size.width;
            window_size.height = size.height;
            XSetNormalHints(this->display_handle, this->window_handle, &window_size);

            this->close_handler = XInternAtom(this->display_handle, "WM_DELETE_WINDOW", False);
            XSetWMProtocols(this->display_handle, this->window_handle, &this->close_handler, 1);

            this->set_title(title);

            if (hardware_accelerated) {
                const int framebuffer_attributes[] = {
                    GLX_X_RENDERABLE,
                    GL_TRUE,
                    GLX_DRAWABLE_TYPE,
                    GLX_WINDOW_BIT,
                    GLX_DOUBLEBUFFER,
                    GL_TRUE,
                    GLX_RENDER_TYPE,
                    GLX_RGBA_BIT,
                    GLX_X_VISUAL_TYPE,
                    GLX_TRUE_COLOR,
                    GLX_BUFFER_SIZE,
                    24,
                    GLX_DEPTH_SIZE,
                    24,
                    GLX_STENCIL_SIZE,
                    8,
                    GLX_SAMPLE_BUFFERS,
                    GL_FALSE,
                    GLX_SAMPLES,
                    0,
                    0
                };

                int config_count = 0;
                GLXFBConfig* configs = glXChooseFBConfig(this->display_handle, DefaultScreen(this->display_handle), framebuffer_attributes, &config_count);
                if ((configs == nullptr) || (config_count <= 0)) {
                    this->close();
                    return false;
                }

                GLXFBConfig config = configs[0];
                XFree(configs);

                int context_attributes[] = {
                    GLX_CONTEXT_MAJOR_VERSION_ARB,
                    3,
                    GLX_CONTEXT_MINOR_VERSION_ARB,
                    3,
                    GLX_CONTEXT_PROFILE_MASK_ARB,
                    GLX_CONTEXT_COMPATIBILITY_PROFILE_BIT_ARB,
                    None
                };

                using PFNGLXCREATECONTEXTATTRIBSARBPROC = GLXContext (*)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
                PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribsARB = reinterpret_cast<PFNGLXCREATECONTEXTATTRIBSARBPROC>(glXGetProcAddressARB(reinterpret_cast<const GLubyte*>("glXCreateContextAttribsARB")));

                if (!glXCreateContextAttribsARB) {
                    this->close();
                    return false;
                }

                this->opengl_context = glXCreateContextAttribsARB(this->display_handle, config, nullptr, GL_TRUE, context_attributes);

                if (!this->opengl_context) {
                    this->close();
                    return false;
                }
            }

            return true;
        }

        void close() override {
            if (!this->is_open())
                return;

            this->set_visible(false);

            if (this->opengl_context != nullptr) {
                glXMakeCurrent(this->display_handle, None, nullptr);
                glXDestroyContext(this->display_handle, this->opengl_context);
                this->opengl_context = nullptr;
            }

            XWindowAttributes window_attributes;
            if (XGetWindowAttributes(this->display_handle, this->window_handle, &window_attributes) != 0) {
                XFreeColormap(this->display_handle, window_attributes.colormap);
            }

            XDestroyWindow(this->display_handle, this->window_handle);
            XFlush(this->display_handle);
            XSync(this->display_handle, False);
            XCloseDisplay(this->display_handle);

            this->window_handle = 0;
            this->display_handle = nullptr;
            this->hidden_cursor = false;
            this->grabbed_input = false;
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
            using PFNGLXSWAPINTERVALEXTPROC = void (*)(Display*, GLXDrawable, int);
            PFNGLXSWAPINTERVALEXTPROC glXSwapIntervalEXT = reinterpret_cast<PFNGLXSWAPINTERVALEXTPROC>(glXGetProcAddress(reinterpret_cast<const GLubyte*>("glXSwapIntervalEXT")));
            if (!glXSwapIntervalEXT) {
                return false;
            }
            glXSwapIntervalEXT(this->display_handle, this->window_handle, static_cast<int>(enable));
            return true;
        }

        bool is_iconised() const override {
            if (!this->is_open())
                return false;

            Atom type;
            int format;
            unsigned long property_count;
            unsigned long property_bytes;
            unsigned long* property_data;

            int status = XGetWindowProperty(
                this->display_handle,
                this->window_handle,
                XInternAtom(this->display_handle, "_NET_WM_STATE", False),
                0,
                (~0L),
                False,
                AnyPropertyType,
                &type,
                &format,
                &property_count,
                &property_bytes,
                reinterpret_cast<unsigned char**>(&property_data)
            );

            if (status != Success) {
                return false;
            }

            bool state_hidden = false;

            for (unsigned long property_index = 0; property_index < property_count; ++property_index) {
                if (property_data[property_index] == XInternAtom(this->display_handle, "_NET_WM_STATE_HIDDEN", False)) {
                    state_hidden = true;
                    break;
                }
            }

            XFree(property_data);

            return state_hidden;
        }

        void set_iconised(bool iconised) override {
            if (!this->is_open())
                return;

            if (iconised) {
                XIconifyWindow(this->display_handle, this->window_handle, DefaultScreen(this->display_handle));
            }
            else {
                XEvent iconify_event = {};
                iconify_event.xany.type = ClientMessage;
                iconify_event.xclient.message_type = XInternAtom(this->display_handle, "_NET_ACTIVE_WINDOW", False);
                iconify_event.xclient.format = 32;
                iconify_event.xclient.window = this->window_handle;
                iconify_event.xclient.data.l[0] = 1;
                iconify_event.xclient.data.l[1] = CurrentTime;
                iconify_event.xclient.data.l[2] = 0;

                XSendEvent(
                    this->display_handle,
                    RootWindow(this->display_handle, DefaultScreen(this->display_handle)),
                    0,
                    SubstructureNotifyMask | SubstructureRedirectMask,
                    &iconify_event
                );
            }
        }

        bool is_visible() const override {
            if (!this->is_open())
                return false;
            XWindowAttributes window_attributes;
            XGetWindowAttributes(this->display_handle, this->window_handle, &window_attributes);
            return (window_attributes.map_state != IsUnmapped);
        }

        void set_visible(bool visible) override {
            if (!this->is_open())
                return;
            if (visible) {
                XMapWindow(this->display_handle, this->window_handle);
            }
            else {
                XUnmapWindow(this->display_handle, this->window_handle);
            }
        }

        bool is_maximised() const override {
            if (!this->is_open())
                return false;

            Atom type;
            int format;
            unsigned long property_count;
            unsigned long property_bytes;
            unsigned long* property_data;

            int status = XGetWindowProperty(
                this->display_handle,
                this->window_handle,
                XInternAtom(this->display_handle, "_NET_WM_STATE", False),
                0,
                (~0L),
                False,
                AnyPropertyType,
                &type,
                &format,
                &property_count,
                &property_bytes,
                reinterpret_cast<unsigned char**>(&property_data)
            );

            if (status != Success) {
                return false;
            }

            bool state_maximised_vert = false;
            bool state_maximised_horz = false;

            for (unsigned long property_index = 0; property_index < property_count; ++property_index) {
                if (property_data[property_index] == XInternAtom(this->display_handle, "_NET_WM_STATE_MAXIMIZED_VERT", False)) {
                    state_maximised_vert = true;
                }
                else if (property_data[property_index] == XInternAtom(this->display_handle, "_NET_WM_STATE_MAXIMIZED_HORZ", False)) {
                    state_maximised_horz = true;
                }
            }

            XFree(property_data);

            return (state_maximised_vert && state_maximised_horz);
        }

        void set_maximised(bool maximised) override {
            if (!this->is_open())
                return;

            XEvent maximise_event = {};
            maximise_event.xany.type = ClientMessage;
            maximise_event.xclient.message_type = XInternAtom(this->display_handle, "_NET_WM_STATE", False);
            maximise_event.xclient.format = 32;
            maximise_event.xclient.window = this->window_handle;
            maximise_event.xclient.data.l[0] = static_cast<int>(maximised);
            maximise_event.xclient.data.l[1] = static_cast<long>(XInternAtom(this->display_handle, "_NET_WM_STATE_MAXIMIZED_VERT", False));
            maximise_event.xclient.data.l[2] = static_cast<long>(XInternAtom(this->display_handle, "_NET_WM_STATE_MAXIMIZED_HORZ", False));

            maximise_event.xclient.data.l[3] = 0;

            XSendEvent(
                this->display_handle,
                RootWindow(this->display_handle, DefaultScreen(this->display_handle)),
                0,
                SubstructureNotifyMask | SubstructureRedirectMask,
                &maximise_event
            );
        }

        bool is_borderless() const override {
            if (!this->is_open())
                return false;

            Atom type;
            int format;
            unsigned long property_count;
            unsigned long property_bytes;
            unsigned long* property_data;

            int status = XGetWindowProperty(
                this->display_handle,
                this->window_handle,
                XInternAtom(this->display_handle, "_MOTIF_WM_HINTS", False),
                0,
                (~0L),
                False,
                AnyPropertyType,
                &type,
                &format,
                &property_count,
                &property_bytes,
                reinterpret_cast<unsigned char**>(&property_data)
            );

            if (status != Success) {
                return false;
            }

            bool state_borderless = false;

            if (property_count == 3) {
                state_borderless = ((property_data[0] == 2) && (property_data[1] == 0) && (property_data[2] == 0));
            }

            XFree(property_data);

            return state_borderless;
        }

        void set_borderless(bool borderless) override {
            if (!this->is_open())
                return;

            if (borderless) {
                unsigned long hints[5] = { 2, 0, 0, 0, 0 };
                XChangeProperty(
                    this->display_handle,
                    this->window_handle,
                    XInternAtom(this->display_handle, "_MOTIF_WM_HINTS", False),
                    XInternAtom(this->display_handle, "_MOTIF_WM_HINTS", False),
                    32,
                    PropModeReplace,
                    reinterpret_cast<unsigned char*>(&hints),
                    3
                );
            }
            else {
                unsigned long hints[5] = { 3, 62, 126, 0, 0 };
                XChangeProperty(
                    this->display_handle,
                    this->window_handle,
                    XInternAtom(this->display_handle, "_MOTIF_WM_HINTS", False),
                    XInternAtom(this->display_handle, "_MOTIF_WM_HINTS", False),
                    32,
                    PropModeReplace,
                    reinterpret_cast<unsigned char*>(&hints),
                    3
                );
            }
        }

        bool is_fullscreen() const override {
            if (!this->is_open())
                return false;

            Atom type;
            int format;
            unsigned long property_count;
            unsigned long property_bytes;
            unsigned long* property_data;

            int status = XGetWindowProperty(
                this->display_handle,
                this->window_handle,
                XInternAtom(this->display_handle, "_NET_WM_STATE", False),
                0,
                (~0L),
                False,
                AnyPropertyType,
                &type,
                &format,
                &property_count,
                &property_bytes,
                reinterpret_cast<unsigned char**>(&property_data)
            );

            if (status != Success) {
                return false;
            }

            bool state_fullscreen = false;

            for (unsigned long property_index = 0; property_index < property_count; ++property_index) {
                if (property_data[property_index] == XInternAtom(this->display_handle, "_NET_WM_STATE_FULLSCREEN", False)) {
                    state_fullscreen = true;
                    break;
                }
            }

            XFree(property_data);

            return state_fullscreen;
        }

        void set_fullscreen(bool fullscreen) override {
            if (!this->is_open())
                return;

            XEvent fullscreen_event = {};
            fullscreen_event.xclient.type = ClientMessage;
            fullscreen_event.xclient.message_type = XInternAtom(this->display_handle, "_NET_WM_STATE", False);
            fullscreen_event.xclient.format = 32;
            fullscreen_event.xclient.window = this->window_handle;
            fullscreen_event.xclient.data.l[0] = static_cast<int>(fullscreen);
            fullscreen_event.xclient.data.l[1] = static_cast<long>(XInternAtom(this->display_handle, "_NET_WM_STATE_FULLSCREEN", False));
            fullscreen_event.xclient.data.l[2] = 0;

            XSendEvent(this->display_handle, DefaultRootWindow(this->display_handle), False, SubstructureRedirectMask | SubstructureNotifyMask, &fullscreen_event);
        }

        bool is_focused() const override {
            if (!this->is_open())
                return false;

            Window focused_window_handle;
            int revert_to_return;
            XGetInputFocus(this->display_handle, &focused_window_handle, &revert_to_return);
            return (focused_window_handle == this->window_handle);
        }

        void set_focused(bool focused) override {
            if (!this->is_open())
                return;
            XWindowAttributes window_attributes;
            if ((XGetWindowAttributes(this->display_handle, this->window_handle, &window_attributes) == 0) || (window_attributes.map_state != IsViewable))
                return;

            if (focused) {
                XSetInputFocus(this->display_handle, this->window_handle, RevertToParent, CurrentTime);
            }
            else {
                XSetInputFocus(this->display_handle, PointerRoot, RevertToParent, CurrentTime);
            }
        }

        bool is_cursor_hidden() const override {
            if (!this->is_open())
                return false;

            return this->hidden_cursor;
        }

        void set_cursor_hidden(bool cursor_hidden) override {
            if (!this->is_open())
                return;

            if (cursor_hidden) {
                XColor cursor_colour = {};
                static char cursor_data[] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                Pixmap cursor_bitmap = XCreateBitmapFromData(this->display_handle, this->window_handle, cursor_data, 8, 8);
                Cursor cursor = XCreatePixmapCursor(this->display_handle, cursor_bitmap, cursor_bitmap, &cursor_colour, &cursor_colour, 0, 0);
                XDefineCursor(this->display_handle, this->window_handle, cursor);
                XFreeCursor(this->display_handle, cursor);
                XFreePixmap(this->display_handle, cursor_bitmap);
            }
            else {
                Cursor cursor = XCreateFontCursor(this->display_handle, XC_left_ptr);
                XDefineCursor(this->display_handle, this->window_handle, cursor);
                XFreeCursor(this->display_handle, cursor);
            }

            this->hidden_cursor = cursor_hidden;
        }

        bool is_grab_input() const override {
            if (!this->is_open())
                return false;

            return this->grabbed_input;
        }

        void set_grab_input(bool grab_input) override {
            if (!this->is_open())
                return;

            if (grab_input) {
                XGrabPointer(
                    this->display_handle,
                    this->window_handle,
                    True,
                    0,
                    GrabModeAsync,
                    GrabModeAsync,
                    this->window_handle,
                    None,
                    CurrentTime
                );
                XGrabKeyboard(
                    this->display_handle,
                    this->window_handle,
                    True,
                    GrabModeAsync,
                    GrabModeAsync,
                    CurrentTime
                );
            }
            else {
                XUngrabPointer(this->display_handle, CurrentTime);
                XUngrabKeyboard(this->display_handle, CurrentTime);
            }

            this->grabbed_input = grab_input;
        }

        position_type get_position() const override {
            if (!this->is_open())
                return { 0, 0 };
            XWindowAttributes window_attributes;
            XGetWindowAttributes(this->display_handle, this->window_handle, &window_attributes);
            return { window_attributes.x, window_attributes.y };
        }

        void set_position(const position_type& position) override {
            if (!this->is_open())
                return;
            XMoveWindow(this->display_handle, this->window_handle, position.x, position.y);
        }

        size_type get_size() const override {
            if (!this->is_open())
                return { 0, 0 };
            XWindowAttributes window_attributes;
            XGetWindowAttributes(this->display_handle, this->window_handle, &window_attributes);
            return { window_attributes.width, window_attributes.height };
        }

        void set_size(const size_type& size) override {
            if (!this->is_open())
                return;
            XResizeWindow(this->display_handle, this->window_handle, static_cast<unsigned int>(size.width), static_cast<unsigned int>(size.height));
        }

        std::string get_title() const override {
            if (!this->is_open())
                return {};
            char* name = nullptr;
            if ((XFetchName(this->display_handle, this->window_handle, &name) == 0) || (name == nullptr))
                return {};
            std::string title(name);
            XFree(name);
            return title;
        }

        void set_title(const std::string& new_title) override {
            if (!this->is_open())
                return;
            XClassHint class_hint{};
            class_hint.res_name = const_cast<char*>(new_title.c_str());
            class_hint.res_class = const_cast<char*>(new_title.c_str());
            XSetClassHint(this->display_handle, this->window_handle, &class_hint);
            XStoreName(this->display_handle, this->window_handle, new_title.c_str());
        }

        void get_icon(int& width, int& height, std::vector<unsigned int>& data) const override {
            static_cast<void>(width);
            static_cast<void>(height);
            static_cast<void>(data);
            // TODO: Get the icon
        }

        void set_icon(int width, int height, const std::vector<unsigned int>& data) override {
            if (static_cast<size_t>(width) * static_cast<size_t>(height) != data.size()) {
                return;
            }
            std::vector<unsigned long int> icon_data;
            icon_data.reserve(2 + data.size());
            icon_data.push_back(static_cast<unsigned long int>(width));
            icon_data.push_back(static_cast<unsigned long int>(height));
            icon_data.insert(icon_data.end(), data.begin(), data.end());
            XChangeProperty(
                this->display_handle,
                this->window_handle,
                XInternAtom(this->display_handle, "_NET_WM_ICON", False),
                XA_CARDINAL,
                32,
                PropModeReplace,
                reinterpret_cast<unsigned char*>(icon_data.data()),
                static_cast<int>(icon_data.size())
            );
        }

        bool process(event_type& event) override {
            if (!this->is_open())
                return false;

            XEvent x11_event = {};

            auto is_window_event = [](Display*, XEvent* current_event, XPointer userData) -> Bool {
                return (current_event->type == MappingNotify) || (current_event->xany.window == reinterpret_cast<Window>(userData));
            };

            while (XCheckIfEvent(this->display_handle, &x11_event, is_window_event, reinterpret_cast<XPointer>(this->window_handle)) != 0) {
                if (XFilterEvent(&x11_event, this->window_handle) == True) {
                    continue;
                }

                switch (x11_event.type) {
                    case ClientMessage: {
                        if ((x11_event.xclient.format == 32) && (x11_event.xclient.data.l[0] == static_cast<long>(this->close_handler))) {
                            this->close();
                            return false;
                        }
                        break;
                    }

                    case MappingNotify: {
                        XRefreshKeyboardMapping(&x11_event.xmapping);
                        break;
                    }

                    case ConfigureNotify: {
                        event.input = event_type::input_type::configuration;
                        event.data.configuration = event_type::configuration_type{ x11_event.xconfigure.x, x11_event.xconfigure.y, x11_event.xconfigure.width, x11_event.xconfigure.height };
                        return true;
                    }

                    case FocusIn: {
                        event.input = event_type::input_type::focus;
                        event.data.focus = event_type::focus_type::gained;
                        return true;
                    }

                    case FocusOut: {
                        event.input = event_type::input_type::focus;
                        event.data.focus = event_type::focus_type::lost;
                        return true;
                    }

                    case KeyPress:
                    case KeyRelease: {
                        event.input = (x11_event.type == KeyPress) ? event_type::input_type::key_press : event_type::input_type::key_release;
                        char buffer[32];
                        KeySym symbol = NoSymbol;
                        XLookupString(&x11_event.xkey, buffer, sizeof(buffer), &symbol, nullptr);
                        event.data.key = static_cast<event_type::key_type>(symbol);
                        return true;
                    }

                    case ButtonPress:
                    case ButtonRelease: {
                        if ((x11_event.xbutton.button == Button4) || (x11_event.xbutton.button == Button5)) {
                            if (x11_event.type != ButtonPress) {
                                continue;
                            }
                            event.input = event_type::input_type::mouse_scroll;
                            event.data.scroll = (x11_event.xbutton.button == Button4) ? 1 : -1;
                            return true;
                        }

                        event.input = (x11_event.type == ButtonPress) ? event_type::input_type::mouse_press : event_type::input_type::mouse_release;

                        switch (x11_event.xbutton.button) {
                            case Button1:
                                event.data.mouse = event_type::mouse_type::left;
                                break;
                            case Button2:
                                event.data.mouse = event_type::mouse_type::middle;
                                break;
                            case Button3:
                                event.data.mouse = event_type::mouse_type::right;
                                break;
                            default:
                                event.data.mouse = event_type::mouse_type::unknown;
                                break;
                        }
                        return true;
                    }

                    case MotionNotify: {
                        event.input = event_type::input_type::mouse_move;
                        event.data.motion.x = x11_event.xmotion.x;
                        event.data.motion.y = x11_event.xmotion.y;
                        return true;
                    }
                }
            }

            return false;
        }

        void prepare() override {
            if (!this->is_open())
                return;
            if (this->is_hardware_accelerated()) {
                glXMakeCurrent(this->display_handle, this->window_handle, this->opengl_context);
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

            size_type size = this->get_size();

            char* buffer = static_cast<char*>(malloc(static_cast<size_t>(size.width * size.height * 4)));
            XImage* image = XCreateImage(
                this->display_handle,
                DefaultVisual(this->display_handle, DefaultScreen(this->display_handle)),
                static_cast<unsigned int>(DefaultDepth(this->display_handle, DefaultScreen(this->display_handle))),
                ZPixmap,
                0,
                buffer,
                static_cast<unsigned int>(size.width),
                static_cast<unsigned int>(size.height),
                32,
                0
            );

            const int draw_width = (width < size.width) ? width : size.width;
            const int draw_height = (height < size.height) ? height : size.height;
            const int data_index_r = (step == 1) ? 0 : 0;
            const int data_index_g = (step == 1) ? 0 : 1;
            const int data_index_b = (step == 1) ? 0 : 2;

            for (int y = 0; y < draw_height; ++y) {
                const unsigned char* __restrict data_line = data + y * stride;
                for (int x = 0; x < draw_width; ++x, data_line += step) {
                    unsigned long pixel = ((static_cast<unsigned long>(data_line[data_index_r]) << 16) | (static_cast<unsigned long>(data_line[data_index_g]) << 8) | (static_cast<unsigned long>(data_line[data_index_b]) << 0));
                    XPutPixel(image, x, y, pixel);
                }
                for (int x = draw_width; x < size.width; ++x) {
                    XPutPixel(image, x, y, 0);
                }
            }

            for (int y = draw_height; y < size.height; ++y) {
                for (int x = 0; x < size.width; ++x) {
                    XPutPixel(image, x, y, 0);
                }
            }

            GC graphics_context = XCreateGC(this->display_handle, this->window_handle, 0, nullptr);
            XPutImage(this->display_handle, this->window_handle, graphics_context, image, 0, 0, 0, 0, static_cast<unsigned int>(size.width), static_cast<unsigned int>(size.height));
            XFreeGC(this->display_handle, graphics_context);
            XDestroyImage(image);
        }

        void present() override {
            if (!this->is_open())
                return;
            if (this->is_hardware_accelerated()) {
                glXSwapBuffers(this->display_handle, this->window_handle);
            }
            XFlush(this->display_handle);
        }

        bool read(unsigned char* __restrict const rgb, const int width, const int height, const int step, const int stride) override {
            if (!this->is_open())
                return false;
            if (this->is_hardware_accelerated())
                return false;
            if (!this->is_visible())
                return false;

            size_type size = this->get_size();

            XFlush(this->display_handle);

            XWindowAttributes window_attributes;
            Status status = XGetWindowAttributes(this->display_handle, this->window_handle, &window_attributes);
            if (status == 0 || window_attributes.map_state != IsViewable)
                return false;

            XImage* image = XGetImage(
                this->display_handle,
                this->window_handle,
                0,
                0,
                static_cast<unsigned int>(size.width),
                static_cast<unsigned int>(size.height),
                AllPlanes,
                ZPixmap
            );

            if (!image)
                return false;

            const int read_width = (width < size.width) ? width : size.width;
            const int read_height = (height < size.height) ? height : size.height;

            for (int y = 0; y < read_height; ++y) {
                unsigned char* __restrict rgb_line = rgb + y * stride;
                unsigned char* __restrict image_line = reinterpret_cast<unsigned char*>(image->data) + y * image->bytes_per_line;
                for (int x = 0; x < read_width; ++x, rgb_line += step, image_line += 4) {
                    rgb_line[0] = image_line[2];
                    rgb_line[1] = image_line[1];
                    rgb_line[2] = image_line[0];
                }
            }

            XDestroyImage(image);
            return true;
        }
    };
}

namespace gtl {
    window::driver_type* window::create_driver_linux_x11() {
        Display* display = XOpenDisplay(nullptr);
        if (!display) {
            return nullptr;
        }
        XCloseDisplay(display);
        return new x11_driver();
    }
}
#else
namespace gtl {
    window::driver_type* window::create_driver_linux_x11() {
        return nullptr;
    }
}
#endif
