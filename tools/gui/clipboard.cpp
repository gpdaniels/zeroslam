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

#include "clipboard.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#if defined(linux) || defined(__linux) || defined(__linux__)
#define GTL_CLIPBOARD_DRIVER_LINUX_X11 0
#if __has_include(<X11/Xlib.h>)
#undef GTL_CLIPBOARD_DRIVER_LINUX_X11
#define GTL_CLIPBOARD_DRIVER_LINUX_X11 1

#include <X11/Xatom.h>
#include <X11/Xlib.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <mutex>
#include <sys/select.h>
#include <thread>
#include <unistd.h>
#endif
#endif

#if defined(_WIN32) || defined(_WIN64)
#undef GTL_CLIPBOARD_DRIVER_WINDOWS_WIN32
#define GTL_CLIPBOARD_DRIVER_WINDOWS_WIN32 1

#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#define STRICT

#include <Windows.h>
#endif

#if defined(__APPLE__)
#undef GTL_CLIPBOARD_DRIVER_APPLE
#define GTL_CLIPBOARD_DRIVER_APPLE 1

#include <objc/NSObjCRuntime.h>
#include <objc/objc-runtime.h>
#include <objc/objc.h>
#endif

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace gtl {
#if defined(GTL_CLIPBOARD_DRIVER_LINUX_X11) && GTL_CLIPBOARD_DRIVER_LINUX_X11
    namespace {
        class selection_owner final {
        private:
            struct atom_set final {
                Atom clipboard;
                Atom targets;
                Atom utf8;
                Atom text_plain;
                Atom text_plain_utf8;
                Atom manager;
                Atom save_targets;
            };

            static inline std::atomic<Display*> owner_display{ nullptr };

            static inline int (*previous_error_handler)(Display*, XErrorEvent*) = nullptr;

            static inline bool error_handler_installed = false;

            std::mutex mutex;

            std::condition_variable acknowledged;

            std::thread thread;

            std::string text;

            unsigned long long requested_generation = 0;

            unsigned long long served_generation = 0;

            bool owning = false;

            bool running = false;

            bool stopping = false;

            int wake[2] = { -1, -1 };

        private:
            static int ignore_owner_errors(Display* display, XErrorEvent* error) {
                if (display == selection_owner::owner_display.load()) {
                    return 0;
                }
                return (selection_owner::previous_error_handler != nullptr) ? selection_owner::previous_error_handler(display, error) : 0;
            }

            static bool is_text_target(const Atom target, const atom_set& atoms) {
                return (target == atoms.utf8) || (target == atoms.text_plain_utf8) || (target == atoms.text_plain) || (target == XA_STRING);
            }

            static void answer(Display* display, const XSelectionRequestEvent& request, const std::string& served, const atom_set& atoms) {
                const Atom property = (request.property != None) ? request.property : request.target;
                XEvent response{};
                response.xselection.type = SelectionNotify;
                response.xselection.display = request.display;
                response.xselection.requestor = request.requestor;
                response.xselection.selection = request.selection;
                response.xselection.target = request.target;
                response.xselection.property = property;
                response.xselection.time = request.time;
                if (request.target == atoms.targets) {
                    Atom list[5] = { atoms.targets, atoms.utf8, atoms.text_plain, atoms.text_plain_utf8, XA_STRING };
                    XChangeProperty(display, request.requestor, property, XA_ATOM, 32, PropModeReplace, reinterpret_cast<unsigned char*>(&list[0]), 5);
                }
                else if (selection_owner::is_text_target(request.target, atoms)) {
                    XChangeProperty(display, request.requestor, property, request.target, 8, PropModeReplace, reinterpret_cast<const unsigned char*>(served.data()), static_cast<int>(served.size()));
                }
                else {
                    response.xselection.property = None;
                }
                XSendEvent(display, request.requestor, False, 0, &response);
                XFlush(display);
            }

            bool wait_for_events(Display* display, const long timeout_microseconds) const {
                const int connection = ConnectionNumber(display);
                fd_set descriptors;
                FD_ZERO(&descriptors);
                FD_SET(connection, &descriptors);
                FD_SET(this->wake[0], &descriptors);
                struct timeval timeout;
                timeout.tv_sec = timeout_microseconds / 1000000;
                timeout.tv_usec = timeout_microseconds % 1000000;
                const int ready = select(std::max(connection, this->wake[0]) + 1, &descriptors, nullptr, nullptr, (timeout_microseconds >= 0) ? &timeout : nullptr);
                if ((ready > 0) && FD_ISSET(this->wake[0], &descriptors)) {
                    char drained[64];
                    while (::read(this->wake[0], &drained[0], sizeof(drained)) > 0) {
                    }
                }
                return (ready > 0) || ((ready < 0) && (errno == EINTR));
            }

            void hand_over(Display* display, const Window window, const std::string& served, const atom_set& atoms) const {
                if (XGetSelectionOwner(display, atoms.manager) == None) {
                    return;
                }
                XConvertSelection(display, atoms.manager, atoms.save_targets, None, window, CurrentTime);
                XFlush(display);
                const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
                while (std::chrono::steady_clock::now() < deadline) {
                    while (XPending(display) > 0) {
                        XEvent event;
                        XNextEvent(display, &event);
                        if (event.type == SelectionRequest) {
                            selection_owner::answer(display, event.xselectionrequest, served, atoms);
                        }
                        else if ((event.type == SelectionNotify) && (event.xselection.selection == atoms.manager)) {
                            return;
                        }
                    }
                    const long remaining_microseconds = static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(deadline - std::chrono::steady_clock::now()).count());
                    if ((remaining_microseconds <= 0) || !this->wait_for_events(display, remaining_microseconds)) {
                        return;
                    }
                }
            }

            void finish(const bool owned) {
                std::lock_guard<std::mutex> lock(this->mutex);
                this->owning = owned;
                this->running = false;
                this->served_generation = this->requested_generation;
                this->acknowledged.notify_all();
            }

            void serve() {
                Display* const display = XOpenDisplay(nullptr);
                if (display == nullptr) {
                    this->finish(false);
                    return;
                }
                selection_owner::owner_display.store(display);
                if (!selection_owner::error_handler_installed) {
                    selection_owner::previous_error_handler = XSetErrorHandler(&selection_owner::ignore_owner_errors);
                    selection_owner::error_handler_installed = true;
                }

                const Window window = XCreateSimpleWindow(display, DefaultRootWindow(display), 0, 0, 1, 1, 0, 0, 0);
                atom_set atoms;
                atoms.clipboard = XInternAtom(display, "CLIPBOARD", False);
                atoms.targets = XInternAtom(display, "TARGETS", False);
                atoms.utf8 = XInternAtom(display, "UTF8_STRING", False);
                atoms.text_plain = XInternAtom(display, "text/plain", False);
                atoms.text_plain_utf8 = XInternAtom(display, "text/plain;charset=utf-8", False);
                atoms.manager = XInternAtom(display, "CLIPBOARD_MANAGER", False);
                atoms.save_targets = XInternAtom(display, "SAVE_TARGETS", False);

                std::string served;
                bool owned = false;
                while (true) {
                    {
                        std::lock_guard<std::mutex> lock(this->mutex);
                        if (this->stopping) {
                            break;
                        }
                        if (this->served_generation != this->requested_generation) {
                            served = this->text;
                            XSetSelectionOwner(display, atoms.clipboard, window, CurrentTime);
                            owned = (XGetSelectionOwner(display, atoms.clipboard) == window);
                            this->owning = owned;
                            this->served_generation = this->requested_generation;
                            this->acknowledged.notify_all();
                        }
                    }
                    while (XPending(display) > 0) {
                        XEvent event;
                        XNextEvent(display, &event);
                        if (event.type == SelectionRequest) {
                            selection_owner::answer(display, event.xselectionrequest, served, atoms);
                        }
                        else if ((event.type == SelectionClear) && (event.xselectionclear.selection == atoms.clipboard)) {
                            owned = false;
                            std::lock_guard<std::mutex> lock(this->mutex);
                            this->owning = false;
                        }
                    }
                    this->wait_for_events(display, -1);
                }

                if (owned) {
                    this->hand_over(display, window, served, atoms);
                }
                XDestroyWindow(display, window);
                XCloseDisplay(display);
                selection_owner::owner_display.store(nullptr);
                this->finish(false);
            }

            void notify() const {
                if (this->wake[1] >= 0) {
                    const char wake_byte = 1;
                    const ssize_t sent = ::write(this->wake[1], &wake_byte, 1);
                    static_cast<void>(sent);
                }
            }

        public:
            ~selection_owner() {
                {
                    std::lock_guard<std::mutex> lock(this->mutex);
                    this->stopping = true;
                }
                this->notify();
                if (this->thread.joinable()) {
                    this->thread.join();
                }
                for (const int descriptor : this->wake) {
                    if (descriptor >= 0) {
                        close(descriptor);
                    }
                }
            }

            selection_owner() = default;

            selection_owner(const selection_owner&) = delete;

            selection_owner(selection_owner&&) = delete;

            selection_owner& operator=(const selection_owner&) = delete;

            selection_owner& operator=(selection_owner&&) = delete;

        public:
            bool publish(const std::string& new_text) {
                std::unique_lock<std::mutex> lock(this->mutex);
                if (this->wake[0] < 0) {
                    if (pipe(&this->wake[0]) != 0) {
                        this->wake[0] = -1;
                        this->wake[1] = -1;
                        return false;
                    }
                    for (const int descriptor : this->wake) {
                        fcntl(descriptor, F_SETFL, fcntl(descriptor, F_GETFL) | O_NONBLOCK);
                        fcntl(descriptor, F_SETFD, FD_CLOEXEC);
                    }
                }
                if (!this->running) {
                    if (this->thread.joinable()) {
                        this->thread.join();
                    }
                    this->running = true;
                    this->thread = std::thread(&selection_owner::serve, this);
                }
                this->text = new_text;
                const unsigned long long generation = ++this->requested_generation;
                this->notify();
                this->acknowledged.wait_for(lock, std::chrono::seconds(1), [&]() {
                    return (this->served_generation >= generation) || !this->running;
                });
                return (this->served_generation >= generation) && this->owning;
            }
        };

        selection_owner& clipboard_owner() {
            static selection_owner owner;
            return owner;
        }
    }
#endif

    bool clipboard::read(std::string& text) {
        text.clear();
#if defined(GTL_CLIPBOARD_DRIVER_LINUX_X11) && GTL_CLIPBOARD_DRIVER_LINUX_X11
        Display* display = XOpenDisplay(nullptr);
        if (!display) {
            return false;
        }

        int screen = DefaultScreen(display);
        Window root = RootWindow(display, screen);

        Atom clipboard_atom = XInternAtom(display, "CLIPBOARD", False);
        Atom utf8_atom = XInternAtom(display, "UTF8_STRING", False);

        Window window = XGetSelectionOwner(display, clipboard_atom);

        if (window == None) {
            window = XGetSelectionOwner(display, XInternAtom(display, "PRIMARY", False));
            clipboard_atom = XInternAtom(display, "PRIMARY", False);
        }

        if (window == None) {
            XCloseDisplay(display);
            return false;
        }

        Window target_window = XCreateSimpleWindow(display, root, -10, -10, 1, 1, 0, 0, 0);
        Atom target_property = XInternAtom(display, "GTL_CLIPBOARD_DATA", False);

        XConvertSelection(display, clipboard_atom, utf8_atom, target_property, target_window, CurrentTime);
        XFlush(display);

        constexpr static const long maximum_wait_microseconds = 1000000;
        const int connection = ConnectionNumber(display);
        XEvent event;
        bool selection_received = false;
        long remaining_microseconds = maximum_wait_microseconds;
        while (!selection_received && (remaining_microseconds > 0)) {
            while (XPending(display) > 0) {
                XNextEvent(display, &event);
                if (event.type == SelectionNotify) {
                    selection_received = true;
                    break;
                }
            }
            if (selection_received) {
                break;
            }

            fd_set descriptors;
            FD_ZERO(&descriptors);
            FD_SET(connection, &descriptors);
            struct timeval timeout;
            timeout.tv_sec = remaining_microseconds / 1000000;
            timeout.tv_usec = remaining_microseconds % 1000000;

            const int ready = select(connection + 1, &descriptors, nullptr, nullptr, &timeout);
            remaining_microseconds = (timeout.tv_sec * 1000000) + timeout.tv_usec;
            if (ready < 0) {
                if (errno == EINTR) {
                    continue;
                }
                break;
            }
            if (ready == 0) {
                break;
            }
        }

        if (!selection_received) {
            XDestroyWindow(display, target_window);
            XCloseDisplay(display);
            return false;
        }

        if (event.xselection.property == None) {
            XDestroyWindow(display, target_window);
            XCloseDisplay(display);
            return false;
        }

        Atom type;
        int format;
        unsigned long nitems, bytes_after;
        unsigned char* data = nullptr;

        Status status = XGetWindowProperty(
            display,
            target_window,
            target_property,
            0,
            65536,
            False,
            AnyPropertyType,
            &type,
            &format,
            &nitems,
            &bytes_after,
            &data
        );

        if (status != Success || data == nullptr) {
            if (data)
                XFree(data);
            XDestroyWindow(display, target_window);
            XCloseDisplay(display);
            return false;
        }

        if (type == utf8_atom || type == XA_STRING) {
            text.assign(reinterpret_cast<char*>(data), nitems);
        }

        XFree(data);
        XDeleteProperty(display, target_window, target_property);
        XDestroyWindow(display, target_window);
        XCloseDisplay(display);
        return true;

#elif defined(GTL_CLIPBOARD_DRIVER_WINDOWS_WIN32) && GTL_CLIPBOARD_DRIVER_WINDOWS_WIN32
        if (!OpenClipboard(nullptr)) {
            return false;
        }

        HANDLE data = GetClipboardData(CF_TEXT);
        if (!data) {
            CloseClipboard();
            return false;
        }

        char* clipboard_text = static_cast<char*>(GlobalLock(data));
        if (!clipboard_text) {
            GlobalUnlock(data);
            CloseClipboard();
            return false;
        }

        text = clipboard_text;

        GlobalUnlock(data);
        CloseClipboard();
        return true;

#elif defined(GTL_CLIPBOARD_DRIVER_APPLE) && GTL_CLIPBOARD_DRIVER_APPLE
        static id (*const msg_send)(id, SEL) = reinterpret_cast<id (*)(id, SEL)>(reinterpret_cast<void*>(objc_msgSend));
        static void (*const msg_send_void)(id, SEL) = reinterpret_cast<void (*)(id, SEL)>(reinterpret_cast<void*>(objc_msgSend));
        static id (*const msg_send_class)(Class, SEL) = reinterpret_cast<id (*)(Class, SEL)>(reinterpret_cast<void*>(objc_msgSend));
        static id (*const msg_send_class_utf8)(Class, SEL, const char*) = reinterpret_cast<id (*)(Class, SEL, const char*)>(reinterpret_cast<void*>(objc_msgSend));
        static id (*const msg_send_id)(id, SEL, id) = reinterpret_cast<id (*)(id, SEL, id)>(reinterpret_cast<void*>(objc_msgSend));
        static const char* (*const msg_send_utf8)(id, SEL) = reinterpret_cast<const char* (*)(id, SEL)>(reinterpret_cast<void*>(objc_msgSend));

        static Class ns_autorelease_pool_class = objc_getClass("NSAutoreleasePool");
        static Class ns_string_class = objc_getClass("NSString");
        static Class ns_pasteboard_class = objc_getClass("NSPasteboard");
        static SEL sel_alloc = sel_getUid("alloc");
        static SEL sel_init = sel_getUid("init");
        static SEL sel_drain = sel_getUid("drain");
        static SEL sel_string_with_utf8 = sel_getUid("stringWithUTF8String:");
        static SEL sel_general_pasteboard = sel_getUid("generalPasteboard");
        static SEL sel_string_for_type = sel_getUid("stringForType:");
        static SEL sel_utf8 = sel_getUid("UTF8String");

        id pool = msg_send(msg_send_class(ns_autorelease_pool_class, sel_alloc), sel_init);

        bool success = false;
        id general_pasteboard = msg_send_class(ns_pasteboard_class, sel_general_pasteboard);
        id string_type = msg_send_class_utf8(ns_string_class, sel_string_with_utf8, "public.utf8-plain-text");
        if (general_pasteboard && string_type) {
            id clipboard_string = msg_send_id(general_pasteboard, sel_string_for_type, string_type);
            const char* utf8_string = clipboard_string ? msg_send_utf8(clipboard_string, sel_utf8) : nullptr;
            if (utf8_string) {
                text = utf8_string;
                success = true;
            }
        }

        msg_send_void(pool, sel_drain);
        return success;
#else
        return false;
#endif
    }

    bool clipboard::write(const std::string& text) {
#if defined(GTL_CLIPBOARD_DRIVER_LINUX_X11) && GTL_CLIPBOARD_DRIVER_LINUX_X11
        return clipboard_owner().publish(text);

#elif defined(GTL_CLIPBOARD_DRIVER_WINDOWS_WIN32) && GTL_CLIPBOARD_DRIVER_WINDOWS_WIN32
        if (!OpenClipboard(nullptr)) {
            return false;
        }

        if (!EmptyClipboard()) {
            CloseClipboard();
            return false;
        }

        HGLOBAL global_memory = GlobalAlloc(GMEM_MOVEABLE, text.size() + 1);
        if (!global_memory) {
            CloseClipboard();
            return false;
        }

        char* clipboard_text = static_cast<char*>(GlobalLock(global_memory));
        if (!clipboard_text) {
            GlobalFree(global_memory);
            CloseClipboard();
            return false;
        }

        memcpy(clipboard_text, text.data(), text.size());
        clipboard_text[text.size()] = '\0';

        GlobalUnlock(global_memory);

        if (!SetClipboardData(CF_TEXT, global_memory)) {
            GlobalFree(global_memory);
            CloseClipboard();
            return false;
        }

        CloseClipboard();
        return true;

#elif defined(GTL_CLIPBOARD_DRIVER_APPLE) && GTL_CLIPBOARD_DRIVER_APPLE
        static id (*const msg_send)(id, SEL) = reinterpret_cast<id (*)(id, SEL)>(reinterpret_cast<void*>(objc_msgSend));
        static void (*const msg_send_void)(id, SEL) = reinterpret_cast<void (*)(id, SEL)>(reinterpret_cast<void*>(objc_msgSend));
        static NSInteger (*const msg_send_integer)(id, SEL) = reinterpret_cast<NSInteger (*)(id, SEL)>(reinterpret_cast<void*>(objc_msgSend));
        static id (*const msg_send_class)(Class, SEL) = reinterpret_cast<id (*)(Class, SEL)>(reinterpret_cast<void*>(objc_msgSend));
        static id (*const msg_send_class_utf8)(Class, SEL, const char*) = reinterpret_cast<id (*)(Class, SEL, const char*)>(reinterpret_cast<void*>(objc_msgSend));
        static BOOL (*const msg_send_bool_id_id)(id, SEL, id, id) = reinterpret_cast<BOOL (*)(id, SEL, id, id)>(reinterpret_cast<void*>(objc_msgSend));

        static Class ns_autorelease_pool_class = objc_getClass("NSAutoreleasePool");
        static Class ns_string_class = objc_getClass("NSString");
        static Class ns_pasteboard_class = objc_getClass("NSPasteboard");
        static SEL sel_alloc = sel_getUid("alloc");
        static SEL sel_init = sel_getUid("init");
        static SEL sel_drain = sel_getUid("drain");
        static SEL sel_string_with_utf8 = sel_getUid("stringWithUTF8String:");
        static SEL sel_general_pasteboard = sel_getUid("generalPasteboard");
        static SEL sel_clear = sel_getUid("clearContents");
        static SEL sel_set_string = sel_getUid("setString:forType:");

        id pool = msg_send(msg_send_class(ns_autorelease_pool_class, sel_alloc), sel_init);

        bool success = false;
        id general_pasteboard = msg_send_class(ns_pasteboard_class, sel_general_pasteboard);
        id string_type = msg_send_class_utf8(ns_string_class, sel_string_with_utf8, "public.utf8-plain-text");
        id clipboard_string = msg_send_class_utf8(ns_string_class, sel_string_with_utf8, text.c_str());
        if (general_pasteboard && string_type && clipboard_string) {
            static_cast<void>(msg_send_integer(general_pasteboard, sel_clear));
            success = static_cast<bool>(msg_send_bool_id_id(general_pasteboard, sel_set_string, clipboard_string, string_type));
        }

        msg_send_void(pool, sel_drain);
        return success;
#else
        static_cast<void>(text);
        return false;
#endif
    }
}
