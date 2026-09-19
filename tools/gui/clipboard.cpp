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
#include <cerrno>
#include <csignal>
#include <cstring>
#include <sys/select.h>
#include <sys/wait.h>
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
        pid_t g_clipboard_owner_pid = 0;

        void reap_clipboard_owner() {
            if (g_clipboard_owner_pid > 0) {
                if (waitpid(g_clipboard_owner_pid, nullptr, WNOHANG) == g_clipboard_owner_pid) {
                    g_clipboard_owner_pid = 0;
                }
            }
        }

        void retire_clipboard_owner(const pid_t pid) {
            if (pid > 0) {
                kill(pid, SIGTERM);
                waitpid(pid, nullptr, 0);
            }
        }

        void host_clipboard(const std::string& text) {
            reap_clipboard_owner();

            const pid_t outgoing_owner_pid = g_clipboard_owner_pid;

            int handshake[2] = { -1, -1 };
            if (pipe(&handshake[0]) != 0) {
                handshake[0] = -1;
                handshake[1] = -1;
            }

            pid_t pid = fork();
            if (pid < 0) {
                if (handshake[0] >= 0) {
                    close(handshake[0]);
                    close(handshake[1]);
                }
                return;
            }
            if (pid > 0) {
                g_clipboard_owner_pid = pid;
                if (handshake[0] >= 0) {
                    close(handshake[1]);
                    constexpr static const long maximum_wait_microseconds = 1000000;
                    long remaining_microseconds = maximum_wait_microseconds;
                    while (remaining_microseconds > 0) {
                        fd_set descriptors;
                        FD_ZERO(&descriptors);
                        FD_SET(handshake[0], &descriptors);
                        struct timeval timeout;
                        timeout.tv_sec = remaining_microseconds / 1000000;
                        timeout.tv_usec = remaining_microseconds % 1000000;
                        const int ready = select(handshake[0] + 1, &descriptors, nullptr, nullptr, &timeout);
                        remaining_microseconds = (timeout.tv_sec * 1000000) + timeout.tv_usec;
                        if ((ready < 0) && (errno == EINTR)) {
                            continue;
                        }
                        if (ready > 0) {
                            char acknowledgement = 0;
                            const ssize_t received = ::read(handshake[0], &acknowledgement, 1);
                            static_cast<void>(received);
                        }
                        break;
                    }
                    close(handshake[0]);
                }
                retire_clipboard_owner(outgoing_owner_pid);
                return;
            }

            if (handshake[0] >= 0) {
                close(handshake[0]);
            }

            Display* d = XOpenDisplay(nullptr);
            if (!d) {
                _exit(1);
            }

            Window w = XCreateSimpleWindow(d, DefaultRootWindow(d), 0, 0, 1, 1, 0, 0, 0);

            Atom clipboard = XInternAtom(d, "CLIPBOARD", False);
            Atom targets = XInternAtom(d, "TARGETS", False);
            Atom utf8 = XInternAtom(d, "UTF8_STRING", False);
            Atom text_plain = XInternAtom(d, "text/plain", False);
            Atom text_plain_utf8 = XInternAtom(d, "text/plain;charset=utf-8", False);

            XSetSelectionOwner(d, clipboard, w, CurrentTime);
            XFlush(d);

            static_cast<void>(XGetSelectionOwner(d, clipboard));
            if (handshake[1] >= 0) {
                const char acknowledgement = 1;
                const ssize_t sent = ::write(handshake[1], &acknowledgement, 1);
                static_cast<void>(sent);
                close(handshake[1]);
            }

            int fd = ConnectionNumber(d);
            fd_set readfds;
            struct timeval tv;
            int max_seconds = 60;
            int elapsed = 0;

            XEvent ev;
            while (elapsed < max_seconds) {
                FD_ZERO(&readfds);
                FD_SET(fd, &readfds);
                tv.tv_sec = 1;
                tv.tv_usec = 0;

                int select_result = select(fd + 1, &readfds, nullptr, nullptr, &tv);
                if (select_result > 0) {
                    while (XPending(d) > 0) {
                        XNextEvent(d, &ev);

                        if (ev.type == SelectionRequest) {
                            XSelectionRequestEvent* req = &ev.xselectionrequest;
                            XEvent res{};
                            res.xselection.type = SelectionNotify;
                            res.xselection.display = req->display;
                            res.xselection.requestor = req->requestor;
                            res.xselection.selection = req->selection;
                            res.xselection.target = req->target;
                            res.xselection.property = req->property;
                            res.xselection.time = req->time;

                            if (req->target == utf8 || req->target == text_plain_utf8 || req->target == text_plain || req->target == XA_STRING) {
                                XChangeProperty(d, req->requestor, req->property, utf8, 8, PropModeReplace, reinterpret_cast<const unsigned char*>(text.c_str()), static_cast<int>(text.size()));
                            }
                            else if (req->target == targets) {
                                Atom list[5] = { targets, utf8, text_plain, text_plain_utf8, XA_STRING };
                                XChangeProperty(d, req->requestor, req->property, XA_ATOM, 32, PropModeReplace, reinterpret_cast<unsigned char*>(list), 5);
                            }
                            else {
                                res.xselection.property = None;
                            }

                            XSendEvent(d, req->requestor, False, 0, &res);
                            XFlush(d);
                        }

                        if (ev.type == SelectionClear) {
                            elapsed = max_seconds;
                            break;
                        }
                    }
                }
                else {
                    elapsed++;
                }
            }

            XCloseDisplay(d);
            _exit(0);
        }
    }
#endif

    bool clipboard::read(std::string& text) {
        text.clear();
#if defined(GTL_CLIPBOARD_DRIVER_LINUX_X11) && GTL_CLIPBOARD_DRIVER_LINUX_X11
        reap_clipboard_owner();

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
        host_clipboard(text);
        return true;

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
