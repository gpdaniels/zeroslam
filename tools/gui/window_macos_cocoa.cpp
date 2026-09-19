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

#if defined(__APPLE__)
#ifndef NDEBUG
#if defined(_MSC_VER)
#define __builtin_trap() __debugbreak()
#endif
#define GTL_WINDOW_ASSERT(ASSERTION, MESSAGE) static_cast<void>((ASSERTION) || (__builtin_trap(), 0))
#else
#define GTL_WINDOW_ASSERT(ASSERTION, MESSAGE) static_cast<void>(0)
#endif

#include "opengl.hpp"

#include <CoreGraphics/CoreGraphics.h>
#include <cstdint>
#include <deque>
#include <objc/NSObjCRuntime.h>
#include <objc/objc-runtime.h>
#include <objc/objc.h>
#include <string>
#include <unordered_map>
#include <vector>

extern id NSApp;
extern const id NSDefaultRunLoopMode;

namespace {
    constexpr BOOL objc_yes = 1;
    constexpr BOOL objc_no = 0;

    template <typename function_type>
    struct msg;

    template <typename return_type, typename... argument_types>
    struct msg<return_type(argument_types...)> final {
        static return_type send(argument_types... arguments) {
            using pointer_type = return_type (*)(argument_types...);
            return reinterpret_cast<pointer_type>(reinterpret_cast<void*>(objc_msgSend))(arguments...);
        }
    };

    template <typename... argument_types>
    struct msg<CGRect(argument_types...)> final {
        static CGRect send(argument_types... arguments) {
            using pointer_type = CGRect (*)(argument_types...);
#if defined(__arm64__)
            return reinterpret_cast<pointer_type>(reinterpret_cast<void*>(objc_msgSend))(arguments...);
#else
            return reinterpret_cast<pointer_type>(reinterpret_cast<void*>(objc_msgSend_stret))(arguments...);
#endif
        }
    };

    SEL sel(const char* name) {
        return sel_registerName(name);
    }

    Class cls(const char* name) {
        return objc_getClass(name);
    }

    template <typename function_type>
    IMP imp(function_type function) {
        return reinterpret_cast<IMP>(reinterpret_cast<void*>(function));
    }

    id string(const char* utf8) {
        return msg<id(Class, SEL, const char*)>::send(cls("NSString"), sel("stringWithUTF8String:"), utf8);
    }

    CGFloat screen_height() {
        id screen = msg<id(Class, SEL)>::send(cls("NSScreen"), sel("mainScreen"));
        if (!screen) {
            return 0;
        }
        return msg<CGRect(id, SEL)>::send(screen, sel("frame")).size.height;
    }

    CGImageRef create_image(const unsigned char* rgba, int width, int height) {
        CGColorSpaceRef colour_space = CGColorSpaceCreateDeviceRGB();
        CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, rgba, static_cast<size_t>(width) * static_cast<size_t>(height) * 4, nullptr);
        CGImageRef image = CGImageCreate(
            static_cast<size_t>(width),
            static_cast<size_t>(height),
            8,
            32,
            static_cast<size_t>(width) * 4,
            colour_space,
            static_cast<CGBitmapInfo>(kCGImageAlphaNoneSkipLast),
            provider,
            nullptr,
            false,
            kCGRenderingIntentDefault
        );
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(colour_space);
        return image;
    }

    constexpr NSUInteger event_left_mouse_down = 1;
    constexpr NSUInteger event_left_mouse_up = 2;
    constexpr NSUInteger event_right_mouse_down = 3;
    constexpr NSUInteger event_right_mouse_up = 4;
    constexpr NSUInteger event_mouse_moved = 5;
    constexpr NSUInteger event_left_mouse_dragged = 6;
    constexpr NSUInteger event_right_mouse_dragged = 7;
    constexpr NSUInteger event_key_down = 10;
    constexpr NSUInteger event_key_up = 11;
    constexpr NSUInteger event_flags_changed = 12;
    constexpr NSUInteger event_scroll_wheel = 22;
    constexpr NSUInteger event_other_mouse_down = 25;
    constexpr NSUInteger event_other_mouse_up = 26;
    constexpr NSUInteger event_other_mouse_dragged = 27;

    constexpr NSUInteger modifier_caps_lock = 1u << 16;
    constexpr NSUInteger modifier_shift = 1u << 17;
    constexpr NSUInteger modifier_control = 1u << 18;
    constexpr NSUInteger modifier_option = 1u << 19;
    constexpr NSUInteger modifier_command = 1u << 20;

    constexpr NSUInteger style_titled = 1u << 0;
    constexpr NSUInteger style_closable = 1u << 1;
    constexpr NSUInteger style_miniaturizable = 1u << 2;
    constexpr NSUInteger style_resizable = 1u << 3;
    constexpr NSUInteger style_fullscreen = 1u << 14;
    constexpr NSUInteger style_standard = style_titled | style_closable | style_miniaturizable | style_resizable;

    constexpr NSUInteger backing_store_buffered = 2;
    constexpr NSInteger activation_policy_regular = 0;
    constexpr NSUInteger terminate_cancel = 0;

    constexpr uint32_t pixel_format_double_buffer = 5;
    constexpr uint32_t pixel_format_colour_size = 8;
    constexpr uint32_t pixel_format_alpha_size = 11;
    constexpr uint32_t pixel_format_depth_size = 12;
    constexpr uint32_t pixel_format_stencil_size = 13;
    constexpr uint32_t pixel_format_accelerated = 73;
    constexpr uint32_t pixel_format_profile = 99;
    constexpr uint32_t profile_legacy = 0x1000;
    constexpr NSInteger context_parameter_swap_interval = 222;

    constexpr char32_t keysym_backspace = 65288;
    constexpr char32_t keysym_tab = 65289;
    constexpr char32_t keysym_return = 65293;
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
    constexpr char32_t keysym_keypad_enter = 65421;
    constexpr char32_t keysym_f1 = 65470;
    constexpr char32_t keysym_shift_left = 65505;
    constexpr char32_t keysym_control_left = 65507;
    constexpr char32_t keysym_caps_lock = 65509;
    constexpr char32_t keysym_alt_left = 65513;
    constexpr char32_t keysym_delete = 65535;
    constexpr char32_t keysym_unicode = 0x01000000;

    constexpr unsigned short function_key_up = 0xF700;
    constexpr unsigned short function_key_down = 0xF701;
    constexpr unsigned short function_key_left = 0xF702;
    constexpr unsigned short function_key_right = 0xF703;
    constexpr unsigned short function_key_f1 = 0xF704;
    constexpr unsigned short function_key_f12 = 0xF70F;
    constexpr unsigned short function_key_insert = 0xF727;
    constexpr unsigned short function_key_delete = 0xF728;
    constexpr unsigned short function_key_home = 0xF729;
    constexpr unsigned short function_key_end = 0xF72B;
    constexpr unsigned short function_key_page_up = 0xF72C;
    constexpr unsigned short function_key_page_down = 0xF72D;

    char32_t translate_key(id event) {
        id characters = msg<id(id, SEL)>::send(event, sel("charactersIgnoringModifiers"));
        if (!characters) {
            return 0;
        }
        const NSUInteger length = msg<NSUInteger(id, SEL)>::send(characters, sel("length"));
        if (length == 0) {
            return 0;
        }
        const unsigned short unit = msg<unsigned short(id, SEL, NSUInteger)>::send(characters, sel("characterAtIndex:"), 0);
        switch (unit) {
            case 0x03:
                return keysym_keypad_enter;
            case 0x08:
            case 0x7F:
                return keysym_backspace;
            case 0x09:
                return keysym_tab;
            case 0x0D:
                return keysym_return;
            case 0x1B:
                return keysym_escape;
            case function_key_up:
                return keysym_up;
            case function_key_down:
                return keysym_down;
            case function_key_left:
                return keysym_left;
            case function_key_right:
                return keysym_right;
            case function_key_insert:
                return keysym_insert;
            case function_key_delete:
                return keysym_delete;
            case function_key_home:
                return keysym_home;
            case function_key_end:
                return keysym_end;
            case function_key_page_up:
                return keysym_page_up;
            case function_key_page_down:
                return keysym_page_down;
            default:
                break;
        }
        if ((unit >= function_key_f1) && (unit <= function_key_f12)) {
            return keysym_f1 + static_cast<char32_t>(unit - function_key_f1);
        }
        if ((unit >= 0xD800) && (unit <= 0xDBFF) && (length >= 2)) {
            const unsigned short low = msg<unsigned short(id, SEL, NSUInteger)>::send(characters, sel("characterAtIndex:"), 1);
            const char32_t code_point = 0x10000 + ((static_cast<char32_t>(unit) - 0xD800) << 10) + (static_cast<char32_t>(low) - 0xDC00);
            return keysym_unicode | code_point;
        }
        if (unit < 0x100) {
            return unit;
        }
        return keysym_unicode | unit;
    }
}

namespace {
    class cocoa_driver final : public gtl::window::driver_type {
    public:
        class application_type final {
        private:
            static inline unsigned int create_count = 0;
            static inline id pool = nullptr;
            static inline id frame_pool = nullptr;
            static inline id delegate = nullptr;

        public:
            static inline std::unordered_map<id, cocoa_driver*> registry;

        public:
            ~application_type() {
                GTL_WINDOW_ASSERT(application_type::create_count != 0, "Application window reference count has become corrupted.");
                if (--application_type::create_count != 0) {
                    return;
                }
                if (application_type::frame_pool) {
                    msg<void(id, SEL)>::send(application_type::frame_pool, sel("drain"));
                    application_type::frame_pool = nullptr;
                }
                if (application_type::pool) {
                    msg<void(id, SEL)>::send(application_type::pool, sel("drain"));
                    application_type::pool = nullptr;
                }
            }

            application_type() {
                if (application_type::create_count++ != 0) {
                    return;
                }

                application_type::pool = msg<id(id, SEL)>::send(msg<id(Class, SEL)>::send(cls("NSAutoreleasePool"), sel("alloc")), sel("init"));

                msg<id(Class, SEL)>::send(cls("NSApplication"), sel("sharedApplication"));
                msg<void(id, SEL, NSInteger)>::send(NSApp, sel("setActivationPolicy:"), activation_policy_regular);

                static Class delegate_class = nullptr;
                if (!delegate_class) {
                    delegate_class = objc_allocateClassPair(cls("NSObject"), "GtlApplicationDelegate", 0);
                    class_addMethod(
                        delegate_class,
                        sel("applicationShouldTerminate:"),
                        imp(static_cast<NSUInteger (*)(id, SEL, id)>([](id, SEL, id) -> NSUInteger {
                            for (std::pair<const id, cocoa_driver*>& entry : application_type::registry) {
                                entry.second->closed = true;
                            }
                            return terminate_cancel;
                        })),
                        "Q@:@"
                    );
                    objc_registerClassPair(delegate_class);
                }
                application_type::delegate = msg<id(id, SEL)>::send(msg<id(Class, SEL)>::send(delegate_class, sel("alloc")), sel("init"));
                msg<void(id, SEL, id)>::send(NSApp, sel("setDelegate:"), application_type::delegate);

                id menubar = msg<id(id, SEL)>::send(msg<id(Class, SEL)>::send(cls("NSMenu"), sel("alloc")), sel("init"));
                id application_item = msg<id(id, SEL)>::send(msg<id(Class, SEL)>::send(cls("NSMenuItem"), sel("alloc")), sel("init"));
                msg<void(id, SEL, id)>::send(menubar, sel("addItem:"), application_item);
                msg<void(id, SEL, id)>::send(NSApp, sel("setMainMenu:"), menubar);

                id application_menu = msg<id(id, SEL)>::send(msg<id(Class, SEL)>::send(cls("NSMenu"), sel("alloc")), sel("init"));
                id process_info = msg<id(Class, SEL)>::send(cls("NSProcessInfo"), sel("processInfo"));
                id process_name = msg<id(id, SEL)>::send(process_info, sel("processName"));
                id quit_title = msg<id(id, SEL, id)>::send(string("Quit "), sel("stringByAppendingString:"), process_name);
                id quit_item = msg<id(id, SEL, id, SEL, id)>::send(msg<id(Class, SEL)>::send(cls("NSMenuItem"), sel("alloc")), sel("initWithTitle:action:keyEquivalent:"), quit_title, sel("terminate:"), string("q"));
                msg<void(id, SEL, id)>::send(application_menu, sel("addItem:"), quit_item);
                msg<void(id, SEL, id)>::send(application_item, sel("setSubmenu:"), application_menu);
                msg<void(id, SEL)>::send(quit_item, sel("release"));
                msg<void(id, SEL)>::send(application_menu, sel("release"));
                msg<void(id, SEL)>::send(application_item, sel("release"));
                msg<void(id, SEL)>::send(menubar, sel("release"));

                msg<void(id, SEL)>::send(NSApp, sel("finishLaunching"));
                msg<void(id, SEL, BOOL)>::send(NSApp, sel("activateIgnoringOtherApps:"), objc_yes);
            }

            application_type(const application_type&) = delete;
            application_type(application_type&&) = delete;
            application_type& operator=(const application_type&) = delete;
            application_type& operator=(application_type&&) = delete;

        public:
            static Class window_delegate_class() {
                static Class delegate_class = nullptr;
                if (delegate_class) {
                    return delegate_class;
                }
                delegate_class = objc_allocateClassPair(cls("NSObject"), "GtlWindowDelegate", 0);
                class_addMethod(
                    delegate_class,
                    sel("windowWillClose:"),
                    imp(static_cast<void (*)(id, SEL, id)>([](id, SEL, id notification) {
                        if (cocoa_driver* platform = application_type::find(notification)) {
                            platform->closed = true;
                        }
                    })),
                    "v@:@"
                );
                class_addMethod(
                    delegate_class,
                    sel("windowDidResize:"),
                    imp(static_cast<void (*)(id, SEL, id)>([](id, SEL, id notification) {
                        if (cocoa_driver* platform = application_type::find(notification)) {
                            if (platform->opengl_context) {
                                msg<void(id, SEL)>::send(platform->opengl_context, sel("update"));
                            }
                            platform->push_configuration();
                        }
                    })),
                    "v@:@"
                );
                class_addMethod(
                    delegate_class,
                    sel("windowDidMove:"),
                    imp(static_cast<void (*)(id, SEL, id)>([](id, SEL, id notification) {
                        if (cocoa_driver* platform = application_type::find(notification)) {
                            platform->push_configuration();
                        }
                    })),
                    "v@:@"
                );
                class_addMethod(
                    delegate_class,
                    sel("windowDidBecomeKey:"),
                    imp(static_cast<void (*)(id, SEL, id)>([](id, SEL, id notification) {
                        if (cocoa_driver* platform = application_type::find(notification)) {
                            platform->push_focus(true);
                        }
                    })),
                    "v@:@"
                );
                class_addMethod(
                    delegate_class,
                    sel("windowDidResignKey:"),
                    imp(static_cast<void (*)(id, SEL, id)>([](id, SEL, id notification) {
                        if (cocoa_driver* platform = application_type::find(notification)) {
                            platform->push_focus(false);
                        }
                    })),
                    "v@:@"
                );
                objc_registerClassPair(delegate_class);
                return delegate_class;
            }

            static cocoa_driver* find(id notification) {
                id handle = msg<id(id, SEL)>::send(notification, sel("object"));
                std::unordered_map<id, cocoa_driver*>::iterator entry = application_type::registry.find(handle);
                return (entry != application_type::registry.end()) ? entry->second : nullptr;
            }

        public:
            static void pump() {
                if (application_type::frame_pool) {
                    msg<void(id, SEL)>::send(application_type::frame_pool, sel("drain"));
                }
                application_type::frame_pool = msg<id(id, SEL)>::send(msg<id(Class, SEL)>::send(cls("NSAutoreleasePool"), sel("alloc")), sel("init"));

                for (;;) {
                    id event = msg<id(id, SEL, NSUInteger, id, id, BOOL)>::send(NSApp, sel("nextEventMatchingMask:untilDate:inMode:dequeue:"), NSUIntegerMax, nullptr, NSDefaultRunLoopMode, objc_yes);
                    if (!event) {
                        break;
                    }
                    if (application_type::translate(event)) {
                        msg<void(id, SEL, id)>::send(NSApp, sel("sendEvent:"), event);
                    }
                }
                msg<void(id, SEL)>::send(NSApp, sel("updateWindows"));
            }

        private:
            static bool translate(id event) {
                const NSUInteger type = msg<NSUInteger(id, SEL)>::send(event, sel("type"));
                id handle = msg<id(id, SEL)>::send(event, sel("window"));
                std::unordered_map<id, cocoa_driver*>::iterator entry = application_type::registry.find(handle);
                cocoa_driver* platform = (entry != application_type::registry.end()) ? entry->second : nullptr;
                if (!platform) {
                    return true;
                }

                switch (type) {
                    case event_left_mouse_down:
                    case event_right_mouse_down:
                    case event_other_mouse_down:
                    case event_left_mouse_up:
                    case event_right_mouse_up:
                    case event_other_mouse_up: {
                        const bool press = (type == event_left_mouse_down) || (type == event_right_mouse_down) || (type == event_other_mouse_down);
                        event_type translated;
                        translated.input = press ? event_type::input_type::mouse_press : event_type::input_type::mouse_release;
                        if ((type == event_left_mouse_down) || (type == event_left_mouse_up)) {
                            translated.data.mouse = event_type::mouse_type::left;
                        }
                        else if ((type == event_right_mouse_down) || (type == event_right_mouse_up)) {
                            translated.data.mouse = event_type::mouse_type::right;
                        }
                        else {
                            const NSInteger button = msg<NSInteger(id, SEL)>::send(event, sel("buttonNumber"));
                            translated.data.mouse = (button == 2) ? event_type::mouse_type::middle : event_type::mouse_type::unknown;
                        }
                        platform->events.push_back(translated);
                        return true;
                    }

                    case event_mouse_moved:
                    case event_left_mouse_dragged:
                    case event_right_mouse_dragged:
                    case event_other_mouse_dragged: {
                        const CGPoint location = msg<CGPoint(id, SEL)>::send(event, sel("locationInWindow"));
                        const CGRect bounds = msg<CGRect(id, SEL)>::send(platform->content_view, sel("bounds"));
                        event_type translated;
                        translated.input = event_type::input_type::mouse_move;
                        translated.data.motion.x = static_cast<int>(location.x);
                        translated.data.motion.y = static_cast<int>(bounds.size.height - location.y);
                        platform->events.push_back(translated);
                        return true;
                    }

                    case event_scroll_wheel: {
                        const CGFloat delta = msg<CGFloat(id, SEL)>::send(event, sel("scrollingDeltaY"));
                        if (delta != 0) {
                            event_type translated;
                            translated.input = event_type::input_type::mouse_scroll;
                            translated.data.scroll = (delta > 0) ? 1 : -1;
                            platform->events.push_back(translated);
                        }
                        return true;
                    }

                    case event_key_down:
                    case event_key_up: {
                        const char32_t key = translate_key(event);
                        if (key != 0) {
                            event_type translated;
                            translated.input = (type == event_key_down) ? event_type::input_type::key_press : event_type::input_type::key_release;
                            translated.data.key = key;
                            platform->events.push_back(translated);
                        }
                        const NSUInteger flags = msg<NSUInteger(id, SEL)>::send(event, sel("modifierFlags"));
                        return (flags & modifier_command) != 0;
                    }

                    case event_flags_changed: {
                        const NSUInteger flags = msg<NSUInteger(id, SEL)>::send(event, sel("modifierFlags"));
                        const NSUInteger previous = platform->modifier_flags;
                        platform->modifier_flags = flags;
                        platform->push_modifier((previous & modifier_shift) != 0, (flags & modifier_shift) != 0, keysym_shift_left);
                        platform->push_modifier((previous & (modifier_control | modifier_command)) != 0, (flags & (modifier_control | modifier_command)) != 0, keysym_control_left);
                        platform->push_modifier((previous & modifier_option) != 0, (flags & modifier_option) != 0, keysym_alt_left);
                        platform->push_modifier((previous & modifier_caps_lock) != 0, (flags & modifier_caps_lock) != 0, keysym_caps_lock);
                        return true;
                    }

                    default:
                        return true;
                }
            }
        };

    public:
        void push_configuration() {
            const CGRect frame = msg<CGRect(id, SEL)>::send(this->window_handle, sel("frame"));
            const CGRect bounds = msg<CGRect(id, SEL)>::send(this->content_view, sel("bounds"));
            event_type translated;
            translated.input = event_type::input_type::configuration;
            translated.data.configuration.x = static_cast<int>(frame.origin.x);
            translated.data.configuration.y = static_cast<int>(screen_height() - (frame.origin.y + frame.size.height));
            translated.data.configuration.width = static_cast<int>(bounds.size.width);
            translated.data.configuration.height = static_cast<int>(bounds.size.height);
            this->events.push_back(translated);
        }

        void push_focus(bool gained) {
            event_type translated;
            translated.input = event_type::input_type::focus;
            translated.data.focus = gained ? event_type::focus_type::gained : event_type::focus_type::lost;
            this->events.push_back(translated);
        }

        void push_modifier(bool before, bool after, char32_t key) {
            if (before == after) {
                return;
            }
            event_type translated;
            translated.input = after ? event_type::input_type::key_press : event_type::input_type::key_release;
            translated.data.key = key;
            this->events.push_back(translated);
        }

    public:
        application_type application;
        id window_handle = nullptr;
        id content_view = nullptr;
        id delegate = nullptr;
        id opengl_context = nullptr;
        bool closed = false;
        bool hidden_cursor = false;
        bool grabbed_input = false;
        NSUInteger modifier_flags = 0;
        std::deque<event_type> events;
        std::vector<unsigned char> pixel_buffer;
        int icon_width = 0;
        int icon_height = 0;
        std::vector<unsigned int> icon_data;

        bool is_open() const override {
            return (this->window_handle != nullptr);
        }

        bool open(const position_type& position, const size_type& size, const std::string& title, const bool hardware_accelerated) override {
            if (this->is_open())
                return false;

            if (NSApp == nullptr) {
                return false;
            }

            const CGRect content_rect = { { static_cast<CGFloat>(position.x), 0 }, { static_cast<CGFloat>(size.width), static_cast<CGFloat>(size.height) } };
            this->window_handle = msg<id(id, SEL, CGRect, NSUInteger, NSUInteger, BOOL)>::send(
                msg<id(Class, SEL)>::send(cls("NSWindow"), sel("alloc")),
                sel("initWithContentRect:styleMask:backing:defer:"),
                content_rect,
                style_standard,
                backing_store_buffered,
                objc_no
            );
            if (!this->window_handle) {
                return false;
            }

            this->closed = false;
            this->modifier_flags = 0;
            this->events.clear();
            this->hidden_cursor = false;
            this->grabbed_input = false;

            msg<void(id, SEL, BOOL)>::send(this->window_handle, sel("setReleasedWhenClosed:"), objc_no);
            msg<void(id, SEL, BOOL)>::send(this->window_handle, sel("setAcceptsMouseMovedEvents:"), objc_yes);
            msg<void(id, SEL, CGPoint)>::send(this->window_handle, sel("setFrameTopLeftPoint:"), CGPoint{ static_cast<CGFloat>(position.x), screen_height() - static_cast<CGFloat>(position.y) });

            this->content_view = msg<id(id, SEL)>::send(this->window_handle, sel("contentView"));

            this->delegate = msg<id(id, SEL)>::send(msg<id(Class, SEL)>::send(cocoa_driver::application_type::window_delegate_class(), sel("alloc")), sel("init"));
            msg<void(id, SEL, id)>::send(this->window_handle, sel("setDelegate:"), this->delegate);

            cocoa_driver::application_type::registry[this->window_handle] = this;

            this->set_title(title);

            if (hardware_accelerated) {
                const uint32_t attributes[] = {
                    pixel_format_profile,
                    profile_legacy,
                    pixel_format_double_buffer,
                    pixel_format_colour_size,
                    24,
                    pixel_format_alpha_size,
                    8,
                    pixel_format_depth_size,
                    24,
                    pixel_format_stencil_size,
                    8,
                    pixel_format_accelerated,
                    0
                };

                id pixel_format = msg<id(id, SEL, const uint32_t*)>::send(msg<id(Class, SEL)>::send(cls("NSOpenGLPixelFormat"), sel("alloc")), sel("initWithAttributes:"), attributes);
                if (!pixel_format) {
                    this->close();
                    return false;
                }

                this->opengl_context = msg<id(id, SEL, id, id)>::send(msg<id(Class, SEL)>::send(cls("NSOpenGLContext"), sel("alloc")), sel("initWithFormat:shareContext:"), pixel_format, nullptr);
                msg<void(id, SEL)>::send(pixel_format, sel("release"));

                if (!this->opengl_context) {
                    this->close();
                    return false;
                }

                msg<void(id, SEL, id)>::send(this->opengl_context, sel("setView:"), this->content_view);
                msg<void(id, SEL)>::send(this->opengl_context, sel("makeCurrentContext"));
            }

            this->push_configuration();

            return true;
        }

        void close() override {
            if (!this->is_open())
                return;

            if (this->opengl_context) {
                msg<void(id, SEL)>::send(this->opengl_context, sel("clearDrawable"));
                msg<void(id, SEL)>::send(this->opengl_context, sel("release"));
                this->opengl_context = nullptr;
            }

            cocoa_driver::application_type::registry.erase(this->window_handle);

            msg<void(id, SEL, id)>::send(this->window_handle, sel("setDelegate:"), nullptr);
            if (this->delegate) {
                msg<void(id, SEL)>::send(this->delegate, sel("release"));
                this->delegate = nullptr;
            }

            if (!this->closed) {
                msg<void(id, SEL)>::send(this->window_handle, sel("close"));
            }
            msg<void(id, SEL)>::send(this->window_handle, sel("release"));

            this->window_handle = nullptr;
            this->content_view = nullptr;
            this->closed = false;
            this->events.clear();
            this->pixel_buffer.clear();
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

            const GLint swap_interval = enable ? 1 : 0;
            msg<void(id, SEL, const GLint*, NSInteger)>::send(this->opengl_context, sel("setValues:forParameter:"), &swap_interval, context_parameter_swap_interval);
            return true;
        }

        bool is_iconised() const override {
            if (!this->is_open())
                return false;
            return msg<BOOL(id, SEL)>::send(this->window_handle, sel("isMiniaturized")) != 0;
        }

        void set_iconised(bool iconised) override {
            if (!this->is_open())
                return;
            if (iconised) {
                msg<void(id, SEL, id)>::send(this->window_handle, sel("miniaturize:"), nullptr);
            }
            else {
                msg<void(id, SEL, id)>::send(this->window_handle, sel("deminiaturize:"), nullptr);
            }
        }

        bool is_visible() const override {
            if (!this->is_open())
                return false;
            return msg<BOOL(id, SEL)>::send(this->window_handle, sel("isVisible")) != 0;
        }

        void set_visible(bool visible) override {
            if (!this->is_open())
                return;
            if (visible) {
                msg<void(id, SEL, id)>::send(this->window_handle, sel("makeKeyAndOrderFront:"), nullptr);
                msg<void(id, SEL, BOOL)>::send(NSApp, sel("activateIgnoringOtherApps:"), objc_yes);
            }
            else {
                msg<void(id, SEL, id)>::send(this->window_handle, sel("orderOut:"), nullptr);
            }
        }

        bool is_maximised() const override {
            if (!this->is_open())
                return false;
            return msg<BOOL(id, SEL)>::send(this->window_handle, sel("isZoomed")) != 0;
        }

        void set_maximised(bool maximised) override {
            if (!this->is_open())
                return;
            if (this->is_maximised() != maximised) {
                msg<void(id, SEL, id)>::send(this->window_handle, sel("zoom:"), nullptr);
            }
        }

        bool is_borderless() const override {
            if (!this->is_open())
                return false;
            const NSUInteger style_mask = msg<NSUInteger(id, SEL)>::send(this->window_handle, sel("styleMask"));
            return (style_mask & style_titled) == 0;
        }

        void set_borderless(bool borderless) override {
            if (!this->is_open())
                return;
            NSUInteger style_mask = msg<NSUInteger(id, SEL)>::send(this->window_handle, sel("styleMask"));
            if (borderless) {
                style_mask &= ~style_standard;
            }
            else {
                style_mask |= style_standard;
            }
            msg<void(id, SEL, NSUInteger)>::send(this->window_handle, sel("setStyleMask:"), style_mask);
        }

        bool is_fullscreen() const override {
            if (!this->is_open())
                return false;
            const NSUInteger style_mask = msg<NSUInteger(id, SEL)>::send(this->window_handle, sel("styleMask"));
            return (style_mask & style_fullscreen) != 0;
        }

        void set_fullscreen(bool fullscreen) override {
            if (!this->is_open())
                return;
            if (this->is_fullscreen() != fullscreen) {
                msg<void(id, SEL, id)>::send(this->window_handle, sel("toggleFullScreen:"), nullptr);
            }
        }

        bool is_focused() const override {
            if (!this->is_open())
                return false;
            return msg<BOOL(id, SEL)>::send(this->window_handle, sel("isKeyWindow")) != 0;
        }

        void set_focused(bool focused) override {
            if (!this->is_open())
                return;
            if (focused) {
                msg<void(id, SEL, id)>::send(this->window_handle, sel("makeKeyAndOrderFront:"), nullptr);
                msg<void(id, SEL, BOOL)>::send(NSApp, sel("activateIgnoringOtherApps:"), objc_yes);
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
            if (this->hidden_cursor == cursor_hidden) {
                return;
            }
            this->hidden_cursor = cursor_hidden;
            msg<void(Class, SEL)>::send(cls("NSCursor"), sel(cursor_hidden ? "hide" : "unhide"));
        }

        bool is_grab_input() const override {
            if (!this->is_open())
                return false;
            return this->grabbed_input;
        }

        void set_grab_input(bool grab_input) override {
            if (!this->is_open())
                return;
            this->grabbed_input = grab_input;
            CGAssociateMouseAndMouseCursorPosition(static_cast<boolean_t>(!grab_input));
        }

        position_type get_position() const override {
            if (!this->is_open())
                return { 0, 0 };
            const CGRect frame = msg<CGRect(id, SEL)>::send(this->window_handle, sel("frame"));
            return { static_cast<int>(frame.origin.x), static_cast<int>(screen_height() - (frame.origin.y + frame.size.height)) };
        }

        void set_position(const position_type& position) override {
            if (!this->is_open())
                return;
            msg<void(id, SEL, CGPoint)>::send(this->window_handle, sel("setFrameTopLeftPoint:"), CGPoint{ static_cast<CGFloat>(position.x), screen_height() - static_cast<CGFloat>(position.y) });
        }

        size_type get_size() const override {
            if (!this->is_open())
                return { 0, 0 };
            const CGRect bounds = msg<CGRect(id, SEL)>::send(this->content_view, sel("bounds"));
            return { static_cast<int>(bounds.size.width), static_cast<int>(bounds.size.height) };
        }

        void set_size(const size_type& size) override {
            if (!this->is_open())
                return;
            msg<void(id, SEL, CGSize)>::send(this->window_handle, sel("setContentSize:"), CGSize{ static_cast<CGFloat>(size.width), static_cast<CGFloat>(size.height) });
        }

        std::string get_title() const override {
            if (!this->is_open())
                return {};
            id title = msg<id(id, SEL)>::send(this->window_handle, sel("title"));
            const char* utf8 = title ? msg<const char*(id, SEL)>::send(title, sel("UTF8String")) : nullptr;
            return utf8 ? std::string(utf8) : std::string();
        }

        void set_title(const std::string& new_title) override {
            if (!this->is_open())
                return;
            id title = string(new_title.c_str());
            if (title) {
                msg<void(id, SEL, id)>::send(this->window_handle, sel("setTitle:"), title);
            }
        }

        void get_icon(int& width, int& height, std::vector<unsigned int>& data) const override {
            width = this->icon_width;
            height = this->icon_height;
            data = this->icon_data;
        }

        void set_icon(int width, int height, const std::vector<unsigned int>& data) override {
            if ((width <= 0) || (height <= 0)) {
                return;
            }
            if (static_cast<size_t>(width) * static_cast<size_t>(height) != data.size()) {
                return;
            }

            this->icon_width = width;
            this->icon_height = height;
            this->icon_data = data;

            std::vector<unsigned char> rgba(data.size() * 4);
            for (size_t index = 0; index < data.size(); ++index) {
                const unsigned int pixel = data[index];
                rgba[index * 4 + 0] = static_cast<unsigned char>((pixel >> 16) & 0xFF);
                rgba[index * 4 + 1] = static_cast<unsigned char>((pixel >> 8) & 0xFF);
                rgba[index * 4 + 2] = static_cast<unsigned char>((pixel >> 0) & 0xFF);
                rgba[index * 4 + 3] = static_cast<unsigned char>((pixel >> 24) & 0xFF);
            }

            CGColorSpaceRef colour_space = CGColorSpaceCreateDeviceRGB();
            CGDataProviderRef provider = CGDataProviderCreateWithData(nullptr, rgba.data(), rgba.size(), nullptr);
            CGImageRef image = CGImageCreate(
                static_cast<size_t>(width),
                static_cast<size_t>(height),
                8,
                32,
                static_cast<size_t>(width) * 4,
                colour_space,
                static_cast<CGBitmapInfo>(kCGImageAlphaLast),
                provider,
                nullptr,
                false,
                kCGRenderingIntentDefault
            );
            CGDataProviderRelease(provider);
            CGColorSpaceRelease(colour_space);
            if (!image) {
                return;
            }

            id icon = msg<id(id, SEL, CGImageRef, CGSize)>::send(msg<id(Class, SEL)>::send(cls("NSImage"), sel("alloc")), sel("initWithCGImage:size:"), image, CGSize{ static_cast<CGFloat>(width), static_cast<CGFloat>(height) });
            CGImageRelease(image);
            if (!icon) {
                return;
            }
            msg<void(id, SEL, id)>::send(NSApp, sel("setApplicationIconImage:"), icon);
            msg<void(id, SEL)>::send(icon, sel("release"));
        }

        bool process(event_type& event) override {
            if (!this->is_open())
                return false;

            if (this->closed) {
                this->close();
                return false;
            }

            if (this->events.empty()) {
                cocoa_driver::application_type::pump();
            }

            if (this->closed) {
                this->close();
                return false;
            }

            if (this->events.empty()) {
                return false;
            }

            event = this->events.front();
            this->events.pop_front();
            return true;
        }

        void prepare() override {
            if (!this->is_open())
                return;
            if (this->is_hardware_accelerated()) {
                msg<void(id, SEL)>::send(this->opengl_context, sel("makeCurrentContext"));
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

            std::vector<unsigned char>& buffer = this->pixel_buffer;
            buffer.assign(static_cast<size_t>(size.width) * static_cast<size_t>(size.height) * 4, 0);

            const int draw_width = (width < size.width) ? width : size.width;
            const int draw_height = (height < size.height) ? height : size.height;
            const int data_index_r = 0;
            const int data_index_g = (step == 1) ? 0 : 1;
            const int data_index_b = (step == 1) ? 0 : 2;

            for (int y = 0; y < draw_height; ++y) {
                const unsigned char* __restrict data_line = data + y * stride;
                unsigned char* __restrict buffer_line = buffer.data() + static_cast<size_t>(y) * static_cast<size_t>(size.width) * 4;
                for (int x = 0; x < draw_width; ++x, data_line += step, buffer_line += 4) {
                    buffer_line[0] = data_line[data_index_r];
                    buffer_line[1] = data_line[data_index_g];
                    buffer_line[2] = data_line[data_index_b];
                    buffer_line[3] = 255;
                }
            }

            CGImageRef image = create_image(buffer.data(), size.width, size.height);
            if (!image) {
                return;
            }

            msg<void(id, SEL, BOOL)>::send(this->content_view, sel("setWantsLayer:"), objc_yes);
            id layer = msg<id(id, SEL)>::send(this->content_view, sel("layer"));
            if (layer) {
                msg<void(id, SEL, id)>::send(layer, sel("setContents:"), reinterpret_cast<id>(image));
            }
            CGImageRelease(image);
        }

        void present() override {
            if (!this->is_open())
                return;
            if (this->is_hardware_accelerated()) {
                msg<void(id, SEL)>::send(this->opengl_context, sel("flushBuffer"));
            }
        }

        bool read(unsigned char* __restrict const rgb, const int width, const int height, const int step, const int stride) override {
            if (!this->is_open())
                return false;
            if (this->is_hardware_accelerated())
                return false;

            const size_type size = this->get_size();
            const std::vector<unsigned char>& buffer = this->pixel_buffer;
            if (buffer.size() != static_cast<size_t>(size.width) * static_cast<size_t>(size.height) * 4) {
                return false;
            }

            const int read_width = (width < size.width) ? width : size.width;
            const int read_height = (height < size.height) ? height : size.height;

            for (int y = 0; y < read_height; ++y) {
                unsigned char* __restrict rgb_line = rgb + y * stride;
                const unsigned char* __restrict buffer_line = buffer.data() + static_cast<size_t>(y) * static_cast<size_t>(size.width) * 4;
                for (int x = 0; x < read_width; ++x, rgb_line += step, buffer_line += 4) {
                    rgb_line[0] = buffer_line[0];
                    if (step > 1) {
                        rgb_line[1] = buffer_line[1];
                        rgb_line[2] = buffer_line[2];
                    }
                }
            }

            return true;
        }
    };
}

namespace gtl {
    window::driver_type* window::create_driver_macos_cocoa() {
        return new cocoa_driver();
    }
}

#undef GTL_WINDOW_ASSERT
#else
namespace gtl {
    window::driver_type* window::create_driver_macos_cocoa() {
        return nullptr;
    }
}
#endif
