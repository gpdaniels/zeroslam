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
#ifndef ZEROSLAM_TOOLS_GUI_ICON_HPP
#define ZEROSLAM_TOOLS_GUI_ICON_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace icon {

    constexpr int W = 64;
    constexpr int H = 64;

    constexpr float VIEW = 264.6f;
    constexpr float S = W / VIEW;
    constexpr float PI = 3.14159265358979323846f;

    struct Pt {
        float x, y;
    };

    struct Path {
        std::vector<std::vector<Pt>> contours;
    };

    static Pt P(float x, float y) {
        return {
            (x - 2196.0f) * S,
            (y - 441.0f) * S
        };
    }

    static void arc(Path& path, Pt a, float rx, float ry, bool largeArc, bool sweep, Pt b) {
        rx *= S;
        ry *= S;

        if (rx <= 0 || ry <= 0) {
            path.contours.back().push_back(b);
            return;
        }

        float dx = (a.x - b.x) * 0.5f;
        float dy = (a.y - b.y) * 0.5f;

        float rx2 = rx * rx;
        float ry2 = ry * ry;
        float dx2 = dx * dx;
        float dy2 = dy * dy;

        float lambda = dx2 / rx2 + dy2 / ry2;

        if (lambda > 1.0f) {
            float k = std::sqrt(lambda);
            rx *= k;
            ry *= k;
            rx2 = rx * rx;
            ry2 = ry * ry;
        }

        float n = rx2 * ry2 - rx2 * dy2 - ry2 * dx2;
        float d = rx2 * dy2 + ry2 * dx2;

        float k = d > 0
                      ? std::sqrt(std::max(0.0f, n / d))
                      : 0.0f;

        if (largeArc == sweep)
            k = -k;

        float cx = k * rx * dy / ry;
        float cy = -k * ry * dx / rx;

        cx += (a.x + b.x) * 0.5f;
        cy += (a.y + b.y) * 0.5f;

        auto angle = [](float ux, float uy, float vx, float vy) {
            float dot = ux * vx + uy * vy;
            float l = std::sqrt(
                (ux * ux + uy * uy) *
                (vx * vx + vy * vy)
            );

            if (l == 0)
                return 0.0f;

            float r = std::acos(
                std::clamp(dot / l, -1.0f, 1.0f)
            );

            if (ux * vy - uy * vx < 0)
                r = -r;

            return r;
        };

        float ux = (a.x - cx) / rx;
        float uy = (a.y - cy) / ry;
        float vx = (b.x - cx) / rx;
        float vy = (b.y - cy) / ry;

        float start = angle(1, 0, ux, uy);
        float delta = angle(ux, uy, vx, vy);

        if (!sweep && delta > 0)
            delta -= 2 * PI;
        if (sweep && delta < 0)
            delta += 2 * PI;

        int steps = std::max(
            1,
            int(std::ceil(std::fabs(delta) / (PI / 90.0f)))
        );

        for (int i = 1; i <= steps; ++i) {
            float t = float(i) / static_cast<float>(steps);
            float theta = start + delta * t;

            path.contours.back().push_back({ cx + rx * std::cos(theta), cy + ry * std::sin(theta) });
        }
    }

    static bool inside(float x, float y, const Path& path) {
        bool result = false;

        for (const auto& c : path.contours) {
            if (c.size() < 2)
                continue;

            for (size_t i = 0, j = c.size() - 1;
                 i < c.size();
                 j = i++) {
                const Pt& a = c[i];
                const Pt& b = c[j];

                if (((a.y > y) != (b.y > y)) &&
                    x < (b.x - a.x) *
                                (y - a.y) /
                                (b.y - a.y) +
                            a.x) {
                    result = !result;
                }
            }
        }

        return result;
    }

    static float segDist(float x, float y, Pt a, Pt b) {
        float dx = b.x - a.x;
        float dy = b.y - a.y;
        float d2 = dx * dx + dy * dy;

        if (d2 == 0)
            return std::hypot(x - a.x, y - a.y);

        float t = ((x - a.x) * dx +
                   (y - a.y) * dy) /
                  d2;

        t = std::clamp(t, 0.0f, 1.0f);

        float qx = a.x + dx * t;
        float qy = a.y + dy * t;

        return std::hypot(x - qx, y - qy);
    }

    static bool stroke(float x, float y, const Path& path, float width) {
        float r = width * 0.5f;

        for (const auto& c : path.contours) {
            for (size_t i = 0; i + 1 < c.size(); ++i) {
                if (segDist(x, y, c[i], c[i + 1]) <= r)
                    return true;
            }

            for (const Pt& p : c) {
                if (std::hypot(x - p.x, y - p.y) <= r)
                    return true;
            }
        }

        return false;
    }

    static void draw(
        std::vector<unsigned int>& image,
        const Path& path,
        unsigned int fill,
        unsigned int strokeColor,
        float strokeWidth
    ) {
        float minX = float(W);
        float minY = float(H);
        float maxX = 0;
        float maxY = 0;

        for (const auto& c : path.contours) {
            for (const Pt& p : c) {
                minX = std::min(minX, p.x);
                minY = std::min(minY, p.y);
                maxX = std::max(maxX, p.x);
                maxY = std::max(maxY, p.y);
            }
        }

        int x0 = std::max(
            0,
            int(std::floor(minX - strokeWidth))
        );
        int x1 = std::min(
            W - 1,
            int(std::ceil(maxX + strokeWidth))
        );
        int y0 = std::max(
            0,
            int(std::floor(minY - strokeWidth))
        );
        int y1 = std::min(
            H - 1,
            int(std::ceil(maxY + strokeWidth))
        );

        for (int y = y0; y <= y1; ++y) {
            for (int x = x0; x <= x1; ++x) {
                float px = static_cast<float>(x) + 0.5f;
                float py = static_cast<float>(y) + 0.5f;

                unsigned int& dst = image[static_cast<size_t>(y * W + x)];

                if (inside(px, py, path))
                    dst = fill;

                if (strokeWidth > 0 &&
                    stroke(px, py, path, strokeWidth)) {
                    dst = strokeColor;
                }
            }
        }
    }

    static Path outer1() {
        Path p;
        p.contours.push_back({});

        auto& c = p.contours.back();

        Pt cur = P(2449, 514.4f);
        c.push_back(cur);

        Pt next = P(
            2449 - 8.714f,
            514.4f + 68.68f
        );

        arc(p, cur, 23.81f, 50.28f, false, true, next);
        cur = next;

        next = P(
            cur.x / S + 2196 - 32.52f,
            cur.y / S + 441 + (-18.4f)
        );

        arc(p, cur, 23.81f, 50.28f, false, true, next);
        cur = next;

        next = P(
            cur.x / S + 2196 + 8.714f,
            cur.y / S + 441 - 68.68f
        );

        arc(p, cur, 23.81f, 50.28f, false, true, next);
        cur = next;

        next = P(2449, 514.4f);

        arc(p, cur, 23.81f, 50.28f, false, true, next);

        c.push_back(c.front());
        return p;
    }

    static Path outer2() {
        Path p;
        p.contours.push_back({});

        auto& c = p.contours.back();

        Pt cur = P(2249, 581.9f);
        c.push_back(cur);

        Pt next = P(
            2249 - 8.714f,
            581.9f + 68.68f
        );

        arc(p, cur, 23.81f, 50.28f, false, true, next);
        cur = next;

        next = P(
            cur.x / S + 2196 - 32.52f,
            cur.y / S + 441 - 18.4f
        );

        arc(p, cur, 23.81f, 50.28f, false, true, next);
        cur = next;

        next = P(
            cur.x / S + 2196 + 8.714f,
            cur.y / S + 441 - 68.68f
        );

        arc(p, cur, 23.81f, 50.28f, false, true, next);
        cur = next;

        next = P(2249, 581.9f);

        arc(p, cur, 23.81f, 50.28f, false, true, next);

        c.push_back(c.front());
        return p;
    }

    static Path mainShape() {
        Path p;

        p.contours.push_back({});
        auto& o = p.contours.back();

        Pt cur = P(2329, 489.2f);
        o.push_back(cur);

        cur.y += 0.001f * S;
        o.push_back(cur);

        Pt next = P(2278.73f, 539.47f);

        arc(p, cur, 50.27f, 50.27f, false, false, next);
        cur = next;

        next = P(2290.13f, 571.34f);

        arc(p, cur, 50.27f, 50.27f, false, false, next);
        cur = next;

        next = P(2228.45f, 606.95f);
        o.push_back(next);
        cur = next;

        next = P(2303.85f, 650.48f);
        o.push_back(next);
        cur = next;

        next = P(2303.86f, 650.484f);
        o.push_back(next);
        cur = next;

        cur.y -= 0.00051f * S;
        o.push_back(cur);

        next = P(2329.0f, 657.22f);

        arc(p, cur, 50.27f, 50.27f, false, false, next);
        cur = next;

        cur.y -= 0.001f * S;
        o.push_back(cur);

        next = P(2379.27f, 606.949f);

        arc(p, cur, 50.27f, 50.27f, false, false, next);
        cur = next;

        next = P(2367.87f, 575.079f);

        arc(p, cur, 50.27f, 50.27f, false, false, next);
        cur = next;

        next = P(2429.55f, 539.469f);
        o.push_back(next);
        cur = next;

        next = P(2354.15f, 495.939f);
        o.push_back(next);
        cur = next;

        next = P(2354.14f, 495.935f);
        o.push_back(next);
        cur = next;

        cur.y += 0.00052f * S;
        o.push_back(cur);

        next = P(2329.0f, 489.2f);

        arc(p, cur, 50.27f, 50.27f, false, false, next);

        o.push_back(o.front());

        p.contours.push_back({});
        auto& a = p.contours.back();

        cur = P(2329, 522.27f);
        a.push_back(cur);

        cur.y += 0.00052f * S;
        a.push_back(cur);

        Pt q = P(2337.604f, 524.575f);

        arc(p, cur, 17.21f, 17.21f, false, true, q);
        cur = q;

        cur.x += 0.0005f * S;
        a.push_back(cur);

        cur.y += 0.002f * S;
        a.push_back(cur);

        cur = P(2363.415f, 541.5f);
        a.push_back(cur);

        cur = P(2337.635f, 556.39f);
        a.push_back(cur);

        cur.x -= 0.03f * S;
        a.push_back(cur);

        cur.y -= 0.003f * S;
        a.push_back(cur);

        q = P(2329.031f, 556.688f);

        arc(p, cur, 17.21f, 17.21f, false, true, q);
        cur = q;

        q = P(2311.79f, 539.48f);

        arc(p, cur, 17.21f, 17.21f, false, true, q);

        a.push_back(a.front());

        p.contours.push_back({});
        auto& b = p.contours.back();

        cur = P(2329, 589.75f);
        b.push_back(cur);

        q = P(2346.21f, 606.96f);

        arc(p, cur, 17.21f, 17.21f, false, true, q);
        cur = q;

        q = P(2329, 624.17f);

        arc(p, cur, 17.21f, 17.21f, false, true, q);
        cur = q;

        cur.y -= 0.00052f * S;
        b.push_back(cur);

        cur.x -= 8.604f * S / 17.21f;

        b.clear();

        cur = P(2329, 589.75f);
        b.push_back(cur);

        q = P(2346.21f, 606.96f);
        arc(p, cur, 17.21f, 17.21f, false, true, q);

        q = P(2329, 624.17f);
        arc(p, b.back(), 17.21f, 17.21f, false, true, q);

        q = P(2329, 624.16948f);
        b.push_back(q);

        q = P(2320.396f, 621.864f);
        arc(p, b.back(), 17.21f, 17.21f, false, true, q);

        q = P(2294.585f, 606.95f);
        b.push_back(q);

        q = P(2320.365f, 592.06f);
        b.push_back(q);

        q = P(2320.395f, 592.0424f);
        b.push_back(q);

        q = P(2320.395f, 592.0454f);
        b.push_back(q);

        q = P(2329, 589.75f);
        arc(p, b.back(), 17.21f, 17.21f, false, true, q);

        b.push_back(b.front());

        return p;
    }

    inline std::vector<unsigned int> make() {
        std::vector<unsigned int> image(
            W * H,
            0x00000000u
        );

        constexpr unsigned int BLACK = 0xff000000u;
        constexpr unsigned int WHITE = 0xffffffffu;

        const float strokeWidth = 5.292f * S;

        draw(image, outer1(), BLACK, BLACK, strokeWidth);
        draw(image, outer2(), BLACK, BLACK, strokeWidth);

        draw(image, mainShape(), WHITE, BLACK, strokeWidth);

        return image;
    }
}

#endif // ZEROSLAM_TOOLS_GUI_ICON_HPP
