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
#ifndef ZEROSLAM_CORE_THREAD_POOL_HPP
#define ZEROSLAM_CORE_THREAD_POOL_HPP

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <set>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    using size_t = decltype(sizeof(0));
}

namespace core {
    class thread_pool final {
    public:
        class queue final {
        private:
            friend class thread_pool;

        private:
            struct comparison final {
                bool operator()(const queue* lhs, const queue* rhs) const {
                    return (lhs->priority != rhs->priority) ? (lhs->priority < rhs->priority) : (lhs < rhs);
                }
            };

        private:
            thread_pool& pool;
            int priority;
            unsigned int inserted;
            std::atomic<unsigned int> completed;
            mutable std::mutex tasks_mutex;
            std::queue<std::function<void()>> tasks;

        public:
            ~queue();
            explicit queue(thread_pool& target_pool, int queue_priority = 0);
            queue(const queue&) = delete;
            queue(queue&&) = delete;
            queue& operator=(const queue&) = delete;
            queue& operator=(queue&&) = delete;

        public:
            void push(const std::function<void()>& task);
            void drain();
            bool empty() const;
            bool finished() const;
        };

    private:
        bool running;
        std::vector<std::thread> threads;
        std::mutex queue_mutex;
        std::condition_variable queue_available;
        std::set<queue*, queue::comparison> queues;

    private:
        explicit thread_pool(unsigned int thread_count);
        ~thread_pool();
        thread_pool(const thread_pool&) = delete;
        thread_pool(thread_pool&&) = delete;
        thread_pool& operator=(const thread_pool&) = delete;
        thread_pool& operator=(thread_pool&&) = delete;

        void thread_loop();
        void drain(queue& target);
        void join();

    public:
        static thread_pool& instance();

        size_t thread_count() const;

        template <typename function_type>
        void parallel_for(const size_t count, const size_t grain, const function_type& body) {
            if (count == 0) {
                return;
            }
            const size_t workers = this->thread_count() + 1;
            const size_t chunk = (((count + workers - 1) / workers) < grain) ? grain : ((count + workers - 1) / workers);
            if ((this->thread_count() == 0) || (chunk >= count)) {
                for (size_t index = 0; index < count; ++index) {
                    body(index);
                }
                return;
            }
            queue tasks(*this);
            for (size_t begin = 0; begin < count; begin += chunk) {
                const size_t end = ((begin + chunk) < count) ? (begin + chunk) : count;
                tasks.push([&body, begin, end]() {
                    for (size_t index = begin; index < end; ++index) {
                        body(index);
                    }
                });
            }
            tasks.drain();
        }
    };
}

#endif // ZEROSLAM_CORE_THREAD_POOL_HPP
