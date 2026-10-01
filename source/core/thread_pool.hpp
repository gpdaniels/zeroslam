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
                    return (lhs->priority != rhs->priority) ? (lhs->priority < rhs->priority) : std::less<const queue*>()(lhs, rhs);
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

        // The workers run while the pool is referenced, releasing the last reference joins them.
        // On windows the pool destructor runs under the loader lock where it cannot join, so a dll releases every reference before it is unloaded.
        class reference final {
        public:
            reference();
            ~reference();
            reference(const reference&) = delete;
            reference(reference&&) = delete;
            reference& operator=(const reference&) = delete;
            reference& operator=(reference&&) = delete;
        };

    private:
        // A parallel for, the caller and the workers claim chunks from the shared counter until it passes the count.
        struct job final {
            void (*const invoke)(const void*, size_t, size_t);
            const void* const body;
            const size_t count;
            const size_t chunk;
            std::atomic<size_t> next;
            std::atomic<size_t> participants;
            bool listed;

            job(void (*const job_invoke)(const void*, size_t, size_t), const void* const job_body, const size_t job_count, const size_t job_chunk);
            job(const job&) = delete;
            job(job&&) = delete;
            job& operator=(const job&) = delete;
            job& operator=(job&&) = delete;

            void run();
        };

    private:
        const unsigned int requested_workers;
        std::mutex lifetime_mutex;
        size_t references;
        std::vector<std::thread> threads;
        std::atomic<size_t> workers;
        std::atomic<size_t> generation;
        bool running;
        size_t spinning;
        size_t sleeping;
        std::mutex queue_mutex;
        std::condition_variable queue_available;
        std::condition_variable job_finished;
        std::set<queue*, queue::comparison> queues;
        std::vector<job*> jobs;

    private:
        explicit thread_pool(unsigned int thread_count);
        ~thread_pool();
        thread_pool(const thread_pool&) = delete;
        thread_pool(thread_pool&&) = delete;
        thread_pool& operator=(const thread_pool&) = delete;
        thread_pool& operator=(thread_pool&&) = delete;

        void thread_loop();
        job* claimable_job();
        void execute(job& work);
        void drain(queue& target);
        void start();
        void join();
        void acquire();
        void release();

    public:
        static thread_pool& instance();

        size_t thread_count() const;

        template <typename function_type>
        void parallel_for(const size_t count, const size_t grain, const function_type& body) {
            if (count == 0) {
                return;
            }
            const size_t threads_running = this->thread_count();
            const size_t participants = threads_running + 1;
            const size_t chunk = (((count + participants - 1) / participants) < grain) ? grain : ((count + participants - 1) / participants);
            if ((threads_running == 0) || (chunk >= count)) {
                for (size_t index = 0; index < count; ++index) {
                    body(index);
                }
                return;
            }
            job work(
                [](const void* const context, const size_t begin, const size_t end) {
                    const function_type& target = *static_cast<const function_type*>(context);
                    for (size_t index = begin; index < end; ++index) {
                        target(index);
                    }
                },
                &body,
                count,
                chunk
            );
            this->execute(work);
        }
    };
}

#endif // ZEROSLAM_CORE_THREAD_POOL_HPP
