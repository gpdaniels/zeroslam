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

#include "core/thread_pool.hpp"

#include "core/assert.hpp"

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace core {
    thread_pool::queue::~queue() {
        ASSERT(this->empty(), "Thread pool queue still contains pending tasks.");
        ASSERT(this->finished(), "Thread pool queue is still being processed.");
        std::lock_guard<std::mutex> lock(this->pool.queue_mutex);
        this->pool.queues.erase(this);
    }

    thread_pool::queue::queue(thread_pool& target_pool, int queue_priority)
        : pool(target_pool)
        , priority(queue_priority)
        , inserted(0)
        , completed(0)
        , tasks_mutex()
        , tasks() {
    }

    void thread_pool::queue::push(const std::function<void()>& task) {
        {
            std::lock_guard<std::mutex> lock(this->tasks_mutex);
            ++this->inserted;
            this->tasks.push(task);
        }
        {
            std::lock_guard<std::mutex> lock(this->pool.queue_mutex);
            this->pool.queues.emplace(this);
            this->pool.queue_available.notify_one();
        }
    }

    void thread_pool::queue::drain() {
        this->pool.drain(*this);
    }

    bool thread_pool::queue::empty() const {
        std::lock_guard<std::mutex> lock(this->tasks_mutex);
        return this->tasks.empty();
    }

    bool thread_pool::queue::finished() const {
        std::lock_guard<std::mutex> lock(this->tasks_mutex);
        return (this->inserted == this->completed);
    }

    thread_pool::thread_pool(unsigned int count)
        : running(true)
        , threads()
        , queue_mutex()
        , queue_available()
        , queues() {
        this->threads.reserve(count);
        for (unsigned int thread_index = 0; thread_index < count; ++thread_index) {
            this->threads.emplace_back(&thread_pool::thread_loop, this);
        }
    }

    thread_pool::~thread_pool() {
        this->join();
    }

    void thread_pool::thread_loop() {
        queue* current = nullptr;
        std::function<void()> task;
        for (;;) {
            {
                std::lock_guard<std::mutex> lock(this->queue_mutex);
                if (!this->queues.empty()) {
                    current = *this->queues.begin();
                    std::lock_guard<std::mutex> lock_tasks(current->tasks_mutex);
                    if (!current->tasks.empty()) {
                        task = static_cast<std::function<void()>&&>(current->tasks.front());
                        current->tasks.pop();
                    }
                    if (current->tasks.empty()) {
                        this->queues.erase(current);
                    }
                }
            }
            if (task) {
                task();
                ++current->completed;
            }
            else {
                std::unique_lock<std::mutex> lock(this->queue_mutex);
                this->queue_available.wait(lock, [this] {
                    return !this->running || !this->queues.empty();
                });
                if (!this->running && this->queues.empty()) {
                    break;
                }
            }
            task = nullptr;
        }
    }

    void thread_pool::drain(queue& target) {
        std::function<void()> task;
        for (;;) {
            {
                std::lock_guard<std::mutex> lock(this->queue_mutex);
                std::lock_guard<std::mutex> lock_tasks(target.tasks_mutex);
                if (!target.tasks.empty()) {
                    task = static_cast<std::function<void()>&&>(target.tasks.front());
                    target.tasks.pop();
                }
                if (target.tasks.empty()) {
                    this->queues.erase(&target);
                }
            }
            if (task) {
                task();
                ++target.completed;
                task = nullptr;
            }
            else {
                break;
            }
        }
        while (!target.finished()) {
            std::this_thread::yield();
        }
    }

    void thread_pool::join() {
        {
            std::lock_guard<std::mutex> lock(this->queue_mutex);
            this->running = false;
            this->queue_available.notify_all();
        }
        this->thread_loop();
        for (std::thread& thread : this->threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    thread_pool& thread_pool::instance() {
        static thread_pool singleton([]() -> unsigned int {
            const char* const requested = std::getenv("ZEROSLAM_THREADS");
            if (requested != nullptr) {
                const long value = std::strtol(requested, nullptr, 10);
                return (value > 1) ? static_cast<unsigned int>(value - 1) : 0u;
            }
            return (std::thread::hardware_concurrency() > 1u) ? (std::thread::hardware_concurrency() - 1u) : 0u;
        }());
        return singleton;
    }

    size_t thread_pool::thread_count() const {
        return this->threads.size();
    }
}
