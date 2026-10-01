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

#include <algorithm>
#include <chrono>
#include <cstdlib>

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace {
    // Waking a sleeping thread costs tens of microseconds, so the workers and a waiting caller spin for about as long before they sleep.
    constexpr std::chrono::microseconds spin_duration(50);
}

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
            this->pool.generation.fetch_add(1, std::memory_order_relaxed);
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

    thread_pool::reference::reference() {
        thread_pool::instance().acquire();
    }

    thread_pool::reference::~reference() {
        thread_pool::instance().release();
    }

    thread_pool::job::job(void (*const job_invoke)(const void*, size_t, size_t), const void* const job_body, const size_t job_count, const size_t job_chunk)
        : invoke(job_invoke)
        , body(job_body)
        , count(job_count)
        , chunk(job_chunk)
        , next(0)
        , participants(0)
        , listed(false) {
    }

    void thread_pool::job::run() {
        for (;;) {
            const size_t begin = this->next.fetch_add(this->chunk, std::memory_order_relaxed);
            if (begin >= this->count) {
                return;
            }
            const size_t end = ((this->count - begin) > this->chunk) ? (begin + this->chunk) : this->count;
            this->invoke(this->body, begin, end);
        }
    }

    thread_pool::thread_pool(unsigned int count)
        : requested_workers(count)
        , lifetime_mutex()
        , references(0)
        , threads()
        , workers(0)
        , generation(0)
        , running(false)
        , spinning(0)
        , sleeping(0)
        , queue_mutex()
        , queue_available()
        , job_finished()
        , queues()
        , jobs() {
        this->jobs.reserve(16);
        this->start();
    }

    thread_pool::~thread_pool() {
#if defined(_WIN32)
        // A dll runs its static destructors under the loader lock, which an exiting thread needs, so joining here would deadlock.
        // The workers are already terminated at process exit, and releasing the last reference joins them before an unload.
        for (std::thread& thread : this->threads) {
            if (thread.joinable()) {
                thread.detach();
            }
        }
#else
        this->join();
#endif
    }

    void thread_pool::thread_loop() {
        std::unique_lock<std::mutex> lock(this->queue_mutex);
        for (;;) {
            queue* const current = this->queues.empty() ? nullptr : *this->queues.begin();
            // A parallel for runs at the default priority, ahead of the queues of equal priority.
            job* const work = ((current == nullptr) || (current->priority >= 0)) ? this->claimable_job() : nullptr;
            if (work != nullptr) {
                work->participants.fetch_add(1, std::memory_order_relaxed);
                lock.unlock();
                work->run();
                lock.lock();
                // The caller may release the job as soon as it reads zero, so it is not touched after this.
                if (work->participants.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    this->job_finished.notify_all();
                }
                continue;
            }
            if (current != nullptr) {
                std::function<void()> task;
                {
                    std::lock_guard<std::mutex> lock_tasks(current->tasks_mutex);
                    if (!current->tasks.empty()) {
                        task = static_cast<std::function<void()>&&>(current->tasks.front());
                        current->tasks.pop();
                    }
                    if (current->tasks.empty()) {
                        this->queues.erase(current);
                    }
                }
                if (task) {
                    lock.unlock();
                    task();
                    ++current->completed;
                    task = nullptr;
                    lock.lock();
                }
                continue;
            }
            if (!this->running) {
                break;
            }
            // Parallel fors tend to follow each other closely, so look for work a little longer before sleeping.
            const size_t seen = this->generation.load(std::memory_order_relaxed);
            ++this->spinning;
            lock.unlock();
            const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + spin_duration;
            while ((this->generation.load(std::memory_order_relaxed) == seen) && (std::chrono::steady_clock::now() < deadline)) {
                std::this_thread::yield();
            }
            lock.lock();
            --this->spinning;
            if (this->generation.load(std::memory_order_relaxed) != seen) {
                continue;
            }
            ++this->sleeping;
            this->queue_available.wait(lock);
            --this->sleeping;
        }
    }

    thread_pool::job* thread_pool::claimable_job() {
        while (!this->jobs.empty()) {
            job* const work = this->jobs.front();
            if (work->next.load(std::memory_order_relaxed) < work->count) {
                return work;
            }
            work->listed = false;
            this->jobs.erase(this->jobs.begin());
        }
        return nullptr;
    }

    void thread_pool::execute(job& work) {
        size_t wake = 0;
        {
            std::lock_guard<std::mutex> lock(this->queue_mutex);
            this->jobs.push_back(&work);
            work.listed = true;
            this->generation.fetch_add(1, std::memory_order_relaxed);
            // The spinning workers find the job themselves, wake sleeping ones for the rest of the chunks.
            const size_t helpers = (work.count - 1) / work.chunk;
            const size_t missing = (helpers > this->spinning) ? (helpers - this->spinning) : 0;
            wake = (this->sleeping < missing) ? this->sleeping : missing;
        }
        for (size_t index = 0; index < wake; ++index) {
            this->queue_available.notify_one();
        }
        work.run();
        {
            std::lock_guard<std::mutex> lock(this->queue_mutex);
            if (work.listed) {
                work.listed = false;
                this->jobs.erase(std::find(this->jobs.begin(), this->jobs.end(), &work));
            }
        }
        // Only the chunks the workers already claimed are left, so wait a little before sleeping.
        const std::chrono::steady_clock::time_point deadline = std::chrono::steady_clock::now() + spin_duration;
        while ((work.participants.load(std::memory_order_acquire) != 0) && (std::chrono::steady_clock::now() < deadline)) {
            std::this_thread::yield();
        }
        if (work.participants.load(std::memory_order_acquire) != 0) {
            std::unique_lock<std::mutex> lock(this->queue_mutex);
            while (work.participants.load(std::memory_order_acquire) != 0) {
                this->job_finished.wait(lock);
            }
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

    void thread_pool::start() {
        {
            std::lock_guard<std::mutex> lock(this->queue_mutex);
            this->running = true;
        }
        this->threads.reserve(this->requested_workers);
        for (unsigned int thread_index = 0; thread_index < this->requested_workers; ++thread_index) {
            this->threads.emplace_back(&thread_pool::thread_loop, this);
        }
        this->workers = this->threads.size();
    }

    void thread_pool::join() {
        this->workers = 0;
        {
            std::lock_guard<std::mutex> lock(this->queue_mutex);
            this->running = false;
            this->generation.fetch_add(1, std::memory_order_relaxed);
            this->queue_available.notify_all();
        }
        this->thread_loop();
        for (std::thread& thread : this->threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        this->threads.clear();
    }

    void thread_pool::acquire() {
        std::lock_guard<std::mutex> lock(this->lifetime_mutex);
        ++this->references;
        if (this->threads.empty()) {
            this->start();
        }
    }

    void thread_pool::release() {
        std::lock_guard<std::mutex> lock(this->lifetime_mutex);
        ASSERT(this->references > 0, "Thread pool reference released without being acquired.");
        --this->references;
        if (this->references == 0) {
            this->join();
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
        return this->workers;
    }
}
