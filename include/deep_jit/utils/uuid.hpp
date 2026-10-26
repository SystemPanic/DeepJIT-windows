#pragma once

#include <chrono>
#include <cstdint>
#include <format>
#include <functional>
#include <random>
#include <string>
#include <thread>

#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

namespace deep_jit {

inline int get_process_id() {
#ifdef _WIN32
    return ::_getpid();
#else
    return static_cast<int>(::getpid());
#endif
}

// Process-unique id: pid + three random words. The generator is thread_local
// and seeded per thread so concurrent callers never collide.
inline std::string get_uuid() {
    thread_local std::mt19937 generator([]() {
        std::random_device rd;
        return static_cast<std::mt19937::result_type>(rd()) ^
            static_cast<std::mt19937::result_type>(std::chrono::steady_clock::now().time_since_epoch().count()) ^
            static_cast<std::mt19937::result_type>(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }());
    thread_local std::uniform_int_distribution<uint32_t> distribution;
    return std::format("{}-{:08x}-{:08x}-{:08x}", get_process_id(),
                       distribution(generator), distribution(generator), distribution(generator));
}

}  // namespace deep_jit
