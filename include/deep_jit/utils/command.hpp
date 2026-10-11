#pragma once

#include <array>
#include <cstdio>
#include <memory>
#include <string>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#include <deep_jit/utils/exception.hpp>

namespace deep_jit {

namespace detail {

inline FILE* open_process(const char* command, const char* mode) {
#ifdef _WIN32
    return _popen(command, mode);
#else
    return popen(command, mode);
#endif
}

inline int close_process(FILE* file) {
#ifdef _WIN32
    return _pclose(file);
#else
    return pclose(file);
#endif
}

inline int to_exit_code(const int status) {
#ifdef _WIN32
    // NOTES: on Windows, _pclose returns the child's exit code directly (or -1 on error)
    return status;
#else
    // NOTES: if the child was killed by a signal (e.g., SIGINT from Ctrl+C),
    // WEXITSTATUS would incorrectly return 0. Treat signal death as failure.
    return WIFEXITED(status) ? WEXITSTATUS(status) : 128 + WTERMSIG(status);
#endif
}

}  // namespace detail

inline std::string call_external_command(std::string command, const bool print_command = false) {
    DJ_HOST_ASSERT(!command.empty(), "command must not be empty");
    if (print_command)
        std::printf("Running command: %s\n", command.c_str());

    command += " 2>&1";
    const auto deleter = [](FILE* file) {
        if (file != nullptr)
            detail::close_process(file);
    };
    std::unique_ptr<FILE, decltype(deleter)> pipe(detail::open_process(command.c_str(), "r"), deleter);
    DJ_HOST_ASSERT(pipe != nullptr, "failed to run command: {}", command);

    std::array<char, 512> buffer;
    std::string output;
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr)
        output += buffer.data();

    const auto status = detail::close_process(pipe.release());
    const auto exit_code = detail::to_exit_code(status);
    DJ_HOST_ASSERT(exit_code == 0, "command failed with exit code {}:\n{}\n{}", exit_code, command, output);
    return output;
}

}  // namespace deep_jit