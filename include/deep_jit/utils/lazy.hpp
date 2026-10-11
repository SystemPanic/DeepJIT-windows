#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

#include <deep_jit/utils/exception.hpp>

#define DJ_DECLARE_STATIC_VAR_IN_CLASS(cls, name) decltype(cls::name) cls::name

namespace deep_jit {

template <typename T>
class LazyInit {
    std::shared_ptr<T> ptr;
    std::function<std::shared_ptr<T>()> factory;

public:
    explicit LazyInit(std::nullptr_t) {}

    explicit LazyInit(std::function<std::shared_ptr<T>()> factory)
        : factory(std::move(factory)) {}

    T* operator->() {
        DJ_HOST_ASSERT(factory != nullptr, "lazy object must be initialized before use");
        if (ptr == nullptr)
            ptr = factory();
        DJ_HOST_ASSERT(ptr != nullptr, "lazy factory must not return nullptr");
        return ptr.get();
    }

    std::shared_ptr<T> get() {
        (void)operator->();
        return ptr;
    }
};

// ---------------------------------------------------------------------------
// Portable dynamic-library helpers (dlopen/dlsym on POSIX, LoadLibrary/GetProcAddress on Windows)
// ---------------------------------------------------------------------------
namespace detail {

#if defined(_WIN32)

inline std::string dl_last_error() {
    const DWORD code = ::GetLastError();
    if (code == 0)
        return "unknown";
    LPSTR buffer = nullptr;
    const DWORD size = ::FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        reinterpret_cast<LPSTR>(&buffer), 0, nullptr);
    std::string message = (size != 0 && buffer != nullptr)
        ? std::string(buffer, size)
        : "error code " + std::to_string(code);
    if (buffer != nullptr)
        ::LocalFree(buffer);
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r'))
        message.pop_back();
    return message;
}

inline void* dl_open(const char* lib) {
    ::SetLastError(0);
    return reinterpret_cast<void*>(::LoadLibraryA(lib));
}

inline void* dl_sym(void* handle, const char* name) {
    ::SetLastError(0);
    return reinterpret_cast<void*>(::GetProcAddress(static_cast<HMODULE>(handle), name));
}

#else

inline std::string dl_last_error() {
    const char* error = ::dlerror();
    return error == nullptr ? "unknown" : error;
}

inline void* dl_open(const char* lib) {
    ::dlerror();
    return ::dlopen(lib, RTLD_LAZY | RTLD_LOCAL);
}

inline void* dl_sym(void* handle, const char* name) {
    ::dlerror();
    return ::dlsym(handle, name);
}

#endif

}  // namespace detail

}  // namespace deep_jit

#define DJ_DECL_LAZY_DL_HANDLE(handle_func_name, lib)                                                 \
    inline void* handle_func_name() {                                                                 \
        static void* handle = [] {                                                                    \
            void* value = ::deep_jit::detail::dl_open(lib);                                           \
            if (value == nullptr)                                                                     \
                DJ_PANIC("failed to load {}: {}", lib, ::deep_jit::detail::dl_last_error());          \
            return value;                                                                             \
        }();                                                                                          \
        return handle;                                                                                \
    }

#define DJ_DECL_LAZY_DL_FUNCTION(handle_func_name, name)                                              \
    template <typename... Args>                                                                       \
    static auto lazy_##name(Args&&... args) {                                                         \
        static const auto func = []() {                                                               \
            void* symbol = ::deep_jit::detail::dl_sym(handle_func_name(), #name);                     \
            if (symbol == nullptr)                                                                    \
                DJ_PANIC("failed to load {} from {}: {}", #name, #handle_func_name,                   \
                         ::deep_jit::detail::dl_last_error());                                        \
            return reinterpret_cast<decltype(&name)>(symbol);                                         \
        }();                                                                                          \
        return func(std::forward<Args>(args)...);                                                     \
    }