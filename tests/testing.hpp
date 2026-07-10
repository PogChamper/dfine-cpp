#pragma once

// Tiny assertion harness for the dfine test binaries — no framework dependency.
// DFINE_CHECK records a failure and continues; DFINE_EXPECT_THROW asserts a
// statement throws a std::exception whose what() contains a substring. A test
// binary returns finish() from main: 0 = all checks passed, 1 = failures.
// Exit code 77 marks a skipped test (CTest SKIP_RETURN_CODE), used when a
// prerequisite — a GPU or a DFINE_TEST_ENGINE path — is missing.

#include <cstdio>
#include <exception>
#include <functional>
#include <string>

namespace dfine::testing {

constexpr int kSkipExitCode = 77;

inline int g_failures = 0;

inline void check(bool ok, const char* expr, const char* file, int line) {
    if (!ok) {
        ++g_failures;
        std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
    }
}

inline void expect_throw(const std::function<void()>& fn, const std::string& needle,
                         const char* stmt, const char* file, int line) {
    try {
        fn();
    } catch (const std::exception& e) {
        if (needle.empty() || std::string(e.what()).find(needle) != std::string::npos) return;
        ++g_failures;
        std::fprintf(stderr, "FAIL %s:%d: %s threw \"%s\" (expected substring \"%s\")\n", file,
                     line, stmt, e.what(), needle.c_str());
        return;
    }
    ++g_failures;
    std::fprintf(stderr, "FAIL %s:%d: %s did not throw\n", file, line, stmt);
}

inline int finish(const char* name) {
    if (g_failures) {
        std::fprintf(stderr, "%s: %d check(s) FAILED\n", name, g_failures);
        return 1;
    }
    std::printf("%s: OK\n", name);
    return 0;
}

}  // namespace dfine::testing

#define DFINE_CHECK(cond) \
    ::dfine::testing::check(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
#define DFINE_EXPECT_THROW(stmt, needle) \
    ::dfine::testing::expect_throw([&] { stmt; }, (needle), #stmt, __FILE__, __LINE__)
