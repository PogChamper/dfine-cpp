#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

inline bool starts_with(std::string_view s, std::string_view p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

// Accepts both "--flag=value" and "--flag value".
inline const char* next_value(int argc, char** argv, int& i, std::string_view flag) {
    std::string_view a = argv[i];
    if (a.size() > flag.size() + 1 && a[flag.size()] == '=') {
        return argv[i] + flag.size() + 1;
    }
    if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + std::string(flag));
    }
    return argv[++i];
}

inline int parse_int(const char* s, std::string_view flag) {
    try { return std::stoi(s); }
    catch (...) { throw std::runtime_error("invalid integer for " + std::string(flag)); }
}

inline float parse_float(const char* s, std::string_view flag) {
    try { return std::stof(s); }
    catch (...) { throw std::runtime_error("invalid float for " + std::string(flag)); }
}
