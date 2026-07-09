#include "internal/failpoint.hpp"

#include <atomic>
#include <cstring>

namespace dfine::testing {

namespace {

struct Point {
    char name[56] = {};
    std::atomic<int> remaining{0};
};

Point g_points[4];
std::atomic<int> g_armed{0};  // number of live points; 0 short-circuits every check

}  // namespace

void arm_failpoint(const char* name, int nth) noexcept {
    if (!name || !*name) return;
    Point* slot = nullptr;
    for (auto& p : g_points) {
        if (std::strncmp(p.name, name, sizeof(p.name)) == 0) {
            slot = &p;
            break;
        }
        if (!slot && p.name[0] == '\0') slot = &p;
    }
    if (!slot) return;  // all slots busy; tests arm at most two points
    if (nth <= 0) {
        if (slot->name[0] != '\0') {
            slot->name[0] = '\0';
            g_armed.fetch_sub(1, std::memory_order_release);
        }
        return;
    }
    const bool fresh = slot->name[0] == '\0';
    std::strncpy(slot->name, name, sizeof(slot->name) - 1);
    slot->name[sizeof(slot->name) - 1] = '\0';
    slot->remaining.store(nth, std::memory_order_relaxed);
    if (fresh) g_armed.fetch_add(1, std::memory_order_release);
}

bool failpoint(const char* name) noexcept {
    if (g_armed.load(std::memory_order_acquire) == 0) return false;
    for (auto& p : g_points) {
        if (p.name[0] == '\0' || std::strncmp(p.name, name, sizeof(p.name)) != 0) continue;
        if (p.remaining.fetch_sub(1, std::memory_order_relaxed) == 1) {
            p.name[0] = '\0';
            g_armed.fetch_sub(1, std::memory_order_release);
            return true;
        }
        return false;
    }
    return false;
}

}  // namespace dfine::testing
