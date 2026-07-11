#pragma once

// Test-only fault injection for error paths that are impractical to trigger
// for real (a cudaMalloc failure at an exact moment, a setTensorAddress
// rejection). A disarmed process pays one atomic load per check.
//
// Tests link libdfine and arm points directly:
//   dfine::testing::arm_failpoint("trt_session.dev_alloc", 1);  // 1st hit fires
// A fired point disarms itself. Arming is not synchronized against concurrent
// hits — tests are single-threaded by design.

namespace dfine::testing {

// Arm `name` to fire on its nth upcoming hit (1-based). nth <= 0 disarms the
// point. Up to 4 points can be armed at once (re-arming a name resets it).
void arm_failpoint(const char* name, int nth) noexcept;

// True when an armed point's hit count is reached; fires once, then disarms.
bool failpoint(const char* name) noexcept;

}  // namespace dfine::testing
