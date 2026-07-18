#include "diffusion.h"

#include <cstdio>

int main() {
    if (!diffusion_logical_kv_self_test() || !diffusion_logical_kv_self_test()) {
        std::fprintf(stderr, "LOGICAL_KV_STALE_SELF_TEST=FAIL\n");
        return 1;
    }

    std::printf("LOGICAL_KV_STALE_SELF_TEST=PASS\n");
    return 0;
}
