#pragma once

#include "../config/hardware_config.h"

// Keep the NAFNet worker count local to the NAFNet simulator while
// reusing the shared tile geometry, bandwidth, and accelerator timing
// constants defined for the standalone kernel simulators.
static const int N_WORKERS = 4;
