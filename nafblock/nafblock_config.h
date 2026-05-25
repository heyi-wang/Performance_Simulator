#pragma once

// Override at build time via -DNAFBLOCK_N_WORKERS=N (used by full_sweep.py).
#ifndef NAFBLOCK_N_WORKERS
#define NAFBLOCK_N_WORKERS 4
#endif

namespace nafblock_cfg
{
constexpr int N_WORKERS    = NAFBLOCK_N_WORKERS;
constexpr int DEFAULT_C    = 32;
constexpr int DEFAULT_H    = 64;
constexpr int DEFAULT_W    = 64;
}  // namespace nafblock_cfg
