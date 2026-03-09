#pragma once

#include "memory.h"
#include "interconnect.h"
#include "accelerator.h"
#include "worker.h"
#include "config.h"
#include <vector>

// ============================================================
// Top — instantiates and connects all sub-modules
// ============================================================
struct Top : sc_module
{
    AcceleratorTLM mat_acc;
    AcceleratorTLM vec_acc;
    Interconnect   noc;
    Memory         memory;

    std::vector<Worker *> workers;

    SC_CTOR(Top);

    ~Top() override;
};
