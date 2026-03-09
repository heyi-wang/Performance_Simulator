#include "top.h"
#include <iostream>

int sc_main(int /*argc*/, char * /*argv*/[])
{
    Top top("top");
    sc_start();

    std::cout << "\n=== Accelerator stats ===\n";
    std::cout << "mat_acc: reqs="             << top.mat_acc.req_count
              << " busy_cycles="              << top.mat_acc.busy_cycles
              << " queue_wait_cycles="        << top.mat_acc.queue_wait_cycles
              << "\n";

    std::cout << "vec_acc: reqs="             << top.vec_acc.req_count
              << " busy_cycles="              << top.vec_acc.busy_cycles
              << " queue_wait_cycles="        << top.vec_acc.queue_wait_cycles
              << "\n";

    std::cout << "memory : reqs="             << top.memory.reqs
              << " busy_cycles="              << top.memory.busy_cycles
              << " queue_wait_cycles="        << top.memory.qwait_cycles
              << "\n";

    return 0;
}
