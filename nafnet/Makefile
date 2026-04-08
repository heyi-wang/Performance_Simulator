CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-parameter \
            -DSC_INCLUDE_DYNAMIC_PROCESSES \
            -MMD -MP
KERNEL_CPPFLAGS := -DKERNEL_STANDALONE_MAIN=0
INCLUDES := -I/usr/include -I../src -I../config -I../kernel -I../kernel/matmul \
            -I../kernel/dw_conv2d -I../kernel/layer_norm -I../kernel/pooling \
            -I../kernel/vec_ops -I.
LDFLAGS  := -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu
LIBS     := -lsystemc -lpthread

TARGET   := build/nafnet_perf_sim
TEST_TARGET := build/nafnet_kernel_bridge_test
BUILDDIR := build

OBJS := $(BUILDDIR)/memory.o               \
        $(BUILDDIR)/accelerator.o          \
        $(BUILDDIR)/accelerator_pool.o     \
        $(BUILDDIR)/interconnect.o         \
        $(BUILDDIR)/worker.o               \
        $(BUILDDIR)/matmul_top.o           \
        $(BUILDDIR)/accum_coordinator.o    \
        $(BUILDDIR)/dw_conv2d_top.o        \
        $(BUILDDIR)/layer_norm_top.o       \
        $(BUILDDIR)/pooling_top.o          \
        $(BUILDDIR)/vec_ops_top.o          \
        $(BUILDDIR)/nafnet_sim.o

.PHONY: all run test clean

all: $(TARGET) $(TEST_TARGET)

$(TARGET): $(OBJS) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(TEST_TARGET): $(BUILDDIR)/nafnet_kernel_bridge_test.o | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(BUILDDIR)/memory.o: ../src/memory.cpp ../src/memory.h ../src/common.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/accelerator.o: ../src/accelerator.cpp ../src/accelerator.h ../src/interconnect.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/accelerator_pool.o: ../src/accelerator_pool.cpp ../src/accelerator_pool.h ../src/accelerator.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/interconnect.o: ../src/interconnect.cpp ../src/interconnect.h ../src/extensions.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/worker.o: ../src/worker.cpp ../src/worker.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/matmul_top.o: ../kernel/matmul/matmul_top.cpp ../kernel/matmul/matmul_top.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(KERNEL_CPPFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/accum_coordinator.o: ../kernel/matmul/accum_coordinator.cpp ../kernel/matmul/accum_coordinator.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/dw_conv2d_top.o: ../kernel/dw_conv2d/dw_conv2d_top.cpp ../kernel/dw_conv2d/dw_conv2d_top.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(KERNEL_CPPFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/layer_norm_top.o: ../kernel/layer_norm/layer_norm_top.cpp ../kernel/layer_norm/layer_norm_top.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(KERNEL_CPPFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/pooling_top.o: ../kernel/pooling/pooling_top.cpp ../kernel/pooling/pooling_top.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(KERNEL_CPPFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/vec_ops_top.o: ../kernel/vec_ops/vec_ops_top.cpp ../kernel/vec_ops/vec_ops_top.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(KERNEL_CPPFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/nafnet_sim.o: nafnet_sim.cpp nafnet_layers.h nafnet_hw_config.h nafnet_kernel_bridge.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/nafnet_kernel_bridge_test.o: nafnet_kernel_bridge_test.cpp nafnet_kernel_bridge.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

run: $(TARGET)
	./$(TARGET)

test: $(TARGET) $(TEST_TARGET)
	./$(TEST_TARGET)

clean:
	rm -rf $(BUILDDIR)

# Auto-generated header dependency rules
-include $(OBJS:.o=.d)
