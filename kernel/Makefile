CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-parameter \
            -DSC_INCLUDE_DYNAMIC_PROCESSES \
            -MMD -MP
EXTRA_CXXFLAGS ?=
INCLUDES := -I/usr/include -I.. -I../src -I../config -I.
LDFLAGS  := -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu
LIBS     := -lsystemc -lpthread

BUILDDIR := build

# src/ modules shared by all kernel simulators
COMMON_HW_OBJS := $(BUILDDIR)/src/memory.o             \
                  $(BUILDDIR)/src/accelerator.o        \
                  $(BUILDDIR)/src/accelerator_pool.o   \
                  $(BUILDDIR)/src/interconnect.o

MATMUL_HW_OBJS := $(COMMON_HW_OBJS) $(BUILDDIR)/src/worker.o

TARGET_LN     := $(BUILDDIR)/layer_norm_sim
TARGET_DW     := $(BUILDDIR)/dw_conv2d_sim
TARGET_POOL   := $(BUILDDIR)/pooling_sim
TARGET_VEC    := $(BUILDDIR)/vec_ops_sim
TARGET_MATMUL := $(BUILDDIR)/matmul_sim

OBJS_LN := $(BUILDDIR)/layer_norm/layer_norm_top.o $(COMMON_HW_OBJS)
OBJS_DW := $(BUILDDIR)/dw_conv2d/dw_conv2d_top.o $(COMMON_HW_OBJS)
OBJS_POOL := $(BUILDDIR)/pooling/pooling_top.o $(COMMON_HW_OBJS)
OBJS_VEC := $(BUILDDIR)/vec_ops/vec_ops_top.o $(COMMON_HW_OBJS)
OBJS_MATMUL := $(BUILDDIR)/matmul/main.o \
               $(BUILDDIR)/matmul/matmul_top.o \
               $(BUILDDIR)/matmul/accum_coordinator.o \
               $(MATMUL_HW_OBJS)

DEPS := $(OBJS_LN:.o=.d) $(OBJS_DW:.o=.d) $(OBJS_POOL:.o=.d) \
        $(OBJS_VEC:.o=.d) $(OBJS_MATMUL:.o=.d)

.PHONY: all layer_norm dw_conv2d pooling vec_ops matmul clean

$(TARGET_LN): $(OBJS_LN) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(TARGET_DW): $(OBJS_DW) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(TARGET_POOL): $(OBJS_POOL) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(TARGET_VEC): $(OBJS_VEC) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(TARGET_MATMUL): $(OBJS_MATMUL) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

all: $(TARGET_LN) $(TARGET_DW) $(TARGET_POOL) $(TARGET_VEC) $(TARGET_MATMUL)

layer_norm: $(TARGET_LN)
dw_conv2d: $(TARGET_DW)
pooling: $(TARGET_POOL)
vec_ops: $(TARGET_VEC)
matmul: $(TARGET_MATMUL)

$(BUILDDIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/src/%.o: ../src/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $(EXTRA_CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR)

-include $(DEPS)
