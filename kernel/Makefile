CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-parameter \
            -DSC_INCLUDE_DYNAMIC_PROCESSES \
            -MMD -MP
INCLUDES := -I/usr/include -I.. -I../src -I.
LDFLAGS  := -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu
LIBS     := -lsystemc -lpthread

BUILDDIR := build

# src/ modules shared by all kernel simulators
SRC_OBJS := $(BUILDDIR)/src_memory.o            \
            $(BUILDDIR)/src_accelerator.o        \
            $(BUILDDIR)/src_accelerator_pool.o   \
            $(BUILDDIR)/src_interconnect.o

# ── Targets ────────────────────────────────────────────────
TARGET_LN   := $(BUILDDIR)/layer_norm_sim
TARGET_DW   := $(BUILDDIR)/dw_conv2d_sim
TARGET_POOL := $(BUILDDIR)/pooling_sim
TARGET_VEC  := $(BUILDDIR)/vec_ops_sim

OBJS_LN   := $(BUILDDIR)/layer_norm_sim.o $(SRC_OBJS)
OBJS_DW   := $(BUILDDIR)/dw_conv2d_sim.o  $(SRC_OBJS)
OBJS_POOL := $(BUILDDIR)/pooling_sim.o    $(SRC_OBJS)
OBJS_VEC  := $(BUILDDIR)/vec_ops_sim.o    $(SRC_OBJS)

DEPS := $(BUILDDIR)/layer_norm_sim.d  \
        $(BUILDDIR)/dw_conv2d_sim.d   \
        $(BUILDDIR)/pooling_sim.d     \
        $(BUILDDIR)/vec_ops_sim.d     \
        $(SRC_OBJS:.o=.d)

.PHONY: all run run_ln run_dw run_pool run_vec clean

all: $(TARGET_LN) $(TARGET_DW) $(TARGET_POOL) $(TARGET_VEC)

$(TARGET_LN): $(OBJS_LN) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(TARGET_DW): $(OBJS_DW) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(TARGET_POOL): $(OBJS_POOL) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(TARGET_VEC): $(OBJS_VEC) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

# Compile kernel/*.cpp
$(BUILDDIR)/%.o: %.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Compile src/*.cpp (prefixed with src_ to avoid name collisions)
$(BUILDDIR)/src_%.o: ../src/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

run: run_vec

run_ln: $(TARGET_LN)
	./$(TARGET_LN)

run_dw: $(TARGET_DW)
	./$(TARGET_DW)

run_pool: $(TARGET_POOL)
	./$(TARGET_POOL)

run_vec: $(TARGET_VEC)
	./$(TARGET_VEC)

clean:
	rm -rf $(BUILDDIR)

-include $(DEPS)
