CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-parameter \
            -DSC_INCLUDE_DYNAMIC_PROCESSES \
            -MMD -MP
INCLUDES := -I/usr/include -I.. -I../src -I.
LDFLAGS  := -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu
LIBS     := -lsystemc -lpthread

TARGET   := build/layer_norm_sim
BUILDDIR := build

# src/ modules required by layer_norm_sim (excludes src/main.cpp and src/worker.cpp)
SRC_OBJS := $(BUILDDIR)/src_memory.o       \
            $(BUILDDIR)/src_accelerator.o  \
            $(BUILDDIR)/src_accelerator_pool.o \
            $(BUILDDIR)/src_interconnect.o

KERNEL_OBJS := $(BUILDDIR)/layer_norm_sim.o

OBJS := $(KERNEL_OBJS) $(SRC_OBJS)
DEPS := $(OBJS:.o=.d)

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

# Compile kernel/*.cpp
$(BUILDDIR)/%.o: %.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# Compile src/*.cpp (prefixed with src_ to avoid name collisions)
$(BUILDDIR)/src_%.o: ../src/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILDDIR)

-include $(DEPS)
