CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-parameter \
            -DSC_INCLUDE_DYNAMIC_PROCESSES \
            -MMD -MP
INCLUDES := -I/usr/include -I../src -I.
LDFLAGS  := -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu
LIBS     := -lsystemc -lpthread

TARGET   := build/nafnet_perf_sim
BUILDDIR := build

OBJS := $(BUILDDIR)/memory.o      \
        $(BUILDDIR)/accelerator.o \
        $(BUILDDIR)/interconnect.o \
        $(BUILDDIR)/nafnet_sim.o

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

$(BUILDDIR)/memory.o: ../src/memory.cpp ../src/memory.h ../src/common.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/accelerator.o: ../src/accelerator.cpp ../src/accelerator.h ../src/interconnect.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/interconnect.o: ../src/interconnect.cpp ../src/interconnect.h ../src/extensions.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR)/nafnet_sim.o: nafnet_sim.cpp nafnet_layers.h nafnet_hw_config.h | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILDDIR)

# Auto-generated header dependency rules
-include $(OBJS:.o=.d)
