CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-parameter \
            -DSC_INCLUDE_DYNAMIC_PROCESSES \
            -MMD -MP
INCLUDES := -I/usr/include -I.. -I.
LDFLAGS  := -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu
LIBS     := -lsystemc -lpthread

TARGET   := build/matmul_sim
BUILDDIR := build

# Source files: new matmul_sim files + all src/ modules (excluding src/main.cpp)
MATMUL_SRCS := matmul_main.cpp \
               matmul_top.cpp  \
               accum_coordinator.cpp

SRC_SRCS := $(filter-out ../src/main.cpp, $(wildcard ../src/*.cpp))

ALL_SRCS := $(MATMUL_SRCS) $(SRC_SRCS)

# Object files placed in build/
MATMUL_OBJS := $(MATMUL_SRCS:%.cpp=$(BUILDDIR)/%.o)
SRC_OBJS    := $(SRC_SRCS:../src/%.cpp=$(BUILDDIR)/src_%.o)
OBJS        := $(MATMUL_OBJS) $(SRC_OBJS)
DEPS        := $(OBJS:.o=.d)

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(OBJS) | $(BUILDDIR)
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

# Compile matmul_sim/*.cpp
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
