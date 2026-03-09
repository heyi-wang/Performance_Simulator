CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wno-unused-parameter
INCLUDES := -I/usr/include
LDFLAGS  := -L/usr/lib/x86_64-linux-gnu -Wl,-rpath,/usr/lib/x86_64-linux-gnu
LIBS     := -lsystemc -lpthread

TARGET   := perf_sim
SRCS     := perf_sim.cpp
DEPS     := config.h

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRCS) $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $(SRCS) $(LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
