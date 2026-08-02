ARCH := x86
BPF_CFLAGS := -O2 -g -target bpf -D__TARGET_ARCH_$(ARCH) -I/usr/include/$(shell uname -m)-linux-gnu
CFLAGS := -O2 -g -Wall -Wextra $(shell pkg-config --cflags libbpf) -I/usr/include/$(shell uname -m)-linux-gnu
LDFLAGS := $(shell pkg-config --libs libbpf) -lelf -lz -pthread

.PHONY: all clean

all: fs_monitor.bpf.o fs_monitor

fs_monitor.bpf.o: fs_monitor.bpf.c
	clang $(BPF_CFLAGS) -c $< -o $@

fs_monitor: fs_monitor.c
	gcc $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f fs_monitor fs_monitor.bpf.o
