ARCH := x86
LIBBPF_HDR := /usr/src/linux-headers-$(shell uname -r)/tools/bpf/resolve_btfids/libbpf/include
BPF_CFLAGS := -O2 -g -target bpf -D__TARGET_ARCH_$(ARCH) -I$(LIBBPF_HDR) -I/usr/include/$(shell uname -m)-linux-gnu
CFLAGS := -O2 -g -Wall -Wextra -I$(LIBBPF_HDR) -I/usr/include/$(shell uname -m)-linux-gnu
LIBBPF_SO := $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libbpf.so.* /lib/x86_64-linux-gnu/libbpf.so.*))
LIBELF_SO := $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libelf.so.* /lib/x86_64-linux-gnu/libelf.so.*))
LIBZ_SO := $(firstword $(wildcard /usr/lib/x86_64-linux-gnu/libz.so.* /lib/x86_64-linux-gnu/libz.so.*))
LDFLAGS := $(LIBBPF_SO) $(LIBELF_SO) $(LIBZ_SO) -pthread

.PHONY: all clean

all: fs_monitor.bpf.o fs_monitor

fs_monitor.bpf.o: fs_monitor.bpf.c
	clang $(BPF_CFLAGS) -c $< -o $@

fs_monitor: fs_monitor.c
	gcc $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f fs_monitor fs_monitor.bpf.o
