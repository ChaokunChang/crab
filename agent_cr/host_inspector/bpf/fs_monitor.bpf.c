#include <asm/unistd.h>
#include <linux/bpf.h>
#include <linux/fcntl.h>
#include <stdbool.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char LICENSE[] SEC("license") = "Dual BSD/GPL";

struct trace_event_raw_sys_enter {
  __u16 common_type;
  __u8 common_flags;
  __u8 common_preempt_count;
  __s32 common_pid;
  long id;
  unsigned long args[6];
};

struct trace_event_raw_sys_exit {
  __u16 common_type;
  __u8 common_flags;
  __u8 common_preempt_count;
  __s32 common_pid;
  long id;
  long ret;
};

struct open_how_min {
  __u64 flags;
  __u64 mode;
  __u64 resolve;
};

struct pending_op {
  __u32 syscall_nr;
  __s32 fd;
  __u64 flags;
};

struct fs_event {
  __u64 cgroup_id;
  __u64 ts_ns;
  __u32 syscall_nr;
  __u32 pid;
  __s32 fd;
  __u64 flags;
};

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 1024);
  __type(key, __u64);
  __type(value, __u8);
} registered_cgroups SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_HASH);
  __uint(max_entries, 8192);
  __type(key, __u32);
  __type(value, struct pending_op);
} pending_ops SEC(".maps");

struct {
  __uint(type, BPF_MAP_TYPE_RINGBUF);
  __uint(max_entries, 1 << 24);
} events SEC(".maps");

static __always_inline bool is_registered_cgroup(__u64 cgroup_id)
{
  __u8 *value = bpf_map_lookup_elem(&registered_cgroups, &cgroup_id);
  return value != 0;
}

static __always_inline bool is_tracked_syscall(long syscall_nr)
{
  switch (syscall_nr) {
    case __NR_write:
    case __NR_pwrite64:
    case __NR_writev:
    case __NR_pwritev:
    case __NR_pwritev2:
    case __NR_open:
    case __NR_openat:
    case __NR_openat2:
    case __NR_creat:
    case __NR_truncate:
    case __NR_ftruncate:
    case __NR_rename:
    case __NR_renameat:
    case __NR_renameat2:
    case __NR_unlink:
    case __NR_unlinkat:
    case __NR_mkdir:
    case __NR_mkdirat:
    case __NR_rmdir:
    case __NR_link:
    case __NR_linkat:
    case __NR_symlink:
    case __NR_symlinkat:
    case __NR_mknod:
    case __NR_mknodat:
    case __NR_chmod:
    case __NR_fchmod:
    case __NR_fchmodat:
    case __NR_chown:
    case __NR_fchown:
    case __NR_fchownat:
    case __NR_lchown:
    case __NR_setxattr:
    case __NR_lsetxattr:
    case __NR_fsetxattr:
    case __NR_removexattr:
    case __NR_lremovexattr:
    case __NR_fremovexattr:
      return true;
    default:
      return false;
  }
}

static __always_inline __u64 open_flags_from_enter(long syscall_nr, struct trace_event_raw_sys_enter *ctx)
{
  struct open_how_min how = {};

  switch (syscall_nr) {
    case __NR_open:
      return (__u64)ctx->args[1];
    case __NR_openat:
      return (__u64)ctx->args[2];
    case __NR_openat2:
      bpf_probe_read_user(&how, sizeof(how), (const void *)ctx->args[2]);
      return how.flags;
    case __NR_creat:
      return O_WRONLY | O_CREAT | O_TRUNC;
    default:
      return 0;
  }
}

static __always_inline __s32 fd_from_enter(long syscall_nr, struct trace_event_raw_sys_enter *ctx)
{
  switch (syscall_nr) {
    case __NR_write:
    case __NR_pwrite64:
    case __NR_writev:
    case __NR_pwritev:
    case __NR_pwritev2:
      return (__s32)ctx->args[0];
    default:
      return -1;
  }
}

static __always_inline bool should_emit(const struct pending_op *pending, long ret)
{
  __u64 mutating_open_flags_mask = O_CREAT | O_TRUNC;
#ifdef O_TMPFILE
  mutating_open_flags_mask |= O_TMPFILE;
#endif

  if (ret < 0)
    return false;

  switch (pending->syscall_nr) {
    case __NR_open:
    case __NR_openat:
    case __NR_openat2:
    case __NR_creat:
      return (pending->flags & mutating_open_flags_mask) != 0;
    case __NR_write:
    case __NR_pwrite64:
    case __NR_writev:
    case __NR_pwritev:
    case __NR_pwritev2:
      return pending->fd != 1 && pending->fd != 2;
    default:
      return true;
  }
}

static __always_inline bool is_open_syscall(__u32 syscall_nr)
{
  switch (syscall_nr) {
    case __NR_open:
    case __NR_openat:
    case __NR_openat2:
    case __NR_creat:
      return true;
    default:
      return false;
  }
}

SEC("tracepoint/raw_syscalls/sys_enter")
int handle_sys_enter(struct trace_event_raw_sys_enter *ctx)
{
  __u64 cgroup_id = bpf_get_current_cgroup_id();
  __u32 tid = (__u32)bpf_get_current_pid_tgid();
  struct pending_op pending = {};

  if (!is_registered_cgroup(cgroup_id))
    return 0;
  if (!is_tracked_syscall(ctx->id))
    return 0;

  pending.syscall_nr = (__u32)ctx->id;
  pending.fd = fd_from_enter(ctx->id, ctx);
  pending.flags = open_flags_from_enter(ctx->id, ctx);
  bpf_map_update_elem(&pending_ops, &tid, &pending, BPF_ANY);
  return 0;
}

SEC("tracepoint/raw_syscalls/sys_exit")
int handle_sys_exit(struct trace_event_raw_sys_exit *ctx)
{
  __u32 tid = (__u32)bpf_get_current_pid_tgid();
  struct pending_op *pending;
  struct fs_event *event;
  __u64 cgroup_id;

  pending = bpf_map_lookup_elem(&pending_ops, &tid);
  if (!pending)
    return 0;

  if ((__u32)ctx->id != pending->syscall_nr) {
    bpf_map_delete_elem(&pending_ops, &tid);
    return 0;
  }

  cgroup_id = bpf_get_current_cgroup_id();
  if (!is_registered_cgroup(cgroup_id)) {
    bpf_map_delete_elem(&pending_ops, &tid);
    return 0;
  }

  if (!should_emit(pending, ctx->ret)) {
    bpf_map_delete_elem(&pending_ops, &tid);
    return 0;
  }

  event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
  if (!event) {
    bpf_map_delete_elem(&pending_ops, &tid);
    return 0;
  }

  event->cgroup_id = cgroup_id;
  event->ts_ns = bpf_ktime_get_ns();
  event->syscall_nr = pending->syscall_nr;
  event->pid = (__u32)(bpf_get_current_pid_tgid() >> 32);
  event->fd = is_open_syscall(pending->syscall_nr) ? (__s32)ctx->ret : pending->fd;
  event->flags = pending->flags;
  bpf_ringbuf_submit(event, 0);
  bpf_map_delete_elem(&pending_ops, &tid);
  return 0;
}
