#include <asm/unistd.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define PATH_LEN 128

struct fs_event {
  uint64_t cgroup_id;
  uint64_t ts_ns;
  uint32_t syscall_nr;
  uint32_t pid;
  int32_t fd;
  int32_t dirfd_primary;
  int32_t dirfd_secondary;
  uint64_t flags;
  char path[PATH_LEN];
  char path_secondary[PATH_LEN];
};

struct registration {
  char sandbox_id[128];
  uint64_t cgroup_id;
  struct registration *next;
};

#define PENDING_SYNC_CAPACITY 1024

struct helper_state {
  int registered_cgroups_fd;
  int ignored_pids_fd;
  struct registration *registrations;
  pthread_mutex_t lock;
  uint64_t pending_sync_ids[PENDING_SYNC_CAPACITY];
  size_t pending_sync_count;
  size_t pending_sync_overflow;
  /* Instrumentation: accumulated wall time (microseconds) spent inside
   * ring_buffer__poll + ring_buffer__consume between two sync_ack
   * emissions, and the number of events processed in that window. Both
   * are reset after emit_sync_ack so Python can attribute "helper CPU
   * time" and "event backlog drained" to each sync() round-trip. */
  uint64_t drain_us_since_prev_sync;
  uint64_t events_since_prev_sync;
};

static volatile sig_atomic_t stop_flag;

static void on_signal(int signo)
{
  (void)signo;
  stop_flag = 1;
}

static const char *syscall_name(uint32_t syscall_nr)
{
  switch (syscall_nr) {
    case __NR_write:
      return "write";
    case __NR_pwrite64:
      return "pwrite64";
    case __NR_writev:
      return "writev";
    case __NR_pwritev:
      return "pwritev";
    case __NR_pwritev2:
      return "pwritev2";
    case __NR_open:
      return "open";
    case __NR_openat:
      return "openat";
    case __NR_openat2:
      return "openat2";
    case __NR_creat:
      return "creat";
    case __NR_truncate:
      return "truncate";
    case __NR_ftruncate:
      return "ftruncate";
    case __NR_rename:
      return "rename";
    case __NR_renameat:
      return "renameat";
    case __NR_renameat2:
      return "renameat2";
    case __NR_unlink:
      return "unlink";
    case __NR_unlinkat:
      return "unlinkat";
    case __NR_mkdir:
      return "mkdir";
    case __NR_mkdirat:
      return "mkdirat";
    case __NR_rmdir:
      return "rmdir";
    case __NR_link:
      return "link";
    case __NR_linkat:
      return "linkat";
    case __NR_symlink:
      return "symlink";
    case __NR_symlinkat:
      return "symlinkat";
    case __NR_mknod:
      return "mknod";
    case __NR_mknodat:
      return "mknodat";
    case __NR_chmod:
      return "chmod";
    case __NR_fchmod:
      return "fchmod";
    case __NR_fchmodat:
      return "fchmodat";
    case __NR_chown:
      return "chown";
    case __NR_fchown:
      return "fchown";
    case __NR_fchownat:
      return "fchownat";
    case __NR_lchown:
      return "lchown";
    case __NR_setxattr:
      return "setxattr";
    case __NR_lsetxattr:
      return "lsetxattr";
    case __NR_fsetxattr:
      return "fsetxattr";
    case __NR_removexattr:
      return "removexattr";
    case __NR_lremovexattr:
      return "lremovexattr";
    case __NR_fremovexattr:
      return "fremovexattr";
    default:
      return "unknown";
  }
}

static int current_executable_dir(char *buffer, size_t buffer_len)
{
  ssize_t len = readlink("/proc/self/exe", buffer, buffer_len - 1);
  char *slash;

  if (len < 0)
    return -errno;
  buffer[len] = '\0';
  slash = strrchr(buffer, '/');
  if (!slash)
    return -EINVAL;
  *slash = '\0';
  return 0;
}

static int update_registration_map(struct helper_state *state, const char *sandbox_id, uint64_t cgroup_id)
{
  struct registration *item = state->registrations;
  uint8_t present = 1;

  pthread_mutex_lock(&state->lock);
  while (item) {
    if (strcmp(item->sandbox_id, sandbox_id) == 0) {
      if (item->cgroup_id != cgroup_id)
        bpf_map_delete_elem(state->registered_cgroups_fd, &item->cgroup_id);
      item->cgroup_id = cgroup_id;
      pthread_mutex_unlock(&state->lock);
      return bpf_map_update_elem(state->registered_cgroups_fd, &cgroup_id, &present, BPF_ANY);
    }
    item = item->next;
  }

  item = calloc(1, sizeof(*item));
  if (!item) {
    pthread_mutex_unlock(&state->lock);
    return -ENOMEM;
  }
  snprintf(item->sandbox_id, sizeof(item->sandbox_id), "%s", sandbox_id);
  item->cgroup_id = cgroup_id;
  item->next = state->registrations;
  state->registrations = item;
  pthread_mutex_unlock(&state->lock);
  return bpf_map_update_elem(state->registered_cgroups_fd, &cgroup_id, &present, BPF_ANY);
}

static int remove_registration_map(struct helper_state *state, const char *sandbox_id)
{
  struct registration *item = NULL;
  struct registration *prev = NULL;
  int rc = -ENOENT;

  pthread_mutex_lock(&state->lock);
  item = state->registrations;
  while (item) {
    if (strcmp(item->sandbox_id, sandbox_id) == 0) {
      if (prev)
        prev->next = item->next;
      else
        state->registrations = item->next;
      bpf_map_delete_elem(state->registered_cgroups_fd, &item->cgroup_id);
      free(item);
      rc = 0;
      break;
    }
    prev = item;
    item = item->next;
  }
  pthread_mutex_unlock(&state->lock);
  return rc;
}

static int add_ignored_pid_map(struct helper_state *state, uint32_t pid)
{
  uint8_t present = 1;
  if (pid == 0)
    return -EINVAL;
  return bpf_map_update_elem(state->ignored_pids_fd, &pid, &present, BPF_ANY);
}

static int remove_ignored_pid_map(struct helper_state *state, uint32_t pid)
{
  if (pid == 0)
    return -EINVAL;
  return bpf_map_delete_elem(state->ignored_pids_fd, &pid);
}

static int find_sandbox_id(struct helper_state *state, uint64_t cgroup_id, char *output, size_t output_len)
{
  struct registration *item;

  pthread_mutex_lock(&state->lock);
  item = state->registrations;
  while (item) {
    if (item->cgroup_id == cgroup_id) {
      snprintf(output, output_len, "%s", item->sandbox_id);
      pthread_mutex_unlock(&state->lock);
      return 0;
      break;
    }
    item = item->next;
  }
  pthread_mutex_unlock(&state->lock);
  return -ENOENT;
}

static int extract_string_field(const char *line, const char *key, char *output, size_t output_len)
{
  const char *start;
  const char *end;
  size_t len;
  char pattern[64];

  snprintf(pattern, sizeof(pattern), "\"%s\":\"", key);
  start = strstr(line, pattern);
  if (!start)
    return -ENOENT;
  start += strlen(pattern);
  end = strchr(start, '"');
  if (!end)
    return -EINVAL;
  len = (size_t)(end - start);
  if (len >= output_len)
    len = output_len - 1;
  memcpy(output, start, len);
  output[len] = '\0';
  return 0;
}

static int extract_u64_field(const char *line, const char *key, uint64_t *value)
{
  const char *start;
  char pattern[64];

  snprintf(pattern, sizeof(pattern), "\"%s\":", key);
  start = strstr(line, pattern);
  if (!start)
    return -ENOENT;
  start += strlen(pattern);
  *value = strtoull(start, NULL, 10);
  return 0;
}

static void *stdin_loop(void *arg)
{
  struct helper_state *state = arg;
  char *line = NULL;
  size_t cap = 0;

  while (!stop_flag && getline(&line, &cap, stdin) != -1) {
    char op[64];
    char sandbox_id[128];
    uint64_t cgroup_id;

    if (extract_string_field(line, "op", op, sizeof(op)) != 0)
      continue;

    if (strcmp(op, "sync") == 0) {
      uint64_t sync_id = 0;
      if (extract_u64_field(line, "sync_id", &sync_id) != 0 || sync_id == 0)
        continue;
      pthread_mutex_lock(&state->lock);
      if (state->pending_sync_count < PENDING_SYNC_CAPACITY) {
        state->pending_sync_ids[state->pending_sync_count++] = sync_id;
      } else {
        state->pending_sync_overflow++;
      }
      pthread_mutex_unlock(&state->lock);
      continue;
    }

    if (strcmp(op, "add_ignored_pid") == 0) {
      uint64_t pid = 0;
      if (extract_u64_field(line, "pid", &pid) != 0 || pid == 0)
        continue;
      if (add_ignored_pid_map(state, (uint32_t)pid) != 0)
        fprintf(stderr, "failed to add ignored pid %llu\n", (unsigned long long)pid);
      continue;
    }

    if (strcmp(op, "remove_ignored_pid") == 0) {
      uint64_t pid = 0;
      if (extract_u64_field(line, "pid", &pid) != 0 || pid == 0)
        continue;
      remove_ignored_pid_map(state, (uint32_t)pid);
      continue;
    }

    if (extract_string_field(line, "sandbox_id", sandbox_id, sizeof(sandbox_id)) != 0)
      continue;

    if (strcmp(op, "upsert_sandbox") == 0) {
      if (extract_u64_field(line, "cgroup_id", &cgroup_id) != 0)
        continue;
      if (update_registration_map(state, sandbox_id, cgroup_id) != 0)
        fprintf(stderr, "failed to upsert sandbox %s\n", sandbox_id);
      continue;
    }

    if (strcmp(op, "remove_sandbox") == 0) {
      remove_registration_map(state, sandbox_id);
      continue;
    }
  }

  free(line);
  stop_flag = 1;
  return NULL;
}

static void format_now_iso8601(char *buffer, size_t buffer_len)
{
  struct timespec ts;
  struct tm tm;

  clock_gettime(CLOCK_REALTIME, &ts);
  gmtime_r(&ts.tv_sec, &tm);
  snprintf(
    buffer,
    buffer_len,
    "%04d-%02d-%02dT%02d:%02d:%02d.%03ld+00:00",
    tm.tm_year + 1900,
    tm.tm_mon + 1,
    tm.tm_mday,
    tm.tm_hour,
    tm.tm_min,
    tm.tm_sec,
    ts.tv_nsec / 1000000L
  );
}

static void json_print_string_or_null(const char *value)
{
  const unsigned char *cursor = (const unsigned char *)value;

  if (!value || value[0] == '\0') {
    fputs("null", stdout);
    return;
  }

  fputc('"', stdout);
  while (*cursor) {
    switch (*cursor) {
      case '\\':
        fputs("\\\\", stdout);
        break;
      case '"':
        fputs("\\\"", stdout);
        break;
      case '\n':
        fputs("\\n", stdout);
        break;
      case '\r':
        fputs("\\r", stdout);
        break;
      case '\t':
        fputs("\\t", stdout);
        break;
      default:
        if (*cursor < 0x20)
          fprintf(stdout, "\\u%04x", (unsigned int)*cursor);
        else
          fputc(*cursor, stdout);
        break;
    }
    cursor++;
  }
  fputc('"', stdout);
}

static void strip_deleted_suffix(char *path)
{
  static const char suffix[] = " (deleted)";
  char *found = strstr(path, suffix);

  if (found)
    *found = '\0';
}

static int read_proc_link(const char *path, char *output, size_t output_len)
{
  ssize_t len = readlink(path, output, output_len - 1);
  if (len < 0)
    return -errno;
  output[len] = '\0';
  strip_deleted_suffix(output);
  return 0;
}

static int resolve_base_path(uint32_t pid, int32_t dirfd, char *output, size_t output_len)
{
  char proc_path[128];

  if (pid == 0)
    return -ENOENT;
  if (dirfd == AT_FDCWD)
    snprintf(proc_path, sizeof(proc_path), "/proc/%u/cwd", pid);
  else if (dirfd >= 0)
    snprintf(proc_path, sizeof(proc_path), "/proc/%u/fd/%d", pid, dirfd);
  else
    return -ENOENT;
  return read_proc_link(proc_path, output, output_len);
}

static void join_paths(char *output, size_t output_len, const char *base, const char *suffix)
{
  if (!suffix || suffix[0] == '\0') {
    output[0] = '\0';
    return;
  }
  if (suffix[0] == '/') {
    snprintf(output, output_len, "%s", suffix);
    return;
  }
  if (!base || base[0] == '\0') {
    snprintf(output, output_len, "%s", suffix);
    return;
  }
  snprintf(output, output_len, "%s/%s", base, suffix);
}

static void resolve_event_path(
  uint32_t pid,
  int32_t dirfd,
  const char *raw_path,
  char *output,
  size_t output_len
)
{
  char base[PATH_MAX];

  output[0] = '\0';
  if (!raw_path || raw_path[0] == '\0')
    return;
  if (raw_path[0] == '/') {
    snprintf(output, output_len, "%s", raw_path);
    return;
  }
  if (resolve_base_path(pid, dirfd, base, sizeof(base)) == 0) {
    join_paths(output, output_len, base, raw_path);
    return;
  }
  snprintf(output, output_len, "%s", raw_path);
}

static void fd_kind_from_mode(mode_t mode, char *fd_kind, size_t fd_kind_len)
{
  if (S_ISREG(mode))
    snprintf(fd_kind, fd_kind_len, "regular");
  else if (S_ISDIR(mode))
    snprintf(fd_kind, fd_kind_len, "directory");
  else if (S_ISLNK(mode))
    snprintf(fd_kind, fd_kind_len, "symlink");
  else if (S_ISCHR(mode))
    snprintf(fd_kind, fd_kind_len, "char");
  else if (S_ISBLK(mode))
    snprintf(fd_kind, fd_kind_len, "block");
  else if (S_ISFIFO(mode))
    snprintf(fd_kind, fd_kind_len, "fifo");
  else if (S_ISSOCK(mode))
    snprintf(fd_kind, fd_kind_len, "socket");
}

static int resolve_fd_identity(
  uint32_t pid,
  int32_t fd,
  char *path_output,
  size_t path_output_len,
  char *fd_kind,
  size_t fd_kind_len,
  struct stat *st
)
{
  char fd_path[128];

  if (pid == 0 || fd < 0)
    return -ENOENT;
  snprintf(fd_path, sizeof(fd_path), "/proc/%u/fd/%d", pid, fd);
  if (read_proc_link(fd_path, path_output, path_output_len) != 0)
    path_output[0] = '\0';
  if (stat(fd_path, st) != 0)
    return -errno;
  fd_kind_from_mode(st->st_mode, fd_kind, fd_kind_len);
  return 0;
}

static int resolve_path_identity(const char *path, struct stat *st)
{
  if (!path || path[0] == '\0')
    return -ENOENT;
  if (lstat(path, st) != 0)
    return -errno;
  return 0;
}

static int handle_event(void *ctx, void *data, size_t data_sz)
{
  struct helper_state *state = ctx;
  const struct fs_event *event = data;
  char sandbox_id[128];
  char fd_kind[16] = "";
  char primary_path[PATH_MAX] = "";
  char secondary_path[PATH_MAX] = "";
  char fd_resolved_path[PATH_MAX] = "";
  char ts[64];
  struct stat st = {};
  struct stat path_st = {};
  bool have_identity = false;
  uint64_t device = 0;
  uint64_t inode = 0;

  (void)data_sz;
  if (find_sandbox_id(state, event->cgroup_id, sandbox_id, sizeof(sandbox_id)) != 0)
    return 0;

  /* Count only events that belong to a registered sandbox (the path
   * that actually performs /proc syscalls + stdout write). Events for
   * unregistered cgroups return above and cost ~zero. */
  state->events_since_prev_sync++;

  resolve_event_path(event->pid, event->dirfd_primary, event->path, primary_path, sizeof(primary_path));
  resolve_event_path(
    event->pid,
    event->dirfd_secondary,
    event->path_secondary,
    secondary_path,
    sizeof(secondary_path)
  );

  if (resolve_fd_identity(
        event->pid,
        event->fd,
        fd_resolved_path,
        sizeof(fd_resolved_path),
        fd_kind,
        sizeof(fd_kind),
        &st
      ) == 0) {
    device = (uint64_t)st.st_dev;
    inode = (uint64_t)st.st_ino;
    have_identity = true;
  } else if (
    (event->syscall_nr == __NR_rename || event->syscall_nr == __NR_renameat || event->syscall_nr == __NR_renameat2 ||
     event->syscall_nr == __NR_link || event->syscall_nr == __NR_linkat || event->syscall_nr == __NR_symlink ||
     event->syscall_nr == __NR_symlinkat) &&
    resolve_path_identity(secondary_path, &path_st) == 0
  ) {
    device = (uint64_t)path_st.st_dev;
    inode = (uint64_t)path_st.st_ino;
    if (fd_kind[0] == '\0')
      fd_kind_from_mode(path_st.st_mode, fd_kind, sizeof(fd_kind));
    have_identity = true;
  } else if (resolve_path_identity(primary_path, &path_st) == 0) {
    device = (uint64_t)path_st.st_dev;
    inode = (uint64_t)path_st.st_ino;
    if (fd_kind[0] == '\0')
      fd_kind_from_mode(path_st.st_mode, fd_kind, sizeof(fd_kind));
    have_identity = true;
  }

  format_now_iso8601(ts, sizeof(ts));
  printf("{\"cgroup_id\":%llu", (unsigned long long)event->cgroup_id);
  printf(",\"fd\":%d", event->fd);
  printf(",\"fd_kind\":");
  json_print_string_or_null(fd_kind[0] == '\0' ? NULL : fd_kind);
  printf(",\"flags\":%llu", (unsigned long long)event->flags);
  if (have_identity) {
    printf(",\"device\":%llu", (unsigned long long)device);
    printf(",\"inode\":%llu", (unsigned long long)inode);
  } else {
    printf(",\"device\":null,\"inode\":null");
  }
  printf(",\"kind\":\"filesystem_change\"");
  printf(",\"path\":");
  if (primary_path[0] != '\0')
    json_print_string_or_null(primary_path);
  else if (fd_resolved_path[0] != '\0')
    json_print_string_or_null(fd_resolved_path);
  else
    json_print_string_or_null(NULL);
  printf(",\"path_secondary\":");
  json_print_string_or_null(secondary_path[0] == '\0' ? NULL : secondary_path);
  printf(",\"pid\":%u", event->pid);
  printf(",\"sandbox_id\":");
  json_print_string_or_null(sandbox_id);
  printf(",\"syscall\":");
  json_print_string_or_null(syscall_name(event->syscall_nr));
  printf(",\"timestamp\":");
  json_print_string_or_null(ts);
  printf("}\n");
  fflush(stdout);
  return 0;
}

static void emit_sync_ack(struct helper_state *state, uint64_t sync_id)
{
  /* Attach per-round-trip instrumentation so Python can log a latency
   * breakdown (helper drain vs worker drain) without extra syscalls. */
  printf(
    "{\"kind\":\"sync_ack\",\"sync_id\":%llu,\"drain_us\":%llu,\"events\":%llu}\n",
    (unsigned long long)sync_id,
    (unsigned long long)state->drain_us_since_prev_sync,
    (unsigned long long)state->events_since_prev_sync
  );
  fflush(stdout);
  state->drain_us_since_prev_sync = 0;
  state->events_since_prev_sync = 0;
}

static uint64_t monotonic_us(void)
{
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
    return 0;
  return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

int main(void)
{
  struct helper_state state = {};
  struct bpf_object *obj = NULL;
  struct bpf_program *prog;
  struct bpf_link *links[8] = {};
  size_t link_count = 0;
  struct ring_buffer *rb = NULL;
  pthread_t thread;
  int err = 0;
  char object_path[512];

  signal(SIGINT, on_signal);
  signal(SIGTERM, on_signal);
  setvbuf(stdout, NULL, _IOLBF, 0);
  pthread_mutex_init(&state.lock, NULL);

  libbpf_set_strict_mode(LIBBPF_STRICT_ALL);
  err = current_executable_dir(object_path, sizeof(object_path));
  if (err != 0) {
    fprintf(stderr, "failed to resolve executable dir: %d\n", err);
    goto cleanup;
  }
  strncat(object_path, "/fs_monitor.bpf.o", sizeof(object_path) - strlen(object_path) - 1);

  obj = bpf_object__open_file(object_path, NULL);
  if (libbpf_get_error(obj)) {
    err = (int)libbpf_get_error(obj);
    obj = NULL;
    fprintf(stderr, "failed to open BPF object: %d\n", err);
    goto cleanup;
  }

  err = bpf_object__load(obj);
  if (err != 0) {
    fprintf(stderr, "failed to load BPF object: %d\n", err);
    goto cleanup;
  }

  state.registered_cgroups_fd = bpf_object__find_map_fd_by_name(obj, "registered_cgroups");
  if (state.registered_cgroups_fd < 0) {
    err = state.registered_cgroups_fd;
    fprintf(stderr, "failed to find registered_cgroups map: %d\n", err);
    goto cleanup;
  }

  state.ignored_pids_fd = bpf_object__find_map_fd_by_name(obj, "ignored_pids");
  if (state.ignored_pids_fd < 0) {
    err = state.ignored_pids_fd;
    fprintf(stderr, "failed to find ignored_pids map: %d\n", err);
    goto cleanup;
  }

  bpf_object__for_each_program(prog, obj) {
    struct bpf_link *link = bpf_program__attach(prog);
    if (libbpf_get_error(link)) {
      err = (int)libbpf_get_error(link);
      fprintf(stderr, "failed to attach program %s: %d\n", bpf_program__name(prog), err);
      goto cleanup;
    }
    links[link_count++] = link;
  }

  rb = ring_buffer__new(bpf_object__find_map_fd_by_name(obj, "events"), handle_event, &state, NULL);
  if (!rb) {
    err = -errno;
    fprintf(stderr, "failed to create ring buffer: %d\n", err);
    goto cleanup;
  }

  if (pthread_create(&thread, NULL, stdin_loop, &state) != 0) {
    err = -errno;
    fprintf(stderr, "failed to create stdin thread: %d\n", err);
    goto cleanup;
  }

  while (!stop_flag) {
    uint64_t sync_batch[PENDING_SYNC_CAPACITY];
    size_t sync_batch_count = 0;
    size_t sync_overflow = 0;
    uint64_t poll_start_us = monotonic_us();

    err = ring_buffer__poll(rb, 50);
    if (err == -EINTR)
      continue;
    if (err < 0) {
      fprintf(stderr, "ring buffer poll failed: %d\n", err);
      break;
    }
    /* Only count wall-clock time as "drain work" when poll actually
     * returned events — an idle poll (returned 0) just burned the 50ms
     * timeout and isn't interesting for the bottleneck analysis. */
    if (err > 0)
      state.drain_us_since_prev_sync += monotonic_us() - poll_start_us;

    pthread_mutex_lock(&state.lock);
    sync_batch_count = state.pending_sync_count;
    if (sync_batch_count > 0) {
      memcpy(sync_batch, state.pending_sync_ids, sync_batch_count * sizeof(sync_batch[0]));
      state.pending_sync_count = 0;
    }
    sync_overflow = state.pending_sync_overflow;
    state.pending_sync_overflow = 0;
    pthread_mutex_unlock(&state.lock);

    if (sync_overflow > 0)
      fprintf(stderr, "sync queue overflow: %zu syncs dropped\n", sync_overflow);

    if (sync_batch_count > 0) {
      /* Drain any events that landed between the poll return and the
       * sync request so the ack truly means "all in-flight events are
       * processed". ring_buffer__consume is non-blocking and runs the
       * callback for each ready event. One drain satisfies every queued
       * sync because each ack only promises "events up to this point
       * have been processed" — and the latest drain covers all earlier
       * ones. */
      uint64_t consume_start_us = monotonic_us();
      int consumed = ring_buffer__consume(rb);
      if (consumed < 0) {
        fprintf(stderr, "ring buffer consume failed: %d\n", consumed);
        break;
      }
      if (consumed > 0)
        state.drain_us_since_prev_sync += monotonic_us() - consume_start_us;
      for (size_t i = 0; i < sync_batch_count; i++)
        emit_sync_ack(&state, sync_batch[i]);
    }
  }

  stop_flag = 1;
  pthread_join(thread, NULL);
  err = 0;

cleanup:
  if (rb)
    ring_buffer__free(rb);
  while (link_count > 0)
    bpf_link__destroy(links[--link_count]);
  if (obj)
    bpf_object__close(obj);
  return err != 0;
}
