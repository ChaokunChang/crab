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
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

struct fs_event {
  uint64_t cgroup_id;
  uint64_t ts_ns;
  uint32_t syscall_nr;
  uint32_t pid;
  int32_t fd;
  uint64_t flags;
};

struct registration {
  char sandbox_id[128];
  uint64_t cgroup_id;
  struct registration *next;
};

struct helper_state {
  int registered_cgroups_fd;
  struct registration *registrations;
  pthread_mutex_t lock;
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

static int handle_event(void *ctx, void *data, size_t data_sz)
{
  struct helper_state *state = ctx;
  const struct fs_event *event = data;
  char sandbox_id[128];
  char fd_kind[16] = "unknown";
  char fd_path[128];
  char ts[64];
  struct stat st;

  (void)data_sz;
  if (find_sandbox_id(state, event->cgroup_id, sandbox_id, sizeof(sandbox_id)) != 0)
    return 0;

  if (event->pid > 0 && event->fd >= 0) {
    snprintf(fd_path, sizeof(fd_path), "/proc/%u/fd/%d", event->pid, event->fd);
    if (stat(fd_path, &st) == 0) {
      if (S_ISREG(st.st_mode))
        snprintf(fd_kind, sizeof(fd_kind), "regular");
      else if (S_ISDIR(st.st_mode))
        snprintf(fd_kind, sizeof(fd_kind), "directory");
      else if (S_ISCHR(st.st_mode))
        snprintf(fd_kind, sizeof(fd_kind), "char");
      else if (S_ISBLK(st.st_mode))
        snprintf(fd_kind, sizeof(fd_kind), "block");
      else if (S_ISFIFO(st.st_mode))
        snprintf(fd_kind, sizeof(fd_kind), "fifo");
      else if (S_ISSOCK(st.st_mode))
        snprintf(fd_kind, sizeof(fd_kind), "socket");
    }
  }

  format_now_iso8601(ts, sizeof(ts));
  printf(
    "{\"cgroup_id\":%llu,\"fd\":%d,\"fd_kind\":\"%s\",\"flags\":%llu,\"kind\":\"filesystem_change\",\"pid\":%u,\"sandbox_id\":\"%s\",\"syscall\":\"%s\",\"timestamp\":\"%s\"}\n",
    (unsigned long long)event->cgroup_id,
    event->fd,
    fd_kind,
    (unsigned long long)event->flags,
    event->pid,
    sandbox_id,
    syscall_name(event->syscall_nr),
    ts
  );
  fflush(stdout);
  return 0;
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
    err = ring_buffer__poll(rb, 250);
    if (err == -EINTR)
      continue;
    if (err < 0) {
      fprintf(stderr, "ring buffer poll failed: %d\n", err);
      break;
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
