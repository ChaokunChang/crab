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
#define IGNORED_PATH_PREFIX_MAX_LEN 256
#define IGNORED_PATH_PREFIXES_PER_SANDBOX 16

/* Process-ignore-rule limits. Mirror server.py:ProcessIgnoreRule shape:
 *   - executable_basename, executable_path_contains, ancestor_executable_basename
 *     are single strings; cmdline_contains is a list of substrings, all of
 *     which must appear in the NUL-joined cmdline.
 * Sized for the terminus tmux integration's actual rule set (≤4 rules,
 * each cmdline_contains ≤2 entries). */
#define IGNORE_RULES_PER_SANDBOX 16
#define RULE_FIELD_MAX_LEN 256
#define CMDLINE_CONTAINS_PARTS_MAX 8

struct ignore_rule {
  char executable_basename[RULE_FIELD_MAX_LEN];
  char executable_path_contains[RULE_FIELD_MAX_LEN];
  size_t cmdline_part_count;
  char cmdline_parts[CMDLINE_CONTAINS_PARTS_MAX][RULE_FIELD_MAX_LEN];
  char ancestor_executable_basename[RULE_FIELD_MAX_LEN];
};

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
  /* Per-sandbox path-prefix filter for fs events. Targets host-side helper
   * writes (CRIU's `dump.log` to the per-sandbox checkpoint dir, runc
   * state metadata, etc.) that get attributed to the sandbox cgroup
   * because they are written by tasks that transiently joined the
   * sandbox cgroup (CRIU parasite injected into container processes,
   * runc init helpers, etc.). Filtering here in the C helper avoids the
   * JSON serialization + Python parse + per-event sandbox_lock
   * acquisition cost that would otherwise burn through the per-worker
   * fs_monitor.sync budget — all for events that have nothing to do
   * with the sandbox's own filesystem state. */
  size_t ignored_path_prefix_count;
  char ignored_path_prefixes[IGNORED_PATH_PREFIXES_PER_SANDBOX][IGNORED_PATH_PREFIX_MAX_LEN];
  size_t ignored_path_prefix_lens[IGNORED_PATH_PREFIXES_PER_SANDBOX];
  /* Per-sandbox process-ignore-rule filter. Mirrors the fs-eligible
   * subset (scope=all) of server.py:ignore_process_rules. Events whose
   * pid identity matches any rule are dropped here so the JSON encode +
   * IPC + Python decode + sandbox_lock cost never gets paid. The Python
   * daemon still re-evaluates the same rules in `_handle_fs_event` as
   * belt-and-suspenders for any path that bypasses the helper.
   * `any_rule_needs_ancestors` is computed at push time so the per-event
   * fast path can skip the parent-chain walk when no rule needs it. */
  size_t ignore_rule_count;
  struct ignore_rule ignore_rules[IGNORE_RULES_PER_SANDBOX];
  bool any_rule_needs_ancestors;
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

/* Mirror of server.py:_has_mutating_open_flags. Open-family events that
 * lack O_CREAT|O_TRUNC|O_TMPFILE cannot mutate the filesystem and must
 * not count toward filesystem_changed. Note: fs_monitor.bpf.c already
 * filters these at the kernel side via should_emit, so this branch
 * mostly serves as defense-in-depth for any open that slips through. */
static bool has_mutating_open_flags(uint64_t flags)
{
  uint64_t mask = O_CREAT | O_TRUNC;
#ifdef O_TMPFILE
  mask |= O_TMPFILE;
#endif
  return (flags & mask) != 0;
}

/* Mirror of server.py:_path_is_likely_persistent. /dev, /proc, /sys are
 * pseudo-filesystems whose mutating opens never represent real on-disk
 * state changes; they must be dropped even if fd_kind happens to look
 * "regular" (e.g. /proc/<pid>/fd/<n> targets). */
static bool path_is_likely_persistent(const char *path)
{
  if (!path || path[0] != '/')
    return false;
  if (strcmp(path, "/dev") == 0 || strcmp(path, "/proc") == 0 || strcmp(path, "/sys") == 0)
    return false;
  if (strncmp(path, "/dev/", 5) == 0)
    return false;
  if (strncmp(path, "/proc/", 6) == 0)
    return false;
  if (strncmp(path, "/sys/", 5) == 0)
    return false;
  return true;
}

static bool fd_kind_is_device_or_stream(const char *fd_kind)
{
  if (!fd_kind || fd_kind[0] == '\0')
    return false;
  return strcmp(fd_kind, "char") == 0 || strcmp(fd_kind, "block") == 0 ||
         strcmp(fd_kind, "fifo") == 0 || strcmp(fd_kind, "socket") == 0;
}

/* Mirror of server.py:_MUTATING_FD_KINDS. Empty/unresolved fd_kind maps
 * to Python's `event.fd_kind or "unknown"` fallback — kept by design so
 * a stat() race on /proc/<pid>/fd cannot drop a real write. */
static bool fd_kind_is_mutating(const char *fd_kind)
{
  if (!fd_kind || fd_kind[0] == '\0')
    return true;
  return strcmp(fd_kind, "regular") == 0 || strcmp(fd_kind, "directory") == 0 ||
         strcmp(fd_kind, "symlink") == 0 || strcmp(fd_kind, "unknown") == 0;
}

/* Mirror of server.py:_is_countable_fs_event. Returns true iff this
 * event represents a real filesystem mutation; non-countable events
 * are dropped here so they never cross IPC, never get JSON-decoded in
 * Python, and never sit in a per-sandbox worker queue waiting for
 * sandbox_lock. Pre-port, every such event was emitted and rejected
 * downstream — wasted work that bled the fs_sync budget under burst
 * workloads (apt-get, tar, build). */
static bool is_countable_fs_event(
  uint32_t syscall_nr,
  uint64_t flags,
  int32_t fd,
  const char *fd_kind,
  const char *primary_path,
  const char *secondary_path
)
{
  switch (syscall_nr) {
    case __NR_open:
    case __NR_openat:
    case __NR_openat2:
    case __NR_creat:
      if (!has_mutating_open_flags(flags))
        return false;
      /* When the absolute path is a real file, trust it over fd_kind:
       * resolve_fd_identity stat()s /proc/<pid>/fd/<fd> AFTER the
       * syscall returned and races with dup2/close/exec in the tracee.
       * `cat > file << 'EOF'` opens the real file then dup2/close/pipe-
       * reuses the slot, so the post-hoc stat sees a fifo for what was
       * actually a regular-file write. */
      if (path_is_likely_persistent(primary_path))
        return true;
      return !fd_kind_is_device_or_stream(fd_kind);
    case __NR_write:
    case __NR_pwrite64:
    case __NR_writev:
    case __NR_pwritev:
    case __NR_pwritev2:
    case __NR_ftruncate:
    case __NR_fchmod:
    case __NR_fchown:
    case __NR_fsetxattr:
    case __NR_fremovexattr:
      /* Match Python's `int(event.fd or -1) < 0` short-circuit: fd 0
       * (stdin) and missing fd both drop. Real shell-redirected fd-1/2
       * writes still pass — fd_kind disambiguates them. */
      if (fd <= 0)
        return false;
      return fd_kind_is_mutating(fd_kind);
    /* Path-based mutations: unconditional keep. */
    case __NR_truncate:
    case __NR_chmod:
    case __NR_fchmodat:
    case __NR_chown:
    case __NR_fchownat:
    case __NR_lchown:
    case __NR_setxattr:
    case __NR_lsetxattr:
    case __NR_removexattr:
    case __NR_lremovexattr:
    case __NR_mkdir:
    case __NR_mkdirat:
    case __NR_mknod:
    case __NR_mknodat:
    case __NR_unlink:
    case __NR_unlinkat:
    case __NR_rmdir:
    case __NR_rename:
    case __NR_renameat:
    case __NR_renameat2:
    case __NR_link:
    case __NR_linkat:
    case __NR_symlink:
    case __NR_symlinkat:
      return true;
    default:
      return (primary_path && primary_path[0] != '\0') ||
             (secondary_path && secondary_path[0] != '\0');
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

/* Forward declaration; defined later in file alongside other timing helpers. */
static uint64_t monotonic_us(void);
/* Forward declaration; defined later in file alongside other path helpers. */
static void strip_deleted_suffix(char *path);

/* ---- pid identity cache + rule evaluation ----------------------------
 *
 * Mirror of process_filter.py:PidIdentityCache. Without caching, every
 * fs event would issue 2 syscalls in `read_pid_identity` (readlink
 * /proc/<pid>/exe + read /proc/<pid>/cmdline) — and a parent-chain
 * walk on top of that whenever a rule needs ancestors. Under tmux pane
 * workloads the helper sees ~10–40k events/sec, almost all of them
 * concentrated on a few PIDs, so caching collapses this to one read
 * per (pid, ttl-window) pair.
 *
 * Layout: open-addressed hash table keyed by pid. Slot is rebound on
 * collision (LRU-by-recency-of-touch is overkill at this scale). TTL
 * handles PID reuse — at the 4M Linux PID space, reuse inside 5s is
 * essentially impossible on a normal system, and the worst-case cost
 * of a stale hit is one or two events going through the wrong rule
 * for a freshly-recycled PID. */

#define PID_CACHE_SLOTS 4096
#define PID_CACHE_TTL_US 5000000ULL
#define MAX_CMDLINE_PARTS 32
#define MAX_ANCESTORS 32
#define IDENTITY_FIELD_LEN 256
#define CMDLINE_BUF_LEN 4096

struct pid_identity_entry {
  uint32_t pid;
  uint64_t deadline_us;
  bool resolved;
  char executable_basename[IDENTITY_FIELD_LEN];
  char executable_path[PATH_MAX];
  size_t cmdline_count;
  char cmdline_parts[MAX_CMDLINE_PARTS][IDENTITY_FIELD_LEN];
  /* Combined cmdline buffer for substring matching: parts joined by
   * '\0'. Faster than per-part scans when cmdline_contains entries
   * have to match across the whole cmdline (Python uses the same
   * NUL-joined form). */
  size_t cmdline_buf_len;
  char cmdline_buf[CMDLINE_BUF_LEN];
  bool ancestors_filled;
  size_t ancestor_count;
  char ancestor_basenames[MAX_ANCESTORS][IDENTITY_FIELD_LEN];
};

static struct pid_identity_entry pid_cache[PID_CACHE_SLOTS];
static pthread_mutex_t pid_cache_lock = PTHREAD_MUTEX_INITIALIZER;

static const char *path_basename(const char *path)
{
  const char *slash = strrchr(path, '/');
  return slash ? slash + 1 : path;
}

/* Read /proc/<pid>/exe (real path) into `path_out` and populate the
 * basename copy in `basename_out`. Returns 0 on success, negative
 * errno on failure (process exited / permission denied / etc.). */
static int read_pid_executable(uint32_t pid, char *path_out, size_t path_out_len, char *basename_out, size_t basename_out_len)
{
  char proc_path[64];
  ssize_t len;

  snprintf(proc_path, sizeof(proc_path), "/proc/%u/exe", pid);
  len = readlink(proc_path, path_out, path_out_len - 1);
  if (len < 0)
    return -errno;
  path_out[len] = '\0';
  /* Match Python's str(Path(...).resolve(strict=True)).name semantics:
   * strip a possible trailing " (deleted)" suffix that the kernel
   * appends when the executable was unlinked while still in use. */
  strip_deleted_suffix(path_out);
  snprintf(basename_out, basename_out_len, "%s", path_basename(path_out));
  return 0;
}

/* Read /proc/<pid>/cmdline (NUL-separated) into the entry's parts +
 * the joined buffer. Returns 0 on success, negative errno on failure. */
static int read_pid_cmdline(uint32_t pid, struct pid_identity_entry *entry)
{
  char proc_path[64];
  int fd;
  ssize_t total_read = 0;

  snprintf(proc_path, sizeof(proc_path), "/proc/%u/cmdline", pid);
  fd = open(proc_path, O_RDONLY);
  if (fd < 0)
    return -errno;
  for (;;) {
    ssize_t n = read(fd, entry->cmdline_buf + total_read,
                     sizeof(entry->cmdline_buf) - 1 - (size_t)total_read);
    if (n <= 0)
      break;
    total_read += n;
    if ((size_t)total_read >= sizeof(entry->cmdline_buf) - 1)
      break;
  }
  close(fd);
  entry->cmdline_buf_len = (size_t)total_read;
  entry->cmdline_buf[total_read] = '\0';

  /* Split into parts (NUL-separated). Trailing NUL on a clean
   * cmdline produces an empty final part — skip empties. */
  entry->cmdline_count = 0;
  size_t start = 0;
  for (size_t i = 0; i <= (size_t)total_read; i++) {
    if (i == (size_t)total_read || entry->cmdline_buf[i] == '\0') {
      if (i > start) {
        size_t part_len = i - start;
        if (part_len >= IDENTITY_FIELD_LEN)
          part_len = IDENTITY_FIELD_LEN - 1;
        memcpy(entry->cmdline_parts[entry->cmdline_count], entry->cmdline_buf + start, part_len);
        entry->cmdline_parts[entry->cmdline_count][part_len] = '\0';
        if (++entry->cmdline_count >= MAX_CMDLINE_PARTS)
          break;
      }
      start = i + 1;
    }
  }
  return 0;
}

static int read_ppid(uint32_t pid)
{
  char proc_path[64];
  FILE *fp;
  char line[256];
  int ppid = -1;

  snprintf(proc_path, sizeof(proc_path), "/proc/%u/status", pid);
  fp = fopen(proc_path, "r");
  if (!fp)
    return -1;
  while (fgets(line, sizeof(line), fp)) {
    if (strncmp(line, "PPid:", 5) == 0) {
      ppid = atoi(line + 5);
      break;
    }
  }
  fclose(fp);
  return ppid;
}

/* Walk up to 32 ancestors. Mirror of process_filter.py:read_ancestor_basenames.
 * Stops at PID 1, depth cap, or visited cycle. Permission errors on a
 * single ancestor are non-fatal — keep walking, an ancestor higher up
 * may still be readable. */
static void fill_pid_ancestors(uint32_t pid, struct pid_identity_entry *entry)
{
  uint32_t visited[64];
  size_t visited_count = 0;
  int current = read_ppid(pid);

  entry->ancestor_count = 0;
  entry->ancestors_filled = true;
  for (int depth = 0; depth < 32; depth++) {
    if (current <= 1)
      break;
    bool seen = false;
    for (size_t i = 0; i < visited_count; i++) {
      if (visited[i] == (uint32_t)current) {
        seen = true;
        break;
      }
    }
    if (seen || visited_count >= sizeof(visited) / sizeof(visited[0]))
      break;
    visited[visited_count++] = (uint32_t)current;
    char path[PATH_MAX] = {};
    char basename[IDENTITY_FIELD_LEN] = {};
    if (read_pid_executable((uint32_t)current, path, sizeof(path), basename, sizeof(basename)) == 0) {
      if (entry->ancestor_count < MAX_ANCESTORS) {
        snprintf(entry->ancestor_basenames[entry->ancestor_count], IDENTITY_FIELD_LEN, "%s", basename);
        entry->ancestor_count++;
      }
    }
    int parent = read_ppid((uint32_t)current);
    if (parent < 0)
      break;
    current = parent;
  }
}

/* Look up `pid` in the cache, populating it if missing or expired.
 * If `needs_ancestors` is true and the cache hit lacks ancestors, this
 * also extends the entry with the ancestor walk (mirrors Python's
 * "two-shot" cache contract).
 *
 * Returns a pointer to the cache slot. The lock is held only across
 * the slot mutation; the returned pointer remains stable as long as
 * no other thread evicts the same slot, which is true because the
 * helper's stdout reader (where this is called) is single-threaded. */
static struct pid_identity_entry *pid_identity_cache_get(uint32_t pid, bool needs_ancestors)
{
  if (pid == 0)
    return NULL;

  struct pid_identity_entry *slot = &pid_cache[pid % PID_CACHE_SLOTS];
  uint64_t now = monotonic_us();

  pthread_mutex_lock(&pid_cache_lock);
  bool fresh = (slot->pid == pid && slot->deadline_us > now);
  if (!fresh) {
    /* Rebind the slot. Clear identity then refill below. */
    memset(slot, 0, sizeof(*slot));
    slot->pid = pid;
    slot->deadline_us = now + PID_CACHE_TTL_US;
  }
  pthread_mutex_unlock(&pid_cache_lock);

  if (!fresh) {
    bool ok = (read_pid_executable(pid, slot->executable_path, sizeof(slot->executable_path),
                                   slot->executable_basename, sizeof(slot->executable_basename)) == 0);
    int cmd_rc = read_pid_cmdline(pid, slot);
    if (cmd_rc != 0)
      slot->cmdline_count = 0;
    /* Mirror Python: identity is "resolved" if we got either an exe
     * path or a non-empty cmdline. */
    slot->resolved = ok || slot->cmdline_count > 0;
    if (!ok && slot->cmdline_count > 0) {
      /* Fallback basename from cmdline[0], like Python does. */
      snprintf(slot->executable_basename, sizeof(slot->executable_basename),
               "%s", path_basename(slot->cmdline_parts[0]));
    }
  }

  if (needs_ancestors && !slot->ancestors_filled && slot->resolved)
    fill_pid_ancestors(pid, slot);

  return slot->resolved ? slot : NULL;
}

static bool ancestor_set_contains(const struct pid_identity_entry *entry, const char *needle)
{
  for (size_t i = 0; i < entry->ancestor_count; i++) {
    if (strcmp(entry->ancestor_basenames[i], needle) == 0)
      return true;
  }
  return false;
}

/* Mirror of ProcessIgnoreRule.matches. All declared rule fields must
 * match for the rule to fire; an empty field is "no constraint". */
static bool rule_matches(const struct ignore_rule *rule, const struct pid_identity_entry *identity)
{
  if (rule->executable_basename[0] != '\0' &&
      strcmp(rule->executable_basename, identity->executable_basename) != 0)
    return false;
  if (rule->executable_path_contains[0] != '\0') {
    if (identity->executable_path[0] == '\0' ||
        strstr(identity->executable_path, rule->executable_path_contains) == NULL)
      return false;
  }
  for (size_t i = 0; i < rule->cmdline_part_count; i++) {
    /* memmem is non-portable but available on glibc. cmdline_buf is
     * NUL-separated, so a needle that spans multiple argv entries
     * (the Python form joins on '\0') still matches if we use
     * memmem rather than strstr. */
    if (memmem(identity->cmdline_buf, identity->cmdline_buf_len,
               rule->cmdline_parts[i], strlen(rule->cmdline_parts[i])) == NULL)
      return false;
  }
  if (rule->ancestor_executable_basename[0] != '\0') {
    if (!ancestor_set_contains(identity, rule->ancestor_executable_basename))
      return false;
  }
  return true;
}

/* Match `pid` against the registered fs-eligible rules for the
 * sandbox owning `cgroup_id`. Returns true iff at least one rule
 * matches — at which point the event must be dropped. Caller must NOT
 * hold state->lock; we acquire it briefly to snapshot the rule set
 * (the array is rule-set-bounded so this stays cheap). */
static bool pid_matches_sandbox_rules(struct helper_state *state, uint64_t cgroup_id, uint32_t pid)
{
  struct ignore_rule rules_snapshot[IGNORE_RULES_PER_SANDBOX];
  size_t rule_count = 0;
  bool needs_ancestors = false;

  pthread_mutex_lock(&state->lock);
  struct registration *item = state->registrations;
  while (item) {
    if (item->cgroup_id == cgroup_id) {
      rule_count = item->ignore_rule_count;
      if (rule_count > IGNORE_RULES_PER_SANDBOX)
        rule_count = IGNORE_RULES_PER_SANDBOX;
      memcpy(rules_snapshot, item->ignore_rules, rule_count * sizeof(struct ignore_rule));
      needs_ancestors = item->any_rule_needs_ancestors;
      break;
    }
    item = item->next;
  }
  pthread_mutex_unlock(&state->lock);
  if (rule_count == 0)
    return false;

  struct pid_identity_entry *identity = pid_identity_cache_get(pid, needs_ancestors);
  if (!identity)
    return false;
  for (size_t i = 0; i < rule_count; i++) {
    if (rule_matches(&rules_snapshot[i], identity))
      return true;
  }
  return false;
}

/* ---- duplicate-event coalescing ------------------------------------
 *
 * Build / tar / apt-get workloads emit dense bursts of events on the
 * same path: a single tar extract is open(O_CREAT) → write* → close →
 * open(O_TRUNC) → write* across thousands of files, with each file
 * generating tens of write events that all collapse to the SAME
 * `live_dirty_entries` row in Python (keyed by (device, inode) or
 * path). Pre-port, every duplicate paid for JSON encode + IPC + JSON
 * decode + sandbox_lock acquisition + dirty-entry update.
 *
 * Drop them at the helper. Key: (cgroup_id, syscall_nr, primary_path,
 * secondary_path), 64-bit FNV-1a hash. Open-addressed table with a
 * 50ms time window — long enough to absorb a burst, short enough that
 * sequenced operations on the same path (write → unlink → recreate)
 * still get a fresh dirty signal in <0.1s. Hash collisions could
 * cause false-positive coalesce; at 64 bits that's ~1-in-2^32 per
 * pair, which is negligible at our event scale.
 *
 * What this loses: `event_count` in the Python dirty-entry no longer
 * reflects the true number of writes — it counts emissions, not raw
 * events. This is diagnostic only; nothing in the scheduler hot path
 * depends on it. */

#define COALESCE_SLOTS 16384
#define COALESCE_WINDOW_US 50000ULL

struct coalesce_slot {
  uint64_t hash;
  uint64_t deadline_us;
};

static struct coalesce_slot coalesce_table[COALESCE_SLOTS];
static pthread_mutex_t coalesce_lock = PTHREAD_MUTEX_INITIALIZER;

static uint64_t fnv1a64_update(const void *data, size_t len, uint64_t seed)
{
  const uint8_t *p = data;
  uint64_t h = seed;
  for (size_t i = 0; i < len; i++) {
    h ^= p[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}

/* Returns true if this event is a recent duplicate and should be
 * dropped. Caller emits otherwise; the slot is rebound either way so
 * the next duplicate within the window collapses to this emission. */
static bool coalesce_event(
  uint64_t cgroup_id,
  uint32_t syscall_nr,
  const char *primary_path,
  const char *secondary_path
)
{
  uint64_t h = 0xcbf29ce484222325ULL;
  h = fnv1a64_update(&cgroup_id, sizeof(cgroup_id), h);
  h = fnv1a64_update(&syscall_nr, sizeof(syscall_nr), h);
  if (primary_path && primary_path[0] != '\0')
    h = fnv1a64_update(primary_path, strlen(primary_path), h);
  /* Separator byte so "ab|c" and "a|bc" can't hash-collide via
   * concatenation; FNV mixing already separates them, but the explicit
   * sentinel is cheap insurance. */
  h = fnv1a64_update("\xff", 1, h);
  if (secondary_path && secondary_path[0] != '\0')
    h = fnv1a64_update(secondary_path, strlen(secondary_path), h);

  uint64_t now = monotonic_us();
  size_t slot = (size_t)(h % COALESCE_SLOTS);
  bool drop;

  pthread_mutex_lock(&coalesce_lock);
  struct coalesce_slot *entry = &coalesce_table[slot];
  drop = (entry->hash == h && entry->deadline_us > now);
  /* Rebind regardless: a hash mismatch means a different event landed
   * on the same slot (different paths or syscalls), and we want to
   * coalesce subsequent duplicates of THIS event, not the prior one. */
  entry->hash = h;
  entry->deadline_us = now + COALESCE_WINDOW_US;
  pthread_mutex_unlock(&coalesce_lock);
  return drop;
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

static struct registration *find_registration_by_id_locked(
  struct helper_state *state, const char *sandbox_id
)
{
  struct registration *item = state->registrations;
  while (item) {
    if (strcmp(item->sandbox_id, sandbox_id) == 0)
      return item;
    item = item->next;
  }
  return NULL;
}

/* Returns 1 if `path` matches any of the per-sandbox ignored prefixes
 * for the registration whose cgroup is `cgroup_id`, else 0. Caller must
 * NOT hold state->lock. */
static int path_matches_ignored_prefix(
  struct helper_state *state, uint64_t cgroup_id, const char *path
)
{
  struct registration *item;
  size_t path_len;
  int matched = 0;

  if (!path || path[0] == '\0')
    return 0;
  path_len = strlen(path);
  pthread_mutex_lock(&state->lock);
  item = state->registrations;
  while (item) {
    if (item->cgroup_id == cgroup_id) {
      for (size_t i = 0; i < item->ignored_path_prefix_count; i++) {
        size_t plen = item->ignored_path_prefix_lens[i];
        if (plen > 0 && plen <= path_len &&
            memcmp(item->ignored_path_prefixes[i], path, plen) == 0) {
          matched = 1;
          break;
        }
      }
      break;
    }
    item = item->next;
  }
  pthread_mutex_unlock(&state->lock);
  return matched;
}

static int set_ignored_path_prefixes_for_sandbox(
  struct helper_state *state,
  const char *sandbox_id,
  /* prefixes is a "prefix1\tprefix2\t...\tprefixN" string, tab-separated;
   * the daemon ships them this way to keep the JSON wire format simple
   * (no need to escape '\t' inside paths since paths can't contain it
   * naturally). */
  const char *prefixes
)
{
  struct registration *item;
  const char *cursor;
  const char *next;
  size_t count = 0;

  pthread_mutex_lock(&state->lock);
  item = find_registration_by_id_locked(state, sandbox_id);
  if (!item) {
    pthread_mutex_unlock(&state->lock);
    return -ENOENT;
  }
  cursor = prefixes;
  while (cursor && *cursor && count < IGNORED_PATH_PREFIXES_PER_SANDBOX) {
    size_t len;
    next = strchr(cursor, '\t');
    len = next ? (size_t)(next - cursor) : strlen(cursor);
    if (len == 0) {
      if (!next)
        break;
      cursor = next + 1;
      continue;
    }
    if (len >= IGNORED_PATH_PREFIX_MAX_LEN)
      len = IGNORED_PATH_PREFIX_MAX_LEN - 1;
    memcpy(item->ignored_path_prefixes[count], cursor, len);
    item->ignored_path_prefixes[count][len] = '\0';
    item->ignored_path_prefix_lens[count] = len;
    count++;
    if (!next)
      break;
    cursor = next + 1;
  }
  item->ignored_path_prefix_count = count;
  pthread_mutex_unlock(&state->lock);
  return 0;
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

    if (strcmp(op, "set_ignored_path_prefixes") == 0) {
      char prefixes[IGNORED_PATH_PREFIX_MAX_LEN * IGNORED_PATH_PREFIXES_PER_SANDBOX];
      if (extract_string_field(line, "prefixes", prefixes, sizeof(prefixes)) != 0) {
        /* Empty value: clear the filter. */
        prefixes[0] = '\0';
      }
      if (set_ignored_path_prefixes_for_sandbox(state, sandbox_id, prefixes) != 0)
        fprintf(stderr, "failed to set path prefixes for sandbox %s\n", sandbox_id);
      continue;
    }

    if (strcmp(op, "clear_ignore_process_rules") == 0) {
      pthread_mutex_lock(&state->lock);
      struct registration *item = find_registration_by_id_locked(state, sandbox_id);
      if (item) {
        item->ignore_rule_count = 0;
        item->any_rule_needs_ancestors = false;
        memset(item->ignore_rules, 0, sizeof(item->ignore_rules));
      }
      pthread_mutex_unlock(&state->lock);
      continue;
    }

    if (strcmp(op, "add_ignore_process_rule") == 0) {
      char executable_basename[RULE_FIELD_MAX_LEN];
      char executable_path_contains[RULE_FIELD_MAX_LEN];
      char cmdline_contains[RULE_FIELD_MAX_LEN * CMDLINE_CONTAINS_PARTS_MAX];
      char ancestor_basename[RULE_FIELD_MAX_LEN];
      if (extract_string_field(line, "executable_basename", executable_basename, sizeof(executable_basename)) != 0)
        executable_basename[0] = '\0';
      if (extract_string_field(line, "executable_path_contains", executable_path_contains, sizeof(executable_path_contains)) != 0)
        executable_path_contains[0] = '\0';
      if (extract_string_field(line, "cmdline_contains", cmdline_contains, sizeof(cmdline_contains)) != 0)
        cmdline_contains[0] = '\0';
      if (extract_string_field(line, "ancestor_executable_basename", ancestor_basename, sizeof(ancestor_basename)) != 0)
        ancestor_basename[0] = '\0';

      pthread_mutex_lock(&state->lock);
      struct registration *item = find_registration_by_id_locked(state, sandbox_id);
      if (item && item->ignore_rule_count < IGNORE_RULES_PER_SANDBOX) {
        struct ignore_rule *rule = &item->ignore_rules[item->ignore_rule_count];
        snprintf(rule->executable_basename, RULE_FIELD_MAX_LEN, "%s", executable_basename);
        snprintf(rule->executable_path_contains, RULE_FIELD_MAX_LEN, "%s", executable_path_contains);
        snprintf(rule->ancestor_executable_basename, RULE_FIELD_MAX_LEN, "%s", ancestor_basename);
        /* Split cmdline_contains on '|'. The wire encoding uses '|'
         * because tab gets passed-through cleanly by the JSON line
         * encoder, but '|' is more visually obvious in the helper
         * stderr if logging is ever added. cmdline_contains entries
         * for the terminus rule set are short literal substrings
         * ("server-1", "session-name") so '|' is safe. */
        rule->cmdline_part_count = 0;
        const char *cursor = cmdline_contains;
        while (cursor && *cursor && rule->cmdline_part_count < CMDLINE_CONTAINS_PARTS_MAX) {
          const char *next = strchr(cursor, '|');
          size_t len = next ? (size_t)(next - cursor) : strlen(cursor);
          if (len > 0) {
            if (len >= RULE_FIELD_MAX_LEN)
              len = RULE_FIELD_MAX_LEN - 1;
            memcpy(rule->cmdline_parts[rule->cmdline_part_count], cursor, len);
            rule->cmdline_parts[rule->cmdline_part_count][len] = '\0';
            rule->cmdline_part_count++;
          }
          if (!next)
            break;
          cursor = next + 1;
        }
        item->ignore_rule_count++;
        if (rule->ancestor_executable_basename[0] != '\0')
          item->any_rule_needs_ancestors = true;
      }
      pthread_mutex_unlock(&state->lock);
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

  /* Per-sandbox path-prefix filter: drop events whose target path is
   * under a host-side helper directory the daemon registered up-front
   * (CRIU's per-sandbox dump.log, runc's state metadata, etc.). These
   * writes get attributed to the sandbox cgroup because the calling
   * task transiently joined it (CRIU parasite injected into container
   * processes, runc init helpers); they are not real sandbox state
   * changes. Filtering here in the C helper saves the JSON
   * serialization + Python parse + per-event sandbox_lock acquisition
   * downstream — at the cost of one strlen + one memcmp per registered
   * prefix, both well-bounded. */
  if (
    (primary_path[0] != '\0' &&
     path_matches_ignored_prefix(state, event->cgroup_id, primary_path)) ||
    (secondary_path[0] != '\0' &&
     path_matches_ignored_prefix(state, event->cgroup_id, secondary_path))
  )
    return 0;

  /* Per-sandbox process-ignore-rule filter. Drops events whose pid
   * identity matches a registered fs-eligible (scope=all) rule, so
   * tmux pane bash / sleep / etc. never produce JSON-encoded events
   * to begin with. We do this BEFORE fd_kind resolution so matched
   * events skip the stat() syscalls too. */
  if (event->pid > 0 && pid_matches_sandbox_rules(state, event->cgroup_id, event->pid))
    return 0;

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

  /* Drop non-countable events at the helper instead of downstream in
   * Python. Identical decision tree to server.py:_is_countable_fs_event;
   * the duplicate Python check stays in place as a belt-and-suspenders
   * for any path that could regress (e.g. someone bypassing the helper
   * with a unit-test stub). */
  if (!is_countable_fs_event(
        event->syscall_nr,
        event->flags,
        event->fd,
        fd_kind,
        primary_path,
        secondary_path
      ))
    return 0;

  /* Final filter: collapse duplicate (cgroup, syscall, paths) events
   * within a short time window. Burst-y workloads emit thousands of
   * writes against the same path; Python folds them into one
   * `live_dirty_entries` row anyway, so emitting more than one per
   * path-window is wasted IPC + decode. Skipped for non-mutation
   * targets (no path) since those would all hash to the same slot. */
  if ((primary_path[0] != '\0' || secondary_path[0] != '\0') &&
      coalesce_event(event->cgroup_id, event->syscall_nr, primary_path, secondary_path))
    return 0;

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
