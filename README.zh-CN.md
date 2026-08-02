# Crab

[English](README.md) | 简体中文

Crab 为 AI agent 沙箱提供可恢复的存档点。它观察 agent 的执行阶段，判断
何时需要保存进程状态、文件系统状态或两者，并协调宿主机上的 checkpoint
和 restore。

Crab v0目前是技术预览版。v0 使用：

- `runc` 运行沙箱；
- CRIU 保存和恢复进程状态；
- ZFS 快照和回滚沙箱根文件系统；
- eBPF host inspector 检测影响恢复的状态变化。

Crab 负责决定“何时保存”和“保存什么粒度”，但 v0 只提供经过完整验证的
runc + CRIU + ZFS 后端，尚未提供 Docker/overlay fallback。

## 快速体验

支持 Ubuntu 24.04/26.04 x86-64，需要 root 权限：

```bash
git clone https://github.com/open-agent-infra/crab.git
cd crab
sudo ./scripts/install-ubuntu.sh
sudo ./scripts/smoke-rollback.sh
```

安装脚本会创建一个名为 `crab` 的专用稀疏文件 ZFS pool。它不会自动选择
机器上的其他 pool，也不会重新分区磁盘。

smoke test 不需要 API key，会真实执行以下流程：

1. 启动 Ubuntu 沙箱；
2. 创建文件和后台进程；
3. 创建包含进程和文件系统状态的 checkpoint；
4. 修改文件并终止进程；
5. restore 并验证文件、PID 和进程都已恢复。

详细选项见[安装说明](docs/installation.md)和[入门教程](docs/getting-started.md)。

## 手动 checkpoint 和 restore

```bash
sudo crab daemon start --config /etc/crab/config.yaml
SBX=$(sudo crab sandbox run --detach ubuntu:22.04)

sudo crab sandbox exec "$SBX" -- sh -lc 'echo before > /root/state.txt'
CKPT=$(sudo crab checkpoint create "$SBX")
sudo crab checkpoint ls "$SBX"

sudo crab sandbox exec "$SBX" -- sh -lc 'echo after > /root/state.txt'
sudo crab restore "$SBX" "$CKPT"
sudo crab sandbox exec "$SBX" -- cat /root/state.txt
# before

sudo crab sandbox rm "$SBX"
sudo crab daemon stop
```

## Python SDK

SDK 和 CLI 连接同一个常驻 daemon：

```python
from crab import Engine, Sandbox

with Engine.connect() as engine:
    sandbox = Sandbox(image="ubuntu:22.04", engine=engine)
    try:
        sandbox.commands.run("echo before > /root/state.txt")
        checkpoint_id = sandbox.checkpoint()
        sandbox.commands.run("echo after > /root/state.txt")
        sandbox.restore(checkpoint_id)
    finally:
        sandbox.kill()
```

Crab 还提供 `Agent` 接入接口，以及 Claude Code 和 iFlow profile。需要注意：
默认 smoke-test 配置关闭了沙箱网络和 LLM interceptor；在沙箱内调用 LLM 的
agent 必须使用 agent 专用配置。参见：

- [SDK](docs/sdk.md)
- [接入自己的 agent](docs/byo-agent.md)
- [无 API key 的 iFlow trace replay](docs/sdk-iflow-replay.md)
- [配置说明](docs/configuration-reference.md)

## checkpoint 的边界

v0 的完整 checkpoint 包含进程状态和 ZFS 管理的沙箱根文件系统。

以下内容不会被回滚：

- `Sandbox(work_dir="./repo")` 或 `--work-dir` 挂载到 `/work` 的宿主机目录；
- GitHub push、云 API、支付、外部数据库等沙箱外部副作用；
- daemon 未管理的其他容器、进程和文件系统。

如果希望项目文件能够随 checkpoint 回滚，应把仓库放在沙箱根文件系统内，
而不是宿主机 bind mount 中。

## v0 范围

当前支持：

- Ubuntu 24.04/26.04 x86-64；
- 单机、root-owned、单用户 daemon；
- 通过 CLI 或 SDK 创建、查看和恢复 checkpoint；
- 基于 LLM request boundary 的语义感知自动 checkpoint；
- runc + CRIU + ZFS 后端。

暂不支持：

- macOS、Windows、rootless 或多用户模式；
- 缺少 CRIU/ZFS 时的自动 fallback；
- daemon 重启后自动重新接管已有沙箱；
- 回滚宿主机 bind mount 或外部副作用；
- 稳定的第三方 checkpoint substrate 插件 API。

## 文档、贡献和论文

- [文档索引](docs/README.md)
- [贡献指南](CONTRIBUTING.md)
- [论文：Crab: A Semantics-Aware Checkpoint/Restore Runtime for Agent Sandboxes](https://arxiv.org/abs/2604.28138)

## License

Crab 使用 [MIT License](LICENSE)。
