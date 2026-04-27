# MiniSTS

MiniSTS 是一个轻量级《Slay the Spire》战斗模拟与强化学习实验环境，当前重点是固定卡组、固定战斗场景下的 Ironclad 出牌策略研究。

这个仓库最初基于原始 MiniStS 项目改造而来，但当前版本已经大幅扩展了卡牌、敌人、战斗机制、RL 环境和手动验证工具。它仍然不是完整 STS 引擎复刻：不包含地图、遗物系统、金币奖励、动画和完整房间流程。

## 当前能力

- Ironclad/red 卡牌：实现了大量原版红卡、状态牌和若干实验牌。
- 第一幕敌人：支持常见小怪、精英和 Boss，包括 `JawWorm`、`Lagavulin`、`GremlinNob`、`Sentry`、`TheGuardian`、`Hexaghost`、`SlimeBoss` 等。
- 第一幕 encounter：支持单怪和组合战斗，例如 `exordium:Gremlin Gang`、`exordium:3 Sentries`。
- 战斗机制：支持抽牌、弃牌、消耗、回合事件、受击事件、Artifact、Angry、Curl Up、Sharp Hide、Guardian defensive mode 等关键机制。
- RL 环境：支持固定动作空间、状态编码、DQN/Double DQN、PPO、MCTS baseline、policy+value+PUCT。
- 手动验证：支持 CLI 手动打牌和本地 Web 手动打牌界面。

## 安装

建议使用虚拟环境：

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

如果你的 Python 环境里已经装好依赖，也可以直接使用现有解释器运行。当前依赖主要包括 `numpy`、`torch`、`pandas`、`matplotlib`、`tqdm`。

## 快速开始

列出支持的怪物和第一幕 encounter：

```bash
.venv/bin/python main.py --list-enemies
```

运行传统 CLI 手动战斗：

```bash
.venv/bin/python main.py --enemy JawWorm
```

运行第一幕组合战斗：

```bash
.venv/bin/python main.py --enemy "exordium:3 Sentries"
```

## RL 场景配置

主要实验配置在：

```text
rl_configs/scenario5_big_jaw_worm.json
```

配置里可以指定：

- `env.enemy`：敌人或 encounter 名称，例如 `Lagavulin`、`TheGuardian`、`exordium:Gremlin Gang`
- `env.ascension`：进阶等级，当前常用 `20`
- `deck`：固定卡组
- `reward`：敌人掉血、自身损血、胜负终局信号
- `training` / `ppo` / `mcts` / `az`：各算法参数

命令行参数优先级高于 JSON 配置。

## 手动验证 RL 场景

CLI 手动打牌：

```bash
.venv/bin/python -m rl.manual_play --config rl_configs/scenario5_big_jaw_worm.json
```

Web 手动打牌：

```bash
.venv/bin/python -m rl.web_play --config rl_configs/scenario5_big_jaw_worm.json --host 127.0.0.1 --port 8765
```

然后打开：

```text
http://127.0.0.1:8765
```

## 训练与评估

训练 DQN：

```bash
.venv/bin/python -m rl.train_dqn --config rl_configs/scenario5_big_jaw_worm.json --no-double-dqn
```

训练 Double DQN：

```bash
.venv/bin/python -m rl.train_dqn --config rl_configs/scenario5_big_jaw_worm.json --double-dqn --save-path rl_runs/ddqn_scenario5_guardian.pt
```

训练 PPO：

```bash
.venv/bin/python -m rl.train_ppo --config rl_configs/scenario5_big_jaw_worm.json
```

运行 MCTS baseline：

```bash
.venv/bin/python -m rl.evaluate_mcts --config rl_configs/scenario5_big_jaw_worm.json
```

训练 policy+value+PUCT：

```bash
.venv/bin/python -m rl.train_az --config rl_configs/scenario5_big_jaw_worm.json
```

评估 DQN checkpoint：

```bash
.venv/bin/python -m rl.evaluate_dqn --config rl_configs/scenario5_big_jaw_worm.json --checkpoint rl_runs/ddqn_scenario5_guardian.pt
```

评估 PPO checkpoint：

```bash
.venv/bin/python -m rl.evaluate_ppo --config rl_configs/scenario5_big_jaw_worm.json --checkpoint rl_runs/ppo_scenario5_guardian.pt
```

评估 policy+value+PUCT checkpoint：

```bash
.venv/bin/python -m rl.evaluate_az --config rl_configs/scenario5_big_jaw_worm.json --checkpoint rl_runs/az_scenario5_guardian.pt --mode puct
```

打印首局 trace：

```bash
.venv/bin/python -m rl.evaluate_dqn --config rl_configs/scenario5_big_jaw_worm.json --episodes 1 --trace
```

## RL 约定

动作空间固定为 11 维：

- `0`：结束回合
- `1..10`：普通状态下打出手牌槽位 `0..9`
- pending hand choice 状态下，`1..10` 表示选择手牌槽位 `0..9`

当前 pending hand choice 覆盖：

- `True Grit+`
- `Burning Pact`
- `Warcry`
- `Armaments`
- `Dual Wield`

`StateEncoder` 当前主要服务固定战斗实验。多敌人 encounter 可以实例化，但如果要泛化到任意战斗，还需要更通用的 enemy 表示和更完整的 target 动作空间。

## 项目结构

```text
card.py                  卡牌定义
agent.py                 敌人与怪物行为
encounters.py            第一幕 encounter
battle.py                战斗流程
status_effecs.py         状态效果和触发机制
main.py                  传统手动战斗入口
rl/                      RL 环境、encoder、算法、训练和评估入口
rl_configs/              RL 实验配置
rl_web/manual_play/      Web 手动打牌前端
tools/                   本地 STS jar 摘要提取工具
```

## 原版参考

部分卡牌、敌人和 encounter 行为参考本机 STS jar 的反编译/摘要数据实现。仓库不提交原始游戏资源或反编译源码。

本地提取红卡摘要：

```bash
.venv/bin/python tools/sts_reference_extract.py --colors red --out local_refs/sts_cards_red_summary.json
```

本地提取第一幕 encounter 摘要：

```bash
.venv/bin/python tools/sts_encounter_extract.py --dungeons exordium --out local_refs/sts_encounters_summary.json
```

## 注意事项

- 旧 checkpoint 可能因 encoder、card vocabulary 或动作语义变更而不兼容。
- `Exhume`、`Headbutt` 等从消耗堆/弃牌堆选择对象的 pending 编码仍待完善。
- `MCTS` 当前是在线规划 baseline，`episodes` 表示评估局数，不会训练或跨局改进。
- `policy+value+PUCT` 是实验性实现，当前更适合用于比较搜索增强是否有价值，不应默认认为会优于 PPO/DQN。

## Attribution

本项目最初基于 MiniStS 改造。原始 MiniStS 由 Bahar Bateni 和 Jim Whitehead 开发，用作动态规则系统和游戏智能体研究的测试环境。当前仓库在此基础上扩展为面向 STS 固定战斗与 RL 训练的实验项目。
