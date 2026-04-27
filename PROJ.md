# MiniSTS 项目说明

## 目标

把原始 MiniSTS 扩展成一个轻量级《Slay the Spire》战斗模拟与 RL 训练环境，用于研究固定卡组、固定战斗场景下的出牌策略，以及后续的选牌价值评估。

## 当前状态

- `card.py`：已实现大部分 Ironclad/red 卡牌、状态牌和若干实验牌；原版牌尽量参考本机 STS jar 的结构与中文名。
- `agent.py`：已实现第一幕主要敌人、精英和 Boss，包括 `TheGuardian`、`Hexaghost`、`SlimeBoss`、`GremlinNob`、`Lagavulin`、`Sentry` 等；数值和关键机制当前按 A20 对齐。
- `encounters.py`：支持第一幕 encounter 名称与组合，例如 `exordium:Gremlin Gang`、`exordium:3 Sentries`。
- `battle.py` / `status_effecs.py`：已补回合事件、抽牌/弃牌/消耗事件、受击事件、Curl Up、Angry、Sharp Hide、Artifact 等战斗机制。
- `rl/`：已有固定场景 RL 环境、DQN/Double DQN、状态编码、训练与评估入口。
- `rl_web/manual_play/` + `rl/web_play.py`：已有本地 Web 手动打牌界面，便于按 RL config 交互式验证战斗。

MiniSTS 仍是轻量模拟，不是完整 STS 引擎复刻；动画、金币奖励、房间流程、部分稀有边界仍未建模。

## RL 约定

- 动作空间固定为 11 维：
  - `0`：结束回合
  - `1..10`：普通状态下打出手牌槽位 `0..9`
  - pending hand choice 状态下选择手牌槽位 `0..9`
- 当前 pending hand choice 已覆盖：
  `True Grit+`、`Burning Pact`、`Warcry`、`Armaments`、`Dual Wield`。
- `StateEncoder` 当前按固定卡组生成 card vocabulary；怪物意图只编码当前可见效果的简化数值，例如攻击、格挡、debuff、塞牌、split/sleep/stun/escape/defensive mode。
- 怪物意图不应暴露内部调度信息，例如 `Set next move to ...`、清状态、模式切换实现细节。
- reward 当前保持简洁，只额外加入敌人掉血奖励：
  `damage_reward_scale * damage_dealt / enemy_max_health`。

## 常用命令

训练普通 DQN：

```bash
.venv/bin/python -m rl.train_dqn --config rl_configs/scenario5_big_jaw_worm.json --no-double-dqn
```

训练 Double DQN：

```bash
.venv/bin/python -m rl.train_dqn --config rl_configs/scenario5_big_jaw_worm.json --double-dqn --save-path rl_runs/ddqn_scenario5_guardian.pt
```

评估 checkpoint：

```bash
.venv/bin/python -m rl.evaluate_dqn --config rl_configs/scenario5_big_jaw_worm.json --checkpoint rl_runs/ddqn_scenario5_guardian.pt
```

打印首局 trace：

```bash
.venv/bin/python -m rl.evaluate_dqn --config rl_configs/scenario5_big_jaw_worm.json --episodes 1 --trace
```

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

列出 main.py 支持的怪物和 encounter：

```bash
.venv/bin/python main.py --list-enemies
```

按 encounter 运行传统手动战斗：

```bash
.venv/bin/python main.py --enemy "exordium:3 Sentries"
```

## 当前默认场景

`rl_configs/scenario5_big_jaw_worm.json` 当前用于 A20 Guardian 固定卡组实验：

- enemy：`TheGuardian`
- ascension：`20`
- reward：只包含敌人掉血奖励
- training：默认 `episodes=5000`
- `training.double_dqn=false`，命令行可用 `--double-dqn` 覆盖

命令行参数优先级高于 JSON 配置。例如临时跑 1500 局：

```bash
.venv/bin/python -m rl.train_dqn --config rl_configs/scenario5_big_jaw_worm.json --episodes 1500
```

## 原版参考

本机 STS jar：

```bash
/Users/charlie/Library/Application Support/Steam/steamapps/common/SlayTheSpire/SlayTheSpire.app/Contents/Resources/desktop-1.0.jar
```

提取卡牌摘要：

```bash
.venv/bin/python tools/sts_reference_extract.py --colors red --out local_refs/sts_cards_red_summary.json
```

提取第一幕 encounter 摘要：

```bash
.venv/bin/python tools/sts_encounter_extract.py --dungeons exordium --out local_refs/sts_encounters_summary.json
```

`local_refs/` 只作为本地实现参考，不提交原始游戏资源或反编译源码。

## 注意事项

- 旧 checkpoint 可能因 encoder/card vocabulary/动作语义变更而不兼容。
- `Exhume`、`Headbutt` 这类从消耗堆/弃牌堆选择的 pending 编码仍待补。
- 多敌人训练已能实例化 encounter，但 encoder 目前仍偏向固定场景，后续若泛化到全局战斗需要更通用的 enemy 表示。
