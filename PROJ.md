# MiniSTS 项目说明

## 目标

本项目基于原始 MiniSTS，目标是把它扩展成可用于《Slay the Spire》选牌决策研究的轻量模拟与训练环境。

当前主要方向：

1. 补全和扩展原项目未完成的卡牌、敌人、状态效果与战斗逻辑。
2. 构造 RL 训练流程：先在固定卡组 + 固定敌人的战斗场景中训练策略，再用训练好的策略评估增删某张牌对胜率、血量、回合数等指标的影响。
3. 未来考虑增加前端或游戏内置 mod 接口，最终目标是为游戏提供选牌指导。

## 当前完成情况

- 原 MiniSTS 核心仍保留：卡牌、行动、目标、状态效果、战斗循环和 GGPA/Backtrack/LLM bot 框架。
- 已新增/整理部分 Ironclad 卡牌定义，主要在 `card.py`：
  - 原有基础牌与常见牌：`Strike`、`Defend`、`Bash`、`Pommel Strike`、`Shrug It Off`、`Bloodletting`、`Cleave`、`Impervious` 等。
  - 实验牌：`Stimulate`、`Batter`、`Tolerate`、`Bomb`、`Suffer`。
  - 固定测试卡组：`CardRepo.get_scenario_0()` 到 `get_scenario_5()`。
- 已新增/整理部分敌人定义，主要在 `agent.py`：
  - 原有小怪：`AcidSlimeSmall`、`SpikeSlimeSmall`、`JawWorm`。
  - 已对照本机 STS jar 修正 `JawWorm` 的部分数值与行动选择逻辑。
  - 已新增/补齐第一幕 Exordium 敌人切片：
    `Cultist`、`AcidSlimeSmall/Medium/Large`、`SpikeSlimeSmall/Medium/Large`、`LouseNormal`、`LouseDefensive`、
    `FungiBeast`、`SlaverBlue`、`SlaverRed`、`Looter`、`GremlinFat`、`GremlinWarrior`、`GremlinThief`、
    `GremlinTsundere`、`GremlinWizard`、`GremlinNob`、`Lagavulin`、`Sentry`、`SlimeBoss`、`Hexaghost`、`TheGuardian`。
  - 扩展/实验敌人：`BigJawWorm`、`Goblin`、`HobGoblin`、`Leech`。
- 已新增状态效果，主要在 `status_effecs.py`：
  - 原有：`VULNERABLE`、`WEAK`、`STRENGTH`。
  - 扩展：`DEXTERITY`、`VIGOR`、`TOLERANCE`、`BOMB`、`NO_DRAW`、`BARRICADE`、`METALLICIZE`、`LOSE_STRENGTH`、`BERSERK`、`DEMON_FORM`、`BRUTALITY`、`RAGE`、`FEEL_NO_PAIN`、`DARK_EMBRACE`、`EVOLVE`、`FIRE_BREATHING`、`JUGGERNAUT`、`FLAME_BARRIER`、`RITUAL`、`FRAIL`、`ENTANGLED`、`CURL_UP`、`ANGER`、`SHARP_HIDE`、`ARTIFACT`。
- 已补一批 Ironclad/red 机制样例卡：
  - `Battle Trance`、`Flex`、`Inflame`、`Metallicize`、`Barricade`、`Seeing Red`、`True Grit`、`Wild Strike`、`Power Through`。
  - `Rage`、`Feel No Pain`、`Dark Embrace`、`Evolve`、`Fire Breathing`、`Juggernaut`、`Flame Barrier`。
  - `Bludgeon`、`Clothesline`、`Thunderclap`、`Twin Strike`、`Uppercut`、`Iron Wave`、`Pummel`、`Hemokinesis`、`Offering`、`Disarm`、`Shockwave`、`Berserk`、`Demon Form`、`Brutality`。
  - 已继续补齐剩余 Ironclad 牌：`Armaments`、`Blood for Blood`、`Body Slam`、`Burning Pact`、`Carnage`、`Clash`、`Combust`、`Corruption`、`Double Tap`、`Dropkick`、`Dual Wield`、`Entrench`、`Exhume`、`Feed`、`Fiend Fire`、`Ghostly Armor`、`Havoc`、`Headbutt`、`Heavy Blade`、`Immolate`、`Infernal Blade`、`Intimidate`、`Limit Break`、`Perfected Strike`、`Rampage`、`Reaper`、`Reckless Charge`、`Rupture`、`Second Wind`、`Sentinel`、`Sever Soul`、`Spot Weakness`、`Sword Boomerang`、`Warcry`、`Whirlwind`，并保留 `Strike_R`、`Defend_R` 别名。
  - `Bloodletting` 已按原版修正为失去 3 HP 后获得 2/3 能量。
  - `card.py` 中已给每张 `CardGen` 卡牌补中文短注释；原版牌参考本机 jar 的 `localization/zhs/cards.json`，实验牌单独标注。
  - 已新增 `Wound`、`Dazed`、`Burn` 等不可打出状态牌基础定义。
- 已新增回合事件与牌堆机制：
  - `BattleState.turn_start_event`、`turn_end_event`。
  - `card_play_event`、`card_exhaust_event`、`card_draw_event`、`block_gain_event`、`attacked_event`。
  - 随机/全部卡牌目标、弃牌、临时生成牌、升级整手牌、HP loss。
  - 已补动态费用、X 费用、虚无、耗尽触发、HP loss 触发、按格挡/力量/Strike 数计算伤害、从弃牌/耗尽取牌、复制手牌、消耗手牌、打出抽牌堆顶牌等机制。
  - 已新增升级切换卡牌目标机制：`True Grit` 未升级时随机消耗手牌，`True Grit+` 改为可选择消耗目标。
  - 已修正普通卡牌战斗内过度升级问题：除 `SearingBlow` 这类特殊牌外，普通牌最多升级一次，避免出现 `True Grit+++` 这类不符合原版的状态。
  - 新增状态：`COMBUST`、`CORRUPTION`、`DOUBLE_TAP`、`RUPTURE`。
- 已有 DQN 训练原型，当前默认配置是固定 6 张牌对战 `BigJawWorm`。
- 已有训练产物示例在 `rl_runs/`：
  - `dqn_scenario5_jawworm.pt`
  - `smoke.pt`
  - `smoke_big.pt`

## RL 代码参考

- `rl/env.py`
  - `MiniSTSEnv`：RL 环境封装。
  - 当前动作空间：`0` 表示结束回合，`1..10` 在普通状态表示打出手牌槽位 `0..9`；进入手牌选择 pending 状态时表示选择手牌槽位 `0..9`。
  - 当前敌人参数：`enemy_name` 优先使用 `agent.py` 中 `MONSTER_FACTORIES` 的怪物类名，例如 `SlaverBlue`、`GremlinNob`、`SlimeBoss`、`BigJawWorm`；旧的小写别名仍保留兼容，也支持 `exordium:<encounter name>` 格式的第一幕 encounter。
  - 当前支持从 JSON 配置传入固定牌组；未传配置时使用 `rl/experiment_config.py` 中的 `DEFAULT_DECK`。
- `rl/experiment_config.py`
  - JSON 配置解析与牌组构造。
  - 可以在配置中的 `deck` 字段直接调整牌组，例如 `{"name": "Pommel Strike", "upgrades": 1}` 或 `{"name": "Strike", "count": 5}`。
- `rl_configs/scenario5_big_jaw_worm.json`
  - 当前默认 RL 实验配置，可复制后修改敌人、训练参数、评估参数和牌组。
- `rl/encoder.py`
  - `StateEncoder`：固定场景状态编码。
  - 当前 card name vocabulary 会根据配置牌组生成；意图集合仍是简化固定集合，后续若要泛化到更多敌人，需要改成更通用的 enemy/action 表示。
- `rl/actions.py`
  - `RLAction` 与 `RLActionType`：RL 动作结构。
- `rl/bot.py`
  - `RLBattleBot`：给需要选目标的卡牌提供目标选择桥接，实际决策由 `MiniSTSEnv.step()` 驱动。
  - 普通 agent target 仍由 env 设置；手牌选择类 card target 已由 env 的 pending hand choice 机制接管。
- `rl/dqn.py`
  - `DQNAgent`、`QNetwork`、`ReplayBuffer`：基础 DQN 实现。
- `rl/train_dqn.py`
  - DQN 训练入口。
  - 在项目根目录使用虚拟环境执行。
  - 示例：

```bash
.venv/bin/python -m rl.train_dqn --config rl_configs/scenario5_big_jaw_worm.json
```

- `rl/evaluate_dqn.py`
  - DQN checkpoint 评估入口。
  - 支持 `--trace` 打印首局决策过程和 Q 值。
  - 示例：

```bash
.venv/bin/python -m rl.evaluate_dqn --config rl_configs/scenario5_big_jaw_worm.json
.venv/bin/python -m rl.evaluate_dqn --config rl_configs/scenario5_big_jaw_worm.json --episodes 1 --trace
```

命令行参数仍可覆盖 JSON 中的设置，例如：

```bash
.venv/bin/python -m rl.evaluate_dqn --config rl_configs/scenario5_big_jaw_worm.json --episodes 500
```

如果虚拟环境需要重建，依赖参考：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

## 传统评估代码参考

- `evaluation/evaluate_deck.py`
  - 使用 `BacktrackBot` 对不同加牌方案做批量模拟。
  - 可作为“某张牌加入卡组后效果如何”的早期 baseline。
- `evaluation/evaluate_bot.py`
  - 用于比较不同 bot 的对战结果。
- `evaluation/evaluate_card_gen.py`
  - 用于评估随机生成卡牌或卡牌生成策略。
- `evaluation/aggregate_metadata.py`、`plot_evaluation.py`、`plot_property.py`
  - 用于整理与可视化实验结果。

## 下一步重点

1. 把更多原版卡牌、敌人和状态效果补齐，并为关键交互补测试。
2. 将 `MiniSTSEnv` 继续推广到可配置奖励函数、多敌人、多种子和更完整的战斗设置。
3. 建立“增删某张牌”的评估协议：固定随机种子或多种子、多局模拟，记录胜率、剩余血量、平均步数和策略差异。
4. 设计更通用的状态编码，避免 encoder 绑定某个固定卡组。
5. 继续扩展 RL pending choice。当前已支持对“手牌槽位”的 pending 选择，覆盖 `True Grit+`、`Armaments`、`Burning Pact`、`Dual Wield`、`Warcry`；`Exhume`、`Headbutt` 这类从消耗牌堆/弃牌堆选择的效果仍需后续补充对应 pending 编码。
6. 未来再考虑前端或 mod 接口，把模型评估结果转化为游戏内选牌建议。

## 原版机制参考

- 本机一代游戏 jar 路径：

```bash
/Users/charlie/Library/Application Support/Steam/steamapps/common/SlayTheSpire/SlayTheSpire.app/Contents/Resources/desktop-1.0.jar
```

- 用 `tools/sts_reference_extract.py` 从 jar 中提取卡牌结构摘要，输出到 git 忽略的 `local_refs/`：

```bash
.venv/bin/python tools/sts_reference_extract.py --colors red --out local_refs/sts_cards_red_summary.json
```

- 当前已提取 Ironclad/red 卡牌摘要：75 张。
- 摘要包含类名、卡牌 ID、费用、类型、稀有度、目标、基础数值、升级方法、`use()` 依赖的 action/power 类。
- `local_refs/` 只作为本地实现参考，不提交原始游戏资源或反编译源码。
- 已新增 `tools/sts_encounter_extract.py`，可从 jar 提取 dungeon encounter/boss 摘要到 `local_refs/sts_encounters_summary.json`：

```bash
.venv/bin/python tools/sts_encounter_extract.py --dungeons exordium --out local_refs/sts_encounters_summary.json
```

## 实机示例

- 新增/实验卡牌效果演示脚本：

```bash
.venv/bin/python examples/new_card_effects_demo.py
```

- 脚本会逐张展示打出前后状态变化，包括玩家/敌人 HP、Block、Status，以及 hand/draw/discard/exhaust 牌堆变化。
- 当前覆盖：全部已实现 Ironclad 卡牌、实验牌、状态牌，并包含 `True Grit+` 的升级后二级选牌示例。
- 现已覆盖全部已实现 Ironclad 卡牌；每张新增牌都有“打出前/打出后/必要触发后”的简洁状态变化示例。

## 当前注意事项

- 机制实现参考了本机游戏 jar 的类结构，但 MiniSTS 仍是轻量模拟，不是完整引擎复刻。
- 多敌人、复杂选择 UI、部分特殊边界仍需后续测试，例如 `Exhume` 的过滤、触发顺序和多卡联动边界等。
- 旧 DQN checkpoint 与当前 encoder/card vocabulary 可能不兼容；评估旧 `dqn_scenario5_jawworm.pt` 时应带上对应 JSON config，例如：

```bash
.venv/bin/python -m rl.evaluate_dqn --config rl_configs/scenario5_big_jaw_worm.json --checkpoint rl_runs/dqn_scenario5_jawworm.pt --episodes 1 --trace
```
