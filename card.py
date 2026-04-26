from __future__ import annotations
from target.agent_target import AgentSet, ChooseAgentTarget, SelfAgentTarget, AllAgentsTarget, RandomAgentTarget
from target.card_target import CardPile, SelfCardTarget, ChooseCardTarget, RandomCardTarget, AllCardsTarget, UpgradeSwitchCardTarget
from action.action import (
    Action,
    AddMana,
    BurningPactAction,
    DoubleBlock,
    DrawCard,
    DualWieldAction,
    ExhaustNonAttackCards,
    ExhumeAction,
    FiendFireAction,
    HavocAction,
    HeadbuttAction,
    InfernalBladeAction,
    LimitBreakAction,
    MakeTempCard,
    SecondWindAction,
    SeverSoulAction,
    UpgradeAllCardsInHand,
    WarcryAction,
    WhirlwindAction,
)
from action.agent_targeted_action import (
    DealAttackDamage,
    DealBodySlamDamage,
    DealHeavyBladeDamage,
    Dropkick,
    Feed,
    PerfectedStrike,
    Reaper,
    Rampage,
    ApplyStatus,
    AddBlock,
    Heal,
    LoseHP,
    SpotWeakness,
    SwordBoomerang,
)
from action.card_targeted_action import CardTargetedL1, Exhaust, AddCopy, UpgradeCard, DiscardCard
from config import CardType, Character, Rarity
from status_effecs import StatusEffectRepo, StatusEffectDefinition
from value import Value, ConstValue, UpgradableOnce, LinearUpgradable
from utility import RandomStr
from typing import TYPE_CHECKING, Callable
if TYPE_CHECKING:
    from game import GameState
    from battle import BattleState

class Card:
    def __init__(
        self,
        name: str,
        card_type: CardType,
        mana_cost: Value,
        character: Character,
        rarity: Rarity,
        *actions: Action|CardTargetedL1,
        desc: str|None = None,
        playable: bool = True,
        ethereal: bool = False,
        exhaust_on_play: bool = False,
        remove_exhaust_on_upgrade: bool = False,
        max_upgrades: int | None = 1,
        play_condition: Callable[[GameState, BattleState, Card], bool] | None = None,
    ):
        self.name = name
        self.card_type = card_type
        self.mana_cost = mana_cost
        self.character = character
        self.rarity = rarity
        self.playable = playable
        self.ethereal = ethereal
        self.exhaust_on_play = exhaust_on_play
        self.remove_exhaust_on_upgrade = remove_exhaust_on_upgrade
        self.max_upgrades = max_upgrades
        self.play_condition = play_condition
        self.cost_override: int|None = None
        self.upgrade_count = 0
        self.mana_action = AddMana(mana_cost.negative())
        self.actions: list[Action] = []
        for action in actions:
            if isinstance(action, Action):
                self.actions.append(action)
            else:
                self.actions.append(action.By(self))
        self.desc = desc if desc is not None else " ".join([f"{action}" for action in self.actions])
    
    def play(self, game_state: GameState, battle_state: BattleState):
        assert self.is_playable(game_state, battle_state)
        battle_state.add_to_mana(-self.effective_cost(game_state, battle_state))
        self.play_actions(game_state, battle_state)

    def play_actions(self, game_state: GameState, battle_state: BattleState):
        if self.name == "Armaments" and self.upgrade_count > 0:
            battle_state.gain_block(game_state.player, 5)
            for card in list(battle_state.hand):
                card.upgrade()
            return
        for action in self.actions:
            action.play(game_state.player, game_state, battle_state)

    def effective_cost(self, game_state: GameState, battle_state: BattleState) -> int:
        if self.cost_override is not None:
            return self.cost_override
        if self.card_type == CardType.SKILL and game_state.player.status_effect_state.has(StatusEffectRepo.CORRUPTION):
            return 0
        if self.name == "Blood for Blood":
            return max(0, self.mana_cost.peek() - battle_state.player_hp_lost_this_combat)
        if self.mana_cost.peek() < 0:
            return 0
        return self.mana_cost.peek()

    def is_playable(self, game_state: GameState, battle_state: BattleState, ignore_mana: bool = False):
        if not self.playable:
            return False
        if self.card_type == CardType.ATTACK and game_state.player.status_effect_state.has(StatusEffectRepo.ENTANGLED):
            return False
        if self.play_condition is not None and not self.play_condition(game_state, battle_state, self):
            return False
        return ignore_mana or self.effective_cost(game_state, battle_state) <= battle_state.mana

    def upgrade(self, times: int = 1):
        if times <= 0:
            return
        if self.max_upgrades is not None:
            times = min(times, self.max_upgrades - self.upgrade_count)
        if times <= 0:
            return
        self.upgrade_count += times
        self.mana_cost.upgrade(times)
        if self.remove_exhaust_on_upgrade and self.upgrade_count > 0:
            self.exhaust_on_play = False
        for action in self.actions:
            for val in action.values:
                val.upgrade(times)

    def get_name(self) -> str:
        return "{}{}".format(self.name, "+"*self.upgrade_count)
    
    def __repr__(self) -> str:
        return "{}-cost:{}-{}-{}\n-".format(self.get_name(), self.mana_cost.peek(), self.card_type, self.rarity) + \
            "\n-".join(['' + action.__repr__() for action in self.actions])

    def get_description(self) -> str:
        return self.desc

def clash_condition(_: GameState, battle_state: BattleState, __: Card) -> bool:
    return all(card.card_type == CardType.ATTACK for card in battle_state.hand)

class CardGen:
    # 中文：打击。造成基础攻击伤害。
    Strike = lambda: Card("Strike", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.STARTER, DealAttackDamage(UpgradableOnce(6, 9)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：防御。获得基础格挡。
    Defend = lambda: Card("Defend", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.STARTER, AddBlock(UpgradableOnce(5, 8)).To(SelfAgentTarget()))
    # 中文：灼热攻击。可反复升级，伤害随升级线性增长。
    Searing_Blow = lambda: Card("SearingBlow", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.UNCOMMON, DealAttackDamage(LinearUpgradable(12, 4)).To(ChooseAgentTarget(AgentSet.ENEMY)), max_upgrades=None)
    # 中文：痛击。造成伤害并给予易伤。
    Bash = lambda: Card("Bash", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.STARTER, DealAttackDamage(UpgradableOnce(8, 10)).And(ApplyStatus(UpgradableOnce(2, 3), StatusEffectRepo.VULNERABLE)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：愤怒。造成伤害，并向弃牌堆加入自身复制。
    Anger = lambda: Card("Anger", CardType.ATTACK, ConstValue(0), Character.IRON_CLAD, Rarity.COMMON, DealAttackDamage(UpgradableOnce(6, 8)).To(ChooseAgentTarget(AgentSet.ENEMY)), AddCopy(CardPile.DISCARD).To(SelfCardTarget()))
    # 中文：剑柄打击。造成伤害并抽牌。
    Pommel_Strike = lambda: Card("Pommel Strike", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, DealAttackDamage(UpgradableOnce(9, 10)).To(ChooseAgentTarget(AgentSet.ENEMY)), DrawCard(UpgradableOnce(1, 2)))
    # 中文：耸肩无视。获得格挡并抽一张牌。
    Shrug_It_Off = lambda: Card("Shrug It Off", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, AddBlock(UpgradableOnce(8, 11)).To(SelfAgentTarget()), DrawCard(ConstValue(1)))
    # 中文：放血。失去生命，获得能量。
    Bloodletting = lambda: Card("Bloodletting", CardType.SKILL, ConstValue(0), Character.IRON_CLAD, Rarity.UNCOMMON, LoseHP(ConstValue(3)).To(SelfAgentTarget()), AddMana(UpgradableOnce(2, 3)))
    # 中文：顺劈斩。对所有敌人造成伤害。
    Cleave = lambda: Card("Cleave", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, DealAttackDamage(UpgradableOnce(8, 11)).To(AllAgentsTarget(AgentSet.ENEMY)))
    # 中文：岿然不动。获得大量格挡并消耗。
    Impervious = lambda: Card("Impervious", CardType.SKILL, ConstValue(2), Character.IRON_CLAD, Rarity.RARE, AddBlock(UpgradableOnce(30, 40)).To(SelfAgentTarget()), Exhaust().To(SelfCardTarget()))
    # 中文：战斗专注。抽牌，本回合之后不能再抽牌。
    Battle_Trance = lambda: Card("Battle Trance", CardType.SKILL, ConstValue(0), Character.IRON_CLAD, Rarity.UNCOMMON, DrawCard(UpgradableOnce(3, 4)), ApplyStatus(ConstValue(1), StatusEffectRepo.NO_DRAW).To(SelfAgentTarget()))
    # 中文：活动肌肉。临时获得力量，回合结束失去。
    Flex = lambda: Card("Flex", CardType.SKILL, ConstValue(0), Character.IRON_CLAD, Rarity.COMMON, ApplyStatus(UpgradableOnce(2, 4), StatusEffectRepo.STRENGTH).And(ApplyStatus(UpgradableOnce(2, 4), StatusEffectRepo.LOSE_STRENGTH)).To(SelfAgentTarget()))
    # 中文：燃烧。永久获得力量。
    Inflame = lambda: Card("Inflame", CardType.POWER, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(2, 3), StatusEffectRepo.STRENGTH).To(SelfAgentTarget()))
    # 中文：金属化。回合结束获得格挡。
    Metallicize = lambda: Card("Metallicize", CardType.POWER, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(3, 4), StatusEffectRepo.METALLICIZE).To(SelfAgentTarget()))
    # 中文：壁垒。格挡不会在回合开始时消失。
    Barricade = lambda: Card("Barricade", CardType.POWER, UpgradableOnce(3, 2), Character.IRON_CLAD, Rarity.RARE, ApplyStatus(ConstValue(1), StatusEffectRepo.BARRICADE).To(SelfAgentTarget()))
    # 中文：盛怒。获得能量并消耗。
    Seeing_Red = lambda: Card("Seeing Red", CardType.SKILL, UpgradableOnce(1, 0), Character.IRON_CLAD, Rarity.UNCOMMON, AddMana(ConstValue(2)), Exhaust().To(SelfCardTarget()))
    # 中文：坚毅。获得格挡，消耗一张手牌；升级后可选牌。
    True_Grit = lambda: Card("True Grit", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, AddBlock(UpgradableOnce(7, 9)).To(SelfAgentTarget()), Exhaust().To(UpgradeSwitchCardTarget(RandomCardTarget(CardPile.HAND), ChooseCardTarget(CardPile.HAND))))
    # 中文：狂野打击。造成高伤害，向抽牌堆加入伤口。
    Wild_Strike = lambda: Card("Wild Strike", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, DealAttackDamage(UpgradableOnce(12, 17)).To(ChooseAgentTarget(AgentSet.ENEMY)), MakeTempCard(lambda: CardGen.Wound(), CardPile.DRAW))
    # 中文：强硬撑住。获得大量格挡，向手牌加入伤口。
    Power_Through = lambda: Card("Power Through", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, AddBlock(UpgradableOnce(15, 20)).To(SelfAgentTarget()), MakeTempCard(lambda: CardGen.Wound(), CardPile.HAND, ConstValue(2)))
    # 中文：狂怒。本回合打出攻击牌时获得格挡。
    Rage = lambda: Card("Rage", CardType.SKILL, ConstValue(0), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(3, 5), StatusEffectRepo.RAGE).To(SelfAgentTarget()))
    # 中文：无惧疼痛。每当有牌被消耗时获得格挡。
    Feel_No_Pain = lambda: Card("Feel No Pain", CardType.POWER, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(3, 4), StatusEffectRepo.FEEL_NO_PAIN).To(SelfAgentTarget()))
    # 中文：黑暗之拥。每当有牌被消耗时抽牌。
    Dark_Embrace = lambda: Card("Dark Embrace", CardType.POWER, UpgradableOnce(2, 1), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(ConstValue(1), StatusEffectRepo.DARK_EMBRACE).To(SelfAgentTarget()))
    # 中文：进化。抽到状态牌时额外抽牌。
    Evolve = lambda: Card("Evolve", CardType.POWER, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(1, 2), StatusEffectRepo.EVOLVE).To(SelfAgentTarget()))
    # 中文：火焰吐息。抽到状态或诅咒时伤害所有敌人。
    Fire_Breathing = lambda: Card("Fire Breathing", CardType.POWER, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(6, 10), StatusEffectRepo.FIRE_BREATHING).To(SelfAgentTarget()))
    # 中文：势不可挡。每当获得格挡时，对随机敌人造成伤害。
    Juggernaut = lambda: Card("Juggernaut", CardType.POWER, ConstValue(2), Character.IRON_CLAD, Rarity.RARE, ApplyStatus(UpgradableOnce(5, 7), StatusEffectRepo.JUGGERNAUT).To(SelfAgentTarget()))
    # 中文：火焰屏障。获得格挡，本回合攻击者受到反伤。
    Flame_Barrier = lambda: Card("Flame Barrier", CardType.SKILL, ConstValue(2), Character.IRON_CLAD, Rarity.UNCOMMON, AddBlock(UpgradableOnce(12, 16)).To(SelfAgentTarget()), ApplyStatus(UpgradableOnce(4, 6), StatusEffectRepo.FLAME_BARRIER).To(SelfAgentTarget()))
    # 中文：重锤。造成大量伤害。
    Bludgeon = lambda: Card("Bludgeon", CardType.ATTACK, ConstValue(3), Character.IRON_CLAD, Rarity.RARE, DealAttackDamage(UpgradableOnce(32, 42)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：金刚臂。造成伤害并给予虚弱。
    Clothesline = lambda: Card("Clothesline", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.COMMON, DealAttackDamage(UpgradableOnce(12, 14)).And(ApplyStatus(UpgradableOnce(2, 3), StatusEffectRepo.WEAK)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：闪电霹雳。对所有敌人造成伤害并给予易伤。
    Thunderclap = lambda: Card("Thunderclap", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, DealAttackDamage(UpgradableOnce(4, 7)).And(ApplyStatus(ConstValue(1), StatusEffectRepo.VULNERABLE)).To(AllAgentsTarget(AgentSet.ENEMY)))
    # 中文：双重打击。对同一敌人造成两次伤害。
    Twin_Strike = lambda: Card("Twin Strike", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, DealAttackDamage(UpgradableOnce(5, 7), ConstValue(2)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：上勾拳。造成伤害并给予虚弱和易伤。
    Uppercut = lambda: Card("Uppercut", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.UNCOMMON, DealAttackDamage(ConstValue(13)).And(ApplyStatus(UpgradableOnce(1, 2), StatusEffectRepo.WEAK)).And(ApplyStatus(UpgradableOnce(1, 2), StatusEffectRepo.VULNERABLE)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：铁斩波。获得格挡并造成伤害。
    Iron_Wave = lambda: Card("Iron Wave", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, AddBlock(UpgradableOnce(5, 7)).To(SelfAgentTarget()), DealAttackDamage(UpgradableOnce(5, 7)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：连续拳。多段低伤害攻击并消耗。
    Pummel = lambda: Card("Pummel", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, DealAttackDamage(ConstValue(2), UpgradableOnce(4, 5)).To(ChooseAgentTarget(AgentSet.ENEMY)), Exhaust().To(SelfCardTarget()))
    # 中文：御血术。失去生命并造成伤害。
    Hemokinesis = lambda: Card("Hemokinesis", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, LoseHP(ConstValue(2)).To(SelfAgentTarget()), DealAttackDamage(UpgradableOnce(15, 20)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：祭品。失去生命，获得能量并抽牌，然后消耗。
    Offering = lambda: Card("Offering", CardType.SKILL, ConstValue(0), Character.IRON_CLAD, Rarity.RARE, LoseHP(ConstValue(6)).To(SelfAgentTarget()), AddMana(ConstValue(2)), DrawCard(UpgradableOnce(3, 5)), Exhaust().To(SelfCardTarget()))
    # 中文：缴械。降低敌人力量并消耗。
    Disarm = lambda: Card("Disarm", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(-2, -3), StatusEffectRepo.STRENGTH).To(ChooseAgentTarget(AgentSet.ENEMY)), Exhaust().To(SelfCardTarget()))
    # 中文：震荡波。对所有敌人给予虚弱和易伤，然后消耗。
    Shockwave = lambda: Card("Shockwave", CardType.SKILL, ConstValue(2), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(3, 5), StatusEffectRepo.WEAK).And(ApplyStatus(UpgradableOnce(3, 5), StatusEffectRepo.VULNERABLE)).To(AllAgentsTarget(AgentSet.ENEMY)), Exhaust().To(SelfCardTarget()))
    # 中文：狂暴。获得易伤；之后每回合开始获得能量。
    Berserk = lambda: Card("Berserk", CardType.POWER, ConstValue(0), Character.IRON_CLAD, Rarity.RARE, ApplyStatus(UpgradableOnce(2, 1), StatusEffectRepo.VULNERABLE).And(ApplyStatus(ConstValue(1), StatusEffectRepo.BERSERK)).To(SelfAgentTarget()))
    # 中文：恶魔形态。每回合开始获得力量。
    Demon_Form = lambda: Card("Demon Form", CardType.POWER, ConstValue(3), Character.IRON_CLAD, Rarity.RARE, ApplyStatus(UpgradableOnce(2, 3), StatusEffectRepo.DEMON_FORM).To(SelfAgentTarget()))
    # 中文：残暴。每回合开始失去生命并抽牌。
    Brutality = lambda: Card("Brutality", CardType.POWER, ConstValue(0), Character.IRON_CLAD, Rarity.RARE, ApplyStatus(ConstValue(1), StatusEffectRepo.BRUTALITY).To(SelfAgentTarget()))

    # 中文：武装。获得格挡，并在本场战斗中升级手牌。
    Armaments = lambda: Card("Armaments", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, AddBlock(ConstValue(5)).To(SelfAgentTarget()), UpgradeCard().To(ChooseCardTarget(CardPile.HAND)))
    # 中文：以血还血。造成伤害；本场战斗每失去生命会降低费用。
    Blood_for_Blood = lambda: Card("Blood for Blood", CardType.ATTACK, UpgradableOnce(4, 3), Character.IRON_CLAD, Rarity.UNCOMMON, DealAttackDamage(UpgradableOnce(18, 22)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：全身撞击。造成等同当前格挡的攻击伤害。
    Body_Slam = lambda: Card("Body Slam", CardType.ATTACK, UpgradableOnce(1, 0), Character.IRON_CLAD, Rarity.COMMON, DealBodySlamDamage().To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：燃烧契约。消耗一张手牌并抽牌。
    Burning_Pact = lambda: Card("Burning Pact", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, BurningPactAction(UpgradableOnce(2, 3)))
    # 中文：残杀。造成高伤害；留在手牌到回合结束会因虚无被消耗。
    Carnage = lambda: Card("Carnage", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.UNCOMMON, DealAttackDamage(UpgradableOnce(20, 28)).To(ChooseAgentTarget(AgentSet.ENEMY)), ethereal=True)
    # 中文：交锋。只有手牌全是攻击牌时可打出。
    Clash = lambda: Card("Clash", CardType.ATTACK, ConstValue(0), Character.IRON_CLAD, Rarity.COMMON, DealAttackDamage(UpgradableOnce(14, 18)).To(ChooseAgentTarget(AgentSet.ENEMY)), play_condition=clash_condition)
    # 中文：燃烧。回合结束失去生命并伤害所有敌人。
    Combust = lambda: Card("Combust", CardType.POWER, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(5, 7), StatusEffectRepo.COMBUST).To(SelfAgentTarget()))
    # 中文：腐化。技能牌变为免费，打出后消耗。
    Corruption = lambda: Card("Corruption", CardType.POWER, UpgradableOnce(3, 2), Character.IRON_CLAD, Rarity.RARE, ApplyStatus(ConstValue(1), StatusEffectRepo.CORRUPTION).To(SelfAgentTarget()))
    # 中文：双发。本回合下一张或下几张攻击牌打出两次。
    Double_Tap = lambda: Card("Double Tap", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.RARE, ApplyStatus(UpgradableOnce(1, 2), StatusEffectRepo.DOUBLE_TAP).To(SelfAgentTarget()))
    # 中文：飞身踢。目标有易伤时，攻击后获得能量并抽牌。
    Dropkick = lambda: Card("Dropkick", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, Dropkick(UpgradableOnce(5, 8)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：双持。复制手牌中的攻击牌或能力牌。
    Dual_Wield = lambda: Card("Dual Wield", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, DualWieldAction(UpgradableOnce(1, 2)))
    # 中文：巩固。将当前格挡翻倍。
    Entrench = lambda: Card("Entrench", CardType.SKILL, UpgradableOnce(2, 1), Character.IRON_CLAD, Rarity.UNCOMMON, DoubleBlock())
    # 中文：发掘。从消耗牌堆取回一张牌；未升级会消耗自身。
    Exhume = lambda: Card("Exhume", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.RARE, ExhumeAction(), exhaust_on_play=True, remove_exhaust_on_upgrade=True)
    # 中文：饲育。造成伤害；若击杀敌人则提高最大生命。
    Feed = lambda: Card("Feed", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.RARE, Feed(UpgradableOnce(10, 12), UpgradableOnce(3, 4)).To(ChooseAgentTarget(AgentSet.ENEMY)), exhaust_on_play=True)
    # 中文：恶魔之焰。消耗所有手牌，并按消耗张数造成多次伤害。
    Fiend_Fire = lambda: Card("Fiend Fire", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.RARE, FiendFireAction(UpgradableOnce(7, 10)), exhaust_on_play=True)
    # 中文：幽灵铠甲。获得格挡；留在手牌到回合结束会虚无。
    Ghostly_Armor = lambda: Card("Ghostly Armor", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, AddBlock(UpgradableOnce(10, 13)).To(SelfAgentTarget()), ethereal=True)
    # 中文：破灭。打出抽牌堆顶牌并消耗它。
    Havoc = lambda: Card("Havoc", CardType.SKILL, UpgradableOnce(1, 0), Character.IRON_CLAD, Rarity.COMMON, HavocAction())
    # 中文：头槌。造成伤害，并把弃牌堆一张牌放回抽牌堆顶。
    Headbutt = lambda: Card("Headbutt", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, HeadbuttAction(UpgradableOnce(9, 12)))
    # 中文：重刃。造成伤害，力量加成按倍数生效。
    Heavy_Blade = lambda: Card("Heavy Blade", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.COMMON, DealHeavyBladeDamage(ConstValue(14), UpgradableOnce(3, 5)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：燔祭。对所有敌人造成大量伤害，并加入灼伤。
    Immolate = lambda: Card("Immolate", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.RARE, DealAttackDamage(UpgradableOnce(21, 28)).To(AllAgentsTarget(AgentSet.ENEMY)), MakeTempCard(lambda: CardGen.Burn(), CardPile.DISCARD))
    # 中文：地狱之刃。加入一张随机攻击牌，本回合费用为零。
    Infernal_Blade = lambda: Card("Infernal Blade", CardType.SKILL, UpgradableOnce(1, 0), Character.IRON_CLAD, Rarity.UNCOMMON, InfernalBladeAction())
    # 中文：威吓。对所有敌人给予虚弱并消耗。
    Intimidate = lambda: Card("Intimidate", CardType.SKILL, ConstValue(0), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(1, 2), StatusEffectRepo.WEAK).To(AllAgentsTarget(AgentSet.ENEMY)), exhaust_on_play=True)
    # 中文：突破极限。将当前力量翻倍；未升级会消耗。
    Limit_Break = lambda: Card("Limit Break", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.RARE, LimitBreakAction(), exhaust_on_play=True, remove_exhaust_on_upgrade=True)
    # 中文：完美打击。伤害随包含“打击”的牌数量提高。
    Perfected_Strike = lambda: Card("Perfected Strike", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.COMMON, PerfectedStrike(UpgradableOnce(6, 6), UpgradableOnce(2, 3)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：暴走。本场战斗中每次打出后提高自身伤害。
    Rampage = lambda: Card("Rampage", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, Rampage(ConstValue(8), UpgradableOnce(5, 8)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：收割。伤害所有敌人，并按未被格挡的伤害回复生命。
    Reaper = lambda: Card("Reaper", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.RARE, Reaper(UpgradableOnce(4, 5)).To(AllAgentsTarget(AgentSet.ENEMY)), exhaust_on_play=True)
    # 中文：鲁莽冲锋。造成伤害，向抽牌堆加入晕眩。
    Reckless_Charge = lambda: Card("Reckless Charge", CardType.ATTACK, ConstValue(0), Character.IRON_CLAD, Rarity.COMMON, DealAttackDamage(UpgradableOnce(7, 10)).To(ChooseAgentTarget(AgentSet.ENEMY)), MakeTempCard(lambda: CardGen.Dazed(), CardPile.DRAW))
    # 中文：撕裂。每当因牌失去生命时获得力量。
    Rupture = lambda: Card("Rupture", CardType.POWER, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, ApplyStatus(UpgradableOnce(1, 2), StatusEffectRepo.RUPTURE).To(SelfAgentTarget()))
    # 中文：重振精神。消耗所有非攻击手牌，并按张数获得格挡。
    Second_Wind = lambda: Card("Second Wind", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, SecondWindAction(UpgradableOnce(5, 7)))
    # 中文：哨卫。获得格挡；被消耗时获得能量。
    Sentinel = lambda: Card("Sentinel", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, AddBlock(UpgradableOnce(5, 8)).To(SelfAgentTarget()))
    # 中文：断魂斩。消耗非攻击非能力手牌并造成伤害。
    Sever_Soul = lambda: Card("Sever Soul", CardType.ATTACK, ConstValue(2), Character.IRON_CLAD, Rarity.UNCOMMON, SeverSoulAction(UpgradableOnce(16, 22)))
    # 中文：观察弱点。若目标意图攻击，则获得力量。
    Spot_Weakness = lambda: Card("Spot Weakness", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.UNCOMMON, SpotWeakness(UpgradableOnce(3, 4)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：回旋镖。多次随机攻击敌人。
    Sword_Boomerang = lambda: Card("Sword Boomerang", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, SwordBoomerang(ConstValue(3), UpgradableOnce(3, 4)).To(RandomAgentTarget(AgentSet.ENEMY)))
    # 中文：战吼。抽牌，然后把一张手牌放回抽牌堆顶；消耗。
    Warcry = lambda: Card("Warcry", CardType.SKILL, ConstValue(0), Character.IRON_CLAD, Rarity.COMMON, WarcryAction(UpgradableOnce(1, 2)), exhaust_on_play=True)
    # 中文：旋风斩。消耗所有能量，每点能量对所有敌人造成一次伤害。
    Whirlwind = lambda: Card("Whirlwind", CardType.ATTACK, ConstValue(-1), Character.IRON_CLAD, Rarity.UNCOMMON, WhirlwindAction(UpgradableOnce(5, 8)))
    # 中文：红色打击别名；对应 Ironclad 初始牌 Strike_Red。
    Strike_R = Strike
    # 中文：红色防御别名；对应 Ironclad 初始牌 Defend_Red。
    Defend_R = Defend

    # TODO this doesn't work yet, here for reference
    # Survivor = lambda: Card("Survivor", CardType.SKILL, ConstValue(1), Character.SILENT, Rarity.COMMON, AddBlock(ConstValue(8)).To(SelfAgentTarget()), DiscardCard().To(ChooseCardTarget(CardPile.HAND)))

    # STATUS CARDS
    # 中文：伤口。不可打出状态牌。
    Wound = lambda: Card("Wound", CardType.STATUS, ConstValue(0), Character.IRON_CLAD, Rarity.COMMON, desc="Unplayable.", playable=False)
    # 中文：粘液。状态牌；可花费 1 点能量打出并消耗。
    Slimed = lambda: Card("Slimed", CardType.STATUS, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, desc="Exhaust.", exhaust_on_play=True)
    # 中文：晕眩。不可打出，回合结束虚无。
    Dazed = lambda: Card("Dazed", CardType.STATUS, ConstValue(0), Character.IRON_CLAD, Rarity.COMMON, desc="Unplayable. Ethereal.", playable=False, ethereal=True)
    # 中文：灼伤。不可打出，回合结束造成生命损失。
    Burn = lambda: Card("Burn", CardType.STATUS, ConstValue(0), Character.IRON_CLAD, Rarity.COMMON, desc="Unplayable. At the end of your turn, take 2 damage.", playable=False)
    
    # NEW CARDS
    # 中文：实验牌。获得活力，下次攻击增加伤害。
    Stimulate = lambda: Card("Stimulate", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, ApplyStatus(ConstValue(4), StatusEffectRepo.VIGOR).To(SelfAgentTarget()))
    # 中文：实验牌。进行多段零基础伤害攻击，主要用于测试力量/活力叠加。
    Batter = lambda: Card("Batter", CardType.SKILL, ConstValue(1), Character.IRON_CLAD, Rarity.COMMON, DealAttackDamage(ConstValue(0), ConstValue(10)).To(ChooseAgentTarget(AgentSet.ENEMY)))
    # 中文：实验牌。每回合获得递增格挡。
    Tolerate = lambda: Card("Tolerate", CardType.POWER, ConstValue(3), Character.IRON_CLAD, Rarity.COMMON, ApplyStatus(ConstValue(1), StatusEffectRepo.TOLERANCE).To(SelfAgentTarget()), desc="Gain 1 block every turn and increase this gain by 2.")
    # 中文：实验牌。倒计时后对所有敌人造成大量伤害。
    Bomb = lambda: Card("Bomb", CardType.SKILL, ConstValue(2), Character.IRON_CLAD, Rarity.COMMON, ApplyStatus(ConstValue(3), StatusEffectRepo.BOMB).To(SelfAgentTarget()), desc="At the end of 3 turns, deal 40 damage to all enemies.")
    # 中文：实验牌。造成高额单体伤害。
    Suffer = lambda: Card("Suffer", CardType.ATTACK, ConstValue(1), Character.IRON_CLAD, Rarity.STARTER, DealAttackDamage(UpgradableOnce(15, 30)).To(ChooseAgentTarget(AgentSet.ENEMY)))

class CardRepo:
    @staticmethod
    def get_random() -> Callable[[], Card]:
        import random, string
        import numpy as np
        def get_random_pile():
            return random.choice([CardPile.HAND, CardPile.DISCARD, CardPile.DRAW])
        def get_random_target():
            return random.choice([AllAgentsTarget(AgentSet.ALL), AllAgentsTarget(AgentSet.ENEMY), SelfAgentTarget(), ChooseAgentTarget(AgentSet.ENEMY), RandomAgentTarget(AgentSet.ENEMY)])
        def get_deal_damage(cost: Value) -> Action:
            val = random.randint(0, int((cost.peek() + 1) * 10))
            multi = 1 if random.randint(0, 3) != 0 else random.randint(2, 10)
            target = get_random_target()
            return DealAttackDamage(ConstValue(int(val/multi)), ConstValue(multi)).To(target)
        def get_add_copy():
            return AddCopy(CardPile.DISCARD).To(SelfCardTarget())
        def get_add_block(cost: Value) -> Action:
            val = random.randint(0, int((cost.peek() + 1) * 7))
            return AddBlock(ConstValue(val)).To(get_random_target())
        def get_apply_status(cost: Value) -> Action:
            val = random.randint(0, int((cost.peek() + 1) * 5))
            ses: list[StatusEffectDefinition] = [StatusEffectRepo.STRENGTH, StatusEffectRepo.VIGOR, StatusEffectRepo.VULNERABLE, StatusEffectRepo.WEAK]
            status = random.choice(ses)
            return ApplyStatus(ConstValue(val), status).To(get_random_target())
        def get_random_action(cost: Value) -> Action|CardTargetedL1:
            return random.choice([
                get_deal_damage(cost), get_add_copy(), get_add_block(cost), get_apply_status(cost)
                ])
        name = RandomStr.get_random()
        type = random.choice([CardType.ATTACK, CardType.POWER, CardType.SKILL])
        p = np.array([1, 1, 0.8, 0.3, 0.1, 0.05])
        p /= sum(p)
        cost = np.random.choice([0, 1, 2, 3, 4, 5], p=p)
        char = Character.IRON_CLAD
        rarity = Rarity.COMMON
        acs: list[CardTargetedL1|Action] = []
        p = np.array([1, 0.5, 0.1])
        p /= sum(p)
        ac_count = np.random.choice([1, 2, 3], p=p)
        if type == CardType.ATTACK:
            acs.append(get_deal_damage(ConstValue(int(cost/ac_count))))
        while len(acs) != ac_count:
            acs.append(get_random_action(ConstValue(int(cost/ac_count))))
        if random.randint(0, 3) == 0:
            acs.append(Exhaust().To(SelfCardTarget()))
        # TODO check copy
        gen = lambda: Card(name, type, ConstValue(cost), char, rarity, *acs)
        return gen

    @staticmethod
    def get_starter(character: Character) -> list[Card]:
        starter: list[Card] = []
        if character == Character.IRON_CLAD:
            starter += [CardGen.Strike() for _ in range(5)]
            starter += [CardGen.Defend() for _ in range(4)]
            starter += [CardGen.Bash() for _ in range(1)]
            return starter
        else:
            raise Exception("Undefined started deck for character {}.".format(character))
        
    @staticmethod
    def get_basics() -> list[Card]:
        deck: list[Card] = []
        deck += [CardGen.Strike() for _ in range(5)]
        deck += [CardGen.Defend() for _ in range(4)]
        return deck
    
    @staticmethod
    def get_scenario_0() -> tuple[str, list[Card]]:
        deck: list[Card] = CardRepo.get_starter(Character.IRON_CLAD)
        return "starter-ironclad", deck

    @staticmethod
    def get_scenario_1() -> tuple[str, list[Card]]:
        deck: list[Card] = CardRepo.get_basics()
        deck += [CardGen.Batter(), CardGen.Stimulate()]
        return "basics-batter-stimulate", deck
    
    @staticmethod
    def get_scenario_2() -> tuple[str, list[Card]]:
        deck: list[Card] = []
        deck += [CardGen.Strike() for _ in range(1)]
        deck += [CardGen.Defend() for _ in range(3)]
        deck += [CardGen.Tolerate()]
        return "1s3d-tolerate", deck

    @staticmethod
    def get_scenario_3() -> tuple[str, list[Card]]:
        deck: list[Card] = CardRepo.get_basics()
        deck += [CardGen.Bomb()]
        return "basics-bomb", deck

    @staticmethod
    def get_scenario_4() -> tuple[str, list[Card]]:
        deck: list[Card] = CardRepo.get_basics()
        deck += [CardGen.Suffer()]
        return "basics-suffer", deck

    @staticmethod
    def get_scenario_5() -> tuple[str, list[Card]]:
        pommel_strike = CardGen.Pommel_Strike()
        pommel_strike.upgrade()
        deck: list[Card] = [
            CardGen.Shrug_It_Off(),
            CardGen.Defend(),
            CardGen.Strike(),
            CardGen.Bash(),
            CardGen.Bloodletting(),
            CardGen.Pommel_Strike(),
        ]
        return "shrug-defend-strike-bash-bloodletting-pommel+", deck
    
    @staticmethod
    def anonymize_scenario(scenario: tuple[str, list[Card]]) -> tuple[str, list[Card]]:
        name, cards = scenario
        cards = CardRepo.anonymize_deck(cards)
        return name, cards
    
    @staticmethod
    def anonymize_deck(cards: list[Card]):
        for card in cards:
            card.name = RandomStr.get_hashed(card.name)
        return cards
