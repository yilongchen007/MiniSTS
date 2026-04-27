const state = {
  data: null,
};

const els = {
  scenarioLine: document.querySelector("#scenarioLine"),
  resetButton: document.querySelector("#resetButton"),
  playerName: document.querySelector("#playerName"),
  playerHp: document.querySelector("#playerHp"),
  playerBlock: document.querySelector("#playerBlock"),
  playerStatus: document.querySelector("#playerStatus"),
  mana: document.querySelector("#mana"),
  outcome: document.querySelector("#outcome"),
  turn: document.querySelector("#turn"),
  steps: document.querySelector("#steps"),
  hpLoss: document.querySelector("#hpLoss"),
  message: document.querySelector("#message"),
  enemies: document.querySelector("#enemies"),
  pendingBanner: document.querySelector("#pendingBanner"),
  hand: document.querySelector("#hand"),
  drawPile: document.querySelector("#drawPile"),
  discardPile: document.querySelector("#discardPile"),
  exhaustPile: document.querySelector("#exhaustPile"),
  endTurnButton: document.querySelector("#endTurnButton"),
  legalActions: document.querySelector("#legalActions"),
  actionHistory: document.querySelector("#actionHistory"),
};

function text(value) {
  return value === "" || value == null ? "-" : String(value);
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || response.statusText);
  }
  return payload;
}

function legalAction(index) {
  return state.data?.legal_actions?.find((action) => action.index === index) || null;
}

async function loadState() {
  state.data = await fetchJson("/api/state");
  render();
}

async function resetBattle() {
  state.data = await fetchJson("/api/reset", { method: "POST", body: "{}" });
  render();
}

async function takeAction(index) {
  const legal = legalAction(index);
  if (!legal || state.data.done) {
    return;
  }
  state.data = await fetchJson("/api/action", {
    method: "POST",
    body: JSON.stringify({ action: index }),
  });
  render();
}

function render() {
  const data = state.data;
  if (!data) {
    return;
  }

  els.scenarioLine.textContent = `${data.config_path} | ${data.enemy_name} | A${data.ascension}`;
  els.playerName.textContent = data.player.name;
  els.playerHp.textContent = `${data.player.health}/${data.player.max_health}`;
  els.playerBlock.textContent = data.player.block;
  els.playerStatus.textContent = `Status ${text(data.player.status)}`;
  els.mana.textContent = `${data.mana}/${data.max_mana}`;
  els.outcome.textContent = data.outcome;
  els.turn.textContent = data.turn;
  els.steps.textContent = `${data.steps}/${data.max_steps}`;
  els.hpLoss.textContent = data.hp_loss;
  els.message.textContent = data.done ? "Battle ended" : data.last_reward == null ? "" : `Last reward ${data.last_reward.toFixed(4)}`;

  renderEnemies(data.enemies);
  renderPending(data.pending);
  renderHand(data.hand);
  renderPiles(data);
  renderActions(data.legal_actions);
  els.actionHistory.textContent = data.actions?.length ? `actions: ${data.actions.join(",")}` : "actions: -";
}

function renderEnemies(enemies) {
  els.enemies.replaceChildren(...enemies.map((enemy) => {
    const card = document.createElement("article");
    card.className = "enemy-card";
    card.innerHTML = `
      <div class="enemy-heading">
        <span class="enemy-name">${escapeHtml(enemy.name)}</span>
        <span class="label">Enemy ${enemy.index}</span>
      </div>
      <div class="meter-row">
        <div><span class="metric-label">HP</span><span class="metric-value">${enemy.health}/${enemy.max_health}</span></div>
        <div><span class="metric-label">Block</span><span class="metric-value">${enemy.block}</span></div>
        <div><span class="metric-label">Status</span><span class="metric-value small">${escapeHtml(text(enemy.status))}</span></div>
      </div>
      <div class="intent">${escapeHtml(enemy.intent)}</div>
    `;
    return card;
  }));
}

function renderPending(pending) {
  if (!pending) {
    els.pendingBanner.classList.add("hidden");
    els.pendingBanner.textContent = "";
    return;
  }
  els.pendingBanner.classList.remove("hidden");
  els.pendingBanner.textContent = `Choose a hand card for ${pending.purpose}: slots ${pending.hand_indices.join(", ")}`;
}

function renderHand(hand) {
  els.hand.replaceChildren(...hand.map((card) => {
    const actionIndex = card.index + 1;
    const legal = legalAction(actionIndex);
    const button = document.createElement("button");
    button.type = "button";
    button.className = `card ${card.playable ? "" : "not-playable"} ${state.data.pending ? "is-choice" : ""}`;
    button.disabled = !legal || state.data.done;
    button.addEventListener("click", () => takeAction(actionIndex));
    button.innerHTML = `
      <div class="card-heading">
        <span class="card-name">${escapeHtml(card.display_name)}</span>
        <span class="cost">${card.cost}</span>
      </div>
      <div class="card-body">
        <div class="card-meta">${card.type} | ${card.rarity} | slot ${card.index}</div>
        <div class="card-desc">${escapeHtml(card.desc)}</div>
      </div>
    `;
    return button;
  }));
}

function renderPiles(data) {
  els.drawPile.textContent = data.draw_pile.length ? data.draw_pile.join(", ") : "-empty-";
  els.discardPile.textContent = data.discard_pile.length ? data.discard_pile.join(", ") : "-empty-";
  els.exhaustPile.textContent = data.exhaust_pile.length ? data.exhaust_pile.join(", ") : "-empty-";
}

function renderActions(actions) {
  const endTurn = legalAction(0);
  els.endTurnButton.disabled = !endTurn || state.data.done;
  els.endTurnButton.onclick = () => takeAction(0);
  els.legalActions.replaceChildren(...actions.map((action) => {
    const chip = document.createElement("span");
    chip.className = "action-chip";
    chip.textContent = `${action.index}: ${action.label}`;
    return chip;
  }));
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

els.resetButton.addEventListener("click", resetBattle);
loadState().catch((error) => {
  els.message.textContent = error.message;
});
