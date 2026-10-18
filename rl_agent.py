"""PyTorch-based reinforcement-learner agent for Conquest.

Kept separate from agents.py so games with only human/LLM/random players
never need to import torch.
"""

import os
import random
import contextlib
from typing import NamedTuple
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents import SELECTED_TERRITORIES, Agent, RandomAgent
from game_types import Adjacency, PlayerInfo, Players, Territories

# ---------------------------------------------------------------------------
# Board constants
# ---------------------------------------------------------------------------

# Fixed territory ordering, every state tensor's rows follow this order.
TERRITORY_NAMES = [
    "Alaska", "Great Lakes", "North America", "South America", "Greenland",
    "Europe", "North Africa", "South Africa", "Arabia", "Ural",
    "Central Asia", "South Asia", "Siberia", "Australia",
]
TERRITORY_INDEX = {name: i for i, name in enumerate(TERRITORY_NAMES)}
NUM_TERRITORIES = len(TERRITORY_NAMES)
MAX_PLAYERS     = 5
UNCLAIMED_COL   = MAX_PLAYERS
UNITS_OFFSET    = MAX_PLAYERS + 1
STATE_WIDTH     = 2 * MAX_PLAYERS + 1
UNIT_NORM       = 10.0      # Divisor used to squash raw unit counts into roughly [0, 1]


def build_adjacency_matrix(adjacency: Adjacency) -> torch.Tensor:
    """Builds the (NUM_TERRITORIES, NUM_TERRITORIES) adjacency matrix. The game's adjacency sets already
    include each territory as its own neighbour, so the diagonal comes out as self-loops for free."""
    matrix = torch.zeros(NUM_TERRITORIES, NUM_TERRITORIES)
    for name, neighbours in adjacency.items():
        row = TERRITORY_INDEX[name]
        for neighbour in neighbours:
            matrix[row, TERRITORY_INDEX[neighbour]] = 1.0
    return matrix


# ---------------------------------------------------------------------------
# Egocentric state encoding
# ---------------------------------------------------------------------------

def rotate_player_ids(player_id: int, players: Players) -> dict[int, int]:
    """Rotates player_ids to an egocentric order (first slot is the agent itself, slot 2 is whoever acts
    right after the agent, slot 3 after that, wrapping around). Eliminated players keep their slot."""
    seat_order = sorted(players.keys())
    start      = seat_order.index(player_id)
    rotated    = seat_order[start:] + seat_order[:start]
    return {real_id: slot for slot, real_id in enumerate(rotated, start=1)}


def encode_state(territories: Territories, players: Players, player_id: int, unit_norm: float = UNIT_NORM) -> torch.Tensor:
    """Builds the (NUM_TERRITORIES, STATE_WIDTH) egocentric board tensor for player_id: one-hot ownership
    rotated so that this player is always slot 1, an "unclaimed" flag, and the normalized unit count in
    the owner's units column."""
    rotation = rotate_player_ids(player_id, players)
    state    = torch.zeros(NUM_TERRITORIES, STATE_WIDTH)

    for name, info in territories.items():
        row   = TERRITORY_INDEX[name]
        owner = info["owned_by"]
        if owner == 0:
            state[row, UNCLAIMED_COL] = 1.0
        else:
            slot = rotation[owner] - 1  # 0-based column within each block
            state[row, slot] = 1.0
            state[row, UNITS_OFFSET + slot] = info["units"] / unit_norm
    return state



# ---------------------------------------------------------------------------
# GNN encoder
# ---------------------------------------------------------------------------

HIDDEN_DIM     = 32
NUM_GNN_LAYERS = 3


def _neighbour_sum_matrix(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """STRICT-neighbour adjacency (self-loops stripped out, 1.0 per neighbour) so `matrix @ x` gives the
    SUM of a node's neighbours' features, excluding the node itself."""
    neighbours_only = adjacency_matrix.clone()
    neighbours_only.fill_diagonal_(0.0)
    return neighbours_only


class GNNLayer(nn.Module):
    """One message-passing step: concat(self features, sum of neighbours' features) -> linear -> ReLU."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim * 2, out_dim)

    def forward(self, x: torch.Tensor, neighbour_matrix: torch.Tensor) -> torch.Tensor:
        neighbour_sum = neighbour_matrix @ x
        combined      = torch.cat([x, neighbour_sum], dim=-1)
        return F.relu(self.linear(combined))


class TerritoryGNN(nn.Module):
    """Stacks NUM_GNN_LAYERS GNNLayers and concatenates every layer's output (plus the original input)
    into one per-territory representation, so nothing gets averaged away across hops: 
    [state, 1-hop, 2-hop, 3-hop]."""

    def __init__(self, adjacency_matrix: torch.Tensor, input_dim: int = STATE_WIDTH, hidden_dim: int = HIDDEN_DIM, num_layers: int = NUM_GNN_LAYERS):
        super().__init__()
        # neighbour_matrix is part of the model's state, but not a trained parameter,
        # register it as a buffer to save/load with the checkpoint.
        self.register_buffer("neighbour_matrix", _neighbour_sum_matrix(adjacency_matrix)) 

        self.layers = nn.ModuleList()
        in_dim      = input_dim
        for _ in range(num_layers):
            self.layers.append(GNNLayer(in_dim, hidden_dim))
            in_dim = hidden_dim

        self.output_dim = input_dim + num_layers * hidden_dim

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (..., NUM_TERRITORIES, input_dim) -> (..., NUM_TERRITORIES, output_dim)."""
        features = [state]
        x = state
        for layer in self.layers:
            x = layer(x, self.neighbour_matrix)
            features.append(x)
        return torch.cat(features, dim=-1)



# ---------------------------------------------------------------------------
# Decision context (macro-action state)
# ---------------------------------------------------------------------------

def broadcast_concat(node_embeddings: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Appends `context` (a single (..., C) vector -- e.g. a previously selected territory's embedding, or
    a scalar like remaining reinforcement units) onto every one of the NUM_TERRITORIES rows of node_embeddings."""
    expand_shape = list(node_embeddings.shape[:-1]) + [context.shape[-1]]
    tiled        = context.unsqueeze(-2).expand(*expand_shape)
    return torch.cat([node_embeddings, tiled], dim=-1)


def gather_territory(node_embeddings: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """Pulls one territory's row out of (..., NUM_TERRITORIES, F) embeddings
    for broadcast_concat."""
    feature_dim = node_embeddings.shape[-1]
    lead_shape  = index.shape
    index = index.reshape(*lead_shape, 1, 1).expand(*lead_shape, 1, feature_dim)
    return node_embeddings.gather(-2, index).squeeze(-2)


# ---------------------------------------------------------------------------
# Decision heads
# ---------------------------------------------------------------------------

def mask_logits(logits: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Sets logits at invalid positions to -inf so they get exactly zero probability after softmax. valid_mask
    is a bool tensor the same shape as logits' last dim."""
    return logits.masked_fill(~valid_mask, float("-inf"))


# Width of one territory's embedding coming out of TerritoryGNN: the raw state
# row plus every layer's output, 11 + 3 * 32 = 107 (== TerritoryGNN.output_dim).
NODE_EMBED_DIM = STATE_WIDTH + NUM_GNN_LAYERS * HIDDEN_DIM

# Width each territory row (embedding + context) is compressed to inside a
# head, before the 14 rows are flattened into one vector.
HEAD_HIDDEN_DIM = 32

# Heads that allow skipping (attack_source, reposition_source) have one extra
# output after the 14 territories; this index means SKIP.
SKIP_INDEX = NUM_TERRITORIES

# Unit-count heads output NUM_COUNT_BINS bins spread evenly over the legal
# range [min_units, max_units], 1, 10%, ..., 100% of the max instead of 
# absolute counts, because stack sizes have no upper bound.
# See bin_to_unit_count.
NUM_COUNT_BINS = 11


class HeadSpec(NamedTuple):
    num_options:  int          # size of the head's output layer
    num_selected: int = 0      # previously selected territories appended to every row
    uses_units:  bool = False  # remaining-units scalar appended to every row


# One head per decision_type string the game passes to the Agent methods.
# The comments give the order the `selected` territories are passed in.
HEAD_SPECS: dict[str, HeadSpec] = {
    "placement":              HeadSpec(NUM_TERRITORIES, uses_units=True),    # units = placements left (incl. this one)
    "attack_source":          HeadSpec(NUM_TERRITORIES + 1),                 # 14 territories + SKIP
    "attack_target":          HeadSpec(NUM_TERRITORIES, num_selected=1),     # (source,)
    "continue_attack":        HeadSpec(2, num_selected=2),                   # (source, target); 0 = stop, 1 = continue
    "attack_occupy":          HeadSpec(NUM_COUNT_BINS, num_selected=2),      # (source, conquered)
    "reposition_source":      HeadSpec(NUM_TERRITORIES + 1),                 # 14 territories + SKIP
    "reposition_destination": HeadSpec(NUM_TERRITORIES, num_selected=1),     # (source,)
    "reposition_count":       HeadSpec(NUM_COUNT_BINS, num_selected=2),      # (source, destination)
    "reinforce_target":       HeadSpec(NUM_TERRITORIES, uses_units=True),    # units = remaining reinforcements
    "reinforce_count":        HeadSpec(NUM_COUNT_BINS, num_selected=1,       # (target,)
                                       uses_units=True),                     # units = remaining reinforcements
}

MAX_SELECTED = 2

# Game phase of every decision (the critic gets it as a one-hot).
PHASES = ("expansion", "combat", "repositioning", "reinforcement")

NUM_PHASES = 4
PHASE_INDEX = {
    "placement":              0, # expansion
    "attack_source":          1, # combat
    "attack_target":          1, # combat
    "continue_attack":        1, # combat
    "attack_occupy":          1, # combat
    "reposition_source":      2, # repositioning
    "reposition_destination": 2, # repositioning
    "reposition_count":       2, # repositioning
    "reinforce_target":       3, # reinforcement
    "reinforce_count":        3, # reinforcement
}

# Columns the critic appends to every territory row: phase one-hot, units still
# to place, and one marker column per territory selected earlier in the move.
VALUE_ROW_CONTEXT_DIM = 7


class _HeadBase(nn.Module):
    """Body shared by every head: each territory row (its embedding + the decision context)
    -> Linear, shared across territories -> ReLU -> flatten the 14 rows into one vector -> Linear."""

    def __init__(
        self,
        out_dim:       int,
        num_selected:  int = 0,
        uses_units:   bool = False,
        node_dim:      int = NODE_EMBED_DIM,
        hidden_dim:    int = HEAD_HIDDEN_DIM,
        output_gain: float = 1.0,
        extra_row_dim: int = 0,
    ):
        super().__init__()
        self.num_selected = num_selected
        self.uses_units   = uses_units
        in_dim            = node_dim * (1 + num_selected) + int(uses_units) + extra_row_dim
        self.perTerritory = nn.Linear(in_dim, hidden_dim)
        self.output       = nn.Linear(NUM_TERRITORIES * hidden_dim, out_dim)
        nn.init.orthogonal_(self.output.weight, gain=output_gain)
        nn.init.zeros_(self.output.bias)

    def _body(
        self,
        node_embeddings: torch.Tensor,
        selected: Sequence[int | torch.Tensor] = (),
        units: int | float | torch.Tensor | None = None,
        row_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """node_embeddings: (NUM_TERRITORIES, node_dim), or a batch (B, NUM_TERRITORIES, node_dim).
           selected: indices (TERRITORY_INDEX) of the previously selected territories,
                     in selection order, ints for a single state, (B,) long tensors for a batch.
           units: raw remaining-unit count, an int, or a (B,) tensor.
           row_features: (..., NUM_TERRITORIES, extra_row_dim) columns appended to the rows as they are (used by the critic).
        Returns (out_dim,) or (B, out_dim)."""
        if len(selected) != self.num_selected:
            raise ValueError(f"expected {self.num_selected} selected territories, got {len(selected)}")
        if (units is not None) != self.uses_units:
            raise ValueError("pass units exactly when the head's spec has uses_units=True")

        batch_shape = node_embeddings.shape[:-2]
        device      = node_embeddings.device
        context = [
            gather_territory(
                node_embeddings,
                torch.as_tensor(index, dtype=torch.long, device=device).expand(batch_shape), # as_tensor -> no copy if already a tensor
            )
            for index in selected
        ]
        if units is not None:
            units = torch.as_tensor(units, dtype=node_embeddings.dtype, device=device).expand(batch_shape)
            context.append((units / UNIT_NORM).unsqueeze(-1))

        x = broadcast_concat(node_embeddings, torch.cat(context, dim=-1)) if context else node_embeddings
        if row_features is not None:
            x = torch.cat([x, row_features], dim=-1)
        x = F.relu(self.perTerritory(x))             # (..., NUM_TERRITORIES, hidden_dim)
        return self.output(x.flatten(start_dim=-2))  # (..., out_dim)


class PolicyHead(_HeadBase):
    """One decision head. Returns a Categorical over its options with the invalid ones masked out (the softmax
    happens inside Categorical). During self-play sample from it and keep log_prob(action) for PPO; for greedy
    play take dist.probs.argmax(-1)."""

    def __init__(
        self,
        num_options:  int,
        num_selected: int = 0,
        uses_units:  bool = False,
        node_dim:     int = NODE_EMBED_DIM,
        hidden_dim:   int = HEAD_HIDDEN_DIM,
    ):
        super().__init__(num_options, num_selected, uses_units, node_dim, hidden_dim, output_gain=0.01)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        selected: Sequence[int | torch.Tensor] = (),
        units: int | float | torch.Tensor | None = None,
    ) -> Categorical:
        """valid_mask: bool (num_options,) or (B, num_options). None means every option is valid
        (continue_attack and the count heads)."""
        logits = self._body(node_embeddings, selected, units)
        if valid_mask is not None:
            logits = mask_logits(logits, valid_mask.to(logits.device))
        return Categorical(logits=logits)


class ValueHead(_HeadBase):
    """PPO critic on the same body: one scalar per state, no masking. The same board is worth different
    amounts at different points of a turn, so each territory row also gets (VALUE_ROW_CONTEXT_DIM columns):
      - the game phase, one-hot over PHASES (same on every row);
      - the units still to place: initial placements left, or reinforcements left this turn; 0 otherwise,
        normalized by UNIT_NORM (every row);
      - one marker column per territory chosen earlier in the current move (attack source/target, reposition
        source/destination, reinforce target): 1 on that territory's row, 0 elsewhere, all 0 when there's no
        such selection."""

    def __init__(self, node_dim: int = NODE_EMBED_DIM, hidden_dim: int = HEAD_HIDDEN_DIM):
        super().__init__(1, node_dim=node_dim, hidden_dim=hidden_dim, output_gain=1.0, extra_row_dim=VALUE_ROW_CONTEXT_DIM)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        phase: int | torch.Tensor,
        selected: Sequence[int | torch.Tensor] = (),
        units: int | float | torch.Tensor = 0,
    ) -> torch.Tensor:
        """node_embeddings: (NUM_TERRITORIES, node_dim) -> () value, or (B, NUM_TERRITORIES, node_dim) -> (B,) values.
           phase: index into NUM_PHASES, an int, or a (B,) tensor.
           selected: up to MAX_SELECTED territory indices in selection order, ints, or (B,) tensors where -1 means "no selection".
           units: raw count of units still to place (0 if none), int or (B,)."""
        batch_size    = node_embeddings.shape[:-2]
        device, dtype = node_embeddings.device, node_embeddings.dtype

        # expand(batch_size) broadcasts a single int (e.g. one phase for every state of position_values' batch) to (B,)
        phase     = torch.as_tensor(phase, dtype=torch.long, device=device).expand(batch_size)
        units     = torch.as_tensor(units, dtype=dtype, device=device).expand(batch_size)
        per_state = torch.cat([F.one_hot(phase, NUM_PHASES).to(dtype), (units / UNIT_NORM).unsqueeze(-1)], dim=-1)
        tiled     = per_state.unsqueeze(-2).expand(*batch_size, NUM_TERRITORIES, per_state.shape[-1])

        rows      = torch.arange(NUM_TERRITORIES, device=device)
        padded    = list(selected) + [-1] * (MAX_SELECTED - len(selected)) # [-1], so it does not match any index when there is no selection
        markers   = [
            (torch.as_tensor(index, dtype=torch.long, device=device).expand(batch_size).unsqueeze(-1) == rows)
            for index in padded
        ]                                                                        # each (..., 14) bool
        row_features = torch.cat([tiled, torch.stack(markers, dim=-1).to(dtype)], dim=-1)  # (..., 14, 7)
        return self._body(node_embeddings, row_features=row_features).squeeze(-1)


def build_policy_heads(node_dim: int = NODE_EMBED_DIM, hidden_dim: int = HEAD_HIDDEN_DIM) -> nn.ModuleDict:
    """All ten decision heads keyed by decision_type, so routing is just heads[decision_type]."""
    return nn.ModuleDict({
        decision: PolicyHead(spec.num_options, spec.num_selected, spec.uses_units, node_dim, hidden_dim)
        for decision, spec in HEAD_SPECS.items()
    }) # ModuleDict so the heads' parameters are registered by PyTorch


# ---------------------------------------------------------------------------
# Output conventions (game options <-> head output indices)
# ---------------------------------------------------------------------------

def territory_mask(options: list[str], allow_skip: bool = False) -> torch.Tensor:
    """Bool mask over a territory head's outputs, in TERRITORY_NAMES order, plus the SKIP slot at the end for
    heads that allow skipping."""
    mask = torch.zeros(NUM_TERRITORIES + int(allow_skip), dtype=torch.bool)
    for name in options:
        mask[TERRITORY_INDEX[name]] = True
    if allow_skip:
        mask[SKIP_INDEX] = True
    return mask


def index_to_territory(index: int) -> str | None:
    """Territory head output index -> territory name, or None for SKIP."""
    return None if index == SKIP_INDEX else TERRITORY_NAMES[index]


def bin_to_unit_count(bin_index: int, min_units: int, max_units: int) -> int:
    """Count head output bin -> unit count. Bin 0 is always min_units and the last bin always max_units ("everything"),
    whatever the stack size."""
    fraction = bin_index / (NUM_COUNT_BINS - 1)
    return min_units + int(fraction * (max_units - min_units) + 0.5) # + 0.5 for rounding


# ---------------------------------------------------------------------------
# Full network
# ---------------------------------------------------------------------------

class ConquestNet(nn.Module):
    """Shared GNN trunk feeding the ten decision heads and the critic."""

    def __init__(self, adjacency_matrix: torch.Tensor):
        super().__init__()
        self.gnn    = TerritoryGNN(adjacency_matrix)
        self.heads  = build_policy_heads(self.gnn.output_dim)
        self.critic = ValueHead(self.gnn.output_dim)


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = "rl_agent.pt"

# not sure if this is necessary, but it makes the checkpoint more robust to changes in the code
def network_architecture() -> dict:
    """Returns what a checkpoint's weights depend on: tensor shapes (including every head's spec), plus the state encoding
    and aggregation, which change what the weights mean without changing their shapes. Used as checkpoint metadata to
    detect mismatches."""
    return {
        "state_encoding":  "positional-units-v2",
        "aggregation":     "sum",
        "state_width":     STATE_WIDTH,
        "hidden_dim":      HIDDEN_DIM,
        "num_gnn_layers":  NUM_GNN_LAYERS,
        "head_hidden_dim": HEAD_HIDDEN_DIM,
        "num_count_bins":  NUM_COUNT_BINS,
        "value_context":   "phase+units+selection-markers-v1",
        "head_specs":      {name: list(spec) for name, spec in HEAD_SPECS.items()},
    }

# Training games still running after this many full rounds are stopped and scored by territory share.
MAX_ROUNDS = 60

# Stable integer id per decision type, used when batching transitions.
DECISION_ORDER       = list(HEAD_SPECS)
DECISION_ID          = {name: i for i, name in enumerate(DECISION_ORDER)}           # name -> decision id
PHASE_BY_DECISION_ID = torch.tensor([PHASE_INDEX[name] for name in DECISION_ORDER]) # name -> phase id
MAX_OPTIONS          = max(spec.num_options for spec in HEAD_SPECS.values())

# The game passes the territories chosen earlier in an action as `selected`
# (agents.SELECTED_TERRITORIES); each head takes exactly that many as context.
assert all(len(SELECTED_TERRITORIES.get(name, ())) == spec.num_selected for name, spec in HEAD_SPECS.items()), \
    "HEAD_SPECS num_selected out of sync with agents.SELECTED_TERRITORIES"


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    lr_decay:      float = 0.98  # multiplies the learning rate after every update() (one per training iteration)
    gamma:         float = 0.999 # close to 1: hundreds of decisions per player, reward only at the end
    gae_lambda:    float = 0.95  # bias-variance tradeoff for GAE: 0 = high bias, 1 = high variance
    clip_eps:      float = 0.2      
    value_coef:    float = 0.5
    entropy_coef:  float = 0.01
    epochs:          int = 4
    max_grad_norm: float = 0.5
    minibatch_size:  int = 512


class Transition(NamedTuple):
    """One recorded decision: everything update() needs to re-evaluate it."""
    state:    torch.Tensor     # (NUM_TERRITORIES, STATE_WIDTH), from the deciding player's view
    decision: int              # DECISION_ID
    selected: tuple[int, ...]  # territory indices of the head's context
    units:    int              # units still to place (placement / reinforcement), else 0
    mask:     torch.Tensor     # (num_options,) bool
    action:   int              # the action taken (head output index)
    log_prob: float            # for exp(logp_new − logp_old)
    value:    float            # for GAE


def final_reward(player_id: int, winner: int | None, territories: Territories, players: Players) -> float:
    """+1 for the winner, -1 for everyone else. In a game stopped at MAX_ROUNDS, eliminated players get -1 and survivors their
    territory share minus 0.5, so holding more land still beats holding less. Zero-sum for two players."""
    if winner is not None:
        return 1.0 if player_id == winner else -1.0
    if not players[player_id]["is_playing"]:
        return -1.0
    owned = sum(info["owned_by"] == player_id for info in territories.values())
    return owned / NUM_TERRITORIES - 0.5


class RLAgent(Agent):
    """PPO self-play agent behind the standard Agent interface.

    Stateless as a player: every decision is computed only from the call's arguments: the board, the territories chosen earlier
    in the same action (`selected`) and the units left to place, so one instance can play any number of seats and games. The only
    thing it keeps is, while recording, the decisions of the games being played, until finish_game() scores them.

    The network is built for one fixed map: the adjacency given here is baked into the GNN and saved with the checkpoint, so the
    adjacency passed with each decision isn't used.

    explore:   sample actions (training) instead of taking the argmax (play).
    recording: store every decision as a Transition for the next PPO update.
    """

    def __init__(self, adjacency: Adjacency | dict[str, list[str]], config: PPOConfig | None = None, explore: bool = False,
    ):
        self.adjacency = {name: set(neighbours) for name, neighbours in adjacency.items()}
        self.net       = ConquestNet(build_adjacency_matrix(self.adjacency))
        self.config    = config or PPOConfig()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.lr_decay)
        self.explore   = explore
        self.recording = False
        self.new_game()

    def new_game(self) -> None:
        """Drop decisions recorded in a previous game that was never scored."""
        self._trajectories: dict[int, list[Transition]] = defaultdict(list)


    # ----- Agent interface --------------------------------------------------
    def choose_territory(self, question: str, options: list[str], territories: Territories, adjacency: Adjacency,
                         players: Players, player_id: int, allow_skip: bool = False, decision_type: str = "",
                         selected: tuple[str, ...] = (), remaining_units: int | None = None) -> str | None:
        if len(options) == 1 and not allow_skip:
            return options[0]  # forced move: nothing to decide or learn from, the critic does not see these steps
        mask = territory_mask(options, allow_skip)
        return index_to_territory(self._act(territories, players, player_id, decision_type, mask, selected, remaining_units))

    def choose_unit_count(self, question: str, territories: Territories, adjacency: Adjacency, players: Players,
                          player_id: int, min_units: int, max_units: int, decision_type: str = "",
                          selected: tuple[str, ...] = ()) -> int:
        if min_units == max_units:
            return min_units
        mask      = torch.ones(NUM_COUNT_BINS, dtype=torch.bool)
        units     = max_units if decision_type == "reinforce_count" else None  # should double-check
        bin_index = self._act(territories, players, player_id, decision_type, mask, selected, units)
        return bin_to_unit_count(bin_index, min_units, max_units)

    def choose_yes_no(self, question: str, territories: Territories, adjacency: Adjacency, players: Players,
                      player_id: int, decision_type: str = "", selected: tuple[str, ...] = ()) -> bool:
        mask = torch.ones(2, dtype=torch.bool)
        return bool(self._act(territories, players, player_id, decision_type, mask, selected))


    # ----- forward pass (one decision) ----------------------------------------
    @torch.no_grad()
    def _act(self, territories: Territories, players: Players, player_id: int, decision_type: str,
             mask: torch.Tensor, selected_names: tuple[str, ...] = (), units: int | None = None) -> int:
        # validation
        if decision_type not in HEAD_SPECS:
            raise ValueError(f"RLAgent has no head for decision_type {decision_type!r}")
        spec = HEAD_SPECS[decision_type]
        if len(selected_names) != spec.num_selected:
            raise ValueError(f"{decision_type} needs selected={SELECTED_TERRITORIES.get(decision_type, ())}, "
                             f"got {selected_names!r}")
        selected = tuple(TERRITORY_INDEX[name] for name in selected_names)
        if spec.uses_units and units is None:
            raise ValueError(f"{decision_type} needs remaining_units")
        units = units or 0  # units still to place; 0 for decisions without any

        # inference pass
        state      = encode_state(territories, players, player_id)             # encode
        embeddings = self.net.gnn(state)                                       # GNN
        dist       = self.net.heads[decision_type](embeddings, mask, selected=selected, units=units if spec.uses_units else None) # head
        action     = dist.sample() if self.explore else dist.probs.argmax(-1)  # exploration vs. exploitation (sample or greedy)

        if self.recording:
            self._trajectories[player_id].append(Transition(
                state, DECISION_ID[decision_type], selected, units, mask,
                int(action), float(dist.log_prob(action)),
                float(self.net.critic(embeddings, PHASE_INDEX[decision_type], selected, units)),
            ))
        return int(action)


    # ----- critic-only scoring (map legend) ----------------------------------------
    @torch.no_grad()
    def position_values(self, territories: Territories, players: Players, phase: str) -> dict[int, float]:
        """The critic's estimate of the final reward (-1 = certain loss, +1 = certain win) for every player still in the game,
        each from their own point of view, at a moment in `phase` (one of PHASES) with no move in progress and no units waiting
        to be placed. Runs the GNN and the critic only, all players in one batch; no policy head is used and nothing is recorded.

        The critic learned from self-play, so the estimates assume everyone plays like this agent, and the critic was only
        trained on states where it was the evaluated player's turn to decide."""
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
        active = [pid for pid, player in players.items() if player["is_playing"]]
        if not active:
            return {}
        states = torch.stack([encode_state(territories, players, pid) for pid in active])
        values = self.net.critic(self.net.gnn(states), PHASES.index(phase))
        return dict(zip(active, values.tolist()))


    # ----- rollouts -------------------------------------------------------------
    def finish_game(self, winner: int | None, territories: Territories,
                    players: Players) -> list[tuple[Transition, float, float]]:
        """Scores the game just played and turns every player's recorded decisions into (transition, advantage, return) samples via
        GAE. The only reward is the final one, on each player's last decision."""
        gamma, lam = self.config.gamma, self.config.gae_lambda
        samples    = []
        for player_id, trajectory in self._trajectories.items():
            if not trajectory:
                continue
            reward     = final_reward(player_id, winner, territories, players)
            advantages = [0.0] * len(trajectory)
            advantage, next_value = 0.0, 0.0  # terminal: nothing after the last decision
            for t in reversed(range(len(trajectory))):
                r             = reward if t == len(trajectory) - 1 else 0.0
                delta         = r + gamma * next_value - trajectory[t].value
                advantage     = delta + gamma * lam * advantage
                advantages[t] = advantage
                next_value    = trajectory[t].value
            samples += [(tr, adv, adv + tr.value) for tr, adv in zip(trajectory, advantages)] # recheck adv + tr.value = return
        self._trajectories.clear()
        return samples


    # ----- backward pass (PPO update) ------------------------------------------
    def _evaluate(self, states, decisions, selected, units, masks, actions):
        """Batched forward pass with gradients: log-probs and entropies of the stored actions under the current policy, plus critic
        values. Rows are grouped by decision type so each head runs once per minibatch."""
        embeddings = self.net.gnn(states)                                 # one shared GNN pass for the whole batch
        phases     = PHASE_BY_DECISION_ID.to(decisions.device)[decisions] # ID of the game phase for every row
        values     = self.net.critic(embeddings, phases, tuple(selected[:, k] for k in range(MAX_SELECTED)), units)
        log_probs  = torch.zeros(len(states), device=states.device)
        entropy    = torch.zeros(len(states), device=states.device)
        for decision_id, name in enumerate(DECISION_ORDER):               # group rows by decision type and run each head once
            rows = (decisions == decision_id).nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                continue
            spec = HEAD_SPECS[name]
            dist = self.net.heads[name](
                embeddings[rows],
                masks[rows, :spec.num_options],
                selected=tuple(selected[rows, k] for k in range(spec.num_selected)),
                units=units[rows] if spec.uses_units else None,
            )
            log_probs = log_probs.index_put((rows,), dist.log_prob(actions[rows]))
            entropy   = entropy.index_put((rows,), dist.entropy())
        return log_probs, entropy, values

    def update(self, samples: list[tuple[Transition, float, float]]) -> dict[str, float]:
        """One PPO update (clipped policy loss + value loss - entropy bonus) over the samples of an iteration's games.
        Returns mean loss stats."""
        transitions, advantages, returns = zip(*samples)
        cfg = self.config
        n   = len(transitions)

        states    = torch.stack([t.state for t in transitions])
        decisions = torch.tensor([t.decision for t in transitions])
        selected  = torch.tensor([t.selected + (-1,) * (MAX_SELECTED - len(t.selected)) for t in transitions])
        units     = torch.tensor([t.units for t in transitions], dtype=torch.float)
        masks     = torch.zeros(n, MAX_OPTIONS, dtype=torch.bool)

        for i, t in enumerate(transitions):
            masks[i, :len(t.mask)] = t.mask

        actions       = torch.tensor([t.action for t in transitions])
        old_log_probs = torch.tensor([t.log_prob for t in transitions])
        returns       = torch.tensor(returns, dtype=torch.float)
        advantages    = torch.tensor(advantages, dtype=torch.float)

        # advantage normalisation
        if n > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        stats = defaultdict(list)
        for _ in range(cfg.epochs):
            for batch in torch.randperm(n).split(cfg.minibatch_size):
                log_probs, entropy, values = self._evaluate(
                    states[batch], decisions[batch], selected[batch], units[batch], masks[batch], actions[batch])
                log_ratio   = log_probs - old_log_probs[batch]
                ratio       = log_ratio.exp()
                adv         = advantages[batch]
                policy_loss = -torch.min(ratio * adv, ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv).mean()
                value_loss  = 0.5 * (returns[batch] - values).pow(2).mean()
                loss        = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy.mean()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                stats["policy_loss"].append(policy_loss.item())
                stats["value_loss"].append(value_loss.item())
                stats["entropy"].append(entropy.mean().item())
        self.scheduler.step()
        return {key: sum(values) / len(values) for key, values in stats.items()}


    # ----- training loop ----------------------------------------------------------
    def train(
        self,
        iterations: int = 200,
        games_per_iteration: int = 32,
        player_counts: Sequence[int] = (2, 3, 4, 5),
        max_rounds: int = MAX_ROUNDS,
        eval_every: int = 1,
        eval_games: int = 50,
        checkpoint_path: str | None = DEFAULT_CHECKPOINT,
    ) -> None:
        """Self-play PPO: every seat of every training game is this agent (one shared network, one trajectory per seat). Each iteration
        plays games_per_iteration games with a random player count, then runs one PPO update. Every eval_every iterations the greedy
        policy plays eval_games 2-player games against RandomAgent and a numbered checkpoint is saved next to checkpoint_path
        (rl_agent_iter0001.pt, ...); the final weights also go to checkpoint_path itself."""
        for iteration in range(1, iterations + 1):
            self.explore, self.recording = True, True
            samples, rounds, truncated   = [], 0, 0
            for _ in range(games_per_iteration):
                seats = ["rl"] * random.choice(player_counts)
                winner, territories, players, played = run_headless_game(seats, {"rl": self}, max_rounds)
                samples   += self.finish_game(winner, territories, players)
                rounds    += played
                truncated += winner is None
            self.recording = False

            stats = self.update(samples)
            print(f"iter {iteration:4d} | {len(samples):6d} decisions, {rounds / games_per_iteration:4.1f} rounds/game, "
                  f"{truncated / games_per_iteration:4.0%} hit the round cap | policy {stats['policy_loss']:+.4f}  "
                  f"value {stats['value_loss']:.4f}  entropy {stats['entropy']:.3f}  lr {self.scheduler.get_last_lr()[0]:.2e}", flush=True)

            if eval_every and iteration % eval_every == 0:
                print(f"           win rate vs RandomAgent (2 players, greedy): {self.evaluate(eval_games):.0%}", flush=True)
                if checkpoint_path:
                    # one file per iteration, numbered by the total updates so far (which carries over on
                    # --resume, through the scheduler state), e.g. rl_agent_iter0042.pt
                    stem, ext = os.path.splitext(checkpoint_path)
                    self.save(f"{stem}_iter{self.scheduler.last_epoch:04d}{ext}")
        if checkpoint_path:
            self.save(checkpoint_path)

    def evaluate(self, games: int = 50, num_players: int = 2, max_rounds: int = MAX_ROUNDS) -> float:
        """Win rate of the greedy policy against RandomAgent opponents, with the agent's seat randomized.
        Games that hit max_rounds count as losses."""
        explore, recording = self.explore, self.recording
        self.explore, self.recording = False, False
        agents = {"rl": self, "random": RandomAgent()}
        wins   = 0
        for _ in range(games):
            seats           = ["random"] * num_players
            my_seat         = random.randrange(num_players)
            seats[my_seat]  = "rl"
            winner, _, _, _ = run_headless_game(seats, agents, max_rounds)
            wins += winner == my_seat + 1
        self.explore, self.recording = explore, recording
        return wins / games


    # ----- persistence -----------------------------------------------------------------
    def save(self, path: str = DEFAULT_CHECKPOINT) -> None:
        """Saves network weights, optimizer state (to resume training), PPO config and the map the network was built for."""
        torch.save({
            "model":       self.net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "config":      asdict(self.config),
            "adjacency":   {name: sorted(neighbours) for name, neighbours in self.adjacency.items()},
            "architecture": network_architecture(),
        }, path)

    @classmethod
    def load(cls, path: str = DEFAULT_CHECKPOINT, explore: bool = False) -> "RLAgent":
        """Rebuilds a saved agent. explore=False (play) picks the most likely action at every decision;
        True samples, as in training."""
        checkpoint = torch.load(path, map_location="cpu")
        saved, current = checkpoint.get("architecture"), network_architecture()
        if saved != current:
            raise ValueError(
                f"{path} was trained with a different network (saved: {saved}, current code: {current}). "
                f"Retrain it with `python rl_agent.py`.")
        agent = cls(checkpoint["adjacency"], PPOConfig(**checkpoint["config"]), explore)
        agent.net.load_state_dict(checkpoint["model"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            agent.scheduler.load_state_dict(checkpoint["scheduler"])
        return agent


# ---------------------------------------------------------------------------
# Headless games (training / evaluation)
# ---------------------------------------------------------------------------
def run_headless_game(seat_models: list[str], agents: dict[str, Agent],
                      max_rounds: int = MAX_ROUNDS) -> tuple[int | None, Territories, Players, int]:
    """Plays one all-agent game with the real engine, console output silenced. seat_models[i] is the model of player i + 1.
    Returns (winner id, or None if stopped at max_rounds; final territories; players; rounds played)."""
    import conquest  # imported lazily: only training/evaluation needs the engine from here

    territories = conquest.build_territories()
    adjacency   = conquest.build_adjacency()
    players: Players = {
        pid: PlayerInfo(name=f"{model}-{pid}", is_agent=True, is_playing=True, model=model,
                        game_events=[], interaction_id=None)
        for pid, model in enumerate(seat_models, start=1)
    }
    for agent in set(agents.values()):
        if isinstance(agent, RLAgent):
            agent.new_game()

    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):     # silence the engine's print() calls
        conquest.assign_starting_territories(territories, players)
        conquest.run_initial_expansion_phase(territories, adjacency, players,
                                             conquest.STARTING_UNITS[len(players)], agents)
        for round_num in range(1, max_rounds + 1):
            conquest.run_attacking_phase(territories, adjacency, players, agents)
            conquest.check_player_status(territories, players)
            alive = [pid for pid, player in players.items() if player["is_playing"]]
            if len(alive) == 1:
                return alive[0], territories, players, round_num
            conquest.run_repositioning_phase(territories, adjacency, players, agents)
            conquest.run_reinforcement_phase(territories, adjacency, players, agents)
    return None, territories, players, max_rounds


# ---------------------------------------------------------------------------
# Entry point: python rl_agent.py [--iterations N] [--games N] [--resume]
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the Conquest RL agent by self-play (PPO).")
    parser.add_argument("--iterations", type=int, default=200, help="PPO iterations")
    parser.add_argument("--games", type=int, default=32, help="self-play games per iteration")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="where to save (and --resume from)")
    parser.add_argument("--resume", action="store_true", help="continue training from --checkpoint")
    args = parser.parse_args()

    torch.set_num_threads(1)  # the network is tiny: thread overhead costs more than it saves
    if args.resume:
        rl_agent = RLAgent.load(args.checkpoint)
    else:
        from conquest import build_adjacency
        rl_agent = RLAgent(build_adjacency())
    rl_agent.train(args.iterations, args.games, checkpoint_path=args.checkpoint)