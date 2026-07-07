import random
from typing import TypeAlias

territories_dict: TypeAlias = dict[str: [dict[str: int, str: int]]]
adjacency_dict:   TypeAlias = dict[str: [set[str]]]
players_dict:     TypeAlias = dict[int: [dict[str: str, str: bool, str: bool]]]

class Agent:
    """Agent parent class"""
    pass

# real AI agents with APIs will be implemented here

class RandomAgent(Agent):
    """Implements a mock interface as if it were an AI agent, but makes uniformly random valid choices, 
    with no API calls. Useful for automated testing of the game logic without spending API credits."""

    def choose_territory(self, state_description: str, options: list[str], allow_skip: bool=False, skip_meaning: str="skip") -> {str | None}:
        options = list(options)
        if allow_skip and random.random() < 1/(len(options)+1):
            return None
        return random.choice(options)

    def choose_unit_count(self, state_description: str, min_units: int, max_units: int):
        return random.randint(min_units, max_units)

    def choose_yes_no(self, state_description: str, question: str):
        return random.random() < 0.5


def describe_state_for_agent(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, player_id: int, situation: str) -> str:
    """Build a plain-text summary of the current game state from one
    player's point of view. Reused for every decision type so the agent
    always sees the board in a consistent format."""
    lines = [f"You are player {player_id} ({players[player_id]['name']})."]
    lines.append(situation)
    lines.append("Territories in the current game state (owner, units, neighbours):")
    for name, info in territories.items():
        owner = "unclaimed" if info["owned_by"] == 0 else f"player {info['owned_by']}"
        neighbours = ", ".join(sorted(t for t in adjacency[name] if t != name))
        lines.append(f"- {name}: owner={owner}, units={info['units']}, neighbours=[{neighbours}]")
    return "\n".join(lines)