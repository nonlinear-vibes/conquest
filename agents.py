import random
import json
import random
from google import genai

from typing import TypeAlias

territories_dict: TypeAlias = dict[str: [dict[str: int, str: int]]]
adjacency_dict:   TypeAlias = dict[str: [set[str]]]
players_dict:     TypeAlias = dict[int: [dict[str: str, str: bool, str: bool]]]

class Agent:
    """Agent parent class"""
    pass

class GeminiAgent(Agent):
    """Wraps a google-genai client so the game logic is entirely separated."""
    pass

def describe_state_for_agent(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, player_id: int, situation: str, options: list[str], allow_skip: bool=False, skip_meaning: str="skip") -> str:
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
    lines.append(f"Valid choices: {', '.join(str(x) for x in options)}")
    return "\n".join(lines)