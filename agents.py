import random
import os
import json
from google import genai
from dotenv import load_dotenv

from typing import TypeAlias

territories_dict: TypeAlias = dict[str: [dict[str: int, str: int]]]
adjacency_dict:   TypeAlias = dict[str: [set[str]]]
players_dict:     TypeAlias = dict[int: [dict[str: str, str: bool, str: bool]]]

class Agent:
    """Agent parent class"""
    pass

SYSTEM_PROMPT = """You are an AI player in a simplified version of the board game Risk.

RULES:
- The world has 14 territories, each owned by a player and holding some number of units.
- After an initial expansion phase where players take turns placing starting units and reinforcing or occupying territories,
  each turn of the game loop consists of phases: battle phase, repositioning phase and reinforcement phase. Players take turn
  in order of their player numbers.
- In combat phase, players take turns in starting an attack or skipping their turn. This means if you leave a territory with
  too few units after a battle, an opponent may immediately attack it, but so can you in the opposite situation. Battle phase
  continues until no more attacking move is possible or each player has skipped their last attacking chance.
- You can only attack from a territory you own with at least 2 units, into an adjacent territory owned by another player.
  The outcome of a battle is decided by random-generated numbers from 1 to 6. If the attacker's roll is higher, the defender
  loses a unit in the attacked territory, otherwise the attacker loses a unit in the attacking territory. This means in case 
  of equal rolls, the defender wins, and thus has a small advantage. Battle continues until the attacker has only one unit 
  left in the attacking territory or the defender lost all their units in the attacked territory. The attacker can also call
  off the attack after each roll.
- In repositioning phase, each player has one turn to reposition units between their territories as many times as they want,
  but once they finish repositioning, the next player's turn begins and they will not get another turn to reposition. Players
  can also occupy unclaimed territories in this phase by moving units into them.
- In reinforcement phase, each player can place reinforcement units in territories they control. The number of reinforcements
  each player can allocate is equal to the number of territories they control. Each player has one turn to allocate all their
  reinforcements before passing the turn to the next player.
- You must always leave at least 1 unit behind in a territory you move units out of.
- Your overall goal is to eliminate all other players by capturing every territory.

You will always be given the current game state and a specific decision to make, along with the exact set of valid choices.
Respond only in the requested JSON format -- do not include any other text."""


class AgentError(Exception):
    """Raised for problems talking to the underlying model provider."""


class GeminiAgent(Agent):
    """Wraps a google-genai client so the game logic is entirely separated."""

    def __init__(self, model="gemini-2.5-flash", show_thoughts=True, thinking_level="low"):
        load_dotenv()
        api_key = os.environ.get("API_KEY")

        self.client = genai.Client(api_key=api_key)
        self.model  = model
        self.thinking_level = thinking_level
        self.show_thoughts  = show_thoughts

    def _generate(self, input, schema, name):
        response = self.client.interactions.create(
            model = self.model,
            input = input,
            response_format  = schema,
            generation_config={
            "thinking_level": "medium",        # Options: "low", "medium", "high"
            "thinking_summaries": "auto"       # Requests the model to include internal reasoning
            }
        )
        if self.show_thoughts:
            for i in range(len(response.steps)-1):
                print(f"\n{name}: {response.steps[i].summary[0].text}\n")
        return response

    def choose_territory(self, state_description, options, name, allow_skip=False):
        """Ask the agent to pick one territory name from the options. Returns
        None if allow_skip is True and the agent chooses to skip."""
        options = list(options)
        enum_values = options + (["SKIP"] if allow_skip else [])
        schema = {
            "type": "object",
            "properties": {"territory": {"type": "string", "enum": enum_values}},
            "required": ["territory"],
        }
        response = self._generate(state_description, schema, name)
        response = json.loads(response.output_text)
        return response["territory"]

    def choose_unit_count(self, name, state_description, min_units, max_units):
        """Ask the agent to pick a whole number of units in [min_units, max_units]."""
        schema = {
            "type": "object",
            "properties": {"units": {"type": "integer", "minimum": min_units, "maximum": max_units}},
            "required": ["units"],
        }
        response = self._generate(state_description, schema, name)
        response = json.loads(response.output_text)
        return response["units"]


    def choose_yes_no(self, state_description, name):
        """Ask the agent a yes/no question. Returns True or False."""
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "boolean"}},
            "required": ["answer"],
        }
        response = self._generate(state_description, schema, name)
        response = json.loads(response.output_text)
        return response["answer"]

class RandomAgent(Agent):
    """Implements a mock interface as if it were an AI agent, but makes uniformly random valid choices, 
    with no API calls. Useful for automated testing of the game logic without spending API credits."""

    def choose_territory(self, state_description: str, options: list[str], allow_skip: bool=False, skip_meaning: str="skip") -> {str | None}:
        options = list(options)
        print(state_description)
        if allow_skip and random.random() < 1/(len(options)+1):
            return None
        return random.choice(options)

    def choose_unit_count(self, state_description: str, min_units: int, max_units: int):
        print(state_description)
        return random.randint(min_units, max_units)

    def choose_yes_no(self, state_description: str):
        print(state_description)
        return random.random() < 0.5


def describe_state_for_agent(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, player_id: int, situation: str, options: list[str], allow_skip: bool=False, skip_meaning: str="skip") -> str:
    """Build a plain-text summary of the current game state from one
    player's point of view. Reused for every decision type so the agent
    always sees the board in a consistent format."""
    lines = [f"You are player {player_id} ({players[player_id]['name']})."]
    lines.append(situation)
    lines.append("Territories in the current game state (owner, units, neighbours):")
    for name, info in territories.items():
        owner = "0 (unclaimed)" if info["owned_by"] == 0 else f"{players[info['owned_by']]["name"]} (player {info['owned_by']})"
        neighbours = ", ".join(sorted(t for t in adjacency[name] if t != name))
        lines.append(f"- {name}: owner={owner}, units={info['units']}, neighbours=[{neighbours}]")
    lines.append(f"Legal choices: {', '.join(str(x) for x in options)}")
    return "\n".join(lines)