import os
import json
import random
from abc import ABC, abstractmethod

from game_types import Adjacency, Players, Territories

class Agent(ABC):
    """Common interface every player-controlling agent must implement.
    Every decision in the game reduces to one of these three primitives."""
 
    @abstractmethod
    def choose_territory(
        self,
        state_description: str,
        question: str,
        options: list[str],
        players: Players,
        player_id: int,
        allow_skip: bool = False,
    ) -> str | None:
        """Pick one territory name from options. Returns None only if
        allow_skip is True and the agent chooses to skip."""
 
    @abstractmethod
    def choose_unit_count(
        self,
        question: str,
        players: Players,
        player_id: int,
        min_units: int,
        max_units: int,
    ) -> int:
        """Pick a whole number of units in [min_units, max_units]."""
 
    @abstractmethod
    def choose_yes_no(self, question: str, players: Players, player_id: int) -> bool:
        """Answer a yes/no question."""


def get_system_prompt(name: str, player_id: int) -> str:
    return f'''You are {name} (Player ID: {player_id}), a highly strategic AI competitor playing a simplified version of the board game Risk.

Your ultimate objective is absolute global domination: eliminate all other players by capturing all their territories.

---

### THE RULES

### 1. Map & Territories
* There are 14 territories. 
* A territory is either **unclaimed**, or **owned** by a specific player with a certain number of units (troops) stationed on it.
* You must ALWAYS leave at least 1 unit behind in any territory you move units out of (whether attacking, repositioning, or occupying). You can never leave a territory vacant.

---

### GAME PHASES
The game loop cycles through three distinct phases:

## Phase I: Combat (Attack or Skip)
* Combat is a multi-turn sequence where players take turns either starting a battle or skipping. If you skip, you can still attack later if an opponent acts and opens up a new opportunity. Combat ends when each player run out of attacking possibilities or skip their turn.
* You can only attack from a territory you own with **at least 2 units** into an **adjacent** enemy-owned territory.
* The Dice Mechanics:
  * Battles are resolved unit-by-unit via 1-to-6 dice rolls. 
  * If the attacker rolls higher, the defender loses 1 unit.
  * If the defender rolls higher or equal, the attacker loses 1 unit (defenders win ties).
* A battle continues until either the defender is wiped out, or the attacker has only 1 unit left. However, you can choose to call off an ongoing attack at any point.

### Phase II: Repositioning
* You have one turn to move units between your adjacent territories. You may also move units into adjacent **unclaimed** territories to occupy them.
* Once you declare you are finished repositioning, your turn ends.

### Phase III: Reinforcement
* You receive a pool of new units to place. 
* The number of reinforcement units you get is equal to the number of territories you currently control. 
* You must allocate all of your reinforcement units to your controlled territories before passing your turn.

---

### INPUT & OUTPUT FORMAT
You will be provided with:
1. Recent **Game Events** (what happened while you were "asleep").
2. The current **Board State** (who owns what and where).
3. A specific **Decision** to make.
4. The set of **Valid Options**.

**Output Constraint:** You must choose exactly one of the provided valid options. Return your choice in the requested JSON schema format. Do not include any markdown formatting or text outside of the JSON block.'''


class GeminiAgent(Agent):
    """Wraps a google-genai client so the game logic is entirely separated from prompts, schemas, and the Gemini SDK. 
    Uses the Interactions API (previous_interaction_id) so each player's agent retains memory of its own game across calls."""

    def __init__(self, model: str = "gemini-2.5-flash", show_thoughts: bool = False, thinking_level: str = "low"):
        from dotenv import load_dotenv
        from google import genai

        load_dotenv()
        api_key = os.environ.get("API_KEY")
        if api_key is None:
            raise RuntimeError("API key not found. Set API_KEY in your environment or .env file.")


        self.client = genai.Client(api_key=api_key)
        self.model  = model
        self.thinking_level = thinking_level
        self.show_thoughts  = show_thoughts

    def _generate(self, prompt_text: str, schema: dict, name: str, player_id: int, interaction_id: str | None):
        response = self.client.interactions.create(
            model = self.model,
            input = prompt_text,
            response_format    = schema,
            system_instruction = get_system_prompt(name, player_id),
            previous_interaction_id = interaction_id,
            generation_config  = {
            "thinking_level": self.thinking_level, # Options: "low", "medium", "high"
            "thinking_summaries": "auto"           # Requests the model to include internal reasoning
            }
        )
        if self.show_thoughts:
            for step in response.steps[:-1]:
                print(f"\n{name}: {step.summary[0].text}\n")
        return response

    
    def choose_territory(self, state_description: str, question: str, options: list[str], players: Players, player_id: int, allow_skip: bool = False) -> str | None:
        """Asks the agent to pick one territory name from the options. Returns
        None if allow_skip is True and the agent chooses to skip."""

        options = list(options)
        enum_values = options + (["SKIP"] if allow_skip else [])
 
        schema = {
            "type": "object",
            "properties": {"territory": {"type": "string", "enum": enum_values}},
            "required": ["territory"],
        }

        prompt = build_prompt(players, player_id, enum_values, state_description, question)
        response = self._generate(prompt, schema, players[player_id]["name"], player_id, players[player_id]["interaction_id"])
        players[player_id]["interaction_id"] = response.id

        result = json.loads(response.output_text)
        if result["territory"] == "SKIP":
            return None
        return result["territory"]

    def choose_unit_count(self, question: str, players: Players, player_id: int, min_units: int, max_units: int) -> int:
        """Ask the agent to pick a whole number of units in [min_units, max_units]."""

        enum_values = [str(n) for n in range(min_units, max_units + 1)]
        schema = {
            "type": "object",
            "properties": {"units": {"type": "integer", "minimum": min_units, "maximum": max_units}},
            "required": ["units"],
        }

        prompt = build_prompt(players, player_id, enum_values, "", question)
        response = self._generate(prompt, schema, players[player_id]["name"], player_id, players[player_id]["interaction_id"])
        players[player_id]["interaction_id"] = response.id

        result = json.loads(response.output_text)
        return result["units"]

    def choose_yes_no(self, question: str, players: Players, player_id: int) -> bool:
        """Ask the agent a yes/no question. Returns True or False."""

        enum_values = ["yes (True)", "no (False)"]
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "boolean"}},
            "required": ["answer"],
        }
 
        prompt = build_prompt(players, player_id, enum_values, "", question)
        response = self._generate(prompt, schema, players[player_id]["name"], player_id, players[player_id]["interaction_id"])
        players[player_id]["interaction_id"] = response.id

        result = json.loads(response.output_text)
        return result["answer"]


class RandomAgent(Agent):
    """Implements a mock interface as if it were an AI agent, but makes uniformly random valid choices, with no API calls. 
    Useful for automated testing of the game logic without spending API credits, and as a rules-only opponent."""
    
    def choose_territory(self, state_description: str, question: str, options: list[str], players: Players, player_id: int, allow_skip: bool = False) -> str | None:
        options = list(options)
        if allow_skip and random.random() < 1 / (len(options) + 1):
            return None
        return random.choice(options)

    def choose_unit_count(self, question: str, players: Players, player_id: int, min_units: int, max_units: int) -> int:
        return random.randint(min_units, max_units)


    def choose_yes_no(self, question: str, players: Players, player_id: int) -> bool:
        return random.random() < 0.5


def describe_state_for_agent(territories: Territories, adjacency: Adjacency, players: Players) -> str:
    """Build a plain-text summary of the current game state from one
    player's point of view. Reused for every decision type so the agent
    always sees the board in a consistent format."""

    lines = ["### Current game state (owner, units, neighbours):"]
    for name, info in territories.items():
        owner = "0 (unclaimed)" if info["owned_by"] == 0 else f"{players[info['owned_by']]['name']} (player {info['owned_by']})"
        neighbours = ", ".join(sorted(t for t in adjacency[name] if t != name))
        lines.append(f"- {name}: owner = {owner}, units = {info['units']}, neighbours = [{neighbours}]")
    lines.append("")
    return "\n".join(lines)


def build_prompt(players: Players, player_id: int, enum_values: list[str], state_description: str, question: str) -> str:
    game_events = players[player_id]["game_events"]
    if game_events:
        events_list = "\n".join(f"- {event}" for event in game_events)
        events_formatted = f"### Game events since your last turn:\n{events_list}\n"
    else:
        events_formatted = ""
    options_formatted = ", ".join(enum_values)
    return f"""
{events_formatted}
{state_description}
{question}
 
### Legal choices: {options_formatted}"""
