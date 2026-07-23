import random
 
from agents import Agent, GeminiAgent, OpenAIAgent, RandomAgent, describe_state_for_agent
from game_types import Adjacency, PlayerInfo, Players, Territories
 
# ---------------------------------------------------------------------------
# Input validation helpers
# ---------------------------------------------------------------------------
 
 
def get_int_input(prompt: str, min_val: int, max_val: int) -> int:
    """Prompts until the player enters an integer within [min_val, max_val]."""
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print(f"'{raw}' is not a valid whole number. Please try again.")
            continue
        if value < min_val:
            print(f"Please enter a number that is at least {min_val}.")
            continue
        if value > max_val:
            print(f"Please enter a number that is at most {max_val}.")
            continue
        return value
 
 
def get_yes_no_input(prompt: str) -> bool:
    """Prompts until the player enters something recognizable as yes/no.
    Returns True for yes, False for no."""
    while True:
        raw = input(prompt).strip().lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")
 
 
def get_nonempty_string_input(prompt: str) -> str:
    """Prompts until the player enters a non-blank string."""
    while True:
        raw = input(prompt).strip()
        if raw:
            return raw
        print("Please enter a valid name.")
 
 
def get_choice_input(prompt: str, valid_choices: list[str]) -> str:
    """Prompts until the player enters one of valid_choices (case-insensitive
    match). Returns the matching entry with its original casing."""
    lookup = {choice.strip().lower(): choice for choice in valid_choices}
    while True:
        raw = input(prompt).strip().lower()
        if raw in lookup:
            return lookup[raw]
        print(f"'{raw}' is not one of the valid choices ({', '.join(valid_choices)}). Please try again.")
 
 
def prompt_territory(prompt: str, territories: Territories) -> str:
    """Prompts until the player enters a valid territory name."""
    while True:
        raw = input(prompt).strip()
        if raw not in territories:
            print("Enter a valid region.")
            continue
        return raw
 
 
def prompt_territory_or_cancel(prompt: str, territories: Territories, cancel_str: str = "X") -> str | None:
    """Prompts until the player enters a valid territory name, or the cancel
    string. Returns None if the player cancelled."""
    while True:
        raw = input(prompt).strip()
        if raw.upper() == cancel_str:
            return None
        if raw not in territories:
            print("Enter a valid region.")
            continue
        return raw
 
 
# ---------------------------------------------------------------------------
# Game setup
# ---------------------------------------------------------------------------
 
STARTING_UNITS = {
    2: 7,
    3: 5,
    4: 4,
    5: 3,
}
 
MIN_PLAYERS = min(STARTING_UNITS.keys())
MAX_PLAYERS = max(STARTING_UNITS.keys())
 
SUPPORTED_MODELS = ["gemini", "openai", "random"]
 
 
def build_territories() -> Territories:
    return {
        "Alaska":        {"owned_by": 0, "units": 0},
        "Great Lakes":   {"owned_by": 0, "units": 0},
        "North America": {"owned_by": 0, "units": 0},
        "South America": {"owned_by": 0, "units": 0},
        "Greenland":     {"owned_by": 0, "units": 0},
        "Europe":        {"owned_by": 0, "units": 0},
        "North Africa":  {"owned_by": 0, "units": 0},
        "South Africa":  {"owned_by": 0, "units": 0},
        "Arabia":        {"owned_by": 0, "units": 0},
        "West Asia":     {"owned_by": 0, "units": 0},
        "Central Asia":  {"owned_by": 0, "units": 0},
        "South Asia":    {"owned_by": 0, "units": 0},
        "Siberia":       {"owned_by": 0, "units": 0},
        "Australia":     {"owned_by": 0, "units": 0},
    }
 
 
def build_adjacency() -> Adjacency:
    return {
        "Alaska":        {"Alaska", "Great Lakes", "Siberia"},
        "Great Lakes":   {"Great Lakes", "Alaska", "Greenland", "North America"},
        "North America": {"North America", "Great Lakes", "South America"},
        "South America": {"South America", "North America", "North Africa"},
        "Greenland":     {"Greenland", "Great Lakes", "Europe"},
        "Europe":        {"Europe", "Greenland", "North Africa", "Arabia", "West Asia"},
        "North Africa":  {"North Africa", "South Africa", "South America", "Europe", "Arabia"},
        "South Africa":  {"South Africa", "North Africa"},
        "Arabia":        {"Arabia", "Europe", "North Africa", "West Asia", "South Asia"},
        "West Asia":     {"West Asia", "Europe", "Arabia", "South Asia", "Central Asia"},
        "Central Asia":  {"Central Asia", "West Asia", "South Asia", "Siberia"},
        "South Asia":    {"South Asia", "Arabia", "West Asia", "Central Asia", "Australia"},
        "Siberia":       {"Siberia", "Central Asia", "Alaska"},
        "Australia":     {"Australia", "South Asia"},
    }
 
 
def setup_players() -> Players:
    """Ask for the number of players and their details."""
    num_players = get_int_input(
        f"Number of players ({MIN_PLAYERS}-{MAX_PLAYERS}): ",
        min_val=MIN_PLAYERS,
        max_val=MAX_PLAYERS,
    )
 
    players: Players = {}
    for i in range(1, num_players + 1):
        name = get_nonempty_string_input(f"Name of player {i}: ")
        is_agent = get_yes_no_input(f"Is {name} AI? (y/n): ")
        if is_agent:
            options_str = ", ".join(SUPPORTED_MODELS)
            model = get_choice_input(f"Select model for {name} ({options_str}): ", SUPPORTED_MODELS)
        else:
            model = None
        players[i] = PlayerInfo(name=name, is_agent=is_agent, is_playing=True, model=model, game_events=[], interaction_id=None)
 
    return players
 
 
def build_agents(players: Players) -> dict[str, Agent]:
    """Construct exactly the agent backends actually needed for this game.
    A GeminiAgent is only built (and its API key only required) if some
    player actually selected the "gemini" model, so all-human or
    all-random games never need an API key."""
    selected_models = {p["model"] for p in players.values() if p["is_agent"]}
    agents: dict[str, Agent] = {}
    if "gemini" in selected_models:
        agents["gemini"] = GeminiAgent()
    if "openai" in selected_models:
        agents["openai"] = OpenAIAgent()
    if "random" in selected_models:
        agents["random"] = RandomAgent()
    return agents
 
 
def assign_starting_territories(territories: Territories, players: Players) -> None:
    """Give each player a random unclaimed starting territory."""
    unclaimed = [t for t, info in territories.items() if info["owned_by"] == 0]
    game_events = []
    for player_id in players:
        starting_region = random.choice(unclaimed)
        unclaimed.remove(starting_region)
        territories[starting_region] = {"owned_by": player_id, "units": 1}
        game_events.append(f"{players[player_id]['name']} ({player_id}) starts from {starting_region}.")
    for player_id in players:
        players[player_id]["game_events"] = game_events.copy()
    print("\n-------------------------------")
    print("\n".join(game_events))


# ---------------------------------------------------------------------------
# Map/status helpers
# ---------------------------------------------------------------------------

def print_map(territories: Territories, players: Players) -> None:
    player_colors = {0: "⚪", 1: "🔴", 2: "🔵", 3: "🟢", 4: "🟡", 5: "🟣"}

    alaska =        f"{player_colors[territories['Alaska']['owned_by']]} {territories['Alaska']['units']}"
    great_lakes =   f"{player_colors[territories['Great Lakes']['owned_by']]} {territories['Great Lakes']['units']}"
    north_america = f"{player_colors[territories['North America']['owned_by']]} {territories['North America']['units']}"
    south_america = f"{player_colors[territories['South America']['owned_by']]} {territories['South America']['units']}"
    greenland =     f"{player_colors[territories['Greenland']['owned_by']]} {territories['Greenland']['units']}"
    europe =        f"{player_colors[territories['Europe']['owned_by']]} {territories['Europe']['units']}"
    north_africa =  f"{player_colors[territories['North Africa']['owned_by']]} {territories['North Africa']['units']}"
    south_africa =  f"{player_colors[territories['South Africa']['owned_by']]} {territories['South Africa']['units']}"
    arabia =        f"{player_colors[territories['Arabia']['owned_by']]} {territories['Arabia']['units']}"
    west_asia =     f"{player_colors[territories['West Asia']['owned_by']]} {territories['West Asia']['units']}"
    central_asia =  f"{player_colors[territories['Central Asia']['owned_by']]} {territories['Central Asia']['units']}"
    south_asia =    f"{player_colors[territories['South Asia']['owned_by']]} {territories['South Asia']['units']}"
    siberia =       f"{player_colors[territories['Siberia']['owned_by']]} {territories['Siberia']['units']}"
    australia =     f"{player_colors[territories['Australia']['owned_by']]} {territories['Australia']['units']}"

    
    legend_lines = []
    for player_id in range(1, MAX_PLAYERS + 1):
        if player_id in players:
            text = (f"({player_id}) {player_colors[player_id]} {players[player_id]['name']}: "
                     f"{count_territories_owned(territories, player_id)}")
            line = text[:30].ljust(30)
        else:
            text = ""
            line = text.ljust(31)
        
        legend_lines.append(line)

    game_map = rf""" 	                                ..::::::'':::..'''''''''''''''::....
                          ....::: ':.:....::::.'':.::..	              .::                                    ....          :::::...            ...
                         .'..::::: .:.::''::::...     ''.    {greenland}   ...'                                 .:'''''   .. .''''       .::.......   :::::'
...   ..''''''''':....:''::..::.:::.:':.:::..'.:':.    '': Greenland:::'                 ...'''...   . ...::....:':::'''                   ''''     '''''''...''....
...''..::   {alaska}               '' ''  :...':. ...:':':  '.    .'''    .:':.           ..'  .    ':::::'         '''  '                              {siberia}          .:
   <- ':   Alaska..                    :'  ''':'..:''     '..:         ''       .   :'   :':                                                       Siberia :....''   ->
        '::':'     '':.           {great_lakes} '...  .:    ''.                    \   :.   ':::..:..:''                                {central_asia}             .''''''':  :
 .. .:  '            :''.     Great Lakes  '.:        '..                   .::::  .'''''                                  Central Asia       '.:      :.'    . .
                       '::.           .:...       ..:::: :                     :::'     {europe}                 {west_asia}                               .::     '
                         :              '':::...:::::'' ''                     .'      Europe .....:   ::  West Asia                          ..':.
                         :                  ''' .:                           .'   ..::''.: ...:....'  : '.                              .  .''  ::'
                         '.        {north_america}       .'                             '...:.... '' :: :...      '''                            ':.:.' ...':
                           ':  North America .'                              :        '..  .:   :'                                     :  '.:''''
                             ::.      ...''..:                              :            ''  '''.  {arabia}'.                {south_asia}           :
                              :.'.   :       '..                          .'                    ':Arabia :'.....      South Asia      .:'
           :.                    '.  :.  .:''':..                        :                       ':.      ':    '.        :       ::''
                                   ''..'':.   : ''''                     :            {north_africa}        ':.    ..'      :   ..'' ''.   :     .'.
                                       ''.:.     .                       :        North Africa     '::.::         '. :       ::. :     :''
                                           '::..''''''..                  :                            :           :':       ': ''    .'.::
                                              :         ':            /    ''...'''..                .'              '      '.::    .':   '
                                              :           :          /              '.          .   .'                       '.': .'  :
                                             :             '''..                     '.         :  :                           ''' '.'      ''::''.
                                             '.                 ''.                   :           :                              ':...          ''.:.
                                              '.                 :                     :   {south_africa}    :   .                                    .....'. ''
                                               '.     {south_america}      :                      South Africa   ::                                 .''   :' '.
                                                 :South America.'                     :          .'  .:'                             ..''           :
                                                 :          ..''                       :        .'   :'                             .'      {australia}     '.
                                                 :         .'                          '.      .'                                   :     Australia   :
                                                 :        .'                            :     :                                      :.   ..'.       :
                                                :      .''                               '''''                                         '''    ''..  .'      :..
{legend_lines[0]}                 :    .'                                                                                           ::        .:'
{legend_lines[1]}                 :   .'                                                                                             ''      ::
{legend_lines[2]}                 :  :                                                                                                      ''
{legend_lines[3]}                  '.:  ''
{legend_lines[4]}                    ''"""
    print(game_map)


def count_territories_owned(territories: Territories, player_id: int) -> int:
    return sum(1 for info in territories.values() if info["owned_by"] == player_id)


def is_valid_placement(territories: Territories, adjacency: Adjacency, player_id: int, region: str) -> bool:
    """A placement is valid if the region is unclaimed or already owned by the player, and it is adjacent to (or is) 
    a territory the player already owns."""
    if territories[region]["owned_by"] not in (0, player_id):
        return False
    for territory, info in territories.items():
        if info["owned_by"] == player_id and region in adjacency[territory]:
            return True
    return False


def is_attack_possible(territories: Territories, adjacency: Adjacency) -> bool:
    """Attack is possible for a player if they have more than one troops in a territory next to an enemy territory."""
    for territory, info in territories.items():
        if info["units"] > 1:
            for neighbour in adjacency[territory]:
                if territories[neighbour]["owned_by"] not in {0, info["owned_by"]}:
                    return True
    return False


def player_can_move(territories: Territories, player_id: int) -> bool:
    """A player can move if they have more than one troops in a region."""
    for info in territories.values():
        if info["units"] > 1 and info["owned_by"] == player_id:
            return True
    return False


def check_player_status(territories: Territories, players: Players) -> None:
    """Mark any player with zero territories as eliminated."""
    for player_id, player in players.items():
        if not player["is_playing"]:
            continue
        if count_territories_owned(territories, player_id) == 0:
            player["is_playing"] = False
            event = f"{player['name']} ({player_id}) has been eliminated!"
            broadcast_event(event, players)


def check_for_human_player(players: Players) -> bool:
    for id in players:
        if not players[id]["is_agent"] and players[id]["is_playing"]:
            return True
    return False


def broadcast_event(event: str, players: Players) -> None:
    for id in players:
        if players[id]["is_agent"]:
            players[id]["game_events"].append(event)
    print(event)


# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------

def do_battle(attacker: str, defender: str, territories: Territories, players: Players, adjacency: Adjacency, agent: Agent | None) -> None:
    """Resolve combat between attacker and defender by rolls until the attacker backs down, the defender is conquered,
    or the attacker no longer has enough units to continue."""
    attacking_player_id = territories[attacker]["owned_by"]
    attacking_player = players[attacking_player_id]
    defending_player_name = players[territories[defender]["owned_by"]]["name"]
 
    event = f"{attacking_player['name']} ({attacking_player_id}) attacks {defender} ({defending_player_name}) from {attacker}:"
    broadcast_event(event, players)

    keep_attacking = True
    while keep_attacking and territories[attacker]["units"] > 1 and territories[defender]["units"] > 0:
        attack_roll = random.randint(1, 6)
        defend_roll = random.randint(1, 6)
        att_event = f"- Attacker roll: {attack_roll}"
        def_event = f"- Defender roll: {defend_roll}"
        if players[attacking_player_id]["is_agent"]:
            players[attacking_player_id]["game_events"].append(att_event)
            players[attacking_player_id]["game_events"].append(def_event)
        else:
            print(att_event)
            print(def_event)

        if attack_roll > defend_roll:
            territories[defender]["units"] -= 1
        else:
            territories[attacker]["units"] -= 1
        event = f"- Remaining units: Attacker: {territories[attacker]["units"]}, Defender: {territories[defender]["units"]}"
        broadcast_event(event, players)

        if territories[defender]["units"] == 0:
            territories[defender]["owned_by"] = attacking_player_id
            max_occupy = territories[attacker]["units"] - 1
            if attacking_player["is_agent"]:
                question = f"You've conquered {defender} from {attacker}. Select the number of troops you send in."
                num_units = agent.choose_unit_count(question, players, attacking_player_id, min_units=1, max_units=max_occupy)
                players[attacking_player_id]["game_events"] = []
                territories[attacker]["units"] -= num_units
                territories[defender]["units"] += num_units
            else:
                num_units = get_int_input(f"{attacking_player["name"]} ({attacking_player_id}), number of units to occupy the region (1-{max_occupy}): ", min_val=1, max_val=max_occupy)
                territories[attacker]["units"] -= num_units
                territories[defender]["units"] += num_units
                print_map(territories, players)

            event = f"{defender} has been conquered by {players[attacking_player_id]["name"]} ({attacking_player_id}) with {num_units} units."
            broadcast_event(event, players)
            return

        if territories[attacker]["units"] == 1:
            event = f"{attacking_player["name"]} ({attacking_player_id}) has not enough units in {attacker} to maintain the attack."
            broadcast_event(event, players)
            return

        if attacking_player["is_agent"]:
            keep_attacking = agent.choose_yes_no("Do you want to continue the attack?", players, attacking_player_id)
            players[attacking_player_id]["game_events"] = []
        else:
            print_map(territories, players)
            keep_attacking = get_yes_no_input("Continue the attack? (y/n): ")

        if keep_attacking:
            broadcast_event(f"- {attacking_player["name"]} ({attacking_player_id}) continues the attack.", players)
        else:
            broadcast_event(f"- {attacking_player["name"]} ({attacking_player_id}) called off the attack.", players)


# ---------------------------------------------------------------------------
# Game phases
# ---------------------------------------------------------------------------

def run_initial_expansion_phase(territories: Territories, adjacency: Adjacency, players: Players, units_per_player: int, agents: dict[str, Agent]) -> None:
    broadcast_event("\n-------------------------------\n"
                     "    Initial expansion phase    \n"
                     "-------------------------------", players)

    for round_num in range(units_per_player):
        remaining = units_per_player - round_num
        for player_id, player in players.items():
            if player["is_agent"]:
                agent = agents[player["model"]]
                region = agent_choose_placement(territories, adjacency, players, player_id, remaining, agent)
                player["game_events"] = []
            else:
                print_map(territories, players)
                region = human_choose_placement(territories, adjacency, players, player_id, remaining)

            if territories[region]["owned_by"] == 0:
                event = f"{player["name"]} ({player_id}) claimed {region}."
                broadcast_event(event, players)
            else:
                event = f"{player["name"]} ({player_id}) reinforced {region}."
                broadcast_event(event, players)

            territories[region]["owned_by"] = player_id
            territories[region]["units"] += 1
    print_map(territories, players)


def human_choose_placement(territories: Territories, adjacency: Adjacency, players: Players, player_id: int, remaining: int) -> str:
    """Prompts a human for an initial expansion placement by choosing a territory."""
    player = players[player_id]
    while True:
        region = prompt_territory(
            f"{player['name']} ({player_id}), place a starting unit ({remaining} remaining): ", territories)
        if is_valid_placement(territories, adjacency, player_id, region):
            return region
        print("Please place a unit in one of your regions or an unoccupied neighbouring region.")


def agent_choose_placement(territories: Territories, adjacency: Adjacency, players: Players, player_id: int, remaining: int, agent: Agent) -> str:
    """Prompts an agent for an initial expansion placement by choosing a territory."""
    valid_options = [t for t in territories if is_valid_placement(territories, adjacency, player_id, t)]
    question = f"The game is in the initial expansion phase. Place a unit to a region you control or an unclaimed neighbouring one. You have {remaining} unit(s) remaining."
    state = describe_state_for_agent(territories, adjacency, players)
    return agent.choose_territory(state, question, valid_options, players, player_id)


def get_attacker_options(territories: Territories, adjacency: Adjacency, player_id: int) -> list[str]:
    """Territories the player could legally attack from right now (owned, >= 2 units, has at least one adjacent enemy territory)."""
    return [
        t for t, info in territories.items()
        if info["owned_by"] == player_id and info["units"] >= 2
        and any(territories[n]["owned_by"] not in (0, player_id) for n in adjacency[t])
    ]


def get_defender_options(territories: Territories, adjacency: Adjacency, attacker: str, player_id: int) -> list[str]:
    """Enemy territories adjacent to attacker."""
    return [n for n in adjacency[attacker] if territories[n]["owned_by"] not in (0, player_id)]


def human_handle_attack(territories: Territories, adjacency: Adjacency, players: Players, player_id: int, agent: Agent | None = None) -> bool:
    """Lets a human player either launch a single attack or skip their turn for this round.
    Returns True if done for this round (skipped), False if they completed an attack."""
    player = players[player_id]
    print_map(territories, players)
    while True:
        attacker = prompt_territory_or_cancel(
            f"{player['name']} ({player_id}), select a region to initiate an attack from or skip your attacking turn (type 'X'): ",
            territories
        )
        if attacker is None:
            event = f"{players[player_id]["name"]} ({player_id}) skipped their attacking turn."
            broadcast_event(event, players)
            return True

        if territories[attacker]["owned_by"] != player_id:
            print("Select a region you control.")
            continue
        if territories[attacker]["units"] < 2:
            print("Select a region with at least 2 units.")
            continue

        defender = prompt_territory_or_cancel(f"{player['name']} ({player_id}), select a region to attack from {attacker} ('X' - cancel): ", territories)
        if defender is None:
            continue
        if territories[defender]["owned_by"] in (0, player_id):
            print("Select an enemy region.")
            continue
        if defender not in adjacency[attacker]:
            print("Select a neighbouring region.")
            continue

        do_battle(attacker, defender, territories, players, adjacency, agent)
        check_player_status(territories, players)
        return False


def agent_handle_attack(territories: Territories, adjacency: Adjacency, players: Players, attacker_options: list[str], player_id: int, agent: Agent) -> bool:
    """Lets an agent player either launch a single attack or skip their turn for this round.
    Returns True if done for this round (skipped), False if they completed an attack."""

    question = f"Select a territory to initiate an attack from or skip your attacking turn."
    state = describe_state_for_agent(territories, adjacency, players)

    attacker = agent.choose_territory(state, question, attacker_options, players, player_id, allow_skip=True)
    players[player_id]["game_events"] = []
    if attacker is None:
        event = f"{players[player_id]["name"]} ({player_id}) skipped their attacking turn."
        broadcast_event(event, players)
        return True

    defender_options = get_defender_options(territories, adjacency, attacker, player_id)
    question = f"You chose to attack from {attacker}. Chose a territory to attack."
    state = describe_state_for_agent(territories, adjacency, players)
    defender = agent.choose_territory("", question, defender_options, players, player_id)
    players[player_id]["game_events"] = []

    do_battle(attacker, defender, territories, players, adjacency, agent)
    check_player_status(territories, players)
    return False


def run_attacking_phase(territories: Territories, adjacency: Adjacency, players: Players, agents: dict[str, Agent]) -> None:
    broadcast_event("\n-------------------------------\n"
                     "         Combat phase          \n"
                     "-------------------------------", players)
    
    while True:
        if not is_attack_possible(territories, adjacency):
            event = "No attack is possible."
            broadcast_event(event, players)
            break

        round_done = {}
        for player_id, player in players.items():
            if not player["is_playing"]:
                round_done[player_id] = True
                continue
            attacker_options = get_attacker_options(territories, adjacency, player_id)
            if not attacker_options:
                event = f"{player["name"]} ({player_id}) has no attacking possibility."
                broadcast_event(event, players)
                round_done[player_id] = True
                continue
            if player["is_agent"]:
                agent = agents[player["model"]]
                round_done[player_id] = agent_handle_attack(territories, adjacency, players, attacker_options, player_id, agent)
            else:
                round_done[player_id] = human_handle_attack(territories, adjacency, players, player_id)

        if all(round_done.values()):
            break


def get_reposition_source_options(territories: Territories, player_id: int) -> list[str]:
    return [t for t, info in territories.items() if info["owned_by"] == player_id and info["units"] >= 2]


def get_reposition_destination_options(territories: Territories, adjacency: Adjacency, region_from: str, player_id: int) -> list[str]:
    return [n for n in adjacency[region_from] if territories[n]["owned_by"] in (0, player_id)]


def apply_reposition(territories: Territories, region_from: str, region_to: str, num_units: int, player_id: int) -> None:
    if territories[region_to]["owned_by"] == 0:
        territories[region_to]["owned_by"] = player_id
    territories[region_from]["units"] -= num_units
    territories[region_to]["units"] += num_units


def human_reposition_turn(territories: Territories, adjacency: Adjacency, players: Players, player_id: int) -> None:
    player = players[player_id]
    while True:
        if not player_can_move(territories, player_id):
            event = f"{player['name']} ({player_id}) has no troops to reposition. Skipping turn."
            broadcast_event(event, players)
            return
        region_from = prompt_territory_or_cancel(
            f"{player['name']} ({player_id}), select a region to move units from or finish repositioning (type 'X'): ",
            territories
        )
        if region_from is None:
            event = f"{players[player_id]["name"]} ({player_id}) finished repositioning."
            broadcast_event(event, players)
            return

        if territories[region_from]["owned_by"] != player_id:
            print("Choose a region you control.")
            continue

        if territories[region_from]["units"] < 2:
            print("You need a region with at least 2 units.")
            continue

        region_to = prompt_territory(f"{player['name']} ({player_id}), select a region to move units to: ", territories)
        valid_destination = (region_to in adjacency[region_from] and territories[region_to]["owned_by"] in (0, player_id))
        if not valid_destination:
            print(f"{region_to} is not a valid destination. Choose a neutral or your own neighbouring region.")
            continue

        max_movable = territories[region_from]["units"] - 1
        num_units = get_int_input(f"Number of units to move (1-{max_movable}): ", min_val=1, max_val=max_movable)

        if territories[region_to]["owned_by"] == 0:
            event = f"{players[player_id]["name"]} ({player_id}) moved {num_units} units from {region_from} to {region_to}, claiming the territory."
        else:
            event = f"{players[player_id]["name"]} ({player_id}) moved {num_units} units from {region_from} to {region_to}."

        apply_reposition(territories, region_from, region_to, num_units, player_id)
        broadcast_event(event, players)
        print_map(territories, players)


def agent_reposition_turn(territories: Territories, adjacency: Adjacency, players: Players, player_id: int, agent: Agent) -> None:
    # Agents get a bounded number of reposition moves per turn -- unlike a
    # human, an agent has no natural "I've thought about it enough" stopping
    # point, so this caps how many decisions (and API calls) one turn can take.

    moves_remaining = 3
    while moves_remaining > 0:
        source_options = get_reposition_source_options(territories, player_id)
        if not source_options:
            event = f"{players[player_id]['name']} ({player_id}) has no troops to reposition. Skipping turn."
            broadcast_event(event, players)
            return

        question = f"Select a territory you want to move troops from or finish repositioning."
        state = describe_state_for_agent(territories, adjacency, players)
        region_from = agent.choose_territory(state, question, source_options, players, player_id, allow_skip=True)
        players[player_id]["game_events"] = []
        if region_from is None:
            event = f"{players[player_id]["name"]} ({player_id}) finished repositioning."
            broadcast_event(event, players)
            return

        destination_options = get_reposition_destination_options(territories, adjacency, region_from, player_id)

        question = f"You chose to move units from {region_from}. Select a territory you want to move troops to."
        region_to = agent.choose_territory("", question, destination_options, players, player_id)
        players[player_id]["game_events"] = []

        max_movable = territories[region_from]["units"] - 1
        question = f"You chose to move units from {region_from} to {region_to}. Select the number of units to move."
        num_units = agent.choose_unit_count(question, players, player_id, min_units=1, max_units=max_movable)
        players[player_id]["game_events"] = []

        if territories[region_to]["owned_by"] == 0:
            event = f"{players[player_id]["name"]} ({player_id}) moved {num_units} units from {region_from} to {region_to}, claiming the territory."
        else:
            event = f"{players[player_id]["name"]} ({player_id}) moved {num_units} units from {region_from} to {region_to}."

        apply_reposition(territories, region_from, region_to, num_units, player_id)
        moves_remaining -= 1
        broadcast_event(event, players)


def run_repositioning_phase(territories: Territories, adjacency: Adjacency, players: Players, agents: dict[str, Agent]) -> None:
    broadcast_event("\n-------------------------------\n"
                     "      Repositioning phase      \n"
                     "-------------------------------", players)

    for player_id, player in players.items():
        if not player["is_playing"]:
            continue
        if not player_can_move(territories, player_id):
            event = f"{player['name']} ({player_id}) has no troops to reposition. Skipping turn."
            broadcast_event(event, players)
            continue

        if player["is_agent"]:
            agent = agents[player["model"]]
            agent_reposition_turn(territories, adjacency, players, player_id, agent)
        else:
            print_map(territories, players)
            human_reposition_turn(territories, adjacency, players, player_id)


def human_reinforce_turn(territories: Territories, players: Players, player_id: int, reinforcements: int) -> None:
    player = players[player_id]
    while reinforcements > 0:
        region = prompt_territory(f"{player['name']} ({player_id}), select a region to reinforce ({reinforcements} troops remaining): ", territories)
        if territories[region]["owned_by"] != player_id:
            print("Select a region you control.")
            continue

        num_units = get_int_input(f"Number of reinforcement troops (1-{reinforcements}): ", min_val=1, max_val=reinforcements)
        territories[region]["units"] += num_units
        reinforcements -= num_units

        event = f"{players[player_id]["name"]} ({player_id}) reinforced {region} with {num_units} units."
        broadcast_event(event, players)
        print_map(territories, players)


def agent_reinforce_turn(territories: Territories, adjacency: Adjacency, players: Players, player_id: int, reinforcements: int, agent: Agent) -> None:
    owned_options = [t for t, info in territories.items() if info["owned_by"] == player_id]
    while reinforcements > 0:
        state = describe_state_for_agent(territories, adjacency, players)
        question = f"You have {reinforcements} reinforcement troops left to place. Select a region you want to fortify."
        region = agent.choose_territory(state, question, owned_options, players, player_id)
        players[player_id]["game_events"] = []

        question = f"You chose to reinforce {region}. Select the number of troops."
        num_units = agent.choose_unit_count(question, players, player_id, min_units=1, max_units=reinforcements)
        players[player_id]["game_events"] = []

        territories[region]["units"] += num_units
        reinforcements -= num_units
        event = f"{players[player_id]["name"]} ({player_id}) reinforced {region} with {num_units} units."
        broadcast_event(event, players)


def run_reinforcement_phase(territories: Territories, adjacency: Adjacency, players: Players, agents: dict[str, Agent]) -> None:
    broadcast_event("\n-------------------------------\n"
                     "      Reinforcement phase      \n"
                     "-------------------------------", players)

    for player_id, player in players.items():
        if not player["is_playing"]:
            continue

        reinforcements = count_territories_owned(territories, player_id)
        if player["is_agent"]:
            agent = agents[players[player_id]["model"]]
            agent_reinforce_turn(territories, adjacency, players, player_id, reinforcements, agent)
        else:
            print_map(territories, players)
            human_reinforce_turn(territories, players, player_id, reinforcements)


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def main():
    territories = build_territories()
    adjacency   = build_adjacency()

    players = setup_players()
    agents = build_agents(players)
    assign_starting_territories(territories, players)

    units_per_player = STARTING_UNITS[len(players)]
    run_initial_expansion_phase(territories, adjacency, players, units_per_player, agents)

    winner_name = None
    while winner_name is None:

        if not check_for_human_player(players):
            print_map(territories, players)

        run_attacking_phase(territories, adjacency, players, agents)
        check_player_status(territories, players)

        active_players = [p for p in players.values() if p["is_playing"]]
        if len(active_players) == 1:
            winner_name = active_players[0]["name"]
            break

        run_repositioning_phase(territories, adjacency, players, agents)
        run_reinforcement_phase(territories, adjacency, players, agents)

    print("Game over:")
    print(f"{winner_name} won!")


if __name__ == "__main__":
    main()