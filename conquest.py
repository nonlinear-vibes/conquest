import os
import random
from typing import TypeAlias

from agents import describe_state_for_agent, RandomAgent, Agent, GeminiAgent

territories_dict: TypeAlias = dict[str: [dict[str: int, str: int]]]
adjacency_dict:   TypeAlias = dict[str: [set[str]]]
players_dict:     TypeAlias = dict[int: [dict[str: str, str: bool, str: bool]]]

# ---------------------------------------------------------------------------
# Input validatior helpers
# ---------------------------------------------------------------------------


def get_int_input(prompt: str, min_val: int, max_val: int) -> int:
    """Prompts until the player enters an integer within [min_val, max_val]."""
    while True:
        raw = input(prompt)
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
        raw = input(prompt).lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Please answer 'y' or 'n'.")


def get_nonempty_string_input(prompt: str) -> str:
    """Prompts until the player enters a non-blank string as name."""
    while True:
        raw = input(prompt)
        if raw:
            return raw
        print("Please enter a valid name.")


def prompt_territory(prompt: str, valid_territories: list[str]) -> str:
    """Prompts until the player enters a valid territory name."""
    while True:
        raw = input(prompt)
        if raw not in valid_territories:
            print("Enter a valid region.")
            continue
        return raw


def prompt_territory_or_cancel(prompt: str, valid_territories: list[str], cancel_str: str="X") -> {str | None}:
    """Prompts until the player enters a valid territory name, or the cancel string.
    Returns 'None' if the user cancelled."""
    while True:
        raw = input(prompt)
        if raw == cancel_str:
            return None
        if raw not in valid_territories:
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
    5: 3
}

MIN_PLAYERS = min(STARTING_UNITS.keys())
MAX_PLAYERS = max(STARTING_UNITS.keys())


def build_territories() -> territories_dict:
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
        "Australia":     {"owned_by": 0, "units": 0}
    }


def build_adjacency() -> adjacency_dict:
    return {
        "Alaska":        {"Alaska", "Great Lakes", "North America", "Siberia"},
        "Great Lakes":   {"Great Lakes", "Alaska", "Greenland", "North America"},
        "North America": {"North America", "Great Lakes", "Alaska", "South America"},
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
        "Australia":     {"Australia", "South Asia"}
    }


def setup_players() -> players_dict:
    """Ask for the number of players and their details."""
    num_players = get_int_input(
        f"Number of players ({MIN_PLAYERS}-{MAX_PLAYERS}): ",
        min_val=MIN_PLAYERS,
        max_val=MAX_PLAYERS
    )

    players = {}
    for i in range(1, num_players + 1):
        name = get_nonempty_string_input(f"Name of player {i}: ")
        is_agent = get_yes_no_input(f"Is {name} AI? (y/n): ")
        players[i] = {"name": name, "is_agent": is_agent, "is_playing": True}

    return players


def assign_starting_territories(territories: territories_dict, players: players_dict):
    """Give each player a random unclaimed starting territory."""
    unclaimed = [t for t, info in territories.items() if info["owned_by"] == 0]
    for player_id in players:
        starting_region = random.choice(unclaimed)
        unclaimed.remove(starting_region)
        territories[starting_region] = {"owned_by": player_id, "units": 1}


# ---------------------------------------------------------------------------
# Map/status helpers
# ---------------------------------------------------------------------------

def print_map(territories, players):
    alaska =        f"{territories['Alaska']['owned_by']}: {territories['Alaska']['units']}"
    great_lakes =   f"{territories['Great Lakes']['owned_by']}: {territories['Great Lakes']['units']}"
    north_america = f"{territories['North America']['owned_by']}: {territories['North America']['units']}"
    south_america = f"{territories['South America']['owned_by']}: {territories['South America']['units']}"
    greenland =     f"{territories['Greenland']['owned_by']}: {territories['Greenland']['units']}"
    europe =        f"{territories['Europe']['owned_by']}: {territories['Europe']['units']}"
    north_africa =  f"{territories['North Africa']['owned_by']}: {territories['North Africa']['units']}"
    south_africa =  f"{territories['South Africa']['owned_by']}: {territories['South Africa']['units']}"
    arabia =        f"{territories['Arabia']['owned_by']}: {territories['Arabia']['units']}"
    west_asia =     f"{territories['West Asia']['owned_by']}: {territories['West Asia']['units']}"
    central_asia =  f"{territories['Central Asia']['owned_by']}: {territories['Central Asia']['units']}"
    south_asia =    f"{territories['South Asia']['owned_by']}: {territories['South Asia']['units']}"
    siberia =       f"{territories['Siberia']['owned_by']}: {territories['Siberia']['units']}"
    australia =     f"{territories['Australia']['owned_by']}: {territories['Australia']['units']}"

    legend_lines = []
    for player_id in range(1, MAX_PLAYERS + 1):
        if player_id in players:
            text = (f"{player_id}: {players[player_id]['name']}, "
                     f"{count_territories_owned(territories, player_id)}")
        else:
            text = ""
        line = text[:25].ljust(25)
        legend_lines.append(line)

    game_map = f""" 	                                ..::::::'':::..'''''''''''''''::....
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
{legend_lines[0]}                       :    .'                                                                                           ::        .:'
{legend_lines[1]}                       :   .'                                                                                             ''      ::
{legend_lines[2]}                       :  :                                                                                                      ''
{legend_lines[3]}                        '.:  ''
{legend_lines[4]}                          ''"""
    print(game_map)


def count_territories_owned(territories: territories_dict, player_id: int) -> int:
    return sum(1 for info in territories.values() if info["owned_by"] == player_id)


def is_valid_placement(territories: territories_dict, adjacency: adjacency_dict, player_id: int, region: str) -> bool:
    """A placement is valid if the region is unclaimed or already owned by the player, and it is adjacent to (or is) 
    a territory the player already owns."""
    if territories[region]["owned_by"] not in (0, player_id):
        return False
    for territory, info in territories.items():
        if info["owned_by"] == player_id and region in adjacency[territory]:
            return True
    return False


def is_attack_possible(territories: territories_dict, adjacency: adjacency_dict) -> bool:
    """Attack is possible for a player if they have more than one troops in a territory next to an enemy territory."""
    for territory, info in territories.items():
        if info["units"] > 1:
            for neighbour in adjacency[territory]:
                if territories[neighbour]["owned_by"] not in {0, info["owned_by"]}:
                    return True
    return False


def player_can_move(territories: territories_dict, player_id: int) -> bool:
    """A player can move if they have more than one troops in a region."""
    for info in territories.values():
        if info["units"] > 1 and info["owned_by"] == player_id:
            return True
    return False


def check_player_status(territories: territories_dict, players: players_dict) -> bool:
    """Mark any player with zero territories as eliminated."""
    for player_id, player in players.items():
        if not player["is_playing"]:
            continue
        if count_territories_owned(territories, player_id) == 0:
            player["is_playing"] = False
            print(f"{player['name']} ({player_id}) has been eliminated!")


# ---------------------------------------------------------------------------
# Combat
# ---------------------------------------------------------------------------

def do_battle(attacker: str, defender: str, territories: territories_dict, players: players_dict, adjacency: adjacency_dict, agent: {Agent | None}) -> None:
    """Resolve combat between attacker and defender by rolls until the attacker backs down, the defender is conquered,
    or the attacker no longer has enough units to continue."""
    attacking_player_id = territories[attacker]["owned_by"]
    attacking_player = players[attacking_player_id]

    print(f"{attacking_player["name"]} ({attacking_player_id}) attacks {defender} from {attacker}.")

    keep_attacking = True
    while keep_attacking and territories[attacker]["units"] > 1 and territories[defender]["units"] > 0:
        attack_roll = random.randint(1, 6)
        defend_roll = random.randint(1, 6)
        print(f"Attacker roll: {attack_roll}")
        print(f"Defender roll: {defend_roll}")

        if attack_roll > defend_roll:
            territories[defender]["units"] -= 1
        else:
            territories[attacker]["units"] -= 1

        if territories[defender]["units"] == 0:
            territories[defender]["owned_by"] = attacking_player_id
            max_occupy = territories[attacker]["units"] - 1
            if attacking_player["is_agent"]:
                situation = f"You've just conquered {defender} from {attacker}. Select the number of troops you send in."
                state = describe_state_for_agent(territories, adjacency, players, attacking_player_id, situation, range(1, max_occupy+1))
                num_units = agent.choose_unit_count(state, min_units=1, max_units=max_occupy)
            else:
                num_units = get_int_input(
                    f"{attacking_player["name"]} ({attacking_player_id}), number of units to occupy the region (1-{max_occupy}): ",
                    min_val=1,
                    max_val=max_occupy
                )
                print_map(territories, players)
            print(f"{defender} has been conquered with {num_units} units.")
            territories[attacker]["units"] -= num_units
            territories[defender]["units"] += num_units
            return

        if territories[attacker]["units"] == 1:
            print(f"{attacking_player["name"]} ({attacking_player_id}) has not enough units in {attacker} to maintain the attack.")
            return

        if attacking_player["is_agent"]:
            situation = f"The game is in the combat phase, you are attacking {defender} from {attacker}. Do you want to continue the attack?"
            state = describe_state_for_agent(territories, adjacency, players, attacking_player_id, situation, ["yes", "no"])
            keep_attacking = agent.choose_yes_no(state)
        else:
            print_map(territories, players)
            keep_attacking = get_yes_no_input("Continue the attack? (y/n): ")

        if keep_attacking:
            print(f"{attacking_player["name"]} ({attacking_player_id}) continues the attack.")
        else:
            print(f"{attacking_player["name"]} ({attacking_player_id}) called off the attack.")


# ---------------------------------------------------------------------------
# Game phases
# ---------------------------------------------------------------------------

def run_initial_expansion_phase(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, units_per_player: int, agent: {Agent | None}):
    print("")
    print("-------------------------------")
    print("    Initial expansion phase    ")
    print("-------------------------------")

    for round_num in range(units_per_player):
        remaining = units_per_player - round_num
        for player_id, player in players.items():
            if player["is_agent"]:
                region = agent_choose_placement(territories, adjacency, players, player_id, remaining, agent)
            else:
                print_map(territories, players)
                region = human_choose_placement(territories, adjacency, players, player_id, remaining)
            if territories[region]["owned_by"] == 0:
                print(f"{player["name"]} ({player_id}) claimed {region}.")
            else:
                print(f"{player["name"]} ({player_id}) reinforced {region}.")
            territories[region]["owned_by"] = player_id
            territories[region]["units"] += 1
    print_map(territories, players)


def human_choose_placement(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, player_id: int, remaining: int) -> str:
    """Prompts a human for an initial expansion placement by choosing a territory."""
    player = players[player_id]
    while True:
        region = prompt_territory(
            f"{player['name']} ({player_id}), place a starting unit ({remaining} remaining): ",
            territories
        )
        if is_valid_placement(territories, adjacency, player_id, region):
            return region
        print("Please place a unit in one of your regions or an unoccupied neighbouring region.")


def agent_choose_placement(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, player_id: int, remaining: int, agent: Agent) -> str:
    """Prompts an agent for an initial expansion placement by choosing a territory."""
    valid_options = [t for t in territories if is_valid_placement(territories, adjacency, player_id, t)]
    situation = f"The game is in the initial expansion phase, place a unit to a region you control or an unclaimed neighbouring one. You have {remaining} unit(s) remaining."
    state = describe_state_for_agent(territories, adjacency, players, player_id, situation, valid_options, valid_options)
    return agent.choose_territory(state, valid_options)


def get_attacker_options(territories: territories_dict, adjacency: adjacency_dict, player_id: int) -> list[str]:
    """Territories the player could legally attack from right now (owned, >= 2 units, has at least one adjacent enemy territory)."""
    return [
        t for t, info in territories.items()
        if info["owned_by"] == player_id and info["units"] >= 2
        and any(territories[n]["owned_by"] not in (0, player_id) for n in adjacency[t])
    ]


def get_defender_options(territories: territories_dict, adjacency: adjacency_dict, attacker: str, player_id: int) -> list[str]:
    """Enemy territories adjacent to attacker."""
    return [n for n in adjacency[attacker] if territories[n]["owned_by"] not in (0, player_id)]


def human_handle_attack(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, player_id: int, agent: None) -> bool:
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
            return True

        if territories[attacker]["owned_by"] != player_id:
            print("Select a region you control.")
            continue
        if territories[attacker]["units"] < 2:
            print("Select a region with at least 2 units.")
            continue

        defender = prompt_territory_or_cancel(
            f"{player['name']} ({player_id}), select a region to attack from {attacker} ('X' - cancel): ",
            territories
        )
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


def agent_handle_attack(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, player_id: int, agent: Agent) -> bool:
    """Lets an agent player either launch a single attack or skip their turn for this round.
    Returns True if done for this round (skipped), False if they completed an attack."""
    attacker_options = get_attacker_options(territories, adjacency, player_id)
    if not attacker_options:
        print(f"{players[player_id]["name"]} ({player_id}) has no attacking possibility.")
        return True

    situation = f"The game is in the combat phase, it is your turn to attack. Select a territory to initiate an attack from or skip your attacking turn."
    state = describe_state_for_agent(territories, adjacency, players, player_id, situation, attacker_options, True, "skip attacking turn")
    attacker = agent.choose_territory(state, attacker_options, allow_skip=True, skip_meaning="skip attacking for this round")
    if attacker is None:
        print(f"{players[player_id]["name"]} ({player_id}) skipped their attacking turn.")
        return True

    defender_options = get_defender_options(territories, adjacency, attacker, player_id)
    situation = f"The game is in the combat phase, you chose to attack from {attacker}. Chose a territory to attack."
    state = describe_state_for_agent(territories, adjacency, players, player_id, situation, defender_options)
    defender = agent.choose_territory(state, defender_options)

    do_battle(attacker, defender, territories, players, adjacency, agent)
    check_player_status(territories, players)
    return False


def run_attacking_phase(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, agent: {Agent | None}):
    print("")
    print("-----------------------")
    print("    Combat phase    ")
    print("-----------------------")
    while True:
        if not is_attack_possible(territories, adjacency):
            print("No attack is possible.")
            break

        round_done = {}
        for player_id, player in players.items():
            if not player["is_playing"]:
                round_done[player_id] = True
                continue
            if not player_can_move(territories, player_id):
                print(f"{player['name']} ({player_id}) has not enough troops to attack. Skipping turn.")
                round_done[player_id] = True
                continue
            if player["is_agent"]:
                round_done[player_id] = agent_handle_attack(territories, adjacency, players, player_id, agent)
            else:
                round_done[player_id] = human_handle_attack(territories, adjacency, players, player_id, agent)

        if all(round_done.values()):
            break


def get_reposition_source_options(territories: territories_dict, player_id: int) -> list[str]:
    return [t for t, info in territories.items() if info["owned_by"] == player_id and info["units"] >= 2]


def get_reposition_destination_options(territories: territories_dict, adjacency: adjacency_dict, region_from: str, player_id: int) -> list[str]:
    return [n for n in adjacency[region_from] if territories[n]["owned_by"] in (0, player_id)]


def apply_reposition(territories: territories_dict, region_from: str, region_to: str, num_units: int, player_id: int):
    if territories[region_to]["owned_by"] == 0:
        territories[region_to]["owned_by"] = player_id
    territories[region_from]["units"] -= num_units
    territories[region_to]["units"] += num_units


def human_reposition_turn(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, player_id: int):
    player = players[player_id]
    while True:
        region_from = prompt_territory_or_cancel(
            f"{player['name']} ({player_id}), select a region to move units from or finish repositioning (type 'X'): ",
            territories
        )
        if region_from is None:
            return

        if territories[region_from]["owned_by"] != player_id:
            print("Choose a region you control.")
            continue
        if territories[region_from]["units"] < 2:
            print("You need a region with at least 2 units.")
            continue

        region_to = prompt_territory(
            f"{player['name']} ({player_id}), select a region to move units to: ",
            territories
        )
        valid_destination = (
            region_to in adjacency[region_from]
            and territories[region_to]["owned_by"] in (0, player_id)
        )
        if not valid_destination:
            print(f"{region_to} is not a valid destination. Choose a neutral or your own neighbouring region.")
            continue

        max_movable = territories[region_from]["units"] - 1
        num_units = get_int_input(
            f"Number of units to move (1-{max_movable}): ",
            min_val=1,
            max_val=max_movable
        )

        apply_reposition(territories, region_from, region_to, num_units, player_id)
        print_map(territories, players)


def agent_reposition_turn(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, player_id: int, agent: Agent):
    while True:
        source_options = get_reposition_source_options(territories, player_id)
        if not source_options:
            return

        situation = f"The game is in the repositioning phase. Select a territory you want to move troops from or finish repositioning."
        state = describe_state_for_agent(territories, adjacency, players, player_id, situation, source_options, True, "finish repositioning")
        region_from = agent.choose_territory(
            state, source_options, allow_skip=True, skip_meaning="finish repositioning"
        )
        if region_from is None:
            print(f"{players[player_id]["name"]} ({player_id}) finished repositioning.")
            return

        destination_options = get_reposition_destination_options(territories, adjacency, region_from, player_id)

        situation = f"The game is in the repositioning phase. You chose to move units from {region_from}. Select a territory you want to move troops to."
        state = describe_state_for_agent(territories, adjacency, players, player_id, situation, destination_options)
        region_to = agent.choose_territory(state, destination_options)

        max_movable = territories[region_from]["units"] - 1
        situation = f"The game is in the repositioning phase. You chose to move units from {region_from} to {region_to}. Select the number of units to move."
        state = describe_state_for_agent(territories, adjacency, players, player_id, situation, range(1, max_movable+1))
        num_units = agent.choose_unit_count(state, min_units=1, max_units=max_movable)

        print(f"{players[player_id]["name"]} ({player_id}) moved {num_units} units from {region_from} to {region_to}.")
        apply_reposition(territories, region_from, region_to, num_units, player_id)


def run_repositioning_phase(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, agent: {Agent | None}):
    print("")
    print("---------------------------")
    print("    Repositioning phase    ")
    print("---------------------------")

    for player_id, player in players.items():
        if not player["is_playing"]:
            continue
        if not player_can_move(territories, player_id):
            print(f"{player['name']} ({player_id}) has no troops to reposition. Skipping turn.")
            continue

        if player["is_agent"]:
            agent_reposition_turn(territories, adjacency, players, player_id, agent)
        else:
            print_map(territories, players)
            human_reposition_turn(territories, adjacency, players, player_id)


def human_reinforce_turn(territories: territories_dict, players: players_dict, player_id: int, reinforcements: int):
    player = players[player_id]
    while reinforcements > 0:
        region = prompt_territory(
            f"{player['name']} ({player_id}), select a region to reinforce "
            f"({reinforcements} troops remaining): ",
            territories
        )
        if territories[region]["owned_by"] != player_id:
            print("Select a region you control.")
            continue

        num_units = get_int_input(
            f"Number of reinforcement troops (1-{reinforcements}): ",
            min_val=1,
            max_val=reinforcements
        )
        territories[region]["units"] += num_units
        reinforcements -= num_units
        print_map(territories, players)


def agent_reinforce_turn(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, player_id: int, reinforcements: int, agent: Agent):
    owned_options = [t for t, info in territories.items() if info["owned_by"] == player_id]
    while reinforcements > 0:
        situation = f"The game is in the reinforcement phase. You have {reinforcements} reinforcement troops left to place. Select a region you want to reinforce."
        state = describe_state_for_agent(territories, adjacency, players, player_id, situation, owned_options)
        region = agent.choose_territory(state, owned_options)

        situation = f"The game is in the reinforcement phase. You have choosen to reinforce {region}. Select the number of troops."
        state = describe_state_for_agent(territories, adjacency, players, player_id, situation, range(1, reinforcements+1))
        num_units = agent.choose_unit_count(state, min_units=1, max_units=reinforcements)
        territories[region]["units"] += num_units
        reinforcements -= num_units
        
        print(f"{players[player_id]["name"]} ({player_id}) reinforced {region} with {num_units} units.")


def run_reinforcement_phase(territories: territories_dict, adjacency: adjacency_dict, players: players_dict, agent: {Agent | None}):
    print("")
    print("---------------------------")
    print("    Reinforcement phase    ")
    print("---------------------------")

    for player_id, player in players.items():
        if not player["is_playing"]:
            continue

        reinforcements = count_territories_owned(territories, player_id)
        if player["is_agent"]:
            agent_reinforce_turn(territories, adjacency, players, player_id, reinforcements, agent)
        else:
            print_map(territories, players)
            human_reinforce_turn(territories, players, player_id, reinforcements)


# ---------------------------------------------------------------------------
# Main game loop
# ---------------------------------------------------------------------------

def main():
    territories = build_territories()
    adjacency = build_adjacency()

    players = setup_players()
    assign_starting_territories(territories, players)

    agent = RandomAgent()

    units_per_player = STARTING_UNITS[len(players)]
    run_initial_expansion_phase(territories, adjacency, players, units_per_player, agent)

    winner_name = None
    while winner_name is None:
        run_attacking_phase(territories, adjacency, players, agent)
        check_player_status(territories, players)

        active_players = [p for p in players.values() if p["is_playing"]]
        if len(active_players) == 1:
            winner_name = active_players[0]["name"]
            break

        run_repositioning_phase(territories, adjacency, players, agent)
        run_reinforcement_phase(territories, adjacency, players, agent)

    print("Game over:")
    print(f"{winner_name} won!")


if __name__ == "__main__":
    main()