"""Shared type definitions for Conquest."""

from typing import TypedDict


class TerritoryInfo(TypedDict):
    owned_by: int
    units: int


class PlayerInfo(TypedDict):
    name: str
    is_agent: bool
    is_playing: bool
    model: str | None           # e.g. "gemini", "random"; None for human players
    game_events: list[str]      # events queued up since this player's last turn
    interaction_id: str | None  # Gemini Interactions API conversation handle


Territories = dict[str, TerritoryInfo]
Adjacency = dict[str, set[str]]
Players = dict[int, PlayerInfo]
