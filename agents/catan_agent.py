"""Catan Tournament 1v1 agent.

How a decision is made (details in team/README.md):

1. BoardModel      static facts about this game's board, built once per game.
2. OpponentTracker logic over the public log: what the rival must be holding.
3. TurnSituation   one decision's view: my network, legal spots, roads, robber.
4. Planner         classical planning over resources: each build has preconditions
                   (its cost, a legal place) and effects (VP, production, reach).
                   Maritime trades are searched as steps toward those effects.
5. Evaluator       afterstate value V(s) = structure(s) + hand(s), in VP-hundredths.
6. Q-table         learned offline by Q-learning over a small abstraction
                   (state_features x action kind); added as a bonus.

The judge gives each player only 5 seconds for the WHOLE game, so nothing here
simulates the engine: every decision is a few small loops over 54 nodes.
"""
from __future__ import annotations

import time

# ---------------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------------

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}
COSTS = {
    "ROAD": {"WOOD": 1, "BRICK": 1},
    "SETTLEMENT": {"WOOD": 1, "BRICK": 1, "SHEEP": 1, "WHEAT": 1},
    "CITY": {"WHEAT": 2, "ORE": 3},
    "DEVELOPMENT_CARD": {"SHEEP": 1, "WHEAT": 1, "ORE": 1},
}
DEV_DECK = {"KNIGHT": 14, "YEAR_OF_PLENTY": 2, "ROAD_BUILDING": 2, "MONOPOLY": 2, "VICTORY_POINT": 5}
LONGEST_ROAD_MINIMUM = 5
LARGEST_ARMY_MINIMUM = 3
DISCARD_LIMIT = 7
SEVEN_BEFORE_MY_NEXT_TURN = 1 - (5 / 6) ** 2  # rival's roll, then mine

# Below this much clock, the O(1) baseline answers: losing on time loses the game.
CLOCK_RESERVE_S = 0.4

# Every value is in hundredths of a victory point, so one VP = 100.
# Hand-set first, then refined by hill climbing (training/tune.py) on paired seeds.
WEIGHTS = {
    "vp": 100.0,
    "win": 10000.0,
    "pip": 3.202,                 # one pip of production of an average resource
    "missing_resource": 5.743,    # penalty per resource I produce nothing of
    "expansion_spot": 0.55,     # share of the best reachable spot's value
    "expansion_far": 0.29,      # share of the best spot one road further away
    "road_length": 2.604,
    "card": 4.288,                # one resource card in hand
    "goal_city": 34.0,
    "goal_settlement": 33.007,
    "goal_development": 10.413,
    "knight_in_hand": 12.943,
    "progress_card_in_hand": 19.86,
    "knight_played": 6.027,
    "robber_pip": 3.0,
    "steal": 8.0,
    "port_specific": 0.671,       # per pip of the port's resource I produce
    "port_generic": 3.951,
    "q_bonus": 0.1,             # chosen by training/compare.py: no worse, best head-to-head
}

# Placement favours ore and wheat a little: 7 VP rewards cities, which need them.
PLACEMENT_RESOURCE_WEIGHT = {"WOOD": 1.232, "BRICK": 1.0, "SHEEP": 0.829, "WHEAT": 1.267, "ORE": 1.1}
PLAY_RESOURCE_WEIGHT = {"WOOD": 1.0, "BRICK": 1.0, "SHEEP": 0.85, "WHEAT": 1.1, "ORE": 1.05}


def _key(tile) -> tuple:
    return tuple(tile)


def _edge(edge) -> tuple:
    a, b = edge
    return (a, b) if a < b else (b, a)


def _pay(hand: tuple, cost: dict):
    """The hand after paying `cost`, or None when the precondition fails."""
    paid = list(hand)
    for index, resource in enumerate(RESOURCES):
        paid[index] -= cost.get(resource, 0)
        if paid[index] < 0:
            return None
    return tuple(paid)


def _hand_tuple(resources: dict) -> tuple:
    return tuple(int(resources.get(resource, 0)) for resource in RESOURCES)


# ---------------------------------------------------------------------------------
# 1. Board model: static per game
# ---------------------------------------------------------------------------------


class BoardModel:
    """Adjacency, production and ports. The board never changes within a game."""

    def __init__(self, board: dict):
        self.tiles = {}
        self.node_tiles = {}
        self.neighbours = {}
        self.edges = set()
        for tile in board["tiles"]:
            coordinate = _key(tile["tile"])
            self.tiles[coordinate] = (tile["resource"], tile["number"], tuple(tile["nodes"]))
            for node in tile["nodes"]:
                self.node_tiles.setdefault(node, []).append(coordinate)
            for a, b in tile["edges"]:
                self.edges.add(_edge((a, b)))
                self.neighbours.setdefault(a, set()).add(b)
                self.neighbours.setdefault(b, set()).add(a)
        self.nodes = sorted(self.node_tiles)
        self.port_of = {}
        for port in board["ports"]:
            for node in port["nodes"]:
                self.port_of[node] = port["resource"]
        self.scarcity = self._scarcity()

    def _scarcity(self) -> dict:
        """Rare resources are worth more: weight = average supply / this supply."""
        supply = {resource: 0 for resource in RESOURCES}
        for resource, number, _ in self.tiles.values():
            if resource is not None:
                supply[resource] += PIPS.get(number, 0)
        average = sum(supply.values()) / len(RESOURCES)
        return {r: max(0.7, min(1.6, (average / supply[r]) ** 0.5)) if supply[r] else 1.6 for r in RESOURCES}

    def node_production(self, node: int, robber=None) -> dict:
        """Pips per resource that a settlement on `node` would collect."""
        production = {}
        for coordinate in self.node_tiles.get(node, ()):
            if coordinate == robber:
                continue
            resource, number, _ = self.tiles[coordinate]
            if resource is not None:
                production[resource] = production.get(resource, 0) + PIPS.get(number, 0)
        return production

    @staticmethod
    def signature(board: dict) -> tuple:
        return tuple((tuple(t["tile"]), t["resource"], t["number"]) for t in board["tiles"])


# ---------------------------------------------------------------------------------
# 2. Opponent tracker: inference from public events
# ---------------------------------------------------------------------------------


class OpponentTracker:
    """Replays the public log to estimate the rival's hand.

    Certain: production from every roll (we know its buildings and the robber),
    every build cost, maritime trades, Year of Plenty, cards we stole or lost.
    Uncertain: discarded cards and cards stolen when we did not take part; those are
    spread in proportion. The public card total is ground truth, so the estimate is
    rescaled to it after each update.
    """

    def __init__(self, board: BoardModel):
        self.board = board
        self.cursor = 0
        self.hand = {resource: 0.0 for resource in RESOURCES}
        self.settlements, self.cities = set(), set()
        self.robber = next((c for c, (resource, _, _) in board.tiles.items() if resource is None), None)
        self.rolled = False
        self.initial_settlements = 0
        self.free_roads = 0

    def update(self, log: list, card_total: int) -> None:
        for entry in log[self.cursor:]:
            self._apply(entry)
        self.cursor = len(log)
        self._rescale(card_total)

    def _apply(self, entry: dict) -> None:
        kind, theirs = entry["type"], entry["player"] == "opponent"
        if kind == "ROLL":
            self.rolled = True
            self._produce(sum(entry["dice"]))
        elif kind == "MOVE_ROBBER":
            self.robber = _key(entry["tile"])
            self._steal(entry, theirs)
        elif not theirs:
            if kind == "PLAY_MONOPOLY":
                self.hand[entry["resource"]] = 0.0
            return
        elif kind == "BUILD_SETTLEMENT":
            self.settlements.add(entry["node"])
            if not self.rolled:
                self.initial_settlements += 1
                if self.initial_settlements == 2:  # the second one pays its tiles
                    for resource, pips in self.board.node_production(entry["node"]).items():
                        self.hand[resource] += 1
                return
            self._spend(COSTS["SETTLEMENT"])
        elif kind == "BUILD_CITY":
            self.settlements.discard(entry["node"])
            self.cities.add(entry["node"])
            self._spend(COSTS["CITY"])
        elif kind == "BUILD_ROAD":
            if not self.rolled:
                return
            if self.free_roads:
                self.free_roads -= 1
                return
            self._spend(COSTS["ROAD"])
        elif kind == "BUY_DEVELOPMENT_CARD":
            self._spend(COSTS["DEVELOPMENT_CARD"])
        elif kind == "MARITIME_TRADE":
            self.hand[entry["give"]] -= entry["give_count"]
            self.hand[entry["receive"]] += 1
        elif kind == "PLAY_YEAR_OF_PLENTY":
            for resource in entry["resources"]:
                self.hand[resource] += 1
        elif kind == "PLAY_MONOPOLY":
            # Our own loss is not logged; the rescale to the public total corrects it.
            self.hand[entry["resource"]] += 2.0
        elif kind == "PLAY_ROAD_BUILDING":
            self.free_roads = 2
        elif kind == "DISCARD":
            self._lose_unknown(entry["count"])
        elif kind == "END_TURN":
            self.free_roads = 0

    def _produce(self, number: int) -> None:
        if number == 7:
            return
        for coordinate, (resource, tile_number, nodes) in self.board.tiles.items():
            if tile_number != number or coordinate == self.robber or resource is None:
                continue
            for node in nodes:
                if node in self.settlements:
                    self.hand[resource] += 1
                elif node in self.cities:
                    self.hand[resource] += 2

    def _steal(self, entry: dict, theirs: bool) -> None:
        stolen = entry.get("stolen_resource")
        if entry.get("stole_from") is None:
            return
        if theirs and stolen:          # they robbed me: I know the card
            self.hand[stolen] += 1
        elif not theirs and stolen:    # I robbed them: I know the card
            self.hand[stolen] -= 1
        elif not theirs:
            self._lose_unknown(1)

    def _spend(self, cost: dict) -> None:
        for resource, count in cost.items():
            self.hand[resource] -= count

    def _lose_unknown(self, count: int) -> None:
        total = sum(max(0.0, v) for v in self.hand.values())
        if total <= 0:
            return
        for resource in RESOURCES:
            self.hand[resource] -= count * max(0.0, self.hand[resource]) / total

    def _rescale(self, card_total: int) -> None:
        for resource in RESOURCES:
            self.hand[resource] = max(0.0, self.hand[resource])
        total = sum(self.hand.values())
        if card_total <= 0:
            self.hand = {resource: 0.0 for resource in RESOURCES}
        elif total <= 0:
            self.hand = {resource: card_total / len(RESOURCES) for resource in RESOURCES}
        else:
            self.hand = {resource: value * card_total / total for resource, value in self.hand.items()}


# ---------------------------------------------------------------------------------
# Per-game memory. The judge keeps the module alive between calls.
# ---------------------------------------------------------------------------------


class GameMemory:
    """Holds the board model and tracker for the game in progress."""

    def __init__(self):
        self.signature = None
        self.board = None
        self.tracker = None

    def refresh(self, observation: dict):
        signature = BoardModel.signature(observation["board"])
        restarted = self.tracker is not None and len(observation["log"]) < self.tracker.cursor
        if signature != self.signature or restarted:
            self.signature = signature
            self.board = BoardModel(observation["board"])
            self.tracker = OpponentTracker(self.board)
        self.tracker.update(observation["log"], observation["opponent"]["resource_cards"])
        return self.board, self.tracker


MEMORY = GameMemory()


# ---------------------------------------------------------------------------------
# 3. One decision's situation
# ---------------------------------------------------------------------------------


class TurnSituation:
    """Derived facts for one observation: networks, legal spots, production."""

    def __init__(self, observation: dict, board: BoardModel, tracker: OpponentTracker):
        self.observation = observation
        self.board = board
        self.tracker = tracker
        me, rival = observation["me"], observation["opponent"]
        self.me, self.rival = me, rival
        self.robber = _key(observation["robber_tile"])
        self.my_settlements = set(me["settlements"])
        self.my_cities = set(me["cities"])
        self.rival_buildings = set(rival["settlements"]) | set(rival["cities"])
        self.occupied = self.my_settlements | self.my_cities | self.rival_buildings
        self.my_roads = {_edge(e) for e in me["roads"]}
        self.all_roads = self.my_roads | {_edge(e) for e in rival["roads"]}
        self.hand = _hand_tuple(me["resources"])
        self.production = self._production()
        self.resource_value = self._resource_value()
        self.network = self._network(self.my_roads)
        self.spot_value_cache = {}

    # -- production and the worth of each resource to me right now ------------------

    def _production(self) -> dict:
        production = {resource: 0 for resource in RESOURCES}
        for node in self.my_settlements | self.my_cities:
            factor = 2 if node in self.my_cities else 1
            for resource, pips in self.board.node_production(node, self.robber).items():
                production[resource] += factor * pips
        return production

    def _resource_value(self) -> dict:
        """What one pip of each resource is worth: scarce on the board and to me."""
        value = {}
        for resource in RESOURCES:
            own_supply = self.production[resource]
            need = 1.25 if own_supply == 0 else (1.0 if own_supply < 6 else 0.85)
            value[resource] = PLAY_RESOURCE_WEIGHT[resource] * self.board.scarcity[resource] * need
        return value

    # -- networks and legal places -------------------------------------------------

    def _network(self, roads: set) -> set:
        nodes = set(self.my_settlements | self.my_cities)
        for a, b in roads:
            nodes.add(a)
            nodes.add(b)
        return nodes

    def is_settlement_spot(self, node: int) -> bool:
        """Distance rule: empty node with no building on any neighbour."""
        if node in self.occupied:
            return False
        return not any(n in self.occupied for n in self.board.neighbours.get(node, ()))

    def buildable_edges(self, roads: set, network: set) -> set:
        edges = set()
        for node in network:
            if node in self.rival_buildings:
                continue  # a rival building cuts my network at that node
            for other in self.board.neighbours.get(node, ()):
                edge = _edge((node, other))
                if edge not in self.all_roads and edge not in roads:
                    edges.add(edge)
        return edges

    def spot_value(self, node: int) -> float:
        """Afterstate production value of a settlement on `node` (not counting its VP)."""
        cached = self.spot_value_cache.get(node)
        if cached is not None:
            return cached
        production = self.board.node_production(node, self.robber)
        value = sum(WEIGHTS["pip"] * pips * self.resource_value[r] for r, pips in production.items())
        for resource in production:
            if self.production[resource] == 0:
                value += WEIGHTS["missing_resource"]
        value += self.port_value(node)
        self.spot_value_cache[node] = value
        return value

    def port_value(self, node: int) -> float:
        if node not in self.board.port_of:
            return 0.0
        resource = self.board.port_of[node]
        if resource is None:
            return WEIGHTS["port_generic"]
        return WEIGHTS["port_specific"] * (self.production[resource] + self.board.node_production(node).get(resource, 0))

    def expansion_value(self, roads: set) -> float:
        """How good the settlement spots my roads reach are, now and one road away."""
        network = self._network(roads)
        best_now, best_far = 0.0, 0.0
        for node in network:
            if node in self.rival_buildings:
                continue
            if self.is_settlement_spot(node):
                best_now = max(best_now, self.spot_value(node))
            for other in self.board.neighbours.get(node, ()):
                edge = _edge((node, other))
                if edge in self.all_roads or other in network:
                    continue
                if self.is_settlement_spot(other):
                    best_far = max(best_far, self.spot_value(other))
        return WEIGHTS["expansion_spot"] * best_now + WEIGHTS["expansion_far"] * best_far

    def longest_road(self, roads: set) -> int:
        """Longest trail through my roads; rival buildings break it."""
        adjacency = {}
        for a, b in roads:
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)
        best = 0

        def walk(node, used, length):
            nonlocal best
            best = max(best, length)
            if node in self.rival_buildings and length > 0:
                return
            for other in adjacency.get(node, ()):
                edge = _edge((node, other))
                if edge not in used:
                    used.add(edge)
                    walk(other, used, length + 1)
                    used.discard(edge)

        for start in adjacency:
            walk(start, set(), 0)
        return best

    def robber_value(self, tile, steal: bool, current=None) -> float:
        """Rival production I block, minus mine, plus a stolen card."""
        current = self.robber if current is None else current

        def blocked(coordinate):
            if coordinate is None or coordinate not in self.board.tiles:
                return 0.0
            resource, number, nodes = self.board.tiles[coordinate]
            if resource is None:
                return 0.0
            pips = PIPS.get(number, 0)
            rival = sum((2 if n in self.rival["cities"] else 1) for n in nodes if n in self.rival_buildings)
            mine = sum((2 if n in self.my_cities else 1) for n in nodes if n in self.my_settlements | self.my_cities)
            return pips * (rival - 1.3 * mine)

        value = WEIGHTS["robber_pip"] * (blocked(tile) - blocked(current))
        if steal:
            value += WEIGHTS["steal"] + 0.5 * min(self.rival["resource_cards"], 8)
        return value


# ---------------------------------------------------------------------------------
# 4-5. Planner and evaluator
# ---------------------------------------------------------------------------------


class Planner:
    """Searches builds and maritime trades over my hand.

    A build's precondition is its cost (and a legal place, already checked); its
    effect is the structural gain `delta`. A trade's effect is a hand change. The
    search keeps the best afterstate value reachable within a few steps this turn.
    """

    def __init__(self, situation: TurnSituation):
        self.s = situation
        self.trade_rates = self._trade_rates()
        self.deltas = self._build_deltas()
        self.memo = {}

    # -- trades ----------------------------------------------------------------

    def _trade_rates(self) -> dict:
        rates = {resource: 4 for resource in RESOURCES}
        for node in self.s.my_settlements | self.s.my_cities:
            port = self.s.board.port_of.get(node, "none")
            if port is None:
                rates = {r: min(rate, 3) for r, rate in rates.items()}
            elif port in rates:
                rates[port] = 2
        return rates

    def trades(self, hand: tuple):
        for give_index, give in enumerate(RESOURCES):
            rate = self.trade_rates[give]
            if hand[give_index] < rate:
                continue
            for receive_index, receive in enumerate(RESOURCES):
                if receive_index == give_index:
                    continue
                new = list(hand)
                new[give_index] -= rate
                new[receive_index] += 1
                yield (give, rate, receive), tuple(new)

    # -- the value of a hand ------------------------------------------------------

    def hand_value(self, hand: tuple) -> float:
        cards = sum(hand)
        value = WEIGHTS["card"] * sum(min(count, 4) + 0.4 * max(0, count - 4) for count in hand)
        value += WEIGHTS["goal_city"] * self._progress(hand, COSTS["CITY"]) ** 2
        if self.deltas.get("SETTLEMENT") is not None:
            value += WEIGHTS["goal_settlement"] * self._progress(hand, COSTS["SETTLEMENT"]) ** 2
        value += WEIGHTS["goal_development"] * self._progress(hand, COSTS["DEVELOPMENT_CARD"]) ** 2
        if cards > DISCARD_LIMIT:  # half the hand is lost at random on a 7
            value -= SEVEN_BEFORE_MY_NEXT_TURN * (cards // 2) * WEIGHTS["card"] * 1.6
        return value

    @staticmethod
    def _progress(hand: tuple, cost: dict) -> float:
        have = sum(min(hand[RESOURCES.index(r)], n) for r, n in cost.items())
        return have / sum(cost.values())

    # -- structural effects of each build ------------------------------------------

    def _build_deltas(self) -> dict:
        """Gain of the best instance of each build, assuming its place is legal."""
        s = self.s
        deltas = {}
        if s.my_settlements:
            node = max(s.my_settlements, key=s.spot_value)
            deltas["CITY"] = (node, self._vp_gain(1) + s.spot_value(node) - s.port_value(node))
        spots = [n for n in s.network if n not in s.rival_buildings and s.is_settlement_spot(n)]
        if spots:
            node = max(spots, key=s.spot_value)
            deltas["SETTLEMENT"] = (node, self._vp_gain(1) + s.spot_value(node))
        else:
            deltas["SETTLEMENT"] = None
        deltas["DEVELOPMENT_CARD"] = (None, self.development_card_value())
        road = self.best_road()
        if road is not None:
            deltas["ROAD"] = road
        return deltas

    def _vp_gain(self, points: int) -> float:
        gain = WEIGHTS["vp"] * points
        if self.s.me["victory_points"] + points >= self.s.observation["vps_to_win"]:
            gain += WEIGHTS["win"]
        return gain

    def road_delta(self, edge: tuple, roads: set = None) -> float:
        s = self.s
        roads = s.my_roads if roads is None else roads
        after = roads | {edge}
        gain = s.expansion_value(after) - s.expansion_value(roads)
        old_length = s.longest_road(roads)
        new_length = s.longest_road(after)
        gain += WEIGHTS["road_length"] * (new_length - old_length)
        rival_length = s.rival["longest_road_length"]
        if not s.me["has_longest_road"] and new_length >= LONGEST_ROAD_MINIMUM and new_length > rival_length:
            gain += self._vp_gain(2)
        elif not s.me["has_longest_road"] and new_length >= LONGEST_ROAD_MINIMUM - 1 and new_length >= rival_length:
            gain += 0.2 * WEIGHTS["vp"]  # one road from the bonus
        return gain

    def best_road(self, roads: set = None):
        s = self.s
        roads = s.my_roads if roads is None else roads
        network = s._network(roads)
        best = None
        for edge in s.buildable_edges(roads, network):
            delta = self.road_delta(edge, roads)
            if best is None or delta > best[1]:
                best = (edge, delta)
        return best

    def development_card_value(self) -> float:
        """Expected value of a card drawn from the unknown part of the deck."""
        pool = self.unknown_development_cards()
        total = sum(pool.values())
        if total <= 0 or self.s.observation["development_cards_left"] <= 0:
            return -1000.0
        army = self.army_gain(extra_knights=1)
        value = {
            "VICTORY_POINT": self._vp_gain(1),
            "KNIGHT": WEIGHTS["knight_in_hand"] + 0.5 * army,
            "YEAR_OF_PLENTY": WEIGHTS["progress_card_in_hand"] + 6,
            "MONOPOLY": WEIGHTS["progress_card_in_hand"] + 6,
            "ROAD_BUILDING": WEIGHTS["progress_card_in_hand"],
        }
        return sum(pool[card] * value[card] for card in pool) / total

    def unknown_development_cards(self) -> dict:
        """Deck composition minus every card whose identity I know."""
        me, rival = self.s.me, self.s.rival
        pool = dict(DEV_DECK)
        for card in pool:
            pool[card] -= me["development_cards_in_hand"].get(card, 0)
            pool[card] -= me["played_development_cards"].get(card, 0)
            pool[card] -= rival["played_development_cards"].get(card, 0)
            pool[card] = max(0, pool[card])
        return pool

    def army_gain(self, extra_knights: int = 1) -> float:
        me, rival = self.s.me, self.s.rival
        mine = me["played_development_cards"].get("KNIGHT", 0) + extra_knights
        theirs = rival["played_development_cards"].get("KNIGHT", 0)
        gain = WEIGHTS["knight_played"] * extra_knights
        if not me["has_largest_army"] and mine >= LARGEST_ARMY_MINIMUM and mine > theirs:
            gain += self._vp_gain(2)
        elif not me["has_largest_army"] and mine >= theirs:
            gain += 0.15 * WEIGHTS["vp"] * min(mine, LARGEST_ARMY_MINIMUM) / LARGEST_ARMY_MINIMUM
        return gain

    # -- search ----------------------------------------------------------------

    def plan_value(self, hand: tuple, trades_left: int = 2, builds_left: int = 1) -> float:
        """Best afterstate hand+build value reachable from `hand` this turn."""
        key = (hand, trades_left, builds_left)
        cached = self.memo.get(key)
        if cached is not None:
            return cached
        best = self.hand_value(hand)
        if builds_left:
            for kind, entry in self.deltas.items():
                if entry is None:
                    continue
                paid = _pay(hand, COSTS[kind])
                if paid is not None:
                    best = max(best, entry[1] + self.plan_value(paid, trades_left, builds_left - 1))
        if trades_left:
            for _, traded in self.trades(hand):
                best = max(best, self.plan_value(traded, trades_left - 1, builds_left))
        self.memo[key] = best
        return best


class Evaluator:
    """Scores every legal action as the value of the state it leads to."""

    def __init__(self, observation: dict):
        board, tracker = MEMORY.refresh(observation)
        self.observation = observation
        self.situation = TurnSituation(observation, board, tracker)
        self.planner = Planner(self.situation)
        self.hand = self.situation.hand

    def value(self, action: dict) -> float:
        handler = getattr(self, "_" + action["type"].lower(), None)
        return handler(action) if handler else 0.0

    # -- turn structure -------------------------------------------------------------

    def _roll(self, action):
        return self.planner.plan_value(self.hand, 0, 0)

    def _end_turn(self, action):
        return self.planner.plan_value(self.hand, 0, 0)

    # -- builds: effect + what the remaining hand can still do -----------------------

    def _after_paying(self, kind: str) -> float:
        paid = _pay(self.hand, COSTS[kind])
        return self.planner.plan_value(paid, 1, 1) if paid is not None else -1e9

    def _build_city(self, action):
        s = self.situation
        node = action["node"]
        gain = self.planner._vp_gain(1) + s.spot_value(node) - s.port_value(node)
        return gain + self._after_paying("CITY")

    def _build_settlement(self, action):
        s = self.situation
        node = action["node"]
        gain = self.planner._vp_gain(1) + s.spot_value(node)
        return gain + self._after_paying("SETTLEMENT")

    def _build_road(self, action):
        edge = _edge(action["edge"])
        gain = self.planner.road_delta(edge)
        if self.observation["phase"] == "free_roads":
            return gain
        return gain + self._after_paying("ROAD")

    def _buy_development_card(self, action):
        return self.planner.development_card_value() + self._after_paying("DEVELOPMENT_CARD")

    def _maritime_trade(self, action):
        hand = list(self.hand)
        hand[RESOURCES.index(action["give"])] -= action["give_count"]
        hand[RESOURCES.index(action["receive"])] += 1
        return self.planner.plan_value(tuple(hand), 1, 1)

    # -- development cards: value of playing now against keeping the card ------------

    def _play_knight_card(self, action):
        s = self.situation
        best_robber = max((s.robber_value(tile, steal=True) for tile in s.board.tiles if tile != s.robber), default=0)
        keep = WEIGHTS["knight_in_hand"]
        return self._hand_after_card() + self.planner.army_gain(1) + best_robber - keep

    def _hand_after_card(self) -> float:
        """Before the roll, the alternative (ROLL) is valued on the hand as it is;
        after the roll, playing a card still leaves this turn's builds and trades."""
        if any(a["type"] == "ROLL" for a in self.observation["actions"]):
            return self.planner.plan_value(self.hand, 0, 0)
        return self.planner.plan_value(self.hand, 2, 1)

    def _play_year_of_plenty(self, action):
        hand = list(self.hand)
        for resource in action["resources"]:
            hand[RESOURCES.index(resource)] += 1
        return self.planner.plan_value(tuple(hand), 1, 1) - WEIGHTS["progress_card_in_hand"]

    def _play_monopoly(self, action):
        expected = self.situation.tracker.hand.get(action["resource"], 0.0)
        hand = list(self.hand)
        hand[RESOURCES.index(action["resource"])] += int(round(expected))
        value = self.planner.plan_value(tuple(hand), 1, 1) - WEIGHTS["progress_card_in_hand"]
        return value if expected >= 1.5 else value - 50

    def _play_road_building(self, action):
        first = self.planner.best_road()
        if first is None:
            return -1e9
        roads = self.situation.my_roads | {first[0]}
        second = self.planner.best_road(roads)
        gain = first[1] + (second[1] if second else 0.0)
        return self._hand_after_card() + gain - WEIGHTS["progress_card_in_hand"]

    def _move_robber(self, action):
        return self.situation.robber_value(_key(action["tile"]), action.get("steal_from") == "opponent")


# ---------------------------------------------------------------------------------
# Initial placement
# ---------------------------------------------------------------------------------


def placement_value(board: BoardModel, node: int, my_settlements: list, occupied: set) -> float:
    """Pips weighted by resource and scarcity, with diversity, ports and complement."""
    production = board.node_production(node)
    value = sum(pips * PLACEMENT_RESOURCE_WEIGHT[r] * board.scarcity[r] for r, pips in production.items())
    value += 1.2 * len(production)
    numbers = [board.tiles[c][1] for c in board.node_tiles[node] if board.tiles[c][1]]
    if len(set(numbers)) < len(numbers):
        value -= 1.0  # the same number twice is feast or famine
    covered = set()
    for existing in my_settlements:
        covered |= set(board.node_production(existing))
    if my_settlements:
        value += 2.0 * len(set(production) - covered)
        if not ({"WOOD", "BRICK"} <= covered | set(production)):
            value -= 2.5  # without wood and brick I cannot expand at all
    port = board.port_of.get(node, "none")
    if port is None:
        value += 1.0
    elif port in RESOURCES:
        value += 0.25 * (production.get(port, 0) + sum(board.node_production(n).get(port, 0) for n in my_settlements))
    return value


def _placement_spot_free(board: BoardModel, node: int, occupied: set) -> bool:
    return node not in occupied and not any(n in occupied for n in board.neighbours.get(node, ()))


def choose_initial(observation: dict) -> int:
    board, _ = MEMORY.refresh(observation)
    me, rival = observation["me"], observation["opponent"]
    occupied = set(me["settlements"]) | set(rival["settlements"]) | set(me["cities"]) | set(rival["cities"])
    actions = observation["actions"]
    if actions[0]["type"] == "BUILD_SETTLEMENT":
        return max(actions, key=lambda a: placement_value(board, a["node"], me["settlements"], occupied))["id"]

    def road_score(action):
        a, b = action["edge"]
        far = b if a in me["settlements"] else a
        options = [placement_value(board, n, me["settlements"], occupied)
                   for n in board.neighbours.get(far, ()) if _placement_spot_free(board, n, occupied | {far})]
        return max(options, default=0.0)

    return max(actions, key=road_score)["id"]


# ---------------------------------------------------------------------------------
# Required algorithms: abstraction, learned values, Q-learning, baseline
# ---------------------------------------------------------------------------------

# Learned offline by training/train_q.py (Q-learning, see README). Keys are
# "state_features|action kind"; values are in victory points.
Q = {"('free_roads', 2, -1, 3, 0, 0, 0)|BUILD_ROAD": 1.2141, "('free_roads', 2, -1, 4, 0, 0, 0)|BUILD_ROAD": 1.8624, "('free_roads', 2, -1, 4, 0, 1, 0)|BUILD_ROAD": 1.3001, "('free_roads', 2, -1, 5, 0, 0, 0)|BUILD_ROAD": 0.8813, "('free_roads', 2, -2, 3, 0, 0, 0)|BUILD_ROAD": 0.7919, "('free_roads', 2, -2, 4, 0, 0, 0)|BUILD_ROAD": 0.8082, "('free_roads', 2, -3, 3, 0, 0, 0)|BUILD_ROAD": 0.1376, "('free_roads', 2, -3, 4, 0, 0, 0)|BUILD_ROAD": 0.7104, "('free_roads', 2, 0, 3, 0, 0, 0)|BUILD_ROAD": 1.4797, "('free_roads', 2, 0, 4, 0, 0, 0)|BUILD_ROAD": 2.2338, "('free_roads', 2, 0, 4, 0, 1, 0)|BUILD_ROAD": 1.6652, "('free_roads', 2, 0, 4, 0, 1, 1)|BUILD_ROAD": 1.3743, "('free_roads', 2, 0, 5, 0, 0, 0)|BUILD_ROAD": 1.1492, "('free_roads', 3, -1, 3, 0, 0, 0)|BUILD_ROAD": 0.15, "('free_roads', 3, -1, 4, 0, 0, 0)|BUILD_ROAD": 0.7098, "('free_roads', 3, -1, 5, 0, 0, 0)|BUILD_ROAD": 0.6122, "('free_roads', 3, -1, 6, 0, 0, 0)|BUILD_ROAD": 1.7864, "('free_roads', 3, -2, 3, 0, 0, 0)|BUILD_ROAD": 0.2369, "('free_roads', 3, -2, 5, 0, 0, 0)|BUILD_ROAD": 0.8506, "('free_roads', 3, -2, 6, 0, 0, 0)|BUILD_ROAD": 1.2515, "('free_roads', 3, -3, 4, 0, 0, 0)|BUILD_ROAD": 0.0313, "('free_roads', 3, 0, 3, 0, 0, 0)|BUILD_ROAD": 0.1179, "('free_roads', 3, 0, 4, 0, 0, 0)|BUILD_ROAD": 0.7959, "('free_roads', 3, 0, 5, 0, 0, 0)|BUILD_ROAD": 0.7156, "('free_roads', 3, 0, 5, 0, 1, 0)|BUILD_ROAD": 0.7875, "('free_roads', 3, 0, 6, 0, 0, 0)|BUILD_ROAD": 1.7578, "('free_roads', 3, 0, 6, 0, 1, 0)|BUILD_ROAD": 1.0327, "('free_roads', 3, 1, 3, 0, 0, 0)|BUILD_ROAD": 0.7444, "('free_roads', 3, 1, 4, 0, 0, 0)|BUILD_ROAD": 1.7899, "('free_roads', 3, 1, 4, 0, 1, 0)|BUILD_ROAD": 0.5333, "('free_roads', 3, 1, 5, 0, 0, 0)|BUILD_ROAD": 1.384, "('free_roads', 3, 1, 6, 0, 0, 0)|BUILD_ROAD": 3.0453, "('free_roads', 3, 1, 6, 0, 1, 1)|BUILD_ROAD": 0.9918, "('free_roads', 4, -1, 5, 0, 0, 0)|BUILD_ROAD": 0.501, "('free_roads', 4, -1, 5, 0, 1, 0)|BUILD_ROAD": 0.7323, "('free_roads', 4, -1, 6, 0, 0, 0)|BUILD_ROAD": 0.5156, "('free_roads', 4, -1, 6, 0, 1, 0)|BUILD_ROAD": 0.8119, "('free_roads', 4, -2, 6, 0, 0, 0)|BUILD_ROAD": 0.7943, "('free_roads', 4, -2, 6, 0, 1, 0)|BUILD_ROAD": 0.4626, "('free_roads', 4, 0, 3, 0, 0, 0)|BUILD_ROAD": 0.1475, "('free_roads', 4, 0, 4, 0, 0, 0)|BUILD_ROAD": 0.3035, "('free_roads', 4, 0, 6, 0, 0, 0)|BUILD_ROAD": 1.843, "('free_roads', 4, 0, 6, 0, 1, 0)|BUILD_ROAD": 1.4426, "('free_roads', 4, 0, 6, 0, 1, 1)|BUILD_ROAD": 0.826, "('free_roads', 4, 1, 4, 0, 0, 0)|BUILD_ROAD": 0.2661, "('free_roads', 4, 1, 5, 0, 0, 0)|BUILD_ROAD": 0.6475, "('free_roads', 4, 1, 6, 0, 0, 0)|BUILD_ROAD": 2.1309, "('free_roads', 4, 1, 6, 0, 1, 0)|BUILD_ROAD": 0.7375, "('free_roads', 4, 2, 4, 0, 0, 0)|BUILD_ROAD": 0.8176, "('free_roads', 4, 2, 5, 0, 0, 0)|BUILD_ROAD": 0.9393, "('free_roads', 4, 2, 6, 0, 0, 0)|BUILD_ROAD": 2.7973, "('free_roads', 4, 2, 6, 0, 1, 0)|BUILD_ROAD": 1.3382, "('free_roads', 5, -1, 6, 0, 0, 0)|BUILD_ROAD": 0.8592, "('free_roads', 5, 0, 6, 0, 0, 0)|BUILD_ROAD": 0.783, "('free_roads', 5, 0, 6, 0, 1, 0)|BUILD_ROAD": 0.379, "('free_roads', 5, 1, 5, 0, 0, 0)|BUILD_ROAD": 0.1741, "('free_roads', 5, 1, 6, 0, 0, 0)|BUILD_ROAD": 1.8133, "('free_roads', 5, 2, 4, 0, 0, 0)|BUILD_ROAD": 0.3254, "('free_roads', 5, 2, 6, 0, 0, 0)|BUILD_ROAD": 1.8487, "('free_roads', 5, 2, 6, 0, 1, 0)|BUILD_ROAD": 0.6131, "('free_roads', 5, 3, 4, 0, 0, 0)|BUILD_ROAD": 0.6416, "('free_roads', 5, 3, 6, 0, 0, 0)|BUILD_ROAD": 2.1944, "('free_roads', 5, 3, 6, 0, 1, 0)|BUILD_ROAD": 0.8092, "('free_roads', 5, 3, 6, 0, 1, 1)|BUILD_ROAD": 0.8092, "('free_roads', 6, 0, 6, 0, 0, 0)|BUILD_ROAD": 0.648, "('free_roads', 6, 1, 6, 0, 0, 0)|BUILD_ROAD": 1.6914, "('free_roads', 6, 1, 6, 0, 1, 0)|BUILD_ROAD": 0.5771, "('free_roads', 6, 2, 6, 0, 0, 0)|BUILD_ROAD": 1.3652, "('free_roads', 6, 2, 6, 0, 1, 0)|BUILD_ROAD": 1.3206, "('free_roads', 6, 3, 4, 0, 0, 0)|BUILD_ROAD": 0.5735, "('free_roads', 6, 3, 5, 0, 0, 0)|BUILD_ROAD": 0.8504, "('free_roads', 6, 3, 6, 0, 0, 0)|BUILD_ROAD": 1.8523, "('free_roads', 6, 3, 6, 0, 1, 0)|BUILD_ROAD": 1.0502, "('free_roads', 6, 3, 6, 0, 1, 1)|BUILD_ROAD": 1.1034, "('move_robber', 2, -1, 2, 0, 0, 0)|MOVE_ROBBER": 1.3915, "('move_robber', 2, -1, 3, 0, 0, 0)|MOVE_ROBBER": 1.6478, "('move_robber', 2, -1, 3, 0, 0, 1)|MOVE_ROBBER": 1.3883, "('move_robber', 2, -1, 3, 0, 1, 0)|MOVE_ROBBER": 1.3684, "('move_robber', 2, -1, 4, 0, 0, 0)|MOVE_ROBBER": 1.5737, "('move_robber', 2, -1, 4, 0, 0, 1)|MOVE_ROBBER": 1.3403, "('move_robber', 2, -1, 4, 0, 1, 0)|MOVE_ROBBER": 1.3919, "('move_robber', 2, -1, 4, 1, 0, 0)|MOVE_ROBBER": 0.6937, "('move_robber', 2, -1, 5, 0, 0, 0)|MOVE_ROBBER": 0.9646, "('move_robber', 2, -2, 2, 0, 0, 0)|MOVE_ROBBER": 0.7673, "('move_robber', 2, -2, 3, 0, 0, 0)|MOVE_ROBBER": 1.1208, "('move_robber', 2, -2, 3, 0, 0, 1)|MOVE_ROBBER": 0.4914, "('move_robber', 2, -2, 4, 0, 0, 0)|MOVE_ROBBER": 1.0689, "('move_robber', 2, -2, 4, 0, 0, 1)|MOVE_ROBBER": 0.5675, "('move_robber', 2, -2, 5, 0, 0, 0)|MOVE_ROBBER": 0.3261, "('move_robber', 2, -3, 2, 0, 0, 0)|MOVE_ROBBER": 0.3584, "('move_robber', 2, -3, 3, 0, 0, 0)|MOVE_ROBBER": 0.6923, "('move_robber', 2, -3, 3, 0, 0, 1)|MOVE_ROBBER": 0.2907, "('move_robber', 2, -3, 4, 0, 0, 0)|MOVE_ROBBER": 0.5641, "('move_robber', 2, -3, 4, 0, 0, 1)|MOVE_ROBBER": 0.2828, "('move_robber', 2, -3, 5, 0, 0, 0)|MOVE_ROBBER": 0.0872, "('move_robber', 2, 0, 2, 0, 0, 0)|MOVE_ROBBER": 1.8347, "('move_robber', 2, 0, 3, 0, 0, 0)|MOVE_ROBBER": 2.1631, "('move_robber', 2, 0, 3, 0, 0, 1)|MOVE_ROBBER": 2.3593, "('move_robber', 2, 0, 3, 0, 1, 0)|MOVE_ROBBER": 1.9325, "('move_robber', 2, 0, 3, 0, 1, 1)|MOVE_ROBBER": 1.7291, "('move_robber', 2, 0, 3, 1, 0, 0)|MOVE_ROBBER": 1.197, "('move_robber', 2, 0, 4, 0, 0, 0)|MOVE_ROBBER": 2.0697, "('move_robber', 2, 0, 4, 0, 0, 1)|MOVE_ROBBER": 2.4053, "('move_robber', 2, 0, 4, 0, 1, 0)|MOVE_ROBBER": 2.5432, "('move_robber', 2, 0, 4, 0, 1, 1)|MOVE_ROBBER": 2.4443, "('move_robber', 2, 0, 4, 1, 0, 0)|MOVE_ROBBER": 2.448, "('move_robber', 2, 0, 5, 0, 0, 0)|MOVE_ROBBER": 2.0726, "('move_robber', 2, 0, 5, 0, 0, 1)|MOVE_ROBBER": 1.7806, "('move_robber', 2, 0, 5, 0, 1, 0)|MOVE_ROBBER": 1.1211, "('move_robber', 3, -1, 2, 0, 0, 0)|MOVE_ROBBER": 0.6017, "('move_robber', 3, -1, 3, 0, 0, 0)|MOVE_ROBBER": 1.1908, "('move_robber', 3, -1, 3, 0, 0, 1)|MOVE_ROBBER": 0.5246, "('move_robber', 3, -1, 4, 0, 0, 0)|MOVE_ROBBER": 1.2386, "('move_robber', 3, -1, 4, 0, 0, 1)|MOVE_ROBBER": 0.836, "('move_robber', 3, -1, 5, 0, 0, 0)|MOVE_ROBBER": 1.2603, "('move_robber', 3, -1, 5, 0, 0, 1)|MOVE_ROBBER": 1.0276, "('move_robber', 3, -1, 6, 0, 0, 0)|MOVE_ROBBER": 1.5737, "('move_robber', 3, -1, 6, 0, 0, 1)|MOVE_ROBBER": 1.0459, "('move_robber', 3, -2, 2, 0, 0, 0)|MOVE_ROBBER": 0.0588, "('move_robber', 3, -2, 3, 0, 0, 0)|MOVE_ROBBER": 0.4307, "('move_robber', 3, -2, 3, 0, 0, 1)|MOVE_ROBBER": 0.1367, "('move_robber', 3, -2, 4, 0, 0, 0)|MOVE_ROBBER": 0.422, "('move_robber', 3, -2, 4, 0, 0, 1)|MOVE_ROBBER": 0.3429, "('move_robber', 3, -2, 5, 0, 0, 0)|MOVE_ROBBER": 0.9822, "('move_robber', 3, -2, 5, 0, 0, 1)|MOVE_ROBBER": 0.2899, "('move_robber', 3, -2, 6, 0, 0, 0)|MOVE_ROBBER": 0.6882, "('move_robber', 3, -2, 6, 0, 0, 1)|MOVE_ROBBER": 0.3307, "('move_robber', 3, -3, 3, 0, 0, 0)|MOVE_ROBBER": 0.0449, "('move_robber', 3, -3, 4, 0, 0, 0)|MOVE_ROBBER": 0.2996, "('move_robber', 3, -3, 4, 0, 0, 1)|MOVE_ROBBER": 0.0567, "('move_robber', 3, -3, 5, 0, 0, 0)|MOVE_ROBBER": 0.0751, "('move_robber', 3, -3, 6, 0, 0, 0)|MOVE_ROBBER": 0.2985, "('move_robber', 3, 0, 2, 0, 0, 0)|MOVE_ROBBER": 0.453, "('move_robber', 3, 0, 3, 0, 0, 0)|MOVE_ROBBER": 1.3786, "('move_robber', 3, 0, 3, 0, 0, 1)|MOVE_ROBBER": 0.3505, "('move_robber', 3, 0, 4, 0, 0, 0)|MOVE_ROBBER": 1.6535, "('move_robber', 3, 0, 4, 0, 0, 1)|MOVE_ROBBER": 0.9161, "('move_robber', 3, 0, 4, 0, 1, 0)|MOVE_ROBBER": 0.551, "('move_robber', 3, 0, 5, 0, 0, 0)|MOVE_ROBBER": 1.6389, "('move_robber', 3, 0, 5, 0, 0, 1)|MOVE_ROBBER": 1.116, "('move_robber', 3, 0, 5, 0, 1, 0)|MOVE_ROBBER": 0.916, "('move_robber', 3, 0, 6, 0, 0, 0)|MOVE_ROBBER": 1.7792, "('move_robber', 3, 0, 6, 0, 0, 1)|MOVE_ROBBER": 1.4852, "('move_robber', 3, 0, 6, 0, 1, 0)|MOVE_ROBBER": 0.9494, "('move_robber', 3, 1, 2, 0, 0, 0)|MOVE_ROBBER": 0.8871, "('move_robber', 3, 1, 3, 0, 0, 0)|MOVE_ROBBER": 1.9298, "('move_robber', 3, 1, 3, 0, 0, 1)|MOVE_ROBBER": 0.7251, "('move_robber', 3, 1, 4, 0, 0, 0)|MOVE_ROBBER": 1.812, "('move_robber', 3, 1, 4, 0, 0, 1)|MOVE_ROBBER": 1.6124, "('move_robber', 3, 1, 4, 0, 1, 0)|MOVE_ROBBER": 1.4247, "('move_robber', 3, 1, 4, 0, 1, 1)|MOVE_ROBBER": 0.7842, "('move_robber', 3, 1, 5, 0, 0, 0)|MOVE_ROBBER": 2.1018, "('move_robber', 3, 1, 5, 0, 0, 1)|MOVE_ROBBER": 1.8743, "('move_robber', 3, 1, 5, 0, 1, 0)|MOVE_ROBBER": 1.2755, "('move_robber', 3, 1, 6, 0, 0, 0)|MOVE_ROBBER": 2.3762, "('move_robber', 3, 1, 6, 0, 0, 1)|MOVE_ROBBER": 2.2729, "('move_robber', 3, 1, 6, 0, 1, 0)|MOVE_ROBBER": 1.4941, "('move_robber', 3, 1, 6, 0, 1, 1)|MOVE_ROBBER": 0.522, "('move_robber', 3, 1, 6, 1, 0, 0)|MOVE_ROBBER": 1.4757, "('move_robber', 4, -1, 2, 0, 0, 0)|MOVE_ROBBER": 0.1041, "('move_robber', 4, -1, 3, 0, 0, 0)|MOVE_ROBBER": 0.3516, "('move_robber', 4, -1, 3, 0, 0, 1)|MOVE_ROBBER": 0.1247, "('move_robber', 4, -1, 4, 0, 0, 0)|MOVE_ROBBER": 0.4421, "('move_robber', 4, -1, 4, 0, 0, 1)|MOVE_ROBBER": 0.1699, "('move_robber', 4, -1, 5, 0, 0, 0)|MOVE_ROBBER": 0.3762, "('move_robber', 4, -1, 6, 0, 0, 0)|MOVE_ROBBER": 1.1307, "('move_robber', 4, -1, 6, 0, 0, 1)|MOVE_ROBBER": 0.7174, "('move_robber', 4, -1, 6, 1, 0, 0)|MOVE_ROBBER": 0.1128, "('move_robber', 4, -2, 3, 0, 0, 0)|MOVE_ROBBER": -0.2467, "('move_robber', 4, -2, 4, 0, 0, 0)|MOVE_ROBBER": 0.3112, "('move_robber', 4, -2, 4, 0, 0, 1)|MOVE_ROBBER": 0.0922, "('move_robber', 4, -2, 5, 0, 0, 0)|MOVE_ROBBER": 0.2671, "('move_robber', 4, -2, 5, 0, 0, 1)|MOVE_ROBBER": 0.1029, "('move_robber', 4, -2, 6, 0, 0, 0)|MOVE_ROBBER": 0.2639, "('move_robber', 4, -2, 6, 0, 0, 1)|MOVE_ROBBER": 0.3478, "('move_robber', 4, -2, 6, 0, 1, 0)|MOVE_ROBBER": 0.1671, "('move_robber', 4, 0, 2, 0, 0, 0)|MOVE_ROBBER": 0.4146, "('move_robber', 4, 0, 3, 0, 0, 0)|MOVE_ROBBER": 0.6726, "('move_robber', 4, 0, 3, 0, 0, 1)|MOVE_ROBBER": 0.2082, "('move_robber', 4, 0, 4, 0, 0, 0)|MOVE_ROBBER": 0.8402, "('move_robber', 4, 0, 4, 0, 0, 1)|MOVE_ROBBER": 0.0705, "('move_robber', 4, 0, 5, 0, 0, 0)|MOVE_ROBBER": 1.5131, "('move_robber', 4, 0, 6, 0, 0, 0)|MOVE_ROBBER": 1.4537, "('move_robber', 4, 0, 6, 0, 0, 1)|MOVE_ROBBER": 1.2499, "('move_robber', 4, 0, 6, 0, 1, 0)|MOVE_ROBBER": 1.1922, "('move_robber', 4, 1, 2, 0, 0, 0)|MOVE_ROBBER": 0.5948, "('move_robber', 4, 1, 3, 0, 0, 0)|MOVE_ROBBER": 1.0404, "('move_robber', 4, 1, 3, 0, 0, 1)|MOVE_ROBBER": 0.3675, "('move_robber', 4, 1, 4, 0, 0, 0)|MOVE_ROBBER": 0.9365, "('move_robber', 4, 1, 4, 0, 0, 1)|MOVE_ROBBER": 0.332, "('move_robber', 4, 1, 4, 0, 1, 0)|MOVE_ROBBER": 0.7312, "('move_robber', 4, 1, 5, 0, 0, 0)|MOVE_ROBBER": 1.5646, "('move_robber', 4, 1, 5, 0, 0, 1)|MOVE_ROBBER": 0.5126, "('move_robber', 4, 1, 6, 0, 0, 0)|MOVE_ROBBER": 1.3726, "('move_robber', 4, 1, 6, 0, 0, 1)|MOVE_ROBBER": 1.5053, "('move_robber', 4, 1, 6, 0, 1, 0)|MOVE_ROBBER": 0.9791, "('move_robber', 4, 2, 2, 0, 0, 0)|MOVE_ROBBER": 0.8798, "('move_robber', 4, 2, 3, 0, 0, 0)|MOVE_ROBBER": 1.559, "('move_robber', 4, 2, 3, 0, 0, 1)|MOVE_ROBBER": 0.502, "('move_robber', 4, 2, 4, 0, 0, 0)|MOVE_ROBBER": 1.7882, "('move_robber', 4, 2, 4, 0, 0, 1)|MOVE_ROBBER": 0.8497, "('move_robber', 4, 2, 4, 0, 1, 0)|MOVE_ROBBER": 0.8267, "('move_robber', 4, 2, 5, 0, 0, 0)|MOVE_ROBBER": 2.005, "('move_robber', 4, 2, 6, 0, 0, 0)|MOVE_ROBBER": 2.1813, "('move_robber', 4, 2, 6, 0, 0, 1)|MOVE_ROBBER": 2.1566, "('move_robber', 4, 2, 6, 0, 1, 0)|MOVE_ROBBER": 1.2752, "('move_robber', 4, 2, 6, 1, 0, 0)|MOVE_ROBBER": 0.8644, "('move_robber', 5, -1, 3, 0, 0, 0)|MOVE_ROBBER": -0.3496, "('move_robber', 5, -1, 4, 0, 0, 0)|MOVE_ROBBER": -0.162, "('move_robber', 5, -1, 5, 0, 0, 0)|MOVE_ROBBER": 0.4533, "('move_robber', 5, -1, 6, 0, 0, 0)|MOVE_ROBBER": 0.5006, "('move_robber', 5, -1, 6, 0, 0, 1)|MOVE_ROBBER": 0.4487, "('move_robber', 5, 0, 3, 0, 0, 0)|MOVE_ROBBER": 0.218, "('move_robber', 5, 0, 4, 0, 0, 0)|MOVE_ROBBER": 0.5985, "('move_robber', 5, 0, 5, 0, 0, 0)|MOVE_ROBBER": 0.8419, "('move_robber', 5, 0, 5, 0, 0, 1)|MOVE_ROBBER": 0.6807, "('move_robber', 5, 0, 6, 0, 0, 0)|MOVE_ROBBER": 1.008, "('move_robber', 5, 0, 6, 0, 0, 1)|MOVE_ROBBER": 0.6914, "('move_robber', 5, 1, 3, 0, 0, 0)|MOVE_ROBBER": 0.3554, "('move_robber', 5, 1, 3, 0, 0, 1)|MOVE_ROBBER": 0.2073, "('move_robber', 5, 1, 4, 0, 0, 0)|MOVE_ROBBER": 0.9599, "('move_robber', 5, 1, 4, 0, 0, 1)|MOVE_ROBBER": 0.2403, "('move_robber', 5, 1, 5, 0, 0, 0)|MOVE_ROBBER": 0.9814, "('move_robber', 5, 1, 5, 0, 0, 1)|MOVE_ROBBER": 0.7212, "('move_robber', 5, 1, 6, 0, 0, 0)|MOVE_ROBBER": 1.5033, "('move_robber', 5, 1, 6, 0, 0, 1)|MOVE_ROBBER": 1.429, "('move_robber', 5, 1, 6, 0, 1, 0)|MOVE_ROBBER": 0.7116, "('move_robber', 5, 2, 2, 0, 0, 0)|MOVE_ROBBER": 0.3498, "('move_robber', 5, 2, 3, 0, 0, 0)|MOVE_ROBBER": 0.8297, "('move_robber', 5, 2, 4, 0, 0, 0)|MOVE_ROBBER": 1.22, "('move_robber', 5, 2, 4, 0, 0, 1)|MOVE_ROBBER": 0.3936, "('move_robber', 5, 2, 5, 0, 0, 0)|MOVE_ROBBER": 1.53, "('move_robber', 5, 2, 5, 0, 0, 1)|MOVE_ROBBER": 0.748, "('move_robber', 5, 2, 6, 0, 0, 0)|MOVE_ROBBER": 1.677, "('move_robber', 5, 2, 6, 0, 0, 1)|MOVE_ROBBER": 1.4698, "('move_robber', 5, 2, 6, 0, 1, 0)|MOVE_ROBBER": 0.7449, "('move_robber', 5, 2, 6, 1, 0, 0)|MOVE_ROBBER": 1.0887, "('move_robber', 5, 3, 2, 0, 0, 0)|MOVE_ROBBER": 0.277, "('move_robber', 5, 3, 3, 0, 0, 0)|MOVE_ROBBER": 1.3946, "('move_robber', 5, 3, 4, 0, 0, 0)|MOVE_ROBBER": 1.4872, "('move_robber', 5, 3, 4, 0, 0, 1)|MOVE_ROBBER": 0.3903, "('move_robber', 5, 3, 5, 0, 0, 0)|MOVE_ROBBER": 1.8432, "('move_robber', 5, 3, 5, 0, 0, 1)|MOVE_ROBBER": 0.7973, "('move_robber', 5, 3, 6, 0, 0, 0)|MOVE_ROBBER": 1.9295, "('move_robber', 5, 3, 6, 0, 0, 1)|MOVE_ROBBER": 1.9951, "('move_robber', 5, 3, 6, 0, 1, 0)|MOVE_ROBBER": 1.8933, "('move_robber', 5, 3, 6, 0, 1, 1)|MOVE_ROBBER": 0.8702, "('move_robber', 5, 3, 6, 1, 0, 0)|MOVE_ROBBER": 0.7652, "('move_robber', 6, 0, 3, 0, 0, 0)|MOVE_ROBBER": -0.0505, "('move_robber', 6, 0, 4, 0, 0, 0)|MOVE_ROBBER": 0.2119, "('move_robber', 6, 0, 5, 0, 0, 0)|MOVE_ROBBER": 0.4909, "('move_robber', 6, 0, 6, 0, 0, 0)|MOVE_ROBBER": 0.8157, "('move_robber', 6, 0, 6, 0, 0, 1)|MOVE_ROBBER": 0.5545, "('move_robber', 6, 1, 3, 0, 0, 0)|MOVE_ROBBER": 0.0036, "('move_robber', 6, 1, 4, 0, 0, 0)|MOVE_ROBBER": 0.6416, "('move_robber', 6, 1, 5, 0, 0, 0)|MOVE_ROBBER": 0.7861, "('move_robber', 6, 1, 6, 0, 0, 0)|MOVE_ROBBER": 1.0973, "('move_robber', 6, 1, 6, 0, 0, 1)|MOVE_ROBBER": 1.0418, "('move_robber', 6, 1, 6, 0, 1, 0)|MOVE_ROBBER": 0.5685, "('move_robber', 6, 2, 3, 0, 0, 0)|MOVE_ROBBER": 0.0531, "('move_robber', 6, 2, 4, 0, 0, 0)|MOVE_ROBBER": 0.3855, "('move_robber', 6, 2, 5, 0, 0, 0)|MOVE_ROBBER": 1.3597, "('move_robber', 6, 2, 5, 0, 0, 1)|MOVE_ROBBER": 0.3697, "('move_robber', 6, 2, 6, 0, 0, 0)|MOVE_ROBBER": 1.3725, "('move_robber', 6, 2, 6, 0, 0, 1)|MOVE_ROBBER": 1.4354, "('move_robber', 6, 2, 6, 0, 1, 0)|MOVE_ROBBER": 1.0575, "('move_robber', 6, 2, 6, 1, 0, 0)|MOVE_ROBBER": 0.8113, "('move_robber', 6, 3, 2, 0, 0, 0)|MOVE_ROBBER": 0.1837, "('move_robber', 6, 3, 3, 0, 0, 0)|MOVE_ROBBER": 1.0121, "('move_robber', 6, 3, 4, 0, 0, 0)|MOVE_ROBBER": 0.7667, "('move_robber', 6, 3, 4, 0, 0, 1)|MOVE_ROBBER": 0.6179, "('move_robber', 6, 3, 5, 0, 0, 0)|MOVE_ROBBER": 1.5418, "('move_robber', 6, 3, 5, 0, 0, 1)|MOVE_ROBBER": 0.8933, "('move_robber', 6, 3, 6, 0, 0, 0)|MOVE_ROBBER": 1.6962, "('move_robber', 6, 3, 6, 0, 0, 1)|MOVE_ROBBER": 1.7001, "('move_robber', 6, 3, 6, 0, 1, 0)|MOVE_ROBBER": 1.6319, "('move_robber', 6, 3, 6, 0, 1, 1)|MOVE_ROBBER": 1.1485, "('move_robber', 6, 3, 6, 1, 0, 0)|MOVE_ROBBER": 0.7989, "('play_turn', 2, -1, 2, 0, 0, 0)|BUILD_ROAD": 1.0387, "('play_turn', 2, -1, 2, 0, 0, 0)|END_TURN": 0.5968, "('play_turn', 2, -1, 2, 0, 0, 0)|MARITIME_TRADE": 0.3737, "('play_turn', 2, -1, 2, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.4046, "('play_turn', 2, -1, 2, 0, 0, 1)|END_TURN": 0.3801, "('play_turn', 2, -1, 2, 0, 1, 0)|BUILD_SETTLEMENT": 0.9733, "('play_turn', 2, -1, 2, 0, 1, 1)|BUILD_SETTLEMENT": 0.7555, "('play_turn', 2, -1, 2, 1, 0, 0)|BUILD_CITY": 1.0015, "('play_turn', 2, -1, 3, 0, 0, 0)|BUILD_ROAD": 1.6593, "('play_turn', 2, -1, 3, 0, 0, 0)|END_TURN": 1.294, "('play_turn', 2, -1, 3, 0, 0, 0)|MARITIME_TRADE": 1.461, "('play_turn', 2, -1, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.5479, "('play_turn', 2, -1, 3, 0, 0, 0)|PLAY_MONOPOLY": 1.0234, "('play_turn', 2, -1, 3, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.3703, "('play_turn', 2, -1, 3, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.4057, "('play_turn', 2, -1, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.4353, "('play_turn', 2, -1, 3, 0, 0, 1)|END_TURN": 0.8971, "('play_turn', 2, -1, 3, 0, 0, 1)|MARITIME_TRADE": 0.3727, "('play_turn', 2, -1, 3, 0, 1, 0)|BUILD_ROAD": 1.6873, "('play_turn', 2, -1, 3, 0, 1, 0)|BUILD_SETTLEMENT": 2.1438, "('play_turn', 2, -1, 3, 0, 1, 0)|END_TURN": 0.331, "('play_turn', 2, -1, 3, 0, 1, 1)|BUILD_ROAD": 0.9686, "('play_turn', 2, -1, 3, 0, 1, 1)|BUILD_SETTLEMENT": 1.9592, "('play_turn', 2, -1, 3, 1, 0, 0)|BUILD_CITY": 2.4847, "('play_turn', 2, -1, 3, 1, 0, 0)|END_TURN": 0.716, "('play_turn', 2, -1, 3, 1, 0, 1)|BUILD_CITY": 0.9309, "('play_turn', 2, -1, 4, 0, 0, 0)|BUILD_ROAD": 1.6997, "('play_turn', 2, -1, 4, 0, 0, 0)|END_TURN": 1.385, "('play_turn', 2, -1, 4, 0, 0, 0)|MARITIME_TRADE": 1.7811, "('play_turn', 2, -1, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.6965, "('play_turn', 2, -1, 4, 0, 0, 0)|PLAY_MONOPOLY": 1.6536, "('play_turn', 2, -1, 4, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.8164, "('play_turn', 2, -1, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.6981, "('play_turn', 2, -1, 4, 0, 0, 0)|ROLL": 0.2747, "('play_turn', 2, -1, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.6319, "('play_turn', 2, -1, 4, 0, 0, 1)|END_TURN": 1.2467, "('play_turn', 2, -1, 4, 0, 0, 1)|MARITIME_TRADE": 1.7683, "('play_turn', 2, -1, 4, 0, 1, 0)|BUILD_ROAD": 1.9207, "('play_turn', 2, -1, 4, 0, 1, 0)|BUILD_SETTLEMENT": 2.6525, "('play_turn', 2, -1, 4, 0, 1, 0)|END_TURN": 0.8584, "('play_turn', 2, -1, 4, 0, 1, 0)|MARITIME_TRADE": 0.5081, "('play_turn', 2, -1, 4, 0, 1, 0)|PLAY_ROAD_BUILDING": 0.2772, "('play_turn', 2, -1, 4, 0, 1, 1)|BUILD_ROAD": 1.4527, "('play_turn', 2, -1, 4, 0, 1, 1)|BUILD_SETTLEMENT": 2.5414, "('play_turn', 2, -1, 4, 0, 1, 1)|END_TURN": 0.8581, "('play_turn', 2, -1, 4, 1, 0, 0)|BUILD_CITY": 2.6633, "('play_turn', 2, -1, 4, 1, 0, 0)|END_TURN": 0.5538, "('play_turn', 2, -1, 4, 1, 0, 1)|BUILD_CITY": 1.9348, "('play_turn', 2, -1, 5, 0, 0, 0)|BUILD_ROAD": 1.3666, "('play_turn', 2, -1, 5, 0, 0, 0)|END_TURN": 1.1656, "('play_turn', 2, -1, 5, 0, 0, 0)|MARITIME_TRADE": 1.1821, "('play_turn', 2, -1, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.2047, "('play_turn', 2, -1, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.546, "('play_turn', 2, -1, 5, 0, 0, 0)|ROLL": 0.1201, "('play_turn', 2, -1, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.5567, "('play_turn', 2, -1, 5, 0, 0, 1)|END_TURN": 0.4421, "('play_turn', 2, -1, 5, 0, 0, 1)|MARITIME_TRADE": 0.1777, "('play_turn', 2, -1, 5, 0, 1, 0)|BUILD_SETTLEMENT": 2.2984, "('play_turn', 2, -1, 5, 0, 1, 1)|BUILD_SETTLEMENT": 1.7402, "('play_turn', 2, -1, 5, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.6256, "('play_turn', 2, -1, 5, 1, 0, 0)|BUILD_CITY": 1.9278, "('play_turn', 2, -2, 2, 0, 0, 0)|BUILD_ROAD": 0.079, "('play_turn', 2, -2, 2, 0, 0, 0)|END_TURN": 0.3634, "('play_turn', 2, -2, 2, 0, 0, 1)|END_TURN": 0.0785, "('play_turn', 2, -2, 3, 0, 0, 0)|BUILD_ROAD": 1.2033, "('play_turn', 2, -2, 3, 0, 0, 0)|END_TURN": 0.971, "('play_turn', 2, -2, 3, 0, 0, 0)|MARITIME_TRADE": 1.1304, "('play_turn', 2, -2, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.0613, "('play_turn', 2, -2, 3, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.1803, "('play_turn', 2, -2, 3, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.7287, "('play_turn', 2, -2, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.9557, "('play_turn', 2, -2, 3, 0, 0, 1)|END_TURN": 0.3608, "('play_turn', 2, -2, 3, 0, 0, 1)|MARITIME_TRADE": 0.4138, "('play_turn', 2, -2, 3, 0, 1, 0)|BUILD_ROAD": 0.1382, "('play_turn', 2, -2, 3, 0, 1, 0)|BUILD_SETTLEMENT": 1.9136, "('play_turn', 2, -2, 3, 0, 1, 1)|BUILD_SETTLEMENT": 1.9683, "('play_turn', 2, -2, 3, 1, 0, 0)|BUILD_CITY": 1.8138, "('play_turn', 2, -2, 4, 0, 0, 0)|BUILD_ROAD": 1.5273, "('play_turn', 2, -2, 4, 0, 0, 0)|END_TURN": 1.2004, "('play_turn', 2, -2, 4, 0, 0, 0)|MARITIME_TRADE": 1.4543, "('play_turn', 2, -2, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.1338, "('play_turn', 2, -2, 4, 0, 0, 0)|PLAY_MONOPOLY": 0.7446, "('play_turn', 2, -2, 4, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.1508, "('play_turn', 2, -2, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.859, "('play_turn', 2, -2, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.2149, "('play_turn', 2, -2, 4, 0, 0, 1)|END_TURN": 0.4938, "('play_turn', 2, -2, 4, 0, 0, 1)|MARITIME_TRADE": 0.9056, "('play_turn', 2, -2, 4, 0, 1, 0)|BUILD_ROAD": 0.891, "('play_turn', 2, -2, 4, 0, 1, 0)|BUILD_SETTLEMENT": 2.1753, "('play_turn', 2, -2, 4, 0, 1, 0)|END_TURN": 0.2189, "('play_turn', 2, -2, 4, 0, 1, 1)|BUILD_ROAD": 0.4664, "('play_turn', 2, -2, 4, 0, 1, 1)|BUILD_SETTLEMENT": 2.0056, "('play_turn', 2, -2, 4, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.9614, "('play_turn', 2, -2, 4, 1, 0, 0)|BUILD_CITY": 1.9501, "('play_turn', 2, -2, 4, 1, 0, 1)|BUILD_CITY": 1.1637, "('play_turn', 2, -2, 5, 0, 0, 0)|BUILD_ROAD": 0.8292, "('play_turn', 2, -2, 5, 0, 0, 0)|END_TURN": 0.6981, "('play_turn', 2, -2, 5, 0, 0, 0)|MARITIME_TRADE": 0.2622, "('play_turn', 2, -2, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.6595, "('play_turn', 2, -2, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.8521, "('play_turn', 2, -2, 5, 0, 0, 1)|END_TURN": 0.0865, "('play_turn', 2, -2, 5, 0, 1, 0)|BUILD_SETTLEMENT": 0.9918, "('play_turn', 2, -2, 5, 1, 0, 0)|BUILD_CITY": 0.9486, "('play_turn', 2, -3, 2, 0, 0, 0)|BUILD_ROAD": 0.0335, "('play_turn', 2, -3, 2, 0, 0, 0)|END_TURN": 0.0493, "('play_turn', 2, -3, 2, 0, 0, 0)|MARITIME_TRADE": 0.0726, "('play_turn', 2, -3, 2, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.2988, "('play_turn', 2, -3, 3, 0, 0, 0)|BUILD_ROAD": 0.5195, "('play_turn', 2, -3, 3, 0, 0, 0)|END_TURN": 0.2073, "('play_turn', 2, -3, 3, 0, 0, 0)|MARITIME_TRADE": 0.2992, "('play_turn', 2, -3, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.8914, "('play_turn', 2, -3, 3, 0, 0, 0)|PLAY_MONOPOLY": 0.2052, "('play_turn', 2, -3, 3, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.0157, "('play_turn', 2, -3, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.6765, "('play_turn', 2, -3, 3, 0, 0, 1)|END_TURN": -0.0714, "('play_turn', 2, -3, 3, 0, 1, 0)|BUILD_ROAD": 0.3602, "('play_turn', 2, -3, 3, 0, 1, 0)|BUILD_SETTLEMENT": 0.6448, "('play_turn', 2, -3, 3, 0, 1, 1)|BUILD_SETTLEMENT": 0.7547, "('play_turn', 2, -3, 3, 1, 0, 0)|BUILD_CITY": 1.3294, "('play_turn', 2, -3, 4, 0, 0, 0)|BUILD_ROAD": 0.7401, "('play_turn', 2, -3, 4, 0, 0, 0)|END_TURN": 0.634, "('play_turn', 2, -3, 4, 0, 0, 0)|MARITIME_TRADE": 0.7087, "('play_turn', 2, -3, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.9842, "('play_turn', 2, -3, 4, 0, 0, 0)|PLAY_MONOPOLY": 0.2856, "('play_turn', 2, -3, 4, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.1109, "('play_turn', 2, -3, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.4752, "('play_turn', 2, -3, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.8038, "('play_turn', 2, -3, 4, 0, 0, 1)|END_TURN": 0.36, "('play_turn', 2, -3, 4, 0, 1, 0)|BUILD_ROAD": 0.8194, "('play_turn', 2, -3, 4, 0, 1, 0)|BUILD_SETTLEMENT": 1.3737, "('play_turn', 2, -3, 4, 0, 1, 0)|END_TURN": 0.1049, "('play_turn', 2, -3, 4, 0, 1, 1)|BUILD_SETTLEMENT": 1.3149, "('play_turn', 2, -3, 4, 1, 0, 0)|BUILD_CITY": 1.5753, "('play_turn', 2, -3, 4, 1, 0, 1)|BUILD_CITY": 0.7007, "('play_turn', 2, -3, 5, 0, 0, 0)|BUILD_ROAD": -0.0021, "('play_turn', 2, -3, 5, 0, 0, 0)|END_TURN": 0.1019, "('play_turn', 2, -3, 5, 0, 0, 0)|MARITIME_TRADE": 0.1135, "('play_turn', 2, -3, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.4856, "('play_turn', 2, -3, 5, 0, 0, 1)|END_TURN": 0.1217, "('play_turn', 2, -3, 5, 0, 1, 1)|BUILD_SETTLEMENT": 0.5375, "('play_turn', 2, 0, 2, 0, 0, 0)|BUILD_ROAD": 1.4366, "('play_turn', 2, 0, 2, 0, 0, 0)|END_TURN": 1.5322, "('play_turn', 2, 0, 2, 0, 0, 0)|MARITIME_TRADE": 0.5918, "('play_turn', 2, 0, 2, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.4887, "('play_turn', 2, 0, 2, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.2905, "('play_turn', 2, 0, 2, 0, 0, 1)|END_TURN": 0.115, "('play_turn', 2, 0, 2, 0, 1, 0)|BUILD_SETTLEMENT": 1.229, "('play_turn', 2, 0, 2, 1, 0, 0)|BUILD_CITY": 1.3751, "('play_turn', 2, 0, 3, 0, 0, 0)|BUILD_ROAD": 1.606, "('play_turn', 2, 0, 3, 0, 0, 0)|END_TURN": 1.7683, "('play_turn', 2, 0, 3, 0, 0, 0)|MARITIME_TRADE": 1.9067, "('play_turn', 2, 0, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.9361, "('play_turn', 2, 0, 3, 0, 0, 0)|PLAY_MONOPOLY": 1.8301, "('play_turn', 2, 0, 3, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.5664, "('play_turn', 2, 0, 3, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 2.0219, "('play_turn', 2, 0, 3, 0, 0, 0)|ROLL": 0.9496, "('play_turn', 2, 0, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.7777, "('play_turn', 2, 0, 3, 0, 0, 1)|END_TURN": 1.7497, "('play_turn', 2, 0, 3, 0, 0, 1)|MARITIME_TRADE": 1.6671, "('play_turn', 2, 0, 3, 0, 0, 1)|PLAY_KNIGHT_CARD": 0.8418, "('play_turn', 2, 0, 3, 0, 1, 0)|BUILD_ROAD": 1.7593, "('play_turn', 2, 0, 3, 0, 1, 0)|BUILD_SETTLEMENT": 2.6195, "('play_turn', 2, 0, 3, 0, 1, 0)|END_TURN": 1.2577, "('play_turn', 2, 0, 3, 0, 1, 1)|BUILD_ROAD": 0.6471, "('play_turn', 2, 0, 3, 0, 1, 1)|BUILD_SETTLEMENT": 2.9101, "('play_turn', 2, 0, 3, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 1.671, "('play_turn', 2, 0, 3, 0, 1, 1)|END_TURN": 0.9424, "('play_turn', 2, 0, 3, 1, 0, 0)|BUILD_CITY": 2.9462, "('play_turn', 2, 0, 3, 1, 0, 0)|END_TURN": 0.9354, "('play_turn', 2, 0, 3, 1, 0, 1)|BUILD_CITY": 2.2947, "('play_turn', 2, 0, 4, 0, 0, 0)|BUILD_ROAD": 2.0561, "('play_turn', 2, 0, 4, 0, 0, 0)|END_TURN": 2.106, "('play_turn', 2, 0, 4, 0, 0, 0)|MARITIME_TRADE": 2.1879, "('play_turn', 2, 0, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.1384, "('play_turn', 2, 0, 4, 0, 0, 0)|PLAY_MONOPOLY": 2.2151, "('play_turn', 2, 0, 4, 0, 0, 0)|PLAY_ROAD_BUILDING": 1.4412, "('play_turn', 2, 0, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 2.1471, "('play_turn', 2, 0, 4, 0, 0, 0)|ROLL": 1.5469, "('play_turn', 2, 0, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 2.209, "('play_turn', 2, 0, 4, 0, 0, 1)|END_TURN": 2.1453, "('play_turn', 2, 0, 4, 0, 0, 1)|MARITIME_TRADE": 2.5474, "('play_turn', 2, 0, 4, 0, 0, 1)|PLAY_KNIGHT_CARD": 1.8635, "('play_turn', 2, 0, 4, 0, 0, 1)|PLAY_MONOPOLY": 0.834, "('play_turn', 2, 0, 4, 0, 0, 1)|PLAY_YEAR_OF_PLENTY": 1.7609, "('play_turn', 2, 0, 4, 0, 1, 0)|BUILD_ROAD": 2.0755, "('play_turn', 2, 0, 4, 0, 1, 0)|BUILD_SETTLEMENT": 3.1609, "('play_turn', 2, 0, 4, 0, 1, 0)|END_TURN": 2.0948, "('play_turn', 2, 0, 4, 0, 1, 0)|MARITIME_TRADE": 0.66, "('play_turn', 2, 0, 4, 0, 1, 0)|PLAY_ROAD_BUILDING": 0.422, "('play_turn', 2, 0, 4, 0, 1, 1)|BUILD_ROAD": 2.2636, "('play_turn', 2, 0, 4, 0, 1, 1)|BUILD_SETTLEMENT": 3.1594, "('play_turn', 2, 0, 4, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 1.9239, "('play_turn', 2, 0, 4, 0, 1, 1)|END_TURN": 2.0389, "('play_turn', 2, 0, 4, 0, 1, 1)|MARITIME_TRADE": 0.9901, "('play_turn', 2, 0, 4, 1, 0, 0)|BUILD_CITY": 3.2143, "('play_turn', 2, 0, 4, 1, 0, 0)|BUILD_ROAD": 0.959, "('play_turn', 2, 0, 4, 1, 0, 0)|END_TURN": 1.957, "('play_turn', 2, 0, 4, 1, 0, 0)|MARITIME_TRADE": 1.2036, "('play_turn', 2, 0, 4, 1, 0, 1)|BUILD_CITY": 3.2598, "('play_turn', 2, 0, 4, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 1.0642, "('play_turn', 2, 0, 4, 1, 1, 1)|BUILD_CITY": 1.3856, "('play_turn', 2, 0, 4, 1, 1, 1)|BUILD_SETTLEMENT": 1.9251, "('play_turn', 2, 0, 5, 0, 0, 0)|BUILD_ROAD": 1.9091, "('play_turn', 2, 0, 5, 0, 0, 0)|END_TURN": 2.0329, "('play_turn', 2, 0, 5, 0, 0, 0)|MARITIME_TRADE": 2.3301, "('play_turn', 2, 0, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.0068, "('play_turn', 2, 0, 5, 0, 0, 0)|PLAY_MONOPOLY": 1.871, "('play_turn', 2, 0, 5, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.1282, "('play_turn', 2, 0, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.967, "('play_turn', 2, 0, 5, 0, 0, 0)|ROLL": 0.3354, "('play_turn', 2, 0, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 2.1786, "('play_turn', 2, 0, 5, 0, 0, 1)|END_TURN": 1.7568, "('play_turn', 2, 0, 5, 0, 0, 1)|MARITIME_TRADE": 0.9145, "('play_turn', 2, 0, 5, 0, 1, 0)|BUILD_ROAD": 2.2147, "('play_turn', 2, 0, 5, 0, 1, 0)|BUILD_SETTLEMENT": 3.1008, "('play_turn', 2, 0, 5, 0, 1, 0)|END_TURN": 1.4938, "('play_turn', 2, 0, 5, 0, 1, 1)|BUILD_ROAD": 0.5789, "('play_turn', 2, 0, 5, 0, 1, 1)|BUILD_SETTLEMENT": 3.2321, "('play_turn', 2, 0, 5, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 2.1266, "('play_turn', 2, 0, 5, 0, 1, 1)|END_TURN": 0.4213, "('play_turn', 2, 0, 5, 1, 0, 0)|BUILD_CITY": 3.0897, "('play_turn', 2, 0, 5, 1, 0, 1)|BUILD_CITY": 1.9941, "('play_turn', 3, -1, 2, 0, 0, 0)|BUILD_ROAD": -0.0547, "('play_turn', 3, -1, 2, 0, 0, 0)|END_TURN": 0.1005, "('play_turn', 3, -1, 2, 0, 0, 1)|END_TURN": 0.0, "('play_turn', 3, -1, 3, 0, 0, 0)|BUILD_ROAD": 0.9683, "('play_turn', 3, -1, 3, 0, 0, 0)|END_TURN": 0.5313, "('play_turn', 3, -1, 3, 0, 0, 0)|MARITIME_TRADE": 0.4002, "('play_turn', 3, -1, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.0017, "('play_turn', 3, -1, 3, 0, 0, 0)|PLAY_MONOPOLY": 0.1299, "('play_turn', 3, -1, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.4787, "('play_turn', 3, -1, 3, 0, 0, 1)|END_TURN": 0.1966, "('play_turn', 3, -1, 3, 0, 1, 0)|BUILD_SETTLEMENT": 1.3493, "('play_turn', 3, -1, 3, 0, 1, 1)|BUILD_SETTLEMENT": 0.5896, "('play_turn', 3, -1, 3, 1, 0, 0)|BUILD_CITY": 0.7966, "('play_turn', 3, -1, 3, 1, 0, 1)|BUILD_CITY": 0.8787, "('play_turn', 3, -1, 4, 0, 0, 0)|BUILD_ROAD": 1.2206, "('play_turn', 3, -1, 4, 0, 0, 0)|END_TURN": 1.0544, "('play_turn', 3, -1, 4, 0, 0, 0)|MARITIME_TRADE": 1.2753, "('play_turn', 3, -1, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.2424, "('play_turn', 3, -1, 4, 0, 0, 0)|PLAY_MONOPOLY": 0.5679, "('play_turn', 3, -1, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.7171, "('play_turn', 3, -1, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.2974, "('play_turn', 3, -1, 4, 0, 0, 1)|END_TURN": 0.4562, "('play_turn', 3, -1, 4, 0, 0, 1)|MARITIME_TRADE": 0.586, "('play_turn', 3, -1, 4, 0, 1, 0)|BUILD_ROAD": 0.6942, "('play_turn', 3, -1, 4, 0, 1, 0)|BUILD_SETTLEMENT": 1.9581, "('play_turn', 3, -1, 4, 0, 1, 0)|END_TURN": 0.7566, "('play_turn', 3, -1, 4, 0, 1, 1)|BUILD_SETTLEMENT": 1.9373, "('play_turn', 3, -1, 4, 1, 0, 0)|BUILD_CITY": 2.0347, "('play_turn', 3, -1, 4, 1, 0, 1)|BUILD_CITY": 0.4614, "('play_turn', 3, -1, 5, 0, 0, 0)|BUILD_ROAD": 1.416, "('play_turn', 3, -1, 5, 0, 0, 0)|END_TURN": 1.1021, "('play_turn', 3, -1, 5, 0, 0, 0)|MARITIME_TRADE": 1.0691, "('play_turn', 3, -1, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.5043, "('play_turn', 3, -1, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.6492, "('play_turn', 3, -1, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.6498, "('play_turn', 3, -1, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.4919, "('play_turn', 3, -1, 5, 0, 0, 1)|END_TURN": 0.4706, "('play_turn', 3, -1, 5, 0, 0, 1)|MARITIME_TRADE": 0.8499, "('play_turn', 3, -1, 5, 0, 1, 0)|BUILD_ROAD": 1.3918, "('play_turn', 3, -1, 5, 0, 1, 0)|BUILD_SETTLEMENT": 2.0484, "('play_turn', 3, -1, 5, 0, 1, 0)|END_TURN": 0.3909, "('play_turn', 3, -1, 5, 0, 1, 1)|BUILD_SETTLEMENT": 1.9001, "('play_turn', 3, -1, 5, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.7851, "('play_turn', 3, -1, 5, 1, 0, 0)|BUILD_CITY": 1.9808, "('play_turn', 3, -1, 5, 1, 0, 1)|BUILD_CITY": 0.7162, "('play_turn', 3, -1, 6, 0, 0, 0)|BUILD_ROAD": 1.6121, "('play_turn', 3, -1, 6, 0, 0, 0)|END_TURN": 1.2155, "('play_turn', 3, -1, 6, 0, 0, 0)|MARITIME_TRADE": 1.767, "('play_turn', 3, -1, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.9468, "('play_turn', 3, -1, 6, 0, 0, 0)|PLAY_MONOPOLY": 0.9786, "('play_turn', 3, -1, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.6135, "('play_turn', 3, -1, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.3384, "('play_turn', 3, -1, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.9237, "('play_turn', 3, -1, 6, 0, 0, 1)|END_TURN": 1.1794, "('play_turn', 3, -1, 6, 0, 0, 1)|MARITIME_TRADE": 0.8519, "('play_turn', 3, -1, 6, 0, 1, 0)|BUILD_ROAD": 1.6548, "('play_turn', 3, -1, 6, 0, 1, 0)|BUILD_SETTLEMENT": 2.4388, "('play_turn', 3, -1, 6, 0, 1, 0)|END_TURN": 0.5135, "('play_turn', 3, -1, 6, 0, 1, 1)|BUILD_ROAD": 1.5621, "('play_turn', 3, -1, 6, 0, 1, 1)|BUILD_SETTLEMENT": 2.2311, "('play_turn', 3, -1, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.7572, "('play_turn', 3, -1, 6, 1, 0, 0)|BUILD_CITY": 2.4827, "('play_turn', 3, -1, 6, 1, 0, 0)|MARITIME_TRADE": 0.5256, "('play_turn', 3, -1, 6, 1, 0, 1)|BUILD_CITY": 1.7677, "('play_turn', 3, -2, 3, 0, 0, 0)|BUILD_ROAD": 0.3202, "('play_turn', 3, -2, 3, 0, 0, 0)|END_TURN": 0.1065, "('play_turn', 3, -2, 3, 0, 0, 0)|MARITIME_TRADE": 0.1662, "('play_turn', 3, -2, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.3922, "('play_turn', 3, -2, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.0193, "('play_turn', 3, -2, 3, 0, 0, 1)|END_TURN": -0.1495, "('play_turn', 3, -2, 3, 0, 1, 0)|BUILD_SETTLEMENT": 0.5188, "('play_turn', 3, -2, 3, 1, 0, 0)|BUILD_CITY": 0.4326, "('play_turn', 3, -2, 4, 0, 0, 0)|BUILD_ROAD": 1.1782, "('play_turn', 3, -2, 4, 0, 0, 0)|END_TURN": 0.5725, "('play_turn', 3, -2, 4, 0, 0, 0)|MARITIME_TRADE": 0.5906, "('play_turn', 3, -2, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.8656, "('play_turn', 3, -2, 4, 0, 0, 0)|PLAY_MONOPOLY": 0.2583, "('play_turn', 3, -2, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.4312, "('play_turn', 3, -2, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.6645, "('play_turn', 3, -2, 4, 0, 0, 1)|END_TURN": 0.1531, "('play_turn', 3, -2, 4, 0, 0, 1)|MARITIME_TRADE": 0.368, "('play_turn', 3, -2, 4, 0, 1, 0)|BUILD_ROAD": 0.5432, "('play_turn', 3, -2, 4, 0, 1, 0)|BUILD_SETTLEMENT": 1.1271, "('play_turn', 3, -2, 4, 0, 1, 1)|BUILD_ROAD": 0.6851, "('play_turn', 3, -2, 4, 0, 1, 1)|BUILD_SETTLEMENT": 1.2303, "('play_turn', 3, -2, 4, 1, 0, 0)|BUILD_CITY": 1.0541, "('play_turn', 3, -2, 5, 0, 0, 0)|BUILD_ROAD": 1.0662, "('play_turn', 3, -2, 5, 0, 0, 0)|END_TURN": 0.4375, "('play_turn', 3, -2, 5, 0, 0, 0)|MARITIME_TRADE": 0.6939, "('play_turn', 3, -2, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.105, "('play_turn', 3, -2, 5, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.1731, "('play_turn', 3, -2, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.2603, "('play_turn', 3, -2, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.8089, "('play_turn', 3, -2, 5, 0, 0, 1)|END_TURN": 0.1817, "('play_turn', 3, -2, 5, 0, 1, 0)|BUILD_ROAD": 0.5458, "('play_turn', 3, -2, 5, 0, 1, 0)|BUILD_SETTLEMENT": 1.1783, "('play_turn', 3, -2, 5, 0, 1, 1)|BUILD_SETTLEMENT": 1.18, "('play_turn', 3, -2, 5, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.3754, "('play_turn', 3, -2, 5, 1, 0, 0)|BUILD_CITY": 1.4212, "('play_turn', 3, -2, 5, 1, 0, 1)|BUILD_CITY": 1.0179, "('play_turn', 3, -2, 6, 0, 0, 0)|BUILD_ROAD": 1.0457, "('play_turn', 3, -2, 6, 0, 0, 0)|END_TURN": 0.8027, "('play_turn', 3, -2, 6, 0, 0, 0)|MARITIME_TRADE": 1.1181, "('play_turn', 3, -2, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.0638, "('play_turn', 3, -2, 6, 0, 0, 0)|PLAY_MONOPOLY": 0.4074, "('play_turn', 3, -2, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.3586, "('play_turn', 3, -2, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.4879, "('play_turn', 3, -2, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.1119, "('play_turn', 3, -2, 6, 0, 0, 1)|END_TURN": 0.2999, "('play_turn', 3, -2, 6, 0, 0, 1)|MARITIME_TRADE": 0.6031, "('play_turn', 3, -2, 6, 0, 1, 0)|BUILD_ROAD": 0.6259, "('play_turn', 3, -2, 6, 0, 1, 0)|BUILD_SETTLEMENT": 1.6777, "('play_turn', 3, -2, 6, 0, 1, 1)|BUILD_ROAD": 0.5687, "('play_turn', 3, -2, 6, 0, 1, 1)|BUILD_SETTLEMENT": 1.4434, "('play_turn', 3, -2, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.5469, "('play_turn', 3, -2, 6, 1, 0, 0)|BUILD_CITY": 1.5616, "('play_turn', 3, -2, 6, 1, 0, 1)|BUILD_CITY": 1.2457, "('play_turn', 3, -3, 3, 0, 0, 0)|END_TURN": -0.2768, "('play_turn', 3, -3, 3, 0, 0, 0)|MARITIME_TRADE": -0.0072, "('play_turn', 3, -3, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.124, "('play_turn', 3, -3, 3, 0, 0, 1)|END_TURN": -0.2979, "('play_turn', 3, -3, 4, 0, 0, 0)|BUILD_ROAD": 0.0404, "('play_turn', 3, -3, 4, 0, 0, 0)|END_TURN": -0.2636, "('play_turn', 3, -3, 4, 0, 0, 0)|MARITIME_TRADE": 0.0937, "('play_turn', 3, -3, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.6028, "('play_turn', 3, -3, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.225, "('play_turn', 3, -3, 4, 0, 0, 1)|END_TURN": -0.1937, "('play_turn', 3, -3, 4, 0, 1, 0)|BUILD_SETTLEMENT": 0.3082, "('play_turn', 3, -3, 4, 0, 1, 1)|BUILD_SETTLEMENT": 0.4631, "('play_turn', 3, -3, 4, 1, 0, 0)|BUILD_CITY": 0.3351, "('play_turn', 3, -3, 5, 0, 0, 0)|BUILD_ROAD": 0.5062, "('play_turn', 3, -3, 5, 0, 0, 0)|END_TURN": -0.3396, "('play_turn', 3, -3, 5, 0, 0, 0)|MARITIME_TRADE": 0.0525, "('play_turn', 3, -3, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.1601, "('play_turn', 3, -3, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.2884, "('play_turn', 3, -3, 5, 0, 1, 0)|BUILD_SETTLEMENT": 0.7125, "('play_turn', 3, -3, 5, 1, 0, 0)|BUILD_CITY": 0.6636, "('play_turn', 3, -3, 6, 0, 0, 0)|BUILD_ROAD": 0.3695, "('play_turn', 3, -3, 6, 0, 0, 0)|END_TURN": -0.1329, "('play_turn', 3, -3, 6, 0, 0, 0)|MARITIME_TRADE": 0.3744, "('play_turn', 3, -3, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.4787, "('play_turn', 3, -3, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.0395, "('play_turn', 3, -3, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.5262, "('play_turn', 3, -3, 6, 0, 0, 1)|END_TURN": 0.0575, "('play_turn', 3, -3, 6, 0, 0, 1)|MARITIME_TRADE": 0.17, "('play_turn', 3, -3, 6, 0, 1, 0)|BUILD_ROAD": 0.2274, "('play_turn', 3, -3, 6, 0, 1, 0)|BUILD_SETTLEMENT": 0.9136, "('play_turn', 3, -3, 6, 0, 1, 1)|BUILD_ROAD": 0.2947, "('play_turn', 3, -3, 6, 0, 1, 1)|BUILD_SETTLEMENT": 0.6444, "('play_turn', 3, -3, 6, 1, 0, 0)|BUILD_CITY": 1.0276, "('play_turn', 3, 0, 2, 0, 0, 0)|END_TURN": 0.0584, "('play_turn', 3, 0, 3, 0, 0, 0)|BUILD_ROAD": 0.8665, "('play_turn', 3, 0, 3, 0, 0, 0)|END_TURN": 0.7299, "('play_turn', 3, 0, 3, 0, 0, 0)|MARITIME_TRADE": 0.6272, "('play_turn', 3, 0, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.9811, "('play_turn', 3, 0, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.7459, "('play_turn', 3, 0, 3, 0, 0, 1)|END_TURN": 0.3214, "('play_turn', 3, 0, 3, 0, 1, 0)|BUILD_ROAD": 0.3395, "('play_turn', 3, 0, 3, 0, 1, 0)|BUILD_SETTLEMENT": 1.5909, "('play_turn', 3, 0, 3, 0, 1, 1)|BUILD_SETTLEMENT": 1.1838, "('play_turn', 3, 0, 3, 1, 0, 0)|BUILD_CITY": 1.4505, "('play_turn', 3, 0, 4, 0, 0, 0)|BUILD_ROAD": 1.5598, "('play_turn', 3, 0, 4, 0, 0, 0)|END_TURN": 1.4271, "('play_turn', 3, 0, 4, 0, 0, 0)|MARITIME_TRADE": 1.621, "('play_turn', 3, 0, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.029, "('play_turn', 3, 0, 4, 0, 0, 0)|PLAY_MONOPOLY": 0.5475, "('play_turn', 3, 0, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.0917, "('play_turn', 3, 0, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.8027, "('play_turn', 3, 0, 4, 0, 0, 1)|END_TURN": 0.4335, "('play_turn', 3, 0, 4, 0, 0, 1)|MARITIME_TRADE": 0.3181, "('play_turn', 3, 0, 4, 0, 1, 0)|BUILD_ROAD": 0.8991, "('play_turn', 3, 0, 4, 0, 1, 0)|BUILD_SETTLEMENT": 2.0172, "('play_turn', 3, 0, 4, 0, 1, 0)|END_TURN": 0.2709, "('play_turn', 3, 0, 4, 0, 1, 1)|BUILD_SETTLEMENT": 2.0619, "('play_turn', 3, 0, 4, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.7884, "('play_turn', 3, 0, 4, 1, 0, 0)|BUILD_CITY": 2.3211, "('play_turn', 3, 0, 4, 1, 0, 1)|BUILD_CITY": 1.2017, "('play_turn', 3, 0, 5, 0, 0, 0)|BUILD_ROAD": 1.6001, "('play_turn', 3, 0, 5, 0, 0, 0)|END_TURN": 1.4281, "('play_turn', 3, 0, 5, 0, 0, 0)|MARITIME_TRADE": 1.5577, "('play_turn', 3, 0, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.8269, "('play_turn', 3, 0, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.7996, "('play_turn', 3, 0, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.4037, "('play_turn', 3, 0, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.9246, "('play_turn', 3, 0, 5, 0, 0, 1)|END_TURN": 0.9256, "('play_turn', 3, 0, 5, 0, 0, 1)|MARITIME_TRADE": 0.4322, "('play_turn', 3, 0, 5, 0, 0, 1)|PLAY_KNIGHT_CARD": 0.5545, "('play_turn', 3, 0, 5, 0, 1, 0)|BUILD_ROAD": 1.6833, "('play_turn', 3, 0, 5, 0, 1, 0)|BUILD_SETTLEMENT": 2.5676, "('play_turn', 3, 0, 5, 0, 1, 0)|END_TURN": 0.0927, "('play_turn', 3, 0, 5, 0, 1, 1)|BUILD_SETTLEMENT": 2.2739, "('play_turn', 3, 0, 5, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 1.2686, "('play_turn', 3, 0, 5, 1, 0, 0)|BUILD_CITY": 2.6162, "('play_turn', 3, 0, 5, 1, 0, 0)|END_TURN": 0.4014, "('play_turn', 3, 0, 5, 1, 0, 1)|BUILD_CITY": 1.5241, "('play_turn', 3, 0, 6, 0, 0, 0)|BUILD_ROAD": 2.0309, "('play_turn', 3, 0, 6, 0, 0, 0)|END_TURN": 1.7841, "('play_turn', 3, 0, 6, 0, 0, 0)|MARITIME_TRADE": 1.9649, "('play_turn', 3, 0, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.1694, "('play_turn', 3, 0, 6, 0, 0, 0)|PLAY_MONOPOLY": 1.4664, "('play_turn', 3, 0, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.4768, "('play_turn', 3, 0, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.5159, "('play_turn', 3, 0, 6, 0, 0, 0)|ROLL": 0.2972, "('play_turn', 3, 0, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 2.1352, "('play_turn', 3, 0, 6, 0, 0, 1)|END_TURN": 1.2556, "('play_turn', 3, 0, 6, 0, 0, 1)|MARITIME_TRADE": 1.7435, "('play_turn', 3, 0, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 0.8367, "('play_turn', 3, 0, 6, 0, 1, 0)|BUILD_ROAD": 1.7686, "('play_turn', 3, 0, 6, 0, 1, 0)|BUILD_SETTLEMENT": 2.5633, "('play_turn', 3, 0, 6, 0, 1, 0)|END_TURN": 0.824, "('play_turn', 3, 0, 6, 0, 1, 0)|MARITIME_TRADE": 0.4911, "('play_turn', 3, 0, 6, 0, 1, 1)|BUILD_ROAD": 1.4131, "('play_turn', 3, 0, 6, 0, 1, 1)|BUILD_SETTLEMENT": 2.5935, "('play_turn', 3, 0, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 2.0735, "('play_turn', 3, 0, 6, 0, 1, 1)|END_TURN": 0.402, "('play_turn', 3, 0, 6, 1, 0, 0)|BUILD_CITY": 2.7144, "('play_turn', 3, 0, 6, 1, 0, 0)|END_TURN": 0.6808, "('play_turn', 3, 0, 6, 1, 0, 1)|BUILD_CITY": 2.4186, "('play_turn', 3, 1, 2, 0, 0, 0)|BUILD_ROAD": 0.1526, "('play_turn', 3, 1, 2, 0, 0, 0)|END_TURN": 0.4076, "('play_turn', 3, 1, 3, 0, 0, 0)|BUILD_ROAD": 1.4951, "('play_turn', 3, 1, 3, 0, 0, 0)|END_TURN": 1.421, "('play_turn', 3, 1, 3, 0, 0, 0)|MARITIME_TRADE": 1.371, "('play_turn', 3, 1, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.6498, "('play_turn', 3, 1, 3, 0, 0, 0)|PLAY_MONOPOLY": 0.3433, "('play_turn', 3, 1, 3, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.1105, "('play_turn', 3, 1, 3, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.7252, "('play_turn', 3, 1, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.3959, "('play_turn', 3, 1, 3, 0, 0, 1)|END_TURN": 0.3162, "('play_turn', 3, 1, 3, 0, 0, 1)|MARITIME_TRADE": 0.4158, "('play_turn', 3, 1, 3, 0, 1, 0)|BUILD_ROAD": 0.9373, "('play_turn', 3, 1, 3, 0, 1, 0)|BUILD_SETTLEMENT": 1.8756, "('play_turn', 3, 1, 3, 0, 1, 1)|BUILD_SETTLEMENT": 1.7884, "('play_turn', 3, 1, 3, 1, 0, 0)|BUILD_CITY": 2.3148, "('play_turn', 3, 1, 4, 0, 0, 0)|BUILD_ROAD": 1.8352, "('play_turn', 3, 1, 4, 0, 0, 0)|END_TURN": 1.71, "('play_turn', 3, 1, 4, 0, 0, 0)|MARITIME_TRADE": 2.0458, "('play_turn', 3, 1, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.0038, "('play_turn', 3, 1, 4, 0, 0, 0)|PLAY_MONOPOLY": 1.6776, "('play_turn', 3, 1, 4, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.7551, "('play_turn', 3, 1, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.8209, "('play_turn', 3, 1, 4, 0, 0, 0)|ROLL": 0.4365, "('play_turn', 3, 1, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.9848, "('play_turn', 3, 1, 4, 0, 0, 1)|END_TURN": 1.4078, "('play_turn', 3, 1, 4, 0, 0, 1)|MARITIME_TRADE": 1.8522, "('play_turn', 3, 1, 4, 0, 1, 0)|BUILD_ROAD": 1.8546, "('play_turn', 3, 1, 4, 0, 1, 0)|BUILD_SETTLEMENT": 2.8945, "('play_turn', 3, 1, 4, 0, 1, 0)|END_TURN": 0.8872, "('play_turn', 3, 1, 4, 0, 1, 1)|BUILD_ROAD": 0.8613, "('play_turn', 3, 1, 4, 0, 1, 1)|BUILD_SETTLEMENT": 2.9139, "('play_turn', 3, 1, 4, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 1.6055, "('play_turn', 3, 1, 4, 0, 1, 1)|END_TURN": 0.3537, "('play_turn', 3, 1, 4, 1, 0, 0)|BUILD_CITY": 2.9311, "('play_turn', 3, 1, 4, 1, 0, 1)|BUILD_CITY": 1.9267, "('play_turn', 3, 1, 5, 0, 0, 0)|BUILD_ROAD": 2.3, "('play_turn', 3, 1, 5, 0, 0, 0)|END_TURN": 1.9377, "('play_turn', 3, 1, 5, 0, 0, 0)|MARITIME_TRADE": 2.1951, "('play_turn', 3, 1, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.105, "('play_turn', 3, 1, 5, 0, 0, 0)|PLAY_MONOPOLY": 1.6198, "('play_turn', 3, 1, 5, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.3881, "('play_turn', 3, 1, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 2.0951, "('play_turn', 3, 1, 5, 0, 0, 0)|ROLL": 1.0181, "('play_turn', 3, 1, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 2.0842, "('play_turn', 3, 1, 5, 0, 0, 1)|END_TURN": 1.6316, "('play_turn', 3, 1, 5, 0, 0, 1)|MARITIME_TRADE": 1.614, "('play_turn', 3, 1, 5, 0, 0, 1)|PLAY_KNIGHT_CARD": 0.6373, "('play_turn', 3, 1, 5, 0, 0, 1)|PLAY_MONOPOLY": 0.6057, "('play_turn', 3, 1, 5, 0, 1, 0)|BUILD_ROAD": 2.0748, "('play_turn', 3, 1, 5, 0, 1, 0)|BUILD_SETTLEMENT": 2.9393, "('play_turn', 3, 1, 5, 0, 1, 0)|END_TURN": 0.8149, "('play_turn', 3, 1, 5, 0, 1, 0)|MARITIME_TRADE": 0.5654, "('play_turn', 3, 1, 5, 0, 1, 1)|BUILD_SETTLEMENT": 2.932, "('play_turn', 3, 1, 5, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 1.9119, "('play_turn', 3, 1, 5, 1, 0, 0)|BUILD_CITY": 3.1794, "('play_turn', 3, 1, 5, 1, 0, 0)|END_TURN": 0.594, "('play_turn', 3, 1, 5, 1, 0, 1)|BUILD_CITY": 2.254, "('play_turn', 3, 1, 6, 0, 0, 0)|BUILD_ROAD": 2.5231, "('play_turn', 3, 1, 6, 0, 0, 0)|END_TURN": 2.4612, "('play_turn', 3, 1, 6, 0, 0, 0)|MARITIME_TRADE": 2.4697, "('play_turn', 3, 1, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.4411, "('play_turn', 3, 1, 6, 0, 0, 0)|PLAY_MONOPOLY": 2.213, "('play_turn', 3, 1, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 1.7321, "('play_turn', 3, 1, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 2.3337, "('play_turn', 3, 1, 6, 0, 0, 0)|ROLL": 1.1545, "('play_turn', 3, 1, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 2.3018, "('play_turn', 3, 1, 6, 0, 0, 1)|END_TURN": 2.2969, "('play_turn', 3, 1, 6, 0, 0, 1)|MARITIME_TRADE": 2.6593, "('play_turn', 3, 1, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 1.9595, "('play_turn', 3, 1, 6, 0, 1, 0)|BUILD_ROAD": 2.8399, "('play_turn', 3, 1, 6, 0, 1, 0)|BUILD_SETTLEMENT": 3.0438, "('play_turn', 3, 1, 6, 0, 1, 0)|END_TURN": 1.9325, "('play_turn', 3, 1, 6, 0, 1, 1)|BUILD_ROAD": 1.764, "('play_turn', 3, 1, 6, 0, 1, 1)|BUILD_SETTLEMENT": 3.038, "('play_turn', 3, 1, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 2.529, "('play_turn', 3, 1, 6, 0, 1, 1)|END_TURN": 0.9632, "('play_turn', 3, 1, 6, 0, 1, 1)|PLAY_ROAD_BUILDING": 0.1546, "('play_turn', 3, 1, 6, 1, 0, 0)|BUILD_CITY": 3.2351, "('play_turn', 3, 1, 6, 1, 0, 0)|END_TURN": 1.905, "('play_turn', 3, 1, 6, 1, 0, 0)|MARITIME_TRADE": 1.1657, "('play_turn', 3, 1, 6, 1, 0, 1)|BUILD_CITY": 2.9763, "('play_turn', 3, 1, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 0.6237, "('play_turn', 3, 1, 6, 1, 0, 1)|END_TURN": 0.8687, "('play_turn', 4, -1, 2, 0, 0, 0)|END_TURN": -0.1013, "('play_turn', 4, -1, 3, 0, 0, 0)|BUILD_ROAD": -0.0397, "('play_turn', 4, -1, 3, 0, 0, 0)|END_TURN": -0.3152, "('play_turn', 4, -1, 3, 0, 0, 0)|MARITIME_TRADE": -0.0325, "('play_turn', 4, -1, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.3237, "('play_turn', 4, -1, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.2278, "('play_turn', 4, -1, 3, 0, 0, 1)|END_TURN": -0.1559, "('play_turn', 4, -1, 3, 0, 1, 0)|BUILD_SETTLEMENT": 0.7304, "('play_turn', 4, -1, 3, 1, 0, 0)|BUILD_CITY": 0.3107, "('play_turn', 4, -1, 4, 0, 0, 0)|BUILD_ROAD": 0.2688, "('play_turn', 4, -1, 4, 0, 0, 0)|END_TURN": 0.1583, "('play_turn', 4, -1, 4, 0, 0, 0)|MARITIME_TRADE": 0.6553, "('play_turn', 4, -1, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.4521, "('play_turn', 4, -1, 4, 0, 0, 0)|PLAY_MONOPOLY": -0.1009, "('play_turn', 4, -1, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.2241, "('play_turn', 4, -1, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.4007, "('play_turn', 4, -1, 4, 0, 0, 1)|END_TURN": 0.2971, "('play_turn', 4, -1, 4, 0, 0, 1)|MARITIME_TRADE": 0.2672, "('play_turn', 4, -1, 4, 0, 1, 0)|BUILD_ROAD": 0.0381, "('play_turn', 4, -1, 4, 0, 1, 0)|BUILD_SETTLEMENT": 1.2555, "('play_turn', 4, -1, 4, 0, 1, 1)|BUILD_SETTLEMENT": 1.3178, "('play_turn', 4, -1, 4, 1, 0, 0)|BUILD_CITY": 0.679, "('play_turn', 4, -1, 5, 0, 0, 0)|BUILD_ROAD": 0.6635, "('play_turn', 4, -1, 5, 0, 0, 0)|END_TURN": 0.3352, "('play_turn', 4, -1, 5, 0, 0, 0)|MARITIME_TRADE": 0.4559, "('play_turn', 4, -1, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.1102, "('play_turn', 4, -1, 5, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.0966, "('play_turn', 4, -1, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.9448, "('play_turn', 4, -1, 5, 0, 0, 1)|END_TURN": 0.1237, "('play_turn', 4, -1, 5, 0, 1, 0)|BUILD_ROAD": 0.4011, "('play_turn', 4, -1, 5, 0, 1, 0)|BUILD_SETTLEMENT": 1.1538, "('play_turn', 4, -1, 5, 0, 1, 1)|BUILD_ROAD": 0.5813, "('play_turn', 4, -1, 5, 0, 1, 1)|BUILD_SETTLEMENT": 0.7654, "('play_turn', 4, -1, 5, 1, 0, 0)|BUILD_CITY": 1.1094, "('play_turn', 4, -1, 6, 0, 0, 0)|BUILD_ROAD": 1.019, "('play_turn', 4, -1, 6, 0, 0, 0)|END_TURN": 0.9202, "('play_turn', 4, -1, 6, 0, 0, 0)|MARITIME_TRADE": 0.9642, "('play_turn', 4, -1, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.3958, "('play_turn', 4, -1, 6, 0, 0, 0)|PLAY_MONOPOLY": 0.9676, "('play_turn', 4, -1, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.0634, "('play_turn', 4, -1, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.0288, "('play_turn', 4, -1, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.3415, "('play_turn', 4, -1, 6, 0, 0, 1)|END_TURN": 0.5255, "('play_turn', 4, -1, 6, 0, 0, 1)|MARITIME_TRADE": 0.839, "('play_turn', 4, -1, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 0.1573, "('play_turn', 4, -1, 6, 0, 1, 0)|BUILD_ROAD": 1.4188, "('play_turn', 4, -1, 6, 0, 1, 0)|BUILD_SETTLEMENT": 1.8201, "('play_turn', 4, -1, 6, 0, 1, 0)|END_TURN": 0.0121, "('play_turn', 4, -1, 6, 0, 1, 1)|BUILD_ROAD": 1.1698, "('play_turn', 4, -1, 6, 0, 1, 1)|BUILD_SETTLEMENT": 1.731, "('play_turn', 4, -1, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.2521, "('play_turn', 4, -1, 6, 1, 0, 0)|BUILD_CITY": 1.8284, "('play_turn', 4, -1, 6, 1, 0, 0)|END_TURN": 0.2746, "('play_turn', 4, -1, 6, 1, 0, 0)|MARITIME_TRADE": 0.3252, "('play_turn', 4, -1, 6, 1, 0, 1)|BUILD_CITY": 1.5081, "('play_turn', 4, -1, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 0.1516, "('play_turn', 4, -1, 6, 1, 0, 1)|MARITIME_TRADE": 0.4375, "('play_turn', 4, -2, 3, 0, 0, 0)|END_TURN": -0.1763, "('play_turn', 4, -2, 3, 0, 0, 0)|MARITIME_TRADE": -0.0803, "('play_turn', 4, -2, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.1888, "('play_turn', 4, -2, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.0001, "('play_turn', 4, -2, 3, 0, 0, 1)|END_TURN": -0.1453, "('play_turn', 4, -2, 3, 0, 1, 0)|BUILD_SETTLEMENT": 0.0732, "('play_turn', 4, -2, 4, 0, 0, 0)|BUILD_ROAD": 0.2641, "('play_turn', 4, -2, 4, 0, 0, 0)|END_TURN": -0.1922, "('play_turn', 4, -2, 4, 0, 0, 0)|MARITIME_TRADE": 0.0798, "('play_turn', 4, -2, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.5066, "('play_turn', 4, -2, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.3664, "('play_turn', 4, -2, 4, 0, 0, 1)|END_TURN": -0.1583, "('play_turn', 4, -2, 4, 0, 1, 0)|BUILD_SETTLEMENT": 0.4125, "('play_turn', 4, -2, 4, 0, 1, 1)|BUILD_SETTLEMENT": 0.281, "('play_turn', 4, -2, 5, 0, 0, 0)|BUILD_ROAD": 0.4391, "('play_turn', 4, -2, 5, 0, 0, 0)|END_TURN": 0.089, "('play_turn', 4, -2, 5, 0, 0, 0)|MARITIME_TRADE": 0.2092, "('play_turn', 4, -2, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.2075, "('play_turn', 4, -2, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.0417, "('play_turn', 4, -2, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.2989, "('play_turn', 4, -2, 5, 0, 0, 1)|END_TURN": -0.2376, "('play_turn', 4, -2, 5, 0, 1, 0)|BUILD_ROAD": -0.1023, "('play_turn', 4, -2, 5, 0, 1, 0)|BUILD_SETTLEMENT": 0.571, "('play_turn', 4, -2, 5, 0, 1, 1)|BUILD_SETTLEMENT": 0.7585, "('play_turn', 4, -2, 6, 0, 0, 0)|BUILD_ROAD": 0.1763, "('play_turn', 4, -2, 6, 0, 0, 0)|END_TURN": 0.0819, "('play_turn', 4, -2, 6, 0, 0, 0)|MARITIME_TRADE": 0.5158, "('play_turn', 4, -2, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.6527, "('play_turn', 4, -2, 6, 0, 0, 0)|PLAY_MONOPOLY": 0.0847, "('play_turn', 4, -2, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.302, "('play_turn', 4, -2, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.551, "('play_turn', 4, -2, 6, 0, 0, 1)|END_TURN": 0.0927, "('play_turn', 4, -2, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 0.3843, "('play_turn', 4, -2, 6, 0, 1, 0)|BUILD_ROAD": 0.542, "('play_turn', 4, -2, 6, 0, 1, 0)|BUILD_SETTLEMENT": 1.2249, "('play_turn', 4, -2, 6, 0, 1, 1)|BUILD_SETTLEMENT": 0.7209, "('play_turn', 4, -2, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.4396, "('play_turn', 4, -2, 6, 1, 0, 0)|BUILD_CITY": 1.1978, "('play_turn', 4, -2, 6, 1, 0, 0)|MARITIME_TRADE": 0.1672, "('play_turn', 4, -2, 6, 1, 0, 1)|BUILD_CITY": 0.6699, "('play_turn', 4, 0, 2, 0, 0, 0)|BUILD_ROAD": 0.0609, "('play_turn', 4, 0, 2, 0, 0, 0)|END_TURN": 0.0645, "('play_turn', 4, 0, 2, 0, 0, 0)|MARITIME_TRADE": -0.081, "('play_turn', 4, 0, 3, 0, 0, 0)|BUILD_ROAD": 0.3655, "('play_turn', 4, 0, 3, 0, 0, 0)|END_TURN": 0.0783, "('play_turn', 4, 0, 3, 0, 0, 0)|MARITIME_TRADE": 0.4876, "('play_turn', 4, 0, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.5189, "('play_turn', 4, 0, 3, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.1025, "('play_turn', 4, 0, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.3323, "('play_turn', 4, 0, 3, 0, 0, 1)|END_TURN": 0.0865, "('play_turn', 4, 0, 3, 0, 1, 0)|BUILD_SETTLEMENT": 1.1446, "('play_turn', 4, 0, 3, 0, 1, 1)|BUILD_SETTLEMENT": 0.9727, "('play_turn', 4, 0, 3, 1, 0, 0)|BUILD_CITY": 0.9115, "('play_turn', 4, 0, 4, 0, 0, 0)|BUILD_ROAD": 0.3431, "('play_turn', 4, 0, 4, 0, 0, 0)|END_TURN": 0.4671, "('play_turn', 4, 0, 4, 0, 0, 0)|MARITIME_TRADE": 0.7403, "('play_turn', 4, 0, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.9882, "('play_turn', 4, 0, 4, 0, 0, 0)|PLAY_MONOPOLY": 0.3353, "('play_turn', 4, 0, 4, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.0557, "('play_turn', 4, 0, 4, 0, 0, 0)|ROLL": 0.2122, "('play_turn', 4, 0, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.865, "('play_turn', 4, 0, 4, 0, 0, 1)|END_TURN": 0.0481, "('play_turn', 4, 0, 4, 0, 0, 1)|MARITIME_TRADE": 0.413, "('play_turn', 4, 0, 4, 0, 1, 0)|BUILD_ROAD": 0.5899, "('play_turn', 4, 0, 4, 0, 1, 0)|BUILD_SETTLEMENT": 1.7911, "('play_turn', 4, 0, 4, 0, 1, 0)|END_TURN": 0.1405, "('play_turn', 4, 0, 4, 0, 1, 1)|BUILD_ROAD": 0.3333, "('play_turn', 4, 0, 4, 0, 1, 1)|BUILD_SETTLEMENT": 1.6963, "('play_turn', 4, 0, 4, 1, 0, 0)|BUILD_CITY": 1.8058, "('play_turn', 4, 0, 5, 0, 0, 0)|BUILD_ROAD": 1.0351, "('play_turn', 4, 0, 5, 0, 0, 0)|END_TURN": 0.9329, "('play_turn', 4, 0, 5, 0, 0, 0)|MARITIME_TRADE": 0.9476, "('play_turn', 4, 0, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.8845, "('play_turn', 4, 0, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.4506, "('play_turn', 4, 0, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.374, "('play_turn', 4, 0, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.4034, "('play_turn', 4, 0, 5, 0, 0, 1)|END_TURN": 0.285, "('play_turn', 4, 0, 5, 0, 1, 0)|BUILD_ROAD": 0.7718, "('play_turn', 4, 0, 5, 0, 1, 0)|BUILD_SETTLEMENT": 1.475, "('play_turn', 4, 0, 5, 0, 1, 1)|BUILD_SETTLEMENT": 1.5612, "('play_turn', 4, 0, 5, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.6298, "('play_turn', 4, 0, 5, 1, 0, 0)|BUILD_CITY": 1.4539, "('play_turn', 4, 0, 5, 1, 0, 1)|BUILD_CITY": 0.9379, "('play_turn', 4, 0, 6, 0, 0, 0)|BUILD_ROAD": 1.7763, "('play_turn', 4, 0, 6, 0, 0, 0)|END_TURN": 1.509, "('play_turn', 4, 0, 6, 0, 0, 0)|MARITIME_TRADE": 1.6266, "('play_turn', 4, 0, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.9242, "('play_turn', 4, 0, 6, 0, 0, 0)|PLAY_MONOPOLY": 1.1002, "('play_turn', 4, 0, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.678, "('play_turn', 4, 0, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.6266, "('play_turn', 4, 0, 6, 0, 0, 0)|ROLL": 0.6249, "('play_turn', 4, 0, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.8592, "('play_turn', 4, 0, 6, 0, 0, 1)|END_TURN": 0.992, "('play_turn', 4, 0, 6, 0, 0, 1)|MARITIME_TRADE": 1.6166, "('play_turn', 4, 0, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 1.0014, "('play_turn', 4, 0, 6, 0, 1, 0)|BUILD_ROAD": 2.0917, "('play_turn', 4, 0, 6, 0, 1, 0)|BUILD_SETTLEMENT": 2.4132, "('play_turn', 4, 0, 6, 0, 1, 0)|END_TURN": 0.6863, "('play_turn', 4, 0, 6, 0, 1, 0)|PLAY_ROAD_BUILDING": 0.2697, "('play_turn', 4, 0, 6, 0, 1, 1)|BUILD_ROAD": 0.8632, "('play_turn', 4, 0, 6, 0, 1, 1)|BUILD_SETTLEMENT": 2.1537, "('play_turn', 4, 0, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 1.4773, "('play_turn', 4, 0, 6, 1, 0, 0)|BUILD_CITY": 2.3526, "('play_turn', 4, 0, 6, 1, 0, 0)|END_TURN": 0.4329, "('play_turn', 4, 0, 6, 1, 0, 0)|MARITIME_TRADE": 1.0997, "('play_turn', 4, 0, 6, 1, 0, 1)|BUILD_CITY": 2.0952, "('play_turn', 4, 0, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 1.09, "('play_turn', 4, 0, 6, 1, 1, 1)|BUILD_SETTLEMENT": 0.9614, "('play_turn', 4, 1, 2, 0, 0, 0)|BUILD_ROAD": 0.2992, "('play_turn', 4, 1, 2, 0, 0, 0)|END_TURN": 0.1337, "('play_turn', 4, 1, 3, 0, 0, 0)|BUILD_ROAD": 0.735, "('play_turn', 4, 1, 3, 0, 0, 0)|END_TURN": 0.6969, "('play_turn', 4, 1, 3, 0, 0, 0)|MARITIME_TRADE": 0.66, "('play_turn', 4, 1, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.8566, "('play_turn', 4, 1, 3, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.3175, "('play_turn', 4, 1, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.6148, "('play_turn', 4, 1, 3, 0, 0, 1)|END_TURN": 0.3269, "('play_turn', 4, 1, 3, 0, 0, 1)|MARITIME_TRADE": 0.1968, "('play_turn', 4, 1, 3, 0, 1, 0)|BUILD_SETTLEMENT": 1.8365, "('play_turn', 4, 1, 3, 0, 1, 1)|BUILD_SETTLEMENT": 1.2801, "('play_turn', 4, 1, 3, 1, 0, 0)|BUILD_CITY": 1.4337, "('play_turn', 4, 1, 3, 1, 0, 1)|BUILD_CITY": 0.5351, "('play_turn', 4, 1, 4, 0, 0, 0)|BUILD_ROAD": 0.891, "('play_turn', 4, 1, 4, 0, 0, 0)|END_TURN": 0.8023, "('play_turn', 4, 1, 4, 0, 0, 0)|MARITIME_TRADE": 1.1584, "('play_turn', 4, 1, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.2754, "('play_turn', 4, 1, 4, 0, 0, 0)|PLAY_MONOPOLY": 0.4925, "('play_turn', 4, 1, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.874, "('play_turn', 4, 1, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.1433, "('play_turn', 4, 1, 4, 0, 0, 1)|END_TURN": 0.6487, "('play_turn', 4, 1, 4, 0, 1, 0)|BUILD_ROAD": 0.9983, "('play_turn', 4, 1, 4, 0, 1, 0)|BUILD_SETTLEMENT": 2.5039, "('play_turn', 4, 1, 4, 0, 1, 1)|BUILD_SETTLEMENT": 2.303, "('play_turn', 4, 1, 4, 1, 0, 0)|BUILD_CITY": 2.0192, "('play_turn', 4, 1, 4, 1, 0, 1)|BUILD_CITY": 0.8713, "('play_turn', 4, 1, 5, 0, 0, 0)|BUILD_ROAD": 1.374, "('play_turn', 4, 1, 5, 0, 0, 0)|END_TURN": 1.1024, "('play_turn', 4, 1, 5, 0, 0, 0)|MARITIME_TRADE": 1.4081, "('play_turn', 4, 1, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.7438, "('play_turn', 4, 1, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.3438, "('play_turn', 4, 1, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.3683, "('play_turn', 4, 1, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.6526, "('play_turn', 4, 1, 5, 0, 0, 1)|END_TURN": 0.2016, "('play_turn', 4, 1, 5, 0, 1, 0)|BUILD_ROAD": 0.5658, "('play_turn', 4, 1, 5, 0, 1, 0)|BUILD_SETTLEMENT": 2.2823, "('play_turn', 4, 1, 5, 0, 1, 1)|BUILD_SETTLEMENT": 1.5697, "('play_turn', 4, 1, 5, 1, 0, 0)|BUILD_CITY": 2.2299, "('play_turn', 4, 1, 5, 1, 0, 1)|BUILD_CITY": 0.9975, "('play_turn', 4, 1, 6, 0, 0, 0)|BUILD_ROAD": 1.9038, "('play_turn', 4, 1, 6, 0, 0, 0)|END_TURN": 1.5978, "('play_turn', 4, 1, 6, 0, 0, 0)|MARITIME_TRADE": 1.9812, "('play_turn', 4, 1, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.8812, "('play_turn', 4, 1, 6, 0, 0, 0)|PLAY_MONOPOLY": 1.3792, "('play_turn', 4, 1, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.9785, "('play_turn', 4, 1, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.5613, "('play_turn', 4, 1, 6, 0, 0, 0)|ROLL": 0.4567, "('play_turn', 4, 1, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.9983, "('play_turn', 4, 1, 6, 0, 0, 1)|END_TURN": 1.2386, "('play_turn', 4, 1, 6, 0, 0, 1)|MARITIME_TRADE": 1.3761, "('play_turn', 4, 1, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 0.9723, "('play_turn', 4, 1, 6, 0, 1, 0)|BUILD_ROAD": 1.8711, "('play_turn', 4, 1, 6, 0, 1, 0)|BUILD_SETTLEMENT": 2.6228, "('play_turn', 4, 1, 6, 0, 1, 0)|END_TURN": 0.8377, "('play_turn', 4, 1, 6, 0, 1, 1)|BUILD_ROAD": 0.7046, "('play_turn', 4, 1, 6, 0, 1, 1)|BUILD_SETTLEMENT": 2.466, "('play_turn', 4, 1, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 1.4304, "('play_turn', 4, 1, 6, 0, 1, 1)|END_TURN": 0.4227, "('play_turn', 4, 1, 6, 1, 0, 0)|BUILD_CITY": 2.5508, "('play_turn', 4, 1, 6, 1, 0, 0)|END_TURN": 0.7324, "('play_turn', 4, 1, 6, 1, 0, 0)|MARITIME_TRADE": 1.2349, "('play_turn', 4, 1, 6, 1, 0, 1)|BUILD_CITY": 2.3856, "('play_turn', 4, 1, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 0.994, "('play_turn', 4, 1, 6, 1, 0, 1)|END_TURN": 0.3411, "('play_turn', 4, 1, 6, 1, 0, 1)|MARITIME_TRADE": 0.8979, "('play_turn', 4, 2, 2, 0, 0, 0)|BUILD_ROAD": -0.1166, "('play_turn', 4, 2, 2, 0, 0, 0)|END_TURN": 0.1997, "('play_turn', 4, 2, 3, 0, 0, 0)|BUILD_ROAD": 0.7449, "('play_turn', 4, 2, 3, 0, 0, 0)|END_TURN": 0.9568, "('play_turn', 4, 2, 3, 0, 0, 0)|MARITIME_TRADE": 1.0691, "('play_turn', 4, 2, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.323, "('play_turn', 4, 2, 3, 0, 0, 0)|PLAY_MONOPOLY": 0.2762, "('play_turn', 4, 2, 3, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.4157, "('play_turn', 4, 2, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.046, "('play_turn', 4, 2, 3, 0, 0, 1)|END_TURN": 0.2901, "('play_turn', 4, 2, 3, 0, 1, 0)|BUILD_SETTLEMENT": 1.8613, "('play_turn', 4, 2, 3, 0, 1, 1)|BUILD_SETTLEMENT": 1.8636, "('play_turn', 4, 2, 3, 1, 0, 0)|BUILD_CITY": 1.0247, "('play_turn', 4, 2, 4, 0, 0, 0)|BUILD_ROAD": 1.6734, "('play_turn', 4, 2, 4, 0, 0, 0)|END_TURN": 1.6936, "('play_turn', 4, 2, 4, 0, 0, 0)|MARITIME_TRADE": 1.8115, "('play_turn', 4, 2, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.5883, "('play_turn', 4, 2, 4, 0, 0, 0)|PLAY_MONOPOLY": 1.0689, "('play_turn', 4, 2, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.402, "('play_turn', 4, 2, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.7481, "('play_turn', 4, 2, 4, 0, 0, 1)|END_TURN": 0.978, "('play_turn', 4, 2, 4, 0, 0, 1)|MARITIME_TRADE": 1.0648, "('play_turn', 4, 2, 4, 0, 1, 0)|BUILD_ROAD": 1.4335, "('play_turn', 4, 2, 4, 0, 1, 0)|BUILD_SETTLEMENT": 2.5837, "('play_turn', 4, 2, 4, 0, 1, 0)|END_TURN": 0.6069, "('play_turn', 4, 2, 4, 0, 1, 1)|BUILD_SETTLEMENT": 2.7805, "('play_turn', 4, 2, 4, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.7245, "('play_turn', 4, 2, 4, 1, 0, 0)|BUILD_CITY": 2.7753, "('play_turn', 4, 2, 4, 1, 0, 1)|BUILD_CITY": 1.2682, "('play_turn', 4, 2, 5, 0, 0, 0)|BUILD_ROAD": 1.9169, "('play_turn', 4, 2, 5, 0, 0, 0)|END_TURN": 1.8227, "('play_turn', 4, 2, 5, 0, 0, 0)|MARITIME_TRADE": 1.8476, "('play_turn', 4, 2, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.1792, "('play_turn', 4, 2, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.5204, "('play_turn', 4, 2, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.1659, "('play_turn', 4, 2, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.9532, "('play_turn', 4, 2, 5, 0, 0, 1)|END_TURN": 0.5686, "('play_turn', 4, 2, 5, 0, 1, 0)|BUILD_ROAD": 1.4804, "('play_turn', 4, 2, 5, 0, 1, 0)|BUILD_SETTLEMENT": 2.7951, "('play_turn', 4, 2, 5, 0, 1, 0)|END_TURN": 0.6932, "('play_turn', 4, 2, 5, 0, 1, 1)|BUILD_ROAD": 1.3561, "('play_turn', 4, 2, 5, 0, 1, 1)|BUILD_SETTLEMENT": 2.364, "('play_turn', 4, 2, 5, 1, 0, 0)|BUILD_CITY": 2.4033, "('play_turn', 4, 2, 5, 1, 0, 1)|BUILD_CITY": 1.0752, "('play_turn', 4, 2, 6, 0, 0, 0)|BUILD_ROAD": 2.6114, "('play_turn', 4, 2, 6, 0, 0, 0)|END_TURN": 2.1614, "('play_turn', 4, 2, 6, 0, 0, 0)|MARITIME_TRADE": 2.3794, "('play_turn', 4, 2, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.4583, "('play_turn', 4, 2, 6, 0, 0, 0)|PLAY_MONOPOLY": 2.1112, "('play_turn', 4, 2, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 2.1807, "('play_turn', 4, 2, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 2.2536, "('play_turn', 4, 2, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 2.4194, "('play_turn', 4, 2, 6, 0, 0, 1)|END_TURN": 2.1295, "('play_turn', 4, 2, 6, 0, 0, 1)|MARITIME_TRADE": 2.4653, "('play_turn', 4, 2, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 1.9386, "('play_turn', 4, 2, 6, 0, 1, 0)|BUILD_ROAD": 2.6766, "('play_turn', 4, 2, 6, 0, 1, 0)|BUILD_SETTLEMENT": 2.9446, "('play_turn', 4, 2, 6, 0, 1, 0)|END_TURN": 1.7175, "('play_turn', 4, 2, 6, 0, 1, 0)|PLAY_ROAD_BUILDING": 0.2631, "('play_turn', 4, 2, 6, 0, 1, 1)|BUILD_ROAD": 1.5022, "('play_turn', 4, 2, 6, 0, 1, 1)|BUILD_SETTLEMENT": 2.9616, "('play_turn', 4, 2, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 2.422, "('play_turn', 4, 2, 6, 0, 1, 1)|END_TURN": 1.1759, "('play_turn', 4, 2, 6, 1, 0, 0)|BUILD_CITY": 2.9282, "('play_turn', 4, 2, 6, 1, 0, 0)|BUILD_ROAD": 0.7155, "('play_turn', 4, 2, 6, 1, 0, 0)|END_TURN": 1.3246, "('play_turn', 4, 2, 6, 1, 0, 0)|MARITIME_TRADE": 2.0447, "('play_turn', 4, 2, 6, 1, 0, 1)|BUILD_CITY": 2.8235, "('play_turn', 4, 2, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 1.8486, "('play_turn', 4, 2, 6, 1, 0, 1)|END_TURN": 0.4624, "('play_turn', 4, 2, 6, 1, 0, 1)|MARITIME_TRADE": 0.9062, "('play_turn', 4, 2, 6, 1, 1, 1)|BUILD_SETTLEMENT": 1.1734, "('play_turn', 4, 2, 6, 1, 1, 1)|BUY_DEVELOPMENT_CARD": 0.5885, "('play_turn', 5, -1, 3, 0, 0, 0)|END_TURN": -0.3807, "('play_turn', 5, -1, 4, 0, 0, 0)|BUILD_ROAD": 0.2029, "('play_turn', 5, -1, 4, 0, 0, 0)|END_TURN": -0.4476, "('play_turn', 5, -1, 4, 0, 0, 0)|MARITIME_TRADE": 0.0271, "('play_turn', 5, -1, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.8974, "('play_turn', 5, -1, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.2235, "('play_turn', 5, -1, 4, 0, 0, 1)|END_TURN": -0.0655, "('play_turn', 5, -1, 4, 0, 1, 0)|BUILD_SETTLEMENT": 0.4148, "('play_turn', 5, -1, 4, 0, 1, 1)|BUILD_SETTLEMENT": 0.2575, "('play_turn', 5, -1, 5, 0, 0, 0)|BUILD_ROAD": 0.4352, "('play_turn', 5, -1, 5, 0, 0, 0)|END_TURN": -0.1235, "('play_turn', 5, -1, 5, 0, 0, 0)|MARITIME_TRADE": 0.0081, "('play_turn', 5, -1, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.3026, "('play_turn', 5, -1, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.0983, "('play_turn', 5, -1, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.1435, "('play_turn', 5, -1, 5, 0, 0, 1)|END_TURN": -0.0735, "('play_turn', 5, -1, 5, 0, 1, 0)|BUILD_SETTLEMENT": 0.632, "('play_turn', 5, -1, 5, 1, 0, 0)|BUILD_CITY": 0.7241, "('play_turn', 5, -1, 6, 0, 0, 0)|BUILD_ROAD": 1.1764, "('play_turn', 5, -1, 6, 0, 0, 0)|END_TURN": 0.7103, "('play_turn', 5, -1, 6, 0, 0, 0)|MARITIME_TRADE": 0.9759, "('play_turn', 5, -1, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.3198, "('play_turn', 5, -1, 6, 0, 0, 0)|PLAY_MONOPOLY": 0.2532, "('play_turn', 5, -1, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.1311, "('play_turn', 5, -1, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.0874, "('play_turn', 5, -1, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.1408, "('play_turn', 5, -1, 6, 0, 0, 1)|END_TURN": 0.4162, "('play_turn', 5, -1, 6, 0, 1, 0)|BUILD_ROAD": 0.8463, "('play_turn', 5, -1, 6, 0, 1, 0)|BUILD_SETTLEMENT": 1.2536, "('play_turn', 5, -1, 6, 0, 1, 0)|END_TURN": 0.0054, "('play_turn', 5, -1, 6, 0, 1, 1)|BUILD_SETTLEMENT": 1.0952, "('play_turn', 5, -1, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.6561, "('play_turn', 5, -1, 6, 1, 0, 0)|BUILD_CITY": 1.1824, "('play_turn', 5, -1, 6, 1, 0, 0)|END_TURN": -0.1713, "('play_turn', 5, -1, 6, 1, 0, 1)|BUILD_CITY": 0.654, "('play_turn', 5, 0, 3, 0, 0, 0)|BUILD_ROAD": 0.005, "('play_turn', 5, 0, 3, 0, 0, 0)|END_TURN": -0.0018, "('play_turn', 5, 0, 3, 0, 0, 0)|MARITIME_TRADE": 0.0817, "('play_turn', 5, 0, 3, 0, 1, 0)|BUILD_SETTLEMENT": 0.535, "('play_turn', 5, 0, 3, 0, 1, 1)|BUILD_SETTLEMENT": 0.3069, "('play_turn', 5, 0, 4, 0, 0, 0)|BUILD_ROAD": 0.2523, "('play_turn', 5, 0, 4, 0, 0, 0)|END_TURN": 0.1635, "('play_turn', 5, 0, 4, 0, 0, 0)|MARITIME_TRADE": 0.3735, "('play_turn', 5, 0, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.7115, "('play_turn', 5, 0, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.2558, "('play_turn', 5, 0, 4, 0, 0, 1)|END_TURN": 0.1326, "('play_turn', 5, 0, 4, 0, 1, 0)|BUILD_SETTLEMENT": 0.9389, "('play_turn', 5, 0, 4, 0, 1, 1)|BUILD_SETTLEMENT": 1.0405, "('play_turn', 5, 0, 4, 1, 0, 0)|BUILD_CITY": 0.5077, "('play_turn', 5, 0, 5, 0, 0, 0)|BUILD_ROAD": 0.4044, "('play_turn', 5, 0, 5, 0, 0, 0)|END_TURN": 0.1851, "('play_turn', 5, 0, 5, 0, 0, 0)|MARITIME_TRADE": 0.6419, "('play_turn', 5, 0, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.4796, "('play_turn', 5, 0, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.2108, "('play_turn', 5, 0, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.0699, "('play_turn', 5, 0, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.9399, "('play_turn', 5, 0, 5, 0, 0, 1)|END_TURN": 0.0833, "('play_turn', 5, 0, 5, 0, 1, 0)|BUILD_SETTLEMENT": 1.7225, "('play_turn', 5, 0, 5, 0, 1, 1)|BUILD_SETTLEMENT": 0.3039, "('play_turn', 5, 0, 5, 1, 0, 0)|BUILD_CITY": 0.4231, "('play_turn', 5, 0, 6, 0, 0, 0)|BUILD_ROAD": 1.2873, "('play_turn', 5, 0, 6, 0, 0, 0)|END_TURN": 0.8364, "('play_turn', 5, 0, 6, 0, 0, 0)|MARITIME_TRADE": 1.241, "('play_turn', 5, 0, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.9067, "('play_turn', 5, 0, 6, 0, 0, 0)|PLAY_MONOPOLY": 0.5152, "('play_turn', 5, 0, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.0753, "('play_turn', 5, 0, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.0998, "('play_turn', 5, 0, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.5185, "('play_turn', 5, 0, 6, 0, 0, 1)|END_TURN": 0.9613, "('play_turn', 5, 0, 6, 0, 0, 1)|MARITIME_TRADE": 0.9872, "('play_turn', 5, 0, 6, 0, 1, 0)|BUILD_ROAD": 1.1198, "('play_turn', 5, 0, 6, 0, 1, 0)|BUILD_SETTLEMENT": 2.0488, "('play_turn', 5, 0, 6, 0, 1, 0)|END_TURN": 0.6282, "('play_turn', 5, 0, 6, 0, 1, 1)|BUILD_ROAD": 1.4256, "('play_turn', 5, 0, 6, 0, 1, 1)|BUILD_SETTLEMENT": 1.7024, "('play_turn', 5, 0, 6, 1, 0, 0)|BUILD_CITY": 2.0665, "('play_turn', 5, 0, 6, 1, 0, 0)|END_TURN": 0.3444, "('play_turn', 5, 0, 6, 1, 0, 0)|MARITIME_TRADE": 0.5954, "('play_turn', 5, 0, 6, 1, 0, 1)|BUILD_CITY": 1.5436, "('play_turn', 5, 0, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 0.5972, "('play_turn', 5, 1, 2, 0, 0, 0)|END_TURN": -0.2953, "('play_turn', 5, 1, 3, 0, 0, 0)|END_TURN": 0.1967, "('play_turn', 5, 1, 3, 0, 0, 0)|MARITIME_TRADE": 0.0708, "('play_turn', 5, 1, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.5275, "('play_turn', 5, 1, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.1578, "('play_turn', 5, 1, 3, 0, 0, 1)|END_TURN": 0.0327, "('play_turn', 5, 1, 3, 0, 1, 0)|BUILD_SETTLEMENT": 0.6139, "('play_turn', 5, 1, 3, 0, 1, 1)|BUILD_SETTLEMENT": 0.3826, "('play_turn', 5, 1, 4, 0, 0, 0)|BUILD_ROAD": 0.4568, "('play_turn', 5, 1, 4, 0, 0, 0)|END_TURN": 0.6471, "('play_turn', 5, 1, 4, 0, 0, 0)|MARITIME_TRADE": 0.7636, "('play_turn', 5, 1, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.4601, "('play_turn', 5, 1, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.3194, "('play_turn', 5, 1, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.8281, "('play_turn', 5, 1, 4, 0, 0, 1)|END_TURN": 0.5292, "('play_turn', 5, 1, 4, 0, 1, 0)|BUILD_ROAD": 0.4065, "('play_turn', 5, 1, 4, 0, 1, 0)|BUILD_SETTLEMENT": 1.7021, "('play_turn', 5, 1, 4, 0, 1, 1)|BUILD_SETTLEMENT": 1.2655, "('play_turn', 5, 1, 4, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.3489, "('play_turn', 5, 1, 4, 1, 0, 0)|BUILD_CITY": 1.2613, "('play_turn', 5, 1, 5, 0, 0, 0)|BUILD_ROAD": 1.1428, "('play_turn', 5, 1, 5, 0, 0, 0)|END_TURN": 0.9092, "('play_turn', 5, 1, 5, 0, 0, 0)|MARITIME_TRADE": 1.0148, "('play_turn', 5, 1, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.2396, "('play_turn', 5, 1, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.2726, "('play_turn', 5, 1, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.025, "('play_turn', 5, 1, 5, 0, 0, 1)|END_TURN": 0.3814, "('play_turn', 5, 1, 5, 0, 0, 1)|MARITIME_TRADE": 0.2013, "('play_turn', 5, 1, 5, 0, 1, 0)|BUILD_ROAD": 0.9562, "('play_turn', 5, 1, 5, 0, 1, 0)|BUILD_SETTLEMENT": 2.1301, "('play_turn', 5, 1, 5, 0, 1, 1)|BUILD_SETTLEMENT": 1.7142, "('play_turn', 5, 1, 5, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.5894, "('play_turn', 5, 1, 5, 1, 0, 0)|BUILD_CITY": 1.8599, "('play_turn', 5, 1, 6, 0, 0, 0)|BUILD_ROAD": 1.4811, "('play_turn', 5, 1, 6, 0, 0, 0)|END_TURN": 1.5208, "('play_turn', 5, 1, 6, 0, 0, 0)|MARITIME_TRADE": 1.6121, "('play_turn', 5, 1, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.9788, "('play_turn', 5, 1, 6, 0, 0, 0)|PLAY_MONOPOLY": 1.3773, "('play_turn', 5, 1, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 1.041, "('play_turn', 5, 1, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.511, "('play_turn', 5, 1, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.7086, "('play_turn', 5, 1, 6, 0, 0, 1)|END_TURN": 0.9289, "('play_turn', 5, 1, 6, 0, 0, 1)|MARITIME_TRADE": 1.6635, "('play_turn', 5, 1, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 1.7336, "('play_turn', 5, 1, 6, 0, 1, 0)|BUILD_ROAD": 1.9699, "('play_turn', 5, 1, 6, 0, 1, 0)|BUILD_SETTLEMENT": 1.7866, "('play_turn', 5, 1, 6, 0, 1, 0)|END_TURN": 0.7429, "('play_turn', 5, 1, 6, 0, 1, 1)|BUILD_ROAD": 1.5391, "('play_turn', 5, 1, 6, 0, 1, 1)|BUILD_SETTLEMENT": 2.1918, "('play_turn', 5, 1, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.9458, "('play_turn', 5, 1, 6, 0, 1, 1)|END_TURN": 0.4325, "('play_turn', 5, 1, 6, 1, 0, 0)|BUILD_CITY": 2.3306, "('play_turn', 5, 1, 6, 1, 0, 0)|END_TURN": 0.6334, "('play_turn', 5, 1, 6, 1, 0, 0)|MARITIME_TRADE": 1.2262, "('play_turn', 5, 1, 6, 1, 0, 1)|BUILD_CITY": 2.1925, "('play_turn', 5, 1, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 1.0968, "('play_turn', 5, 1, 6, 1, 0, 1)|END_TURN": 0.2745, "('play_turn', 5, 1, 6, 1, 1, 1)|BUILD_SETTLEMENT": 1.1757, "('play_turn', 5, 1, 6, 1, 1, 1)|BUY_DEVELOPMENT_CARD": 0.8109, "('play_turn', 5, 2, 2, 0, 0, 0)|END_TURN": -0.0773, "('play_turn', 5, 2, 3, 0, 0, 0)|BUILD_ROAD": 0.4479, "('play_turn', 5, 2, 3, 0, 0, 0)|END_TURN": 0.2488, "('play_turn', 5, 2, 3, 0, 0, 0)|MARITIME_TRADE": 0.1149, "('play_turn', 5, 2, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.0231, "('play_turn', 5, 2, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.411, "('play_turn', 5, 2, 3, 0, 0, 1)|END_TURN": -0.0383, "('play_turn', 5, 2, 3, 0, 1, 0)|BUILD_SETTLEMENT": 0.6818, "('play_turn', 5, 2, 4, 0, 0, 0)|BUILD_ROAD": 0.9153, "('play_turn', 5, 2, 4, 0, 0, 0)|END_TURN": 0.7876, "('play_turn', 5, 2, 4, 0, 0, 0)|MARITIME_TRADE": 1.0708, "('play_turn', 5, 2, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.3069, "('play_turn', 5, 2, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.4876, "('play_turn', 5, 2, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.4144, "('play_turn', 5, 2, 4, 0, 0, 1)|END_TURN": 0.1373, "('play_turn', 5, 2, 4, 0, 1, 0)|BUILD_ROAD": 0.4027, "('play_turn', 5, 2, 4, 0, 1, 0)|BUILD_SETTLEMENT": 1.9876, "('play_turn', 5, 2, 4, 0, 1, 1)|BUILD_SETTLEMENT": 1.7471, "('play_turn', 5, 2, 4, 1, 0, 0)|BUILD_CITY": 1.8758, "('play_turn', 5, 2, 5, 0, 0, 0)|BUILD_ROAD": 0.8494, "('play_turn', 5, 2, 5, 0, 0, 0)|END_TURN": 1.0748, "('play_turn', 5, 2, 5, 0, 0, 0)|MARITIME_TRADE": 1.2629, "('play_turn', 5, 2, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.522, "('play_turn', 5, 2, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.6906, "('play_turn', 5, 2, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.6417, "('play_turn', 5, 2, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.3601, "('play_turn', 5, 2, 5, 0, 0, 1)|END_TURN": 0.2545, "('play_turn', 5, 2, 5, 0, 0, 1)|MARITIME_TRADE": 0.4397, "('play_turn', 5, 2, 5, 0, 1, 0)|BUILD_ROAD": 1.4393, "('play_turn', 5, 2, 5, 0, 1, 0)|BUILD_SETTLEMENT": 2.3581, "('play_turn', 5, 2, 5, 0, 1, 0)|END_TURN": 0.6398, "('play_turn', 5, 2, 5, 0, 1, 1)|BUILD_SETTLEMENT": 2.0944, "('play_turn', 5, 2, 5, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.9342, "('play_turn', 5, 2, 5, 1, 0, 0)|BUILD_CITY": 1.9418, "('play_turn', 5, 2, 6, 0, 0, 0)|BUILD_ROAD": 2.027, "('play_turn', 5, 2, 6, 0, 0, 0)|END_TURN": 1.6309, "('play_turn', 5, 2, 6, 0, 0, 0)|MARITIME_TRADE": 1.8782, "('play_turn', 5, 2, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.9117, "('play_turn', 5, 2, 6, 0, 0, 0)|PLAY_MONOPOLY": 1.5562, "('play_turn', 5, 2, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.7481, "('play_turn', 5, 2, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.652, "('play_turn', 5, 2, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 2.0487, "('play_turn', 5, 2, 6, 0, 0, 1)|END_TURN": 1.2896, "('play_turn', 5, 2, 6, 0, 0, 1)|MARITIME_TRADE": 1.6658, "('play_turn', 5, 2, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 1.0447, "('play_turn', 5, 2, 6, 0, 1, 0)|BUILD_ROAD": 1.9155, "('play_turn', 5, 2, 6, 0, 1, 0)|BUILD_SETTLEMENT": 2.5883, "('play_turn', 5, 2, 6, 0, 1, 0)|END_TURN": 1.0489, "('play_turn', 5, 2, 6, 0, 1, 1)|BUILD_ROAD": 1.8625, "('play_turn', 5, 2, 6, 0, 1, 1)|BUILD_SETTLEMENT": 2.6616, "('play_turn', 5, 2, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.5882, "('play_turn', 5, 2, 6, 1, 0, 0)|BUILD_CITY": 2.62, "('play_turn', 5, 2, 6, 1, 0, 0)|END_TURN": 0.7857, "('play_turn', 5, 2, 6, 1, 0, 0)|MARITIME_TRADE": 0.6483, "('play_turn', 5, 2, 6, 1, 0, 1)|BUILD_CITY": 2.2483, "('play_turn', 5, 2, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 1.159, "('play_turn', 5, 3, 2, 0, 0, 0)|END_TURN": 0.0098, "('play_turn', 5, 3, 3, 0, 0, 0)|BUILD_ROAD": 0.3146, "('play_turn', 5, 3, 3, 0, 0, 0)|END_TURN": 0.4976, "('play_turn', 5, 3, 3, 0, 0, 0)|MARITIME_TRADE": 0.1358, "('play_turn', 5, 3, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.8556, "('play_turn', 5, 3, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.5353, "('play_turn', 5, 3, 3, 0, 1, 0)|BUILD_SETTLEMENT": 0.8802, "('play_turn', 5, 3, 3, 0, 1, 1)|BUILD_SETTLEMENT": 0.8968, "('play_turn', 5, 3, 4, 0, 0, 0)|BUILD_ROAD": 1.3795, "('play_turn', 5, 3, 4, 0, 0, 0)|END_TURN": 1.4435, "('play_turn', 5, 3, 4, 0, 0, 0)|MARITIME_TRADE": 1.3205, "('play_turn', 5, 3, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.515, "('play_turn', 5, 3, 4, 0, 0, 0)|PLAY_MONOPOLY": 0.2968, "('play_turn', 5, 3, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.5081, "('play_turn', 5, 3, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.4906, "('play_turn', 5, 3, 4, 0, 0, 1)|END_TURN": 0.4303, "('play_turn', 5, 3, 4, 0, 0, 1)|MARITIME_TRADE": 0.2473, "('play_turn', 5, 3, 4, 0, 1, 0)|BUILD_ROAD": 0.3504, "('play_turn', 5, 3, 4, 0, 1, 0)|BUILD_SETTLEMENT": 2.3148, "('play_turn', 5, 3, 4, 0, 1, 1)|BUILD_SETTLEMENT": 2.0104, "('play_turn', 5, 3, 4, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.7576, "('play_turn', 5, 3, 4, 1, 0, 0)|BUILD_CITY": 1.8253, "('play_turn', 5, 3, 5, 0, 0, 0)|BUILD_ROAD": 1.4422, "('play_turn', 5, 3, 5, 0, 0, 0)|END_TURN": 1.6936, "('play_turn', 5, 3, 5, 0, 0, 0)|MARITIME_TRADE": 1.71, "('play_turn', 5, 3, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.1124, "('play_turn', 5, 3, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.5002, "('play_turn', 5, 3, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.2081, "('play_turn', 5, 3, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.854, "('play_turn', 5, 3, 5, 0, 0, 1)|END_TURN": 0.8659, "('play_turn', 5, 3, 5, 0, 1, 0)|BUILD_ROAD": 1.0669, "('play_turn', 5, 3, 5, 0, 1, 0)|BUILD_SETTLEMENT": 2.3668, "('play_turn', 5, 3, 5, 0, 1, 1)|BUILD_SETTLEMENT": 2.4922, "('play_turn', 5, 3, 5, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 1.2648, "('play_turn', 5, 3, 5, 1, 0, 0)|BUILD_CITY": 2.1291, "('play_turn', 5, 3, 5, 1, 0, 1)|BUILD_CITY": 1.4003, "('play_turn', 5, 3, 6, 0, 0, 0)|BUILD_ROAD": 2.2367, "('play_turn', 5, 3, 6, 0, 0, 0)|END_TURN": 1.9739, "('play_turn', 5, 3, 6, 0, 0, 0)|MARITIME_TRADE": 2.2225, "('play_turn', 5, 3, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 2.363, "('play_turn', 5, 3, 6, 0, 0, 0)|PLAY_MONOPOLY": 1.9563, "('play_turn', 5, 3, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 1.6417, "('play_turn', 5, 3, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 2.107, "('play_turn', 5, 3, 6, 0, 0, 0)|ROLL": 0.9661, "('play_turn', 5, 3, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 2.2893, "('play_turn', 5, 3, 6, 0, 0, 1)|END_TURN": 2.0254, "('play_turn', 5, 3, 6, 0, 0, 1)|MARITIME_TRADE": 2.0803, "('play_turn', 5, 3, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 1.7635, "('play_turn', 5, 3, 6, 0, 1, 0)|BUILD_ROAD": 2.079, "('play_turn', 5, 3, 6, 0, 1, 0)|BUILD_SETTLEMENT": 2.5562, "('play_turn', 5, 3, 6, 0, 1, 0)|END_TURN": 1.5122, "('play_turn', 5, 3, 6, 0, 1, 0)|MARITIME_TRADE": 0.8861, "('play_turn', 5, 3, 6, 0, 1, 1)|BUILD_ROAD": 1.7056, "('play_turn', 5, 3, 6, 0, 1, 1)|BUILD_SETTLEMENT": 2.5801, "('play_turn', 5, 3, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 2.2688, "('play_turn', 5, 3, 6, 0, 1, 1)|END_TURN": 0.5095, "('play_turn', 5, 3, 6, 1, 0, 0)|BUILD_CITY": 2.6071, "('play_turn', 5, 3, 6, 1, 0, 0)|BUILD_ROAD": 0.7602, "('play_turn', 5, 3, 6, 1, 0, 0)|END_TURN": 0.8059, "('play_turn', 5, 3, 6, 1, 0, 0)|MARITIME_TRADE": 0.9672, "('play_turn', 5, 3, 6, 1, 0, 1)|BUILD_CITY": 2.5641, "('play_turn', 5, 3, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 1.4184, "('play_turn', 5, 3, 6, 1, 0, 1)|END_TURN": 0.465, "('play_turn', 5, 3, 6, 1, 0, 1)|MARITIME_TRADE": 0.7984, "('play_turn', 5, 3, 6, 1, 1, 1)|BUILD_ROAD": 0.8594, "('play_turn', 5, 3, 6, 1, 1, 1)|BUILD_SETTLEMENT": 0.9728, "('play_turn', 6, 0, 4, 0, 0, 0)|BUILD_ROAD": 0.2162, "('play_turn', 6, 0, 4, 0, 0, 0)|END_TURN": -0.0929, "('play_turn', 6, 0, 4, 0, 0, 0)|MARITIME_TRADE": -0.1355, "('play_turn', 6, 0, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.337, "('play_turn', 6, 0, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.1319, "('play_turn', 6, 0, 5, 0, 0, 0)|BUILD_ROAD": 0.0952, "('play_turn', 6, 0, 5, 0, 0, 0)|END_TURN": -0.1035, "('play_turn', 6, 0, 5, 0, 0, 0)|MARITIME_TRADE": 0.1653, "('play_turn', 6, 0, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.0909, "('play_turn', 6, 0, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.2731, "('play_turn', 6, 0, 6, 0, 0, 0)|BUILD_ROAD": 0.6553, "('play_turn', 6, 0, 6, 0, 0, 0)|END_TURN": 0.1383, "('play_turn', 6, 0, 6, 0, 0, 0)|MARITIME_TRADE": 0.9937, "('play_turn', 6, 0, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.8308, "('play_turn', 6, 0, 6, 0, 0, 0)|PLAY_MONOPOLY": 0.4743, "('play_turn', 6, 0, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.1009, "('play_turn', 6, 0, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.6342, "('play_turn', 6, 0, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.9765, "('play_turn', 6, 0, 6, 0, 0, 1)|END_TURN": 0.492, "('play_turn', 6, 0, 6, 0, 0, 1)|MARITIME_TRADE": 0.6242, "('play_turn', 6, 0, 6, 0, 1, 0)|BUILD_ROAD": 0.9139, "('play_turn', 6, 0, 6, 0, 1, 0)|BUILD_SETTLEMENT": 1.9313, "('play_turn', 6, 0, 6, 0, 1, 0)|END_TURN": 0.4405, "('play_turn', 6, 0, 6, 0, 1, 1)|BUILD_SETTLEMENT": 1.7298, "('play_turn', 6, 0, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 1.0539, "('play_turn', 6, 0, 6, 1, 0, 0)|BUILD_CITY": 1.8953, "('play_turn', 6, 0, 6, 1, 0, 0)|MARITIME_TRADE": 0.3957, "('play_turn', 6, 0, 6, 1, 0, 1)|BUILD_CITY": 1.2252, "('play_turn', 6, 1, 3, 0, 0, 0)|END_TURN": 0.0463, "('play_turn', 6, 1, 4, 0, 0, 0)|BUILD_ROAD": 0.0377, "('play_turn', 6, 1, 4, 0, 0, 0)|END_TURN": 0.167, "('play_turn', 6, 1, 4, 0, 0, 0)|MARITIME_TRADE": 0.3299, "('play_turn', 6, 1, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.4809, "('play_turn', 6, 1, 4, 0, 0, 0)|ROLL": 0.131, "('play_turn', 6, 1, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.736, "('play_turn', 6, 1, 4, 0, 1, 0)|BUILD_SETTLEMENT": 1.0434, "('play_turn', 6, 1, 4, 0, 1, 1)|BUILD_SETTLEMENT": 0.9371, "('play_turn', 6, 1, 5, 0, 0, 0)|BUILD_ROAD": 0.4809, "('play_turn', 6, 1, 5, 0, 0, 0)|END_TURN": 0.3177, "('play_turn', 6, 1, 5, 0, 0, 0)|MARITIME_TRADE": 0.5371, "('play_turn', 6, 1, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.0302, "('play_turn', 6, 1, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.1923, "('play_turn', 6, 1, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.1617, "('play_turn', 6, 1, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.8485, "('play_turn', 6, 1, 5, 0, 1, 0)|BUILD_SETTLEMENT": 1.4351, "('play_turn', 6, 1, 5, 0, 1, 1)|BUILD_SETTLEMENT": 0.9371, "('play_turn', 6, 1, 6, 0, 0, 0)|BUILD_ROAD": 1.0221, "('play_turn', 6, 1, 6, 0, 0, 0)|END_TURN": 0.6458, "('play_turn', 6, 1, 6, 0, 0, 0)|MARITIME_TRADE": 1.2206, "('play_turn', 6, 1, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.0843, "('play_turn', 6, 1, 6, 0, 0, 0)|PLAY_MONOPOLY": 1.0376, "('play_turn', 6, 1, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.941, "('play_turn', 6, 1, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.1317, "('play_turn', 6, 1, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.8951, "('play_turn', 6, 1, 6, 0, 0, 1)|END_TURN": 0.4419, "('play_turn', 6, 1, 6, 0, 0, 1)|MARITIME_TRADE": 1.3171, "('play_turn', 6, 1, 6, 0, 1, 0)|BUILD_ROAD": 1.3144, "('play_turn', 6, 1, 6, 0, 1, 0)|BUILD_SETTLEMENT": 1.9993, "('play_turn', 6, 1, 6, 0, 1, 0)|END_TURN": 0.3774, "('play_turn', 6, 1, 6, 0, 1, 0)|MARITIME_TRADE": 0.3699, "('play_turn', 6, 1, 6, 0, 1, 1)|BUILD_ROAD": 1.1246, "('play_turn', 6, 1, 6, 0, 1, 1)|BUILD_SETTLEMENT": 1.9907, "('play_turn', 6, 1, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.5541, "('play_turn', 6, 1, 6, 0, 1, 1)|END_TURN": 0.2737, "('play_turn', 6, 1, 6, 1, 0, 0)|BUILD_CITY": 1.996, "('play_turn', 6, 1, 6, 1, 0, 0)|BUILD_ROAD": 0.3904, "('play_turn', 6, 1, 6, 1, 0, 0)|END_TURN": 0.2179, "('play_turn', 6, 1, 6, 1, 0, 0)|MARITIME_TRADE": 0.6899, "('play_turn', 6, 1, 6, 1, 0, 1)|BUILD_CITY": 1.7298, "('play_turn', 6, 1, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 0.8393, "('play_turn', 6, 1, 6, 1, 1, 1)|BUILD_SETTLEMENT": 0.9371, "('play_turn', 6, 2, 3, 0, 0, 0)|BUILD_ROAD": -0.2216, "('play_turn', 6, 2, 3, 0, 0, 0)|END_TURN": -0.2546, "('play_turn', 6, 2, 3, 0, 0, 0)|MARITIME_TRADE": 0.0248, "('play_turn', 6, 2, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.133, "('play_turn', 6, 2, 4, 0, 0, 0)|BUILD_ROAD": 0.0528, "('play_turn', 6, 2, 4, 0, 0, 0)|END_TURN": -0.026, "('play_turn', 6, 2, 4, 0, 0, 0)|MARITIME_TRADE": 0.2669, "('play_turn', 6, 2, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.2331, "('play_turn', 6, 2, 4, 0, 0, 0)|ROLL": 0.0246, "('play_turn', 6, 2, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.5194, "('play_turn', 6, 2, 4, 0, 1, 0)|BUILD_SETTLEMENT": 0.9371, "('play_turn', 6, 2, 4, 0, 1, 1)|BUILD_SETTLEMENT": 1.3724, "('play_turn', 6, 2, 4, 1, 0, 0)|BUILD_CITY": 0.9371, "('play_turn', 6, 2, 5, 0, 0, 0)|BUILD_ROAD": 0.8555, "('play_turn', 6, 2, 5, 0, 0, 0)|END_TURN": 0.5885, "('play_turn', 6, 2, 5, 0, 0, 0)|MARITIME_TRADE": 1.0474, "('play_turn', 6, 2, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.5848, "('play_turn', 6, 2, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.1445, "('play_turn', 6, 2, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.3721, "('play_turn', 6, 2, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.3187, "('play_turn', 6, 2, 5, 0, 1, 0)|BUILD_ROAD": 0.2808, "('play_turn', 6, 2, 5, 0, 1, 0)|BUILD_SETTLEMENT": 1.6665, "('play_turn', 6, 2, 5, 0, 1, 1)|BUILD_SETTLEMENT": 1.2252, "('play_turn', 6, 2, 5, 1, 0, 0)|BUILD_CITY": 1.4916, "('play_turn', 6, 2, 6, 0, 0, 0)|BUILD_ROAD": 1.6695, "('play_turn', 6, 2, 6, 0, 0, 0)|END_TURN": 1.2947, "('play_turn', 6, 2, 6, 0, 0, 0)|MARITIME_TRADE": 1.5047, "('play_turn', 6, 2, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.3366, "('play_turn', 6, 2, 6, 0, 0, 0)|PLAY_MONOPOLY": 1.3177, "('play_turn', 6, 2, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 0.3449, "('play_turn', 6, 2, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.5818, "('play_turn', 6, 2, 6, 0, 0, 0)|ROLL": 0.4975, "('play_turn', 6, 2, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.6206, "('play_turn', 6, 2, 6, 0, 0, 1)|END_TURN": 1.1855, "('play_turn', 6, 2, 6, 0, 0, 1)|MARITIME_TRADE": 1.7443, "('play_turn', 6, 2, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 0.4032, "('play_turn', 6, 2, 6, 0, 1, 0)|BUILD_ROAD": 1.6199, "('play_turn', 6, 2, 6, 0, 1, 0)|BUILD_SETTLEMENT": 2.0, "('play_turn', 6, 2, 6, 0, 1, 0)|END_TURN": 0.6106, "('play_turn', 6, 2, 6, 0, 1, 0)|MARITIME_TRADE": 0.8978, "('play_turn', 6, 2, 6, 0, 1, 0)|PLAY_ROAD_BUILDING": 0.2863, "('play_turn', 6, 2, 6, 0, 1, 1)|BUILD_ROAD": 1.5538, "('play_turn', 6, 2, 6, 0, 1, 1)|BUILD_SETTLEMENT": 2.0, "('play_turn', 6, 2, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 0.8606, "('play_turn', 6, 2, 6, 0, 1, 1)|END_TURN": 0.4916, "('play_turn', 6, 2, 6, 1, 0, 0)|BUILD_CITY": 2.0, "('play_turn', 6, 2, 6, 1, 0, 0)|END_TURN": 0.9662, "('play_turn', 6, 2, 6, 1, 0, 0)|MARITIME_TRADE": 1.3204, "('play_turn', 6, 2, 6, 1, 0, 1)|BUILD_CITY": 1.8405, "('play_turn', 6, 2, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 1.0843, "('play_turn', 6, 3, 3, 0, 0, 0)|BUILD_ROAD": 0.6602, "('play_turn', 6, 3, 3, 0, 0, 0)|END_TURN": 0.412, "('play_turn', 6, 3, 3, 0, 0, 0)|MARITIME_TRADE": 0.1562, "('play_turn', 6, 3, 3, 0, 0, 0)|PLAY_KNIGHT_CARD": 0.3919, "('play_turn', 6, 3, 3, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 0.5119, "('play_turn', 6, 3, 3, 0, 1, 1)|BUILD_SETTLEMENT": 0.819, "('play_turn', 6, 3, 4, 0, 0, 0)|BUILD_ROAD": 0.9265, "('play_turn', 6, 3, 4, 0, 0, 0)|END_TURN": 0.9716, "('play_turn', 6, 3, 4, 0, 0, 0)|MARITIME_TRADE": 1.2054, "('play_turn', 6, 3, 4, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.0741, "('play_turn', 6, 3, 4, 0, 0, 0)|PLAY_MONOPOLY": 0.5697, "('play_turn', 6, 3, 4, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 0.7974, "('play_turn', 6, 3, 4, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.1821, "('play_turn', 6, 3, 4, 0, 0, 1)|END_TURN": 0.1571, "('play_turn', 6, 3, 4, 0, 1, 0)|BUILD_SETTLEMENT": 1.9672, "('play_turn', 6, 3, 4, 0, 1, 1)|BUILD_SETTLEMENT": 1.6998, "('play_turn', 6, 3, 4, 1, 0, 0)|BUILD_CITY": 1.3026, "('play_turn', 6, 3, 5, 0, 0, 0)|BUILD_ROAD": 0.9641, "('play_turn', 6, 3, 5, 0, 0, 0)|END_TURN": 1.1026, "('play_turn', 6, 3, 5, 0, 0, 0)|MARITIME_TRADE": 1.0952, "('play_turn', 6, 3, 5, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.5259, "('play_turn', 6, 3, 5, 0, 0, 0)|PLAY_MONOPOLY": 0.7822, "('play_turn', 6, 3, 5, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.1644, "('play_turn', 6, 3, 5, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.39, "('play_turn', 6, 3, 5, 0, 0, 1)|END_TURN": 0.7354, "('play_turn', 6, 3, 5, 0, 0, 1)|MARITIME_TRADE": 0.6311, "('play_turn', 6, 3, 5, 0, 1, 0)|BUILD_ROAD": 1.2014, "('play_turn', 6, 3, 5, 0, 1, 0)|BUILD_SETTLEMENT": 1.9806, "('play_turn', 6, 3, 5, 0, 1, 1)|BUILD_SETTLEMENT": 1.6294, "('play_turn', 6, 3, 5, 1, 0, 0)|BUILD_CITY": 1.6294, "('play_turn', 6, 3, 5, 1, 0, 0)|END_TURN": 0.4633, "('play_turn', 6, 3, 5, 1, 0, 1)|BUILD_CITY": 1.3026, "('play_turn', 6, 3, 6, 0, 0, 0)|BUILD_ROAD": 1.7018, "('play_turn', 6, 3, 6, 0, 0, 0)|END_TURN": 1.6061, "('play_turn', 6, 3, 6, 0, 0, 0)|MARITIME_TRADE": 1.7296, "('play_turn', 6, 3, 6, 0, 0, 0)|PLAY_KNIGHT_CARD": 1.9747, "('play_turn', 6, 3, 6, 0, 0, 0)|PLAY_MONOPOLY": 1.6049, "('play_turn', 6, 3, 6, 0, 0, 0)|PLAY_ROAD_BUILDING": 1.8041, "('play_turn', 6, 3, 6, 0, 0, 0)|PLAY_YEAR_OF_PLENTY": 1.6821, "('play_turn', 6, 3, 6, 0, 0, 0)|ROLL": 1.2907, "('play_turn', 6, 3, 6, 0, 0, 1)|BUY_DEVELOPMENT_CARD": 1.7728, "('play_turn', 6, 3, 6, 0, 0, 1)|END_TURN": 1.5188, "('play_turn', 6, 3, 6, 0, 0, 1)|MARITIME_TRADE": 1.8383, "('play_turn', 6, 3, 6, 0, 0, 1)|PLAY_KNIGHT_CARD": 2.0529, "('play_turn', 6, 3, 6, 0, 0, 1)|PLAY_MONOPOLY": 0.707, "('play_turn', 6, 3, 6, 0, 0, 1)|PLAY_YEAR_OF_PLENTY": 0.7197, "('play_turn', 6, 3, 6, 0, 1, 0)|BUILD_ROAD": 1.9527, "('play_turn', 6, 3, 6, 0, 1, 0)|BUILD_SETTLEMENT": 2.0, "('play_turn', 6, 3, 6, 0, 1, 0)|END_TURN": 1.5257, "('play_turn', 6, 3, 6, 0, 1, 0)|MARITIME_TRADE": 1.4787, "('play_turn', 6, 3, 6, 0, 1, 1)|BUILD_ROAD": 1.2529, "('play_turn', 6, 3, 6, 0, 1, 1)|BUILD_SETTLEMENT": 2.0, "('play_turn', 6, 3, 6, 0, 1, 1)|BUY_DEVELOPMENT_CARD": 1.8025, "('play_turn', 6, 3, 6, 0, 1, 1)|END_TURN": 1.278, "('play_turn', 6, 3, 6, 0, 1, 1)|MARITIME_TRADE": 0.6222, "('play_turn', 6, 3, 6, 0, 1, 1)|PLAY_ROAD_BUILDING": 0.2699, "('play_turn', 6, 3, 6, 1, 0, 0)|BUILD_CITY": 2.0, "('play_turn', 6, 3, 6, 1, 0, 0)|END_TURN": 1.2863, "('play_turn', 6, 3, 6, 1, 0, 0)|MARITIME_TRADE": 1.5344, "('play_turn', 6, 3, 6, 1, 0, 1)|BUILD_CITY": 1.9981, "('play_turn', 6, 3, 6, 1, 0, 1)|BUY_DEVELOPMENT_CARD": 1.63, "('play_turn', 6, 3, 6, 1, 0, 1)|END_TURN": 0.5892, "('play_turn', 6, 3, 6, 1, 0, 1)|MARITIME_TRADE": 1.1859, "('play_turn', 6, 3, 6, 1, 1, 1)|BUILD_CITY": 0.819, "('play_turn', 6, 3, 6, 1, 1, 1)|BUILD_SETTLEMENT": 1.4916}


def _bucket(value: float, size: float, top: int) -> int:
    return int(max(0, min(top, value // size)))


def state_features(observation: dict) -> tuple:
    """Abstract state: phase, my VP, VP lead, production, what I can afford now."""
    me, rival = observation["me"], observation["opponent"]
    hand = _hand_tuple(me["resources"])
    production = 0
    board = observation["board"]
    robber = _key(observation["robber_tile"])
    buildings = {n: 1 for n in me["settlements"]} | {n: 2 for n in me["cities"]}
    for tile in board["tiles"]:
        if tile["number"] and _key(tile["tile"]) != robber:
            production += PIPS[tile["number"]] * sum(buildings.get(n, 0) for n in tile["nodes"])
    lead = me["victory_points"] - rival["visible_victory_points"]
    return (
        observation["phase"],
        min(me["victory_points"], 7),
        max(-3, min(3, lead)),
        _bucket(production, 5, 6),
        int(_pay(hand, COSTS["CITY"]) is not None),
        int(_pay(hand, COSTS["SETTLEMENT"]) is not None),
        int(_pay(hand, COSTS["DEVELOPMENT_CARD"]) is not None),
    )


def action_kind(action: dict) -> str:
    return action["type"]


def q_value(features: tuple, kind: str) -> float:
    return Q.get(f"{features}|{kind}", 0.0)


def q_update(q, reward, best_next_q, alpha, gamma):
    """Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))."""
    return q + alpha * (reward + gamma * best_next_q - q)


def action_value(observation, action_id):
    """Afterstate value from the planner plus the learned Q bonus, in VP-hundredths."""
    action = next(a for a in observation["actions"] if a["id"] == action_id)
    if observation["phase"] == "initial_placement":
        board, _ = MEMORY.refresh(observation)
        if action["type"] == "BUILD_SETTLEMENT":
            me, rival = observation["me"], observation["opponent"]
            occupied = set(me["settlements"]) | set(rival["settlements"])
            return float(placement_value(board, action["node"], me["settlements"], occupied))
        return 0.0
    value = Evaluator(observation).value(action)
    return float(value + WEIGHTS["vp"] * WEIGHTS["q_bonus"] * q_value(state_features(observation), action_kind(action)))


BASELINE_PRIORITY = (
    "BUILD_CITY", "BUILD_SETTLEMENT", "BUY_DEVELOPMENT_CARD", "PLAY_KNIGHT_CARD",
    "PLAY_YEAR_OF_PLENTY", "PLAY_MONOPOLY", "PLAY_ROAD_BUILDING", "BUILD_ROAD", "ROLL", "END_TURN",
)


def heuristic_policy(observation):
    """Baseline: settle on the most pips, then a fixed build priority, no trades."""
    actions = observation["actions"]
    if actions[0]["type"] == "BUILD_SETTLEMENT" and observation["phase"] == "initial_placement":
        production = {}
        for tile in observation["board"]["tiles"]:
            for node in tile["nodes"]:
                production[node] = production.get(node, 0) + PIPS.get(tile["number"], 0)
        return max(actions, key=lambda a: production.get(a["node"], 0))["id"]
    if observation["phase"] == "move_robber":
        stealing = [a for a in actions if a.get("steal_from") == "opponent"]
        return (stealing or actions)[0]["id"]
    for kind in BASELINE_PRIORITY:
        for action in actions:
            if action["type"] == kind:
                return action["id"]
    return actions[0]["id"]


# ---------------------------------------------------------------------------------
# The advanced policy
# ---------------------------------------------------------------------------------


def advanced_policy(observation: dict, q_bonus: float = None) -> int:
    if observation["phase"] == "initial_placement":
        return choose_initial(observation)
    evaluator = Evaluator(observation)
    weight = WEIGHTS["q_bonus"] if q_bonus is None else q_bonus
    features = state_features(observation) if weight else None
    best_id, best_value = None, None
    for action in observation["actions"]:
        value = evaluator.value(action)
        if weight:
            value += WEIGHTS["vp"] * weight * q_value(features, action_kind(action))
        if best_value is None or value > best_value:
            best_id, best_value = action["id"], value
    return best_id


def act(observation):
    """Advanced policy, with the O(1) baseline when the clock is nearly out.

    Any unexpected error also falls back to the baseline: an exception would
    forfeit the game, a weaker move only costs a little.
    """
    remaining = observation.get("time_remaining_s")
    if remaining is not None and remaining < CLOCK_RESERVE_S:
        return {"action": heuristic_policy(observation)}
    try:
        return {"action": advanced_policy(observation)}
    except Exception:
        return {"action": heuristic_policy(observation)}
