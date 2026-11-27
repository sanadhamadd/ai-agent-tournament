"""European Road Trip agent.

Goal: the shortest road route that starts in Madrid, visits every required city and
returns to Madrid, without passing through any city twice.

How the agent works (the README explains each step):

  1. GRAPH     Cities become integers 0..N-1 and roads an adjacency list. A set of cities
               is stored as a bitmask (one Python int), so "does this path touch a city
               that is already on the route?" is a single AND.
  2. SEARCH    One best-first search, _search(), ordered by f = g + h. With h = 0 it is
               uniform-cost search (ucs); with h = straight-line distance it is A*
               (astar). The agent uses the A* form, with a set of blocked cities, to
               find the shortest path between two cities that avoids the rest of the route.
  3. BUILD     build_by_insertion() makes the first route directly on the road graph:
               a loop through one city, then every other required city is inserted
               where it adds the fewest km. The route is valid by construction and is
               submitted at once.
  4. IMPROVE   _anneal() is simulated annealing on the real route. Its moves are the
               2-opt and Or-opt moves of local_search(), applied to the route itself:
               the legs they break are rebuilt with A* around the rest of the route.
               It runs many short rounds, some from a fresh build and some from the
               best route so far, and submits every improvement.

The state of the search is always a complete valid route, so anything it finds can be
submitted. Only the Python standard library is used.
"""
import heapq
import math
import random
import time
import traceback

EARTH_RADIUS_KM = 6371.0
INFINITY = float("inf")

# Settings of the annealer, chosen by measurement on 400 random draws (see the README).
NEAR = 12        # a move pairs a stop with one of its NEAR nearest required cities
HOT = 500.0      # starting temperature of a round, in km
COLD = 3.0       # final temperature of a round, in km
ROUNDS = 24      # annealing rounds within the time limit
FRESH = 16       # the first FRESH rounds start from a new insertion build


# ---------------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------------


def straight_line_km(a, b):
    """Haversine distance between two entries of problem["cities"]."""
    dlat = math.radians(b["lat"] - a["lat"])
    dlon = math.radians(b["lon"] - a["lon"])
    h = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(a["lat"])) * math.cos(math.radians(b["lat"])) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def _build_graph(graph):
    """Return (ids, index, adjacency): city ids, id -> integer, and for each integer
    a list of (neighbour, km)."""
    ids = sorted(graph["cities"])
    index = {city: i for i, city in enumerate(ids)}
    table = [dict() for _ in ids]
    for road in graph["roads"]:
        a, b, km = index[road["from"]], index[road["to"]], road["km"]
        table[a][b] = km
        if road.get("bidirectional", True):
            table[b][a] = km
    adjacency = [sorted(neighbours.items()) for neighbours in table]
    return ids, index, adjacency


def _heuristic_scale(graph):
    """1.0 when every road is at least as long as the straight line between its ends
    (true on the tournament map). Otherwise the largest factor that makes
    factor * straight_line <= km on every road, which keeps the heuristic admissible
    and consistent on any map."""
    cities, scale = graph["cities"], 1.0
    for road in graph["roads"]:
        straight = straight_line_km(cities[road["from"]], cities[road["to"]])
        if straight > road["km"]:
            scale = min(scale, road["km"] / straight)
    return scale


def _heuristic_to(graph, ids, goal, scale):
    """h[n] = scaled straight-line km from city n to goal, for every city."""
    cities = graph["cities"]
    target = cities[goal]
    return [scale * straight_line_km(cities[city], target) for city in ids]


def _search(adjacency, start, goal, h, blocked=0):
    """Best-first graph search ordered by f = g + h.

    adjacency  list of [(neighbour, km), ...] indexed by city number
    h          h[n] = estimate of the km from n to goal. All zeros = uniform-cost search.
               An admissible, consistent estimate = A*.
    blocked    bitmask of cities the path may not use (start and goal must not be in it)

    A city is expanded when it is removed from the frontier to explore its neighbours.
    With a consistent h, the first time a city is removed its g is optimal, so each city
    is expanded at most once (the closed set) and the search can stop when the goal is
    removed. Ties on f are broken towards the smaller h, that is, the city nearer the goal.

    Returns (km, path, expanded). km and path are None when there is no path.
    """
    best_g = {start: 0}
    parent = {start: -1}
    closed = blocked
    expanded = 0
    frontier = [(h[start], h[start], 0, start)]
    pop, push = heapq.heappop, heapq.heappush
    while frontier:
        _, _, g, city = pop(frontier)
        if closed >> city & 1:
            continue  # an older, worse entry for a city already expanded
        closed |= 1 << city
        expanded += 1
        if city == goal:
            path = [goal]
            while parent[path[-1]] != -1:
                path.append(parent[path[-1]])
            path.reverse()
            return g, path, expanded
        for neighbour, km in adjacency[city]:
            if closed >> neighbour & 1:
                continue
            new_g = g + km
            if new_g < best_g.get(neighbour, INFINITY):
                best_g[neighbour] = new_g
                parent[neighbour] = city
                estimate = h[neighbour]
                push(frontier, (new_g + estimate, estimate, new_g, neighbour))
    return None, None, expanded


def _tree(adjacency, start, dist):
    """Uniform-cost search from start to every city at once (no goal, so it runs until
    the frontier is empty). Fills dist[n] with the km and returns parent[n], the city
    before n on its shortest path (-1 for start and for unreachable cities)."""
    parent = [-1] * len(adjacency)
    dist[start] = 0
    frontier = [(0, start)]
    while frontier:
        g, city = heapq.heappop(frontier)
        if g > dist[city]:
            continue
        for neighbour, km in adjacency[city]:
            if g + km < dist[neighbour]:
                dist[neighbour] = g + km
                parent[neighbour] = city
                heapq.heappush(frontier, (g + km, neighbour))
    return parent


# ---------------------------------------------------------------------------------
# Required algorithms. check_requirements() tests these, and solve() uses the same code.
# ---------------------------------------------------------------------------------


def ucs(graph, start, goal):
    """Uniform-cost search (Dijkstra) on graph = {"cities": ..., "roads": ...}.

    Returns {"path": [start, ..., goal], "km": total_km, "expanded": nodes_expanded}.
    """
    ids, index, adjacency = _build_graph(graph)
    km, path, expanded = _search(adjacency, index[start], index[goal], [0] * len(ids))
    return {"path": [ids[i] for i in path] if path else [], "km": km, "expanded": expanded}


def astar(graph, start, goal):
    """A* with the straight-line (Haversine) distance to the goal as the heuristic.

    Every road is at least as long as the straight line between its ends, so the
    heuristic never overestimates (admissible) and obeys the triangle inequality
    (consistent). A* therefore returns the same km as ucs(). Measured on this map it
    never expands more cities than UCS: 0 cases in all 10,920 ordered pairs, and
    184 against 392 on the judge's 8 test pairs.
    """
    ids, index, adjacency = _build_graph(graph)
    h = _heuristic_to(graph, ids, goal, _heuristic_scale(graph))
    km, path, expanded = _search(adjacency, index[start], index[goal], h)
    return {"path": [ids[i] for i in path] if path else [], "km": km, "expanded": expanded}


def local_search(tour, distance, time_limit_s):
    """Iterated hill climbing on a closed tour. The first city stays first.

    Neighbourhood   2-opt (reverse a stretch of the tour) and Or-opt (move a stretch of
                    1 to 3 cities somewhere else, either way round).
    Acceptance      hill climbing: take a move only if the tour gets shorter, repeat
                    until no move helps (a local optimum).
    Escaping        random restarts from the best tour: a random stretch is reversed and
                    another is moved (a "kick"), then hill climbing runs again. The
                    result is kept if it is no longer than the best so far.

    tour      different city ids in visiting order, first city not repeated at the end
    distance  {a: {b: km}} for every pair of different cities in the tour
    Returns a reordering of tour with the same first city, never longer than the input.
    """
    started = time.monotonic()
    n = len(tour)
    if n <= 3:
        return list(tour)  # with 3 cities or fewer, every order has the same length
    deadline = started + max(0.0, float(time_limit_s)) * 0.8
    d = [[0 if a == b else distance[a][b] for b in tour] for a in tour]
    rng = random.Random(n)

    def length(order):
        return sum(d[order[i - 1]][order[i]] for i in range(n))

    def hill_climb(order):
        """Apply improving 2-opt and Or-opt moves until none is left."""
        improved = True
        while improved:
            improved = False
            # 2-opt: replace roads (a,b) and (c,e) with (a,c) and (b,e), reversing b..c.
            for i in range(1, n - 1):
                a, b = order[i - 1], order[i]
                for j in range(i + 1, n):
                    c, e = order[j], order[(j + 1) % n]
                    if d[a][c] + d[b][e] < d[a][b] + d[c][e] - 1e-9:
                        order[i:j + 1] = order[i:j + 1][::-1]
                        improved = True
                        b = order[i]
            # Or-opt: move a stretch of 1 to 3 cities between two other neighbours.
            for size in (1, 2, 3):
                if n - 1 <= size:
                    break
                i = 1
                while i + size <= n:
                    before, first, last = order[i - 1], order[i], order[i + size - 1]
                    after = order[(i + size) % n]
                    saved = d[before][first] + d[last][after] - d[before][after]
                    rest = order[:i] + order[i + size:]
                    best_gain, best_at, best_flip = 1e-9, -1, False
                    for k in range(len(rest)):
                        p, q = rest[k], rest[(k + 1) % len(rest)]
                        if p == before and q == after:
                            continue  # that is where the stretch already is
                        base = saved + d[p][q]
                        gain = base - d[p][first] - d[last][q]
                        if gain > best_gain:
                            best_gain, best_at, best_flip = gain, k, False
                        gain = base - d[p][last] - d[first][q]
                        if gain > best_gain:
                            best_gain, best_at, best_flip = gain, k, True
                    if best_at >= 0:
                        stretch = order[i:i + size]
                        if best_flip:
                            stretch.reverse()
                        order[:] = rest[:best_at + 1] + stretch + rest[best_at + 1:]
                        improved = True
                    else:
                        i += 1
            if time.monotonic() > deadline:
                break
        return order

    best = hill_climb(list(range(n)))
    best_length = length(best)
    stale = 0
    while stale < 30 * n and time.monotonic() < deadline:
        candidate = best[:]
        i, j = sorted(rng.sample(range(1, n), 2))
        candidate[i:j + 1] = candidate[i:j + 1][::-1]
        i, j = rng.randrange(1, n), rng.randrange(1, n)
        candidate.insert(j, candidate.pop(i))
        candidate = hill_climb(candidate)
        candidate_length = length(candidate)
        if candidate_length < best_length - 1e-9:
            best, best_length, stale = candidate, candidate_length, 0
        else:
            stale += 1
            if candidate_length <= best_length:
                best = candidate  # same length: walk sideways along the plateau
    if best_length > length(list(range(n))):
        return list(tour)
    return [tour[i] for i in best]


# ---------------------------------------------------------------------------------
# The competing agent
# ---------------------------------------------------------------------------------


class _Planner:
    """Everything solve() needs about one problem, with cities as integers."""

    def __init__(self, problem):
        self.ids, self.index, self.adjacency = _build_graph(problem)
        ids, index, adjacency = self.ids, self.index, self.adjacency
        n = len(ids)
        self.start = index[problem["start"]]
        self.start_bit = 1 << self.start
        required = []
        for city in problem["required"]:
            if index[city] not in required:
                required.append(index[city])
        if self.start not in required:
            required.insert(0, self.start)
        self.required = required
        self.stops = [city for city in required if city != self.start]
        self.required_mask = sum(1 << city for city in required)
        self.edge = [dict(neighbours) for neighbours in adjacency]

        # Heuristic tables for A*: h[goal][n] for every required goal.
        scale = _heuristic_scale(problem)
        self.h = [None] * n
        for city in required:
            self.h[city] = _heuristic_to(problem, ids, ids[city], scale)

        # Shortest paths between every pair of required cities (uniform-cost search
        # from each one). dist[a][b] = km, path[a][b] = cities after a up to b,
        # mask[a][b] = those cities as a bitmask.
        self.dist = [None] * n
        self.path = [None] * n
        self.mask = [None] * n
        for a in required:
            self.dist[a] = [INFINITY] * n
            self.path[a] = [None] * n
            self.mask[a] = [0] * n
            parent = _tree(adjacency, a, self.dist[a])
            for b in required:
                if a == b or parent[b] == -1:
                    continue
                way = [b]
                while parent[way[-1]] != a:
                    way.append(parent[way[-1]])
                way.reverse()
                self.path[a][b] = way
                self.mask[a][b] = sum(1 << city for city in way)
        self.cache = {}
        # For each required city, the other required cities nearest to it by road.
        self.near = {a: sorted((b for b in required if b != a), key=lambda b: (self.dist[a][b], b))[:NEAR]
                     for a in required}

    # -- paths that avoid used cities ----------------------------------------------

    def detour(self, a, b, used):
        """Shortest path from a to b that avoids the cities in the bitmask used.
        a and b themselves are always allowed. Returns (km, cities after a up to b,
        their bitmask) or None. Results are cached."""
        blocked = used & ~(1 << a) & ~(1 << b)
        if not self.mask[a][b] & blocked and self.path[a][b] is not None:
            return self.dist[a][b], self.path[a][b], self.mask[a][b]
        key = (a, b, blocked)
        hit = self.cache.get(key, 0)
        if hit == 0:
            km, way, _ = _search(self.adjacency, a, b, self.h[b], blocked)
            hit = None if way is None else (km, way[1:], sum(1 << city for city in way[1:]))
            if len(self.cache) > 300000:
                self.cache.clear()
            self.cache[key] = hit
        return hit

    # -- route helpers ---------------------------------------------------------------

    def legs_km(self, legs):
        total, here = 0, self.start
        for leg in legs:
            for city in leg:
                total += self.edge[here][city]
                here = city
        return total

    def flatten(self, legs):
        cities = [self.start]
        for leg in legs:
            cities.extend(leg)
        return cities

    def is_valid(self, cities):
        """The judge's rules: starts and ends at the start, no city twice, every
        required city present, every step along a real road."""
        if len(cities) < 3 or cities[0] != self.start or cities[-1] != self.start:
            return False
        middle = cities[1:-1]
        if self.start in middle or len(set(middle)) != len(middle):
            return False
        seen = set(cities)
        if any(city not in seen for city in self.required):
            return False
        return all(b in self.edge[a] for a, b in zip(cities, cities[1:]))

    def split(self, cities):
        """Cut a flat route into legs that each end at a required city."""
        legs, leg = [], []
        for city in cities[1:]:
            leg.append(city)
            if self.required_mask >> city & 1:
                legs.append(leg)
                leg = []
        return legs

    def insert(self, legs, stop):
        """Put stop into the stretch where it adds the fewest km. Returns new legs or None.
        For each stretch a..b it tries a->stop then stop->b, and the other way round
        (stop->b first, then a->stop), because the first path chosen can block the second."""
        best, best_legs = INFINITY, None
        everything = self.start_bit
        for leg in legs:
            for city in leg:
                everything |= 1 << city
        for i, leg in enumerate(legs):
            a = self.start if i == 0 else legs[i - 1][-1]
            b = leg[-1]
            inner = 0
            for city in leg[:-1]:
                inner |= 1 << city
            used = everything & ~inner
            old = self.legs_km_between(a, leg)
            for first_in in (True, False):
                if first_in:
                    one = self.detour(a, stop, used)
                    if one is None:
                        continue
                    two = self.detour(stop, b, used | one[2])
                    if two is None:
                        continue
                else:
                    two = self.detour(stop, b, used)
                    if two is None:
                        continue
                    two_cities = two[2] & ~(1 << b) | (1 << stop)
                    one = self.detour(a, stop, used | two_cities)
                    if one is None:
                        continue
                added = one[0] + two[0] - old
                if added < best:
                    best = added
                    best_legs = legs[:i] + [list(one[1]), list(two[1])] + legs[i + 1:]
        return best_legs

    def legs_km_between(self, a, leg):
        total = 0
        for city in leg:
            total += self.edge[a][city]
            a = city
        return total

    def build_by_insertion(self, rng):
        """Cheapest insertion on the real road graph. Start from the loop
        start -> one stop -> start, then insert the other stops one by one in random
        order, each into the leg where it adds the fewest km. Every intermediate
        result is a valid loop, so the final route is valid by construction.
        Returns legs, or None if some stop could not be inserted (then the caller
        tries again with another random order)."""
        stops = self.stops[:]
        rng.shuffle(stops)
        legs = None
        for first in stops:
            out = self.detour(self.start, first, self.start_bit)
            if out is None:
                continue
            back = self.detour(first, self.start, self.start_bit | out[2])
            if back is not None:
                legs = [list(out[1]), list(back[1])]
                break
        if legs is None:
            return None
        for stop in stops:
            # A new leg may pass through a required city that was still waiting. Cutting
            # the route at every required city turns it into a stop, so a later
            # re-route of that leg can never drop it.
            legs = self.split(self.flatten(legs))
            if any(leg[-1] == stop for leg in legs):
                continue
            legs = self.insert(legs, stop)
            if legs is None:
                return None
        return self.split(self.flatten(legs))


def solve(problem, submit):
    """Called once by the judge. Submits a valid route early, then better ones."""
    entered = time.monotonic()
    limit = float(problem.get("time_limit_s") or 5)
    # The judge's clock started slightly before ours and ignores late routes, so stop early.
    deadline = entered + max(limit * 0.5, limit * 0.93 - 0.05)
    rng = random.Random(20261126)
    state = {"km": INFINITY, "legs": None}

    try:
        planner = _Planner(problem)
    except Exception:
        traceback.print_exc()  # goes to the practice log only; nothing can be submitted
        return
    ids = planner.ids

    def offer(legs):
        """Submit these legs if they form a valid route shorter than the best so far."""
        if legs is None:
            return False
        cities = planner.flatten(legs)
        if not planner.is_valid(cities):
            return False
        km = planner.legs_km(legs)
        if km >= state["km"]:
            return False
        state["km"], state["legs"] = km, [list(leg) for leg in legs]
        submit([ids[city] for city in cities])
        return True

    try:
        _optimise(planner, rng, deadline, offer, state)
    except Exception:
        traceback.print_exc()  # whatever was submitted before the error still counts


def _optimise(planner, rng, deadline, offer, state):
    stops = planner.stops
    if not stops:
        return
    now = time.monotonic

    # --- First valid route, as early as possible --------------------------------------
    # Cheapest insertion builds a real route directly, so it is valid by construction.
    # If an insertion order gets stuck, try again with a new random order.
    build_seconds = 0.0
    while now() < deadline and state["legs"] is None:
        build_start = now()
        offer(planner.build_by_insertion(rng))
        build_seconds = now() - build_start
    # With one or two stops there is nothing to reorder: insertion already tried both
    # ways round, so its route is kept.
    if state["legs"] is None or len(stops) < 3:
        return
    _anneal(planner, rng, deadline, offer, state, build_seconds)


def _connect(planner, pairs, blocked):
    """Find a path for each (a, b) pair, avoiding the blocked cities and each other.

    First every pair takes its own shortest path. If those do not share a city, that is
    the best possible answer. If they collide, the pairs are routed one after another,
    each avoiding the ones before it, in forward and in reverse priority, and the
    shorter result wins. Returns a list of (km, cities, bitmask) or None.
    """
    detour = planner.detour
    alone = []
    seen = 0
    clash = False
    for a, b in pairs:
        hit = detour(a, b, blocked)
        if hit is None:
            return None
        inner = hit[2] & ~(1 << b)
        if inner & seen:
            clash = True
        seen |= inner
        alone.append(hit)
    if not clash:
        return alone
    best, best_km = None, INFINITY
    count = len(pairs)
    for priority in (range(count), range(count - 1, -1, -1)):
        taken, km, result = blocked, 0, [None] * count
        for k in priority:
            a, b = pairs[k]
            hit = detour(a, b, taken)
            if hit is None:
                km = INFINITY
                break
            result[k] = hit
            km += hit[0]
            taken |= hit[2] & ~(1 << b)
        if km < best_km:
            best, best_km = result, km
    return best


def _anneal(planner, rng, deadline, offer, state, build_seconds):
    """Simulated annealing on the real route.

    State        the required cities in route order (stops) and, between each pair of
                 consecutive stops, a leg: a path through non-required cities only.
                 Every state is a valid route, so anything it finds can be submitted.
    Moves        2-opt     reverse a run of stops; the two legs at its ends are replaced
                 Or-opt    move a run of 1 to 3 stops elsewhere, either way round; three
                           legs are replaced
                 re-route  replace one leg with the shortest path now available
                 New legs are shortest paths (A*) that avoid every other city on the route.
    Acceptance   always if the route gets shorter, otherwise with probability
                 exp(-extra_km / temperature). The temperature falls geometrically.
    Escaping     the hot start of each round accepts longer routes. The time is split
                 into ROUNDS short rounds: the first FRESH rounds each start from a new
                 random insertion build (random restarts), the others restart from the
                 best route found so far. On a slow machine, or with a short time
                 limit, fewer rounds are used so that a build never takes more than
                 about a sixth of its round.
    """
    now = time.monotonic
    random_number = rng.random
    start = planner.start
    detour = planner.detour
    exp = math.exp

    def load(legs):
        stops = [start] + [leg[-1] for leg in legs[:-1]]
        masks = []
        for leg in legs:
            m = 0
            for city in leg[:-1]:
                m |= 1 << city
            masks.append(m)
        kms = [planner.legs_km_between(a, leg) for a, leg in zip(stops, legs)]
        return stops, [list(leg) for leg in legs], masks, kms

    S, L, M, W = load(state["legs"])
    K = len(S)
    used = planner.required_mask
    for m in M:
        used |= m
    total = sum(W)
    best_total = total  # best of the whole run; total is the route of the current round
    pos = {stop: i for i, stop in enumerate(S)}
    near = planner.near

    anneal_start = now()
    # How many rounds fit: a fresh build (timed in _optimise) should use at most about a
    # sixth of a round. On the tournament limit with a normal machine this gives ROUNDS.
    rounds = ROUNDS
    if build_seconds > 0:
        rounds = max(4, min(ROUNDS, int((deadline - anneal_start) / (6 * build_seconds))))
    fresh = round(rounds * FRESH / ROUNDS)
    round_span = (deadline - anneal_start) / rounds
    hot, cold = HOT, COLD
    for round_number in range(rounds):
        if now() >= deadline:
            break
        if round_number:
            legs = planner.build_by_insertion(rng) if round_number < fresh else None
            S, L, M, W = load(legs or state["legs"])
            used = planner.required_mask
            for m in M:
                used |= m
            total = sum(W)
            pos = {stop: i for i, stop in enumerate(S)}
        # Each round has a fixed slot of the clock; a fresh build uses part of its slot.
        round_start = now()
        round_end = min(deadline, anneal_start + (round_number + 1) * round_span)
        cooling = max(round_end - round_start, 1e-6)
        temperature = hot
        step = 0
        while True:
            if step & 15 == 0:
                clock = now()
                if clock >= round_end:
                    break
                temperature = hot * (cold / hot) ** ((clock - round_start) / cooling)
            step += 1
            kind = random_number()  # 50% 2-opt, 40% Or-opt, 10% re-route
            a = S[int(random_number() * K)]
            c = near[a][int(random_number() * len(near[a]))]
            i, j = pos[a], pos[c]

            if kind < 0.5:
                # ---- 2-opt: reverse stops lo..hi --------------------------------------
                if i > j:
                    i, j = j, i
                if random_number() < 0.5:
                    lo, hi = i + 1, j
                else:
                    lo, hi = i, j - 1
                if lo < 1 or hi <= lo:
                    continue
                after = S[(hi + 1) % K]
                blocked = used & ~M[lo - 1] & ~M[hi]
                found = _connect(planner, ((S[lo - 1], S[hi]), (S[lo], after)), blocked)
                if found is None:
                    continue
                p, q = found
                delta = p[0] + q[0] - W[lo - 1] - W[hi]
                if delta > 0 and random_number() >= exp(-delta / temperature):
                    continue
                # Legs strictly inside the run are kept, travelled the other way.
                inner_L, inner_M, inner_W = [], [], []
                for t in range(hi - 1, lo - 1, -1):
                    leg = L[t][:-1]
                    leg.reverse()
                    leg.append(S[t])
                    inner_L.append(leg)
                    inner_M.append(M[t])
                    inner_W.append(W[t])
                S[lo:hi + 1] = S[lo:hi + 1][::-1]
                L[lo - 1:hi + 1] = [list(p[1])] + inner_L + [list(q[1])]
                M[lo - 1:hi + 1] = [p[2] & ~(1 << S[lo])] + inner_M + [q[2] & ~(1 << after)]
                W[lo - 1:hi + 1] = [p[0]] + inner_W + [q[0]]
                for t in range(lo, hi + 1):
                    pos[S[t]] = t
                used = blocked | M[lo - 1] | M[hi]
                total += delta

            elif kind < 0.9:
                # ---- Or-opt: move stops p..p+size-1 next to stop c -------------------
                size = 1 + int(random_number() * 3)
                p0 = i
                if p0 < 1 or p0 + size > K or K - size < 2:
                    continue
                q0 = j if random_number() < 0.5 else (j - 1) % K
                if p0 - 1 <= q0 <= p0 + size - 1:
                    continue
                first, last = S[p0], S[p0 + size - 1]
                flip = random_number() < 0.5
                if flip:
                    first, last = last, first
                before, after = S[p0 - 1], S[(p0 + size) % K]
                left, right = S[q0], S[(q0 + 1) % K]
                blocked = used & ~M[p0 - 1] & ~M[p0 + size - 1] & ~M[q0]
                found = _connect(planner, ((before, after), (left, first), (last, right)), blocked)
                if found is None:
                    continue
                joined, into, out = found
                delta = joined[0] + into[0] + out[0] - W[p0 - 1] - W[p0 + size - 1] - W[q0]
                if delta > 0 and random_number() >= exp(-delta / temperature):
                    continue
                run_S = S[p0:p0 + size]
                run_L, run_M, run_W = L[p0:p0 + size - 1], M[p0:p0 + size - 1], W[p0:p0 + size - 1]
                if flip:
                    flipped = []
                    for t in range(size - 2, -1, -1):
                        leg = run_L[t][:-1]
                        leg.reverse()
                        leg.append(run_S[t])
                        flipped.append(leg)
                    run_S.reverse()
                    run_L = flipped
                    run_M.reverse()
                    run_W.reverse()
                new_S = S[:p0] + S[p0 + size:]
                new_L = L[:p0 - 1] + [list(joined[1])] + L[p0 + size:]
                new_M = M[:p0 - 1] + [joined[2] & ~(1 << after)] + M[p0 + size:]
                new_W = W[:p0 - 1] + [joined[0]] + W[p0 + size:]
                at = q0 if q0 < p0 else q0 - size
                S = new_S[:at + 1] + run_S + new_S[at + 1:]
                L = new_L[:at] + [list(into[1])] + run_L + [list(out[1])] + new_L[at + 1:]
                M = new_M[:at] + [into[2] & ~(1 << first)] + run_M + [out[2] & ~(1 << right)] + new_M[at + 1:]
                W = new_W[:at] + [into[0]] + run_W + [out[0]] + new_W[at + 1:]
                pos = {stop: t for t, stop in enumerate(S)}
                used = planner.required_mask
                for m in M:
                    used |= m
                total += delta

            else:
                # ---- re-route one leg ------------------------------------------------
                b = S[(i + 1) % K]
                blocked = used & ~M[i]
                hit = detour(a, b, blocked)
                if hit is None or hit[0] >= W[i]:
                    continue
                total += hit[0] - W[i]
                L[i], M[i], W[i] = list(hit[1]), hit[2] & ~(1 << b), hit[0]
                used = blocked | M[i]

            if total < best_total:
                best_total = total
                offer([list(leg) for leg in L])
