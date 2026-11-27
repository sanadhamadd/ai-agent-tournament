"""Battleship agent (moving ships). Team: <your team name>

How it thinks, in one paragraph
-------------------------------
Every enemy ship we have not hit yet is a hidden variable: its (orientation, row, col).
For each one we keep a probability grid over those positions (a belief). Each turn we run
one step of an HMM forward filter:

    predict  the ship may have moved one cell        -> predict_ship_belief()
    update   a miss removes positions covering it     -> Bayes' rule
             a sonar reading reweights positions      -> sonar_posterior()

We do not know how the rival moves its ships, so we run the filter under several
hypotheses ("never moves", "moves sometimes", "moves every turn at random", "its ships are
drawn to the border", "its ships are drawn to the centre") and weigh them by how well each
one predicted our hits and misses (Bayesian model averaging).

A ship that has been hit can never move again, so the damaged ships form a
constraint-satisfaction problem, solved the way we did in class:

    variables    the hits we have not explained yet
    domains      the placements that could be the ship under each hit -> consistent_placements()
    constraints  same ship or no overlap, fleet counts, no miss made after the ship froze
    search       backtracking with MRV and forward checking        -> _backtrack()

Misses made *before* the first hit on a ship are not constraints (the ship may have moved
onto them), misses made *after* it are. Every solution is weighted by the belief, and we
shoot the cell that the solutions cover most.

Defence: the fleet layout is a second CSP (no two ships touch), solved by min-conflicts
local search from a random start, away from the centre (where probability hunters shoot
first). Then one unhit ship moves every turn so that the rival's old misses stop being
true; the agent learns from where our ships get found whether the centre or the border is
the safe zone, hides next to our own sunk ships (the rival has already searched there),
and keeps a simulated copy of the rival's shooter aimed at our own board so that ships
steer away from where it would fire next.

Course methods and where they are
---------------------------------
    CSP formulation, backtracking, MRV, forward checking   _hit_domains, _select_hit,
                                                           _forward_check, _backtrack
    Local search for CSPs (min-conflicts)                  _min_conflicts_layout
    Bayes' rule (prior, likelihood, posterior)             sonar_posterior, _process_shot
    Transition model and belief filtering                  predict_ship_belief, _predict_all
    Expected value of information                          _best_scan
"""
import random
import time

import numpy as np

# %% Settings (found by playing many games against several kinds of rival)

CFG = {
    # Hypotheses about the rival's movement: (P(it moves a ship on its turn), pull, prior weight).
    # pull > 0: its ships are drawn to the border; pull < 0: to the centre; 0: random walk.
    "models": ((0.0, 0.0, 0.3), (0.35, 0.0, 0.2), (1.0, 0.0, 0.2), (1.0, 5.0, 0.25), (1.0, -8.0, 0.05)),
    "weight_floor": 0.02,            # no hypothesis is ever ruled out completely
    "edge_prior": 0.5,               # share of the prior on fleets hugging the border
    "target_budget": 4000,           # most assignments the backtracking search may try
    "target_eps": 1e-3,              # target mode: weight left for "unlikely" placements
    "sonar": True,
    "sonar_below": 0.14,             # only think about scanning when the best shot is this poor
    "sonar_gain": 1.8,               # scan if it should multiply our best hit chance by this
    "sonar_candidates": 6,
    "placement": "mixed",            # "outer", "spaced", "mixed" or "judge"
    "outer_share": 0.5,              # "mixed": how often we choose the outer layout
    "placement_samples": 40,
    "move_mode": "smart",            # "none", "random" or "smart"
    "move_prob": 1.0,                # how often we move a ship
    "drift": 5.0,                    # >0: stay off the centre early, drift into it later
    "drift_turns": 40,
    "drift_flip": True,              # True: the centre becomes the safe place late in the game
    "adapt": True,                   # learn from where our ships are found which zone is safe
    "adapt_hi": 0.7, "adapt_lo": 0.5, "adapt_after": 25,
    "shadow": 2.0,                   # >0: avoid where a simulated rival shooter would aim next
    "shadow_rates": (0.0, 1.0),      # one simulated shooter that remembers misses, one that forgets
    "shadow_sharp": 3.0, "shadow_floor": 0.02, "shadow_trust": 5.0,
    "shelter": 4.0,                  # >0: hide next to our own sunk ships
    "panic_s": 0.35,                 # below this much clock: answer instantly
    "lite_s": 1.0,                   # below this much clock: skip the expensive extras
}

FLEET_LENGTHS = {"carrier": 5, "battleship": 4, "cruiser": 3, "submarine": 3, "destroyer": 2}
DIRECTIONS = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}

_S = None  # everything the agent remembers during one game


def act(observation):
    """Entry point. Any exception would lose the game, so there is always a safe answer."""
    try:
        if observation["phase"] == "placement":
            return {"placements": _choose_placement(observation)}
        return _turn(observation)
    except Exception as problem:  # never reached in testing; a legal answer beats a lost game
        print("battleship agent fallback:", repr(problem))
        if observation.get("phase") == "placement":
            return {"placements": None}
        return {"shot": _unshot_cell(observation)}


# %% Required algorithms. check_requirements() tests these, and the agent uses all three.


def consistent_placements(length, board_size, blocked, required):
    """CSP for one ship. Variable: its placement (row, col, orientation).
    Domain: every placement on the board. Constraints: stay on the board, cover no
    blocked cell, cover every required cell.

    Returns [[row, col, "H" or "V"], ...] with [row, col] the top-left cell.
    """
    blocked = {(int(r), int(c)) for r, c in blocked}
    required = {(int(r), int(c)) for r, c in required}
    if required:
        # Propagation: a placement must cover the first required cell, so only the
        # placements through that cell are worth generating.
        ar, ac = next(iter(required))
        candidates = [(ar, ac - i, "H") for i in range(length)] + [(ar - i, ac, "V") for i in range(length)]
    else:
        candidates = [(r, c, o) for o in "HV" for r in range(board_size) for c in range(board_size)]
    found = []
    for row, col, orientation in candidates:
        if orientation == "H":
            cells = [(row, col + i) for i in range(length)]
        else:
            cells = [(row + i, col) for i in range(length)]
        if not all(0 <= r < board_size and 0 <= c < board_size for r, c in cells):
            continue
        if blocked.intersection(cells):
            continue
        if not required.issubset(cells):
            continue
        found.append([row, col, orientation])
    return found


def _sonar_likelihood(reading, true_count, region_cells, correct_probability):
    """P(reading | true count) for the published sensor, renormalised at the limits."""
    error = (1.0 - correct_probability) / 2.0
    raw = {true_count: correct_probability, true_count - 1: error, true_count + 1: error}
    valid = {r: p for r, p in raw.items() if 0 <= r <= region_cells}
    total = sum(valid.values())
    if total <= 0:
        return 0.0
    return valid.get(reading, 0.0) / total


def sonar_posterior(prior, reading, region_cells, correct_probability):
    """Bayes' rule: P(true count | reading) is proportional to P(reading | count) * P(count)."""
    joint = {}
    for key, p in prior.items():
        joint[key] = float(p) * _sonar_likelihood(int(reading), int(key), int(region_cells), float(correct_probability))
    total = sum(joint.values())
    if total <= 0:
        return {key: float(p) for key, p in prior.items()}  # impossible reading: keep the prior
    return {key: value / total for key, value in joint.items()}


_FIT_CACHE = {}


def _fit_tables(size, length, orientation):
    """fits[r][c]: can a ship start at (r, c)?  options[r][c]: how many moves keep it on the board."""
    key = (size, length, orientation)
    if key not in _FIT_CACHE:
        fits = np.zeros((size, size))
        if length <= size:
            if orientation == "H":
                fits[:, : size - length + 1] = 1.0
            else:
                fits[: size - length + 1, :] = 1.0
        options = np.zeros((size, size))
        options[1:, :] += fits[:-1, :]   # north
        options[:-1, :] += fits[1:, :]   # south
        options[:, :-1] += fits[:, 1:]   # east
        options[:, 1:] += fits[:, :-1]   # west
        _FIT_CACHE[key] = (fits, options)
    return _FIT_CACHE[key]


def _predict_grid(belief, length, orientation, move_probability):
    """Transition model on a numpy grid: the prediction step of the filter."""
    size = belief.shape[0]
    fits, options = _fit_tables(size, length, orientation)
    can_move = options > 0
    share = np.where(can_move, belief * move_probability / np.maximum(options, 1.0), 0.0)
    new = np.where(can_move, belief * (1.0 - move_probability), belief)
    new[:-1, :] += share[1:, :] * fits[:-1, :]   # mass moving north
    new[1:, :] += share[:-1, :] * fits[1:, :]    # mass moving south
    new[:, 1:] += share[:, :-1] * fits[:, 1:]    # mass moving east
    new[:, :-1] += share[:, 1:] * fits[:, :-1]   # mass moving west
    return new


_PULL_CACHE = {}


def _predict_pulled(belief, length, orientation, move_probability, pull):
    """Transition model for a rival that prefers some zones: a move towards a cooler
    (pull > 0) or hotter (pull < 0) placement is exp(pull * change) times more likely."""
    size = belief.shape[0]
    key = (size, length, orientation, pull)
    if key not in _PULL_CACHE:
        fits, _ = _fit_tables(size, length, orientation)
        heat = _heat(size)
        level = np.zeros((size, size))            # heat under the whole ship, per top-left cell
        for i in range(length):
            if orientation == "H":
                level[:, : size - i] += heat[:, i:]
            else:
                level[: size - i, :] += heat[i:, :]
        w = {}
        w["N"] = np.zeros((size, size)); w["N"][1:, :] = fits[:-1, :] * np.exp(-pull * (level[:-1, :] - level[1:, :]))
        w["S"] = np.zeros((size, size)); w["S"][:-1, :] = fits[1:, :] * np.exp(-pull * (level[1:, :] - level[:-1, :]))
        w["E"] = np.zeros((size, size)); w["E"][:, :-1] = fits[:, 1:] * np.exp(-pull * (level[:, 1:] - level[:, :-1]))
        w["W"] = np.zeros((size, size)); w["W"][:, 1:] = fits[:, :-1] * np.exp(-pull * (level[:, :-1] - level[:, 1:]))
        total = w["N"] + w["S"] + w["E"] + w["W"]
        for d in w:
            w[d] = np.where(total > 0, w[d] / np.maximum(total, 1e-12), 0.0) * fits
        _PULL_CACHE[key] = (w, total > 0)
    w, can_move = _PULL_CACHE[key]
    moving = belief * move_probability
    new = np.where(can_move, belief - moving, belief)
    new[:-1, :] += (moving * w["N"])[1:, :]
    new[1:, :] += (moving * w["S"])[:-1, :]
    new[:, 1:] += (moving * w["E"])[:, :-1]
    new[:, :-1] += (moving * w["W"])[:, 1:]
    return new


def predict_ship_belief(belief, length, orientation, move_probability):
    """One prediction step: new_belief[s'] = sum over s of P(s' | s) * belief[s].

    P(s' | s): with move_probability the ship moves one cell, uniformly among the
    directions that keep it on the board; otherwise (or with no such direction) it stays.
    """
    grid = np.array(belief, dtype=float)
    return _predict_grid(grid, int(length), orientation, float(move_probability)).tolist()


# %% Belief state


def _valid_starts(size, length):
    """Boolean grids [H, V] of the cells where a ship of this length can start."""
    valid = np.zeros((2, size, size), dtype=bool)
    valid[0][:, : size - length + 1] = True
    valid[1][: size - length + 1, :] = True
    return valid


def _zero_covering(grid, length, row, col):
    """Removes every placement that covers (row, col). grid is [H, V]."""
    grid[0][row, max(0, col - length + 1): col + 1] = 0
    grid[1][max(0, row - length + 1): row + 1, col] = 0


def _occupancy(grid, length):
    """P(each cell is covered by this ship), from the belief over its top-left cell."""
    size = grid.shape[1]
    occ = np.zeros((size, size))
    for i in range(length):
        occ[:, i:] += grid[0][:, : size - i]
        occ[i:, :] += grid[1][: size - i, :]
    return occ


def _prior(size, length, edge_share=None):
    """Prior over one ship: partly uniform over placements, partly 'hugging the border'."""
    if edge_share is None:
        edge_share = CFG["edge_prior"]
    valid = _valid_starts(size, length).astype(float)
    uniform = valid / valid.sum()
    edge = np.zeros_like(valid)
    edge[0][0, :] = edge[0][size - 1, :] = 1.0   # horizontal on the top or bottom row
    edge[1][:, 0] = edge[1][:, size - 1] = 1.0   # vertical on the left or right column
    edge *= valid
    edge /= edge.sum()
    return (1.0 - edge_share) * uniform + edge_share * edge


def _new_state(observation):
    size = int(observation["board_size"])
    ships = sorted(FLEET_LENGTHS.values(), reverse=True)
    rates = [model[0] for model in CFG["models"]]
    pulls = [model[1] for model in CFG["models"]]
    prior_w = [model[2] for model in CFG["models"]]
    return {
        "size": size,
        "ships": ships,                                    # lengths of the ships not sunk yet
        "rates": rates,
        "pulls": pulls,
        "weights": np.array(prior_w, dtype=float),
        "prior_w": np.array(prior_w, dtype=float),
        "beliefs": [[_prior(size, length) for length in ships] for _ in rates],
        "allowed": {length: _valid_starts(size, length) for length in set(ships)},
        "hits": {},                                        # unexplained hits: cell -> {"t", "snap"}
        "wreck": set(),                                    # cells of ships we have sunk
        "misses": [],                                      # (turn, cell)
        "seen_shots": 0,
        "seen_sonar": 0,
        "last_mode": None,
        "own_turns": 0,
    }


def _renormalise(state, h, i):
    """Applies the 'cannot overlap a damaged ship' constraint and renormalises."""
    length = state["ships"][i]
    grid = state["beliefs"][h][i]
    grid *= state["allowed"][length]
    total = grid.sum()
    if total <= 1e-12:
        # This hypothesis was contradicted by the evidence: restart it from what is allowed.
        fresh = state["allowed"][length].astype(float)
        if fresh.sum() <= 0:
            fresh = _valid_starts(state["size"], length).astype(float)
        state["beliefs"][h][i] = fresh / fresh.sum()
        return False
    grid /= total
    return True


def _update_weights(state, likelihoods):
    weights = state["weights"] * np.maximum(np.array(likelihoods, dtype=float), 1e-9)
    total = weights.sum()
    if total <= 0:
        weights = state["prior_w"].copy()
        total = weights.sum()
    weights = np.maximum(weights / total, CFG["weight_floor"])
    state["weights"] = weights / weights.sum()


def _mixture(state):
    """Belief per ship averaged over the movement hypotheses."""
    return [
        sum(w * state["beliefs"][h][i] for h, w in enumerate(state["weights"]))
        for i in range(len(state["ships"]))
    ]


def _block_cell(state, cell):
    """A damaged segment stays where it is for ever: no other ship can cover that cell."""
    for length, mask in state["allowed"].items():
        _zero_covering(mask, length, cell[0], cell[1])


def _process_shot(state, entry):
    """Update step for one of our shots (the beliefs are still those of that moment)."""
    row, col = int(entry["target"][0]), int(entry["target"][1])
    cell = (row, col)
    result = entry["result"]
    if entry.get("already_damaged"):
        return
    hunting = state["last_mode"] == "hunt"

    # P(cell was occupied) under each hypothesis, before looking at the result.
    occupied = []
    for h in range(len(state["rates"])):
        free = 1.0
        for i, length in enumerate(state["ships"]):
            free *= 1.0 - min(1.0, float(_occupancy(state["beliefs"][h][i], length)[row, col]))
        occupied.append(1.0 - free)

    if result == "MISS":
        state["misses"].append((int(entry["turn"]), cell))
        for h in range(len(state["rates"])):
            for i, length in enumerate(state["ships"]):
                _zero_covering(state["beliefs"][h][i], length, row, col)
                _renormalise(state, h, i)
        if hunting:
            _update_weights(state, [1.0 - p for p in occupied])
        return

    # HIT or SUNK
    if hunting:
        _update_weights(state, occupied)
    snapshot = {}
    for i, grid in enumerate(_mixture(state)):
        length = state["ships"][i]
        snapshot[length] = snapshot.get(length, 0) + grid
    state["hits"][cell] = {"t": int(entry["turn"]), "snap": snapshot}
    _block_cell(state, cell)

    if result == "SUNK":
        length = int(entry.get("sunk_length", 0))
        explained = [cell]
        for r, c, o in consistent_placements(length, state["size"], [], [cell]):
            cells = [(r, c + k) if o == "H" else (r + k, c) for k in range(length)]
            if all(x in state["hits"] for x in cells):
                explained = cells
                break
        for x in explained:
            state["hits"].pop(x, None)
            state["wreck"].add(x)
        if length in state["ships"]:
            index = state["ships"].index(length)
            state["ships"].pop(index)
            for h in range(len(state["rates"])):
                state["beliefs"][h].pop(index)

    for h in range(len(state["rates"])):
        for i in range(len(state["ships"])):
            _renormalise(state, h, i)


def _predict_all(state):
    """Prediction step for every unseen ship, under every movement hypothesis."""
    frozen = 1 if state["hits"] else 0
    mobile = max(1, len(state["ships"]) - frozen)
    for h, rate in enumerate(state["rates"]):
        if rate <= 0:
            continue
        per_ship = min(1.0, rate / mobile)   # the rival moves at most one ship per turn
        for i, length in enumerate(state["ships"]):
            grid = state["beliefs"][h][i]
            pull = state["pulls"][h]
            if pull:
                grid[0] = _predict_pulled(grid[0], length, "H", per_ship, pull)
                grid[1] = _predict_pulled(grid[1], length, "V", per_ship, pull)
            else:
                grid[0] = _predict_grid(grid[0], length, "H", per_ship)
                grid[1] = _predict_grid(grid[1], length, "V", per_ship)
            _renormalise(state, h, i)


# %% Sonar: Bayes update of the beliefs, and the value of a scan


def _region_mask(size, center):
    mask = np.zeros((size, size), dtype=int)
    r, c = center
    mask[max(0, r - 1): r + 2, max(0, c - 1): c + 2] = 1
    return mask


def _count_grid(mask, length):
    """For every placement, how many of its cells fall inside the scanned region."""
    size = mask.shape[0]
    counts = np.zeros((2, size, size), dtype=int)
    for j in range(length):
        counts[0][:, : size - j] += mask[:, j:]
        counts[1][: size - j, :] += mask[j:, :]
    return counts


def _sonar_update(beliefs, lengths, mask, known, reading, correct):
    """Returns (new beliefs, P(reading)) for one scan.

    The reading depends on the total count = known damaged cells + what each ship adds.
    For one ship at one placement the rest of the fleet is summarised by the distribution
    of its count, and P(count | reading) comes from sonar_posterior().
    """
    region_cells = int(mask.sum())
    counts = [_count_grid(mask, length) for length in lengths]
    dists = [np.bincount(k.ravel(), weights=b.ravel(), minlength=4)[:4] for k, b in zip(counts, beliefs)]

    total = np.array([1.0])
    for d in dists:
        total = np.convolve(total, d)
    prior = {}
    for extra, p in enumerate(total):
        if p > 1e-12:
            n = min(region_cells, known + extra)
            prior[str(n)] = prior.get(str(n), 0.0) + float(p)
    if not prior:
        return beliefs, 0.0
    p_reading = sum(p * _sonar_likelihood(reading, int(n), region_cells, correct) for n, p in prior.items())
    if p_reading <= 1e-12:
        return beliefs, 0.0
    posterior = sonar_posterior(prior, reading, region_cells, correct)
    ratio = {int(n): posterior[n] / prior[n] for n in prior}   # proportional to P(reading | n)

    updated = []
    for i, belief in enumerate(beliefs):
        others = np.array([1.0])
        for j, d in enumerate(dists):
            if j != i:
                others = np.convolve(others, d)
        likelihood = np.zeros(4)
        for k in range(4):
            likelihood[k] = sum(
                p * ratio.get(min(region_cells, known + k + m), 0.0) for m, p in enumerate(others) if p > 1e-12
            )
        new = belief * likelihood[counts[i]]
        s = new.sum()
        updated.append(new / s if s > 1e-12 else belief)
    return updated, p_reading


def _known_in_region(state, mask):
    return sum(int(mask[r, c]) for r, c in list(state["wreck"]) + list(state["hits"]))


def _process_sonar(state, entry, correct):
    center = (int(entry["center"][0]), int(entry["center"][1]))
    mask = _region_mask(state["size"], center)
    known = _known_in_region(state, mask)
    for h in range(len(state["rates"])):
        new, _ = _sonar_update(state["beliefs"][h], state["ships"], mask, known, int(entry["reading"]), correct)
        state["beliefs"][h] = [np.array(b) for b in new]
        for i in range(len(state["ships"])):
            _renormalise(state, h, i)


def _best_scan(state, occupancy, correct):
    """Value of information: the scan that most raises our best hit chance, if it is worth a shot."""
    size = state["size"]
    best_now = float(occupancy.max())
    if best_now <= 0 or best_now >= CFG["sonar_below"]:
        return None
    beliefs = _mixture(state)
    lengths = state["ships"]
    # Candidate centres: the 3 x 3 squares holding the most probability mass.
    padded = np.pad(occupancy, 1)
    mass = sum(padded[dr: dr + size, dc: dc + size] for dr in range(3) for dc in range(3))
    order = np.argsort(-mass, axis=None)[: CFG["sonar_candidates"]]
    best_center, best_value = None, 0.0
    for flat in order:
        center = (int(flat) // size, int(flat) % size)
        mask = _region_mask(size, center)
        known = _known_in_region(state, mask)
        expected = 0.0
        for reading in range(int(mask.sum()) + 1):
            new, p = _sonar_update(beliefs, lengths, mask, known, reading, correct)
            if p <= 1e-4:
                continue
            after = sum(_occupancy(b, length) for b, length in zip(new, lengths))
            expected += p * float(after.max())
        if expected > best_value:
            best_center, best_value = center, expected
    if best_center is not None and best_value >= CFG["sonar_gain"] * best_now:
        return best_center
    return None


# %% Choosing where to shoot


def _hit_domains(state, use_misses=True):
    """The damaged ships as a CSP (same shape as the class CSPs).

    Variables   : the hits we have not explained yet (one per damaged cell).
    Domain      : the placements (length, row, col, orientation) that could be the ship
                  under that hit, from consistent_placements().
    Constraints : two hits get the same placement if it covers both, otherwise placements
                  that do not overlap; no more ships of a length than are still afloat;
                  a placement never covers a miss made after its ship was first hit
                  (a hit ship cannot move); a placement covered entirely by hits is
                  impossible, because it would have been reported as SUNK.
    Returns (domains, weight of each value, cells of each value).
    """
    size = state["size"]
    hits = state["hits"]
    domains, weight, cells_of = {}, {}, {}
    for hit, info in hits.items():
        values = []
        for length in sorted(set(state["ships"])):
            for row, col, orientation in consistent_placements(length, size, state["wreck"], [hit]):
                value = (length, row, col, orientation)
                cells = _ship_cells(row, col, orientation, length)
                covered = [cell for cell in cells if cell in hits]
                if len(covered) == length:
                    continue
                first = min(covered, key=lambda cell: hits[cell]["t"])
                frozen_at = hits[first]["t"]
                if use_misses and any(t > frozen_at and miss in cells for t, miss in state["misses"]):
                    continue
                values.append(value)
                cells_of[value] = cells
                grid = hits[first]["snap"].get(length)
                prior = float(grid[0 if orientation == "H" else 1][row, col]) if grid is not None else 0.0
                weight[value] = prior + CFG["target_eps"]
        domains[hit] = values
    return domains, weight, cells_of


def _select_hit(domains, assignment):
    """MRV: the unassigned hit with the fewest remaining values (ties: smallest cell)."""
    unassigned = [hit for hit in domains if hit not in assignment]
    return min(unassigned, key=lambda hit: (len(domains[hit]), hit))


def _forward_check(value, domains, assignment, afloat, cells_of):
    """Filters the domains of the unassigned hits after choosing `value`.

    Returns the new domains, or None if some hit is left with no value (wipeout).
    """
    cells = set(cells_of[value])
    length = value[0]
    new = {}
    for hit, values in domains.items():
        if hit in assignment:
            continue
        if hit in cells:                       # same ship: this hit must take the same value
            kept = [value] if value in values else []
        else:                                  # another ship: no overlap, and one must be left
            kept = [v for v in values
                    if not cells.intersection(cells_of[v]) and (v[0] != length or afloat[length] > 1)]
        if not kept:
            return None
        new[hit] = kept
    return new


def _backtrack(domains, assignment, afloat, product, cells_of, weight, solutions, stats):
    """Backtracking search that collects every solution with its weight."""
    if stats["assignments"] >= CFG["target_budget"]:
        return
    if len(assignment) == stats["variables"]:
        solutions.append((product, set(assignment.values())))
        return
    hit = _select_hit(domains, assignment)
    for value in domains[hit]:
        stats["assignments"] += 1
        new = _forward_check(value, domains, {**assignment, hit: value}, afloat, cells_of)
        if new is None:
            stats["backtracks"] += 1
            continue
        extended = {**assignment, hit: value}
        for other in cells_of[value]:           # hits under the same ship are settled too
            if other in domains:
                extended[other] = value
        left = dict(afloat)
        left[value[0]] -= 1
        _backtrack(new, extended, left, product * weight[value], cells_of, weight, solutions, stats)


def _target_scores(state, use_misses=True):
    """Target mode: P(cell holds a damaged ship) from all the solutions of the CSP."""
    size = state["size"]
    scores = np.zeros((size, size))
    domains, weight, cells_of = _hit_domains(state, use_misses)
    if any(not values for values in domains.values()):
        return scores
    afloat = {length: state["ships"].count(length) for length in set(state["ships"])}
    solutions = []
    stats = {"assignments": 0, "backtracks": 0, "variables": len(domains)}
    _backtrack(domains, {}, afloat, 1.0, cells_of, weight, solutions, stats)
    state["csp_stats"] = stats
    for product, placements in solutions:
        for value in placements:
            for cell in cells_of[value]:
                if cell not in state["hits"]:
                    scores[cell] += product
    return scores


def _hunt_occupancy(state):
    """Hunt mode: expected number of ship segments in each cell, over all hypotheses."""
    size = state["size"]
    occupancy = np.zeros((size, size))
    for h, w in enumerate(state["weights"]):
        for i, length in enumerate(state["ships"]):
            occupancy += w * _occupancy(state["beliefs"][h][i], length)
    for r, c in list(state["wreck"]) + list(state["hits"]):
        occupancy[r, c] = 0.0
    return occupancy


def _argmax_random(scores):
    """Best cell, breaking ties at random so we are not predictable."""
    size = scores.shape[0]
    flat = scores.ravel()
    best = np.flatnonzero(flat >= flat.max() - 1e-12)
    index = int(best[random.randrange(len(best))])
    return [index // size, index % size]


def _choose_action(state, observation, lite):
    if state["hits"]:
        scores = _target_scores(state, use_misses=True)
        if scores.max() <= 0:
            scores = _target_scores(state, use_misses=False)
        if scores.max() > 0:
            state["last_mode"] = "target"
            return {"shot": _argmax_random(scores)}
        # Nothing fits these hits (should not happen): stop chasing them.
        state["wreck"].update(state["hits"])
        state["hits"].clear()

    state["last_mode"] = "hunt"
    occupancy = _hunt_occupancy(state)
    if occupancy.max() <= 0:
        size = state["size"]
        free = [(r, c) for r in range(size) for c in range(size) if (r, c) not in state["wreck"]]
        return {"shot": list(random.choice(free))}
    if CFG["sonar"] and not lite and int(observation.get("sonar_remaining", 0)) > 0 and state["ships"]:
        center = _best_scan(state, occupancy, float(observation.get("sonar_correct_probability", 0.8)))
        if center is not None:
            state["last_mode"] = "sonar"
            return {"sonar": [int(center[0]), int(center[1])]}
    return {"shot": _argmax_random(occupancy)}


# %% Defence: where to put the fleet, and how to move it


def _heat(size):
    """How attractive each cell is to a probability hunter at the start (centre = hot)."""
    heat = np.zeros((size, size))
    for length in FLEET_LENGTHS.values():
        valid = _valid_starts(size, length).astype(float)
        heat += _occupancy(valid, length)
    return heat / heat.max()


def _ship_cells(row, col, orientation, length):
    return [(row, col + i) if orientation == "H" else (row + i, col) for i in range(length)]


_LAYOUT_CACHE = {}


def _layout_values(size, length):
    """Domain of one of our ships: every placement, with its cells and its halo
    (the cells and everything touching them, even diagonally)."""
    key = (size, length)
    if key not in _LAYOUT_CACHE:
        values = []
        for row, col, orientation in consistent_placements(length, size, [], []):
            cells = frozenset(_ship_cells(row, col, orientation, length))
            halo = frozenset((r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1))
            values.append((row, col, orientation, cells, halo))
        _LAYOUT_CACHE[key] = values
    return _LAYOUT_CACHE[key]


def _layout_conflicts(ship, value, current, spaced):
    """Number of other ships this placement overlaps (or touches, when spaced)."""
    zone = value[4] if spaced else value[3]
    return sum(1 for other, placed in current.items() if other != ship and zone.intersection(placed[3]))


def _min_conflicts_layout(size, fleet, spaced, max_steps=200):
    """Fleet placement as a CSP solved by local search (min-conflicts).

    Variables: our ships. Domain: every placement on the board. Constraint: no two ships
    overlap (spaced=True: or touch). Start from a random complete assignment, then repair:
    pick a conflicted ship at random and give it a value with the fewest conflicts.
    """
    domains = {ship: _layout_values(size, length) for ship, length in fleet}
    current = {ship: random.choice(domains[ship]) for ship, _ in fleet}
    for _ in range(max_steps):
        conflicted = [ship for ship in current if _layout_conflicts(ship, current[ship], current, spaced) > 0]
        if not conflicted:
            return [{"ship": ship, "row": current[ship][0], "col": current[ship][1], "orientation": current[ship][2]}
                    for ship, _ in fleet]
        ship = random.choice(conflicted)
        counts = [_layout_conflicts(ship, value, current, spaced) for value in domains[ship]]
        lowest = min(counts)
        current[ship] = random.choice([value for value, count in zip(domains[ship], counts) if count == lowest])
    return None


def _choose_placement(observation):
    global _S
    _S = None
    mode = CFG["placement"]
    if mode == "judge":
        return None
    size = int(observation["board_size"])
    fleet = [(item["ship"], int(item["length"])) for item in observation["fleet"]]
    if mode == "mixed":   # unpredictable on purpose: a rival cannot count on finding us at the border
        mode = "outer" if random.random() < CFG["outer_share"] else "spaced"
    if mode == "spaced":
        return _min_conflicts_layout(size, fleet, spaced=True)
    # "outer": among many random spaced layouts, prefer the ones a hunter reaches last.
    heat = _heat(size)
    layouts = []
    for _ in range(CFG["placement_samples"]):
        layout = _min_conflicts_layout(size, fleet, spaced=True)
        if layout is None:
            continue
        cost = 0.0
        for item, (_, length) in zip(layout, fleet):
            cost += sum(heat[r, c] for r, c in _ship_cells(item["row"], item["col"], item["orientation"], length))
        layouts.append((cost, layout))
    if not layouts:
        return None
    layouts.sort(key=lambda pair: pair[0])
    return random.choice(layouts[: max(1, len(layouts) // 8)])[1]   # random among the best eighth


def _shadow(state, observation):
    """Models of the rival's shooter, aimed at our own board.

    We never see the rival's misses. But on every turn in which none of our ships was hit,
    a probability shooter most likely fired at its best cell that holds none of our ships,
    so we imagine that shot and update the rival's (simulated) belief with a miss there.
    We keep one such model per guess of how fast the rival forgets its misses, and weigh
    the models by how well they explain when we were hit and when we were not.
    Returns P(the rival aims at each cell), mixed over the models.
    """
    size = state["size"]
    ships = observation["my_ships"]
    lengths = {ship["ship"]: len(ship["cells"]) for ship in ships}
    shadow = state.get("shadow")
    if shadow is None:
        shadow = state["shadow"] = {
            "models": [
                {"rate": rate, "beliefs": {ship["ship"]: _prior(size, len(ship["cells"]), 0.0) for ship in ships}}
                for rate in CFG["shadow_rates"]
            ],
            "logw": np.zeros(len(CFG["shadow_rates"])),
            "found": set(),
        }
    damaged = [tuple(cell) for ship in ships for cell in ship["damaged_cells"]]
    first_hits = [tuple(ship["damaged_cells"][0]) for ship in ships
                  if ship["damaged_cells"] and ship["ship"] not in shadow["found"]]
    was_finishing = any(ship["damaged_cells"] and not ship["sunk"] and ship["ship"] in shadow["found"] for ship in ships)
    shadow["found"].update(ship["ship"] for ship in ships if ship["damaged_cells"])
    ours = [tuple(cell) for ship in ships if not ship["damaged_cells"] for cell in ship["cells"]]
    aimed = np.zeros((size, size))
    views = []
    for k, model in enumerate(shadow["models"]):
        beliefs = model["beliefs"]

        def occupancy():
            if not beliefs:
                return np.zeros((size, size))
            return sum(_occupancy(grid, lengths[name]) for name, grid in beliefs.items())

        def rule_out(row, col):
            for name, grid in beliefs.items():
                _zero_covering(grid, lengths[name], row, col)
                total = grid.sum()
                if total > 1e-12:
                    grid /= total
                else:
                    beliefs[name] = _prior(size, lengths[name], 0.0)

        # 1. How well did this model predict what just happened?
        before = occupancy() ** CFG["shadow_sharp"]
        if before.sum() > 0 and state["own_turns"] > 1:
            before = before / before.sum()
            if first_hits:
                likelihood = sum(before[cell] for cell in first_hits)
            elif not was_finishing and len(damaged) == shadow.get("damaged", 0):
                likelihood = 1.0 - sum(before[cell] for cell in ours)
            else:
                likelihood = None
            if likelihood is not None:
                shadow["logw"][k] += np.log(max(float(likelihood), CFG["shadow_floor"]))
        # 2. Update the simulated shooter.
        for ship in ships:                  # a hit ship is no longer hunted
            if ship["damaged_cells"]:
                beliefs.pop(ship["ship"], None)
        for row, col in damaged:
            rule_out(row, col)
        if beliefs and state["own_turns"] > 1 and not first_hits and not was_finishing:
            guess = occupancy()
            for ship in ships:
                for r, c in ship["cells"]:
                    guess[r, c] = -1.0      # it missed, so it did not fire at one of our ships
            row, col = _argmax_random(guess)
            rule_out(row, col)
        if model["rate"] > 0 and beliefs:
            per_ship = min(1.0, model["rate"] / len(beliefs))
            for name, grid in beliefs.items():
                grid[0] = _predict_grid(grid[0], lengths[name], "H", per_ship)
                grid[1] = _predict_grid(grid[1], lengths[name], "V", per_ship)
        views.append(occupancy())
    shadow["damaged"] = len(damaged)
    weights = np.exp(CFG["shadow_trust"] * (shadow["logw"] - shadow["logw"].max()))
    weights /= weights.sum()
    for w, view in zip(weights, views):
        if view.max() > 0:
            aimed += w * view / view.max()
    return aimed


def _choose_move(state, observation):
    """Moves one unhit ship, away from where the rival is probably about to shoot."""
    moves = observation.get("legal_moves") or []
    mode = CFG["move_mode"]
    aimed = _shadow(state, observation) if CFG["shadow"] > 0 else None
    if not moves or mode == "none" or random.random() > CFG["move_prob"]:
        return None
    if mode == "random":
        return random.choice(moves)

    size = state["size"]
    danger = np.zeros((size, size))
    if aimed is not None and aimed.max() > 0:
        danger += CFG["shadow"] * aimed
    # After a hit, the rival shoots next to it and along the same line.
    for ship in observation["my_ships"]:
        if ship["sunk"] or not ship["damaged_cells"]:
            continue
        for r, c in ship["damaged_cells"]:
            for dr, dc in DIRECTIONS.values():
                for step, value in ((1, 3.0), (2, 1.0)):
                    rr, cc = r + dr * step, c + dc * step
                    if 0 <= rr < size and 0 <= cc < size:
                        danger[rr, cc] += value
    heat = state.setdefault("heat", _heat(size))
    if CFG["adapt"]:
        found = state.setdefault("found", set())
        for ship in observation["my_ships"]:
            if ship["damaged_cells"] and ship["ship"] not in found:
                found.add(ship["ship"])
                level = sum(heat[r, c] for r, c in ship["cells"]) / len(ship["cells"])
                if level >= CFG["adapt_hi"] and state["own_turns"] > CFG["adapt_after"]:
                    state["centre_hot"] = state.get("centre_hot", 0) + 1
                elif level <= CFG["adapt_lo"]:
                    state["centre_hot"] = state.get("centre_hot", 0) - 1
    if CFG["shelter"] > 0:
        for ship in observation["my_ships"]:
            if ship["sunk"]:
                for r, c in ship["cells"]:
                    for dr, dc in DIRECTIONS.values():
                        if 0 <= r + dr < size and 0 <= c + dc < size:
                            danger[r + dr, c + dc] -= CFG["shelter"]
    if CFG["drift"] > 0:
        # Early the centre is where hunters look; later it is where they have already looked.
        progress = min(1.0, state["own_turns"] / float(CFG["drift_turns"])) if CFG["drift_flip"] else 0.0
        sign = 1.0 - 2.0 * progress
        evidence = state.get("centre_hot", 0)
        if evidence > 0:
            sign = 1.0
        elif evidence < 0:
            sign = -1.0
        danger += CFG["drift"] * sign * heat

    ships = {ship["ship"]: ship for ship in observation["my_ships"]}
    best, best_score = None, None
    for move in moves:
        ship = ships[move["ship"]]
        dr, dc = DIRECTIONS[move["direction"]]
        before = sum(danger[r, c] for r, c in ship["cells"])
        after = sum(danger[r + dr, c + dc] for r, c in ship["cells"])
        score = (after - before) + random.random() * 0.5
        if best_score is None or score < best_score:
            best, best_score = move, score
    return {"ship": best["ship"], "direction": best["direction"]}


# %% One turn


def _unshot_cell(observation):
    """Emergency shot (low clock or an unexpected error): a random cell we have not
    fired at yet, next to an unsunk hit if there is one, so no shot is wasted."""
    size = int(observation.get("board_size", 10))
    shots = observation.get("my_shots") or []
    fired = {tuple(s["target"]) for s in shots}
    sunk = {tuple(s["target"]) for s in shots if s.get("result") == "SUNK"}
    hits = [tuple(s["target"]) for s in shots if s.get("result") == "HIT" and tuple(s["target"]) not in sunk]
    near = [(r + dr, c + dc) for r, c in hits for dr, dc in DIRECTIONS.values()
            if 0 <= r + dr < size and 0 <= c + dc < size and (r + dr, c + dc) not in fired]
    free = near or [(r, c) for r in range(size) for c in range(size) if (r, c) not in fired]
    return list(random.choice(free)) if free else [random.randrange(size), random.randrange(size)]



def _turn(observation):
    global _S
    started = time.perf_counter()
    remaining = observation.get("time_remaining_s")
    size = int(observation["board_size"])

    if remaining is not None and remaining < CFG["panic_s"]:
        # Almost out of clock: any legal shot is better than losing on time.
        return {"shot": _unshot_cell(observation)}

    if _S is None or len(observation["my_shots"]) < _S["seen_shots"]:
        _S = _new_state(observation)
    state = _S
    lite = remaining is not None and remaining < CFG["lite_s"]

    # 1. Update with the result of our previous action.
    for entry in observation["my_shots"][state["seen_shots"]:]:
        _process_shot(state, entry)
    state["seen_shots"] = len(observation["my_shots"])
    for entry in observation["my_sonar"][state["seen_sonar"]:]:
        _process_sonar(state, entry, float(observation.get("sonar_correct_probability", 0.8)))
    state["seen_sonar"] = len(observation["my_sonar"])

    # 2. Predict: the rival has had one turn to move a ship since then.
    if state["own_turns"] > 0:
        _predict_all(state)
    state["own_turns"] += 1

    # 3. Decide.
    action = _choose_action(state, observation, lite)
    move = _choose_move(state, observation)
    if move is not None:
        action["move"] = move
    state["spent"] = state.get("spent", 0.0) + time.perf_counter() - started
    return action
