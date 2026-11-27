# Team 10: technical report

> This is the README submitted with the project. The only change in this public copy is the members list, which shows first names and who built which agent.

## Members
- Raji (Chess)
- Lorenzo (Catan)
- Serge (Battleship)
- Sanad Hamad (Road Trip)

## Algorithms implemented
| Challenge | Algorithms | Where in the file |
|---|---|---|
| Road Trip | Uniform-cost search (Dijkstra), A* with the Haversine heuristic, hill climbing with 2-opt and Or-opt, cheapest-insertion construction, simulated annealing with random restarts | `ucs`, `astar`, `_search`, `local_search`, `build_by_insertion`, `_anneal`, `_connect`, `detour` |
| Chess | Linear evaluation with depth cutoff, minimax and alpha-beta, iterative deepening, move ordering, transposition table, killer moves, quiescence search | `evaluate`, `minimax`, `alphabeta`, `_search_root`, `_ab`, `_quiesce`, `_order_moves` |
| Battleship | CSP (backtracking with MRV and forward checking; min-conflicts), Bayesian belief filter (prediction + Bayes update), sonar as value of information | `consistent_placements`, `_backtrack`, `_min_conflicts_layout`, `predict_ship_belief`, `sonar_posterior`, `_best_scan` |
| Catan | Classical planning over builds and trades, afterstate evaluation, logical inference of the rival's hand, Q-learning over an abstract MDP, baseline heuristic policy | `Planner`, `Evaluator`, `OpponentTracker`, `state_features`, `q_update`, `heuristic_policy` |

## Training
- **Road Trip:** no learning. The five annealing settings (`NEAR`, `HOT`, `COLD`, `ROUNDS`, `FRESH`) were chosen by measuring on 200 to 400 random draws.
- **Chess:** no learning. Weights are hand-set; features and search extensions were chosen by head-to-head matches (20 games each) against the previous version.
- **Battleship:** no learning. The settings in `CFG` were chosen by playing 300 to 1000 games per rival with the judge's rules.
- **Catan:** Q-learning over the abstract MDP, 4000 training games (half self-play, half against `demo_A`); entries seen fewer than 5 times dropped, 1553 kept. Evaluator weights tuned by hill climbing (60 steps, 480 paired games per step). The Q-table and weights are stored inside `catan_agent.py`.

## Tests
Official judge (`ai_agent_tournament.zip`, real clocks), games per rival as listed; no incidents in any test (no illegal move, wrong format, crash or timeout).

- **Chess:** 10/10 challenges (2 games each) against `demo` and against `demo_b`; 5/5 in the single 10 s game format; 10/10 with the clock halved to 2.5 s. Against Stockfish 19 limited to its calibrated Elo levels (testing only): about 1430 Elo (95% interval 1320–1540).
- **Battleship:** 9/10 against `demo`, 20/20 against `demo_a`, 10/10 against `demo_b`, 10/10 with a 2.5 s clock.
- **Catan:** 19/20 against `demo`, 17/20 against `demo_a`, 20/20 against `demo_b`, 10/10 with a 2.5 s clock.
- **Road Trip:** 50/50 against `demo` (seeds 1 to 50, 12,943 km against 16,050 km on average), 20/20 against `demo` in Google Colab. In an in-process benchmark that uses the judge's route checker: a valid route on 400/400 random draws, and optimal in 83 to 84 of 84 runs on the 28 draws with a known exact optimum.

`check_requirements()` output:
```
PASS  Chess       Heuristic evaluation function           ranks material correctly from White's point of view
PASS  Chess       Minimax with alpha-beta pruning         same value as minimax with fewer nodes on 4 positions
PASS  Chess       Depth-limited search                    depth 1 visits 32 nodes, depth 2 visits 128
PASS  Battleship  Constraints over hidden ship positions  all placements found on 4 constraint sets
PASS  Battleship  Bayesian update from noisy sonar        Bayes update exact on 3 cases
PASS  Battleship  Transition model / belief prediction    belief correctly propagated through the transition model
PASS  Catan       State abstraction                       6 distinct abstract states from 6 observations
PASS  Catan       Value estimate for actions              a finite value for every legal action tested
PASS  Catan       Q-learning update                       matches the Q-learning update rule
PASS  Catan       Baseline heuristic policy               always returns a legal action
PASS  Road Trip   Uniform-cost search / Dijkstra          optimal on 8 city pairs, 392 nodes expanded
PASS  Road Trip   A* with a geographic heuristic          optimal on 8 city pairs, 184 nodes expanded
PASS  Road Trip   A* expands no more nodes than UCS       A* 184 vs UCS 392
PASS  Road Trip   Local search / optimization             tour improved from 16810 to 7480
14/14 requirements pass.
```

## Dependencies and known limitations
- **Libraries:** `chess` (python-chess, chess agent only, for the board and rules); `numpy` (battleship agent). Catan and Road Trip use only the standard library. Stockfish was used only offline to measure the chess agent's rating and is never imported. SciPy was used only offline to compute exact Road Trip optima for testing and is never imported by the agent.
- **Known limitations:** Road Trip is a heuristic that cannot prove its route is optimal and loses some quality on a slower machine; chess reaches only a few plies in Python and its quiescence search does not look at checks; Battleship filters ships independently and its movement hypotheses are fixed in advance; Catan plans one turn ahead and does not block the rival on purpose. Details are in each section below.

## Technical explanation

### Road Trip

#### Problem formulation
- **Environment:** one agent, fully observable and deterministic. An undirected weighted graph with 105 cities and 262 roads (weight = km). Each match gives 25 required cities, including the start, Madrid.
- **What the agent observes:** everything, once: the problem dictionary with the start, the 25 required cities, the time limit, all cities with latitude and longitude, and all roads with their km.
- **State:** a closed route: Madrid, a sequence of different cities, Madrid. It is stored as `stops` (the required cities in route order) plus one `leg` per consecutive pair (the non-required cities between them). A set of cities is a bitmask (one Python integer), so "does this path touch the route?" is a single AND.
- **Actions:** in path search, drive one road; in construction, insert one required city; in optimisation, one move on the route (2-opt, Or-opt, re-route a leg).
- **Goal and cost:** every required city is on the route, no city appears twice (Madrid only at both ends) and every step follows a road; the cost to minimise is the total km.
- **Why it is hard:** it is a travelling-salesman problem with an extra rule. A good route uses about 47 of the 105 cities, so it acts as a wall: a path used early can cut off cities needed later, and the shortest path between two required cities is often not usable.

#### Algorithms
| Algorithm | Course topic | Where in `roadtrip_agent.py` |
|---|---|---|
| Best-first search on f = g + h with a closed set; with h = 0 it is uniform-cost search | UCS (Dijkstra) | `_search`, `ucs`, `_tree` |
| The same search with h = Haversine distance to the goal | A*, admissible and consistent heuristics | `astar`, `straight_line_km` |
| A* with a set of blocked cities: the shortest path that avoids the rest of the route | A* | `detour` |
| Hill climbing on a tour with 2-opt and Or-opt, and random-restart kicks | Local search | `local_search` |
| First route by cheapest insertion on the real road graph | Greedy construction | `build_by_insertion`, `insert` |
| Simulated annealing on the real route: 2-opt, Or-opt of 1 to 3 stops, re-route; up to 24 rounds with random restarts | Simulated annealing, random restarts | `_anneal`, `_connect` |

**UCS and A\*.** UCS expands cities in order of distance from the start, so the first time a city leaves the frontier its distance is final, and the search stops when the goal leaves it. A* orders the frontier by g + h. The Haversine distance is admissible because every road is at least as long as the straight line between its ends (the tightest road has 3.8 km of slack). It is also consistent, so no city is expanded twice and the path is still optimal. On the judge's 8 pairs A* expands 184 cities and UCS 392.

**No city twice.** Every new leg is an A* path with all cities already on the route blocked. After each insertion the route is cut again at every required city, so a required city that a leg passes through becomes a stop and cannot be lost. Every state of the search is therefore a valid loop, and `is_valid` (a copy of the judge's rules) checks each route before it is submitted.

**Optimisation.** *Neighbourhood:* 2-opt reverses a run of stops (two legs are rebuilt), Or-opt moves 1 to 3 stops next to a nearby stop (three legs are rebuilt), and re-route replaces one leg by the best path now free. *Acceptance rule:* a shorter route is always accepted, a longer one with probability exp(−extra km / T), where T falls from 500 km to 3 km in each round. *Escaping local optima:* the hot start of each round, and restarts: the first 16 of the 24 rounds start from a new random insertion build, the rest from the best route so far. `local_search` is the standalone optimiser the course requires. The agent does not call it, because a distance table cannot express "no city twice"; it applies the same two neighbourhoods inside `_anneal` instead (using the `local_search` tour as the insertion order made the first route worse: 14,268 km against 13,730 km).

#### Baseline against improved version
Official judge, `run_game`, seeds 1 to 50, 5 s clock; the final agent against each older version:

| Opponent | What it does | Final wins / draws / losses | Mean km, final vs older |
|---|---|---|---|
| Bundled demo | random insertion with fewest-roads paths | 50 / 0 / 0 | 12,943 vs 16,050 |
| Final code without the improvement step | cheapest insertion with A* legs only | 46 / 4 / 0 | 12,943 vs 13,679 |
| First complete version | annealing over visiting orders with greedy legs | 5 / 45 / 0 | 12,943 vs 12,958 |

The first complete version searched over visiting orders, but most orders have no valid greedy route; searching over real routes removed that problem. Schedule: 24 short rounds with 16 fresh starts missed the best known route on 0 of 200 draws, where 4 long rounds missed it on 7. Tried and rejected: removing and re-inserting a cluster of cities, a 2-opt that also re-routes the neighbouring legs, and Or-opt runs of up to 6 stops.

Other tests (in-process benchmark with the judge's route checker): a valid route on 400/400 random draws; optimal in 83 to 84 of 84 runs on the 28 draws with a known exact optimum (an integer-programming reference used only for testing, never by the agent), the one miss being 63 km (0.5%). In the official judge in Google Colab: 20/20 against `demo` with the same km as on the laptop on those 20 seeds, and the training instance solved at its exact optimum (13,599 km). In the official judge: `check_requirements` 4/4; no incidents and no invalid or late routes in any game. The older versions and the test scripts are kept by the team and are available on request.

#### Time and memory
The search stops at 93% of the limit minus 0.05 s (4.6 s of 5 s), and the clock is read every 16 moves. A first valid route is submitted within 0.1 s on the laptop (0.3 s in Colab), so a route always counts. On a slow machine the first build is timed and the number of rounds is reduced (from 24 down to 4). With the limit cut to 1.25 s to imitate a machine roughly four times slower (Colab measured about 5 times slower than the laptop in one test): 200/200 valid, 17/200 short of the best known route, mean gap 7 km. All work after the first submission runs inside `try/except`, so an error cannot remove a submitted route. Peak memory is 77 MB.

#### Limits
It is a heuristic: it cannot prove that its route is optimal, and on a slower machine it misses the best route more often. Legs are chosen one at a time as shortest paths, so a change that needs two far-apart legs to give way together is hard to reach (the one known miss). Two agents that both find the optimum draw the game. With more time: branch and bound to prove optimality, and a faster first build so that each restart costs less.

### Chess

#### Problem formulation
- **Game:** two-player, zero-sum, deterministic and fully observable. White is MAX, Black is MIN; utility per game is 1 (win), 0.5 (draw), 0 (loss).
- **State:** the board plus the move history. The history is replayed from `history` because a FEN has no past positions, and threefold repetition depends on them.
- **Actions:** the legal moves in UCI; the judge's python-chess generates them and applies the rules.
- **What the agent optimises:** the minimax value of the position, estimated with a depth-limited search and a linear evaluation, within 5 s of decision time per game.

#### Algorithms
| Algorithm | Course topic | Where in `chess_agent.py` |
|---|---|---|
| Evaluation E(s) = 100·M + P + 10·U, White minus Black | Linear evaluation Σ wᵢfᵢ(s), depth cutoff (Session 15) | `_heuristic`, `_material`, `_piece_square`, `_mop_up` |
| Terminal utility: mate = ±(1,000,000 − ply), draws = 0 | U(s) at terminal states | `_terminal_value` |
| Required minimax and alpha-beta (fixed depth, all nodes counted, minimax = alpha-beta without the prune) | Minimax, alpha-beta (Session 14) | `minimax`, `alphabeta`, `_fixed_depth` |
| Iterative deepening, depth 1 to 8 | Iterative deepening (Session 4) | `act` |
| Alpha-beta with move ordering: remembered best move, captures (most valuable victim first), promotions, killer moves | Alpha-beta, move ordering (Session 14) | `_search_root`, `_ab`, `_order_moves`, `_remember_killer` |
| Transposition table: position → (depth, value, exact/lower/upper bound, best move) | Handbook extension | `_TT` in `_ab` |
| Quiescence search: at the cutoff, keep searching captures and promotions (≤ 4 plies) until quiet | Fixes the horizon problem of the depth cutoff | `_quiesce`, `_capture_gain`, `_bad_capture` |

**Evaluation features.** M = material in pawns (1, 3, 3, 5, 9), weight 100. P = piece-square bonus in hundredths of a pawn: one 64-square table per piece type (the standard "simplified evaluation function" values: knights in the centre, pawns advanced, king sheltered in the middlegame and central in the endgame). U = mop-up, only when one side is 4+ pawns ahead with 6 pieces or fewer: the losing king's distance from the centre plus how close the winning king is, weight 10. Pruning never changes the value: a branch is skipped only when α ≥ β, that is, when the player above already has a better option, so it could never be chosen.

#### Baseline against improved version
Each change was tested on its own copy against the previous version (20-game matches from random openings, colours swapped, judge's rules and 5 s clock), then merged only if it helped.

| Version | What changed | Result |
|---|---|---|
| v1 | E = 100·M + 5·mobility + 3·pawn advance; iterative-deepening alpha-beta, captures first | 4/4 vs a fixed depth-2 alpha-beta |
| v2 | + mop-up term, whole root ordered by last scores, predicted time for the next depth, clock reserve and no-search emergency move | Won endgames converted 0/12 → 12/12; 8% fewer nodes; no timeouts |
| v3 | + quiescence search | 18.5/20 vs v2; error on 10 tactical positions 475 → 163 centipawns |
| v3 | + piece-square tables, mobility removed (scoring 6–8× faster) | 17.5/20 vs v2 |
| v3 | + transposition table and killer moves | 42–65% fewer nodes, same values; 2.8× faster to depth 4 |

Rating, measured against Stockfish 19 limited to its calibrated Elo levels (used only for testing, never imported by the agent): v2 about 850–1000; **v3 about 1430 (95% interval 1320–1540)**, 8.5/20 against the 1500 level. Tried and dropped: mobility, bishop pair, pawn-structure and king-shelter features, a centre feature, history ordering (no measurable gain).

In the official judge: `check_requirements` 3/3; 10/10 challenges against `demo`, 10/10 against `demo_b`, 5/5 in the single 10 s game format, 10/10 with the clock halved to 2.5 s; no incidents.

#### Time and memory
Per move: budget = min((clock − 0.8 s) / max(20, 32 − move number / 2), 0.6 s). Depth 1 always finishes; deeper depths stop at the deadline and the unfinished one is discarded; the next depth starts only if its predicted time fits. Below about 0.85 s on the clock the agent plays the best capture or promotion with no search. Lowest clock left in the judge: 1.09 s at 5 s, 0.99 s at 2.5 s. Any error returns a legal move. The transposition table and killers are cleared every move and capped at 300,000 entries.

#### Limits
Python reaches only a few plies in 5 s. Quiescence looks at captures, not checks. The evaluation has no explicit pawn-structure or king-attack knowledge, and there is no opening book. With more time: check extensions in quiescence and a tuned evaluation.

### Battleship

#### Problem formulation
- **Hidden state:** for each enemy ship, its top-left cell, orientation, and whether it can still move. Ships cannot overlap or leave the board.
- **Observations:** the result of each of our shots (`MISS`, `HIT`, `SUNK` with the length), up to two noisy sonar counts (80% exact, 10% one too low, 10% one too high), and the hits our own fleet takes. We never see the rival's moves or misses.
- **Actions:** one shot or one sonar scan per turn, plus an optional one-cell move of one unhit ship.
- **Goal:** destroy all 17 enemy segments first; after 150 rounds, have destroyed more.
- **Why it is hard:** a ship can move until its first hit, so a miss is only reliable for one turn. The game is partially observable, stochastic and adversarial.

#### Algorithms
| Algorithm | Course topic | Where in `battleship_agent.py` |
|---|---|---|
| Placements for one ship | CSP: variable = placement; constraints = on the board, avoid blocked cells, cover required cells | `consistent_placements` |
| Target mode (after a hit) | CSP over the damaged ships, solved by **backtracking with MRV and forward checking**; collects all solutions, weighted by the belief, and shoots the cell they cover most | `_hit_domains`, `_select_hit`, `_forward_check`, `_backtrack`, `_target_scores` |
| Hunt mode (belief filter) | One probability grid per unseen ship. Each turn: **predict** with the transition model, then **update** with Bayes' rule on a miss, hit or sonar reading | `predict_ship_belief`, `_predict_all`, `_process_shot`, `sonar_posterior`, `_sonar_update` |
| Rival movement | The filter runs under five movement hypotheses (never moves, sometimes, every turn, drawn to the border, drawn to the centre), weighted by how well each predicted our hits and misses | `CFG["models"]`, `_predict_pulled`, `_update_weights` |
| Sonar | Value of information: scan only if the expected best hit chance afterwards is at least 1.8× the best shot now | `_best_scan` |
| Fleet placement | CSP solved by **min-conflicts local search** (no two ships touch), biased away from the centre | `_min_conflicts_layout`, `_choose_placement` |
| Defence | One unhit ship moves each turn: away from where a simulated rival shooter would aim next, next to our sunk ships, and towards the zone that has proved safe | `_choose_move`, `_shadow` |

Logic the agent uses with certainty: a hit ship never moves again, so a hit cell stays occupied; a miss made *after* a ship was hit is permanent for that ship, a miss made *before* it is not; `SUNK` removes exactly one ship of that length.

#### Baseline against improved version
Baseline: the classic probability-density hunter (count every placement that fits, treat every miss as permanent, random placement, never move). Win rates with the judge's rules and a 5 s clock, 500 to 1000 games per rival:

| Rival | Baseline | Final agent |
|---|---|---|
| `demo_a` | 28% | 97% |
| Density hunter whose ships move every turn | 5% | 94% |
| Three agents written independently as stand-ins for other teams | 3–25% | 70–91% |
| Our own first complete version | 0.4% | 74–78% |

What each change was worth (300 games per rival): moving a ship every turn (27% → 88% against a moving hunter), learning which zone is safe (78% → 93% against a hunter that forgets misses), the "drawn to the border / centre" hypotheses (47% → 75% against a rival whose ships run along the border), the simulated rival shooter (about +3 points on average against the independent rivals). Sonar adds 1 to 5 points, so it is used only late, when the belief is flat. Tried and rejected: weighting long ships more, learning the rival's placement style, moving less often. The rival agents and test scripts are kept by the team and are available on request.

In the official judge: `check_requirements` 3/3; 9/10 against `demo_a`, 9/10 against `demo`, 10/10 against `demo_b`; no incidents.

#### Time and memory
A game uses under 1 s of the 5 s clock (lowest seen in the judge: 4.2 s left). Below 1 s the agent skips the sonar analysis; below 0.35 s it answers at once with any legal shot. `act` is wrapped so an unexpected error still returns a legal shot. Memory is a few 10 × 10 grids per ship and hypothesis.

#### Limits
Ships are filtered independently, so "unseen ships do not overlap" is only approximate. The movement hypotheses are fixed in advance, so a rival with an unusual plan is harder to track; our hardest test rivals still win about 3 games in 10. With more time: a particle filter over whole fleets, and using the unused clock for look-ahead.

### Catan

#### Problem formulation
- **Environment:** two players, 19-tile board, 7 VP to win, bank and port trades only. Dice and the development deck are random; the rival's hand and cards are hidden. The judge lists the legal actions and we return one id.
- **Observations:** our full state, the rival's public state (buildings, roads, card counts, played cards), the board and the public log.
- **Goal:** reach 7 VP first. Each decision maximises the value of the state *after* the action (an afterstate), in hundredths of a VP.
- **Clock:** 5 s for the whole game, so the agent never simulates the engine; every decision is a few loops over the 54 board nodes.

#### Algorithms
| Part | Course topic | Where in `catan_agent.py` |
|---|---|---|
| Board model: adjacency, pips per node, ports, resource scarcity | State representation | `BoardModel` |
| Rival's hand from the public log | Logic and inference | `OpponentTracker` |
| Builds as operators with **preconditions** (cost, legal place) and **effects** (VP, production, reach); searches up to 2 trades + 1 build per turn | Classical planning | `Planner`, `plan_value` |
| Afterstate value = structure (VP, production weighted by scarcity, reachable spots, road length, knights) + hand (cards, progress toward a city / settlement / dev card, expected discard loss) | Heuristic evaluation | `Evaluator`, `TurnSituation`, `hand_value` |
| Dev card value = expectation over the unknown deck; > 7 cards costs P(a 7 before my next turn) × half the hand | Expected value under uncertainty | `development_card_value`, `hand_value` |
| Robber: rival pips blocked − 1.3 × own pips blocked + value of a steal | Heuristic evaluation | `TurnSituation.robber_value` |
| Initial placement: pips × resource weight × scarcity, diversity, ports | Heuristic evaluation | `placement_value`, `choose_initial` |
| Q-learning over an abstract MDP, used as a small bonus | MDP, Q-learning | `state_features`, `action_kind`, `Q`, `q_update` |
| Baseline policy: most pips, fixed build priority, no trades | Baseline | `heuristic_policy` |

**Logic over public events.** The rival's production is exact (dice, its buildings and the robber are public), as are its build costs, development-card purchases, maritime trades and Year of Plenty. Discards, steals we did not take part in and Monopoly gains are uncertain: they are spread in proportion and the estimate is rescaled to the rival's public card count. This drives the Monopoly choice and the value of a steal.

**MDP and Q-learning.** State = (phase, my VP, VP lead clamped to ±3, production bucket, can afford a city / settlement / dev card). Action = the action type; the planner picks the concrete action. Reward = VP gained between my decisions, ±1 at the end; γ = 0.95, α = 0.1, ε-greedy 0.3 → 0.05. Used as `planner value + 100 · 0.1 · Q(s, type)`.

#### Baseline against improved version
Same 500 held-out seeds for every row (never used for tuning or training), both seats alternated.

| Agent | vs `demo_A` | Mean rounds |
|---|---|---|
| Baseline (`heuristic_policy`) | 36.8% | 42.8 |
| Planner + tracker + tuned weights | 91.6% | 23.7 |
| Final (with Q bonus 0.1) | 89.6% (53.6% head-to-head vs the planner without Q) | 23.8 |

The planner is the big gain (36.8% → 91.6%, games twice as short). Weight tuning by hill climbing gave a small real gain (54.0% head-to-head against the hand-set weights). The Q-learning bonus is within noise (about ±2.2 points): it refines the planner rather than replacing it, and larger Q weights made the agent worse. The training and comparison scripts are kept by the team and are available on request.

In the official judge: `check_requirements` 4/4; 19/20 against `demo`, 17/20 against `demo_a`, 20/20 against `demo_b`; no incidents.

#### Time and memory
About 55 ms of thinking per game (1–2 ms per decision, slowest about 25 ms). In the judge, including the pipe and JSON overhead, the lowest clock left was 4.9 s. Below 0.4 s the agent switches to the baseline policy, and any unexpected error also falls back to it, so it never forfeits (checked with a 0.3 s clock: 9/10 wins, no incidents). The file, Q-table included, is about 125 KB and loads instantly.

#### Limits
One-turn horizon: saving over several turns is only approximated by the "progress toward a goal" terms. The rival model is passive, so we do not block its best spots on purpose. The Q abstraction is coarse. With more time: expectimax over the next roll for build-or-hold decisions, and blocking the rival's best expansion spot.

## Use of generative AI
We used Claude (Anthropic) as a coding assistant for all agents: to help design, write and test the code, to write rival agents for testing (Battleship), and to draft this README. For Road Trip, a second Claude session also attacked and audited the agent's build and performance, and its author set the requirements and the review process. Every team member reviewed the code, and we can explain each algorithm ourselves.
