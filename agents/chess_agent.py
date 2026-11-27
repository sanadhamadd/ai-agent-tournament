"""Chess agent: AI Agent Tournament, AI: Reasoning & Problem Solving.

Minimax with alpha-beta (Session 14), a linear evaluation with a depth cutoff
(Session 15) and iterative deepening (Session 4), plus three standard
extensions: quiescence search, a transposition table and killer moves.
White is MAX, Black is MIN. python-chess only gives the board and the rules.
"""
import time

import chess

INF = float("inf")
MATE = 1_000_000.0                  # utility of a checkmate, U(s)
DRAW = 0.0
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
W_MATERIAL, W_MOPUP = 100.0, 10.0   # piece-square tables already use 1/100 pawn

# ---------------------------------------------------------------------------
# Evaluation E(s) = 100·M + P + 10·U, always White minus Black (Session 15)
#   M  material, P  piece-square bonus, U  mop-up in won endgames
# ---------------------------------------------------------------------------

# Piece-square tables ("simplified evaluation function"), drawn as White sees
# the board: first row = rank 8. Knights like the centre, pawns like to advance,
# the king hides in the middlegame and comes to the centre in the endgame.
PST = {
    chess.PAWN: [0, 0, 0, 0, 0, 0, 0, 0,
                 50, 50, 50, 50, 50, 50, 50, 50,
                 10, 10, 20, 30, 30, 20, 10, 10,
                 5, 5, 10, 25, 25, 10, 5, 5,
                 0, 0, 0, 20, 20, 0, 0, 0,
                 5, -5, -10, 0, 0, -10, -5, 5,
                 5, 10, 10, -20, -20, 10, 10, 5,
                 0, 0, 0, 0, 0, 0, 0, 0],
    chess.KNIGHT: [-50, -40, -30, -30, -30, -30, -40, -50,
                   -40, -20, 0, 0, 0, 0, -20, -40,
                   -30, 0, 10, 15, 15, 10, 0, -30,
                   -30, 5, 15, 20, 20, 15, 5, -30,
                   -30, 0, 15, 20, 20, 15, 0, -30,
                   -30, 5, 10, 15, 15, 10, 5, -30,
                   -40, -20, 0, 5, 5, 0, -20, -40,
                   -50, -40, -30, -30, -30, -30, -40, -50],
    chess.BISHOP: [-20, -10, -10, -10, -10, -10, -10, -20,
                   -10, 0, 0, 0, 0, 0, 0, -10,
                   -10, 0, 5, 10, 10, 5, 0, -10,
                   -10, 5, 5, 10, 10, 5, 5, -10,
                   -10, 0, 10, 10, 10, 10, 0, -10,
                   -10, 10, 10, 10, 10, 10, 10, -10,
                   -10, 5, 0, 0, 0, 0, 5, -10,
                   -20, -10, -10, -10, -10, -10, -10, -20],
    chess.ROOK: [0, 0, 0, 0, 0, 0, 0, 0,
                 5, 10, 10, 10, 10, 10, 10, 5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 -5, 0, 0, 0, 0, 0, 0, -5,
                 0, 0, 0, 5, 5, 0, 0, 0],
    chess.QUEEN: [-20, -10, -10, -5, -5, -10, -10, -20,
                  -10, 0, 0, 0, 0, 0, 0, -10,
                  -10, 0, 5, 5, 5, 5, 0, -10,
                  -5, 0, 5, 5, 5, 5, 0, -5,
                  0, 0, 5, 5, 5, 5, 0, -5,
                  -10, 5, 5, 5, 5, 5, 0, -10,
                  -10, 0, 5, 0, 0, 0, 0, -10,
                  -20, -10, -10, -5, -5, -10, -10, -20],
    chess.KING: [-30, -40, -40, -50, -50, -40, -40, -30,     # middlegame
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -30, -40, -40, -50, -50, -40, -40, -30,
                 -20, -30, -30, -40, -40, -30, -30, -20,
                 -10, -20, -20, -20, -20, -20, -20, -10,
                 20, 20, 0, 0, 0, 0, 20, 20,
                 20, 30, 10, 0, 0, 10, 30, 20],
}
KING_END = [-50, -40, -30, -20, -20, -30, -40, -50,           # endgame
            -30, -20, -10, 0, 0, -10, -20, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 30, 40, 40, 30, -10, -30,
            -30, -10, 20, 30, 30, 20, -10, -30,
            -30, -30, 0, 0, 0, 0, -30, -30,
            -50, -30, -30, -30, -30, -30, -30, -50]
# python-chess numbers squares a1 = 0 ... h8 = 63, so White reads the picture
# at sq ^ 56 (rank flipped) and Black reads it as drawn (its mirror image).
PST_WHITE = {pt: [t[sq ^ 56] for sq in range(64)] for pt, t in PST.items()}
KING_END_WHITE = [KING_END[sq ^ 56] for sq in range(64)]


def _material(board):
    return sum(v * (chess.popcount(board.pieces_mask(pt, chess.WHITE))
                    - chess.popcount(board.pieces_mask(pt, chess.BLACK)))
               for pt, v in PIECE_VALUE.items())


def _piece_square(board):
    # Endgame = no queens, or queens with at most two other non-pawn pieces.
    endgame = not board.queens or chess.popcount(board.knights | board.bishops | board.rooks) <= 2
    p = 0
    for pt in PST:
        white_t, black_t = PST_WHITE[pt], PST[pt]
        if pt == chess.KING and endgame:
            white_t, black_t = KING_END_WHITE, KING_END
        p += sum(white_t[sq] for sq in chess.scan_forward(board.pieces_mask(pt, chess.WHITE)))
        p -= sum(black_t[sq] for sq in chess.scan_forward(board.pieces_mask(pt, chess.BLACK)))
    return p


def _mop_up(board, m):
    """Only when one side is 4+ pawns ahead with 6 pieces or fewer: the losing
    king's distance from the centre (0..6) plus how close the winning king is
    (0..7). Converts won endgames instead of drawing them."""
    if abs(m) < 4 or chess.popcount(board.occupied) > 6:
        return 0
    win = m > 0                                    # True = White is winning
    lose_k, win_k = board.king(not win), board.king(win)
    f, r = chess.square_file(lose_k), chess.square_rank(lose_k)
    score = max(3 - f, f - 4) + max(3 - r, r - 4) + 7 - chess.square_distance(win_k, lose_k)
    return score if win else -score


def _heuristic(board):
    m = _material(board)
    return W_MATERIAL * m + _piece_square(board) + W_MOPUP * _mop_up(board, m)


def _terminal_value(board, ply=0, has_moves=None):
    """U(s) if the game is over, else None. A faster mate scores higher.
    has_moves can be passed in when the caller already generated the moves."""
    if has_moves is None:
        has_moves = any(True for _ in board.generate_legal_moves())
    if not has_moves:                              # checkmate or stalemate
        if not board.is_check():
            return DRAW
        return -(MATE - ply) if board.turn == chess.WHITE else MATE - ply
    if board.is_insufficient_material() or board.is_fifty_moves() or board.is_repetition(3):
        return DRAW
    return None


def _eval_board(board, ply=0):
    u = _terminal_value(board, ply)
    return u if u is not None else _heuristic(board)


def evaluate(fen):
    return float(_eval_board(chess.Board(fen)))


# ---------------------------------------------------------------------------
# Required functions (Session 14): fixed depth, no extras, every position
# visited is counted (root included). Minimax is alpha-beta without the prune.
# ---------------------------------------------------------------------------

def _fixed_depth(fen, depth, prune):
    board, nodes = chess.Board(fen), [0]

    def value(d, ply, alpha, beta):
        nodes[0] += 1
        if d == 0 or board.is_game_over(claim_draw=False):
            return _eval_board(board, ply)
        maximizing = board.turn == chess.WHITE
        best = -INF if maximizing else INF
        for move in list(board.legal_moves):
            board.push(move)
            v = value(d - 1, ply + 1, alpha, beta)
            board.pop()
            if maximizing:
                best, alpha = max(best, v), max(alpha, v)
            else:
                best, beta = min(best, v), min(beta, v)
            if prune and alpha >= beta:            # the player above never allows this
                break
        return best

    return {"value": float(value(depth, 0, -INF, INF)), "nodes": nodes[0]}


def minimax(fen, depth):
    return _fixed_depth(fen, depth, prune=False)


def alphabeta(fen, depth):
    return _fixed_depth(fen, depth, prune=True)


# ---------------------------------------------------------------------------
# Playing search: iterative-deepening alpha-beta with
#   - move ordering (Session 14): likely-best moves first = more pruning
#   - transposition table: position -> (depth, value, bound, best move), so a
#     position reached by another move order is not searched twice
#   - killer moves: quiet moves that caused a cutoff at the same ply
#   - quiescence search at the cutoff: keep searching captures until quiet
# ---------------------------------------------------------------------------

class _OutOfTime(Exception):
    pass


EXACT, LOWER, UPPER = 0, 1, 2
TT_MAX = 300_000
_TT, _KILLERS = {}, {}              # cleared at the start of every move


def _order_moves(board, moves, first=None, ply=None):
    """Remembered best move, then captures (most valuable victim, cheapest
    attacker), promotions, the killer moves of this ply, the rest."""
    killers = _KILLERS.get(ply, ())

    def score(move):
        if move == first:
            return 1000
        s = 0
        if board.is_capture(move):
            victim = board.piece_type_at(move.to_square)
            s += 100 + 10 * (PIECE_VALUE[victim] if victim else 1) \
                - (PIECE_VALUE[board.piece_type_at(move.from_square)] or 10)
        if move.promotion:
            s += 90 + PIECE_VALUE[move.promotion]
        return 80 if s == 0 and move in killers else s
    return sorted(moves, key=score, reverse=True)


def _remember_killer(board, move, ply):
    if not board.is_capture(move) and not move.promotion:
        k = _KILLERS.setdefault(ply, [])
        if move not in k:
            k.insert(0, move)
            del k[2:]


def _search_root(board, depth, deadline, root_order=None):
    """One depth from the root. Returns (value, best move, root moves sorted
    by this depth's scores, used to order the next depth)."""
    maximizing = board.turn == chess.WHITE
    alpha, beta = -INF, INF
    best, best_move, scored = (-INF if maximizing else INF), None, []
    for move in root_order or _order_moves(board, list(board.legal_moves)):
        board.push(move)
        v = _ab(board, depth - 1, 1, alpha, beta, deadline)
        board.pop()
        scored.append((v, move))
        if (v > best) if maximizing else (v < best):
            best, best_move = v, move
            if maximizing:
                alpha = max(alpha, v)
            else:
                beta = min(beta, v)
    scored.sort(key=lambda t: t[0], reverse=maximizing)
    return best, best_move, [m for _, m in scored]


def _ab(board, depth, ply, alpha, beta, deadline):
    if time.perf_counter() > deadline:
        raise _OutOfTime
    moves = list(board.legal_moves) if depth > 0 else None
    u = _terminal_value(board, ply, bool(moves) if depth > 0 else None)
    if u is not None:
        return u
    if depth == 0:                     # cutoff: do not stop mid-exchange
        return _quiesce(board, ply, alpha, beta, deadline, MAX_QDEPTH)

    key = board._transposition_key()
    tt_move = None
    if key in _TT:                     # searched before: reuse if deep enough
        e_depth, e_value, e_bound, tt_move = _TT[key]
        if e_depth >= depth and (e_bound == EXACT
                                 or (e_bound == LOWER and e_value >= beta)
                                 or (e_bound == UPPER and e_value <= alpha)):
            return e_value

    maximizing = board.turn == chess.WHITE
    alpha0, beta0 = alpha, beta
    best, best_move = (-INF if maximizing else INF), None
    for move in _order_moves(board, moves, tt_move, ply):
        board.push(move)
        v = _ab(board, depth - 1, ply + 1, alpha, beta, deadline)
        board.pop()
        if (v > best) if maximizing else (v < best):
            best, best_move = v, move
        if maximizing:
            alpha = max(alpha, best)
        else:
            beta = min(beta, best)
        if alpha >= beta:              # prune: the player above avoids this
            _remember_killer(board, move, ply)
            break

    # Store the result. Mate and draw scores are skipped: they depend on the
    # path (ply, repetitions), not only on the position.
    if best != DRAW and abs(best) < MATE - 1000:
        bound = UPPER if best <= alpha0 else LOWER if best >= beta0 else EXACT
        if len(_TT) >= TT_MAX:
            _TT.clear()
        _TT[key] = (depth, best, bound, best_move)
    return best


# Quiescence: at depth 0 keep searching captures and promotions only, so E(s)
# is never applied while a piece is hanging (the horizon problem).
MAX_QDEPTH = 4                         # at most 4 extra plies
DELTA_MARGIN = 200.0                   # skip captures that cannot matter even +2 pawns


def _capture_gain(board, move):
    victim = board.piece_type_at(move.to_square)
    gain = PIECE_VALUE[victim] if victim else (1 if board.is_capture(move) else 0)
    return W_MATERIAL * (gain + (PIECE_VALUE[move.promotion] - 1 if move.promotion else 0))


def _bad_capture(board, move):
    """A bigger piece taking a smaller, defended one (e.g. queen x defended pawn)."""
    victim = board.piece_type_at(move.to_square)
    return (victim is not None and not move.promotion
            and PIECE_VALUE[victim] < PIECE_VALUE[board.piece_type_at(move.from_square)]
            and board.is_attacked_by(not board.turn, move.to_square))


def _quiesce(board, ply, alpha, beta, deadline, qdepth):
    if time.perf_counter() > deadline:
        raise _OutOfTime
    if board.is_check() and board.is_checkmate():
        return -(MATE - ply) if board.turn == chess.WHITE else MATE - ply
    # Stand pat: not capturing is allowed, so E(s) bounds the side to move.
    best = stand_pat = _heuristic(board)
    if qdepth == 0:
        return best
    maximizing = board.turn == chess.WHITE
    if (best >= beta) if maximizing else (best <= alpha):
        return best
    if maximizing:
        alpha = max(alpha, best)
    else:
        beta = min(beta, best)
    noisy = list(board.generate_legal_captures()) + [
        m for m in board.generate_legal_moves(board.pawns, chess.BB_RANK_1 | chess.BB_RANK_8)
        if not board.is_capture(m)]    # non-capturing promotions
    for move in _order_moves(board, noisy):
        if _bad_capture(board, move):
            continue
        gain = _capture_gain(board, move) + DELTA_MARGIN
        if (stand_pat + gain <= alpha) if maximizing else (stand_pat - gain >= beta):
            continue                   # delta pruning: cannot change the result
        board.push(move)
        v = _quiesce(board, ply + 1, alpha, beta, deadline, qdepth - 1)
        board.pop()
        if maximizing:
            best = max(best, v)
            alpha = max(alpha, best)
        else:
            best = min(best, v)
            beta = min(beta, best)
        if alpha >= beta:
            break
    return best


# ---------------------------------------------------------------------------
# Clock and the move decision
# ---------------------------------------------------------------------------

RESERVE_S = 0.8                        # never touch the last 0.8 s
MAX_DEPTH = 8


def _time_budget(obs, board):
    """Usable clock spread over the moves still expected (at least 20)."""
    remaining = float(obs.get("time_remaining_s", 5.0)) - RESERVE_S
    if remaining <= 0.05:
        return 0.0
    return min(remaining / max(20, 32 - board.fullmove_number // 2), 0.6)


def _board_from(obs):
    """Replay the history so repetitions can be detected (a FEN has no past)."""
    board = chess.Board()
    try:
        for uci in obs.get("history", []):
            board.push_uci(uci)
        if board.board_fen() == obs["fen"].split()[0]:
            return board
    except Exception:
        pass
    return chess.Board(obs["fen"])


def act(observation):
    legal = observation["legal_moves"]
    try:
        if len(legal) == 1:
            return {"move": legal[0]}
        board = _board_from(observation)
        budget = _time_budget(observation, board)
        _TT.clear()
        _KILLERS.clear()
        if budget <= 0.0:
            # Emergency: almost no clock left, so no search at all.
            best_move = _order_moves(board, list(board.legal_moves))[0]
        else:
            # Iterative deepening: each finished depth gives a decision; an
            # unfinished one is dropped. Depth 1 always finishes.
            start = time.perf_counter()
            deadline, best_move, root_order, times = start + budget, None, None, []
            for depth in range(1, MAX_DEPTH + 1):
                t_depth = time.perf_counter()
                try:
                    value, best_move, root_order = _search_root(
                        board, depth, deadline if depth > 1 else INF, root_order)
                except _OutOfTime:
                    break
                if abs(value) >= MATE - 100:   # forced mate found
                    break
                # Start the next depth only if its predicted time fits.
                times.append(time.perf_counter() - t_depth)
                growth = times[-1] / times[-2] if len(times) > 1 and times[-2] > 0 else 6.0
                if time.perf_counter() - start + times[-1] * min(max(growth, 2.0), 12.0) > budget:
                    break
        if best_move is not None and best_move.uci() in legal:
            return {"move": best_move.uci()}
    except Exception as exc:           # never lose a game to an exception
        print("chess_agent error:", exc)
    return {"move": legal[0] if legal else "resign"}
