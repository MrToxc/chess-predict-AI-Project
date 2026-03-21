"""Enhanced chess game feature extractor with deeper positional analysis."""

import json
import os
import logging
from io import StringIO

import chess
import chess.pgn
import pandas as pd

INPUT_FILE_PATH = os.path.join("data", "raw_games.json")
OUTPUT_FILE_PATH = os.path.join("data", "features.csv")
MAXIMUM_FULL_MOVES = 25

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}

WHITE_MINOR_PIECE_STARTING_SQUARES = {chess.B1, chess.G1, chess.C1, chess.F1}
BLACK_MINOR_PIECE_STARTING_SQUARES = {chess.B8, chess.G8, chess.C8, chess.F8}
CENTER_SQUARES = [chess.E4, chess.D4, chess.E5, chess.D5]
EXTENDED_CENTER_SQUARES = [chess.C3, chess.D3, chess.E3, chess.F3,
                           chess.C4, chess.D4, chess.E4, chess.F4,
                           chess.C5, chess.D5, chess.E5, chess.F5,
                           chess.C6, chess.D6, chess.E6, chess.F6]
VALID_RESULTS = ("1-0", "0-1", "1/2-1/2")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Board analysis functions
# ---------------------------------------------------------------------------

def compute_material_difference(board: chess.Board) -> int:
    difference = 0
    for piece_type, value in PIECE_VALUES.items():
        difference += len(board.pieces(piece_type, chess.WHITE)) * value
        difference -= len(board.pieces(piece_type, chess.BLACK)) * value
    return difference


def compute_total_material(board: chess.Board) -> tuple[int, int]:
    white_material = 0
    black_material = 0
    for piece_type, value in PIECE_VALUES.items():
        white_material += len(board.pieces(piece_type, chess.WHITE)) * value
        black_material += len(board.pieces(piece_type, chess.BLACK)) * value
    return white_material, black_material


def count_pieces_by_type(board: chess.Board) -> dict:
    return {
        "white_pawns": len(board.pieces(chess.PAWN, chess.WHITE)),
        "black_pawns": len(board.pieces(chess.PAWN, chess.BLACK)),
        "white_knights": len(board.pieces(chess.KNIGHT, chess.WHITE)),
        "black_knights": len(board.pieces(chess.KNIGHT, chess.BLACK)),
        "white_bishops": len(board.pieces(chess.BISHOP, chess.WHITE)),
        "black_bishops": len(board.pieces(chess.BISHOP, chess.BLACK)),
        "white_rooks": len(board.pieces(chess.ROOK, chess.WHITE)),
        "black_rooks": len(board.pieces(chess.ROOK, chess.BLACK)),
        "white_queens": len(board.pieces(chess.QUEEN, chess.WHITE)),
        "black_queens": len(board.pieces(chess.QUEEN, chess.BLACK)),
    }


def has_bishop_pair(board: chess.Board, color: chess.Color) -> int:
    return 1 if len(board.pieces(chess.BISHOP, color)) >= 2 else 0


def count_developed_pieces(board: chess.Board, color: chess.Color, starting_squares: set[int]) -> int:
    count = 0
    for square in starting_squares:
        piece = board.piece_at(square)
        is_original = (piece is not None and piece.color == color
                       and piece.piece_type in (chess.KNIGHT, chess.BISHOP))
        if not is_original:
            count += 1
    return count


def compute_center_control(board: chess.Board) -> tuple[int, int]:
    white_control = sum(1 for sq in CENTER_SQUARES if board.is_attacked_by(chess.WHITE, sq))
    black_control = sum(1 for sq in CENTER_SQUARES if board.is_attacked_by(chess.BLACK, sq))
    return white_control, black_control


def compute_extended_center_control(board: chess.Board) -> tuple[int, int]:
    white_control = sum(1 for sq in EXTENDED_CENTER_SQUARES if board.is_attacked_by(chess.WHITE, sq))
    black_control = sum(1 for sq in EXTENDED_CENTER_SQUARES if board.is_attacked_by(chess.BLACK, sq))
    return white_control, black_control


def compute_mobility(board: chess.Board) -> tuple[int, int]:
    """Count legal moves for the side to move, then flip to count the other side."""
    if board.turn == chess.WHITE:
        white_mobility = len(list(board.legal_moves))
        board.push(chess.Move.null())
        black_mobility = len(list(board.legal_moves))
        board.pop()
    else:
        black_mobility = len(list(board.legal_moves))
        board.push(chess.Move.null())
        white_mobility = len(list(board.legal_moves))
        board.pop()
    return white_mobility, black_mobility


def count_attacked_squares(board: chess.Board) -> tuple[int, int]:
    white_attacks = 0
    black_attacks = 0
    for square in chess.SQUARES:
        if board.is_attacked_by(chess.WHITE, square):
            white_attacks += 1
        if board.is_attacked_by(chess.BLACK, square):
            black_attacks += 1
    return white_attacks, black_attacks


def count_pawn_structure_features(board: chess.Board, color: chess.Color) -> dict:
    """Count doubled, isolated, and passed pawns for a given color."""
    pawns = board.pieces(chess.PAWN, color)
    opponent_pawns = board.pieces(chess.PAWN, not color)
    direction = 1 if color == chess.WHITE else -1

    doubled_count = 0
    isolated_count = 0
    passed_count = 0

    files_with_pawns = set()
    for pawn_square in pawns:
        files_with_pawns.add(chess.square_file(pawn_square))

    for pawn_square in pawns:
        pawn_file = chess.square_file(pawn_square)
        pawn_rank = chess.square_rank(pawn_square)

        # Doubled: another own pawn on the same file
        same_file_pawns = [sq for sq in pawns if chess.square_file(sq) == pawn_file and sq != pawn_square]
        if same_file_pawns:
            doubled_count += 1

        # Isolated: no own pawns on adjacent files
        adjacent_files = []
        if pawn_file > 0:
            adjacent_files.append(pawn_file - 1)
        if pawn_file < 7:
            adjacent_files.append(pawn_file + 1)
        has_neighbor = any(f in files_with_pawns for f in adjacent_files)
        if not has_neighbor:
            isolated_count += 1

        # Passed: no opponent pawns on same or adjacent files ahead
        is_passed = True
        check_files = [pawn_file] + adjacent_files
        for opponent_square in opponent_pawns:
            opponent_file = chess.square_file(opponent_square)
            opponent_rank = chess.square_rank(opponent_square)
            if opponent_file not in check_files:
                continue
            if color == chess.WHITE and opponent_rank > pawn_rank:
                is_passed = False
                break
            if color == chess.BLACK and opponent_rank < pawn_rank:
                is_passed = False
                break
        if is_passed:
            passed_count += 1

    return {
        "doubled": doubled_count,
        "isolated": isolated_count,
        "passed": passed_count,
    }


def compute_king_safety(board: chess.Board, color: chess.Color) -> int:
    """Count how many pawns are on adjacent squares around the king."""
    king_square = board.king(color)
    if king_square is None:
        return 0

    pawn_shield = 0
    for adjacent_square in chess.SQUARES:
        distance = chess.square_distance(king_square, adjacent_square)
        if distance > 2:
            continue
        piece = board.piece_at(adjacent_square)
        if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
            pawn_shield += 1

    return pawn_shield


def compute_king_exposure(board: chess.Board, color: chess.Color) -> int:
    """Count how many squares around the king are attacked by the opponent."""
    king_square = board.king(color)
    if king_square is None:
        return 0

    opponent_color = not color
    attacked_count = 0
    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)

    for file_offset in range(-1, 2):
        for rank_offset in range(-1, 2):
            if file_offset == 0 and rank_offset == 0:
                continue
            target_file = king_file + file_offset
            target_rank = king_rank + rank_offset
            if not (0 <= target_file <= 7 and 0 <= target_rank <= 7):
                continue
            target_square = chess.square(target_file, target_rank)
            if board.is_attacked_by(opponent_color, target_square):
                attacked_count += 1

    return attacked_count


def parse_eco_category(headers: chess.pgn.Headers) -> int:
    eco_code = headers.get("ECO", "")
    if not eco_code:
        return -1
    first_letter = eco_code[0].upper()
    if first_letter not in "ABCDE":
        return -1
    return ord(first_letter) - ord("A")


# ---------------------------------------------------------------------------
# Move tracking during replay
# ---------------------------------------------------------------------------

def replay_game_moves(board: chess.Board, game: chess.pgn.Game) -> dict | None:
    """Replay the first 25 moves and track statistics."""
    maximum_half_moves = MAXIMUM_FULL_MOVES * 2

    stats = {
        "white_castled": 0,
        "black_castled": 0,
        "white_castled_on_move": 0,
        "black_castled_on_move": 0,
        "pawn_moves_white": 0,
        "pawn_moves_black": 0,
        "piece_moves_white": 0,
        "piece_moves_black": 0,
        "captures_white": 0,
        "captures_black": 0,
        "checks_white": 0,
        "checks_black": 0,
    }

    half_move_index = 0

    for node in game.mainline():
        if half_move_index >= maximum_half_moves:
            break

        move = node.move
        is_white = (half_move_index % 2 == 0)
        full_move_number = (half_move_index // 2) + 1

        # Castling
        if board.is_castling(move):
            if is_white:
                stats["white_castled"] = 1
                stats["white_castled_on_move"] = full_move_number
            else:
                stats["black_castled"] = 1
                stats["black_castled_on_move"] = full_move_number

        # Captures
        if board.is_capture(move):
            if is_white:
                stats["captures_white"] += 1
            else:
                stats["captures_black"] += 1

        # Move type
        moving_piece = board.piece_at(move.from_square)
        if moving_piece is not None:
            if moving_piece.piece_type == chess.PAWN:
                if is_white:
                    stats["pawn_moves_white"] += 1
                else:
                    stats["pawn_moves_black"] += 1
            else:
                if is_white:
                    stats["piece_moves_white"] += 1
                else:
                    stats["piece_moves_black"] += 1

        board.push(move)

        # Checks (after pushing the move)
        if board.is_check():
            if is_white:
                stats["checks_white"] += 1
            else:
                stats["checks_black"] += 1

        half_move_index += 1

    if half_move_index < maximum_half_moves:
        return None

    return stats


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_features(pgn_string: str, result_string: str) -> dict | None:
    try:
        game = chess.pgn.read_game(StringIO(pgn_string))
    except Exception as exception:
        logger.warning("Failed to parse PGN: %s", exception)
        return None

    if game is None:
        return None

    result = result_string if result_string else game.headers.get("Result", "*")
    if result not in VALID_RESULTS:
        return None

    board = game.board()
    move_stats = replay_game_moves(board, game)
    if move_stats is None:
        return None

    # Position analysis after move 25
    material_diff = compute_material_difference(board)
    white_material, black_material = compute_total_material(board)
    piece_counts = count_pieces_by_type(board)

    white_developed = count_developed_pieces(board, chess.WHITE, WHITE_MINOR_PIECE_STARTING_SQUARES)
    black_developed = count_developed_pieces(board, chess.BLACK, BLACK_MINOR_PIECE_STARTING_SQUARES)

    center_white, center_black = compute_center_control(board)
    ext_center_white, ext_center_black = compute_extended_center_control(board)

    white_mobility, black_mobility = compute_mobility(board)
    white_attacks, black_attacks = count_attacked_squares(board)

    white_pawn_structure = count_pawn_structure_features(board, chess.WHITE)
    black_pawn_structure = count_pawn_structure_features(board, chess.BLACK)

    white_king_safety = compute_king_safety(board, chess.WHITE)
    black_king_safety = compute_king_safety(board, chess.BLACK)
    white_king_exposure = compute_king_exposure(board, chess.WHITE)
    black_king_exposure = compute_king_exposure(board, chess.BLACK)

    eco_category = parse_eco_category(game.headers)

    features = {
        # Material
        "material_diff": material_diff,
        "white_material": white_material,
        "black_material": black_material,
        **piece_counts,

        # Bishop pair
        "white_bishop_pair": has_bishop_pair(board, chess.WHITE),
        "black_bishop_pair": has_bishop_pair(board, chess.BLACK),

        # Development
        "white_developed": white_developed,
        "black_developed": black_developed,

        # Move stats from replay
        **move_stats,

        # Center control
        "center_control_white": center_white,
        "center_control_black": center_black,
        "ext_center_white": ext_center_white,
        "ext_center_black": ext_center_black,

        # Mobility and space
        "white_mobility": white_mobility,
        "black_mobility": black_mobility,
        "white_attacked_squares": white_attacks,
        "black_attacked_squares": black_attacks,

        # Pawn structure
        "white_doubled_pawns": white_pawn_structure["doubled"],
        "black_doubled_pawns": black_pawn_structure["doubled"],
        "white_isolated_pawns": white_pawn_structure["isolated"],
        "black_isolated_pawns": black_pawn_structure["isolated"],
        "white_passed_pawns": white_pawn_structure["passed"],
        "black_passed_pawns": black_pawn_structure["passed"],

        # King safety
        "white_king_safety": white_king_safety,
        "black_king_safety": black_king_safety,
        "white_king_exposure": white_king_exposure,
        "black_king_exposure": black_king_exposure,

        # Opening
        "eco_category": eco_category,

        # Result
        "result": result,
    }

    return features


def run_extraction() -> None:
    if not os.path.exists(INPUT_FILE_PATH):
        logger.error("Input file not found: %s. Run crawler.py first.", INPUT_FILE_PATH)
        return

    with open(INPUT_FILE_PATH, "r", encoding="utf-8") as file:
        raw_games = json.load(file)

    logger.info("Loaded %d raw games from %s", len(raw_games), INPUT_FILE_PATH)

    features_list: list[dict] = []
    skipped_count = 0
    error_count = 0

    for game_index, game_record in enumerate(raw_games):
        pgn_string = game_record.get("pgn", "")
        if not pgn_string:
            error_count += 1
            continue

        result_string = ""
        try:
            quick_game = chess.pgn.read_game(StringIO(pgn_string))
            if quick_game:
                result_string = quick_game.headers.get("Result", "")
        except Exception:
            pass

        try:
            features = extract_features(pgn_string, result_string)
        except Exception as exception:
            logger.warning("Error extracting features from game %d: %s", game_index, exception)
            error_count += 1
            continue

        if features is None:
            skipped_count += 1
            continue

        features_list.append(features)

        if (game_index + 1) % 500 == 0:
            logger.info("  Processed %d / %d games", game_index + 1, len(raw_games))

    if features_list:
        dataframe = pd.DataFrame(features_list)
        os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
        dataframe.to_csv(OUTPUT_FILE_PATH, index=False)
        logger.info("Saved %d feature rows (%d columns) to %s", len(dataframe), len(dataframe.columns), OUTPUT_FILE_PATH)
    else:
        logger.warning("No features extracted — output file not created.")

    logger.info(
        "Done. Extracted: %d | Skipped: %d | Errors: %d",
        len(features_list), skipped_count, error_count,
    )


if __name__ == "__main__":
    run_extraction()
