#!/usr/bin/env python3
"""Prepare move-only human PGNs and Stockfish-labelled Glory training records.

This produces a raw Rec[] sidecar in the exact 36-byte little-endian layout
used by glory_nnue.c. The matching C replay-loader integration is documented
in README-teacher-data.md; do not overwrite existing gen_*.bin files or state.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

try:
    import chess
    import chess.engine
    import chess.pgn
except ImportError:
    raise SystemExit("Install dependency: python3 -m pip install python-chess")

SCORE_NONE = 32767
MAX_SCORE_CP = 2000
REC = __import__("struct").Struct("<32Bb h")


def pack_board(board: chess.Board) -> bytes:
    packed = bytearray(32)
    for sq, piece in board.piece_map().items():
        # C enum: white P..K = 0..5, black P..K = 6..11; empty nibble = 0.
        pc = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        nibble = pc + 1
        packed[sq >> 1] |= nibble << ((sq & 1) * 4)
    return bytes(packed)


def result_for_side_to_move(result: str, turn: bool) -> int:
    if result == "1-0":
        white_result = 1
    elif result == "0-1":
        white_result = -1
    else:
        white_result = 0
    return white_result if turn == chess.WHITE else -white_result


def clean_mainline(game: chess.pgn.Game) -> str:
    # Keep only useful PGN headers and the mainline: no comments, NAGs,
    # side variations, clock annotations, or embedded engine analysis.
    result = game.headers.get("Result", "*")
    white = game.headers.get("White", "White")
    black = game.headers.get("Black", "Black")
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    moves = game.accept(exporter).strip()
    return f'[White "{white}"]\n[Black "{black}"]\n[Result "{result}"]\n\n{moves} {result}\n\n'


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pgn", nargs="+", type=Path, help="input PGN files from human/outsider games")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="normally /run/media/brpr/Glory/glory_nnue/data")
    ap.add_argument("--stockfish", default="stockfish", help="Stockfish UCI executable")
    ap.add_argument("--depth", type=int, default=10, help="Stockfish analysis depth per position (0 disables teacher scores)")
    ap.add_argument("--limit-games", type=int, default=0, help="maximum imported games; 0 imports all")
    ap.add_argument("--max-plies", type=int, default=400, help="ignore moves after this ply")
    args = ap.parse_args()
    if args.depth < 0 or args.limit_games < 0 or args.max_plies < 1:
        ap.error("depth and limit-games must be nonnegative; max-plies must be positive")

    records = bytearray()
    cleaned = []
    imported = positions = scored = 0
    engine = None
    if args.depth:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
            if "Threads" in engine.options:
                engine.configure({"Threads": 1})
            if "Hash" in engine.options:
                engine.configure({"Hash": 128})
        except (OSError, chess.engine.EngineError) as exc:
            raise SystemExit(f"Could not start Stockfish ({args.stockfish}): {exc}")

    try:
        for pgn_path in args.pgn:
            with pgn_path.open("r", encoding="utf-8-sig", errors="replace") as src:
                while args.limit_games == 0 or imported < args.limit_games:
                    game = chess.pgn.read_game(src)
                    if game is None:
                        break
                    if game.errors:
                        print(f"Skipping malformed game in {pgn_path}: {game.errors[0]}", file=sys.stderr)
                        continue
                    result = game.headers.get("Result", "*")
                    if result not in ("1-0", "0-1", "1/2-1/2"):
                        print(f"Skipping game without a final result in {pgn_path}", file=sys.stderr)
                        continue

                    board = game.board()
                    game_records = bytearray()
                    for ply, move in enumerate(game.mainline_moves()):
                        if ply >= args.max_plies:
                            break
                        if move not in board.legal_moves:
                            print(f"Skipping illegal mainline in {pgn_path}", file=sys.stderr)
                            game_records.clear()
                            break
                        # Match Glory's self-play record policy: don't train on
                        # positions where the side to move is already in check.
                        if not board.is_check():
                            score_cp = SCORE_NONE
                            if engine is not None:
                                info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
                                cp = info["score"].pov(board.turn).score(mate_score=MAX_SCORE_CP)
                                cp = max(-MAX_SCORE_CP, min(MAX_SCORE_CP, cp if cp is not None else 0))
                                score_cp = cp
                                scored += 1
                            stm = 0 if board.turn == chess.WHITE else 1
                            outcome = result_for_side_to_move(result, board.turn)
                            game_records.extend(REC.pack(pack_board(board), stm, outcome, score_cp))
                        board.push(move)

                    if not game_records:
                        continue
                    records.extend(game_records)
                    positions += len(game_records) // REC.size
                    cleaned.append(clean_mainline(game))
                    imported += 1
                    print(f"Imported outsider game {imported}: {pgn_path.name}, "
                          f"{len(game_records) // REC.size} positions", flush=True)
    finally:
        if engine is not None:
            engine.quit()

    if imported == 0:
        raise SystemExit("No completed, valid PGN games were imported; outputs left unchanged.")

    out_dir = args.out_dir
    atomic_write(out_dir / "outsider.bin", bytes(records))
    # Human-readable, annotation-free mainlines for online replay/analysis.
    pgn_text = "\n".join(cleaned).encode("utf-8")
    atomic_write(out_dir / "outsider_moves.pgn", pgn_text)
    print(f"Imported {imported} outsider games and {positions} positions "
          f"({scored} Stockfish-labelled).\n"
          f"Records: {out_dir / 'outsider.bin'}\n"
          f"Move-only PGN: {out_dir / 'outsider_moves.pgn'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
