#!/usr/bin/env python3
"""Play Glory against Stockfish (or any UCI engine) and estimate a rating.

Setup (Arch):  python -m venv ~/venv && ~/venv/bin/pip install chess
Run:           ~/venv/bin/python match.py --dir /run/media/brpr/Glory --elo 1500 --games 40

Each side gets a fixed time per move. Colours alternate. Prints the running score and,
at the end, a rough Elo estimate relative to the opponent's UCI_Elo setting.
"""
import argparse, math, shlex, sys, chess, chess.engine, chess.pgn

ap = argparse.ArgumentParser()
ap.add_argument("--dir", help="Glory drive mount point (uses the latest net)")
ap.add_argument("--net", help="specific net file instead of --dir")
ap.add_argument("--glory", default="./glory_nnue", help="path to glory_nnue binary")
ap.add_argument("--opp", default="stockfish", help="opponent UCI engine command")
ap.add_argument("--elo", type=int, default=1500, help="opponent UCI_Elo (0 = full strength / no limit)")
ap.add_argument("--games", type=int, default=20)
ap.add_argument("--movetime", type=float, default=0.3, help="seconds per move, both sides")
ap.add_argument("--hash", type=int, default=64)
ap.add_argument("--pgn", default="glory_match.pgn")
a = ap.parse_args()

cmd = [a.glory, "--uci"] + (["--net", a.net] if a.net else ["--dir", a.dir or "."])
glory = chess.engine.SimpleEngine.popen_uci(cmd)
opp = chess.engine.SimpleEngine.popen_uci(shlex.split(a.opp))
glory.configure({"Hash": a.hash})
if a.elo > 0:
    opp.configure({"UCI_LimitStrength": True, "UCI_Elo": a.elo})

limit = chess.engine.Limit(time=a.movetime)
score = 0.0
w = d = l = 0
pgn_out = open(a.pgn, "w")
try:
    for g in range(a.games):
        glory_white = g % 2 == 0
        board = chess.Board()
        while not board.is_game_over(claim_draw=True) and board.ply() < 400:
            eng = glory if (board.turn == chess.WHITE) == glory_white else opp
            board.push(eng.play(board, limit).move)
        outcome = board.outcome(claim_draw=True)
        res = outcome.result() if outcome else "1/2-1/2"
        if res == "1/2-1/2" or res == "*": pts = 0.5
        else: pts = 1.0 if (res == "1-0") == glory_white else 0.0
        score += pts
        w += pts == 1.0; d += pts == 0.5; l += pts == 0.0
        game = chess.pgn.Game.from_board(board)
        game.headers["White"] = "Glory" if glory_white else "Opponent"
        game.headers["Black"] = "Opponent" if glory_white else "Glory"
        game.headers["Result"] = res
        print(game, file=pgn_out, end="\n\n"); pgn_out.flush()
        print(f"game {g+1:3d}/{a.games}: Glory {'W' if glory_white else 'B'} -> {pts:.1f}   "
              f"total +{w} ={d} -{l}  ({score:.1f}/{g+1})", flush=True)
except KeyboardInterrupt:
    pass
finally:
    glory.quit(); opp.quit()

n = w + d + l
if n:
    p = min(max(score / n, 0.5 / n), 1 - 0.5 / n)
    diff = -400 * math.log10(1 / p - 1)
    print(f"\nresult: +{w} ={d} -{l} over {n} games, score {score/n:.1%}")
    if a.elo > 0:
        print(f"rough estimate: ~{a.elo + diff:.0f} on Stockfish's UCI_Elo scale (very noisy with few games;"
              f" this is NOT a chess.com rating)")
