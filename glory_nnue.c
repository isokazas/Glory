/*
 * glory_nnue.c  --  a chess NNUE that learns from scratch by self-play (AlphaZero-style loop)
 *
 * What it does
 *   gen 0   : randomly initialised network (knows nothing, not even piece values)
 *   each gen: 1) self-play games (alpha-beta search + the network as evaluation)
 *                 - part of the games: current net vs itself
 *                 - part of the games: current net vs a sample of PAST versions
 *                   (so it cannot just overfit to beating itself)
 *             2) positions + search score + final game result are written to disk
 *             3) the net is trained on a replay window of the last N generations
 *             4) the new net is saved as the next version
 *   Everything (nets, game data, optimizer state, log) lives under --dir (your "Glory" drive).
 *   You can stop with Ctrl-C at any time and resume later; it carries on where it left off.
 *
 * Network (NNUE-style, "768 -> H x2 -> 1")
 *   inputs : 768 = {own/enemy} x {P,N,B,R,Q,K} x 64 squares, from each side's perspective
 *   layer 1: shared weights, two accumulators (side-to-move / other side), updated INCREMENTALLY
 *            on every move inside the search (the "efficiently updatable" part of NNUE)
 *   clipped ReLU [0,1], concatenate both sides, linear -> 1 output (logit; eval cp = 400*logit)
 *
 * Build:  gcc -O3 -march=native -ffast-math -pthread -o glory_nnue glory_nnue.c -lm
 * Run  :  ./glory_nnue --dir /path/to/Glory        (see run.sh, which finds/mounts the drive)
 * Test :  ./glory_nnue --perft                      (move generator self-check)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <signal.h>
#include <pthread.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <stdarg.h>
#include <poll.h>
#include <sys/wait.h>

/* ------------------------------------------------------------------ config */
#define H            128          /* hidden width per perspective */
#define NF           768          /* input features */
#define NP           (NF*H + H + 2*H + 1)
#define EVAL_SCALE   400.0f
#define EVAL_CLAMP   5000
#define MATE         30000
#define INF          32000
#define MAXPLY       64
#define MAX_GAME_PLIES 450
#define SCORE_NONE   32767
#define MAXOPP       8
#define MAXTHREADS   64

static int g_depth       = 3;     /* search depth for self-play */
static int g_rand_plies  = 4;     /* purely random opening plies (diversity) */
static int g_temp_plies  = 16;    /* afterwards: sample moves by softmax of search score */
static int g_temp_cp     = 120;   /* temperature in centipawns */

/* ------------------------------------------------------------------ basics */
typedef uint16_t Move;
enum { WHITE = 0, BLACK = 1 };
enum { PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING };
#define EMPTY 12

typedef struct { uint64_t s; } Rng;
static inline uint64_t rng_next(Rng *r) {
  uint64_t x = r->s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; r->s = x;
  return x * 2685821657736338717ULL;
}
static inline double rng_f(Rng *r) { return (rng_next(r) >> 11) * (1.0 / 9007199254740992.0); }
static inline int rng_int(Rng *r, int n) { return (int)(rng_next(r) % (uint64_t)n); }
static float rng_gauss(Rng *r) {
  double u = rng_f(r) + 1e-12, v = rng_f(r);
  return (float)(sqrt(-2.0 * log(u)) * cos(6.283185307179586 * v));
}

typedef struct { float W[NF * H]; float b[H]; float wo[2 * H]; float ob; } Net;
_Static_assert(sizeof(Net) == NP * sizeof(float), "Net layout");

static void *amalloc(size_t sz) {
  void *p = NULL;
  if (posix_memalign(&p, 64, sz)) { fprintf(stderr, "out of memory\n"); exit(1); }
  memset(p, 0, sz);
  return p;
}

typedef struct {
  uint8_t b[64];
  uint8_t stm, castle;       /* castle bits: 1=WK 2=WQ 4=BK 8=BQ */
  int8_t  ep;                /* en-passant target square or -1 */
  uint8_t halfmove;
  uint8_t ksq[2];
  uint64_t hash;
  float acc[2][H];           /* NNUE accumulators, one per perspective */
} Pos;

/* ------------------------------------------------------------------ tables */
static uint64_t zk[12][64], zc[16], zep[8], zside;
static int8_t knt[64][9], kgt[64][9];
static int8_t ray[64][8][8], raylen[64][8];
static uint8_t cmask[64];
static const int DF[8] = { 0, 1, 0, -1, 1, 1, -1, -1 };
static const int DR[8] = { 1, 0, -1, 0, 1, -1, -1, 1 };
static const int KNF[8] = { 1, 2, 2, 1, -1, -2, -2, -1 };
static const int KNR[8] = { 2, 1, -1, -2, -2, -1, 1, 2 };
static const int PV[6] = { 100, 320, 330, 500, 900, 0 };   /* only used for MOVE ORDERING */

static void init_tables(void) {
  Rng r = { 0x9E3779B97F4A7C15ULL };
  for (int i = 0; i < 12; i++) for (int s = 0; s < 64; s++) zk[i][s] = rng_next(&r);
  for (int i = 0; i < 16; i++) zc[i] = rng_next(&r);
  for (int i = 0; i < 8; i++) zep[i] = rng_next(&r);
  zside = rng_next(&r);
  for (int sq = 0; sq < 64; sq++) {
    int f = sq & 7, rk = sq >> 3, n = 0;
    for (int k = 0; k < 8; k++) {
      int nf = f + KNF[k], nr = rk + KNR[k];
      if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) knt[sq][n++] = nr * 8 + nf;
    }
    knt[sq][n] = -1; n = 0;
    for (int k = 0; k < 8; k++) {
      int nf = f + DF[k], nr = rk + DR[k];
      if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) kgt[sq][n++] = nr * 8 + nf;
    }
    kgt[sq][n] = -1;
    for (int d = 0; d < 8; d++) {
      int len = 0, nf = f + DF[d], nr = rk + DR[d];
      while (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) {
        ray[sq][d][len++] = nr * 8 + nf; nf += DF[d]; nr += DR[d];
      }
      raylen[sq][d] = len;
    }
  }
  memset(cmask, 15, sizeof cmask);
  cmask[4] = 12; cmask[0] = 13; cmask[7] = 14;
  cmask[60] = 3; cmask[56] = 7; cmask[63] = 11;
}

/* ------------------------------------------------------------------ NNUE features / accumulators */
static __thread const Net *tl_net;

static inline int feat(int persp, int pc, int sq) {
  int c = pc >= 6, t = pc - 6 * c;
  return (c != persp) * 384 + t * 64 + (persp == WHITE ? sq : (sq ^ 56));
}
static inline void put_piece(Pos *p, int sq, int pc) {
  p->b[sq] = (uint8_t)pc; p->hash ^= zk[pc][sq];
  if (pc % 6 == KING) p->ksq[pc / 6] = (uint8_t)sq;
  const float *w0 = tl_net->W + feat(0, pc, sq) * H, *w1 = tl_net->W + feat(1, pc, sq) * H;
  for (int i = 0; i < H; i++) { p->acc[0][i] += w0[i]; p->acc[1][i] += w1[i]; }
}
static inline void remove_piece(Pos *p, int sq) {
  int pc = p->b[sq];
  p->b[sq] = EMPTY; p->hash ^= zk[pc][sq];
  const float *w0 = tl_net->W + feat(0, pc, sq) * H, *w1 = tl_net->W + feat(1, pc, sq) * H;
  for (int i = 0; i < H; i++) { p->acc[0][i] -= w0[i]; p->acc[1][i] -= w1[i]; }
}
static void refresh(Pos *p) {
  for (int c = 0; c < 2; c++) memcpy(p->acc[c], tl_net->b, sizeof(float) * H);
  for (int sq = 0; sq < 64; sq++) {
    int pc = p->b[sq];
    if (pc == EMPTY) continue;
    const float *w0 = tl_net->W + feat(0, pc, sq) * H, *w1 = tl_net->W + feat(1, pc, sq) * H;
    for (int i = 0; i < H; i++) { p->acc[0][i] += w0[i]; p->acc[1][i] += w1[i]; }
  }
}
static inline float clamp01(float x) { return x < 0 ? 0 : (x > 1 ? 1 : x); }
static inline int evaluate(const Pos *p) {
  const Net *n = tl_net;
  const float *a = p->acc[p->stm], *o = p->acc[p->stm ^ 1];
  float s = n->ob;
  for (int i = 0; i < H; i++) s += n->wo[i] * clamp01(a[i]) + n->wo[H + i] * clamp01(o[i]);
  int v = (int)(s * EVAL_SCALE);
  return v > EVAL_CLAMP ? EVAL_CLAMP : (v < -EVAL_CLAMP ? -EVAL_CLAMP : v);
}

/* ------------------------------------------------------------------ position setup */
static uint64_t compute_hash(const Pos *p) {
  uint64_t h = 0;
  for (int s = 0; s < 64; s++) if (p->b[s] != EMPTY) h ^= zk[p->b[s]][s];
  h ^= zc[p->castle];
  if (p->ep >= 0) h ^= zep[p->ep & 7];
  if (p->stm == BLACK) h ^= zside;
  return h;
}
static void startpos(Pos *p) {
  static const int back[8] = { ROOK, KNIGHT, BISHOP, QUEEN, KING, BISHOP, KNIGHT, ROOK };
  memset(p, 0, sizeof *p);
  for (int i = 0; i < 64; i++) p->b[i] = EMPTY;
  for (int f = 0; f < 8; f++) {
    p->b[f] = back[f]; p->b[8 + f] = PAWN;
    p->b[48 + f] = 6 + PAWN; p->b[56 + f] = 6 + back[f];
  }
  p->ksq[0] = 4; p->ksq[1] = 60;
  p->stm = WHITE; p->castle = 15; p->ep = -1;
  p->hash = compute_hash(p);
  refresh(p);
}
static int set_fen(Pos *p, const char *fen) {
  static const char *PCH = "PNBRQKpnbrqk";
  memset(p, 0, sizeof *p);
  for (int i = 0; i < 64; i++) p->b[i] = EMPTY;
  int sq = 56; const char *c = fen;
  for (; *c && *c != ' '; c++) {
    if (*c == '/') sq -= 16;
    else if (isdigit((unsigned char)*c)) sq += *c - '0';
    else { const char *k = strchr(PCH, *c); if (!k) return 0; p->b[sq++] = (uint8_t)(k - PCH); }
  }
  for (int s = 0; s < 64; s++) if (p->b[s] != EMPTY && p->b[s] % 6 == KING) p->ksq[p->b[s] / 6] = (uint8_t)s;
  while (*c == ' ') c++;
  p->stm = (*c == 'b') ? BLACK : WHITE;
  while (*c && *c != ' ') c++;
  while (*c == ' ') c++;
  for (; *c && *c != ' '; c++) {
    { if (*c == 'K') p->castle |= 1; if (*c == 'Q') p->castle |= 2; }
    { if (*c == 'k') p->castle |= 4; if (*c == 'q') p->castle |= 8; }
  }
  while (*c == ' ') c++;
  p->ep = -1;
  if (*c >= 'a' && *c <= 'h' && c[1] >= '1' && c[1] <= '8') p->ep = (int8_t)((c[1] - '1') * 8 + (c[0] - 'a'));
  p->hash = compute_hash(p);
  refresh(p);
  return 1;
}

/* ------------------------------------------------------------------ move generation */
static int attacked(const Pos *p, int sq, int by) {
  int f = sq & 7, r = sq >> 3;
  if (by == WHITE) {
    if (r > 0) { if (f > 0 && p->b[sq - 9] == PAWN) return 1; if (f < 7 && p->b[sq - 7] == PAWN) return 1; }
  } else {
    if (r < 7) { if (f > 0 && p->b[sq + 7] == 6 + PAWN) return 1; if (f < 7 && p->b[sq + 9] == 6 + PAWN) return 1; }
  }
  int n = by * 6;
  for (const int8_t *q = knt[sq]; *q >= 0; q++) if (p->b[*q] == n + KNIGHT) return 1;
  for (const int8_t *q = kgt[sq]; *q >= 0; q++) if (p->b[*q] == n + KING) return 1;
  for (int d = 0; d < 8; d++) {
    int a = d < 4 ? ROOK : BISHOP;
    for (int k = 0; k < raylen[sq][d]; k++) {
      int pc = p->b[(int)ray[sq][d][k]];
      if (pc != EMPTY) { if (pc == n + a || pc == n + QUEEN) return 1; break; }
    }
  }
  return 0;
}

#define ADD(f, t, pr) (mv[n++] = (Move)((f) | ((t) << 6) | ((pr) << 12)))

static int gen_moves(const Pos *p, Move *mv, int caps_only) {
  int n = 0, us = p->stm, base = us * 6;
  for (int from = 0; from < 64; from++) {
    int pc = p->b[from];
    if (pc == EMPTY || pc / 6 != us) continue;
    int t = pc - base, f = from & 7, r = from >> 3;
    if (t == PAWN) {
      int dir = us == WHITE ? 8 : -8, prank = us == WHITE ? 6 : 1, srank = us == WHITE ? 1 : 6;
      int to = from + dir;
      if (p->b[to] == EMPTY) {
        if (r == prank) {
          ADD(from, to, QUEEN);
          if (!caps_only) { ADD(from, to, ROOK); ADD(from, to, BISHOP); ADD(from, to, KNIGHT); }
        } else if (!caps_only) {
          ADD(from, to, 0);
          if (r == srank && p->b[to + dir] == EMPTY) ADD(from, to + dir, 0);
        }
      }
      for (int df = -1; df <= 1; df += 2) {
        int nf = f + df;
        if (nf < 0 || nf > 7) continue;
        to = from + dir + df;
        int tp = p->b[to];
        if (tp != EMPTY && tp / 6 != us) {
          if (r == prank) {
            ADD(from, to, QUEEN);
            if (!caps_only) { ADD(from, to, ROOK); ADD(from, to, BISHOP); ADD(from, to, KNIGHT); }
          } else ADD(from, to, 0);
        } else if (tp == EMPTY && to == p->ep) ADD(from, to, 0);
      }
    } else if (t == KNIGHT || t == KING) {
      const int8_t *lst = t == KNIGHT ? knt[from] : kgt[from];
      for (; *lst >= 0; lst++) {
        int tp = p->b[*lst];
        if (tp == EMPTY) { if (!caps_only) ADD(from, *lst, 0); }
        else if (tp / 6 != us) ADD(from, *lst, 0);
      }
      if (t == KING && !caps_only) {
        int them = us ^ 1;
        if (us == WHITE && from == 4 && !attacked(p, 4, them)) {
          if ((p->castle & 1) && p->b[5] == EMPTY && p->b[6] == EMPTY && p->b[7] == ROOK &&
            !attacked(p, 5, them) && !attacked(p, 6, them)) ADD(4, 6, 0);
          if ((p->castle & 2) && p->b[3] == EMPTY && p->b[2] == EMPTY && p->b[1] == EMPTY && p->b[0] == ROOK &&
            !attacked(p, 3, them) && !attacked(p, 2, them)) ADD(4, 2, 0);
        } else if (us == BLACK && from == 60 && !attacked(p, 60, them)) {
          if ((p->castle & 4) && p->b[61] == EMPTY && p->b[62] == EMPTY && p->b[63] == 6 + ROOK &&
            !attacked(p, 61, them) && !attacked(p, 62, them)) ADD(60, 62, 0);
          if ((p->castle & 8) && p->b[59] == EMPTY && p->b[58] == EMPTY && p->b[57] == EMPTY && p->b[56] == 6 + ROOK &&
            !attacked(p, 59, them) && !attacked(p, 58, them)) ADD(60, 58, 0);
        }
      }
    } else {
      int d0 = t == BISHOP ? 4 : 0, d1 = t == ROOK ? 4 : 8;
      for (int d = d0; d < d1; d++)
        for (int k = 0; k < raylen[from][d]; k++) {
          int s = ray[from][d][k], tp = p->b[s];
          if (tp == EMPTY) { if (!caps_only) ADD(from, s, 0); }
          else { if (tp / 6 != us) ADD(from, s, 0); break; }
        }
    }
  }
  return n;
}

static void make_move(Pos *p, Move m) {
  int from = m & 63, to = (m >> 6) & 63, pr = m >> 12;
  int pc = p->b[from], us = p->stm, t = pc % 6, cap = p->b[to], oldep = p->ep;
  p->hash ^= zc[p->castle];
  if (oldep >= 0) p->hash ^= zep[oldep & 7];
  p->ep = -1; p->halfmove++;
  if (cap != EMPTY) { remove_piece(p, to); p->halfmove = 0; }
  if (t == PAWN) {
    p->halfmove = 0;
    if (to == oldep && cap == EMPTY && (from & 7) != (to & 7)) remove_piece(p, to + (us == WHITE ? -8 : 8));
    else if (abs(to - from) == 16) p->ep = (int8_t)((from + to) / 2);
  }
  remove_piece(p, from);
  put_piece(p, to, pr ? us * 6 + pr : pc);
  if (t == KING && abs(to - from) == 2) {
    int rf = to > from ? to + 1 : to - 2, rt = to > from ? to - 1 : to + 1;
    remove_piece(p, rf); put_piece(p, rt, us * 6 + ROOK);
  }
  p->castle &= cmask[from] & cmask[to];
  p->hash ^= zc[p->castle];
  if (p->ep >= 0) p->hash ^= zep[p->ep & 7];
  p->stm = (uint8_t)(us ^ 1); p->hash ^= zside;
}

static int legal_moves(const Pos *p, Move *out) {
  Move mv[320]; int n = gen_moves(p, mv, 0), k = 0, us = p->stm;
  for (int i = 0; i < n; i++) {
    Pos q = *p; make_move(&q, mv[i]);
    if (!attacked(&q, q.ksq[us], us ^ 1)) out[k++] = mv[i];
  }
  return k;
}
static uint64_t perft(const Pos *p, int d) {
  if (d == 0) return 1;
  Move mv[320]; int n = gen_moves(p, mv, 0), us = p->stm; uint64_t c = 0;
  for (int i = 0; i < n; i++) {
    Pos q = *p; make_move(&q, mv[i]);
    if (attacked(&q, q.ksq[us], us ^ 1)) continue;
    c += d == 1 ? 1 : perft(&q, d - 1);
  }
  return c;
}

/* ------------------------------------------------------------------ search */
static __thread uint64_t hist[1024];
static __thread int hist_n;
static __thread uint64_t tl_nodes;

static void order_moves(const Pos *p, Move *mv, int n) {
  int sc[320];
  for (int i = 0; i < n; i++) {
    int m = mv[i], from = m & 63, to = (m >> 6) & 63, pr = m >> 12, cap = p->b[to], s = 0;
    if (cap != EMPTY) s = 10000 + 10 * PV[cap % 6] - PV[p->b[from] % 6] / 10;
    else if (p->b[from] % 6 == PAWN && to == p->ep && (from & 7) != (to & 7)) s = 10990;
    if (pr) s += 9000 + PV[pr];
    sc[i] = s;
  }
  for (int i = 1; i < n; i++) {
    Move m = mv[i]; int s = sc[i], j = i - 1;
    while (j >= 0 && sc[j] < s) { mv[j + 1] = mv[j]; sc[j + 1] = sc[j]; j--; }
    mv[j + 1] = m; sc[j + 1] = s;
  }
}

static int qsearch(const Pos *p, int alpha, int beta, int ply) {
  tl_nodes++;

  if (p->halfmove >= 100) return 0;
  if (ply >= MAXPLY) return evaluate(p);

  int us = p->stm;
  int incheck = attacked(p, p->ksq[us], us ^ 1);
  int best = -INF;

  /* Never stand pat while in check: search every legal evasion. */
  if (incheck) {
    Move mv[320];
    int n = gen_moves(p, mv, 0);
    order_moves(p, mv, n);

    int legal = 0;
    for (int i = 0; i < n; i++) {
      Pos q = *p;
      make_move(&q, mv[i]);
      if (attacked(&q, q.ksq[us], us ^ 1)) continue;

      legal++;
      int score = -qsearch(&q, -beta, -alpha, ply + 1);
      if (score > best) best = score;
      if (score > alpha) alpha = score;
      if (alpha >= beta) break;
    }

    return legal ? best : -MATE + ply;
  }

  /* Outside check, use a normal stand-pat score and search captures. */
  best = evaluate(p);
  if (best >= beta) return best;
  if (best > alpha) alpha = best;

  Move mv[320];
  int n = gen_moves(p, mv, 1);
  order_moves(p, mv, n);

  for (int i = 0; i < n; i++) {
    Pos q = *p;
    make_move(&q, mv[i]);
    if (attacked(&q, q.ksq[us], us ^ 1)) continue;

    int score = -qsearch(&q, -beta, -alpha, ply + 1);
    if (score > best) best = score;
    if (score > alpha) alpha = score;
    if (alpha >= beta) break;
  }

  return best;
}

static int negamax(const Pos *p, int depth, int alpha, int beta, int ply) {
  if (p->halfmove >= 100) return 0;
  for (int k = 2; k <= p->halfmove && k <= hist_n - 1; k += 2)
    if (hist[hist_n - 1 - k] == p->hash) return 0;
    if (ply >= MAXPLY) return evaluate(p);
    int us = p->stm, incheck = attacked(p, p->ksq[us], us ^ 1);
  if (depth <= 0) { if (!incheck) return qsearch(p, alpha, beta, ply); depth = 1; }
  Move mv[320]; int n = gen_moves(p, mv, 0), legal = 0, best = -INF;
  order_moves(p, mv, n);
  for (int i = 0; i < n; i++) {
    Pos q = *p; make_move(&q, mv[i]);
    if (attacked(&q, q.ksq[us], us ^ 1)) continue;
    legal++;
    hist[hist_n++] = q.hash;
    int sc = -negamax(&q, depth - 1, -beta, -alpha, ply + 1);
    hist_n--;
    if (sc > best) { best = sc; if (sc > alpha) { alpha = sc; if (alpha >= beta) break; } }
  }
  if (!legal) return incheck ? -MATE + ply : 0;
  return best;
}

/* Search the root. temp>0: score every move with a full window and sample by softmax (exploration). */
static Move search_root(const Pos *p, Rng *rng, int temp, int *out_score) {
  Move mv[320], lm[320]; int ls[320], nl = 0, best = -INF, alpha = -INF; Move bm = 0;
  int n = gen_moves(p, mv, 0), us = p->stm;
  order_moves(p, mv, n);
  for (int i = 0; i < n; i++) {
    Pos q = *p; make_move(&q, mv[i]);
    if (attacked(&q, q.ksq[us], us ^ 1)) continue;
    hist[hist_n++] = q.hash;
    int sc = temp > 0 ? -negamax(&q, g_depth - 1, -INF, INF, 1)
    : -negamax(&q, g_depth - 1, -INF, -alpha, 1);
    hist_n--;
    lm[nl] = mv[i]; ls[nl] = sc; nl++;
    if (sc > best) { best = sc; bm = mv[i]; if (sc > alpha) alpha = sc; }
  }
  *out_score = best;
  if (temp > 0 && nl > 1) {
    double w[320], sum = 0;
    for (int i = 0; i < nl; i++) { w[i] = exp((double)(ls[i] - best) / temp); sum += w[i]; }
    double r = rng_f(rng) * sum;
    for (int i = 0; i < nl; i++) { r -= w[i]; if (r <= 0) return lm[i]; }
    return lm[nl - 1];
  }
  return bm;
}

/* ------------------------------------------------------------------ game records */
typedef struct { uint8_t pc[32]; uint8_t stm; int8_t result; int16_t score; } Rec;  /* 36 bytes */
typedef struct { Rec *v; size_t n, cap; } RecVec;
static void rv_push(RecVec *r, const Rec *x) {
  if (r->n == r->cap) {
    r->cap = r->cap ? r->cap * 2 : 4096;
    r->v = realloc(r->v, r->cap * sizeof(Rec));
    if (!r->v) { fprintf(stderr, "out of memory\n"); exit(1); }
  }
  r->v[r->n++] = *x;
}
static void pack_rec(const Pos *p, Rec *r) {
  memset(r, 0, sizeof *r);
  for (int sq = 0; sq < 64; sq++) {
    int pc = p->b[sq], nib = pc == EMPTY ? 0 : pc + 1;
    r->pc[sq >> 1] |= (uint8_t)(nib << ((sq & 1) * 4));
  }
  r->stm = p->stm;
}

static int insufficient(const Pos *p) {
  int minors = 0;
  for (int s = 0; s < 64; s++) {
    int pc = p->b[s]; if (pc == EMPTY) continue;
    int t = pc % 6;
    if (t == PAWN || t == ROOK || t == QUEEN) return 0;
    if (t == KNIGHT || t == BISHOP) minors++;
  }
  return minors <= 1;
}

/* Plays one game. result: +1 white wins, 0 draw, -1 black wins.
 * learn_mask: bit c set => net of colour c is the "learner" whose search scores are valid targets. */
static int play_game(const Net *nw, const Net *nb, int learn_mask, RecVec *out, Rng *rng, int *plies_out) {
  Pos pos; tl_net = nw; startpos(&pos);
  hist_n = 0; hist[hist_n++] = pos.hash;
  size_t first = out ? out->n : 0;
  int result = 0, ply;
  for (ply = 0; ply < MAX_GAME_PLIES; ply++) {
    if (pos.halfmove >= 100 || insufficient(&pos)) { result = 0; break; }
    int cnt = 0;
    for (int i = hist_n - 1; i >= 0; i -= 2) if (hist[i] == pos.hash) cnt++;
    if (cnt >= 3) { result = 0; break; }
    tl_net = pos.stm == WHITE ? nw : nb;
    refresh(&pos);
    int us = pos.stm, incheck = attacked(&pos, pos.ksq[us], us ^ 1);
    Move lm[320]; int nl = legal_moves(&pos, lm);
    if (nl == 0) { result = incheck ? (us == WHITE ? -1 : 1) : 0; break; }
    Move m; int score = 0;
    if (ply < g_rand_plies) m = lm[rng_int(rng, nl)];
    else {
      m = search_root(&pos, rng, ply < g_rand_plies + g_temp_plies ? g_temp_cp : 0, &score);
      if (out && !incheck) {
        Rec r; pack_rec(&pos, &r);
        int sc = score > MATE - 1000 ? MATE - 1000 : (score < -(MATE - 1000) ? -(MATE - 1000) : score);
        r.score = (learn_mask >> us & 1) ? (int16_t)sc : (int16_t)SCORE_NONE;
        rv_push(out, &r);
      }
    }
    make_move(&pos, m);
    hist[hist_n++] = pos.hash;
  }
  if (out) for (size_t i = first; i < out->n; i++) out->v[i].result = (int8_t)result;
  *plies_out = ply;
  return result;
}

/* ------------------------------------------------------------------ self-play workers */
typedef struct {
  const Net *cur; const Net **opps; int nopp;
  int ngames, record; double p_past; uint64_t seed;
  RecVec rv;
  long w[MAXOPP + 1], d[MAXOPP + 1], l[MAXOPP + 1];   /* idx0: self-play (white view); idx i: learner vs opp i */
  long plies, games, decisive;
} Job;

static void *selfplay_worker(void *arg) {
  Job *j = arg; Rng rng = { j->seed ? j->seed : 1 };
  for (int g = 0; g < j->ngames; g++) {
    int idx = 0;
    if (j->nopp > 0 && rng_f(&rng) < j->p_past) idx = 1 + rng_int(&rng, j->nopp);
    int lc = rng_int(&rng, 2);
    const Net *opn = idx ? j->opps[idx - 1] : j->cur;
    const Net *nw = lc == WHITE ? j->cur : opn, *nb = lc == WHITE ? opn : j->cur;
    int mask = idx == 0 ? 3 : (1 << lc), plies = 0;
    int res = play_game(nw, nb, mask, j->record ? &j->rv : NULL, &rng, &plies);
    int r = idx == 0 ? res : (lc == WHITE ? res : -res);
    if (r > 0) j->w[idx]++; else if (r < 0) j->l[idx]++; else j->d[idx]++;
    j->plies += plies; j->games++; if (res) j->decisive++;
  }
  return NULL;
}

static void run_games(Job *jobs, int nthreads, const Net *cur, const Net **opps, int nopp,
                      int ngames, int record, double p_past, uint64_t seed) {
  pthread_t th[MAXTHREADS];
  for (int t = 0; t < nthreads; t++) {
    memset(&jobs[t], 0, sizeof(Job));
    jobs[t].cur = cur; jobs[t].opps = opps; jobs[t].nopp = nopp;
    jobs[t].ngames = ngames / nthreads + (t < ngames % nthreads);
    jobs[t].record = record; jobs[t].p_past = p_past;
    jobs[t].seed = seed * 1000003ULL + (uint64_t)t * 7919ULL + 17;
  }
  for (int t = 0; t < nthreads; t++) pthread_create(&th[t], NULL, selfplay_worker, &jobs[t]);
  for (int t = 0; t < nthreads; t++) pthread_join(th[t], NULL);
                      }

                      /* ------------------------------------------------------------------ training */
                      static void decode(const Rec *r, int *fu, int *fo, int *n) {
                        int us = r->stm, k = 0;
                        for (int sq = 0; sq < 64; sq++) {
                          int nib = (r->pc[sq >> 1] >> ((sq & 1) * 4)) & 15;
                          if (!nib) continue;
                          int pc = nib - 1;
                          fu[k] = feat(us, pc, sq); fo[k] = feat(us ^ 1, pc, sq); k++;
                        }
                        *n = k;
                      }

                      static float sample_grad(const Net *net, const Rec *r, float lambda, float *g, float scale) {
                        int fu[40], fo[40], n; decode(r, fu, fo, &n);
                        float au[H], ao[H], tu[H], to[H];
                        memcpy(au, net->b, sizeof au); memcpy(ao, net->b, sizeof ao);
                        for (int k = 0; k < n; k++) {
                          const float *a = net->W + fu[k] * H, *b = net->W + fo[k] * H;
                          for (int i = 0; i < H; i++) { au[i] += a[i]; ao[i] += b[i]; }
                        }
                        float out = net->ob;
                        for (int i = 0; i < H; i++) out += net->wo[i] * clamp01(au[i]) + net->wo[H + i] * clamp01(ao[i]);
                        float p = 1.0f / (1.0f + expf(-out));
                        float res = (float)(r->stm == WHITE ? r->result : -r->result);
                        res = (res + 1.0f) * 0.5f;
                        float t = res;
                        if (r->score != SCORE_NONE) {
                          float s = 1.0f / (1.0f + expf(-(float)r->score / EVAL_SCALE));
                          t = (1.0f - lambda) * res + lambda * s;
                        }
                        float err = p - t, d = 2.0f * err * p * (1.0f - p) * scale;
                        float *gW = g, *gb = g + NF * H, *gwo = gb + H, *gob = gwo + 2 * H;
                        *gob += d;
                        for (int i = 0; i < H; i++) {
                          gwo[i] += d * clamp01(au[i]); gwo[H + i] += d * clamp01(ao[i]);
                          tu[i] = (au[i] > 0 && au[i] < 1) ? d * net->wo[i] : 0;
                          to[i] = (ao[i] > 0 && ao[i] < 1) ? d * net->wo[H + i] : 0;
                          gb[i] += tu[i] + to[i];
                        }
                        for (int k = 0; k < n; k++) {
                          float *a = gW + fu[k] * H, *b = gW + fo[k] * H;
                          for (int i = 0; i < H; i++) { a[i] += tu[i]; b[i] += to[i]; }
                        }
                        return err * err;
                      }

                      typedef struct {
                        const Net *net; const Rec *recs; const uint32_t *idx;
                        int lo, hi; float lambda, scale; float *g; double loss;
                      } GJob;
                      static void *grad_worker(void *arg) {
                        GJob *j = arg;
                        memset(j->g, 0, sizeof(float) * NP);
                        j->loss = 0;
                        for (int i = j->lo; i < j->hi; i++) j->loss += sample_grad(j->net, &j->recs[j->idx[i]], j->lambda, j->g, j->scale);
                        return NULL;
                      }

                      static double train(Net *net, float *m, float *v, long *adam_t, const Rec *recs, size_t nrec,
                                          int steps, int batch, float lr, float lambda, int nthreads, Rng *rng) {
                        if (nrec == 0) return 0;
                        float *gbuf = amalloc(sizeof(float) * NP * (size_t)nthreads), *gsum = amalloc(sizeof(float) * NP);
                        uint32_t *idx = malloc(sizeof(uint32_t) * (size_t)batch);
                        double total = 0; long cnt = 0;
                        float *w = (float *)net;
                        for (int s = 0; s < steps; s++) {
                          for (int i = 0; i < batch; i++) idx[i] = (uint32_t)(rng_next(rng) % nrec);
                          GJob jobs[MAXTHREADS]; pthread_t th[MAXTHREADS];
                          for (int t = 0; t < nthreads; t++) {
                            jobs[t] = (GJob){ net, recs, idx, batch * t / nthreads, batch * (t + 1) / nthreads,
                              lambda, 1.0f / batch, gbuf + (size_t)t * NP, 0 };
                              pthread_create(&th[t], NULL, grad_worker, &jobs[t]);
                          }
                          double loss = 0;
                          for (int t = 0; t < nthreads; t++) { pthread_join(th[t], NULL); loss += jobs[t].loss; }
                          for (size_t i = 0; i < NP; i++) {
                            float a = 0;
                            for (int t = 0; t < nthreads; t++) a += gbuf[(size_t)t * NP + i];
                            gsum[i] = a;
                          }
                          (*adam_t)++;
                          const float b1 = 0.9f, b2 = 0.999f;
                          float c1 = 1.0f - powf(b1, (float)*adam_t), c2 = 1.0f - powf(b2, (float)*adam_t);
                          for (size_t i = 0; i < NP; i++) {
                            m[i] = b1 * m[i] + (1 - b1) * gsum[i];
                            v[i] = b2 * v[i] + (1 - b2) * gsum[i] * gsum[i];
                            w[i] -= lr * (m[i] / c1) / (sqrtf(v[i] / c2) + 1e-8f);
                          }
                          total += loss / batch; cnt++;
                        }
                        free(gbuf); free(gsum); free(idx);
                        return total / (double)cnt;
                                          }

                                          /* ------------------------------------------------------------------ files */
                                          static char g_root[1024];
                                          static const char *P(char *buf, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
                                          static const char *P(char *buf, const char *fmt, ...) {
                                            va_list ap; va_start(ap, fmt);
                                            char rel[512]; vsnprintf(rel, sizeof rel, fmt, ap); va_end(ap);
                                            snprintf(buf, 1536, "%s/%s", g_root, rel);
                                            return buf;
                                          }
                                          static int write_atomic(const char *path, const void *a, size_t na, const void *b, size_t nb) {
                                            char tmp[1600]; snprintf(tmp, sizeof tmp, "%s.tmp", path);
                                            FILE *f = fopen(tmp, "wb"); if (!f) return 0;
                                            int ok = fwrite(a, 1, na, f) == na && (nb == 0 || fwrite(b, 1, nb, f) == nb);
                                            fflush(f); fsync(fileno(f)); ok &= fclose(f) == 0;
                                            return ok && rename(tmp, path) == 0;
                                          }
                                          typedef struct { char magic[4]; int32_t h, nf; } NetHdr;
                                          static int save_net(const char *path, const Net *net) {
                                            NetHdr h = { { 'G', 'N', 'N', 'U' }, H, NF };
                                            return write_atomic(path, &h, sizeof h, net, sizeof(Net));
                                          }
                                          static int load_net(const char *path, Net *net) {
                                            FILE *f = fopen(path, "rb"); if (!f) return 0;
                                            NetHdr h; int ok = fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "GNNU", 4) && h.h == H && h.nf == NF
                                            && fread(net, sizeof(Net), 1, f) == 1;
                                            fclose(f); return ok;
                                          }
                                          static Rec *load_replay(int gen, int window, size_t *nout) {
                                            Rec *all = NULL; size_t n = 0;
                                            for (int g = gen - window + 1 > 0 ? gen - window + 1 : 0; g <= gen; g++) {
                                              char buf[1600]; FILE *f = fopen(P(buf, "data/gen_%06d.bin", g), "rb");
                                              if (!f) continue;
                                              fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
                                              size_t k = (size_t)sz / sizeof(Rec);
                                              all = realloc(all, (n + k) * sizeof(Rec));
                                              n += fread(all + n, sizeof(Rec), k, f) == k ? k : 0;
                                              fclose(f);
                                            }
                                            *nout = n; return all;
                                          }
                                          static void init_net(Net *net, Rng *rng) {
                                            for (int i = 0; i < NF * H; i++) net->W[i] = 0.1f * rng_gauss(rng);
                                            for (int i = 0; i < H; i++) net->b[i] = 0;
                                            for (int i = 0; i < 2 * H; i++) net->wo[i] = 0.1f * rng_gauss(rng);
                                            net->ob = 0;
                                          }


                                          /* ------------------------------------------------------------------ UCI play mode (stronger search) */
                                          typedef struct { uint64_t key; int16_t score; int16_t depth; uint8_t flag; uint8_t pad; Move move; } TTE;  /* 16 bytes */
                                          static TTE *tt; static size_t tt_n;
                                          static Move killers[MAXPLY + 4][2];
                                          static int histh[2][64][64];
                                          static double u_start, u_soft, u_hard; static int u_stop, u_quit, u_nostdin, u_quiet; static uint64_t u_last, u_maxnodes;
                                          static char inbuf[1 << 16]; static int inlen;

                                          static double uci_now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

                                          /* returns 1 and fills out if a full line is available; block=1 waits for one; -1 on EOF */
                                          static int read_line(char *out, int block) {
                                            for (;;) {
                                              char *nl = memchr(inbuf, '\n', (size_t)inlen);
                                              if (nl) {
                                                int len = (int)(nl - inbuf);
                                                memcpy(out, inbuf, (size_t)len); out[len] = 0;
                                                if (len && out[len - 1] == '\r') out[len - 1] = 0;
                                                memmove(inbuf, nl + 1, (size_t)(inlen - len - 1)); inlen -= len + 1;
                                                return 1;
                                              }
                                              if (inlen >= (int)sizeof inbuf - 1) inlen = 0;
                                              struct pollfd pf = { 0, POLLIN, 0 };
                                              if (!block && poll(&pf, 1, 0) <= 0) return 0;
                                              ssize_t r = read(0, inbuf + inlen, sizeof inbuf - (size_t)inlen - 1);
                                              if (r <= 0) return -1;
                                              inlen += (int)r;
                                            }
                                          }

                                          static void u_check(void) {
                                            if (uci_now() - u_start > u_hard && u_hard > 0) u_stop = 1;
                                            if (u_maxnodes && tl_nodes > u_maxnodes) u_stop = 1;
                                            char line[512];
                                            int r;
                                            while (!u_nostdin && (r = read_line(line, 0)) != 0) {
                                              if (r < 0) { u_stop = 1; u_quit = 1; break; }
                                              if (!strncmp(line, "stop", 4)) u_stop = 1;
                                              else if (!strncmp(line, "quit", 4)) { u_stop = 1; u_quit = 1; }
                                              else if (!strncmp(line, "isready", 7)) { puts("readyok"); fflush(stdout); }
                                            }
                                          }

                                          static int has_np(const Pos *p, int c) {
                                            for (int s = 0; s < 64; s++) { int pc = p->b[s]; if (pc != EMPTY && pc / 6 == c && pc % 6 != PAWN && pc % 6 != KING) return 1; }
                                            return 0;
                                          }
                                          static int tt_adj_in(int s, int ply) { return s > MATE - 1000 ? s - ply : (s < -(MATE - 1000) ? s + ply : s); }
                                          static int tt_adj_out(int s, int ply) { return s > MATE - 1000 ? s + ply : (s < -(MATE - 1000) ? s - ply : s); }

                                          static int usearch(const Pos *p, int depth, int alpha, int beta, int ply, int nullok) {
                                            tl_nodes++;
                                            if (tl_nodes - u_last >= 2048) { u_last = tl_nodes; u_check(); }
                                            if (u_stop) return 0;
                                            if (ply > 0) {
                                              if (p->halfmove >= 100) return 0;
                                              for (int k = 2; k <= p->halfmove && k <= hist_n - 1; k += 2) if (hist[hist_n - 1 - k] == p->hash) return 0;
                                            }
                                            if (ply >= MAXPLY) return evaluate(p);
                                            int us = p->stm, incheck = attacked(p, p->ksq[us], us ^ 1);
                                            if (incheck && depth <= 0) depth = 1;
                                            if (depth <= 0) return qsearch(p, alpha, beta, ply);
                                            int pv = beta - alpha > 1;
                                            TTE *e = &tt[p->hash & (tt_n - 1)]; Move ttm = 0;
                                            if (e->key == p->hash) {
                                              ttm = e->move;
                                              if (ply > 0 && e->depth >= depth) {
                                                int sc = tt_adj_in(e->score, ply);
                                                if (e->flag == 0 || (e->flag == 1 && sc >= beta) || (e->flag == 2 && sc <= alpha)) return sc;
                                              }
                                            }
                                            if (!incheck && !pv && nullok && depth >= 3 && has_np(p, us) && evaluate(p) >= beta) {
                                              Pos q = *p; q.stm ^= 1; q.hash ^= zside;
                                              if (q.ep >= 0) { q.hash ^= zep[q.ep & 7]; q.ep = -1; }
                                              q.halfmove++;
                                              hist[hist_n++] = q.hash;
                                              int sc = -usearch(&q, depth - 3 - depth / 6, -beta, -beta + 1, ply + 1, 0);
                                              hist_n--;
                                              if (u_stop) return 0;
                                              if (sc >= beta) return sc >= MATE - 1000 ? beta : sc;
                                            }
                                            Move mv[320]; int n = gen_moves(p, mv, 0), sc[320];
                                            for (int i = 0; i < n; i++) {
                                              int m = mv[i], from = m & 63, to = (m >> 6) & 63, pr = m >> 12, cap = p->b[to], s;
                                              if (m == ttm) s = 1000000;
                                              else if (cap != EMPTY) s = 100000 + 10 * PV[cap % 6] - PV[p->b[from] % 6] / 10;
                                              else if (p->b[from] % 6 == PAWN && to == p->ep && (from & 7) != (to & 7)) s = 100990;
                                              else if (m == killers[ply][0]) s = 90000;
                                              else if (m == killers[ply][1]) s = 80000;
                                              else s = histh[us][from][to];
                                              if (pr) s += 95000 + PV[pr];
                                              sc[i] = s;
                                            }
                                            int best = -INF, legal = 0, oa = alpha; Move bestm = 0;
                                            for (int i = 0; i < n; i++) {
                                              int bi = i;
                                              for (int j = i + 1; j < n; j++) if (sc[j] > sc[bi]) bi = j;
                                              if (bi != i) { Move tm = mv[i]; mv[i] = mv[bi]; mv[bi] = tm; int ts = sc[i]; sc[i] = sc[bi]; sc[bi] = ts; }
                                              Move m = mv[i]; int from = m & 63, to = (m >> 6) & 63;
                                              Pos q = *p; make_move(&q, m);
                                              if (attacked(&q, q.ksq[us], us ^ 1)) continue;
                                              legal++;
                                              int quiet = p->b[to] == EMPTY && !(m >> 12) && !(p->b[from] % 6 == PAWN && to == p->ep && (from & 7) != (to & 7));
                                              hist[hist_n++] = q.hash;
                                              int s_, nd = depth - 1;
                                              if (legal == 1) s_ = -usearch(&q, nd, -beta, -alpha, ply + 1, 1);
                                              else {
                                                int red = 0;
                                                if (depth >= 3 && legal > 4 && quiet && !incheck && m != killers[ply][0] && m != killers[ply][1]
                                                  && !attacked(&q, q.ksq[us ^ 1], us)) red = 1 + (legal > 10);
                                                s_ = -usearch(&q, nd - red, -alpha - 1, -alpha, ply + 1, 1);
                                                if (s_ > alpha && red) s_ = -usearch(&q, nd, -alpha - 1, -alpha, ply + 1, 1);
                                                if (s_ > alpha && s_ < beta) s_ = -usearch(&q, nd, -beta, -alpha, ply + 1, 1);
                                              }
                                              hist_n--;
                                              if (u_stop) return 0;
                                              if (s_ > best) {
                                                best = s_; bestm = m;
                                                if (s_ > alpha) {
                                                  alpha = s_;
                                                  if (alpha >= beta) {
                                                    if (quiet) {
                                                      if (killers[ply][0] != m) { killers[ply][1] = killers[ply][0]; killers[ply][0] = m; }
                                                      int *h = &histh[us][from][to]; *h += depth * depth; if (*h > 70000) *h = 70000;
                                                    }
                                                    break;
                                                  }
                                                }
                                              }
                                            }
                                            if (!legal) return incheck ? -MATE + ply : 0;
                                            e->key = p->hash; e->score = (int16_t)tt_adj_out(best, ply); e->depth = (int16_t)depth;
                                            e->flag = best <= oa ? 2 : (best >= beta ? 1 : 0); e->move = bestm;
                                            return best;
                                          }

                                          static void move_str(Move m, char *o) {
                                            int from = m & 63, to = (m >> 6) & 63, pr = m >> 12;
                                            o[0] = (char)('a' + (from & 7)); o[1] = (char)('1' + (from >> 3));
                                            o[2] = (char)('a' + (to & 7)); o[3] = (char)('1' + (to >> 3));
                                            if (pr) { o[4] = "nbrq"[pr - 1]; o[5] = 0; } else o[4] = 0;
                                          }

                                          static Move uci_search(const Pos *root, int maxdepth) {
                                            Move rm[320]; int rs[320];
                                            int nr = legal_moves(root, rm), us = root->stm;
                                            if (nr == 0) return 0;
                                            order_moves(root, rm, nr);
                                            Move best = rm[0];
                                            u_stop = 0; u_last = tl_nodes = 0;
                                            if (nr == 1 && u_hard > 0) return best;
                                            for (int d = 1; d <= maxdepth && d < MAXPLY; d++) {
                                              int alpha = -INF, iterscore = -INF; Move iterbest = 0;
                                              for (int i = 0; i < nr; i++) {
                                                Pos q = *root; make_move(&q, rm[i]);
                                                hist[hist_n++] = q.hash;
                                                int sc;
                                                if (i == 0) sc = -usearch(&q, d - 1, -INF, INF, 1, 1);
                                                else {
                                                  sc = -usearch(&q, d - 1, -alpha - 1, -alpha, 1, 1);
                                                  if (!u_stop && sc > alpha) sc = -usearch(&q, d - 1, -INF, -alpha, 1, 1);
                                                }
                                                hist_n--;
                                                if (u_stop) break;
                                                rs[i] = sc;
                                                if (sc > iterscore) { iterscore = sc; iterbest = rm[i]; if (sc > alpha) alpha = sc; }
                                              }
                                              if (iterbest) best = iterbest;
                                              if (u_stop && !iterbest) break;
                                              if (iterbest && !u_quiet) {
                                                double el = uci_now() - u_start; char ms[8]; move_str(best, ms);
                                                if (iterscore > MATE - 1000 || iterscore < -(MATE - 1000))
                                                  printf("info depth %d score mate %d nodes %llu nps %llu time %d pv %s\n", d,
                                                         (iterscore > 0 ? (MATE - iterscore + 1) / 2 : -(MATE + iterscore + 1) / 2),
                                                         (unsigned long long)tl_nodes, (unsigned long long)(tl_nodes / (el > 1e-3 ? el : 1e-3)), (int)(el * 1000), ms);
                                                  else
                                                    printf("info depth %d score cp %d nodes %llu nps %llu time %d pv %s\n", d, iterscore,
                                                           (unsigned long long)tl_nodes, (unsigned long long)(tl_nodes / (el > 1e-3 ? el : 1e-3)), (int)(el * 1000), ms);
                                                    fflush(stdout);
                                              }
                                              if (u_stop) break;
                                              /* put best move first, order the rest by this iteration's scores */
                                              for (int i = 0; i < nr; i++) if (rm[i] == best) { Move t = rm[0]; rm[0] = rm[i]; rm[i] = t; int ts = rs[0]; rs[0] = rs[i]; rs[i] = ts; break; }
                                              for (int i = 2; i < nr; i++) {
                                                Move m = rm[i]; int s = rs[i], j = i - 1;
                                                while (j >= 1 && rs[j] < s) { rm[j + 1] = rm[j]; rs[j + 1] = rs[j]; j--; }
                                                rm[j + 1] = m; rs[j + 1] = s;
                                              }
                                              if (u_soft > 0 && uci_now() - u_start > u_soft * 0.5) break;
                                              if (iterscore > MATE - 100 || iterscore < -(MATE - 100)) break;
                                            }
                                            (void)us;
                                            return best;
                                          }

                                          static void tt_resize(int mb) {
                                            free(tt);
                                            size_t n = ((size_t)mb << 20) / sizeof(TTE), p2 = 1; while (p2 * 2 <= n) p2 *= 2;
                                            tt_n = p2; tt = calloc(tt_n, sizeof(TTE));
                                            if (!tt) { fprintf(stderr, "out of memory\n"); exit(1); }
                                          }

                                          /* ------------------------------------------------------------------ Stockfish ladder test */
                                          typedef struct { pid_t pid; FILE *to, *from; } Ext;
                                          static int ext_send(Ext *e, const char *fmt, ...) {
                                            va_list ap; va_start(ap, fmt); int r = vfprintf(e->to, fmt, ap); va_end(ap);
                                            fputc('\n', e->to); return r >= 0 && fflush(e->to) == 0;
                                          }
                                          static int ext_wait(Ext *e, const char *prefix, char *out, size_t cap) {
                                            while (fgets(out, (int)cap, e->from)) if (!strncmp(out, prefix, strlen(prefix))) return 1;
                                            return 0;
                                          }
                                          static int ext_start(Ext *e, const char *cmd, int elo) {
                                            int in[2], op[2]; memset(e, 0, sizeof *e);
                                            if (pipe(in) || pipe(op)) return 0;
                                            fflush(stdout);
                                            e->pid = fork();
                                            if (e->pid < 0) return 0;
                                            if (e->pid == 0) {
                                              dup2(in[0], 0); dup2(op[1], 1);
                                              close(in[0]); close(in[1]); close(op[0]); close(op[1]);
                                              execl("/bin/sh", "sh", "-c", cmd, (char *)NULL); _exit(127);
                                            }
                                            close(in[0]); close(op[1]);
                                            e->to = fdopen(in[1], "w"); e->from = fdopen(op[0], "r");
                                            char buf[512];
                                            if (!ext_send(e, "uci") || !ext_wait(e, "uciok", buf, sizeof buf)) return 0;
                                            ext_send(e, "setoption name Threads value 1");
                                            ext_send(e, "setoption name Hash value 64");
                                            ext_send(e, "setoption name UCI_LimitStrength value true");
                                            ext_send(e, "setoption name UCI_Elo value %d", elo);
                                            ext_send(e, "isready");
                                            return ext_wait(e, "readyok", buf, sizeof buf);
                                          }
                                          static void ext_stop(Ext *e) {
                                            if (e->to) { ext_send(e, "quit"); fclose(e->to); }
                                            if (e->from) fclose(e->from);
                                            if (e->pid > 0) waitpid(e->pid, NULL, 0);
                                          }

                                          /* Plays ngames (alternating colours) of `net` vs an external UCI engine limited to `elo`.
                                           * Both sides get `movetime` seconds per move. Returns Glory's points, or -1 if the engine failed. */
                                          static double sf_match(const Net *net, const char *cmd, int elo, int ngames, double movetime, int *wo, int *dr, int *lo) {
                                            Ext e; *wo = *dr = *lo = 0;
                                            if (!ext_start(&e, cmd, elo)) { ext_stop(&e); return -1; }
                                            tl_net = net; u_nostdin = u_quiet = 1;
                                            if (!tt) tt_resize(64);
                                            double pts = 0; char buf[1024];
                                            for (int g = 0; g < ngames; g++) {
                                              int gw = g % 2 == 0, res = 0;                      /* res: +1 white wins, -1 black wins */
                                              ext_send(&e, "ucinewgame");
                                              memset(tt, 0, tt_n * sizeof(TTE)); memset(killers, 0, sizeof killers); memset(histh, 0, sizeof histh);
                                              Pos pos; startpos(&pos); hist_n = 0; hist[hist_n++] = pos.hash;
                                              char mvs[4096] = ""; size_t ml = 0;
                                              for (int ply = 0; ply < 400; ply++) {
                                                if (pos.halfmove >= 100 || insufficient(&pos)) break;
                                                int cnt = 0; for (int i = hist_n - 1; i >= 0; i -= 2) if (hist[i] == pos.hash) cnt++;
                                                if (cnt >= 3) break;
                                                Move lm[320]; int nl = legal_moves(&pos, lm);
                                                if (nl == 0) { if (attacked(&pos, pos.ksq[pos.stm], pos.stm ^ 1)) res = pos.stm == WHITE ? -1 : 1; break; }
                                                Move m = 0;
                                                if ((pos.stm == WHITE) == gw) {
                                                  refresh(&pos);
                                                  u_hard = u_soft = movetime * 0.95; u_maxnodes = 0; u_start = uci_now();
                                                  m = uci_search(&pos, 63);
                                                } else {
                                                  ext_send(&e, "position startpos%s%s", ml ? " moves" : "", mvs);
                                                  ext_send(&e, "go movetime %d", (int)(movetime * 1000));
                                                  char ms[16] = "";
                                                  if (!ext_wait(&e, "bestmove", buf, sizeof buf) || sscanf(buf, "bestmove %15s", ms) != 1) { ext_stop(&e); return -1; }
                                                  for (int i = 0; i < nl; i++) { char t[8]; move_str(lm[i], t); if (!strcmp(t, ms)) { m = lm[i]; break; } }
                                                }
                                                if (!m) { ext_stop(&e); return -1; }
                                                make_move(&pos, m); hist[hist_n++] = pos.hash;
                                                char t[8]; move_str(m, t); ml += (size_t)snprintf(mvs + ml, sizeof mvs - ml, " %s", t);
                                              }
                                              double p = res == 0 ? 0.5 : ((res > 0) == gw ? 1.0 : 0.0);
                                              pts += p; if (p == 1.0) (*wo)++; else if (p == 0.0) (*lo)++; else (*dr)++;
                                              printf("  game %2d/%d: Glory %c -> %.1f   total +%d =%d -%d  (%.1f/%d)\n", g + 1, ngames, gw ? 'W' : 'B', p, *wo, *dr, *lo, pts, g + 1);
                                              fflush(stdout);
                                            }
                                            ext_stop(&e);
                                            return pts;
                                          }

                                          static int uci_main(const char *netfile) {
                                            Net *net = amalloc(sizeof(Net));
                                            if (!load_net(netfile, net)) { fprintf(stderr, "cannot load net %s\n", netfile); return 1; }
                                            tl_net = net;
                                            tt_resize(64);
                                            Pos pos; startpos(&pos); hist_n = 0; hist[hist_n++] = pos.hash;
                                            char line[4096];
                                            for (;;) {
                                              int r = read_line(line, 1);
                                              if (r <= 0) break;
                                              char *save = NULL, *tok = strtok_r(line, " \t", &save);
                                              if (!tok) continue;
                                              if (!strcmp(tok, "uci")) {
                                                puts("id name GloryNNUE"); puts("id author glory_nnue");
                                                puts("option name Hash type spin default 64 min 1 max 4096");
                                                puts("uciok");
                                              } else if (!strcmp(tok, "isready")) puts("readyok");
                                              else if (!strcmp(tok, "ucinewgame")) {
                                                memset(tt, 0, tt_n * sizeof(TTE)); memset(killers, 0, sizeof killers); memset(histh, 0, sizeof histh);
                                              } else if (!strcmp(tok, "setoption")) {
                                                char *a = strtok_r(NULL, " \t", &save), *nm = strtok_r(NULL, " \t", &save);
                                                if (a && nm && !strcasecmp(nm, "Hash")) {
                                                  strtok_r(NULL, " \t", &save);
                                                  char *v = strtok_r(NULL, " \t", &save);
                                                  if (v && atoi(v) >= 1) tt_resize(atoi(v) > 4096 ? 4096 : atoi(v));
                                                }
                                              } else if (!strcmp(tok, "position")) {
                                                char *t = strtok_r(NULL, " \t", &save);
                                                if (!t) continue;
                                                if (!strcmp(t, "startpos")) { startpos(&pos); t = strtok_r(NULL, " \t", &save); }
                                                else if (!strcmp(t, "fen")) {
                                                  char fen[256] = ""; int nf = 0;
                                                  while ((t = strtok_r(NULL, " \t", &save)) && strcmp(t, "moves")) {
                                                    if (nf < 6) { strcat(fen, t); strcat(fen, " "); nf++; }
                                                  }
                                                  set_fen(&pos, fen);
                                                }
                                                hist_n = 0; hist[hist_n++] = pos.hash;
                                                if (t && !strcmp(t, "moves")) {
                                                  while ((t = strtok_r(NULL, " \t", &save))) {
                                                    Move lm[320]; int n = legal_moves(&pos, lm);
                                                    for (int i = 0; i < n; i++) {
                                                      char ms[8]; move_str(lm[i], ms);
                                                      if (!strcmp(ms, t)) { make_move(&pos, lm[i]); hist[hist_n++] = pos.hash; break; }
                                                    }
                                                  }
                                                }
                                              } else if (!strcmp(tok, "go")) {
                                                int depth = 0, movetime = 0, wt = -1, bt = -1, wi = 0, bi = 0, mtg = 0, inf = 0; long long nodes = 0;
                                                char *t;
                                                while ((t = strtok_r(NULL, " \t", &save))) {
                                                  char *v;
                                                  if (!strcmp(t, "infinite")) inf = 1;
                                                  else if ((v = strtok_r(NULL, " \t", &save))) {
                                                    if (!strcmp(t, "depth")) depth = atoi(v);
                                                    else if (!strcmp(t, "movetime")) movetime = atoi(v);
                                                    else if (!strcmp(t, "wtime")) wt = atoi(v);
                                                    else if (!strcmp(t, "btime")) bt = atoi(v);
                                                    else if (!strcmp(t, "winc")) wi = atoi(v);
                                                    else if (!strcmp(t, "binc")) bi = atoi(v);
                                                    else if (!strcmp(t, "movestogo")) mtg = atoi(v);
                                                    else if (!strcmp(t, "nodes")) nodes = atoll(v);
                                                  }
                                                }
                                                u_soft = u_hard = 0; u_maxnodes = (uint64_t)nodes; u_start = uci_now();
                                                int maxd = depth ? depth : (movetime || wt >= 0 || bt >= 0 || inf || nodes ? 63 : 6);
                                                if (movetime) { u_hard = movetime / 1000.0 * 0.95; u_soft = u_hard; }
                                                else if (!depth && !inf && (wt >= 0 || bt >= 0)) {
                                                  double my = (pos.stm == WHITE ? wt : bt) / 1000.0, inc = (pos.stm == WHITE ? wi : bi) / 1000.0;
                                                  double alloc = my / (mtg > 0 ? mtg : 30) + inc * 0.75;
                                                  u_soft = alloc; u_hard = alloc * 4 < my * 0.6 ? alloc * 4 : my * 0.6;
                                                  if (u_hard < 0.01) u_hard = 0.01;
                                                }
                                                refresh(&pos);
                                                Move bm = uci_search(&pos, maxd);
                                                char ms[8];
                                                if (bm) { move_str(bm, ms); printf("bestmove %s\n", ms); } else puts("bestmove 0000");
                                                if (u_quit) { fflush(stdout); break; }
                                              } else if (!strcmp(tok, "quit")) break;
                                              fflush(stdout);
                                            }
                                            return 0;
                                          }

                                          /* ------------------------------------------------------------------ main */
                                          static volatile sig_atomic_t g_stop = 0;
                                          static void on_sig(int s) { (void)s; g_stop = 1; }
                                          static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

                                          static int pick_opps(int gen, int *ids, Rng *rng) {      /* past versions are 0..gen-1 */
                                            if (gen == 0) return 0;
                                            int n = 0; ids[n++] = gen - 1;
                                            if (gen - 50 >= 0 && gen - 50 != gen - 1) ids[n++] = gen - 50;
                                            if (gen - 100 >= 0 && gen - 100 != gen - 50) ids[n++] = gen - 100;                         /* always the previous version */
                                              for (int tries = 0; n < MAXOPP && n < gen && tries < 100; tries++) {
                                                int lo = (rng_int(rng, 2) && gen > 11) ? gen - 11 : 0;  /* half recent, half from all history */
                                                int c = lo + rng_int(rng, gen - 1 - lo > 0 ? gen - 1 - lo : 1);
                                                int dup = 0; for (int i = 0; i < n; i++) if (ids[i] == c) dup = 1;
                                                if (!dup && c < gen) ids[n++] = c;
                                              }
                                              return n;
                                          }

                                          static void usage(void) {
                                            puts("usage: glory_nnue --dir PATH [options]\n"
                                            "  --dir PATH         where everything is stored (your 'Glory' drive)\n"
                                            "  --threads N        worker threads (default: all cores)\n"
                                            "  --games N          self-play games per generation (default 128)\n"
                                            "  --depth N          search depth during self-play (default 3)\n"
                                            "  --auto-depth       raise the depth automatically when progress at the current depth plateaus\n"
                                            "  --min-depth N      starting depth for --auto-depth (default 2)\n"
                                            "  --max-depth N      never go deeper than this (default 4)\n"
                                            "  --promo-min-gens N min generations at a depth before it may be raised (default 30)\n"
                                            "  --promo-window N   generations averaged for the vs-past score (default 10)\n"
                                            "  --promo-bench F    required bench score vs the gen-50/gen-100 nets to raise depth (default 0.65)\n"
                                            "  --promo-past F     required avg score vs past nets to raise depth (default 0.55)\n"
                                            "  --rand-plies N     random opening plies for diversity (default 4)\n"
                                            "  --p-past F         fraction of games played against past versions (default 0.5)\n"
                                            "  --steps N          training minibatches per generation (default 250)\n"
                                            "  --batch N          minibatch size (default 2048)\n"
                                            "  --lr F             Adam learning rate (default 0.001)\n"
                                            "  --window N         replay window, in generations (default 12)\n"
                                            "  --lambda-max F     max weight of search-score target vs game result (default 0.5)\n"
                                            "  --bench-every N    every N gens play 40 games vs the nets 50 and 100 gens back (default 5, 0=off)\n"
                                            "  --sf-every N       every N gens play Stockfish at the current ladder level (default 30, 0=off)\n"
                                            "  --sf-games N       games per Stockfish test (default 20)\n"
                                            "  --sf-pass F        score fraction needed to climb a level (default 0.70 = 14/20)\n"
                                            "  --sf-elo N         starting UCI_Elo (default 1320; the saved level wins on resume)\n"
                                            "  --sf-step N        Elo added after each pass (default 100)\n"
                                            "  --sf-max N         top of the ladder (default 3190)\n"
                                            "  --sf-movetime F    seconds per move in the test (default 0.3)\n"
                                            "  --sf-cmd CMD       Stockfish command (default stockfish)\n"
                                            "  --max-gens N       stop after N generations (default: run until Ctrl-C)\n"
                                            "  --seed N           random seed\n"
                                            "  --no-mount-check   don't verify that --dir is a separate mounted filesystem\n"
                                            "  --uci              play mode: speak the UCI protocol (use with --dir or --net FILE)\n"
                                            "  --net FILE         net file to use with --uci (default: latest net in --dir)\n"
                                            "  --perft            run move generator tests and exit");
                                          }

                                          int main(int argc, char **argv) {
                                            const char *dir = NULL, *netfile = NULL; int uci = 0;
                                            int nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN), games = 128, steps = 250, batch = 2048, window = 12;
                                            int bench_every = 5, max_gens = 0, do_perft = 0, mount_check = 1;
                                            int sf_every = 30, sf_games = 20, sf_elo = 1320, sf_step = 100, sf_max = 3190, sf_last = -1; double sf_pass = 0.70, sf_mt = 0.3; const char *sf_cmd = "stockfish";
                                            int auto_depth = 0, min_depth = 2, max_depth = 4, promo_min_gens = 30, promo_window = 10;
                                            float promo_bench = 0.65f, promo_past = 0.55f; int depth_given = 0;
                                            float lr = 1e-3f, lambda_max = 0.5f; double p_past = 0.5; uint64_t seed = (uint64_t)time(NULL);
                                            for (int i = 1; i < argc; i++) {
                                              #define ARG(name) (!strcmp(argv[i], name) && i + 1 < argc)
                                              if (ARG("--dir")) dir = argv[++i];
                                              else if (ARG("--net")) netfile = argv[++i];
                                              else if (!strcmp(argv[i], "--uci")) uci = 1;
                                              else if (ARG("--threads")) nthreads = atoi(argv[++i]);
                                              else if (ARG("--games")) games = atoi(argv[++i]);
                                              else if (ARG("--depth")) { g_depth = atoi(argv[++i]); depth_given = 1; }
                                              else if (!strcmp(argv[i], "--auto-depth")) auto_depth = 1;
                                              else if (ARG("--min-depth")) min_depth = atoi(argv[++i]);
                                              else if (ARG("--max-depth")) max_depth = atoi(argv[++i]);
                                              else if (ARG("--promo-min-gens")) promo_min_gens = atoi(argv[++i]);
                                              else if (ARG("--promo-window")) promo_window = atoi(argv[++i]);
                                              else if (ARG("--promo-bench")) promo_bench = (float)atof(argv[++i]);
                                              else if (ARG("--promo-past")) promo_past = (float)atof(argv[++i]);
                                              else if (ARG("--rand-plies")) g_rand_plies = atoi(argv[++i]);
                                              else if (ARG("--p-past")) p_past = atof(argv[++i]);
                                              else if (ARG("--steps")) steps = atoi(argv[++i]);
                                              else if (ARG("--batch")) batch = atoi(argv[++i]);
                                              else if (ARG("--lr")) lr = (float)atof(argv[++i]);
                                              else if (ARG("--window")) window = atoi(argv[++i]);
                                              else if (ARG("--lambda-max")) lambda_max = (float)atof(argv[++i]);
                                              else if (ARG("--bench-every")) bench_every = atoi(argv[++i]);
                                              else if (ARG("--sf-every")) sf_every = atoi(argv[++i]);
                                              else if (ARG("--sf-games")) sf_games = atoi(argv[++i]);
                                              else if (ARG("--sf-pass")) sf_pass = atof(argv[++i]);
                                              else if (ARG("--sf-elo")) sf_elo = atoi(argv[++i]);
                                              else if (ARG("--sf-step")) sf_step = atoi(argv[++i]);
                                              else if (ARG("--sf-max")) sf_max = atoi(argv[++i]);
                                              else if (ARG("--sf-movetime")) sf_mt = atof(argv[++i]);
                                              else if (ARG("--sf-cmd")) sf_cmd = argv[++i];
                                              else if (ARG("--max-gens")) max_gens = atoi(argv[++i]);
                                              else if (ARG("--seed")) seed = strtoull(argv[++i], NULL, 10);
                                              else if (!strcmp(argv[i], "--no-mount-check")) mount_check = 0;
                                              else if (!strcmp(argv[i], "--perft")) do_perft = 1;
                                              else { usage(); return 1; }
                                            }
                                            if (nthreads < 1) nthreads = 1;
                                            if (nthreads > MAXTHREADS) nthreads = MAXTHREADS;
                                            init_tables();
                                            Rng rng = { seed * 2654435761ULL + 88172645463325252ULL };

                                            if (uci) {
                                              char pb[1600];
                                              if (!netfile && dir) {
                                                snprintf(g_root, sizeof g_root, "%s/glory_nnue", dir);
                                                int g = 0; FILE *f = fopen(P(pb, "state.txt"), "r");
                                                if (!f || fscanf(f, "%d", &g) < 1) { fprintf(stderr, "no trained net found in %s\n", g_root); return 1; }
                                                fclose(f); netfile = P(pb, "nets/net_%06d.bin", g);
                                              }
                                              if (!netfile) { fprintf(stderr, "--uci needs --dir or --net\n"); return 1; }
                                              return uci_main(netfile);
                                            }
                                            if (do_perft) {
                                              Net *tn = amalloc(sizeof(Net)); init_net(tn, &rng); tl_net = tn;
                                              struct { const char *fen; int d; uint64_t exp; } T[] = {
                                                { "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281 },
                                                { "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862 },
                                                { "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 4, 43238 },
                                                { "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467 },
                                                { "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379 },
                                              };
                                              int bad = 0;
                                              for (size_t i = 0; i < sizeof T / sizeof T[0]; i++) {
                                                Pos p; set_fen(&p, T[i].fen);
                                                uint64_t got = perft(&p, T[i].d);
                                                printf("perft(%d) = %llu  expected %llu  %s\n", T[i].d, (unsigned long long)got,
                                                       (unsigned long long)T[i].exp, got == T[i].exp ? "OK" : "FAIL");
                                                bad |= got != T[i].exp;
                                              }
                                              /* incremental accumulator + hash consistency check */
                                              Pos p; startpos(&p); Rng r2 = { 12345 }; int ok = 1;
                                              for (int ply = 0; ply < 200 && ok; ply++) {
                                                Move lm[320]; int n = legal_moves(&p, lm); if (!n) break;
                                                make_move(&p, lm[rng_int(&r2, n)]);
                                                Pos q = p; refresh(&q);
                                                for (int c = 0; c < 2; c++) for (int i = 0; i < H; i++) if (fabsf(q.acc[c][i] - p.acc[c][i]) > 1e-3f) ok = 0;
                                                if (compute_hash(&p) != p.hash) ok = 0;
                                              }
                                              printf("incremental accumulator/hash check: %s\n", ok ? "OK" : "FAIL");
                                              return bad || !ok;
                                            }

                                            if (!dir) { usage(); return 1; }
                                            snprintf(g_root, sizeof g_root, "%s", dir);
                                            size_t L = strlen(g_root); while (L > 1 && g_root[L - 1] == '/') g_root[--L] = 0;
                                            struct stat st, sp;
                                            if (stat(g_root, &st) || !S_ISDIR(st.st_mode)) {
                                              fprintf(stderr, "error: %s does not exist. Is the Glory drive mounted?\n", g_root); return 1;
                                            }
                                            if (mount_check) {
                                              char par[1100]; snprintf(par, sizeof par, "%s/..", g_root);
                                              if (stat(par, &sp) == 0 && sp.st_dev == st.st_dev) {
                                                fprintf(stderr, "error: %s is on the same filesystem as its parent, so the drive does not look mounted.\n"
                                                "       (refusing to write to your system disk; use --no-mount-check to override)\n", g_root);
                                                return 1;
                                              }
                                            }
                                            char b1[1600], b2[1600];
                                            snprintf(g_root + strlen(g_root), sizeof g_root - strlen(g_root), "/glory_nnue");
                                            mkdir(g_root, 0755);
                                            mkdir(P(b1, "nets"), 0755); mkdir(P(b1, "data"), 0755);
                                            if (stat(P(b1, "nets"), &st)) { fprintf(stderr, "error: cannot create directories under %s: %s\n", g_root, strerror(errno)); return 1; }

                                            signal(SIGINT, on_sig); signal(SIGTERM, on_sig); signal(SIGPIPE, SIG_IGN);

                                            Net *cur = amalloc(sizeof(Net));
                                            float *am = amalloc(sizeof(float) * NP), *av = amalloc(sizeof(float) * NP);
                                            long adam_t = 0; int gen = 0;
                                            if (auto_depth && !depth_given) g_depth = min_depth;
                                            if (promo_window > 200) promo_window = 200;
                                            double win_s[200], win_n[200]; int win_c = 0, gens_at_depth = 0; double last_bench = -1;
                                            FILE *sf = fopen(P(b1, "state.txt"), "r");
                                            if (sf) {
                                              int sd = 0;
                                              int sfe = 0; if (fscanf(sf, "%d %d %d", &gen, &sd, &sfe) < 1) gen = 0; if (sfe >= 1000) sf_elo = sfe;
                                              if (auto_depth && !depth_given) g_depth = sd >= min_depth ? sd : min_depth;
                                              fclose(sf);
                                              if (!load_net(P(b1, "nets/net_%06d.bin", gen), cur)) { fprintf(stderr, "error: cannot load net for gen %d\n", gen); return 1; }
                                              FILE *af = fopen(P(b1, "adam.bin"), "rb");
                                              if (af) {
                                                if (fread(&adam_t, sizeof adam_t, 1, af) != 1 || fread(am, sizeof(float), NP, af) != NP || fread(av, sizeof(float), NP, af) != NP)
                                                { adam_t = 0; memset(am, 0, sizeof(float) * NP); memset(av, 0, sizeof(float) * NP); }
                                                fclose(af);
                                              }
                                              printf("resuming at generation %d (data dir: %s)\n", gen, g_root);
                                            } else {
                                              init_net(cur, &rng);
                                              if (!save_net(P(b1, "nets/net_%06d.bin", 0), cur)) { fprintf(stderr, "error: cannot write to %s\n", g_root); return 1; }
                                              printf("new run: generation 0 = untrained random network (data dir: %s)\n", g_root);
                                            }
                                            if (auto_depth) printf("auto-depth ON: min %d, max %d (raise when vs older nets >= %.0f%% and avg vs past >= %.0f%%, after >= %d gens)\n",
                                              min_depth, max_depth, promo_bench * 100, promo_past * 100, promo_min_gens);
                                            printf("threads=%d games/gen=%d depth=%d steps=%d batch=%d window=%d lr=%g  (Ctrl-C stops after the current generation)\n",
                                                   nthreads, games, g_depth, steps, batch, window, lr);

                                            FILE *lf = fopen(P(b1, "log.csv"), "a");
                                            if (lf && ftell(lf) == 0) fprintf(lf, "gen,positions,selfplay_wdl_white,vs_past_W,vs_past_D,vs_past_L,vs_past_score,bench_vs_gen0,avg_plies,loss,lambda,seconds\n");

                                            int done = 0;
                                            while (!g_stop && (max_gens == 0 || done < max_gens)) {
                                              double t0 = now();
                                              /* ---- opponents: sample of past versions ---- */
                                              int ids[MAXOPP], nopp = pick_opps(gen, ids, &rng);
                                              Net *oppnets[MAXOPP]; const Net *opps[MAXOPP];
                                              for (int i = 0; i < nopp; i++) {
                                                oppnets[i] = amalloc(sizeof(Net));
                                                if (!load_net(P(b1, "nets/net_%06d.bin", ids[i]), oppnets[i])) { fprintf(stderr, "cannot load past net %d\n", ids[i]); return 1; }
                                                opps[i] = oppnets[i];
                                              }
                                              /* ---- self-play ---- */
                                              Job jobs[MAXTHREADS];
                                              run_games(jobs, nthreads, cur, opps, nopp, games, 1, p_past, rng_next(&rng));
                                              long sw = 0, sd = 0, sl = 0, pw = 0, pd = 0, pl = 0, plies = 0, ng = 0;
                                              RecVec all = { 0 };
                                              for (int t = 0; t < nthreads; t++) {
                                                sw += jobs[t].w[0]; sd += jobs[t].d[0]; sl += jobs[t].l[0];
                                                for (int i = 1; i <= nopp; i++) { pw += jobs[t].w[i]; pd += jobs[t].d[i]; pl += jobs[t].l[i]; }
                                                plies += jobs[t].plies; ng += jobs[t].games;
                                                for (size_t k = 0; k < jobs[t].rv.n; k++) rv_push(&all, &jobs[t].rv.v[k]);
                                                free(jobs[t].rv.v);
                                              }
                                              if (!write_atomic(P(b1, "data/gen_%06d.bin", gen), all.v, all.n * sizeof(Rec), NULL, 0)) {
                                                fprintf(stderr, "error: could not write game data (drive full / unplugged?)\n"); return 1;
                                              }
                                              size_t npos = all.n; free(all.v);
                                              for (int i = 0; i < nopp; i++) free(oppnets[i]);

                                              /* ---- optional benchmark against the untrained gen-0 net ---- */
                                              double bench = -1;
                                              if (bench_every > 0 && gen > 0 && gen % bench_every == 0) {
                                                int r1 = gen > 50 ? gen - 50 : 0, r2 = gen > 100 ? gen - 100 : 0, nb = r1 == r2 ? 1 : 2;
                                                Net *bn[2] = { amalloc(sizeof(Net)), amalloc(sizeof(Net)) };
                                                if (!load_net(P(b1, "nets/net_%06d.bin", r1), bn[0]) || (nb == 2 && !load_net(P(b1, "nets/net_%06d.bin", r2), bn[1]))) { fprintf(stderr, "cannot load benchmark net\n"); return 1; }
                                                const Net *o0[2] = { bn[0], bn[1] }; Job bj[MAXTHREADS];
                                                run_games(bj, nthreads, cur, o0, nb, 40, 0, 1.0, rng_next(&rng));
                                                long bw = 0, bd = 0, bl = 0;
                                                for (int t = 0; t < nthreads; t++) for (int i = 1; i <= nb; i++) { bw += bj[t].w[i]; bd += bj[t].d[i]; bl += bj[t].l[i]; }
                                                bench = (bw + 0.5 * bd) / (double)(bw + bd + bl);
                                                free(bn[0]); free(bn[1]);
                                              }

                                              /* ---- train on replay window ---- */
                                              size_t nrec; Rec *replay = load_replay(gen, window, &nrec);
                                              float lambda = fminf(lambda_max, 0.01f * (float)gen);
                                              double loss = train(cur, am, av, &adam_t, replay, nrec, steps, batch, lr, lambda, nthreads, &rng);
                                              free(replay);

                                              /* ---- automatic depth progression ---- */
                                              if (auto_depth) {
                                                gens_at_depth++;
                                                if (bench >= 0) last_bench = bench;
                                                if (pw + pd + pl > 0) {
                                                  if (win_c == promo_window) { memmove(win_s, win_s + 1, sizeof(double) * (promo_window - 1)); memmove(win_n, win_n + 1, sizeof(double) * (promo_window - 1)); win_c--; }
                                                  win_s[win_c] = pw + 0.5 * pd; win_n[win_c] = (double)(pw + pd + pl); win_c++;
                                                }
                                                if (g_depth < max_depth && gens_at_depth >= promo_min_gens && win_c >= promo_window && last_bench >= promo_bench) {
                                                  double s_ = 0, n_ = 0; for (int i = 0; i < win_c; i++) { s_ += win_s[i]; n_ += win_n[i]; }
                                                  if (n_ > 0 && s_ / n_ >= promo_past) {
                                                    g_depth++; gens_at_depth = 0; win_c = 0; last_bench = -1;
                                                    printf("*** plateau reached: raising search depth to %d\n", g_depth);
                                                  }
                                                }
                                              }

                                              /* ---- save next version ---- */
                                              gen++; done++;
                                              int ok = save_net(P(b1, "nets/net_%06d.bin", gen), cur);
                                              FILE *af = fopen(P(b2, "adam.bin.tmp"), "wb");
                                              if (af) {
                                                fwrite(&adam_t, sizeof adam_t, 1, af); fwrite(am, sizeof(float), NP, af); fwrite(av, sizeof(float), NP, af);
                                                fflush(af); fsync(fileno(af)); fclose(af);
                                                rename(b2, P(b1, "adam.bin"));
                                              }
                                              char tmp[64]; snprintf(tmp, sizeof tmp, "%d %d %d\n", gen, g_depth, sf_elo);
                                              ok &= write_atomic(P(b1, "state.txt"), tmp, strlen(tmp), NULL, 0);
                                              if (!ok) { fprintf(stderr, "error: failed to save checkpoint\n"); return 1; }

                                              double secs = now() - t0, vs = (pw + pd + pl) ? (pw + 0.5 * pd) / (double)(pw + pd + pl) : -1;
                                              printf("gen %4d | self W/D/B %ld/%ld/%ld", gen - 1, sw, sd, sl);
                                              if (nopp) printf(" | vs %d past nets: +%ld =%ld -%ld (%.0f%%)", nopp, pw, pd, pl, vs * 100);
                                              if (bench >= 0) printf(" | vs gen-50/100: %.0f%%", bench * 100);
                                              printf(" | %zu pos, %.0f plies/game | loss %.4f lam %.2f | %.0fs\n", npos, ng ? (double)plies / ng : 0, loss, lambda, secs);
                                              fflush(stdout);
                                              if (lf) {
                                                fprintf(lf, "%d,%zu,%ld/%ld/%ld,%ld,%ld,%ld,%.3f,%.3f,%.1f,%.4f,%.2f,%.0f\n", gen - 1, npos, sw, sd, sl, pw, pd, pl,
                                                        vs, bench, ng ? (double)plies / ng : 0, loss, lambda, secs);
                                                fflush(lf);
                                              }
                                              /* ---- Stockfish ladder test (net `gen` = the newest saved one) ---- */
                                              if (sf_every > 0 && gen > 0 && gen % sf_every == 0 && sf_last != gen) {
                                                int e_ = sf_elo < 1320 ? 1320 : sf_elo, w_ = 0, d_ = 0, l_ = 0;
                                                printf("stockfish test at gen %d: UCI_Elo %d, %d games ...\n", gen, e_, sf_games); fflush(stdout);
                                                double pts = sf_match(cur, sf_cmd, e_, sf_games, sf_mt, &w_, &d_, &l_);
                                                sf_last = gen;
                                                if (pts < 0) printf("stockfish test failed (could not run '%s'); skipping\n", sf_cmd);
                                                else {
                                                  int pass = pts >= sf_pass * sf_games - 1e-9;
                                                  printf("stockfish test: Glory %.1f/%d (+%d =%d -%d) vs UCI_Elo %d -> %s\n", pts, sf_games, w_, d_, l_, e_, pass ? "PASS" : "not yet");
                                                  FILE *sl = fopen(P(b1, "sf_log.csv"), "a");
                                                  if (sl) { fprintf(sl, "%d,%d,%.1f,%d,%d,%d,%d,%d\n", gen, e_, pts, w_, d_, l_, sf_games, pass); fclose(sl); }
                                                  if (pass && e_ + sf_step <= sf_max) { sf_elo = e_ + sf_step; printf("*** passed: next Stockfish level is UCI_Elo %d\n", sf_elo); }
                                                  char st[64]; snprintf(st, sizeof st, "%d %d %d\n", gen, g_depth, sf_elo);
                                                  write_atomic(P(b1, "state.txt"), st, strlen(st), NULL, 0);
                                                }
                                                fflush(stdout);
                                              }

                                            }
                                            printf("stopped cleanly at generation %d. Run again with the same --dir to continue.\n", gen);
                                            if (lf) fclose(lf);
                                            return 0;
                                          }
