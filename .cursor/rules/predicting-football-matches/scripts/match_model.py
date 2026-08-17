#!/usr/bin/env python3
"""Pre-match football model: Poisson probabilities + value / EV / Kelly.

Supports both market families so the user can pick the style they bet:
  * European (kèo châu Âu): 1X2, over/under, BTTS  -> `outcomes`, `totals`, `value`
  * Asian   (kèo châu Á)  : Asian handicap (kèo chấp) and Asian over/under
                            (tài xỉu) incl. quarter lines -> `asian`, `atotals`

Inputs are YOUR estimates (expected goals per side) and the LIVE decimal odds
you pulled from a bookmaker. Gather real data first — do not invent the inputs.

Usage:
    python match_model.py outcomes --home-xg 1.65 --away-xg 0.80
    python match_model.py totals   --home-xg 1.65 --away-xg 0.80 --line 2.5
    python match_model.py value    --home-xg 1.65 --away-xg 0.80 \
        --odds-home 1.55 --odds-draw 4.00 --odds-away 6.50
    python match_model.py asian    --home-xg 1.65 --away-xg 0.80 \
        --line -0.75 --side home --odds 1.95
    python match_model.py atotals  --home-xg 1.65 --away-xg 0.80 \
        --line 2.25 --side under --odds 1.90
"""
import argparse
from math import exp, factorial, log


# ---------- Poisson core ----------

def poisson_pmf(k, lam):
    return exp(-lam) * lam ** k / factorial(k)


def score_matrix(home_xg, away_xg, max_goals=10):
    return {
        (i, j): poisson_pmf(i, home_xg) * poisson_pmf(j, away_xg)
        for i in range(max_goals + 1)
        for j in range(max_goals + 1)
    }


def margin_probs(home_xg, away_xg, max_goals=10):
    """P(home_goals - away_goals == m)."""
    out = {}
    for (i, j), p in score_matrix(home_xg, away_xg, max_goals).items():
        out[i - j] = out.get(i - j, 0.0) + p
    return out


def total_probs(home_xg, away_xg, max_goals=10):
    """P(home_goals + away_goals == t)."""
    out = {}
    for (i, j), p in score_matrix(home_xg, away_xg, max_goals).items():
        out[i + j] = out.get(i + j, 0.0) + p
    return out


def outcome_probs(home_xg, away_xg, max_goals=10):
    m = margin_probs(home_xg, away_xg, max_goals)
    home = sum(p for k, p in m.items() if k > 0)
    draw = m.get(0, 0.0)
    away = sum(p for k, p in m.items() if k < 0)
    return home, draw, away


def totals(home_xg, away_xg, line, max_goals=10):
    t = total_probs(home_xg, away_xg, max_goals)
    over = sum(p for k, p in t.items() if k > line)
    under = sum(p for k, p in t.items() if k < line)  # .5 line -> no push
    return over, under


def btts(home_xg, away_xg, max_goals=10):
    yes = sum(p for (i, j), p in score_matrix(home_xg, away_xg, max_goals).items()
              if i >= 1 and j >= 1)
    return yes, 1 - yes


def top_scores(home_xg, away_xg, n=6, max_goals=10):
    return sorted(score_matrix(home_xg, away_xg, max_goals).items(),
                  key=lambda kv: kv[1], reverse=True)[:n]


# ---------- European 1X2 value ----------

def no_vig(odds):
    """odds: {label: decimal}. Returns ({label: fair_prob}, margin)."""
    raw = {k: 1.0 / v for k, v in odds.items()}
    s = sum(raw.values())
    return {k: v / s for k, v in raw.items()}, s - 1.0


def ev(prob, decimal_odds):
    """Expected value per 1 unit staked. Positive => +EV."""
    return prob * decimal_odds - 1.0


def kelly(prob, decimal_odds, fraction=0.25):
    """Fractional-Kelly stake (share of bankroll). 0 if no edge."""
    b = decimal_odds - 1.0
    f = (b * prob - (1 - prob)) / b
    return max(0.0, f) * fraction


# ---------- Asian handicap / Asian over-under ----------
# Handle full (push possible), half (no push) and quarter (split-stake) lines.

def _is_quarter(line):
    return abs(round(line * 2) - line * 2) > 1e-9


def _single_net(adj, odds):
    """Net return per unit for a non-quarter outcome. adj>0 win, ==0 push."""
    if adj > 1e-9:
        return odds - 1.0
    if adj < -1e-9:
        return -1.0
    return 0.0


def _net(signed, line, odds):
    """Net per unit for a signed value (margin for AH, total-diff for AU).
    Quarter lines split the stake across the two adjacent lines."""
    if _is_quarter(line):
        return 0.5 * _single_net(signed + (line - 0.25), odds) \
             + 0.5 * _single_net(signed + (line + 0.25), odds)
    return _single_net(signed + line, odds)


def _outcome_dist(items, signed_of, line, odds):
    """Aggregate net-return-per-unit -> probability over the item distribution."""
    dist = {}
    for item, p in items.items():
        ret = round(_net(signed_of(item), line, odds), 6)
        dist[ret] = dist.get(ret, 0.0) + p
    return dist


def asian_handicap(home_xg, away_xg, line, side, odds, max_goals=10):
    """side: 'home' bets home+line ; 'away' bets away+line.
    Returns dict with win/push/lose probs, expected return, EV, suggested Kelly."""
    mp = margin_probs(home_xg, away_xg, max_goals)
    sign = 1 if side == "home" else -1
    dist = _outcome_dist(mp, lambda m: sign * m, line, odds)
    return _summary(dist, odds)


def asian_total(home_xg, away_xg, line, side, odds, max_goals=10):
    """side: 'over' or 'under' (tài / xỉu).
    Win when over: total > line; under: total < line. _net wins when
    (signed + line_param) > 0, so map each side onto that convention."""
    tp = total_probs(home_xg, away_xg, max_goals)
    if side == "over":
        dist = _outcome_dist(tp, lambda t: t, -line, odds)   # t - line > 0
    else:                                                    # under
        dist = _outcome_dist(tp, lambda t: -t, line, odds)   # line - t > 0
    return _summary(dist, odds)


def _summary(dist, odds, fraction=0.25):
    win = sum(p for r, p in dist.items() if r > 1e-9)
    push = sum(p for r, p in dist.items() if abs(r) <= 1e-9)
    lose = sum(p for r, p in dist.items() if r < -1e-9)
    expected_return = sum(r * p for r, p in dist.items())  # net per unit
    stake = _kelly_numeric(dist) * fraction
    return {"win": win, "push": push, "lose": lose,
            "ev": expected_return, "kelly": stake}


def _kelly_numeric(dist):
    """Full-Kelly fraction maximizing expected log growth over the net-return
    distribution. Robust to pushes and quarter-line half results."""
    best_f, best_g = 0.0, 0.0
    f = 0.0
    while f < 0.999:
        f += 0.001
        g = 0.0
        ok = True
        for r, p in dist.items():
            x = 1.0 + f * r
            if x <= 1e-12:
                ok = False
                break
            g += p * log(x)
        if ok and g > best_g:
            best_g, best_f = g, f
    return best_f


# ---------- CLI ----------

def _pct(x):
    return f"{x * 100:5.1f}%"


def _ev_flag(e):
    """Don't celebrate a negligible edge: require >2% EV to mark a bet +EV."""
    if e > 0.02:
        return "  <- +EV"
    if e > 0:
        return "  (thin edge - within model error, likely skip)"
    return "  (no edge)"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("outcomes", "totals", "value", "asian", "atotals"):
        p = sub.add_parser(name)
        p.add_argument("--home-xg", type=float, required=True)
        p.add_argument("--away-xg", type=float, required=True)
        if name == "totals":
            p.add_argument("--line", type=float, default=2.5)
        if name == "value":
            p.add_argument("--odds-home", type=float, required=True)
            p.add_argument("--odds-draw", type=float, required=True)
            p.add_argument("--odds-away", type=float, required=True)
            p.add_argument("--kelly-fraction", type=float, default=0.25)
        if name in ("asian", "atotals"):
            p.add_argument("--line", type=float, required=True)
            p.add_argument("--side", required=True,
                           choices=["home", "away"] if name == "asian"
                           else ["over", "under"])
            p.add_argument("--odds", type=float, required=True)

    a = ap.parse_args()
    h, d, w = outcome_probs(a.home_xg, a.away_xg)

    if a.cmd == "outcomes":
        print(f"Home win {_pct(h)} | Draw {_pct(d)} | Away win {_pct(w)}")
        o25, u25 = totals(a.home_xg, a.away_xg, 2.5)
        by, bn = btts(a.home_xg, a.away_xg)
        print(f"Over 2.5 {_pct(o25)} | Under 2.5 {_pct(u25)} | "
              f"BTTS Yes {_pct(by)} | BTTS No {_pct(bn)}")
        print("Most likely scores:")
        for (i, j), p in top_scores(a.home_xg, a.away_xg):
            print(f"  {i}-{j}: {_pct(p)}")

    elif a.cmd == "totals":
        o, u = totals(a.home_xg, a.away_xg, a.line)
        print(f"Over {a.line} {_pct(o)} | Under {a.line} {_pct(u)}")

    elif a.cmd == "value":
        odds = {"home": a.odds_home, "draw": a.odds_draw, "away": a.odds_away}
        model = {"home": h, "draw": d, "away": w}
        fair, margin = no_vig(odds)
        print(f"Market: European 1X2 | bookmaker margin (vig): {_pct(margin)}\n")
        print(f"{'Outcome':8} {'Model':>7} {'Fair(mkt)':>10} "
              f"{'Edge':>7} {'EV/unit':>8} {'Kelly%':>7}")
        for k in ("home", "draw", "away"):
            e = ev(model[k], odds[k])
            stake = kelly(model[k], odds[k], a.kelly_fraction)
            flag = _ev_flag(e)
            print(f"{k:8} {_pct(model[k]):>7} {_pct(fair[k]):>10} "
                  f"{(model[k]-fair[k])*100:+6.1f}% {e:+8.3f} {stake*100:6.2f}%{flag}")
        print("\nOnly bet +EV rows that also clear a safety margin (see SKILL.md).")

    elif a.cmd == "asian":
        r = asian_handicap(a.home_xg, a.away_xg, a.line, a.side, a.odds)
        print(f"Market: Asian handicap (kèo chấp) | {a.side} {a.line:+g} @ {a.odds}")
        print(f"Win {_pct(r['win'])} | Push/refund {_pct(r['push'])} | "
              f"Lose {_pct(r['lose'])}")
        flag = _ev_flag(r["ev"])
        print(f"EV/unit {r['ev']:+.3f}{flag} | suggested stake {r['kelly']*100:.2f}% bankroll")
        print("Quarter lines (e.g. -0.75) split the stake; win/push/lose fold the "
              "half-results in. Bet only +EV past the safety margin.")

    elif a.cmd == "atotals":
        r = asian_total(a.home_xg, a.away_xg, a.line, a.side, a.odds)
        print(f"Market: Asian over/under (tài xỉu) | {a.side} {a.line:g} @ {a.odds}")
        print(f"Win {_pct(r['win'])} | Push/refund {_pct(r['push'])} | "
              f"Lose {_pct(r['lose'])}")
        flag = _ev_flag(r["ev"])
        print(f"EV/unit {r['ev']:+.3f}{flag} | suggested stake {r['kelly']*100:.2f}% bankroll")


if __name__ == "__main__":
    main()
