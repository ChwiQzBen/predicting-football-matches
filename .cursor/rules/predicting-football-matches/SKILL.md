---
name: predicting-football-matches
description: Use when the user wants a pre-match prediction or betting analysis for an upcoming football match that has not been played yet — e.g. World Cup 2026 fixtures, "who will win", reading European bookmaker odds, Asian handicap / over-under, or Vietnamese requests like "soi kèo", "phân tích trận đấu", "dự đoán tỉ số", "kèo nhà cái", "tỉ lệ thắng cao".
---

# Predicting Football Matches

## Overview

Pre-match analysis done honestly: **gather real data → estimate true outcome
probabilities → compare them to the bookmaker's vig-free implied odds → recommend
a bet ONLY where expected value is positive.**

**Core principle — value, not certainty.** "Find me high-win-rate picks" sounds
like the goal but is a trap. Backing heavy favourites gives a high *hit rate* and
*negative* long-run return, because the market already prices their strength. The
only durable edge is **positive expected value (EV)**: cases where your estimated
probability beats the market's vig-free probability. Always reframe a "sure win"
request into a value-betting analysis, and say so.

## The reframe (state this to the user)

| User asks for | What you actually deliver |
|---|---|
| "High win-rate / sure picks" | +EV bets where your prob > market's fair prob |
| "Bet on the favourite to win" | Only if it's +EV; usually it is fairly priced — say so |
| "Guaranteed odds" | No guarantees; long-run edge + responsible staking |

## Workflow

1. **Pick the target market(s) with the user.** Ask (or honour what they stated)
   which betting style to analyse — they bet different markets in different regions:
   - **Châu Âu / European** → 1X2, Double Chance, European over/under, BTTS
   - **Châu Á / Asian** → Asian handicap (kèo chấp) + Asian over/under (tài xỉu),
     incl. quarter lines (−0.25, −0.75, 2.25…)
   - **Both / không rõ** → default to analysing both.
   See `references/betting-markets.md` for what each market is and which script
   command computes it. The same xG model feeds all of them; only the odds-read and
   value step differ by market.
2. **Gather real data with web search — never invent it.** Use WebSearch /
   WebFetch for live lineups, injuries, H2H, form, and odds. If a data point can't
   be found, say "unknown" — do NOT fabricate a number. (This is the #1 failure
   mode; see Red Flags.)
3. **Apply World Cup 2026 context.** Read `references/worldcup-2026-context.md` —
   format/qualification motivation, heat, altitude, travel, rotation.
4. **Estimate expected goals (xG) for each side**, anchored on data (see Method),
   then run `scripts/match_model.py outcomes` for 1X2 / totals / BTTS / scorelines.
5. **Pull LIVE odds for the chosen market(s)** from 2+ bookmakers, then compute value:
   - European 1X2 → `match_model.py value` (removes vig, shows edge/EV/Kelly)
   - Asian handicap → `match_model.py asian --line <h> --side home|away --odds <o>`
   - Asian over/under → `match_model.py atotals --line <l> --side over|under --odds <o>`
6. **Size stakes with fractional Kelly** (the script defaults to 0.25×). Skip any
   bet whose edge doesn't clear a **safety margin** for model error (require
   model prob ≥ fair prob + ~3–4 percentage points, or EV > ~2%, before betting).
7. **Output using the template below.** Be explicit about which market, confidence,
   and unknowns.

## Required data checklist (do this BEFORE modelling)

| Item | What to find | Why |
|---|---|---|
| **Projected XI** | Confirmed lineup (~1h pre-KO) or projected (1–2 days out) | Missing stars swing xG |
| **Injuries / suspensions** | Key absentees + replacement quality | Directly adjusts xG |
| **Recent form** | Last 5–6 games: results, **xG for/against** if available | Form-weighted strength |
| **Head-to-head** | Last meetings, esp. tournament/neutral venue | Style matchups recur |
| **Context** | Stakes, qualification scenario, rotation risk | Motivation distorts form |
| **WC2026 factors** | Venue heat/altitude/roof, rest days, travel | See reference file |
| **Live odds** | 1X2, Asian handicap (kèo châu Á), over/under (tài xỉu), BTTS, from 2+ books | Defines fair value |

## Method (turning data into xG)

- Anchor each side's strength on public ratings (Elo / SPI-style) and recent **xG**,
  not just results. Adjust up/down for confirmed absentees and the WC2026 context
  factors (heat/altitude lower tempo and goals; rotation lowers a favourite's xG).
- Feed the two xG numbers to the Poisson model in `scripts/match_model.py`. Run
  `--help` for commands. It outputs win/draw/win, over/under any line, BTTS, and
  the most likely scorelines.
- **Reading European odds:** decimal odds include a margin (vig). The `value`
  command strips it to get the book's fair probability, then shows edge, EV per
  unit, and a Kelly stake. Bet only rows flagged `+EV` that clear the safety margin.
- **Reading Asian odds (kèo chấp / tài xỉu):** `asian` and `atotals` handle full,
  half, and quarter lines (quarter lines split the stake → half-win/half-loss).
  They report win/push/lose, EV per unit, and a push-aware Kelly stake. Asian lines
  often carry a lower margin than 1X2 — frequently better value.
- Always compare 2+ books and take the best price (line shopping).

## Output template (bilingual labels for VN users)

```
TRẬN / MATCH: A vs B — [competition, venue, local KO time]
LOẠI KÈO / MARKET: [Châu Âu 1X2 | Châu Á handicap+tài xỉu | both] — as user chose
DỮ LIỆU / DATA: [XI status, key injuries, form, H2H, WC2026 factors — with sources]
DỰ ĐOÁN / PREDICTION: [1X2 lean] — tỉ số khả dĩ nhất / most likely score [x-y]
  Model: Home __% | Draw __% | Away __% | O/U 2.5 __% | BTTS __%
SOI KÈO / ODDS READ: model vs market fair probs; vig __% (per chosen market)
KÈO GIÁ TRỊ / VALUE BETS (only +EV past safety margin):
  - [market+line, e.g. AH home -0.5 / Under 2.5 / 1X2 away] @ [odds, book]
    | model __% vs fair __% | EV +__ | stake __% bankroll
ĐỘ TIN CẬY / CONFIDENCE: [low/med/high] + biggest unknowns
⚠️ No guarantee. +EV is a long-run edge. Bet only what you can afford to lose.
```

## Red Flags — STOP

- **Inventing lineups, form, or odds instead of searching.** If you didn't pull it,
  you don't know it. Say "unknown" and lower confidence.
- **Calling a favourite a "high-win-rate pick"** without an EV check — usually it's
  fairly priced and not a value bet.
- **Recommending accumulators/parlays** — compounding vig destroys EV. Singles only.
- **Skipping the WC2026 context** (heat, altitude, qualification motivation, rotation).
- **Betting a thin edge** inside model error — require the safety margin.
- **Promising certainty.** There are no locks; never imply one.

## Responsible gambling

This is probabilistic analysis, not financial advice. No method guarantees wins;
variance is real and can bankrupt a true edge if over-staked. Recommend flat or
fractional-Kelly stakes, only money the user can afford to lose, and never chasing
losses. If the user shows signs of problem gambling, point them to support
resources rather than more picks.
