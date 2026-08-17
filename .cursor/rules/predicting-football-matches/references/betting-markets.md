# Betting markets — European vs Asian (let the user choose)

Different regions bet different markets. Ask which the user wants, then read the
odds and compute value for *that* market. The same Poisson xG model feeds them all;
only the odds-read differs.

## Odds formats you'll see

- **Decimal (European):** 1.95 → stake×1.95 returned on win. Implied prob = 1/odds.
  This is what `match_model.py` expects — convert other formats to decimal first.
- Hong Kong (0.95), Malay, Indo (−1.05) are just other notations for the same price;
  most European/Asian books also display decimal. Use decimal.

## Châu Âu / European markets

| Market | What it is | Script command |
|---|---|---|
| **1X2** | Home / Draw / Away — three-way, draw is a losing-or-winning outcome | `value` (no-vig, EV, Kelly per outcome) |
| **Double Chance** | Two of the three outcomes (1X, 12, X2) | derive from `outcomes` probs |
| **Over/Under** | Total goals over/under a .5 line (2.5, 3.5) | `totals --line 2.5` |
| **BTTS** | Both teams to score, yes/no | shown by `outcomes` |

European 1X2 is simple but the draw drains value and margins are often higher.

## Châu Á / Asian markets

The draw is removed; you back a team *with a goal head-start/deficit*, or a total
with quarter-goal precision. Lower margins → often better value.

### Asian handicap (kèo chấp) — `asian --line <h> --side home|away --odds <o>`

| Line type | Example | Mechanic |
|---|---|---|
| **Full** | −1.0 | Win by 2+ = win; by exactly 1 = **push** (stake back); else lose |
| **Half** | −0.5, +0.5 | No push — straight win/lose |
| **Quarter** | −0.25, −0.75 | **Stake splits** across the two adjacent lines → can half-win / half-lose |

`-0.25` = half on 0 (draw-no-bet) + half on −0.5. `-0.75` = half on −0.5 + half on −1.

### Asian over/under (tài xỉu) — `atotals --line <l> --side over|under --odds <o>`

Same full/half/quarter mechanics applied to total goals. `2.25` = half on 2.0 (push
if exactly 2 goals) + half on 2.5. `2.75` = half on 2.5 + half on 3.0.

## How the script reports Asian bets

`asian` / `atotals` print **win / push(refund) / lose** probabilities, **EV per
unit** (positive = edge after the real price), and a **push-aware Kelly** stake
(numerically optimised over the full result distribution, so quarter-line half
results are handled correctly). Quarter-line half-wins are folded into the "win"
bucket — read EV, not just win%.

## Picking for the user

- User says "soi kèo châu Á" / "kèo chấp" / "tài xỉu" → use `asian` / `atotals`.
- User says "kèo châu Âu" / "1X2" / "thắng-hòa-thua" → use `value` / `totals`.
- Unspecified → analyse both and show value in each; recommend whichever has the
  larger +EV that clears the safety margin.
