#!/usr/bin/env bash
# pull.sh -- git pull that does not eat the market data.
#
# WHY THIS EXISTS. There is an unpushed commit on this branch that stops
# tracking taker_data/, oi_data/, funding_data/ and price_data/. Every
# `git pull --rebase` replays it, and a replayed removal makes git delete
# the working-tree copies -- 222MB gone, twice so far.
#
# Pushing that commit would fix it permanently. Until then this restores
# from ~/hyro_data_backup afterwards, which takes a few seconds and needs
# no GitHub access. Safe to run every time; it only copies what is
# missing.
set -euo pipefail
cd "$(dirname "$0")"
BK="${HYRO_BACKUP:-$HOME/hyro_data_backup}"

before=$(ls taker_data/*.csv 2>/dev/null | wc -l)
git pull --rebase "$@"
after=$(ls taker_data/*.csv 2>/dev/null | wc -l)

if [ -d "$BK" ]; then
  for d in taker_data oi_data funding_data price_data; do
    [ -d "$BK/$d" ] && mkdir -p "$d" && cp -rn "$BK/$d/." "$d/" 2>/dev/null || true
  done
  now=$(ls taker_data/*.csv 2>/dev/null | wc -l)
  if [ "$after" -lt "$before" ]; then
    echo "  the pull removed $((before-after)) data files; restored to $now"
  else
    echo "  data intact: $now taker files"
  fi
else
  echo "  WARNING: no backup at $BK -- data loss on pull is NOT protected."
  echo "  mkdir -p $BK && cp -r taker_data oi_data funding_data price_data $BK/"
fi
