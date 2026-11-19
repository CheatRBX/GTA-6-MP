#!/usr/bin/env bash
# Переупаковка готового GSI: распаковать → освободить место → положить оверлей → сжать обратно.
#
#   image/repack.sh <вход: .img | .img.xz | .img.gz | .zip> <папка-оверлей> <выход.img>
#
# Поддерживается ext4 (в том числе sparse и с shared_blocks). EROFS пока нет — скрипт
# скажет об этом явно. RAW_ONLY=1 — только распаковать в raw (для сравнения в verify.sh).
# Нужны: e2fsprogs, xz-utils, gzip, unzip, simg2img (android-sdk-libsparse-utils).
set -euo pipefail

in=$1
overlay=$2
out=$3
here=$(cd "$(dirname "$0")" && pwd)
headroom=${HEADROOM:-768M}

log() { printf '\n== %s\n' "$*"; }

# e2fsck: 0 — чисто, 1 — исправлено. Всё остальное — ошибка.
fsck_ok() {
  local rc=0
  e2fsck -fy "$@" || rc=$?
  if (( rc > 1 )); then echo "e2fsck вернул $rc" >&2; exit 1; fi
}

log "Распаковка $in"
case "$in" in
  *.img.xz) xz -dc -T0 "$in" > "$out" ;;
  *.img.gz) gzip -dc "$in" > "$out" ;;
  *.zip)    unzip -p "$in" '*.img' > "$out" ;;
  *.img)    cp --reflink=auto "$in" "$out" ;;
  *) echo "Неизвестный формат: $in" >&2; exit 1 ;;
esac

magic4() { od -An -tx1 -j "$2" -N4 "$1" | tr -d ' \n'; }
if [[ $(magic4 "$out" 0) == "3aff26ed" ]]; then
  log "Sparse-образ → raw"
  simg2img "$out" "$out.raw" && mv "$out.raw" "$out"
fi
if [[ $(magic4 "$out" 1024) == "e2e1f5e0" ]]; then
  echo "EROFS пока не поддерживается: нужен базовый GSI на ext4" >&2
  exit 2
fi
if [[ $(od -An -tx1 -j 1080 -N2 "$out" | tr -d ' \n') != "53ef" ]]; then
  echo "Это не ext4-образ" >&2
  exit 2
fi

if [[ ${RAW_ONLY:-0} == 1 ]]; then
  log "RAW_ONLY: только распаковка → $out"
  exit 0
fi

log "Проверка и расширение на $headroom"
fsck_ok "$out"
truncate -s "+$headroom" "$out"
resize2fs "$out"
if dumpe2fs -h "$out" 2>/dev/null | grep -q shared_blocks; then
  log "Снимаем shared_blocks"
  fsck_ok -E unshare_blocks "$out"
fi

log "Оверлей $overlay"
python3 "$here/inject.py" "$out" "$overlay" --contexts "$here/contexts.txt"

log "Проверка и сжатие до минимума"
fsck_ok "$out"
resize2fs -M "$out"
fsck_ok "$out"
# resize2fs не обрезает сам файл: подгоняем его размер под файловую систему.
blocks=$(dumpe2fs -h "$out" 2>/dev/null | awk -F: '/^Block count/ {gsub(/ /,"",$2); print $2}')
bsize=$(dumpe2fs -h "$out" 2>/dev/null | awk -F: '/^Block size/ {gsub(/ /,"",$2); print $2}')
truncate -s $((blocks * bsize)) "$out"
log "Готово: $out ($(du -h "$out" | cut -f1))"
