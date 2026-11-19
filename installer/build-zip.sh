#!/usr/bin/env bash
# Собрать установочный zip для TWRP / OrangeFox / adb sideload.
#   installer/build-zip.sh <system.img> <выход.zip>
# Образ кладётся в архив как system.img.xz (без повторного сжатия zip) + его размер в system.img.size.
set -euo pipefail
img=$1
out=$(realpath -m "$2")
here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -r "$here/recovery/META-INF" "$work/"
stat -c %s "$img" > "$work/system.img.size"
xz -T0 -6 -c "$img" > "$work/system.img.xz"
rm -f "$out"
(cd "$work" && zip -q -r -0 "$out" META-INF system.img.size system.img.xz)
echo "installer: $out ($(du -h "$out" | cut -f1))"
