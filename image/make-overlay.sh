#!/usr/bin/env bash
# Собрать папку-оверлей Corax OS для конкретного базового GSI.
#
#   image/make-overlay.sh <base.img (raw ext4)> <папка-оверлей> <CoraxCenter.apk> <bootanimation.zip> [ожидаемые.txt]
#
# Кладёт: загрузочную анимацию (туда, где она лежит в этом GSI), Corax Центр в /system/priv-app,
# /system/etc/corax-release и build.prop с метками ro.corax.*. Список путей пишет в ожидаемые.txt
# (для image/verify.sh).
set -euo pipefail
base=$1
ov=$2
apk=$3
anim_src=$4
expected=${5:-}
here=$(cd "$(dirname "$0")/.." && pwd)
name=$(jq -r .os_name "$here/brand/brand.json")
version=$(jq -r .version "$here/brand/brand.json")
stamp=$(date -u +%Y%m%d)

anim=/system/product/media/bootanimation.zip
for p in /system/product/media/bootanimation.zip /system/media/bootanimation.zip; do
  if debugfs -R "stat $p" "$base" 2>/dev/null | grep -q Inode; then anim=$p; break; fi
done
echo "Загрузочная анимация: $anim"

rm -rf "$ov"
mkdir -p "$ov$(dirname "$anim")" "$ov/system/priv-app/CoraxCenter" "$ov/system/etc/init"
cp "$anim_src" "$ov$anim"
cp "$apk" "$ov/system/priv-app/CoraxCenter/CoraxCenter.apk"
cp "$here/image/corax.rc" "$ov/system/etc/init/corax.rc"   # безопасный запуск
printf 'name=%s\nversion=%s\nbuild=%s\ncommit=%s\n' "$name" "$version" "${GITHUB_RUN_ID:-local}" "${GITHUB_SHA:-local}" \
  > "$ov/system/etc/corax-release"

debugfs -R "dump /system/build.prop $ov/system/build.prop" "$base" 2>/dev/null
[[ -s $ov/system/build.prop ]] || { echo "Не удалось прочитать /system/build.prop из $base" >&2; exit 1; }
{
  echo ""
  echo "# Corax OS"
  echo "ro.corax.name=$name"
  echo "ro.corax.version=$version"
  echo "ro.corax.build=$stamp.${GITHUB_RUN_NUMBER:-0}"
  echo "# Безопасный запуск (image/corax.rc): значения по умолчанию"
  echo "persist.corax.boot_attempts=0"
  echo "persist.corax.safemode=0"
  echo "sys.corax.safemode=0"
} >> "$ov/system/build.prop"

if [[ -n $expected ]]; then
  (cd "$ov" && find . -type f; find . -mindepth 1 -type d) | sort > "$expected"
fi
echo "Оверлей: $ov"
