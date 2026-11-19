#!/usr/bin/env bash
# ТОЛЬКО для проверки в эмуляторе: включить ADB в user-сборке GSI, чтобы проверки могли подключиться.
# В образ Corax OS это не попадает (image.yml этот скрипт не вызывает).
#   qa/emu-gsi/test-props.sh <base.img> <папка-оверлей>
set -euo pipefail
base=$1
ov=$2
set_prop() { # set_prop <файл> <имя> <значение>: заменить строку, а если её нет — дописать
  local f=$1 k=$2 v=$3
  if grep -q "^$k=" "$f"; then sed -i "s|^$k=.*|$k=$v|" "$f"; else echo "$k=$v" >> "$f"; fi
}
mkdir -p "$ov/system/etc"
debugfs -R "dump /system/etc/prop.default $ov/system/etc/prop.default" "$base" 2>/dev/null || true
[[ -s $ov/system/etc/prop.default ]] || : > "$ov/system/etc/prop.default"
for f in "$ov/system/etc/prop.default" "$ov/system/build.prop"; do
  [[ -f $f ]] || continue
  set_prop "$f" ro.adb.secure 0
  set_prop "$f" ro.debuggable 1
  set_prop "$f" persist.sys.usb.config adb
  echo "== $f"; grep -E '^(ro.adb.secure|ro.debuggable|persist.sys.usb.config)=' "$f"
done
