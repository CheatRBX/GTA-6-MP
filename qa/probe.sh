#!/system/bin/sh
# Снимок устройства для детектора CoraxOS (шаг 0.2 плана).
# Запуск с компьютера:
#   adb push qa/probe.sh /data/local/tmp/ && adb shell sh /data/local/tmp/probe.sh > device-reports/<модель>.txt
# Ничего не меняет на телефоне, только читает. Root не нужен (с root конфиг ядра читается полнее).

prop() { getprop "$1"; }
yesno() { if [ -n "$1" ] && [ "$1" != "false" ] && [ "$1" != "0" ]; then echo true; else echo false; fi; }
has_part() { [ -e "/dev/block/by-name/$1" ] || [ -e "/dev/block/by-name/${1}_a" ]; }

echo "# CoraxOS probe $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "abi=$(prop ro.product.cpu.abi)"
echo "kernel=$(uname -r)"
first_api=$(prop ro.product.first_api_level)
[ -z "$first_api" ] && first_api=$(prop ro.build.version.sdk)
echo "first_api=$first_api"
echo "vendor_api=$(prop ro.vendor.build.version.sdk)"
echo "board_api=$(prop ro.board.api_level)"
echo "treble=$(yesno "$(prop ro.treble.enabled)")"
echo "ab=$(yesno "$(prop ro.build.ab_update)")"
echo "dynamic=$(yesno "$(prop ro.boot.dynamic_partitions)")"
if has_part init_boot; then echo "init_boot=true"; else echo "init_boot=false"; fi
echo "vndk=$(prop ro.vndk.version)"
echo "boot_header=$(prop ro.boot.boot_header_version)"
echo "manufacturer=$(prop ro.product.vendor.manufacturer)"
echo "model=$(prop ro.product.vendor.model)"
echo "soc=$(prop ro.soc.model)"
echo "ram_mb=$(awk '/MemTotal/ {print int($2/1024)}' /proc/meminfo)"

# Нужные строки конфига ядра (если ядро его отдаёт).
if [ -r /proc/config.gz ]; then
  for opt in KALLSYMS KALLSYMS_ALL MODULES MODULE_SIG_FORCE KSU USB_CONFIGFS_F_HID CFG80211 MAC80211 OVERLAY_FS; do
    val=$(zcat /proc/config.gz 2>/dev/null | grep "^CONFIG_${opt}=" | cut -d= -f2)
    [ -n "$val" ] && echo "config_$(echo "$opt" | tr 'A-Z' 'a-z')=$val"
  done
else
  echo "# /proc/config.gz недоступен"
fi

# Модули Wi-Fi стека, которые уже есть у производителя.
for m in cfg80211 mac80211; do
  found=$(ls /vendor_dlkm/lib/modules/$m.ko /vendor/lib/modules/$m.ko 2>/dev/null | head -1)
  echo "vendor_${m}_ko=${found:-none}"
done
