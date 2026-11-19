#!/usr/bin/env bash
# Проверки GSI Corax OS, загруженного в эмуляторе (после build-disk.sh).
#   qa/emu-gsi/check.sh <ожидаемая версия> <наш bootanimation.zip>
# Затем запускается общий сценарий плагина (мастер + плитка), без установки APK — он уже в образе.
set -euo pipefail
version=$1
anim=$2
OUT=qa-output
mkdir -p "$OUT"
fail() { echo "ОШИБКА: $*"; exit 1; }
p() { adb shell getprop "$1" | tr -d '\r'; }

echo "== Система"
echo "Android: $(p ro.build.version.release)  сборка: $(p ro.build.display.id)" | tee "$OUT/gsi-system.txt"
[[ $(p ro.corax.version) == "$version" ]] || fail "ro.corax.version = '$(p ro.corax.version)', ожидалось $version"
echo "ro.corax.name=$(p ro.corax.name) ro.corax.version=$(p ro.corax.version) ro.corax.build=$(p ro.corax.build)" | tee -a "$OUT/gsi-system.txt"
adb shell cat /system/etc/corax-release | tee -a "$OUT/gsi-system.txt"

echo "== Corax Центр — системное приложение из образа"
path=$(adb shell pm path org.coraxos.center | tr -d '\r')
echo "$path"
[[ $path == *"/system/priv-app/CoraxCenter/"* ]] || fail "Corax Центр не в /system/priv-app"

echo "== Загрузочная анимация"
want=$(stat -c %s "$anim")
have=$(adb shell 'stat -c %s /product/media/bootanimation.zip 2>/dev/null || stat -c %s /system/media/bootanimation.zip' | tr -d '\r')
echo "в образе: $have байт, наша: $want байт"
[[ $have == "$want" ]] || fail "в системе не наша загрузочная анимация"
adb logcat -d -s BootAnimation:* | head -20 > "$OUT/bootanimation-log.txt" || true

echo "== Конец мастера настройки"
# В AOSP-GSI мастер Provision имеет тот же приоритет HOME (1), что и наш экран, поэтому
# здесь завершаем его вручную. В LineageOS приоритет мастера 9 — там это не нужно.
adb shell pm disable-user --user 0 com.android.provision >/dev/null 2>&1 || true
adb shell settings put global device_provisioned 1
adb shell settings put secure user_setup_complete 1

SKIP_INSTALL=1 bash qa/emulator-spike.sh -
