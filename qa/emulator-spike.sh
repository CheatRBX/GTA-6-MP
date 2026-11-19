#!/usr/bin/env bash
# Шаг 0.3 плана: в эмуляторе Android 17 ставим Corax Центр системным приложением (priv-app)
# и проверяем плагином android-emulator-qa:
#   1) после установки HOME открывает наш экран «Добро пожаловать»;
#   2) мастер проходится до конца (профиль «Пентестер»), после чего HOME снова ведёт в лаунчер;
#   3) в Настройках есть плитка «Corax Центр», и она открывает наш экран.
# Доказательства (скриншоты, UI-дампы, logcat) — в qa-output/.
#
#   bash qa/emulator-spike.sh apps/center/build/outputs/apk/debug/center-debug.apk
set -euo pipefail

apk=$1
QA="python3 qa/android-qa/scripts/android-qa.py"
OUT=qa-output
PKG=org.coraxos.center
mkdir -p "$OUT"

step() { printf '\n== %s\n' "$*"; }
wait_boot() {
  adb wait-for-device
  for _ in $(seq 1 120); do
    [[ $(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r') == 1 ]] && { sleep 5; return; }
    sleep 2
  done
  echo "эмулятор не загрузился" >&2; exit 1
}
home_target() {
  adb shell cmd package resolve-activity --brief -a android.intent.action.MAIN -c android.intent.category.HOME \
    | tr -d '\r' | tail -1
}
ui() { # ui <имя>: дамп UI + сводка + скриншот
  $QA dump-ui --serial "$serial" --out "$OUT/$1.xml"
  $QA summarize-ui --input "$OUT/$1.xml" --out "$OUT/$1-summary.txt"
  $QA screenshot --serial "$serial" --out "$OUT/$1.png"
}
tap_text() { # tap_text <имя-дампа> <текст>
  local xy
  xy=$($QA pick --input "$OUT/$1.xml" --target "$2")
  echo "тап «$2» → $xy"
  # shellcheck disable=SC2086
  adb shell input tap $xy
}

step "Устройство"
$QA devices
serial=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
echo "serial: $serial"
adb shell getprop ro.build.version.release

step "Делаем /system записываемым"
adb root; sleep 3
adb remount || true          # первый раз отключает verity и просит перезагрузку
adb reboot; wait_boot
adb root; sleep 3
adb remount

step "Ставим Corax Центр в /system/priv-app"
adb shell mkdir -p /system/priv-app/CoraxCenter
adb push "$apk" /system/priv-app/CoraxCenter/CoraxCenter.apk
adb shell chmod 644 /system/priv-app/CoraxCenter/CoraxCenter.apk
adb reboot; wait_boot
adb shell pm path "$PKG"
adb shell dumpsys package "$PKG" | grep -E 'flags=|privateFlags=' | head -2

step "1) HOME → наш экран"
before=$(home_target); echo "HOME: $before" | tee "$OUT/home-before.txt"
[[ $before == "$PKG/.WelcomeActivity" ]] || { echo "ОШИБКА: HOME ведёт не к нам"; exit 1; }
adb shell input keyevent KEYCODE_HOME; sleep 4
ui welcome
grep -q "CORAX" "$OUT/welcome-summary.txt" || { echo "ОШИБКА: нет экрана приветствия"; exit 1; }

step "2) Мастер: профиль «Пентестер» → … → «Применить» → «На рабочий стол»"
tap_text welcome "Начать"; sleep 3
ui profile
tap_text profile "Пентестер"; sleep 1
for screen in google root kits summary; do
  ui "before-$screen"
  tap_text "before-$screen" "Далее"; sleep 2
done
ui summary
grep -q "Пентестер" "$OUT/summary-summary.txt" || { echo "ОШИБКА: в сводке нет профиля"; exit 1; }
tap_text summary "Применить"; sleep 10
ui done
tap_text done "На рабочий стол"; sleep 4
after=$(home_target); echo "HOME: $after" | tee "$OUT/home-after.txt"
[[ $after != "$PKG/.WelcomeActivity" ]] || { echo "ОШИБКА: экран приветствия не отключился"; exit 1; }
ui launcher

step "3) Плитка в Настройках"
adb shell am start -W -a android.settings.SETTINGS >/dev/null; sleep 4
found=""
for i in 1 2 3 4 5; do
  ui "settings-$i"
  if grep -q "Corax Центр" "$OUT/settings-$i-summary.txt"; then found="settings-$i"; break; fi
  adb shell input swipe 540 1700 540 700 300; sleep 2
done
[[ -n $found ]] || { echo "ОШИБКА: плитки «Corax Центр» нет в Настройках"; exit 1; }
tap_text "$found" "Corax Центр"; sleep 3
ui center
grep -q "Kali NetHunter" "$OUT/center-summary.txt" || { echo "ОШИБКА: плитка не открыла Corax Центр"; exit 1; }

adb logcat -d > "$OUT/logcat.txt"
adb logcat -b crash -d > "$OUT/crash.txt" || true
step "OK: все три проверки прошли"
