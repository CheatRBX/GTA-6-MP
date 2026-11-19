#!/usr/bin/env bash
# Проверка безопасного запуска на userdebug-эмуляторе (нужен adb root и записываемый /system).
#   bash qa/safeboot/test.sh
# 1) ставим image/corax.rc и тестовый «ломатель»; 2) обычная загрузка → счётчик 0, режим 0;
# 3) включаем «ломатель» → две автоматические перезагрузки → третья загрузка доходит до конца
#    в безопасном режиме (sys.corax.safemode=1, persist.corax.safemode=1).
set -euo pipefail
OUT=qa-output
mkdir -p "$OUT"
QA="python3 qa/android-qa/scripts/android-qa.py"
p() { adb shell getprop "$1" 2>/dev/null | tr -d '\r'; }
fail() { echo "ОШИБКА: $*"; adb logcat -d > "$OUT/safeboot-logcat.txt" || true; exit 1; }
wait_boot() { # wait_boot <секунд>: ждём sys.boot_completed=1, переживая перезагрузки
  local deadline=$(( $(date +%s) + $1 ))
  while (( $(date +%s) < deadline )); do
    adb wait-for-device >/dev/null 2>&1 || true
    [[ $(p sys.boot_completed) == 1 ]] && { sleep 5; return 0; }
    sleep 3
  done
  return 1
}

echo "== Ставим безопасный запуск и «ломатель»"
adb root >/dev/null; sleep 3
adb remount >/dev/null
adb push image/corax.rc /system/etc/init/corax.rc >/dev/null
adb push qa/safeboot/breaker.rc /system/etc/init/corax-test-breaker.rc >/dev/null
adb shell "grep -q '^sys.corax.safemode=' /system/build.prop || printf 'persist.corax.boot_attempts=0\npersist.corax.safemode=0\nsys.corax.safemode=0\n' >> /system/build.prop"
adb shell setprop persist.corax.test_breaker 0
adb shell setprop persist.corax.safemode 0
adb shell setprop persist.corax.boot_attempts 0
adb reboot
wait_boot 600 || fail "обычная загрузка не завершилась"

echo "== Обычная загрузка"
adb root >/dev/null; sleep 3
echo "attempts=$(p persist.corax.boot_attempts) safemode=$(p sys.corax.safemode)" | tee "$OUT/safeboot-1-normal.txt"
[[ $(p persist.corax.boot_attempts) == 0 ]] || fail "счётчик не сброшен после загрузки"
[[ $(p sys.corax.safemode) == 0 ]] || fail "безопасный режим включился без причины"

echo "== Включаем «ломатель»: ждём бутлуп и выход из него"
adb shell setprop persist.corax.test_breaker 1
start=$(date +%s)
adb reboot
sleep 20
wait_boot 900 || fail "система так и не загрузилась — безопасный запуск не сработал"
adb root >/dev/null; sleep 3
took=$(( $(date +%s) - start ))
reasons=$(p persist.sys.boot.reason.history | tr '\n' ' ')
{
  echo "загрузилась через ${took} с"
  echo "sys.corax.safemode=$(p sys.corax.safemode) persist.corax.safemode=$(p persist.corax.safemode) attempts=$(p persist.corax.boot_attempts)"
  echo "история причин перезагрузки: $reasons"
} | tee "$OUT/safeboot-2-loop.txt"
[[ $(p sys.corax.safemode) == 1 ]] || fail "система загрузилась, но не в безопасном режиме"
[[ $(p persist.corax.safemode) == 1 ]] || fail "безопасный режим не запомнился"
[[ $(p persist.corax.boot_attempts) == 0 ]] || fail "счётчик не сброшен после загрузки в безопасном режиме"
grep -q corax-test-breaker <<<"$reasons" || echo "(предупреждение: в истории нет перезагрузок «ломателя»)"

echo "== Corax Центр показывает безопасный режим"
adb shell am start -W -n org.coraxos.center/.CenterActivity >/dev/null; sleep 4
serial=$(adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
$QA dump-ui --serial "$serial" --out "$OUT/safeboot-center.xml"
$QA summarize-ui --input "$OUT/safeboot-center.xml" --out "$OUT/safeboot-center-summary.txt"
$QA screenshot --serial "$serial" --out "$OUT/safeboot-center.png"
if grep -q "Безопасный режим" "$OUT/safeboot-center-summary.txt"; then
  echo "Corax Центр: плашка безопасного режима есть"
else
  echo "(предупреждение: плашки безопасного режима в Corax Центре не видно — проверить доступ приложения к свойству)"
fi

echo "== Уборка"
adb shell setprop persist.corax.test_breaker 0
adb shell setprop persist.corax.safemode 0
adb shell rm -f /system/etc/init/corax-test-breaker.rc
echo "OK: безопасный запуск вывел систему из бутлупа"
