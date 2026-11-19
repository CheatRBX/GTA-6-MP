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
A() { timeout 60 adb "$@"; }   # любой вызов adb — не дольше минуты
p() { timeout 15 adb shell getprop "$1" 2>/dev/null | tr -d '\r'; }
fail() { echo "ОШИБКА: $*"; timeout 60 adb logcat -d > "$OUT/safeboot-logcat.txt" || true; exit 1; }
wait_boot() { # wait_boot <секунд>: ждём sys.boot_completed=1, переживая перезагрузки; пишем ход в лог
  local deadline=$(( $(date +%s) + $1 )) last=""
  while (( $(date +%s) < deadline )); do
    timeout 30 adb wait-for-device >/dev/null 2>&1 || true
    local state now
    state=$(timeout 10 adb get-state 2>/dev/null || echo нет)
    now="adb=$state attempts=$(timeout 10 adb shell getprop persist.corax.boot_attempts 2>/dev/null | tr -d '\r') safemode=$(timeout 10 adb shell getprop sys.corax.safemode 2>/dev/null | tr -d '\r') boot=$(timeout 10 adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')"
    [[ $now != "$last" ]] && { echo "  $(date -u +%H:%M:%S) $now"; last=$now; }
    [[ $now == *"boot=1"* ]] && { sleep 5; return 0; }
    sleep 5
  done
  return 1
}

echo "== Ставим безопасный запуск и «ломатель»"
A root >/dev/null; sleep 3
A remount >/dev/null
A push image/corax.rc /system/etc/init/corax.rc >/dev/null
A push qa/safeboot/breaker.rc /system/etc/init/corax-test-breaker.rc >/dev/null
A shell "grep -q '^sys.corax.safemode=' /system/build.prop || printf 'persist.corax.boot_attempts=0\npersist.corax.safemode=0\nsys.corax.safemode=0\n' >> /system/build.prop"
A shell setprop persist.corax.test_breaker 0
A shell setprop persist.corax.safemode 0
A shell setprop persist.corax.boot_attempts 0
A reboot
wait_boot 600 || fail "обычная загрузка не завершилась"

echo "== Обычная загрузка"
A root >/dev/null; sleep 3
echo "attempts=$(p persist.corax.boot_attempts) safemode=$(p sys.corax.safemode)" | tee "$OUT/safeboot-1-normal.txt"
[[ $(p persist.corax.boot_attempts) == 0 ]] || fail "счётчик не сброшен после загрузки"
[[ $(p sys.corax.safemode) == 0 ]] || fail "безопасный режим включился без причины"

echo "== Включаем «ломатель»: ждём бутлуп и выход из него"
A shell setprop persist.corax.test_breaker 1
start=$(date +%s)
A reboot
sleep 20
wait_boot 1200 || fail "система так и не загрузилась — безопасный запуск не сработал"
A root >/dev/null; sleep 3
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
A shell am start -W -n org.coraxos.center/.CenterActivity >/dev/null; sleep 4
serial=$(timeout 20 adb devices | awk 'NR>1 && $2=="device" {print $1; exit}')
$QA dump-ui --serial "$serial" --out "$OUT/safeboot-center.xml"
$QA summarize-ui --input "$OUT/safeboot-center.xml" --out "$OUT/safeboot-center-summary.txt"
$QA screenshot --serial "$serial" --out "$OUT/safeboot-center.png"
if grep -q "Безопасный режим" "$OUT/safeboot-center-summary.txt"; then
  echo "Corax Центр: плашка безопасного режима есть"
else
  echo "(предупреждение: плашки безопасного режима в Corax Центре не видно — проверить доступ приложения к свойству)"
fi

echo "== Уборка"
# После перезагрузок /system снова только для чтения — уборка не должна валить проверку.
A root >/dev/null 2>&1 || true; sleep 3
A shell setprop persist.corax.test_breaker 0 || true
A shell setprop persist.corax.safemode 0 || true
A remount >/dev/null 2>&1 && A shell rm -f /system/etc/init/corax-test-breaker.rc || echo "(уборка: файл «ломателя» не удалён — эмулятор всё равно одноразовый)"
echo "OK: безопасный запуск вывел систему из бутлупа"
