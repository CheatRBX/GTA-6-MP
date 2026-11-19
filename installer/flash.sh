#!/usr/bin/env bash
# Установка CoraxOS через fastboot / fastbootd (Linux, macOS).
#
#   ./flash.sh CoraxOS-*.img.xz [--vbmeta vbmeta.img] [--keep-data] [--yes]
#
#   --vbmeta     vbmeta.img из стоковой прошивки: прошивается с отключённой проверкой (нужно почти всегда)
#   --keep-data  не стирать данные (только для обновления CoraxOS поверх CoraxOS)
#   --yes        не спрашивать подтверждение
set -euo pipefail

img="" vbmeta="" keep_data=0 yes=0
while (( $# )); do
  case $1 in
    --vbmeta) vbmeta=$2; shift 2 ;;
    --keep-data) keep_data=1; shift ;;
    --yes) yes=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) img=$1; shift ;;
  esac
done
[[ -n $img && -f $img ]] || { echo "Укажите образ: ./flash.sh CoraxOS-*.img.xz" >&2; exit 2; }
command -v fastboot >/dev/null || { echo "Не найден fastboot (Android SDK Platform-Tools)" >&2; exit 2; }

var() { fastboot getvar "$1" 2>&1 | awk -F': ' -v k="$1" '$1==k {print $2; exit}' | tr -d '\r'; }

echo "Ищу телефон в режиме fastboot…"
fastboot devices | grep -q . || { echo "Телефон не найден. Перезагрузите его в bootloader: adb reboot bootloader" >&2; exit 1; }
echo "Модель: $(var product)   Разблокирован: $(var unlocked)   Слот: $(var current-slot)"
[[ $(var unlocked) == yes ]] || { echo "Загрузчик заблокирован — сначала разблокируйте его" >&2; exit 1; }

raw=$img
if [[ $img == *.xz ]]; then
  raw=${TMPDIR:-/tmp}/coraxos-system.img
  echo "Распаковываю образ…"; xz -dc -T0 "$img" > "$raw"
  trap 'rm -f "$raw"' EXIT
fi

echo
echo "Будет прошито: $(basename "$img") → раздел system"
(( keep_data )) || echo "ВНИМАНИЕ: все данные на телефоне будут стёрты"
if (( ! yes )); then
  read -r -p "Продолжить? Введите YES: " answer
  [[ $answer == YES ]] || { echo "Отменено"; exit 1; }
fi

if [[ -n $vbmeta ]]; then
  echo "vbmeta с отключённой проверкой…"
  fastboot --disable-verity --disable-verification flash vbmeta "$vbmeta"
fi

if [[ $(var is-userspace) != yes && -n $(var super-partition-name) ]]; then
  echo "Раздел super: перехожу в fastbootd…"
  fastboot reboot fastboot
  sleep 5
fi

fastboot erase system || true
if ! fastboot flash system "$raw"; then
  echo "Не хватило места в super. Освобождаю: удаляю раздел product (в GSI он уже внутри system)…"
  slot=$(var current-slot); suffix=${slot:+_$slot}
  fastboot delete-logical-partition "product$suffix" || true
  fastboot flash system "$raw"
fi

(( keep_data )) || fastboot -w
fastboot reboot
echo "Готово. Первое включение может занять несколько минут."
