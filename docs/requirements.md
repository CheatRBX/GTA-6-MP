# Минимальные требования

> Предварительные требования. Уточняются по отчётам с реальных телефонов (`device-reports/`).

## Обязательно
| Что | Требование | Как проверить |
| --- | --- | --- |
| Процессор | arm64 (64‑бит) | `adb shell getprop ro.product.cpu.abi` → `arm64-v8a` |
| Project Treble | поддерживается | `adb shell getprop ro.treble.enabled` → `true` |
| Заводская прошивка | Android 11 или новее | `adb shell getprop ro.product.first_api_level` → 30 или больше |
| Загрузчик | разблокирован | зависит от производителя; без этого прошить нельзя |
| ОЗУ | 4 ГБ (для Kali NetHunter — 6 ГБ и больше) | `adb shell cat /proc/meminfo` |
| Память | 64 ГБ; ~4 ГБ свободно в разделе `system`/`super` под образ | `adb shell ls -l /dev/block/by-name/` |

## Что зависит от ядра телефона
Мастер первого включения сам проверяет это и показывает только то, что заработает.

| Функция | Нужно |
| --- | --- |
| Root: Magisk | работает почти везде |
| Root: KernelSU | ядро GKI 5.10+ (телефоны, вышедшие с Android 12+) или наше ядро |
| Root: APatch | arm64‑ядро с `CONFIG_KALLSYMS` |
| Kali NetHunter (полный) | наше ядро или ядро с поддержкой NetHunter (HID‑гаджет, инъекция Wi‑Fi) |
| Kali NetHunter Lite | любой root |
| Kali NetHunter Rootless | ничего, работает без root |
| USB Wi‑Fi (rtl8822bu и др.) | OTG и ядро 6.6 / 6.12 / 6.18 (для 6.1 нужен бэкпорт драйвера) |

Узнать версию ядра: `adb shell uname -r`.
