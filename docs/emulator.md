# Как посмотреть Corax OS в эмуляторе

## Способ 1. Ничего не ставить: смотреть проверки на GitHub
Каждая сборка сама гоняет эмулятор и сохраняет скриншоты.
1. Откройте https://github.com/CheatRBX/GTA-6-MP/actions
2. Слева выберите проверку:
   - **qa** — Corax Центр в эмуляторе Android 16: мастер «Добро пожаловать» и плитка в Настройках;
   - **emu-gsi** — GSI Android 17 с изменениями Corax OS, загруженный в эмуляторе.
3. Откройте последний прогон с зелёной галочкой → внизу раздел **Artifacts** → скачайте `qa-output` или `emu-gsi-qa`.
   Внутри скриншоты (`*.png`), дерево экрана (`*.xml`) и логи (`logcat.txt`).
4. Пока прогон идёт, ход проверки виден в логе шага «Emulator QA» / «Boot Corax GSI in emulator».

## Способ 2. Запустить у себя (Windows, macOS, Linux)
Нужен компьютер с включённой виртуализацией (Intel VT-x / AMD-V; на Mac — любой).

1. Установите **Android Studio**: https://developer.android.com/studio
2. Device Manager → Create Virtual Device → Pixel 8 → образ **API 36, Google APIs** (не «Google Play»: в нём нельзя менять систему).
3. Скачайте APK: Actions → проверка **apps** → последний зелёный прогон → Artifacts → `CoraxCenter-apk`.
4. Запустите эмулятор с записываемой системой и поставьте Corax Центр системным приложением
   (команды из папки `platform-tools` Android SDK):
   ```bash
   emulator -avd <имя_эмулятора> -writable-system
   adb root
   adb remount          # первый раз попросит перезагрузку:
   adb reboot
   adb root && adb remount
   adb shell mkdir -p /system/priv-app/CoraxCenter
   adb push center-debug.apk /system/priv-app/CoraxCenter/CoraxCenter.apk
   adb reboot
   ```
5. После перезагрузки нажмите «Домой» — откроется экран «Добро пожаловать в Corax OS».
   В Настройках появится плитка «Corax Центр».

## Способ 3. Загрузить сам GSI Corax OS (arm64) — только Mac на Apple M1–M4
Основной GSI собран под arm64, а такой эмулятор есть только на Mac с Apple Silicon.
Скрипт `qa/emu-gsi/build-disk.sh` собирает диск эмулятора с нашим GSI
(нужны Docker Desktop и `brew install gdisk`); пошаговая инструкция появится,
если такой Mac есть.
