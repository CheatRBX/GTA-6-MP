# Установка

> ⚠️ Устанавливаемого образа пока нет: проект в ранней разработке. Здесь описано, как будет устроена установка.
> Прошивка стирает данные. Перед началом сохраните всё важное и стоковую прошивку своего телефона.

## Перед любым способом
1. Проверьте [минимальные требования](requirements.md).
2. Разблокируйте загрузчик (у каждого производителя по-своему).
3. Скачайте **стоковую прошивку** своей модели: из неё берутся `boot`/`init_boot`/`vbmeta` для отката.
4. Скачайте из релиза образ `CoraxOS-*.img.xz` и `SHA256SUMS`, проверьте сумму:
   `sha256sum -c SHA256SUMS --ignore-missing`.

## Способ 1. fastboot / fastbootd (рекомендуется)
В архиве релиза есть скрипты, которые делают всё по шагам и проверяют телефон перед записью:
- Linux/macOS: `./flash.sh CoraxOS-*.img.xz`
- Windows: `flash.bat CoraxOS-*.img.xz`

Вручную это выглядит так:
```bash
fastboot flash --disable-verity --disable-verification vbmeta vbmeta.img   # vbmeta из стоковой прошивки
fastboot reboot fastboot                                                   # fastbootd: нужен для раздела super
fastboot erase system
fastboot flash system CoraxOS.img
fastboot -w                                                                # сброс данных
fastboot reboot
```

## Способ 2. TWRP или OrangeFox
1. Скопируйте `CoraxOS-*-installer.zip` на телефон.
2. Recovery → Install → выберите zip. Установщик сам находит раздел `system` (в том числе внутри `super`).
3. Wipe → Format Data → `yes`.
4. Reboot → System.

## Способ 3. adb sideload (любой recovery, включая LineageOS Recovery)
1. Recovery → Apply update → Apply from ADB.
2. На компьютере: `adb sideload CoraxOS-*-installer.zip`.
3. Factory reset → Format data, затем перезагрузка.

## Способ 4. DSU — попробовать без стирания данных
Система загружается из временного образа, ваша прошивка не трогается.
1. Установите приложение DSU Sideloader.
2. Выберите `CoraxOS-*.img.xz` и следуйте подсказкам.
3. Перезагрузитесь в DSU. Чтобы вернуться, просто перезагрузите телефон.

## Первое включение
Мастер спросит язык, Wi‑Fi, для чего телефон, сервисы Google (нет / microG / Google Play), root (Magisk / KernelSU / APatch / без root) и наборы (Kali NetHunter, драйверы USB Wi‑Fi). Всё это потом можно поменять в «Настройки → Corax Центр».

Если что-то пошло не так — см. [откат](rollback.md).
