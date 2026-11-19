# Откат

## Убрать root
«Настройки → Corax Центр → Root → Вернуть как было»: восстанавливает раздел `boot`/`init_boot` из бэкапа, который сделан перед установкой root.

Если система не загружается после установки root:
```bash
fastboot flash boot boot.img            # или init_boot init_boot.img — из стоковой прошивки
fastboot reboot
```

## Вернуться на стоковую прошивку
Прошейте стоковую прошивку инструментом вашего производителя (Pixel Flash Tool, Mi Flash, Odin, SP Flash Tool и т.п.)
или через fastboot:
```bash
fastboot flash vbmeta vbmeta.img        # стоковый, без флагов отключения проверки
fastboot reboot fastboot
fastboot flash system system.img        # стоковый system
fastboot -w
fastboot reboot
```

## Если запускали через DSU
Просто перезагрузите телефон — загрузится ваша обычная система.
