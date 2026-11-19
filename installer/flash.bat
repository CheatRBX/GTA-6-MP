@echo off
rem Установка CoraxOS через fastboot / fastbootd (Windows).
rem   flash.bat CoraxOS.img [vbmeta.img]
rem Образ .img.xz сначала распакуйте (например, 7-Zip) до .img.
setlocal EnableDelayedExpansion
chcp 65001 >nul

set "IMG=%~1"
set "VBMETA=%~2"
if "%IMG%"=="" goto usage
if not exist "%IMG%" goto usage
if /I "%IMG:~-3%"==".xz" (
  echo Распакуйте .xz до .img (7-Zip) и запустите снова.
  exit /b 2
)
where fastboot >nul 2>&1 || (echo Не найден fastboot ^(Android SDK Platform-Tools^) & exit /b 2)

fastboot devices | findstr /r "." >nul || (echo Телефон не найден. Перезагрузите его в bootloader: adb reboot bootloader & exit /b 1)
for /f "tokens=2 delims=: " %%a in ('fastboot getvar unlocked 2^>^&1 ^| findstr /b "unlocked:"') do set "UNLOCKED=%%a"
if /I not "%UNLOCKED%"=="yes" (echo Загрузчик заблокирован — сначала разблокируйте его & exit /b 1)

echo.
echo Будет прошито: %IMG% -^> раздел system
echo ВНИМАНИЕ: все данные на телефоне будут стёрты
set /p ANSWER=Продолжить? Введите YES: 
if not "%ANSWER%"=="YES" (echo Отменено & exit /b 1)

if not "%VBMETA%"=="" fastboot --disable-verity --disable-verification flash vbmeta "%VBMETA%"

for /f "tokens=2 delims=: " %%a in ('fastboot getvar is-userspace 2^>^&1 ^| findstr /b "is-userspace:"') do set "USERSPACE=%%a"
if /I not "%USERSPACE%"=="yes" (
  echo Перехожу в fastbootd...
  fastboot reboot fastboot
  timeout /t 5 >nul
)

fastboot erase system
fastboot flash system "%IMG%"
if errorlevel 1 (
  echo Не хватило места: удаляю раздел product и пробую снова...
  for /f "tokens=2 delims=: " %%a in ('fastboot getvar current-slot 2^>^&1 ^| findstr /b "current-slot:"') do set "SLOT=_%%a"
  fastboot delete-logical-partition product!SLOT!
  fastboot flash system "%IMG%" || exit /b 1
)
fastboot -w
fastboot reboot
echo Готово. Первое включение может занять несколько минут.
exit /b 0

:usage
echo Использование: flash.bat CoraxOS.img [vbmeta.img]
exit /b 2
