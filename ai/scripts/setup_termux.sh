#!/data/data/com.termux/files/usr/bin/bash
# Установка всего нужного в Termux и сборка corax.
set -e
cd "$(dirname "$0")/.."

echo "◆ Corax LM: установка пакетов Termux"
pkg update -y
pkg install -y clang make python curl

echo "◆ Сборка corax (оптимизация под ARM: NEON + dotprod)"
make clean >/dev/null 2>&1 || true
if ! make; then
    echo "сборка с dotprod не удалась — собираю общую версию"
    make generic
fi

# Проверка, что процессор понимает dotprod (иначе будет Illegal instruction)
if ! ./corax presets >/dev/null 2>&1; then
    echo "процессор без dotprod — пересобираю общую версию"
    make generic
fi

mkdir -p data models
echo
echo "✔ Готово. Дальше:"
echo "   bash scripts/quickstart.sh     # всё сразу: данные → обучение → дообучение → чат"
echo "   ./corax help                    # список команд"
echo
echo "Совет: чтобы телефон не усыплял Termux во время долгого обучения, выполни termux-wake-lock"
