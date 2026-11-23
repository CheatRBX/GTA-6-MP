#!/usr/bin/env bash
# Полный цикл с нуля: данные → токенизатор → обучение → дообучение на диалогах → квантование → чат.
#   bash scripts/quickstart.sh                 # пресет nano, ~30 минут на Snapdragon 7 Gen 3
#   STEPS=3000 bash scripts/quickstart.sh      # дольше учится — лучше пишет
#   PRESET=small STEPS=4000 bash scripts/quickstart.sh
#   PRESET=m70 STEPS=20000 bash scripts/quickstart.sh   # 70 млн параметров (на телефоне — сутки и больше)
# Ctrl+C — пауза с сохранением; повторный запуск продолжит с того же места.
set -e
cd "$(dirname "$0")/.."
PRESET=${PRESET:-nano}
STEPS=${STEPS:-1500}
FT_STEPS=${FT_STEPS:-200}
QUANT=${QUANT:-q8}
case "$PRESET" in m70|medium|large) VOCAB=${VOCAB:-16384} ;; *) VOCAB=${VOCAB:-8192} ;; esac
case "$PRESET" in
  m70|medium|large) echo "⚠ $PRESET на телефоне учится очень долго (сутки и больше). Быстрее — colab/train_m70.ipynb. Ctrl+C — пауза." ;;
esac

# свои файлы для каждого словаря и пресета (nano — прежние имена)
DATA=data; [ "$VOCAB" != 8192 ] && DATA="data$((VOCAB / 1024))"
if [ "$PRESET" = nano ]; then BASE=models/base.bin; CHAT=models/chat.bin; else BASE="models/$PRESET-base.bin"; CHAT="models/$PRESET-chat.bin"; fi
CHATQ="${CHAT%.bin}-$QUANT.bin"

[ -x ./corax ] || make
mkdir -p data/raw models "$DATA"
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock || true

if [ -z "$(find data/raw -maxdepth 1 \( -name '*.txt' -o -name '*.jsonl' \) 2>/dev/null | head -n 1)" ]; then
    echo "◆ 1/6 Скачиваю данные"
    bash scripts/get_data.sh ru-stories
fi

if [ ! -f "$DATA/train.bin" ]; then
    echo "◆ 2/6 Учу токенизатор ($VOCAB) и готовлю данные (только русские тексты)"
    python3 tools/prepare.py pretrain data/raw --out-dir "$DATA" --vocab "$VOCAB" --min-cyr 0.5
fi

if [ ! -f "$BASE" ] || [ -f "$BASE.opt" ]; then
    echo "◆ 3/6 Обучаю модель с нуля: $PRESET, $STEPS шагов"
    RESUME=""
    [ -f "$BASE.opt" ] && RESUME="--resume"
    ./corax train --preset "$PRESET" --data "$DATA/train.bin" --val "$DATA/val.bin" --tokenizer "$DATA/tokenizer.bin" \
        --out "$BASE" --steps "$STEPS" --sample-prompt "Однажды" $RESUME
    rm -f "$BASE.opt"
fi

if [ -f "$CHATQ" ] && [ "$CHATQ" -nt "$BASE" ] && [ "$CHATQ" -nt data/sft_example.jsonl ]; then
    echo "✔ Модель уже готова: ./corax chat -m $CHATQ"
    exit 0
fi

echo "◆ 4/6 Готовлю диалоги для дообучения"
python3 tools/prepare.py sft data/sft_example.jsonl --tokenizer "$DATA/tokenizer.bin" --out "$DATA/sft"

echo "◆ 5/6 Дообучаю (учу отвечать как ассистент)"
FT_RESUME=""
[ -f "$CHAT.opt" ] && [ "$CHAT" -nt "$BASE" ] && FT_RESUME="--resume"
./corax finetune --model "$BASE" --data "$DATA/sft" --out "$CHAT" --steps "$FT_STEPS" $FT_RESUME
rm -f "$CHAT.opt"

echo "◆ 6/6 Квантую ($QUANT) и проверяю скорость"
./corax quantize "$CHAT" "$CHATQ" "$QUANT"
./corax bench -m "$CHATQ"

echo
echo "✔ Готово! Запуск чата:  ./corax chat -m $CHATQ"
