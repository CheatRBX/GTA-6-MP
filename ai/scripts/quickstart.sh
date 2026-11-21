#!/usr/bin/env bash
# Полный цикл с нуля: данные → токенизатор → обучение → дообучение на диалогах → квантование → чат.
#   bash scripts/quickstart.sh                 # пресет nano, ~30 минут на Snapdragon 7 Gen 3
#   STEPS=3000 bash scripts/quickstart.sh      # дольше учится — лучше пишет
#   PRESET=small STEPS=4000 bash scripts/quickstart.sh
# Ctrl+C — пауза с сохранением; повторный запуск продолжит с того же места.
set -e
cd "$(dirname "$0")/.."
PRESET=${PRESET:-nano}
STEPS=${STEPS:-1500}
FT_STEPS=${FT_STEPS:-100}
QUANT=${QUANT:-q8}

[ -x ./corax ] || make
mkdir -p data/raw models
command -v termux-wake-lock >/dev/null 2>&1 && termux-wake-lock || true

if [ -z "$(find data/raw -maxdepth 1 \( -name '*.txt' -o -name '*.jsonl' \) 2>/dev/null | head -n 1)" ]; then
    echo "◆ 1/6 Скачиваю данные"
    bash scripts/get_data.sh ru-stories
fi

if [ ! -f data/train.bin ]; then
    echo "◆ 2/6 Учу токенизатор и готовлю данные (только русские тексты)"
    python3 tools/prepare.py pretrain data/raw --out-dir data --vocab 8192 --min-cyr 0.5
fi

if [ ! -f models/base.bin ] || [ -f models/base.bin.opt ]; then
    echo "◆ 3/6 Обучаю модель с нуля: $PRESET, $STEPS шагов"
    RESUME=""
    [ -f models/base.bin.opt ] && RESUME="--resume"
    ./corax train --preset "$PRESET" --data data/train.bin --val data/val.bin --tokenizer data/tokenizer.bin \
        --out models/base.bin --steps "$STEPS" --sample-prompt "Однажды" $RESUME
    rm -f models/base.bin.opt
fi

if [ -f "models/chat-$QUANT.bin" ] && [ "models/chat-$QUANT.bin" -nt models/base.bin ] && [ "models/chat-$QUANT.bin" -nt data/sft_example.jsonl ]; then
    echo "✔ Модель уже готова: ./corax chat -m models/chat-$QUANT.bin"
    exit 0
fi

echo "◆ 4/6 Готовлю диалоги для дообучения"
python3 tools/prepare.py sft data/sft_example.jsonl --tokenizer data/tokenizer.bin --out data/sft

echo "◆ 5/6 Дообучаю (учу отвечать как ассистент)"
FT_RESUME=""
[ -f models/chat.bin.opt ] && [ models/chat.bin -nt models/base.bin ] && FT_RESUME="--resume"
./corax finetune --model models/base.bin --data data/sft --out models/chat.bin --steps "$FT_STEPS" $FT_RESUME
rm -f models/chat.bin.opt

echo "◆ 6/6 Квантую ($QUANT) и проверяю скорость"
./corax quantize models/chat.bin "models/chat-$QUANT.bin" "$QUANT"
./corax bench -m "models/chat-$QUANT.bin"

echo
echo "✔ Готово! Запуск чата:  ./corax chat -m models/chat-$QUANT.bin"
