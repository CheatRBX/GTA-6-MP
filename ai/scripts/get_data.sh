#!/usr/bin/env bash
# Скачивание открытых текстов для обучения с нуля.
#   bash scripts/get_data.sh ru-stories   — ~5 тыс. коротких рассказов на русском (+ англ.), 19 МБ, CC-BY-SA-4.0
#   bash scripts/get_data.sh tinystories  — TinyStories (англ.), валидационная часть ~20 МБ
#   bash scripts/get_data.sh tinystories-full — TinyStories целиком, ~2.2 ГБ
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw
HF=https://huggingface.co/datasets
case "${1:-ru-stories}" in
  ru-stories)
    curl -L --fail -o data/raw/ru-stories.jsonl "$HF/georgiyozhegov/stories/resolve/main/dataset.jsonl"
    echo "✔ data/raw/ru-stories.jsonl (лицензия CC-BY-SA-4.0, автор датасета: georgiyozhegov)"
    ;;
  tinystories)
    curl -L --fail -o data/raw/tinystories-valid.txt "$HF/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
    echo "✔ data/raw/tinystories-valid.txt"
    ;;
  tinystories-full)
    curl -L --fail -o data/raw/tinystories-train.txt "$HF/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
    echo "✔ data/raw/tinystories-train.txt"
    ;;
  *)
    echo "неизвестный набор: $1 (есть: ru-stories, tinystories, tinystories-full)"; exit 1 ;;
esac
echo "Свои тексты тоже подходят: положи любые .txt / .jsonl в data/raw/"
