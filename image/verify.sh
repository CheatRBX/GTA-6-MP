#!/usr/bin/env bash
# Сравнить два ext4-образа: какие файлы добавились, исчезли или изменились (путь, права, владелец, размер,
# SELinux-метка). Нужен root (монтирование только для чтения).
#
#   sudo image/verify.sh <исходный.img> <новый.img> [ожидаемые-изменения.txt]
#
# С третьим аргументом — список путей, которые ДОЛЖНЫ быть единственными отличиями; иначе код 1.
set -euo pipefail

a=$1
b=$2
expected=${3:-}
work=$(mktemp -d)
trap 'umount "$work/a" "$work/b" 2>/dev/null || true; rm -rf "$work"' EXIT
mkdir -p "$work/a" "$work/b"
mount -o loop,ro "$a" "$work/a"
mount -o loop,ro "$b" "$work/b"

listing() {
  (cd "$1" && find . -xdev -printf '%p\t%m\t%U:%G\t%s\n' | while IFS=$'\t' read -r p m o s; do
    ctx=$(getfattr --absolute-names --only-values -n security.selinux -h "$p" 2>/dev/null | tr -d '\0' || true)
    printf '%s\t%s\t%s\t%s\t%s\n' "$p" "$m" "$o" "$s" "$ctx"
  done | sort)
}
listing "$work/a" > "$work/a.txt"
listing "$work/b" > "$work/b.txt"

# Изменившиеся пути (добавленные, удалённые или с другими атрибутами).
diff "$work/a.txt" "$work/b.txt" | grep -E '^[<>]' | cut -f1 | sed 's/^[<>] //' | sort -u > "$work/changed.txt" || true
echo "Отличий: $(wc -l < "$work/changed.txt")"
diff "$work/a.txt" "$work/b.txt" | grep -E '^[<>]' | head -50 || true

if [[ -n $expected ]]; then
  # Родительские папки добавленного файла меняют только размер — их не считаем.
  sort -u "$expected" > "$work/expected.txt"
  unexpected=$(comm -23 "$work/changed.txt" "$work/expected.txt" | while read -r p; do
    [[ -d "$work/b/$p" ]] || echo "$p"
  done)
  if [[ -n $unexpected ]]; then
    echo "НЕОЖИДАННЫЕ изменения:"; echo "$unexpected"; exit 1
  fi
  missing=$(comm -13 "$work/changed.txt" "$work/expected.txt")
  if [[ -n $missing ]]; then
    echo "Ожидались, но не найдены:"; echo "$missing"; exit 1
  fi
  echo "OK: отличия только ожидаемые"
fi
