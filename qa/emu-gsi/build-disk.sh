#!/usr/bin/env bash
# Собрать системный диск эмулятора, в котором раздел system заменён нашим GSI.
#
#   qa/emu-gsi/build-disk.sh <папка system-image эмулятора> <наш system.img (raw ext4)> <выходная папка>
#
# Системный диск эмулятора — GPT с разделами vbmeta и super. Мы распаковываем super,
# подменяем system, собираем super заново (lpmake), ставим vbmeta с отключённой проверкой
# и обновляем VerifiedBootParams.textproto. Нужны: sgdisk, docker (android-tools из Alpine).
set -euo pipefail

src=$(realpath "$1")
gsi=$(realpath "$2")
out=$3
work=$(realpath -m work-emu)
rm -rf "$work" "$out"
mkdir -p "$work/parts" "$out"
cp -r "$src"/. "$out"/
out=$(realpath "$out")

echo "== Разделы исходного диска"
sgdisk -p "$src/system.img"
start=$(sgdisk -i 2 "$src/system.img" | awk '/First sector/ {print $3}')
end=$(sgdisk -i 2 "$src/system.img" | awk '/Last sector/ {print $3}')
dd if="$src/system.img" of="$work/super.img" bs=512 skip="$start" count=$((end - start + 1)) status=none

echo "== super: распаковка, замена system, сборка"
docker run --rm -v "$work:/w" -v "$gsi:/gsi.img:ro" -w /w alpine:3.22 sh -euc '
  apk add --no-cache android-tools python3 >/dev/null
  lpdump super.img > lpdump.txt
  lpunpack super.img parts
  cp /gsi.img parts/system.img
  args=""; total=0
  for f in parts/*.img; do
    name=$(basename "$f" .img)
    size=$(stat -c %s "$f")
    size=$(( (size + 1048575) / 1048576 * 1048576 ))
    total=$((total + size))
    if [ "$size" -gt 0 ]; then
      args="$args --partition $name:readonly:$size:main --image $name=$f"
    else
      args="$args --partition $name:readonly:0:main"
    fi
    echo "  $name: $size"
  done
  super=$(( total + 64 * 1048576 ))
  lpmake --metadata-size 65536 --super-name super --metadata-slots 2 \
    --device super:$super --group main:$total $args --output super-new.img
  avbtool make_vbmeta_image --flags 2 --padding_size 4096 --output vbmeta.img
  avbtool calculate_vbmeta_digest --image vbmeta.img --hash_algorithm sha256 > vbmeta.digest
  avbtool info_image --image vbmeta.img > vbmeta.info
'
grep -E 'Name:|Size:' "$work/lpdump.txt" | head -30 || true

echo "== Новый диск"
super_sectors=$(( $(stat -c %s "$work/super-new.img") / 512 ))
rm -f "$out/system.img"
truncate -s $(( (4096 + super_sectors + 2048) * 512 )) "$out/system.img"
sgdisk --clear -n 1:2048:4095 -c 1:vbmeta -n 2:4096:$((4096 + super_sectors - 1)) -c 2:super "$out/system.img"
dd if="$work/vbmeta.img" of="$out/system.img" bs=512 seek=2048 conv=notrunc status=none
dd if="$work/super-new.img" of="$out/system.img" bs=512 seek=4096 conv=notrunc status=none
sgdisk -p "$out/system.img"

# Параметры проверенной загрузки, которые эмулятор передаёт ядру, должны совпадать с новым vbmeta.
vsize=$(awk '/Header Block:|Authentication Block:|Auxiliary Block:/ {s += $(NF-1)} END {print s}' "$work/vbmeta.info")
vdigest=$(tr -d '[:space:]' < "$work/vbmeta.digest")
cat > "$out/VerifiedBootParams.textproto" <<PROTO
major_version: 2
param: "androidboot.vbmeta.size=$vsize"
param: "androidboot.vbmeta.hash_alg=sha256"
param: "androidboot.vbmeta.digest=$vdigest"
PROTO
cat "$out/VerifiedBootParams.textproto"

# Эмулятор по умолчанию ведёт себя как телефон с ЗАБЛОКИРОВАННЫМ загрузчиком: такой не пускает
# систему с отключённой проверкой vbmeta (init: "PUBLIC_KEY_REJECTED isn't allowed").
# GSI ставят только на разблокированные телефоны — делаем эмулятор разблокированным.
printf ' androidboot.vbmeta.device_state=unlocked androidboot.verifiedbootstate=orange' >> "$out/kernel_cmdline.txt"
echo "kernel_cmdline.txt: $(cat "$out/kernel_cmdline.txt")"
rm -rf "$work"
echo "Готово: $out"
