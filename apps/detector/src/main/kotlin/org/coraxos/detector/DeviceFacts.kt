package org.coraxos.detector

/**
 * Факты об устройстве, из которых детектор делает выводы.
 * На телефоне их собирает [DeviceReport] из вывода `qa/probe.sh`
 * (или то же самое делает системный сервис при первом включении).
 */
data class DeviceFacts(
    val abi: String,
    val kernel: KernelVersion?,
    /** `ro.product.first_api_level`: с какой версией Android телефон вышел. */
    val firstApiLevel: Int,
    val trebleEnabled: Boolean,
    val abUpdate: Boolean,
    val dynamicPartitions: Boolean,
    /** Есть раздел `init_boot` (телефоны, вышедшие с Android 13+): root патчит его, а не `boot`. */
    val hasInitBoot: Boolean,
    /** Выбранные строки `/proc/config.gz`; пусто, если конфиг ядра прочитать не удалось. */
    val kernelConfig: Map<String, String> = emptyMap(),
    val ramMb: Int = 0,
    val manufacturer: String = "",
    val model: String = "",
    val soc: String = "",
) {
    val configKnown: Boolean get() = kernelConfig.isNotEmpty()

    /** `true`/`false`, если конфиг прочитан; `null`, если неизвестно. */
    fun configEnabled(name: String): Boolean? =
        if (!configKnown) null else kernelConfig[name].let { it == "y" || it == "m" }

    val androidVersionAtLaunch: String get() = apiToAndroid(firstApiLevel)

    companion object {
        fun apiToAndroid(api: Int): String = when (api) {
            in 0..27 -> "8 или старше"
            28 -> "9"
            29 -> "10"
            30 -> "11"
            31 -> "12"
            32 -> "12L"
            33 -> "13"
            34 -> "14"
            35 -> "15"
            36 -> "16"
            37 -> "17"
            else -> "API $api"
        }
    }
}
