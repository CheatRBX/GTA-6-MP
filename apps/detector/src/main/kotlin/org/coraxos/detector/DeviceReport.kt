package org.coraxos.detector

/**
 * Разбор отчёта `qa/probe.sh`: строки `ключ=значение`, комментарии начинаются с `#`.
 * Строки `config_ИМЯ=y` попадают в конфиг ядра как `CONFIG_ИМЯ`.
 */
object DeviceReport {
    private val REQUIRED = listOf("abi", "kernel", "first_api")

    fun parse(text: String): DeviceFacts {
        val values = mutableMapOf<String, String>()
        val config = mutableMapOf<String, String>()
        for (line in text.lineSequence()) {
            val trimmed = line.trim()
            if (trimmed.isEmpty() || trimmed.startsWith("#")) continue
            val eq = trimmed.indexOf('=')
            if (eq <= 0) continue
            val key = trimmed.substring(0, eq).trim()
            val value = trimmed.substring(eq + 1).trim()
            if (key.startsWith("config_")) {
                config["CONFIG_" + key.removePrefix("config_").uppercase()] = value
            } else {
                values[key] = value
            }
        }
        val missing = REQUIRED.filter { values[it].isNullOrEmpty() }
        require(missing.isEmpty()) { "В отчёте нет обязательных полей: ${missing.joinToString()}" }

        fun flag(key: String) = values[key]?.lowercase() in setOf("1", "true", "yes", "y")

        return DeviceFacts(
            abi = values.getValue("abi"),
            kernel = KernelVersion.parse(values.getValue("kernel")),
            firstApiLevel = values.getValue("first_api").toIntOrNull()
                ?: throw IllegalArgumentException("first_api должен быть числом: ${values["first_api"]}"),
            trebleEnabled = flag("treble"),
            abUpdate = flag("ab"),
            dynamicPartitions = flag("dynamic"),
            hasInitBoot = flag("init_boot"),
            kernelConfig = config,
            ramMb = values["ram_mb"]?.toIntOrNull() ?: 0,
            manufacturer = values["manufacturer"].orEmpty(),
            model = values["model"].orEmpty(),
            soc = values["soc"].orEmpty(),
        )
    }
}
