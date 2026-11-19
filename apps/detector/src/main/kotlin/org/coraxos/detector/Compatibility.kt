package org.coraxos.detector

enum class Support { SUPPORTED, EXPERIMENTAL, UNAVAILABLE }

/** Вывод по одной функции: можно ли, и почему нет (текст показывается пользователю). */
data class Verdict(val support: Support, val reason: String? = null, val detail: String? = null) {
    val usable: Boolean get() = support != Support.UNAVAILABLE

    companion object {
        fun ok(detail: String? = null) = Verdict(Support.SUPPORTED, detail = detail)
        fun experimental(reason: String, detail: String? = null) = Verdict(Support.EXPERIMENTAL, reason, detail)
        fun no(reason: String) = Verdict(Support.UNAVAILABLE, reason)
    }
}

enum class RootMethod { MAGISK, KERNELSU, APATCH, NONE }

enum class GoogleServices { NONE, MICROG, GOOGLE_PLAY }

enum class NetHunterEdition { FULL, LITE, ROOTLESS }

data class Compatibility(
    val device: Verdict,
    val root: Map<RootMethod, Verdict>,
    /** Какой раздел патчить для root: `init_boot` или `boot`. */
    val bootTarget: String,
    val usbWifi: Verdict,
    val usbWifiChips: List<String>,
    val netHunterFull: Verdict,
    /** Загрузка своих модулей ядра («Полный контроль ядра»). Root проверяется отдельно, в [Profiles.validate]. */
    val kernelModules: Verdict,
) {
    /** Редакция Kali NetHunter, которую ставим при выбранном способе root. */
    fun netHunterEdition(root: RootMethod): NetHunterEdition = when {
        root == RootMethod.NONE -> NetHunterEdition.ROOTLESS
        netHunterFull.support == Support.SUPPORTED -> NetHunterEdition.FULL
        else -> NetHunterEdition.LITE
    }
}

object CompatibilityEngine {
    /** Самое новое ядро, на котором APatch заявлен как рабочий (по README APatch; перепроверять при обновлениях). */
    val APATCH_TESTED_MAX = KernelVersion(6, 12, 0, null, null, "6.12")

    private val RTW88_USB_6_6 = listOf("RTL8822BU", "RTL8821CU", "RTL8822CU")
    private val RTW88_USB_6_18 = RTW88_USB_6_6 + listOf("RTL8821AU", "RTL8812AU", "RTL8814AU")

    fun evaluate(f: DeviceFacts): Compatibility {
        val bootTarget = if (f.hasInitBoot) "init_boot" else "boot"
        val (usbWifi, chips) = usbWifi(f)
        return Compatibility(
            device = device(f),
            root = mapOf(
                RootMethod.MAGISK to Verdict.ok("патчит раздел $bootTarget"),
                RootMethod.KERNELSU to kernelSu(f),
                RootMethod.APATCH to aPatch(f),
                RootMethod.NONE to Verdict.ok(),
            ),
            bootTarget = bootTarget,
            usbWifi = usbWifi,
            usbWifiChips = chips,
            netHunterFull = netHunterFull(f),
            kernelModules = kernelModules(f),
        )
    }

    private fun kernelName(f: DeviceFacts) = f.kernel?.short ?: "неизвестно"

    private fun device(f: DeviceFacts): Verdict = when {
        f.abi != "arm64-v8a" -> Verdict.no("Нужен 64-битный процессор (arm64-v8a), у вас ${f.abi}")
        !f.trebleEnabled -> Verdict.no("Телефон не поддерживает Project Treble")
        f.firstApiLevel < 30 -> Verdict.no(
            "Телефон вышел с Android ${f.androidVersionAtLaunch}, нужна заводская прошивка Android 11 или новее",
        )
        f.ramMb in 1 until 3500 -> Verdict.experimental("Мало оперативной памяти: ${f.ramMb} МБ, рекомендуем от 4 ГБ")
        else -> Verdict.ok()
    }

    private fun kernelSu(f: DeviceFacts): Verdict {
        val k = f.kernel
        return when {
            f.configEnabled("CONFIG_KSU") == true -> Verdict.ok("KernelSU уже встроен в ядро")
            k != null && k.isGki -> Verdict.ok("модуль ядра (LKM) для ${k.kmi}")
            else -> Verdict.no("Нужно ядро GKI 5.10+ (у вас ${kernelName(f)}) или ядро Corax со встроенным KernelSU")
        }
    }

    private fun aPatch(f: DeviceFacts): Verdict {
        val k = f.kernel ?: return Verdict.experimental("Версия ядра неизвестна")
        if (!k.atLeast(3, 18)) return Verdict.no("APatch работает на ядрах 3.18+, у вас ${k.short}")
        if (f.configEnabled("CONFIG_KALLSYMS") == false || f.configEnabled("CONFIG_KALLSYMS_ALL") == false) {
            return Verdict.no("В ядре выключены CONFIG_KALLSYMS / CONFIG_KALLSYMS_ALL, без них APatch не работает")
        }
        if (k > APATCH_TESTED_MAX) return Verdict.experimental("APatch на ядре ${k.short} ещё не проверен")
        if (!f.configKnown) return Verdict.experimental("Не удалось прочитать конфиг ядра, совместимость не проверена")
        return Verdict.ok("патчит раздел boot")
    }

    private fun usbWifi(f: DeviceFacts): Pair<Verdict, List<String>> {
        val k = f.kernel ?: return Verdict.no("Версия ядра неизвестна") to emptyList()
        if (f.configEnabled("CONFIG_MODULES") == false) {
            return Verdict.no("Ядро не умеет загружать модули") to emptyList()
        }
        val chips = when {
            k.atLeast(6, 18) -> RTW88_USB_6_18
            k.atLeast(6, 6) -> RTW88_USB_6_6
            else -> emptyList()
        }
        return when {
            chips.isNotEmpty() && k.isGki -> Verdict.ok("драйверы под ${k.kmi}") to chips
            chips.isNotEmpty() -> Verdict.experimental("Ядро не GKI: драйверы под него не собраны заранее") to chips
            k.atLeast(6, 1) -> Verdict.experimental("Ядро 6.1: нужен бэкпорт драйвера rtw88") to RTW88_USB_6_6
            else -> Verdict.no("Нужно ядро 6.6 или новее (у вас ${k.short})") to emptyList()
        }
    }

    private fun netHunterFull(f: DeviceFacts): Verdict = when {
        f.kernel?.isCorax == true -> Verdict.ok("ядро Corax с поддержкой NetHunter")
        f.configEnabled("CONFIG_USB_CONFIGFS_F_HID") == true ->
            Verdict.experimental("HID-гаджет в ядре есть, инъекция Wi-Fi не проверена")
        else -> Verdict.no("Полному NetHunter нужно ядро с его поддержкой (ядро Corax); доступны Lite и Rootless")
    }

    private fun kernelModules(f: DeviceFacts): Verdict = when {
        f.configEnabled("CONFIG_MODULES") == false -> Verdict.no("Ядро не умеет загружать модули")
        f.configEnabled("CONFIG_MODULE_SIG_FORCE") == true ->
            Verdict.experimental("Ядро принимает только подписанные модули")
        else -> Verdict.ok()
    }
}
