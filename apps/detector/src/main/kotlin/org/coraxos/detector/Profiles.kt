package org.coraxos.detector

/** «Для чего телефон?» — экран 4 мастера. Профиль только расставляет галочки, пользователь их меняет. */
enum class Profile { REGULAR, PENTESTER, DEVELOPER, PRIVACY }

data class Selection(
    val google: GoogleServices,
    val root: RootMethod,
    val netHunter: Boolean,
    val usbWifi: Boolean,
    /** «Полный контроль ядра»: загрузка своих модулей. Требует root (решение по вопросу 8 плана). */
    val kernelControl: Boolean,
)

object Profiles {
    /** Лучший доступный root: KernelSU, если поддержан полностью, иначе Magisk. */
    fun bestRoot(c: Compatibility): RootMethod =
        if (c.root[RootMethod.KERNELSU]?.support == Support.SUPPORTED) RootMethod.KERNELSU else RootMethod.MAGISK

    fun defaults(profile: Profile, c: Compatibility): Selection {
        val wifi = c.usbWifi.support == Support.SUPPORTED
        val modules = c.kernelModules.usable
        return when (profile) {
            Profile.REGULAR -> Selection(GoogleServices.GOOGLE_PLAY, RootMethod.NONE, false, false, false)
            Profile.PENTESTER -> Selection(GoogleServices.MICROG, bestRoot(c), true, wifi, modules)
            Profile.DEVELOPER -> Selection(GoogleServices.GOOGLE_PLAY, bestRoot(c), false, false, modules)
            Profile.PRIVACY -> Selection(GoogleServices.NONE, RootMethod.NONE, false, false, false)
        }
    }

    /** Проблемы выбора; пустой список — можно применять. Тексты показываются пользователю. */
    fun validate(s: Selection, c: Compatibility): List<String> = buildList {
        if (!c.device.usable) add(c.device.reason ?: "Устройство не поддерживается")
        val root = c.root[s.root]
        if (root != null && !root.usable) add("Root ${s.root}: ${root.reason}")
        if (s.usbWifi && !c.usbWifi.usable) add("USB Wi-Fi: ${c.usbWifi.reason}")
        if (s.kernelControl && s.root == RootMethod.NONE) add("Полный контроль ядра требует root")
        if (s.kernelControl && !c.kernelModules.usable) add("Полный контроль ядра: ${c.kernelModules.reason}")
    }
}
