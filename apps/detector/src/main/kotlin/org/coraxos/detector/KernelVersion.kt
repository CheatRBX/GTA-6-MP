package org.coraxos.detector

/**
 * Версия ядра из `uname -r`.
 *
 * Ядра GKI пишут в версии ветку KMI и поколение: `6.1.75-android14-11-g8d0e…`
 * → версия 6.1.75, Android 14, поколение 11, KMI `android14-6.1`.
 * Ядра производителей без GKI выглядят иначе (`4.19.157-perf+`): ветки KMI у них нет.
 */
data class KernelVersion(
    val major: Int,
    val minor: Int,
    val patch: Int,
    val androidRelease: Int?,
    val kmiGeneration: Int?,
    val raw: String,
) : Comparable<KernelVersion> {

    /** Ядро GKI 2.0: Android 12+ и Linux 5.10+, с веткой KMI в версии. */
    val isGki: Boolean
        get() = androidRelease != null && androidRelease >= 12 && kmiGeneration != null && atLeast(5, 10)

    /** Ветка KMI, например `android14-6.1`; null для ядер без GKI. */
    val kmi: String?
        get() = if (isGki) "android$androidRelease-$major.$minor" else null

    /** Наше ядро Corax (собирается в фазе 6) помечает версию суффиксом `-corax`. */
    val isCorax: Boolean
        get() = raw.contains("-corax")

    val short: String
        get() = "$major.$minor"

    fun atLeast(major: Int, minor: Int): Boolean =
        this.major > major || (this.major == major && this.minor >= minor)

    override fun compareTo(other: KernelVersion): Int =
        compareValuesBy(this, other, KernelVersion::major, KernelVersion::minor, KernelVersion::patch)

    companion object {
        private val VERSION = Regex("""^(\d+)\.(\d+)(?:\.(\d+))?""")
        private val ANDROID = Regex("""-android(\d+)-(\d+)(?=-|$)""")

        fun parse(raw: String): KernelVersion? {
            val text = raw.trim()
            val v = VERSION.find(text) ?: return null
            val android = ANDROID.find(text)
            return KernelVersion(
                major = v.groupValues[1].toInt(),
                minor = v.groupValues[2].toInt(),
                patch = v.groupValues[3].toIntOrNull() ?: 0,
                androidRelease = android?.groupValues?.get(1)?.toInt(),
                kmiGeneration = android?.groupValues?.get(2)?.toInt(),
                raw = text,
            )
        }
    }
}
