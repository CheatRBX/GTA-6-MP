package org.coraxos.center

import android.content.Context
import android.os.Build
import org.coraxos.detector.DeviceFacts
import org.coraxos.detector.KernelVersion
import java.io.File

/**
 * Факты об этом телефоне для детектора — то, что приложение может прочитать само.
 * Точные данные (конфиг ядра, разделы) позже будет отдавать системная служба Corax;
 * до тех пор недостающее выводим из того, что видно.
 */
object DeviceFactsReader {

    fun read(): DeviceFacts {
        val firstApi = prop("ro.product.first_api_level").toIntOrNull() ?: Build.VERSION.SDK_INT
        return DeviceFacts(
            abi = Build.SUPPORTED_ABIS.firstOrNull().orEmpty(),
            kernel = KernelVersion.parse(System.getProperty("os.version").orEmpty()),
            firstApiLevel = firstApi,
            trebleEnabled = prop("ro.treble.enabled") == "true",
            abUpdate = prop("ro.build.ab_update") == "true",
            dynamicPartitions = prop("ro.boot.dynamic_partitions") == "true",
            // Раздел init_boot есть у телефонов, вышедших с Android 13+.
            hasInitBoot = exists("/dev/block/by-name/init_boot_a") || exists("/dev/block/by-name/init_boot") ||
                firstApi >= 33,
            ramMb = (File("/proc/meminfo").takeIf { it.canRead() }?.useLines { lines ->
                lines.firstOrNull { it.startsWith("MemTotal") }?.filter(Char::isDigit)?.toLongOrNull()
            } ?: 0L).div(1024).toInt(),
            manufacturer = Build.MANUFACTURER,
            model = Build.MODEL,
            soc = Build.SOC_MODEL,
        )
    }

    private fun exists(path: String) = runCatching { File(path).exists() }.getOrDefault(false)

    private fun prop(name: String): String = runCatching {
        val p = ProcessBuilder("getprop", name).redirectErrorStream(true).start()
        p.inputStream.bufferedReader().use { it.readText() }.trim().also { p.waitFor() }
    }.getOrDefault("")
}

/** Выбор пользователя. Сохраняется для системной службы Corax, которая будет его применять. */
data class Choice(
    val profile: org.coraxos.detector.Profile,
    val selection: org.coraxos.detector.Selection,
)

object ChoiceStore {
    private const val PREFS = "corax_choice"

    fun save(context: Context, choice: Choice) {
        val s = choice.selection
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
            .putString("profile", choice.profile.name)
            .putString("google", s.google.name)
            .putString("root", s.root.name)
            .putBoolean("nethunter", s.netHunter)
            .putBoolean("usb_wifi", s.usbWifi)
            .putBoolean("kernel_control", s.kernelControl)
            .putLong("saved_at", System.currentTimeMillis())
            .apply()
    }

    fun load(context: Context): Choice? {
        val p = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        if (!p.contains("saved_at")) return null
        return runCatching {
            Choice(
                profile = org.coraxos.detector.Profile.valueOf(p.getString("profile", null)!!),
                selection = org.coraxos.detector.Selection(
                    google = org.coraxos.detector.GoogleServices.valueOf(p.getString("google", null)!!),
                    root = org.coraxos.detector.RootMethod.valueOf(p.getString("root", null)!!),
                    netHunter = p.getBoolean("nethunter", false),
                    usbWifi = p.getBoolean("usb_wifi", false),
                    kernelControl = p.getBoolean("kernel_control", false),
                ),
            )
        }.getOrNull()
    }
}
