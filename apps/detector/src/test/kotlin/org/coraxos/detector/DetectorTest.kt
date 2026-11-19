package org.coraxos.detector

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

private fun report(name: String): DeviceFacts {
    val text = checkNotNull(DetectorTest::class.java.getResource("/reports/$name.txt")) { "нет отчёта $name" }.readText()
    return DeviceReport.parse(text)
}

private fun evaluate(name: String) = CompatibilityEngine.evaluate(report(name))

class KernelVersionTest {
    @Test
    fun `GKI version gives KMI branch`() {
        val k = assertNotNull(KernelVersion.parse("6.1.75-android14-11-g8d0e1f2a5b6c-ab12345678"))
        assertEquals(6, k.major)
        assertEquals(1, k.minor)
        assertEquals(75, k.patch)
        assertTrue(k.isGki)
        assertEquals("android14-6.1", k.kmi)
    }

    @Test
    fun `vendor kernel without KMI is not GKI`() {
        val k = assertNotNull(KernelVersion.parse("4.19.157-perf+"))
        assertFalse(k.isGki)
        assertNull(k.kmi)
    }

    @Test
    fun `5_4 with android tag is not GKI 2_0`() {
        val k = assertNotNull(KernelVersion.parse("5.4.210-android11-2-gabcdef"))
        assertFalse(k.isGki)
    }

    @Test
    fun `garbage is rejected`() {
        assertNull(KernelVersion.parse("not a kernel"))
    }

    @Test
    fun `corax suffix is detected`() {
        assertTrue(assertNotNull(KernelVersion.parse("6.12.81-android16-5-corax-20261201")).isCorax)
    }
}

class DeviceReportTest {
    @Test
    fun `config lines become CONFIG_ keys`() {
        val f = report("gki-android14-6.1")
        assertEquals("y", f.kernelConfig["CONFIG_KALLSYMS_ALL"])
        assertTrue(f.hasInitBoot)
        assertEquals(7800, f.ramMb)
    }

    @Test
    fun `missing required fields are reported`() {
        val e = assertFailsWith<IllegalArgumentException> { DeviceReport.parse("abi=arm64-v8a\n") }
        assertTrue(e.message!!.contains("kernel"))
        assertTrue(e.message!!.contains("first_api"))
    }
}

class CompatibilityTest {
    @Test
    fun `GKI 6_1 phone - everything but full NetHunter, USB Wi-Fi needs backport`() {
        val c = evaluate("gki-android14-6.1")
        assertEquals(Support.SUPPORTED, c.device.support)
        assertEquals("init_boot", c.bootTarget)
        assertEquals(Support.SUPPORTED, c.root.getValue(RootMethod.MAGISK).support)
        val ksu = c.root.getValue(RootMethod.KERNELSU)
        assertEquals(Support.SUPPORTED, ksu.support)
        assertTrue(ksu.detail!!.contains("android14-6.1"))
        assertEquals(Support.SUPPORTED, c.root.getValue(RootMethod.APATCH).support)
        assertEquals(Support.EXPERIMENTAL, c.usbWifi.support)
        assertEquals(Support.EXPERIMENTAL, c.netHunterFull.support)
        assertEquals(NetHunterEdition.LITE, c.netHunterEdition(RootMethod.KERNELSU))
        assertEquals(NetHunterEdition.ROOTLESS, c.netHunterEdition(RootMethod.NONE))
    }

    @Test
    fun `GKI 6_18 phone gets the extended USB Wi-Fi chip list`() {
        val c = evaluate("gki-android17-6.18")
        assertEquals(Support.SUPPORTED, c.usbWifi.support)
        assertTrue("RTL8822BU" in c.usbWifiChips)
        assertTrue("RTL8812AU" in c.usbWifiChips)
        assertEquals(Support.EXPERIMENTAL, c.root.getValue(RootMethod.APATCH).support, "6.18 новее проверенного APatch")
    }

    @Test
    fun `legacy 4_19 phone - Magisk and APatch only, no USB Wi-Fi`() {
        val c = evaluate("legacy-4.19-android11")
        assertEquals(Support.SUPPORTED, c.device.support)
        assertEquals("boot", c.bootTarget)
        val ksu = c.root.getValue(RootMethod.KERNELSU)
        assertEquals(Support.UNAVAILABLE, ksu.support)
        assertTrue(ksu.reason!!.contains("4.19"))
        assertEquals(Support.SUPPORTED, c.root.getValue(RootMethod.APATCH).support)
        assertEquals(Support.UNAVAILABLE, c.usbWifi.support)
        assertEquals(RootMethod.MAGISK, Profiles.bestRoot(c))
    }

    @Test
    fun `phone launched with Android 10 is below minimum`() {
        val c = evaluate("too-old-android10")
        assertEquals(Support.UNAVAILABLE, c.device.support)
        assertTrue(c.device.reason!!.contains("Android 10"))
    }

    @Test
    fun `32-bit phone is not supported`() {
        val c = evaluate("armv7")
        assertEquals(Support.UNAVAILABLE, c.device.support)
        assertTrue(c.device.reason!!.contains("armeabi-v7a"))
    }

    @Test
    fun `Corax kernel enables full NetHunter even without kernel config`() {
        val c = evaluate("corax-kernel-6.12")
        assertEquals(Support.SUPPORTED, c.netHunterFull.support)
        assertEquals(NetHunterEdition.FULL, c.netHunterEdition(RootMethod.KERNELSU))
        assertEquals(Support.EXPERIMENTAL, c.root.getValue(RootMethod.APATCH).support, "конфиг ядра неизвестен")
    }

    @Test
    fun `KernelSU built into kernel config wins over LKM`() {
        val f = report("legacy-4.19-android11").let { it.copy(kernelConfig = it.kernelConfig + ("CONFIG_KSU" to "y")) }
        val ksu = CompatibilityEngine.evaluate(f).root.getValue(RootMethod.KERNELSU)
        assertEquals(Support.SUPPORTED, ksu.support)
        assertTrue(ksu.detail!!.contains("встроен"))
    }

    @Test
    fun `APatch refuses kernels without kallsyms`() {
        val f = report("legacy-4.19-android11").let { it.copy(kernelConfig = it.kernelConfig + ("CONFIG_KALLSYMS_ALL" to "n")) }
        assertEquals(Support.UNAVAILABLE, CompatibilityEngine.evaluate(f).root.getValue(RootMethod.APATCH).support)
    }
}

class ProfilesTest {
    @Test
    fun `pentester on GKI gets KernelSU and NetHunter`() {
        val c = evaluate("gki-android17-6.18")
        val s = Profiles.defaults(Profile.PENTESTER, c)
        assertEquals(RootMethod.KERNELSU, s.root)
        assertTrue(s.netHunter)
        assertTrue(s.usbWifi)
        assertTrue(s.kernelControl)
        assertEquals(emptyList(), Profiles.validate(s, c))
    }

    @Test
    fun `pentester on legacy phone falls back to Magisk and no USB Wi-Fi`() {
        val c = evaluate("legacy-4.19-android11")
        val s = Profiles.defaults(Profile.PENTESTER, c)
        assertEquals(RootMethod.MAGISK, s.root)
        assertFalse(s.usbWifi)
        assertEquals(emptyList(), Profiles.validate(s, c))
    }

    @Test
    fun `regular and privacy profiles do not root`() {
        val c = evaluate("gki-android14-6.1")
        assertEquals(RootMethod.NONE, Profiles.defaults(Profile.REGULAR, c).root)
        assertEquals(GoogleServices.GOOGLE_PLAY, Profiles.defaults(Profile.REGULAR, c).google)
        assertEquals(GoogleServices.NONE, Profiles.defaults(Profile.PRIVACY, c).google)
    }

    @Test
    fun `kernel control without root is rejected`() {
        val c = evaluate("gki-android14-6.1")
        val s = Profiles.defaults(Profile.REGULAR, c).copy(kernelControl = true)
        assertTrue(Profiles.validate(s, c).any { it.contains("требует root") })
    }

    @Test
    fun `unavailable root choice is rejected with reason`() {
        val c = evaluate("legacy-4.19-android11")
        val s = Profiles.defaults(Profile.REGULAR, c).copy(root = RootMethod.KERNELSU)
        assertTrue(Profiles.validate(s, c).any { it.contains("GKI") })
    }
}
