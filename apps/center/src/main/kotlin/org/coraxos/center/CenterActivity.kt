package org.coraxos.center

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.coraxos.detector.RootMethod

/** «Настройки → Corax Центр»: те же пункты, что в мастере, в любой момент. */
class CenterActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val model = SetupModel(DeviceFactsReader.read(), ChoiceStore.load(this))
        setContent { CenterScreen(model) }
    }
}

private enum class Section(val title: String) { ROOT("Root"), GOOGLE("Сервисы Google"), KITS("Наборы"), DEVICE("Отчёт об устройстве") }

@Composable
private fun CenterScreen(model: SetupModel) {
    val context = LocalContext.current
    var open by remember { mutableStateOf<Section?>(null) }
    BackHandler(enabled = open != null) { open = null }
    Box(Modifier.fillMaxSize().background(Corax.bg).safeDrawingPadding()) {
        Column(Modifier.fillMaxSize().padding(horizontal = 20.dp, vertical = 16.dp)) {
            Column(Modifier.weight(1f).verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                val section = open
                if (section == null) {
                    StepHeader(null, "Corax Центр", null)
                    DeviceChip(describe(model.facts))
                    val s = model.selection
                    Group {
                        Entry("Root", "Сменить способ или удалить", Labels.root.getValue(s.root).first) { open = Section.ROOT }
                        Entry("Сервисы Google", "Нет · microG · Google Play", Labels.google.getValue(s.google).first) { open = Section.GOOGLE }
                        Entry("Kali NetHunter", "Редакция под ваше ядро", if (s.netHunter) Labels.edition(model.edition) else "Выкл") { open = Section.KITS }
                        Entry("Драйверы USB Wi‑Fi", "rtl8822bu и другие", if (s.usbWifi && model.compat.usbWifi.usable) "Вкл" else "Выкл") { open = Section.KITS }
                        Entry("Модули ядра", "Загрузить свой .ko, автозагрузка", if (s.kernelControl && model.kernelControlAllowed) "Вкл" else "Выкл") { open = Section.KITS }
                    }
                    Group {
                        Entry("Вернуть boot как было", "Из бэкапа перед установкой root", "", enabled = s.root != RootMethod.NONE) {}
                        Entry("Отчёт об устройстве", "Ядро, разделы, совместимость", "") { open = Section.DEVICE }
                    }
                } else {
                    StepHeader(null, section.title, null)
                    when (section) {
                        Section.ROOT -> RootOptions(model)
                        Section.GOOGLE -> GoogleOptions(model)
                        Section.KITS -> KitOptions(model)
                        Section.DEVICE -> DeviceReportView(model)
                    }
                }
            }
            if (open != null) {
                VSpace(12.dp)
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    GhostButton("Назад") { open = null }
                    if (open != Section.DEVICE) {
                        PrimaryButton("Сохранить", Modifier.weight(1f)) {
                            ChoiceStore.save(context, Choice(model.profile, model.selection))
                            open = null
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun Group(content: @Composable () -> Unit) {
    Column(Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)), verticalArrangement = Arrangement.spacedBy(2.dp)) { content() }
}

@Composable
private fun Entry(title: String, summary: String, value: String, enabled: Boolean = true, onClick: () -> Unit) {
    Row(
        Modifier.fillMaxWidth().background(Corax.surface).clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, color = if (enabled) Corax.fg else Corax.muted, fontSize = 14.5.sp, fontWeight = FontWeight.SemiBold)
            Text(summary, color = Corax.muted, fontSize = 12.sp)
        }
        Text(value, color = Corax.accent, fontSize = 12.sp)
    }
}

@Composable
private fun DeviceReportView(model: SetupModel) {
    val f = model.facts
    val c = model.compat
    val lines = listOf(
        "Модель" to "${f.manufacturer} ${f.model}",
        "Процессор" to f.soc.ifEmpty { "—" },
        "ABI" to f.abi,
        "Ядро" to (f.kernel?.raw ?: "—"),
        "Ветка KMI" to (f.kernel?.kmi ?: "нет (ядро не GKI)"),
        "Вышел с Android" to f.androidVersionAtLaunch,
        "Раздел для root" to c.bootTarget,
        "Устройство" to (c.device.reason ?: "поддерживается"),
    )
    lines.forEach { (k, v) ->
        Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(Corax.surface).padding(horizontal = 14.dp, vertical = 10.dp)) {
            Text(k, color = Corax.muted, fontSize = 13.sp, modifier = Modifier.weight(1f))
            Text(v, color = Corax.fg, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}
