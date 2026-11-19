package org.coraxos.center

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import org.coraxos.detector.Compatibility
import org.coraxos.detector.CompatibilityEngine
import org.coraxos.detector.DeviceFacts
import org.coraxos.detector.GoogleServices
import org.coraxos.detector.NetHunterEdition
import org.coraxos.detector.Profile
import org.coraxos.detector.Profiles
import org.coraxos.detector.RootMethod
import org.coraxos.detector.Support

enum class Screen { WELCOME, PROFILE, GOOGLE, ROOT, KITS, SUMMARY, APPLY, DONE }

object Labels {
    val profile = mapOf(
        Profile.REGULAR to ("Обычный" to "Всё как у всех, Google Play"),
        Profile.PENTESTER to ("Пентестер" to "Root, Kali NetHunter, USB Wi‑Fi"),
        Profile.DEVELOPER to ("Разработчик" to "Root и полный контроль ядра"),
        Profile.PRIVACY to ("Приватность" to "Без Google и без root"),
    )
    val google = mapOf(
        GoogleServices.NONE to ("Не устанавливать" to "Телефон без сервисов Google"),
        GoogleServices.MICROG to ("microG" to "Открытая замена сервисов Google"),
        GoogleServices.GOOGLE_PLAY to ("Google Play" to "Официальный пакет скачается с сайта его авторов"),
    )
    val root = mapOf(
        RootMethod.MAGISK to ("Magisk" to "Классика: модули, скрытие от приложений"),
        RootMethod.KERNELSU to ("KernelSU" to "Root на уровне ядра"),
        RootMethod.APATCH to ("APatch" to "Root через патч ядра"),
        RootMethod.NONE to ("Без root" to "Максимум совместимости"),
    )
    fun edition(e: NetHunterEdition) = when (e) {
        NetHunterEdition.FULL -> "Full"
        NetHunterEdition.LITE -> "Lite"
        NetHunterEdition.ROOTLESS -> "Rootless"
    }
}

fun describe(f: DeviceFacts): String {
    val k = f.kernel
    return when {
        k == null -> "ядро не определено"
        k.isGki -> "ядро ${k.short} · GKI ${k.kmi}"
        else -> "ядро ${k.short} · без GKI"
    }
}

/** Состояние мастера: профиль + выбор, всегда проверенный по совместимости. */
class SetupModel(val facts: DeviceFacts, initial: Choice?) {
    val compat: Compatibility = CompatibilityEngine.evaluate(facts)
    var profile by mutableStateOf(initial?.profile ?: Profile.REGULAR)
    var selection by mutableStateOf(initial?.selection ?: Profiles.defaults(Profile.REGULAR, compat))

    fun pickProfile(p: Profile) {
        profile = p
        selection = Profiles.defaults(p, compat)
    }

    fun rootSupport(r: RootMethod): Support = compat.root.getValue(r).support
    val edition: NetHunterEdition get() = compat.netHunterEdition(selection.root)
    val kernelControlAllowed: Boolean get() = selection.root != RootMethod.NONE && compat.kernelModules.usable

    /** Что будет сделано при применении, по порядку. */
    fun plan(): List<String> = buildList {
        val s = selection
        if (s.root != RootMethod.NONE) {
            add("Бэкап раздела ${compat.bootTarget}")
            add("Установка ${Labels.root.getValue(s.root).first}")
        }
        when (s.google) {
            GoogleServices.GOOGLE_PLAY -> add("Скачивание Google Play")
            GoogleServices.MICROG -> add("Установка microG")
            GoogleServices.NONE -> Unit
        }
        if (s.netHunter) add("Kali NetHunter ${Labels.edition(edition)}")
        if (s.usbWifi && compat.usbWifi.usable) add("Драйверы USB Wi‑Fi")
        if (s.kernelControl && kernelControlAllowed) add("Полный контроль ядра")
        add("Сохранение настроек")
    }
}

@Composable
fun WelcomeFlow(model: SetupModel, onFinished: () -> Unit) {
    var screen by remember { mutableStateOf(Screen.WELCOME) }
    val go: (Screen) -> Unit = { screen = it }
    Box(Modifier.fillMaxSize().background(Corax.bg).safeDrawingPadding()) {
        AnimatedContent(
            targetState = screen,
            transitionSpec = { (fadeIn() + slideInVertically { it / 20 }) togetherWith fadeOut() },
            label = "screen",
        ) { s ->
            when (s) {
                Screen.WELCOME -> WelcomeScreen { go(Screen.PROFILE) }
                Screen.PROFILE -> Page(1, "Для чего телефон?", "Расставим настройки под вас. Любой пункт можно поменять дальше.",
                    back = { go(Screen.WELCOME) }, next = { go(Screen.GOOGLE) }) { ProfileOptions(model) }
                Screen.GOOGLE -> Page(2, "Сервисы Google", "Нужны для Google Play, уведомлений многих приложений и входа через Google.",
                    back = { go(Screen.PROFILE) }, next = { go(Screen.ROOT) }) { GoogleOptions(model) }
                Screen.ROOT -> Page(3, "Root‑доступ", "Полный доступ к системе. Перед установкой сохраним копию раздела ${model.compat.bootTarget}.",
                    back = { go(Screen.GOOGLE) }, next = { go(Screen.KITS) }) { RootOptions(model) }
                Screen.KITS -> Page(4, "Наборы", "Инструменты под ваши задачи.",
                    back = { go(Screen.ROOT) }, next = { go(Screen.SUMMARY) }) { KitOptions(model) }
                Screen.SUMMARY -> Page(5, "Всё верно?", "Изменить можно и потом: Настройки → Corax Центр.",
                    back = { go(Screen.KITS) }, next = { go(Screen.APPLY) }, nextLabel = "Применить") { Summary(model) }
                Screen.APPLY -> ApplyScreen(model) { go(Screen.DONE) }
                Screen.DONE -> DoneScreen(model, onFinished)
            }
        }
    }
}

@Composable
private fun Page(
    step: Int?,
    title: String,
    lead: String?,
    back: (() -> Unit)?,
    next: () -> Unit,
    nextLabel: String = "Далее",
    content: @Composable ColumnScope.() -> Unit,
) {
    Column(Modifier.fillMaxSize().padding(horizontal = 22.dp, vertical = 16.dp)) {
        Column(
            Modifier.weight(1f).verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            StepHeader(step, title, lead)
            content()
        }
        VSpace(12.dp)
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            if (back != null) GhostButton("Назад", onClick = back)
            PrimaryButton(nextLabel, Modifier.weight(1f), onClick = next)
        }
    }
}

@Composable
fun ProfileOptions(model: SetupModel) {
    Labels.profile.forEach { (p, l) ->
        OptionCard(l.first, l.second, selected = model.profile == p) { model.pickProfile(p) }
    }
}

@Composable
fun GoogleOptions(model: SetupModel) {
    Labels.google.forEach { (g, l) ->
        OptionCard(l.first, l.second, selected = model.selection.google == g) {
            model.selection = model.selection.copy(google = g)
        }
    }
}

@Composable
fun RootOptions(model: SetupModel) {
    DeviceChip(describe(model.facts))
    Labels.root.forEach { (r, l) ->
        val verdict = model.compat.root.getValue(r)
        OptionCard(
            l.first, verdict.reason ?: l.second,
            selected = model.selection.root == r,
            enabled = verdict.usable,
            badge = verdict.support.takeIf { r != RootMethod.NONE },
        ) {
            val keepKernel = model.selection.kernelControl && r != RootMethod.NONE
            model.selection = model.selection.copy(root = r, kernelControl = keepKernel)
        }
    }
}

@Composable
fun KitOptions(model: SetupModel) {
    val s = model.selection
    val c = model.compat
    DeviceChip(describe(model.facts))
    val edition = Labels.edition(model.edition)
    val editionNote = when (model.edition) {
        NetHunterEdition.FULL -> "ядро поддерживает всё"
        NetHunterEdition.LITE -> "полной редакции нужно ядро Corax"
        NetHunterEdition.ROOTLESS -> "без root работает Rootless"
    }
    ToggleRow("Kali NetHunter", "Редакция: $edition — $editionNote", s.netHunter, enabled = true) {
        model.selection = s.copy(netHunter = it)
    }
    ToggleRow(
        "Драйверы USB Wi‑Fi",
        c.usbWifi.reason ?: c.usbWifiChips.joinToString(", ") { it.lowercase() },
        s.usbWifi, enabled = c.usbWifi.usable, badge = c.usbWifi.support,
    ) { model.selection = s.copy(usbWifi = it) }
    ToggleRow(
        "Полный контроль ядра",
        if (model.kernelControlAllowed) "Загрузка своих модулей и драйверов ядра" else "Нужен root — выберите его на прошлом шаге",
        s.kernelControl, enabled = model.kernelControlAllowed,
    ) { model.selection = s.copy(kernelControl = it) }
    if (s.kernelControl && model.kernelControlAllowed) {
        Text(
            "Свои модули ядра могут сломать загрузку. Откат — в Corax Центре.", color = Corax.warn, fontSize = 12.5.sp,
            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Corax.warn.copy(alpha = .08f)).padding(10.dp),
        )
    }
}

@Composable
fun Summary(model: SetupModel) {
    val s = model.selection
    val rows = listOf(
        "Профиль" to Labels.profile.getValue(model.profile).first,
        "Сервисы Google" to Labels.google.getValue(s.google).first,
        "Root" to Labels.root.getValue(s.root).first,
        "Kali NetHunter" to if (s.netHunter) Labels.edition(model.edition) else "Нет",
        "USB Wi‑Fi" to if (s.usbWifi && model.compat.usbWifi.usable) "Да" else "Нет",
        "Полный контроль ядра" to if (s.kernelControl && model.kernelControlAllowed) "Да" else "Нет",
    )
    rows.forEach { (k, v) ->
        Row(Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp)).background(Corax.surface).padding(horizontal = 14.dp, vertical = 12.dp)) {
            Text(k, color = Corax.muted, fontSize = 14.sp, modifier = Modifier.weight(1f))
            Text(v, color = Corax.fg, fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

@Composable
private fun WelcomeScreen(onStart: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(22.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
            CoraxLogo(140.dp)
            VSpace(18.dp)
            FadeIn(1250) { Wordmark() }
            VSpace(12.dp)
            FadeIn(1450) {
                Text("Полный контроль. Твой телефон — твои правила.", color = Corax.muted, fontSize = 14.sp, textAlign = TextAlign.Center)
            }
        }
        PrimaryButton("Начать", Modifier.fillMaxWidth(), onClick = onStart)
    }
}

@Composable
private fun ApplyScreen(model: SetupModel, onDone: () -> Unit) {
    val context = LocalContext.current
    val steps = remember { model.plan() }
    var current by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        // Пока системная служба Corax не готова (фаза 3 плана), здесь выбор только сохраняется:
        // установка root, GApps и наборов включится вместе со службой.
        ChoiceStore.save(context, Choice(model.profile, model.selection))
        for (i in steps.indices) { current = i; delay(700) }
        current = steps.size
        delay(500)
        onDone()
    }
    Column(Modifier.fillMaxSize().padding(22.dp), verticalArrangement = Arrangement.spacedBy(14.dp)) {
        StepHeader(null, "Почти готово", "Не выключайте телефон.")
        LinearProgressIndicator(
            progress = { current.toFloat() / steps.size },
            modifier = Modifier.fillMaxWidth().height(6.dp).clip(RoundedCornerShape(99.dp)),
            color = Corax.accent, trackColor = Corax.surface2,
        )
        steps.forEachIndexed { i, label ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    Modifier.size(20.dp).clip(CircleShape).background(
                        when {
                            i < current -> Corax.ok
                            i == current -> Corax.accent
                            else -> Corax.surface2
                        },
                    ),
                )
                Text(label, color = if (i <= current) Corax.fg else Corax.muted, fontSize = 14.sp, modifier = Modifier.padding(start = 10.dp))
            }
        }
    }
}

@Composable
private fun DoneScreen(model: SetupModel, onFinished: () -> Unit) {
    Column(Modifier.fillMaxSize().padding(22.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.Center, horizontalAlignment = Alignment.CenterHorizontally) {
            CoraxLogo(120.dp, animate = false)
            VSpace(18.dp)
            Text("Всё готово", color = Corax.fg, fontSize = 26.sp, fontWeight = FontWeight.Bold)
            VSpace(8.dp)
            Text(
                "Выбор сохранён в Corax Центре. Установка компонентов включится вместе со службой Corax.",
                color = Corax.muted, fontSize = 14.sp, textAlign = TextAlign.Center,
            )
        }
        PrimaryButton("На рабочий стол", Modifier.fillMaxWidth(), onClick = onFinished)
    }
}
