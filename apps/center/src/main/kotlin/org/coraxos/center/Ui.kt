package org.coraxos.center

import android.provider.Settings
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.Spring
import androidx.compose.animation.core.spring
import androidx.compose.animation.core.tween
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathMeasure
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import org.coraxos.detector.Support

/** Цвета Corax OS — те же, что в brand/brand.json и в макете. */
object Corax {
    val bg = Color(0xFF07090D)
    val surface = Color(0xFF11151C)
    val surface2 = Color(0xFF181D27)
    val fg = Color(0xFFE8EAF0)
    val muted = Color(0xFF8D96A8)
    val primary = Color(0xFF7C5CFF)
    val accent = Color(0xFF22D3EE)
    val ok = Color(0xFF34D399)
    val warn = Color(0xFFFBBF24)
    val no = Color(0xFFF87171)
    val brand = Brush.linearGradient(listOf(primary, accent))
}

/** Пользователь выключил анимации в системе — показываем сразу конечное состояние. */
@Composable
fun animationsOff(): Boolean {
    val ctx = LocalContext.current
    return remember { Settings.Global.getFloat(ctx.contentResolver, Settings.Global.ANIMATOR_DURATION_SCALE, 1f) == 0f }
}

/**
 * Логотип B (shard): шестиугольник без правой грани прорисовывается, затем «встаёт» ромб.
 * Геометрия совпадает с brand/logo/shard.svg (поле 512×512).
 */
@Composable
fun CoraxLogo(size: Dp, animate: Boolean = true) {
    val instant = !animate || animationsOff()
    val draw = remember { Animatable(if (instant) 1f else 0f) }
    val pop = remember { Animatable(if (instant) 1f else 0f) }
    LaunchedEffect(Unit) {
        if (instant) return@LaunchedEffect
        draw.animateTo(1f, tween(1100, delayMillis = 150, easing = CubicBezierEasing(.65f, 0f, .35f, 1f)))
        pop.animateTo(1f, spring(dampingRatio = Spring.DampingRatioMediumBouncy))
    }
    Canvas(Modifier.size(size)) {
        val k = this.size.minDimension / 512f
        val pts = listOf(407.55f to 343.5f, 256f to 431f, 104.45f to 343.5f, 104.45f to 168.5f, 256f to 81f, 407.55f to 168.5f)
        val full = Path().apply {
            moveTo(pts[0].first * k, pts[0].second * k)
            pts.drop(1).forEach { (x, y) -> lineTo(x * k, y * k) }
        }
        val measure = PathMeasure().apply { setPath(full, false) }
        val part = Path().also { measure.getSegment(0f, measure.length * draw.value, it, true) }
        val brush = Brush.linearGradient(listOf(Corax.primary, Corax.accent), Offset(80 * k, 440 * k), Offset(440 * k, 72 * k))
        drawPath(part, brush, style = Stroke(width = 54 * k, cap = StrokeCap.Round, join = StrokeJoin.Round))
        val cx = 407.55f * k
        val cy = 256f * k
        scale(pop.value, pivot = Offset(cx, cy)) {
            drawPath(Path().apply {
                moveTo(cx, cy - 40 * k); lineTo(cx + 40 * k, cy); lineTo(cx, cy + 40 * k); lineTo(cx - 40 * k, cy); close()
            }, Corax.accent)
        }
    }
}

@Composable
fun FadeIn(delayMs: Int, content: @Composable () -> Unit) {
    val instant = animationsOff()
    val a = remember { Animatable(if (instant) 1f else 0f) }
    LaunchedEffect(Unit) { if (!instant) a.animateTo(1f, tween(600, delayMillis = delayMs)) }
    Box(Modifier.alpha(a.value)) { content() }
}

@Composable
fun StepHeader(step: Int?, title: String, lead: String?) {
    Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
        if (step != null) Text("ШАГ $step ИЗ 5", color = Corax.accent, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 1.sp)
        Text(title, color = Corax.fg, fontSize = 26.sp, fontWeight = FontWeight.Bold, lineHeight = 31.sp)
        if (lead != null) Text(lead, color = Corax.muted, fontSize = 14.sp)
    }
}

@Composable
fun PrimaryButton(text: String, modifier: Modifier = Modifier, enabled: Boolean = true, onClick: () -> Unit) {
    Box(
        modifier
            .clip(RoundedCornerShape(16.dp))
            .background(if (enabled) Corax.brand else Brush.linearGradient(listOf(Corax.surface2, Corax.surface2)))
            .clickable(enabled = enabled, role = Role.Button, onClick = onClick)
            .padding(vertical = 15.dp, horizontal = 18.dp),
        contentAlignment = Alignment.Center,
    ) { Text(text, color = if (enabled) Color(0xFF06070A) else Corax.muted, fontWeight = FontWeight.SemiBold, fontSize = 15.sp) }
}

@Composable
fun GhostButton(text: String, modifier: Modifier = Modifier, onClick: () -> Unit) {
    Box(
        modifier.clip(RoundedCornerShape(16.dp)).background(Corax.surface2)
            .clickable(role = Role.Button, onClick = onClick).padding(vertical = 15.dp, horizontal = 18.dp),
        contentAlignment = Alignment.Center,
    ) { Text(text, color = Corax.fg, fontWeight = FontWeight.SemiBold, fontSize = 15.sp) }
}

@Composable
fun StatusBadge(support: Support) {
    val (label, color) = when (support) {
        Support.SUPPORTED -> "Подходит" to Corax.ok
        Support.EXPERIMENTAL -> "Экспериментально" to Corax.warn
        Support.UNAVAILABLE -> "Недоступно" to Corax.no
    }
    Text(
        label, color = color, fontSize = 11.sp, fontWeight = FontWeight.SemiBold,
        modifier = Modifier.clip(RoundedCornerShape(99.dp)).background(color.copy(alpha = .14f)).padding(horizontal = 8.dp, vertical = 3.dp),
    )
}

/** Карточка-вариант для выбора одного из нескольких. */
@Composable
fun OptionCard(
    title: String,
    summary: String,
    selected: Boolean,
    enabled: Boolean = true,
    badge: Support? = null,
    onClick: () -> Unit,
) {
    Column(
        Modifier.fillMaxWidth().alpha(if (enabled) 1f else .5f)
            .clip(RoundedCornerShape(18.dp))
            .background(if (selected) Corax.surface2 else Corax.surface)
            .border(BorderStroke(1.5.dp, if (selected) Corax.primary else Color.Transparent), RoundedCornerShape(18.dp))
            .clickable(enabled = enabled, role = Role.RadioButton, onClick = onClick)
            .padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(title, color = Corax.fg, fontWeight = FontWeight.SemiBold, fontSize = 15.sp, modifier = Modifier.weight(1f))
            if (badge != null) StatusBadge(badge)
        }
        Text(summary, color = Corax.muted, fontSize = 12.5.sp)
    }
}

@Composable
fun ToggleRow(title: String, summary: String, checked: Boolean, enabled: Boolean, badge: Support? = null, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().clip(RoundedCornerShape(18.dp)).background(Corax.surface).padding(14.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(title, color = Corax.fg, fontWeight = FontWeight.SemiBold, fontSize = 15.sp)
                if (badge != null) StatusBadge(badge)
            }
            Text(summary, color = Corax.muted, fontSize = 12.5.sp)
        }
        Spacer(Modifier.width(12.dp))
        Switch(
            checked = checked && enabled, enabled = enabled, onCheckedChange = onChange,
            colors = SwitchDefaults.colors(checkedTrackColor = Corax.primary, checkedThumbColor = Color.White),
        )
    }
}

@Composable
fun DeviceChip(text: String) {
    Text(
        "Ваш телефон: $text", color = Corax.muted, fontSize = 12.sp,
        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp)).background(Corax.surface).padding(horizontal = 12.dp, vertical = 8.dp),
    )
}

@Composable
fun Wordmark() {
    Text("CORAX", color = Corax.fg, fontSize = 22.sp, fontWeight = FontWeight.Bold, letterSpacing = 7.sp, textAlign = TextAlign.Center)
}

@Composable
fun VSpace(h: Dp) = Spacer(Modifier.height(h))
