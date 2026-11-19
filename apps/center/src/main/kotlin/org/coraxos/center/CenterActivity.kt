package org.coraxos.center

import android.app.Activity
import android.graphics.Color
import android.os.Bundle
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView

/**
 * «Настройки → Corax Центр» (проба шага 0.3): доказываем, что плитка появляется
 * на главном экране Настроек и открывает наш экран. Разделы — заготовки,
 * настоящий интерфейс делается после согласования макетов (шаг 2.0).
 */
class CenterActivity : Activity() {

    private val sections = listOf(
        "Root" to "Magisk · KernelSU · APatch · без root",
        "Сервисы Google" to "Нет · microG · Google Play",
        "Kali NetHunter" to "Редакция подбирается под ядро",
        "Драйверы USB Wi-Fi" to "rtl8822bu и другие",
        "Модули ядра" to "Полный контроль ядра",
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (20 * resources.displayMetrics.density).toInt()
        val list = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad * 2, pad, pad)
        }
        list.addView(TextView(this).apply {
            text = getString(R.string.center_title)
            textSize = 28f
            setTextColor(Color.parseColor("#E8EAF0"))
            setPadding(0, 0, 0, pad)
        })
        for ((title, summary) in sections) {
            list.addView(TextView(this).apply {
                text = title
                textSize = 18f
                setTextColor(Color.parseColor("#E8EAF0"))
                setPadding(0, pad / 2, 0, 0)
            })
            list.addView(TextView(this).apply {
                text = summary
                textSize = 14f
                setTextColor(Color.parseColor("#9AA3B2"))
                setPadding(0, 0, 0, pad / 2)
            })
        }
        setContentView(ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#07090D"))
            addView(list)
        })
    }
}
