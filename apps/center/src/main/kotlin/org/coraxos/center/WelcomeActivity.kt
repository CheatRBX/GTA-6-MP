package org.coraxos.center

import android.app.Activity
import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.view.Gravity
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView

/**
 * Экран «Добро пожаловать» после мастера LineageOS (проба шага 0.3).
 *
 * Он зарегистрирован как HOME с приоритетом 1, поэтому система показывает его,
 * как только мастер Lineage закончил и отключил себя. По кнопке «Начать» экран
 * отключает свой компонент и уступает место обычному лаунчеру.
 * Выбор root / GApps / NetHunter появится здесь после согласования макетов.
 */
class WelcomeActivity : Activity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val pad = (24 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(Color.parseColor("#07090D"))
            setPadding(pad, pad, pad, pad)
        }
        root.addView(ImageView(this).apply {
            setImageResource(R.drawable.ic_corax)
            layoutParams = LinearLayout.LayoutParams(pad * 5, pad * 5)
        })
        root.addView(TextView(this).apply {
            text = getString(R.string.welcome_title)
            textSize = 26f
            setTextColor(Color.parseColor("#E8EAF0"))
            gravity = Gravity.CENTER
            setPadding(0, pad, 0, pad / 2)
        })
        root.addView(TextView(this).apply {
            text = getString(R.string.welcome_subtitle)
            textSize = 16f
            setTextColor(Color.parseColor("#9AA3B2"))
            gravity = Gravity.CENTER
        })
        root.addView(Button(this).apply {
            text = getString(R.string.welcome_start)
            isAllCaps = false
            setOnClickListener { finishWelcome() }
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.WRAP_CONTENT,
                LinearLayout.LayoutParams.WRAP_CONTENT,
            ).apply { topMargin = pad * 2 }
        })
        setContentView(root)
    }

    private fun finishWelcome() {
        packageManager.setComponentEnabledSetting(
            ComponentName(this, WelcomeActivity::class.java),
            PackageManager.COMPONENT_ENABLED_STATE_DISABLED,
            PackageManager.DONT_KILL_APP,
        )
        startActivity(
            Intent(Intent.ACTION_MAIN)
                .addCategory(Intent.CATEGORY_HOME)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
        )
        finish()
    }

    @Deprecated("Назад на первом экране не уводит в никуда")
    override fun onBackPressed() = Unit
}
