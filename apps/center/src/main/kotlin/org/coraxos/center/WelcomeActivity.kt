package org.coraxos.center

import android.content.ComponentName
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.OnBackPressedCallback
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge

/**
 * Экран «Добро пожаловать» после мастера LineageOS.
 *
 * Зарегистрирован как HOME с приоритетом 1: система показывает его, как только мастер Lineage
 * закончил и отключил себя. В конце экран отключает свой компонент и уступает место лаунчеру.
 */
class WelcomeActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // «Назад» на первом включении никуда не уводит.
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() = Unit
        })
        val model = SetupModel(DeviceFactsReader.read(), ChoiceStore.load(this))
        setContent { WelcomeFlow(model, onFinished = ::finishWelcome) }
    }

    private fun finishWelcome() {
        packageManager.setComponentEnabledSetting(
            ComponentName(this, WelcomeActivity::class.java),
            PackageManager.COMPONENT_ENABLED_STATE_DISABLED,
            PackageManager.DONT_KILL_APP,
        )
        startActivity(Intent(Intent.ACTION_MAIN).addCategory(Intent.CATEGORY_HOME).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
        finish()
    }
}
