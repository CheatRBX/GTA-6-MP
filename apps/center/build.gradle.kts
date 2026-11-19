// Corax Центр: экран «Добро пожаловать» после мастера LineageOS и раздел в Настройках.
// Интерфейс — Jetpack Compose по утверждённому макету (docs/prototype/first-boot.html).
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "org.coraxos.center"
    compileSdk = 37

    defaultConfig {
        applicationId = "org.coraxos.center"
        minSdk = 35
        targetSdk = 36
        versionCode = 2
        versionName = "0.2.0"
    }

    buildFeatures {
        compose = true
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

dependencies {
    implementation(project(":detector"))
    implementation(platform("androidx.compose:compose-bom:2026.09.00"))
    implementation("androidx.activity:activity-compose:1.12.4")
    implementation("androidx.compose.foundation:foundation")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.ui:ui")
}
