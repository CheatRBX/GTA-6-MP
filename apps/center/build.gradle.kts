// Corax Центр: экран «Добро пожаловать» после мастера LineageOS и плитка в Настройках.
// Пока это проба для шага 0.3 (без зависимостей): доказываем механику в эмуляторе.
// Настоящий интерфейс с анимациями делается после согласования макетов (шаг 2.0).
plugins {
    id("com.android.application")
}

android {
    namespace = "org.coraxos.center"
    compileSdk = 36

    defaultConfig {
        applicationId = "org.coraxos.center"
        minSdk = 35
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}
