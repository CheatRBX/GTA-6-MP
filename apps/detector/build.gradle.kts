// Детектор устройства: чистая логика на Kotlin/JVM без Android SDK,
// чтобы тестироваться где угодно. Мастер и Центр управления подключают его как зависимость.
plugins {
    kotlin("jvm")
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    testImplementation(kotlin("test"))
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("passed", "failed")
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
