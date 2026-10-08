import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("org.jetbrains.kotlin.jvm")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    // The Android app supplies onnxruntime-android (same ai.onnxruntime API);
    // the desktop build is only used to compile and to run the JVM tests.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.30.0")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.30.0")
    testImplementation(kotlin("test"))
}

tasks.test {
    // ./gradlew :core:test -Psilero.assets=<dir with exported assets> -Psilero.vectors=<vectors.json>
    systemProperty("silero.assets", (findProperty("silero.assets") ?: "${rootDir}/app/src/main/assets/silero").toString())
    for (key in listOf("silero.vectors", "silero.wavOut")) findProperty(key)?.let { systemProperty(key, it.toString()) }
    maxHeapSize = "2g"
    testLogging {
        events("passed", "skipped", "failed", "standardOut")
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
