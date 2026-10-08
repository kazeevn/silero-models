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
    // The networks run behind the Network interface: LiteRT CompiledModel in
    // the app, the LiteRT C API (libLiteRt.so of the ai-edge-litert wheel,
    // through JNA) in the desktop tests.
    testImplementation("net.java.dev.jna:jna:5.19.1")
    testImplementation(kotlin("test"))
}

tasks.test {
    // ./gradlew :core:test -Psilero.assets=<dir with exported assets> -Psilero.litert=<libLiteRt.so>
    //     -Psilero.vectors=<vectors.json>
    systemProperty("silero.assets", (findProperty("silero.assets") ?: "${rootDir}/app/src/main/assets/silero").toString())
    for (key in listOf("silero.litert", "silero.vectors", "silero.wavOut")) findProperty(key)?.let { systemProperty(key, it.toString()) }
    maxHeapSize = "2g"
    testLogging {
        events("passed", "skipped", "failed", "standardOut")
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
