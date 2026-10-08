import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val modelAssets = file("src/main/assets/silero")

android {
    namespace = "io.github.kazeevn.silerotts"
    compileSdk = 36

    defaultConfig {
        applicationId = "io.github.kazeevn.silerotts"
        // Pixel 8a ships with Android 14; nothing below 10 is needed for arm64 phones
        minSdk = 29
        targetSdk = 36
        versionCode = 1
        versionName = "1.0.0-v5_5_ru"
        // Tensor G3 is arm64-only; dropping the other ABIs of ONNX Runtime saves ~100 MB
        ndk { abiFilters += "arm64-v8a" }
    }

    signingConfigs {
        // Release signing from environment (CI) or ~/.gradle/gradle.properties;
        // falls back to the debug key so that a locally built APK installs.
        create("release") {
            val store = (findProperty("silero.keystore") ?: System.getenv("SILERO_KEYSTORE"))?.toString()
            if (store != null && file(store).exists()) {
                storeFile = file(store)
                storePassword = (findProperty("silero.keystore.password") ?: System.getenv("SILERO_KEYSTORE_PASSWORD"))?.toString()
                keyAlias = (findProperty("silero.key.alias") ?: System.getenv("SILERO_KEY_ALIAS"))?.toString()
                keyPassword = (findProperty("silero.key.password") ?: System.getenv("SILERO_KEY_PASSWORD"))?.toString()
            } else {
                val debug = getByName("debug")
                storeFile = debug.storeFile
                storePassword = debug.storePassword
                keyAlias = debug.keyAlias
                keyPassword = debug.keyPassword
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            signingConfig = signingConfigs.getByName("release")
        }
    }

    androidResources {
        // models and lookup tables are memory-mapped straight from the APK
        noCompress += listOf("onnx", "bin", "tsv", "json")
    }

    packaging {
        // keep libonnxruntime.so uncompressed and page-aligned in the APK
        // (loaded in place, not extracted to disk)
        jniLibs.useLegacyPackaging = false
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint {
        // the model assets are generated outside of Gradle
        disable += "MissingTranslation"
    }
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

dependencies {
    implementation(project(":core"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.30.0")
}

val checkModelAssets by tasks.registering {
    doLast {
        if (!file("$modelAssets/config.json").exists()) {
            throw GradleException(
                "Model assets are missing in $modelAssets.\n" +
                    "Generate them with:\n" +
                    "  python tools/export_models.py --out build/silero-export\n" +
                    "  python tools/optimize_models.py --src build/silero-export --out app/src/main/assets/silero\n" +
                    "(see README.md)",
            )
        }
    }
}
tasks.named("preBuild") { dependsOn(checkModelAssets) }

// The debug keystore (release fallback) is created by the debug signing validation.
tasks.matching { it.name == "validateSigningRelease" }.configureEach { dependsOn("validateSigningDebug") }
