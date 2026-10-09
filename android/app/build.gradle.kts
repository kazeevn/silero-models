plugins {
    // Kotlin is compiled by AGP's built-in Kotlin support
    id("com.android.application")
}

val modelAssets = file("src/main/assets/silero")

android {
    namespace = "io.github.kazeevn.silerotts"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "io.github.kazeevn.silerotts"
        // built for a Pixel 8a running Android 17
        minSdk = 37
        targetSdk = 37
        versionCode = 4
        versionName = "2.1.1-v5_5_ru"
        // Tensor G3 is arm64-only; drop the other ABIs of the LiteRT runtime
        // (-Psilero.abi=x86_64 builds an APK for smoke tests on the emulator)
        ndk { abiFilters += (findProperty("silero.abi") ?: "arm64-v8a").toString() }
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
        // lookup tables are memory-mapped straight from the APK; the .tflite
        // models stay compressed, they are extracted once (see LiteRtLoader)
        noCompress += listOf("bin", "tsv", "json")
    }

    packaging {
        // keep the LiteRT libraries uncompressed and page-aligned in the APK
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

dependencies {
    implementation(project(":core"))
    // LiteRT CompiledModel API (CPU / XNNPACK)
    implementation("com.google.ai.edge.litert:litert:2.3.0")
}

val checkModelAssets by tasks.registering {
    doLast {
        if (!file("$modelAssets/config.json").exists() || !file("$modelAssets/vocoder.tflite").exists()) {
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
