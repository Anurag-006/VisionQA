import java.util.Properties

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

kotlin {
    compilerOptions {
        jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_11)
    }
}

val localProperties = Properties()
val localPropertiesFile = rootProject.file("local.properties")
if (localPropertiesFile.exists()) {
    localProperties.load(localPropertiesFile.inputStream())
}

android {
    namespace = "com.anurag.visionqa"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.anurag.visionqa"
        minSdk = 26              // MediaPipe LLM Inference API requires API 26+
        targetSdk = 36
        versionCode = 1
        versionName = "1.0"
        val hfToken = localProperties.getProperty("HF_TOKEN") ?: ""
        buildConfigField("String", "HF_TOKEN", "\"$hfToken\"")
        // NDK block removed — no more C++ / JNI
    }

    // externalNativeBuild block removed — CMakeLists.txt no longer needed

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    buildFeatures {
        buildConfig = true
        compose = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        // Prevent Gradle from compressing .litertlm files — MediaPipe requires
        // them to be memory-mapped directly off disk, not unzipped.
        jniLibs {
            useLegacyPackaging = false
        }
    }

    aaptOptions {
        noCompress += "litertlm"
        noCompress += "task"
        noCompress += "tflite"
    }
}

dependencies {
    // Core + Compose
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.6")
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)

    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)

    // CameraX — unchanged
    implementation("androidx.camera:camera-core:1.3.4")
    implementation("androidx.camera:camera-camera2:1.3.4")
    implementation("androidx.camera:camera-lifecycle:1.3.4")
    implementation("androidx.camera:camera-view:1.3.4")

    // ── NEW: MediaPipe LLM Inference API ──────────────────────────────────────
    // Replaces: onnxruntime-android, llama.cpp JNI, all .so files
    // This is the same runtime powering the Google AI Edge Gallery app.
    // Version 0.10.27 is the latest stable as of April 2026.
    implementation("com.google.mediapipe:tasks-genai:0.10.33")
    implementation("com.google.mediapipe:tasks-vision:0.10.21")
    // ─────────────────────────────────────────────────────────────────────────


    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // OkHttp for model download
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // OCR — unchanged
    implementation("com.google.mlkit:text-recognition:16.0.0")
    implementation("androidx.fragment:fragment-ktx:1.8.6")
    implementation("androidx.compose.material:material-icons-extended")

    // Tests — unchanged
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)

    // REMOVED: com.microsoft.onnxruntime:onnxruntime-android — no longer needed
}