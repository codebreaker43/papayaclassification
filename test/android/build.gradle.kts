allprojects {
    repositories {
        google()
        mavenCentral()
        maven { url = uri("https://jitpack.io") }
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

subprojects {
    if (project.name == "flutter_tflite") {
        project.plugins.withId("com.android.library") {
            project.extensions.configure<com.android.build.gradle.BaseExtension> {
                namespace = "sq.flutter.tflite"
            }
        }
    }
    if (project.name == "flutter_native_splash") {
        project.plugins.withId("com.android.library") {
            project.extensions.configure<com.android.build.gradle.BaseExtension> {
                namespace = "net.jonhanson.flutter_native_splash"
            }
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
