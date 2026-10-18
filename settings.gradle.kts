pluginManagement {
    repositories {
        // === 国内镜像：dl.google.com 能建连但会长时间挂起，比失败更糟 ===
        val aliyunGoogle = "https://maven.aliyun.com/repository/google"
        val aliyunPublic = "https://maven.aliyun.com/repository/public"
        val aliyunGradlePlugin = "https://maven.aliyun.com/repository/gradle-plugin"

        maven { url = uri(aliyunGoogle); name = "AliyunGoogle" }
        maven { url = uri(aliyunPublic); name = "AliyunPublic" }
        maven { url = uri(aliyunGradlePlugin); name = "AliyunGradlePlugin" }

        if (System.getenv("DSH_USE_OFFICIAL") == "1") {
            google()
            mavenCentral()
            gradlePluginPortal()
        }
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        val aliyunGoogle = "https://maven.aliyun.com/repository/google"
        val aliyunPublic = "https://maven.aliyun.com/repository/public"

        maven { url = uri(aliyunGoogle); name = "AliyunGoogle" }
        maven { url = uri(aliyunPublic); name = "AliyunPublic" }

        if (System.getenv("DSH_USE_OFFICIAL") == "1") {
            google()
            mavenCentral()
        }
    }
}

rootProject.name = "ScreenAudioTranslate"
include(":app")
