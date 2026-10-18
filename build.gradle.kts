// 插件版本集中声明（与 settings.gradle.kts 的镜像仓库配合）
//
// 版本组合依据（已实测）：
//  - wrapper 发行版为 Gradle 9.3.1（本机 ~/.gradle/wrapper/dists 已有缓存，无需下载）
//  - AGP 8.x 与 Gradle 9 不兼容，因此必须用 AGP 9.x
//  - Kotlin 与 Compose 编译器插件版本必须严格一致
plugins {
    id("com.android.application") version "9.3.1" apply false
    // AGP 9.x 起 Kotlin 支持内置，不再声明 org.jetbrains.kotlin.android
    // Compose 编译器插件版本必须与 AGP 内置的 Kotlin 版本一致
    id("org.jetbrains.kotlin.plugin.compose") version "2.4.10" apply false
}
