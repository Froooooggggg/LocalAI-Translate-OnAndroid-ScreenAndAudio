package com.screenast.app.core

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/**
 * 设置页的两块小零件：分割线 + 可折叠说明。
 *
 * 背景（用户 2026-10-18 的原话）：
 * 「给每个设置选项间加个分割线，现在太乱了」+「把描述也做成一个图标，点一下才显示」。
 *
 * 设置页有十几个选项组，每组长说明都是三五行灰字，滑起来一片糊；
 * 而且同一屏里既有「翻译后端」这种大标题、也有「自动补标点」这种小行，层级看不出来。
 * 所以：组与组之间一根细线，说明默认折起来，只留一个「ⓘ 说明」，点开才展开。
 */

/** 选项组之间的分割线（上下各留 10dp，避免贴字）。 */
@Composable
fun SettingDivider() {
    Spacer(Modifier.height(10.dp))
    Box(
        Modifier
            .fillMaxWidth()
            .height(1.dp)
            .background(Color(0x1FFFFFFF)),
    )
    Spacer(Modifier.height(10.dp))
}

/**
 * 一段「说明」：默认只显示一个 ⓘ，点一下展开成灰字，再点收回。
 *
 * 签名和 `Text(text, fontSize = …, color = …)` 对齐，所以调用方把 `Text(` 换成
 * `SettingHint(` 就行，排版参数不用动。
 */
@Composable
fun SettingHint(
    text: String,
    fontSize: TextUnit = 11.sp,
    color: Color = Color(0xFF9E9EA7),
) {
    var expanded by remember(text) { mutableStateOf(false) }
    Column(Modifier.fillMaxWidth()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                if (expanded) "ⓘ 收起说明" else "ⓘ 说明",
                fontSize = 11.sp,
                color = Color(0xFF6F6C7C),
                modifier = Modifier.clickable { expanded = !expanded },
            )
        }
        if (expanded) {
            Text(text, fontSize = fontSize, color = color)
        }
    }
}
