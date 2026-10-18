# 生成一段**长英语句子**的 TTS 语音，专门用于验证「NLLB 提前收尾」的修复。
#
# 为什么不复用 make_asr_test_wav.ps1：那个脚本的文本是固定的短句，
# 而本脚本要测的正是**长句**——短句触发不了 EOS 救回判据，测了等于没测。
#
# 文本选用 39 词的复合从句：从句边界多，正是 NLLB「以为说完了」的重灾区。
# 注意设备端 ASR 输出的原文**没有标点**（SenseVoice 不吐标点），
# 所以实际喂给 NLLB 的是一长串连写文本——比带逗号的版本更难，更该测。
#
# 用法（必须用 Windows PowerShell 5.1：System.Speech 在 .NET Core 上不存在）：
#   powershell.exe -ExecutionPolicy Bypass -File tools\make_long_test_wav.ps1
param(
    [string]$Out = "$PSScriptRoot\..\测试数据\probe\asr_long.wav"
)
Add-Type -AssemblyName System.Speech

New-Item -ItemType Directory -Force -Path (Split-Path $Out) | Out-Null

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$culture = New-Object System.Globalization.CultureInfo "en-US"
$synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female,
                          [System.Speech.Synthesis.VoiceAge]::Adult, 0, $culture)

# 16kHz / 16bit / 单声道 —— 正好是 SenseVoice 要求的输入格式
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000, ([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen), ([System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile($Out, $fmt)

$text = "I think the most important thing that we need to understand about this whole situation is that nobody really knows what is going to happen next, and that is exactly why we should all be a little more careful."
Write-Host "GROUND TRUTH: $text"
$synth.Speak($text)
$synth.Dispose()

$fi = Get-Item $Out
$secs = ($fi.Length - 44) / (16000 * 2)
Write-Host ("WAV: {0}  {1} bytes  ~{2:N2} s" -f $fi.FullName, $fi.Length, $secs)

# 剥掉 44 字节 WAV 头输出裸 PCM（设备端不必再解析 WAV）
$raw = [System.IO.File]::ReadAllBytes($Out)
$pcm = New-Object byte[] ($raw.Length - 44)
[Array]::Copy($raw, 44, $pcm, 0, $pcm.Length)
$pcmPath = [System.IO.Path]::ChangeExtension($Out, ".pcm")
[System.IO.File]::WriteAllBytes($pcmPath, $pcm)
Write-Host ("PCM: {0}  {1} bytes (16kHz mono s16le)" -f $pcmPath, $pcm.Length)
