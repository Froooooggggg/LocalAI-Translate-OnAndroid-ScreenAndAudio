# 用 Windows 自带 TTS 生成一段**有标准答案**的语音，供听写链路自测。
# 之所以要真语音而不是正弦波：本测试的目的是验证「识别出来的字对不对」，
# 正弦波只能证明「没崩」，证明不了识别正确。
Add-Type -AssemblyName System.Speech

$out = "$PSScriptRoot\..\测试数据\probe\asr_test.wav"
New-Item -ItemType Directory -Force -Path (Split-Path $out) | Out-Null

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer

# 固定用 en-US 语音，句子与界面默认样例无关，避免「猜对」的假阳性
$culture = New-Object System.Globalization.CultureInfo "en-US"
$synth.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female,
                          [System.Speech.Synthesis.VoiceAge]::Adult, 0, $culture)

# 16kHz / 16bit / 单声道 —— 正好是 SenseVoice 要求的输入格式
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo 16000, ([System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen), ([System.Speech.AudioFormat.AudioChannel]::Mono)
$synth.SetOutputToWaveFile($out, $fmt)

$text = "The quick brown fox jumps over the lazy dog. Machine translation runs entirely on this phone."
Write-Host "GROUND TRUTH: $text"
$synth.Speak($text)
$synth.Dispose()

$fi = Get-Item $out
$secs = ($fi.Length - 44) / (16000 * 2)
Write-Host ("WAV: {0}  {1} bytes  ~{2:N2} s" -f $fi.FullName, $fi.Length, $secs)

# 顺手剥掉 44 字节 WAV 头，输出裸 PCM（设备端就不必再解析 WAV 了）
$raw = [System.IO.File]::ReadAllBytes($fi)
$pcm = New-Object byte[] ($raw.Length - 44)
[Array]::Copy($raw, 44, $pcm, 0, $pcm.Length)
$pcmPath = [System.IO.Path]::ChangeExtension($out, ".pcm")
[System.IO.File]::WriteAllBytes($pcmPath, $pcm)
Write-Host ("PCM: {0}  {1} bytes (16kHz mono s16le)" -f $pcmPath, $pcm.Length)
