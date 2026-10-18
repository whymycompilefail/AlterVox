/* AlterVox V1 前端：麦克风开关 + 能量 VAD 自动断句。
   打开麦克风后持续侦听：检测到说话开始，静音 0.8s 判定说完，
   自动把这一段送管线（ASR→TTS），返回的音频在页面内播放。
   播放期间门控麦克风（丢弃采样），防止合成声音被自己再录回去形成回环。 */

const ptt = document.getElementById("ptt");
const pttLabel = document.getElementById("pttLabel");
const vuBars = Array.from(document.querySelectorAll("#vu i"));
const statusEl = document.getElementById("status");
const transcriptEl = document.getElementById("transcript");
const logEl = document.getElementById("log");
const textInput = document.getElementById("textInput");
const sendText = document.getElementById("sendText");

const TARGET_SR = 16000;

// ---- VAD 参数（能量法，配合浏览器 AGC 使用）----
const VAD_START_RMS = 0.013;   // 超过视为开始说话
const VAD_END_SILENCE_MS = 800; // 静音多久判定说完
const VAD_MIN_SPEECH_MS = 250;  // 短于这个时长的“语音”视为噪声，丢弃
const VAD_MAX_UTT_MS = 25000;   // 单句上限，防跑飞
const VAD_HANGOVER_MS = 300;    // 播放结束后再门控这么久

let ws = null;
let micOpen = false;      // 麦克风总开关
let mediaStream = null;
let audioCtx = null;
let processor = null;
let sourceNode = null;

// VAD 状态机
let gated = false;        // 门控：true 时丢弃采样（播放期间/冷却期）
let inSpeech = false;
let speechBuf = [];       // 当前这句话的采样
let speechSamples = 0;
let silenceMs = 0;

let pendingCount = 0;     // 已发送、尚未播放完的句子数
let lastAudioUrl = null;
let queuedAudio = [];     // 待播放的 wav（base64），按顺序播

// ---------------------------------------------------------------- 状态显示
function setStatus(text, cls) {
  statusEl.className = cls || "";
  statusEl.innerHTML = `<span class="dot"></span>${text}`;
}

function refreshStatus() {
  if (!micOpen) { setStatus("麦克风已关闭"); return; }
  if (gated) { setStatus("播放中…（麦克风暂时让路）", "playing"); return; }
  if (pendingCount > 0) { setStatus(`处理中…（${pendingCount} 句排队）`, "busy"); return; }
  setStatus("侦听中… 直接说话即可", "");
}

// ---------------------------------------------------------------- WebSocket
function getWs() {
  return new Promise((resolve, reject) => {
    if (ws && ws.readyState === WebSocket.OPEN) return resolve(ws);
    ws = new WebSocket(`ws://${location.host}/ws`);
    ws.onopen = () => resolve(ws);
    ws.onerror = () => reject(new Error("无法连接后端，请确认 server.py 已启动"));
    ws.onclose = () => { ws = null; if (micOpen) setStatus("连接断开", "busy"); };
    ws.onmessage = (e) => {
      try { handleServerMsg(JSON.parse(e.data)); } catch (err) { console.error(err); }
    };
  });
}

// ---------------------------------------------------------------- 麦克风开关
async function openMic(auto = false) {
  if (micOpen) return;
  try {
    await getWs();
  } catch (e) {
    if (!auto) setStatus(e.message, "busy");
    return;
  }
  try {
    if (!audioCtx) audioCtx = new AudioContext();
  } catch (e) {
    if (!auto) setStatus("无法创建音频上下文: " + e.message, "busy");
    return;
  }
  // 自动播放策略可能让 AudioContext 挂起：不视为致命，等首次用户手势解锁
  if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
  try {

    // 回声消除开：页面自己播的合成声音会被浏览器从麦克分里消掉（第一道防线）
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (e) {
    // 自动开启失败（浏览器未授权/尚无手势）：安静回退，不报错吓用户
    if (!auto) setStatus("无法打开麦克风: " + e.message, "busy");
    return;
  }

  sourceNode = audioCtx.createMediaStreamSource(mediaStream);
  processor = audioCtx.createScriptProcessor(2048, 1, 1);
  processor.onaudioprocess = (e) => onAudioFrame(e.inputBuffer.getChannelData(0));
  sourceNode.connect(processor);
  processor.connect(audioCtx.destination); // 必须连到 destination，回调才会触发

  micOpen = true;
  ptt.classList.add("on");
  pttLabel.textContent = "关闭麦克风";
  resetVad();
  refreshStatus();
}

function closeMic() {
  micOpen = false;
  try { processor.disconnect(); } catch (e) {}
  try { sourceNode.disconnect(); } catch (e) {}
  processor = null;
  sourceNode = null;
  if (mediaStream) { mediaStream.getTracks().forEach((t) => t.stop()); mediaStream = null; }
  resetVad();
  updateVU(0);
  ptt.classList.remove("on");
  pttLabel.textContent = "打开麦克风";
  refreshStatus();
}

// ---------------------------------------------------------------- VU 电平表
let vuLevel = 0; // 平滑后的电平，避免抖动
function updateVU(rms) {
  // 对数感受映射：0.05 RMS 大约顶格
  const target = Math.min(1, Math.pow(rms / 0.05, 0.55));
  vuLevel += (target - vuLevel) * (target > vuLevel ? 0.55 : 0.12); // 快起慢落
  const lit = Math.round(vuLevel * vuBars.length);
  for (let i = 0; i < vuBars.length; i++) {
    const bar = vuBars[i];
    if (i < lit) {
      bar.classList.add("on");
      const frac = i / vuBars.length;
      bar.classList.toggle("mid", frac >= 0.55);
      bar.classList.toggle("hot", frac >= 0.82);
    } else {
      bar.classList.remove("on", "mid", "hot");
    }
  }
}

// ---------------------------------------------------------------- 能量 VAD
function resetVad() {
  inSpeech = false;
  speechBuf = [];
  speechSamples = 0;
  silenceMs = 0;
}

function rmsOf(f32) {
  let sum = 0;
  for (let i = 0; i < f32.length; i++) sum += f32[i] * f32[i];
  return Math.sqrt(sum / f32.length);
}

function onAudioFrame(f32) {
  if (!micOpen) return;

  const rms = rmsOf(f32);
  updateVU(rms);

  if (gated) return; // 播放期间丢弃，防止回环

  const frameMs = (f32.length / audioCtx.sampleRate) * 1000;

  if (!inSpeech) {
    if (rms > VAD_START_RMS) {
      inSpeech = true;
      silenceMs = 0;
      speechBuf = [new Float32Array(f32)];
      speechSamples = f32.length;
    }
    return;
  }

  // 在说话中
  speechBuf.push(new Float32Array(f32));
  speechSamples += f32.length;

  if (rms < VAD_START_RMS) {
    silenceMs += frameMs;
    if (silenceMs >= VAD_END_SILENCE_MS) {
      finishUtterance();
    }
  } else {
    silenceMs = 0;
  }

  if (speechSamples / audioCtx.sampleRate * 1000 >= VAD_MAX_UTT_MS) {
    finishUtterance(); // 到上限强制断句
  }
}

function finishUtterance() {
  inSpeech = false;
  const durMs = speechSamples / audioCtx.sampleRate * 1000;
  const buf = speechBuf;
  speechBuf = [];
  speechSamples = 0;
  silenceMs = 0;

  if (durMs < VAD_MIN_SPEECH_MS) return; // 太短，视为噪声

  const total = buf.reduce((n, c) => n + c.length, 0);
  const merged = new Float32Array(total);
  let off = 0;
  for (const c of buf) { merged.set(c, off); off += c.length; }
  // 掐掉句尾静音，减少 ASR 幻觉
  const trimmed = trimTailSilence(merged, audioCtx.sampleRate);
  const pcm16 = downsampleToInt16(trimmed, audioCtx.sampleRate, TARGET_SR);
  const b64 = floatToB64(pcm16);

  pendingCount++;
  refreshStatus();
  ws.send(JSON.stringify({ type: "utterance", pcm16_b64: b64, sample_rate: TARGET_SR }));
}

function trimTailSilence(f32, sr) {
  // 从尾部往前找最后一个超过阈值的点，向后留 100ms
  const thresh = VAD_START_RMS * 0.6;
  const win = Math.floor(sr * 0.02);
  let end = f32.length;
  for (let i = f32.length - win; i >= 0; i -= win) {
    if (rmsOf(f32.subarray(i, i + win)) > thresh) {
      end = Math.min(f32.length, i + win + Math.floor(sr * 0.1));
      break;
    }
  }
  return f32.subarray(0, end);
}

function downsampleToInt16(f32, fromRate, toRate) {
  const ratio = fromRate / toRate;
  const n = Math.floor(f32.length / ratio);
  const out = new Int16Array(n);
  for (let i = 0; i < n; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(i0 + 1, f32.length - 1);
    const frac = pos - i0;
    const v = f32[i0] * (1 - frac) + f32[i1] * frac;
    const s = Math.max(-1, Math.min(1, v));
    out[i] = s < 0 ? s * 32768 : s * 32767;
  }
  return out;
}

function floatToB64(int16arr) {
  const bytes = new Uint8Array(int16arr.buffer);
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
  }
  return btoa(bin);
}

// ---------------------------------------------------------------- 播放队列
function playWav(b64) {
  if (lastAudioUrl) URL.revokeObjectURL(lastAudioUrl);
  const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
  const blob = new Blob([bytes], { type: "audio/wav" });
  lastAudioUrl = URL.createObjectURL(blob);
  const audio = new Audio(lastAudioUrl);
  const p = audio.play();
  if (p) p.catch(() => {});
  return audio;
}

function drainQueue() {
  if (queuedAudio.length === 0) {
    gated = false;
    refreshStatus();
    return;
  }
  gated = true;
  refreshStatus();
  const item = queuedAudio.shift();
  const audio = playWav(item.wav_b64);
  const done = () => {
    pendingCount = Math.max(0, pendingCount - 1);
    setTimeout(drainQueue, VAD_HANGOVER_MS); // 冷却一小段再放开麦克风
  };
  audio.onended = done;
  audio.onerror = done;
}

// ---------------------------------------------------------------- 服务端消息
function handleServerMsg(msg) {
  if (msg.type === "asr") {
    transcriptEl.textContent = msg.text || "（未识别到内容）";
  } else if (msg.type === "audio") {
    addLog(msg.text, msg.asr_s, msg.tts_s, msg.wav_b64);
    queuedAudio.push(msg);
    if (!gated) drainQueue(); // 空闲则立即播；否则等当前播完
  } else if (msg.type === "error") {
    pendingCount = Math.max(0, pendingCount - 1);
    setStatus(msg.message, "busy");
    if (!gated) refreshStatus();
  }
}

// ---------------------------------------------------------------- 记录
function addLog(text, asrS, ttsS, b64) {
  const entry = document.createElement("div");
  entry.className = "entry";
  const meta = [];
  if (asrS != null) meta.push(`识别 ${asrS}s`);
  if (ttsS != null) meta.push(`合成 ${ttsS}s`);
  entry.innerHTML =
    `<div class="txt"></div><div class="meta"></div>` +
    (b64 ? `<button class="replay">重播</button>` : "");
  entry.querySelector(".txt").textContent = text;
  entry.querySelector(".meta").textContent = meta.join(" · ");
  if (b64) {
    entry.querySelector(".replay").onclick = () => playWav(b64);
  }
  logEl.prepend(entry);
}

// ---------------------------------------------------------------- 事件绑定
ptt.addEventListener("click", () => (micOpen ? closeMic() : openMic()));
window.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !e.repeat && document.activeElement !== textInput) {
    e.preventDefault();
    micOpen ? closeMic() : openMic();
  }
});

sendText.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) return;
  try {
    await getWs();
    setStatus("合成中…", "busy");
    transcriptEl.textContent = text;
    ws.send(JSON.stringify({ type: "text", text }));
  } catch (e) {
    setStatus(e.message, "busy");
  }
});
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendText.click();
});

refreshStatus();

// 使用体验：页面加载即自动开启麦克风。浏览器会弹权限询问；
// 若策略拦下（无手势/未授权），安静回退为手动开关，首次点击页面时解锁音频上下文。
openMic(true);
window.addEventListener("pointerdown", () => {
  if (audioCtx && audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
});
