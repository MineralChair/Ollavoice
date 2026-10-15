/* OllaVoice browser UI.
 *
 * Deliberately dependency-free: no framework, no CDN, no external fonts.
 * The page is served by the local Python server and only ever talks to it, so
 * it works with the network unplugged.
 *
 * Microphone capture uses MediaRecorder (the encoded bytes go straight to
 * /v1/audio/transcriptions, where faster-whisper decodes them with its bundled
 * ffmpeg - no external tools needed). Answer audio comes back as base64 WAV and
 * is played through an <audio> element.
 */
"use strict";

const $ = (id) => document.getElementById(id);
const chat = $("chat");
const micBtn = $("mic");
const sendBtn = $("send");
const textInput = $("text");
const speakChk = $("speak");
const statusEl = $("status");
const meterFill = $("meterFill");

let recorder = null;
let chunks = [];
let recording = false;
let busy = false;
let audioCtx = null;
let analyser = null;
let meterRaf = null;

/* ------------------------------------------------------------------ */
/* rendering                                                           */
/* ------------------------------------------------------------------ */

function addMsg(kind, who, text) {
  const div = document.createElement("div");
  div.className = "msg " + kind;
  if (who) {
    const head = document.createElement("div");
    head.className = "who";
    head.textContent = who;
    div.appendChild(head);
  }
  const body = document.createElement("div");
  body.textContent = text;
  div.appendChild(body);
  chat.appendChild(div);
  chat.parentElement.scrollTop = chat.parentElement.scrollHeight;
  return body;
}

function setStatus(text) {
  statusEl.textContent = text;
}

function setBusy(value, label) {
  busy = value;
  micBtn.disabled = value;
  sendBtn.disabled = value;
  textInput.disabled = value;
  if (label) setStatus(label);
}

/* ------------------------------------------------------------------ */
/* health / config                                                     */
/* ------------------------------------------------------------------ */

async function refreshHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    $("dot").classList.toggle("on", !!data.ollama);
    $("srv").textContent = data.ollama ? "ollama online" : "OLLAMA OFFLINE";
    $("model").textContent = data.model || "(not set)";
    $("stt").textContent = `${data.stt.engine}/${data.stt.model}`;
    $("tts").textContent = `${data.tts.engine}${data.tts.voice ? "/" + data.tts.voice : ""}`;
  } catch (err) {
    $("dot").classList.remove("on");
    $("srv").textContent = "server unreachable";
  }
}
refreshHealth();
setInterval(refreshHealth, 15000);

/* ------------------------------------------------------------------ */
/* microphone                                                          */
/* ------------------------------------------------------------------ */

function pickMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/mp4",
  ];
  for (const c of candidates) {
    if (window.MediaRecorder && MediaRecorder.isTypeSupported(c)) return c;
  }
  return "";
}

async function startRecording() {
  if (recording || busy) return;
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, channelCount: 1 },
    });
  } catch (err) {
    addMsg("err", "", "Microphone access was denied: " + err.message +
      ". On Windows, allow browser apps under Settings > Privacy > Microphone, " +
      "and remember this page must be opened on http://127.0.0.1 (not a LAN IP) " +
      "for the browser to permit capture.");
    return;
  }

  chunks = [];
  const mimeType = pickMimeType();
  try {
    recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
  } catch (err) {
    addMsg("err", "", "This browser cannot record audio: " + err.message);
    stream.getTracks().forEach((t) => t.stop());
    return;
  }
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
  recorder.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    stopMeter();
    const blob = new Blob(chunks, { type: recorder.mimeType || mimeType || "audio/webm" });
    chunks = [];
    recording = false;
    micBtn.classList.remove("rec");
    micBtn.innerHTML = "&#127908; Hold to talk";
    if (blob.size > 1000) submitAudio(blob);
    else setStatus("recording was too short");
  };

  recorder.start(250);
  recording = true;
  micBtn.classList.add("rec");
  micBtn.innerHTML = "&#128308; Release to send";
  setStatus("listening...");
  startMeter(stream);
}

function stopRecording() {
  if (recorder && recording && recorder.state !== "inactive") recorder.stop();
}

function startMeter(stream) {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    const buf = new Uint8Array(analyser.frequencyBinCount);
    const tick = () => {
      if (!analyser) return;
      analyser.getByteTimeDomainData(buf);
      let peak = 0;
      for (let i = 0; i < buf.length; i++) peak = Math.max(peak, Math.abs(buf[i] - 128));
      meterFill.style.width = Math.min(100, (peak / 128) * 140) + "%";
      meterRaf = requestAnimationFrame(tick);
    };
    tick();
  } catch (err) { /* metering is cosmetic */ }
}

function stopMeter() {
  if (meterRaf) cancelAnimationFrame(meterRaf);
  meterRaf = null;
  meterFill.style.width = "0%";
}

/* ------------------------------------------------------------------ */
/* talking to the local server                                         */
/* ------------------------------------------------------------------ */

async function submitAudio(blob) {
  setBusy(true, "transcribing...");
  const form = new FormData();
  const ext = (blob.type || "").includes("ogg") ? "ogg" : (blob.type || "").includes("mp4") ? "m4a" : "webm";
  form.append("file", blob, "recording." + ext);
  form.append("response_format", "json");

  let transcript = "";
  try {
    const res = await fetch("/v1/audio/transcriptions", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error ? data.error.message : res.statusText);
    transcript = (data.text || "").trim();
  } catch (err) {
    setBusy(false);
    addMsg("err", "", "Transcription failed: " + err.message);
    return;
  }
  if (!transcript) {
    setBusy(false);
    addMsg("sys", "", "(nothing was recognised)");
    return;
  }
  addMsg("user", "You", transcript);
  await ask(transcript);
}

async function ask(text) {
  setBusy(true, "asking " + ($("model").textContent || "ollama") + "...");
  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, speak: false, response_audio: speakChk.checked }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error ? data.error.message : res.statusText);
    setBusy(false, data.timings ? `answered in ${data.timings.total_s}s (all local)` : "idle");
    addMsg("bot", "Ollama", data.text || "(empty answer)");
    if (data.audio_b64) playBase64Wav(data.audio_b64);
  } catch (err) {
    setBusy(false);
    addMsg("err", "", "Request failed: " + err.message);
  }
}

function playBase64Wav(b64) {
  const bytes = atob(b64);
  const buf = new Uint8Array(bytes.length);
  for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i);
  const blob = new Blob([buf], { type: "audio/wav" });
  const audio = new Audio(URL.createObjectURL(blob));
  audio.onended = () => URL.revokeObjectURL(audio.src);
  setStatus("speaking...");
  audio.play().then(() => {
    audio.onended = () => { URL.revokeObjectURL(audio.src); setStatus("idle"); };
  }).catch(() => setStatus("playback blocked by the browser - click the page once"));
}

/* ------------------------------------------------------------------ */
/* UI wiring                                                           */
/* ------------------------------------------------------------------ */

sendBtn.addEventListener("click", () => {
  const text = textInput.value.trim();
  if (!text || busy) return;
  textInput.value = "";
  addMsg("user", "You", text);
  ask(text);
});

textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendBtn.click();
});

/* Hold-to-talk with mouse, touch and the Space key. */
micBtn.addEventListener("mousedown", (e) => { e.preventDefault(); startRecording(); });
micBtn.addEventListener("mouseup", () => stopRecording());
micBtn.addEventListener("mouseleave", () => { if (recording) stopRecording(); });
micBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
micBtn.addEventListener("touchend", (e) => { e.preventDefault(); stopRecording(); });

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat) return;
  if (document.activeElement === textInput) return;
  e.preventDefault();
  startRecording();
});
document.addEventListener("keyup", (e) => {
  if (e.code !== "Space") return;
  if (document.activeElement === textInput) return;
  stopRecording();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") stopRecording();
});
