"""无需第三方依赖的 OpenDuck 实时波形页面。

树莓派只运行轻量 HTTP 服务并采集数据；浏览器应在同一局域网的电脑上打开。
页面通过增量轮询显示输入位置、实际位置、跟踪误差和电流。
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


JOINT_LABEL_ZH = {
    "left_hip_yaw": "左髋偏航",
    "left_hip_roll": "左髋横滚",
    "left_hip_pitch": "左髋俯仰",
    "left_knee": "左膝",
    "left_ankle": "左踝",
    "neck_pitch": "颈部俯仰",
    "head_pitch": "头部俯仰",
    "head_yaw": "头部偏航",
    "head_roll": "头部横滚",
    "right_hip_yaw": "右髋偏航",
    "right_hip_roll": "右髋横滚",
    "right_hip_pitch": "右髋俯仰",
    "right_knee": "右膝",
    "right_ankle": "右踝",
}


DASHBOARD_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenDuck Mini 实时波形</title>
<style>
body{font-family:"Microsoft YaHei",Arial,sans-serif;margin:0;background:#f3f6fa;color:#17202a}
header{background:#17365d;color:white;padding:16px 22px}
main{max-width:1180px;margin:18px auto;padding:0 16px}
.status,.panel{background:white;border:1px solid #d8e0e8;border-radius:10px;padding:14px;margin-bottom:14px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.wide{grid-column:1/-1}
canvas{width:100%;height:250px;border:1px solid #e2e8f0;background:white}
.wide canvas{height:360px}
.wave-title{font-size:clamp(18px,2.2vw,28px);line-height:1.45}
.wave-note{color:#475467;margin:6px 0 12px}
label{font-weight:700} select{font-size:16px;padding:5px 10px;margin-left:8px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}
.blue{background:#2563eb}.red{background:#dc2626}.purple{background:#7c3aed}.orange{background:#ea580c}
.downloads{display:flex;flex-wrap:wrap;gap:10px;margin-top:12px}
.download-button{display:inline-block;padding:10px 16px;border-radius:8px;background:#166534;color:#fff;text-decoration:none;font-weight:700}
.download-button:hover{background:#14532d}
@media(max-width:800px){.grid{grid-template-columns:1fr}.wide{grid-column:auto}}
</style>
</head>
<body>
<header><h1>OpenDuck Mini 实时波形</h1></header>
<main>
  <section class="status">
    <p><strong>状态：</strong><span id="status">等待数据</span></p>
    <p><strong>阶段：</strong><span id="stage">-</span></p>
    <p><strong>已采集时间：</strong><span id="elapsed">0.0 s</span></p>
    <label>关节<select id="joint"></select></label>
    <div id="downloads" class="downloads" hidden></div>
    <p>
      <span class="dot blue"></span>输入指定波形：发送的位置指令 θin(t)（蓝线）
      <span class="dot red"></span>舵机响应波形：反馈的实际输出位置 θout(t)（红线）
      <span class="dot purple"></span>跟踪误差
      <span class="dot orange"></span>电流
    </p>
  </section>
  <section class="grid">
    <div class="panel wide">
      <h2 class="wave-title"><span style="color:#2563eb">输入指定波形：位置指令 θin(t)（蓝线）</span> / <span style="color:#dc2626">舵机响应波形：实际输出位置 θout(t)（红线）</span></h2>
      <p class="wave-note">同一横坐标 t：蓝线为该采样时刻发送给舵机的位置指令，红线为随后读取的舵机位置反馈；两条曲线的横向错位表示响应滞后。</p>
      <canvas id="position"></canvas>
    </div>
    <div class="panel wide">
      <h2 class="wave-title">响应时间对齐后的目标/实际位置波形</h2>
      <p class="wave-note">红线按实时估计的响应延时沿时间轴向左平移；角度值保持不变，用于观察去除纯时间滞后后的幅值和波形误差。</p>
      <canvas id="positionAligned"></canvas>
    </div>
    <div class="panel"><h2>跟踪误差（°）</h2><canvas id="error"></canvas></div>
    <div class="panel"><h2>电流（A）</h2><canvas id="current"></canvas></div>
  </section>
</main>
<script>
const maxPoints = 1200;
let selectedJoint = "";
let lastSeq = 0;
let samples = [];
let currentStage = "";
const jointSelect = document.getElementById("joint");

function fitCanvas(canvas) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (canvas.width !== Math.floor(width * ratio) || canvas.height !== Math.floor(height * ratio)) {
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
  }
  return {ctx: canvas.getContext("2d"), width: canvas.width, height: canvas.height};
}

function estimateResponseMarker() {
  const valid = samples.filter(item =>
    Number.isFinite(Number(item.time_s)) &&
    Number.isFinite(Number(item.target_deg)) &&
    Number.isFinite(Number(item.actual_deg))
  );
  if (valid.length < 12) return null;
  const times = valid.map(item => Number(item.time_s));
  const target = valid.map(item => Number(item.target_deg));
  const actual = valid.map(item => Number(item.actual_deg));
  const timeSteps = [];
  for (let index = 1; index < times.length; index += 1) {
    const step = times[index] - times[index-1];
    if (step > 0) timeSteps.push(step);
  }
  if (!timeSteps.length) return null;
  timeSteps.sort((a,b) => a-b);
  const samplePeriod = timeSteps[Math.floor(timeSteps.length/2)];
  const maxLag = Math.min(
    Math.max(1, Math.round(0.5/samplePeriod)),
    Math.max(1, Math.floor(valid.length/3))
  );
  let bestLag = 0;
  let bestCorrelation = -2;
  for (let lag = -maxLag; lag <= maxLag; lag += 1) {
    const targetValues = [];
    const actualValues = [];
    for (let index = 0; index < valid.length; index += 1) {
      const actualIndex = index + lag;
      if (actualIndex < 0 || actualIndex >= valid.length) continue;
      targetValues.push(target[index]);
      actualValues.push(actual[actualIndex]);
    }
    if (targetValues.length < 8) continue;
    const targetMean = targetValues.reduce((sum,value) => sum+value,0)/targetValues.length;
    const actualMean = actualValues.reduce((sum,value) => sum+value,0)/actualValues.length;
    let numerator = 0, targetPower = 0, actualPower = 0;
    for (let index = 0; index < targetValues.length; index += 1) {
      const targetCentered = targetValues[index]-targetMean;
      const actualCentered = actualValues[index]-actualMean;
      numerator += targetCentered*actualCentered;
      targetPower += targetCentered*targetCentered;
      actualPower += actualCentered*actualCentered;
    }
    const denominator = Math.sqrt(targetPower*actualPower);
    if (denominator <= 1e-12) continue;
    const correlation = numerator/denominator;
    if (correlation > bestCorrelation) {
      bestCorrelation = correlation;
      bestLag = lag;
    }
  }
  const startValue = target[0];
  const minimum = Math.min(...target);
  const maximum = Math.max(...target);
  const extreme = Math.abs(maximum-startValue) >= Math.abs(minimum-startValue)
    ? maximum : minimum;
  const direction = extreme >= startValue ? 1 : -1;
  const excursion = Math.abs(extreme-startValue);
  const movementThreshold = Math.min(Math.max(0.5,excursion*0.02),excursion*0.25);
  let inputIndex = 1;
  for (let index = 1; index < valid.length; index += 1) {
    const movement = direction*(target[index]-startValue);
    if (movement >= movementThreshold) {
      inputIndex = index;
      break;
    }
  }
  const outputIndex = Math.max(0, Math.min(valid.length-1, inputIndex+bestLag));
  return {
    inputTime: times[inputIndex],
    outputTime: times[outputIndex],
    inputPosition: target[inputIndex],
    outputPosition: actual[outputIndex],
    delayMs: (times[outputIndex]-times[inputIndex])*1000
  };
}

function drawArrowHead(ctx, x, y, direction, ratio) {
  const size = 5*ratio;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x-direction*size, y-size*0.65);
  ctx.lineTo(x-direction*size, y+size*0.65);
  ctx.closePath();
  ctx.fill();
}

function drawChart(canvasId, series, yAxisLabel, options = {}) {
  const canvas = document.getElementById(canvasId);
  const {ctx, width, height} = fitCanvas(canvas);
  const ratio = window.devicePixelRatio || 1;
  ctx.clearRect(0, 0, width, height);
  const values = [];
  for (const item of samples) {
    for (const definition of series) {
      const value = Number(item[definition.field]);
      if (Number.isFinite(value)) values.push(value);
    }
  }
  if (samples.length < 2 || values.length === 0) {
    ctx.fillStyle = "#667085";
    ctx.font = `${16 * (window.devicePixelRatio || 1)}px sans-serif`;
    ctx.fillText("等待实时数据…", 20, 35);
    return;
  }
  let minY = Math.min(0, ...values);
  let maxY = Math.max(0, ...values);
  if (Math.abs(maxY - minY) < 1e-6) { minY -= 1; maxY += 1; }
  const padding = Math.max(0.1, (maxY - minY) * 0.08);
  minY -= padding; maxY += padding;
  const left = 78 * ratio, right = 22 * ratio;
  const isPositionChart = canvasId === "position" || canvasId === "positionAligned";
  // 为响应延时和输入函数预留独立顶部区域，避免覆盖曲线。
  const top = (canvasId === "position" ? 94 : (isPositionChart ? 76 : 26)) * ratio;
  const bottom = (canvasId === "position" ? 72 : 52) * ratio;
  const responseMarker = options.alignResponse ? estimateResponseMarker() : null;
  const responseShiftS = responseMarker === null ? 0 : Math.max(0, responseMarker.delayMs / 1000);
  const timeOrigin = Number(samples[0].time_s);
  const x0 = 0;
  const x1 = Number(samples[samples.length - 1].time_s)-timeOrigin;
  const spanX = Math.max(0.001, x1 - x0);
  ctx.font = `${11 * ratio}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let tick = 0; tick <= 6; tick += 1) {
    const fraction = tick / 6;
    const x = left + fraction * (width-left-right);
    const value = x0 + fraction * spanX;
    ctx.strokeStyle = "#e2e8f0";
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, height-bottom); ctx.stroke();
    ctx.fillStyle = "#475467";
    ctx.fillText(`${value.toFixed(1)}`, x, height-bottom+7*ratio);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let tick = 0; tick <= 4; tick += 1) {
    const fraction = tick / 4;
    const y = top + fraction * (height-top-bottom);
    const value = maxY - fraction * (maxY-minY);
    ctx.strokeStyle = "#e2e8f0";
    ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(width-right, y); ctx.stroke();
    ctx.fillStyle = "#475467";
    ctx.fillText(value.toFixed(2), left-8*ratio, y);
  }
  const zeroX = left + (0-x0)/spanX*(width-left-right);
  const zeroY = top + (maxY-0)/(maxY-minY)*(height-top-bottom);
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1.8*ratio;
  ctx.beginPath(); ctx.moveTo(zeroX,top); ctx.lineTo(zeroX,height-bottom); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(left,zeroY); ctx.lineTo(width-right,zeroY); ctx.stroke();
  ctx.fillStyle = "#334155";
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  ctx.fillText("0", zeroX+5*ratio, zeroY-4*ratio);
  for (const definition of series) {
    ctx.strokeStyle = definition.color; ctx.lineWidth = 2 * ratio; ctx.beginPath();
    let started = false;
    for (const item of samples) {
      const value = Number(item[definition.field]);
      if (!Number.isFinite(value)) continue;
      const plottedTime = Number(item.time_s)-timeOrigin -
        (options.alignResponse && definition.field === "actual_deg" ? responseShiftS : 0);
      if (plottedTime < x0 || plottedTime > x1) continue;
      const x = left + (plottedTime-x0)/spanX*(width-left-right);
      const y = top + (maxY-value)/(maxY-minY)*(height-top-bottom);
      if (!started) { ctx.moveTo(x,y); started = true; } else { ctx.lineTo(x,y); }
    }
    ctx.stroke();
  }
  if (isPositionChart && options.inputFunctionLabel) {
    ctx.font = `${12*ratio}px sans-serif`;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    const labelX = left+8*ratio;
    const labelY = top-29*ratio;
    const labelWidth = Math.min(
      width-left-right-16*ratio,
      ctx.measureText(options.inputFunctionLabel).width+14*ratio
    );
    ctx.fillStyle = "rgba(255,255,255,0.86)";
    ctx.fillRect(labelX-5*ratio,labelY-3*ratio,labelWidth,20*ratio);
    ctx.fillStyle = "#0f172a";
    ctx.fillText(options.inputFunctionLabel,labelX,labelY);
  }
  if (canvasId === "position") {
    const marker = estimateResponseMarker();
    if (marker !== null) {
      const inputTimeRelative = marker.inputTime-timeOrigin;
      const outputTimeRelative = marker.outputTime-timeOrigin;
      const xInput = left+(inputTimeRelative-x0)/spanX*(width-left-right);
      const xOutput = left+(outputTimeRelative-x0)/spanX*(width-left-right);
      const yInput = top+(maxY-marker.inputPosition)/(maxY-minY)*(height-top-bottom);
      const yOutput = top+(maxY-marker.outputPosition)/(maxY-minY)*(height-top-bottom);
      const axisY = height-bottom;
      ctx.setLineDash([5*ratio,4*ratio]);
      ctx.lineWidth = 1.2*ratio;
      ctx.strokeStyle = "#2563eb";
      ctx.beginPath(); ctx.moveTo(xInput,yInput); ctx.lineTo(xInput,axisY); ctx.stroke();
      ctx.strokeStyle = "#dc2626";
      ctx.beginPath(); ctx.moveTo(xOutput,yOutput); ctx.lineTo(xOutput,axisY); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#2563eb";
      ctx.beginPath(); ctx.arc(xInput,yInput,4*ratio,0,Math.PI*2); ctx.fill();
      ctx.fillStyle = "#dc2626";
      ctx.beginPath(); ctx.arc(xOutput,yOutput,4*ratio,0,Math.PI*2); ctx.fill();
      const arrowY = 30*ratio;
      ctx.strokeStyle = "#0f172a"; ctx.fillStyle = "#0f172a"; ctx.lineWidth = 1.5*ratio;
      ctx.beginPath(); ctx.moveTo(xInput,arrowY); ctx.lineTo(xOutput,arrowY); ctx.stroke();
      drawArrowHead(ctx,xInput,arrowY,-1,ratio);
      drawArrowHead(ctx,xOutput,arrowY,1,ratio);
      ctx.font = `${12*ratio}px sans-serif`;
      ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(`响应延时 Δt=${marker.delayMs.toFixed(1)} ms`, (xInput+xOutput)/2, arrowY-5*ratio);
      ctx.textBaseline = "top";
      ctx.fillStyle = "#2563eb";
      ctx.fillText(
        `↑ 指令点：t输入=${inputTimeRelative.toFixed(2)} s，θin=${marker.inputPosition.toFixed(2)}°`,
        xInput, axisY+6*ratio
      );
      ctx.fillStyle = "#dc2626";
      ctx.fillText(
        `↑ 响应点：t输出=${outputTimeRelative.toFixed(2)} s，θout=${marker.outputPosition.toFixed(2)}°`,
        xOutput, axisY+25*ratio
      );
    }
  }
  if (options.alignResponse) {
    ctx.fillStyle = "#0f172a";
    ctx.font = `${12*ratio}px sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const delayText = responseMarker === null
      ? "正在估计响应延时"
      : `实际输出已向左平移 Δt=${responseMarker.delayMs.toFixed(1)} ms`;
    ctx.fillText(delayText, left+(width-left-right)/2, 6*ratio);
  }
  ctx.fillStyle = "#344054";
  ctx.font = `${12 * ratio}px sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  ctx.fillText("实际测试时间（s）", left+(width-left-right)/2, height-3*ratio);
  ctx.save();
  ctx.translate(15*ratio, top+(height-top-bottom)/2);
  ctx.rotate(-Math.PI/2);
  ctx.fillText(yAxisLabel, 0, 0);
  ctx.restore();
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(`当前显示时长：${spanX.toFixed(1)} s`, width-right, 3*ratio);
}

function currentInputFunctionLabel() {
  if (currentStage.includes("步态") || currentStage.includes("ONNX")) {
    return "输入函数：θin(t)=ONNX策略以50Hz输出的关节目标位置";
  }
  if (currentStage.includes("单舵机")) {
    return "输入函数：θin(t)=s·90°·f(t)，f(u)=[1−cos(πu)]/2（分段半余弦）";
  }
  return "输入函数：θrel=15°·g；g1=(1−cosπu)/2，g2=(1+cosπu)/2，g3=−g1，g4=−g2（分段半余弦，u∈[0,1]）";
}

function redraw() {
  const inputFunctionLabel = currentInputFunctionLabel();
  const positionAxisLabel =
    (currentStage.includes("步态") || currentStage.includes("ONNX"))
      ? "位置角度（°）"
      : "相对测试原点角度（°）";
  drawChart("position", [
    {field:"target_deg", color:"#2563eb"},
    {field:"actual_deg", color:"#dc2626"}
  ], positionAxisLabel, {inputFunctionLabel});
  drawChart("positionAligned", [
    {field:"target_deg", color:"#2563eb"},
    {field:"actual_deg", color:"#dc2626"}
  ], positionAxisLabel, {alignResponse:true,inputFunctionLabel});
  drawChart("error", [{field:"error_deg", color:"#7c3aed"}], "跟踪误差（°）");
  drawChart("current", [{field:"current_a", color:"#ea580c"}], "电流（A）");
}

function updateJointOptions(joints) {
  const previous = selectedJoint;
  const existing = Array.from(jointSelect.options).map(option => option.value);
  if (JSON.stringify(existing) === JSON.stringify(joints)) return;
  jointSelect.innerHTML = "";
  for (const joint of joints) {
    const option = document.createElement("option");
    option.value = joint; option.textContent = joint; jointSelect.appendChild(option);
  }
  selectedJoint = joints.includes(previous) ? previous : (joints[0] || "");
  jointSelect.value = selectedJoint;
}

jointSelect.addEventListener("change", () => {
  selectedJoint = jointSelect.value;
  lastSeq = 0;
  samples = [];
});

async function poll() {
  try {
    const query = new URLSearchParams({joint:selectedJoint, after:String(lastSeq)});
    const response = await fetch(`/api/data?${query.toString()}`, {cache:"no-store"});
    const data = await response.json();
    document.getElementById("status").textContent = data.status || "运行中";
    currentStage = data.stage || "-";
    document.getElementById("stage").textContent = currentStage;
    const downloads = document.getElementById("downloads");
    const downloadItems = data.downloads || [];
    downloads.hidden = downloadItems.length === 0;
    downloads.innerHTML = "";
    for (const item of downloadItems) {
      const link = document.createElement("a");
      link.className = "download-button";
      link.href = item.url;
      link.textContent = item.label;
      if (item.open_new) link.target = "_blank";
      downloads.appendChild(link);
    }
    updateJointOptions(data.joints || []);
    if (!selectedJoint && data.joints && data.joints.length) selectedJoint = data.joints[0];
    if (data.joint === selectedJoint && data.samples) {
      samples.push(...data.samples);
      if (samples.length > maxPoints) samples = samples.slice(-maxPoints);
      if (data.samples.length) lastSeq = data.samples[data.samples.length-1].seq;
      const latestTime = samples.length
        ? Number(samples[samples.length-1].time_s)-Number(samples[0].time_s)
        : 0;
      document.getElementById("elapsed").textContent =
        `${Number.isFinite(latestTime) ? latestTime.toFixed(1) : "0.0"} s`;
      redraw();
    }
  } catch (error) {
    document.getElementById("status").textContent = "实时服务暂不可用";
  }
  window.setTimeout(poll, 150);
}
window.addEventListener("resize", redraw);
poll();
</script>
</body>
</html>
"""


class LiveWaveformDashboard:
    """在线程中运行实时 HTTP 页面，并保存各关节最近的采样点。"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, max_points: int = 1600):
        self.host = host
        self.requested_port = port
        self.max_points = max_points
        self._lock = threading.Lock()
        self._samples: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.max_points)
        )
        self._sequence = 0
        self._status = "准备中"
        self._stage = "-"
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._report_path: Path | None = None
        self._download_paths: dict[str, tuple[str, str, Path]] = {}
        self.url: str | None = None

    def start(self) -> str:
        """绑定第一个可用端口并启动后台服务。"""

        dashboard = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path in ("/", "/index.html"):
                    body = DASHBOARD_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/api/data":
                    query = parse_qs(parsed.query)
                    joint = query.get("joint", [""])[0]
                    try:
                        after = int(query.get("after", ["0"])[0])
                    except ValueError:
                        after = 0
                    payload = dashboard.snapshot(joint, after)
                    body = json.dumps(
                        payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-store")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if parsed.path == "/report.html":
                    report_path = dashboard.report_path()
                    if report_path is not None and report_path.is_file():
                        body = report_path.read_bytes()
                        self.send_response(HTTPStatus.OK)
                        self.send_header(
                            "Content-Type",
                            "text/html; charset=utf-8",
                        )
                        self.send_header("Content-Length", str(len(body)))
                        self.send_header("Cache-Control", "no-store")
                        self.end_headers()
                        self.wfile.write(body)
                        return
                if parsed.path.startswith("/download/"):
                    token = parsed.path.removeprefix("/download/")
                    download = dashboard.download_path(token)
                    if download is not None:
                        _, filename, file_path = download
                        if file_path.is_file():
                            body = file_path.read_bytes()
                            content_type = (
                                "text/csv; charset=utf-8"
                                if file_path.suffix.lower() == ".csv"
                                else "application/json; charset=utf-8"
                            )
                            self.send_response(HTTPStatus.OK)
                            self.send_header("Content-Type", content_type)
                            self.send_header(
                                "Content-Disposition",
                                f'attachment; filename="{filename}"',
                            )
                            self.send_header("Content-Length", str(len(body)))
                            self.send_header("Cache-Control", "no-store")
                            self.end_headers()
                            self.wfile.write(body)
                            return
                self.send_error(HTTPStatus.NOT_FOUND)

            def log_message(self, format, *args):
                return

        last_error = None
        for port in range(self.requested_port, self.requested_port + 20):
            try:
                self._server = ThreadingHTTPServer((self.host, port), Handler)
                break
            except OSError as exc:
                last_error = exc
        if self._server is None:
            raise RuntimeError(f"无法启动实时曲线服务：{last_error}")

        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="openduck-live-waveform",
            daemon=True,
        )
        self._thread.start()
        address = self._lan_address()
        self.url = f"http://{address}:{self._server.server_port}/"
        print(f"实时曲线：{self.url}")
        print("请在与树莓派同一局域网的电脑浏览器中打开该地址")
        return self.url

    def publish(
        self,
        joint: str,
        *,
        time_s: float,
        target_deg: float,
        actual_deg: float,
        error_deg: float,
        current_a: float | None,
        stage: str,
    ) -> None:
        """发布一个关节采样点。"""

        with self._lock:
            self._sequence += 1
            self._stage = stage
            self._status = "测试进行中"
            self._samples[joint].append(
                {
                    "seq": self._sequence,
                    "time_s": round(float(time_s), 6),
                    "target_deg": round(float(target_deg), 6),
                    "actual_deg": round(float(actual_deg), 6),
                    "error_deg": round(float(error_deg), 6),
                    "current_a": (
                        None if current_a is None else round(float(current_a), 6)
                    ),
                }
            )

    def publish_motor_row(self, row: dict[str, Any]) -> None:
        """接收单舵机或逐关节检测的长表数据。"""

        self.publish(
            JOINT_LABEL_ZH.get(
                str(row.get("joint")),
                str(row.get("joint", f"ID {row.get('servo_id', '-')}")),
            ),
            time_s=float(row["time_s"]),
            target_deg=float(row["target_deg"]),
            actual_deg=float(row["actual_deg"]),
            error_deg=float(row["error_deg"]),
            current_a=(
                None if row.get("current_a") is None else float(row["current_a"])
            ),
            stage=str(row.get("stage", "逐关节检测")),
        )

    def publish_gait_row(self, row: dict[str, Any], joints) -> None:
        """从整机步态宽表中发布14个关节采样点。"""

        for joint_name, _ in joints:
            self.publish(
                JOINT_LABEL_ZH.get(joint_name, joint_name),
                time_s=float(row["time_s"]),
                target_deg=float(row[f"{joint_name}_target_deg"]),
                actual_deg=float(row[f"{joint_name}_actual_deg"]),
                error_deg=float(row[f"{joint_name}_error_deg"]),
                current_a=(
                    None
                    if row.get(f"{joint_name}_current_a") is None
                    else float(row[f"{joint_name}_current_a"])
                ),
                stage=str(row.get("stage", "步态验证")),
            )

    def set_status(self, status: str, stage: str | None = None) -> None:
        with self._lock:
            self._status = status
            if stage is not None:
                self._stage = stage

    def set_report(self, report_path: Path | str) -> None:
        """让实时页面在测试结束后提供中文HTML报告入口。"""

        with self._lock:
            self._report_path = Path(report_path).resolve()

    def set_downloads(
        self,
        downloads: list[tuple[str, Path | str]],
    ) -> None:
        """注册测试结束后可由电脑浏览器直接下载的结果文件。"""

        with self._lock:
            self._download_paths = {}
            for index, (label, file_path) in enumerate(downloads, start=1):
                path = Path(file_path).resolve()
                token = f"file-{index}"
                self._download_paths[token] = (label, path.name, path)

    def download_path(self, token: str) -> tuple[str, str, Path] | None:
        """根据页面令牌返回白名单中的固定下载文件。"""

        with self._lock:
            return self._download_paths.get(token)

    def report_path(self) -> Path | None:
        """返回当前允许实时页面读取的固定报告文件。"""

        with self._lock:
            return self._report_path

    def snapshot(self, joint: str = "", after: int = 0) -> dict[str, Any]:
        """返回指定关节在序号 after 之后的增量数据。"""

        with self._lock:
            joints = list(self._samples)
            selected = joint if joint in self._samples else (joints[0] if joints else "")
            samples = [
                dict(item)
                for item in self._samples.get(selected, ())
                if int(item["seq"]) > after
            ]
            downloads = [
                {
                    "label": label,
                    "url": f"/download/{token}",
                    "open_new": False,
                }
                for token, (label, filename, path) in self._download_paths.items()
                if path.is_file()
            ]
            if self._report_path is not None and self._report_path.is_file():
                downloads.append(
                    {
                        "label": "打开中文HTML报告",
                        "url": "/report.html",
                        "open_new": True,
                    }
                )
            return {
                "status": self._status,
                "stage": self._stage,
                "joints": joints,
                "joint": selected,
                "samples": samples[-250:],
                "downloads": downloads,
            }

    def close(self) -> None:
        """停止实时服务。"""

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    @staticmethod
    def _lan_address() -> str:
        """取得局域网地址；无默认路由时回退到主机名解析。"""

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return str(sock.getsockname()[0])
        except OSError:
            try:
                return socket.gethostbyname(socket.gethostname())
            except OSError:
                return "127.0.0.1"
        finally:
            sock.close()


def wait_for_operator(message: str) -> None:
    """交互式终端中等待操作者打开实时页面，非交互环境直接继续。"""

    import sys

    if sys.stdin.isatty():
        input(message)
