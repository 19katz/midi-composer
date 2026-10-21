import { useEffect, useRef, useState, type MouseEvent } from "react"
import { usePlayback } from "./Playback"

const BARS = 72

let sharedContext: AudioContext | null = null

function audioContext() {
  if (!sharedContext) sharedContext = new AudioContext()
  return sharedContext
}

function peaksFrom(buffer: AudioBuffer): number[] {
  const data = buffer.getChannelData(0)
  const size = Math.max(1, Math.floor(data.length / BARS))
  const peaks: number[] = []
  for (let i = 0; i < BARS; i++) {
    let max = 0
    const start = i * size
    for (let j = 0; j < size; j++) max = Math.max(max, Math.abs(data[start + j] ?? 0))
    peaks.push(max)
  }
  const loudest = Math.max(...peaks, 0.01)
  return peaks.map((peak) => peak / loudest)
}

function draw(canvas: HTMLCanvasElement, peaks: number[], progress: number) {
  const dpr = window.devicePixelRatio || 1
  const width = canvas.clientWidth
  const height = canvas.clientHeight
  if (width === 0 || height === 0) return
  canvas.width = Math.floor(width * dpr)
  canvas.height = Math.floor(height * dpr)
  const ctx = canvas.getContext("2d")
  if (!ctx) return
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  ctx.clearRect(0, 0, width, height)
  const mid = height / 2
  const barW = width / peaks.length
  peaks.forEach((peak, i) => {
    const played = (i + 0.5) / peaks.length <= progress
    const paint = getComputedStyle(canvas)
    ctx.fillStyle = played ? paint.getPropertyValue("--accent").trim() : paint.getPropertyValue("--ink").trim()
    const bh = Math.max(3, peak * height * 0.84)
    ctx.fillRect(i * barW + 1, mid - bh / 2, Math.max(2, barW - 2.5), bh)
  })
}

function clock(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00"
  const s = Math.max(0, Math.floor(seconds))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`
}

export function Player({ src, compact }: { src: string; compact?: boolean }) {
  const playback = usePlayback()
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const nowRef = useRef<HTMLSpanElement>(null)
  const totalRef = useRef<HTMLSpanElement>(null)
  const peaksRef = useRef<number[]>(Array.from({ length: BARS }, () => 0.15))
  const durationRef = useRef(0)
  const playingRef = useRef(false)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    durationRef.current = 0
    peaksRef.current = Array.from({ length: BARS }, () => 0.18)
    draw(canvas, peaksRef.current, 0)
    if (totalRef.current) totalRef.current.textContent = "0:00"
    if (nowRef.current) nowRef.current.textContent = "0:00"

    const ctx = audioContext()
    let cancelled = false
    fetch(src)
      .then((res) => res.arrayBuffer())
      .then((buf) => ctx.decodeAudioData(buf.slice(0)))
      .then((buffer) => {
        if (cancelled) return
        durationRef.current = buffer.duration
        peaksRef.current = peaksFrom(buffer)
        if (totalRef.current) totalRef.current.textContent = clock(buffer.duration)
        draw(canvas, peaksRef.current, 0)
      })
      .catch(() => undefined)

    const paint = () => {
      const audio = playback.audio()
      const active = Boolean(audio && playback.activeSrc() === src)
      const time = active ? audio!.currentTime : 0
      const duration = durationRef.current || (active && Number.isFinite(audio!.duration) ? audio!.duration : 0)
      if (duration && !durationRef.current) durationRef.current = duration
      const isPlaying = Boolean(active && !audio!.paused && !audio!.ended)
      if (nowRef.current) nowRef.current.textContent = clock(time)
      if (totalRef.current && duration) totalRef.current.textContent = clock(duration)
      canvas.setAttribute("aria-valuenow", String(Math.round((duration ? time / duration : 0) * 100)))
      draw(canvas, peaksRef.current, duration ? time / duration : 0)
      if (playingRef.current !== isPlaying) {
        playingRef.current = isPlaying
        setPlaying(isPlaying)
      }
    }
    const unsubscribe = playback.subscribe(paint)
    const observer = new ResizeObserver(() => paint())
    observer.observe(canvas)
    return () => {
      cancelled = true
      unsubscribe()
      observer.disconnect()
    }
  }, [playback, src])

  function toggle() {
    void audioContext().resume()
    playback.toggle(src)
  }

  function seek(event: MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current
    const duration = durationRef.current
    if (!canvas || !duration) return
    const rect = canvas.getBoundingClientRect()
    playback.seek(src, ((event.clientX - rect.left) / rect.width) * duration)
  }

  return (
    <div className={compact ? "player compact" : "player"}>
      <button className="play" type="button" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
        {playing ? "Pause" : "Play"}
      </button>
      <div className="player-body">
        <canvas
          ref={canvasRef}
          className="wave"
          onClick={seek}
          role="slider"
          aria-label="Seek"
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={0}
        />
        <div className="clock">
          <span ref={nowRef}>0:00</span>
          <span ref={totalRef}>0:00</span>
        </div>
      </div>
    </div>
  )
}
