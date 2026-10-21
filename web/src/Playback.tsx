import { createContext, useContext, useEffect, useRef, type ReactNode } from "react"

type Listener = () => void

type PlaybackApi = {
  toggle: (src: string) => void
  seek: (src: string, time: number) => void
  subscribe: (listener: Listener) => () => void
  audio: () => HTMLAudioElement | null
  activeSrc: () => string
}

const PlaybackContext = createContext<PlaybackApi | null>(null)

export function PlaybackProvider({ children }: { children: ReactNode }) {
  const audioRef = useRef<HTMLAudioElement>(null)
  const srcRef = useRef("")
  const listeners = useRef(new Set<Listener>())

  const apiRef = useRef<PlaybackApi | null>(null)
  if (!apiRef.current) {
    const notify = () => listeners.current.forEach((listener) => listener())
    apiRef.current = {
      audio: () => audioRef.current,
      activeSrc: () => srcRef.current,
      subscribe: (listener) => {
        listeners.current.add(listener)
        return () => listeners.current.delete(listener)
      },
      toggle: (src) => {
        const audio = audioRef.current
        if (!audio) return
        if (srcRef.current === src && !audio.paused) {
          audio.pause()
          return
        }
        if (srcRef.current !== src) {
          srcRef.current = src
          audio.src = src
        }
        if (audio.ended) audio.currentTime = 0
        void audio.play().catch(() => notify())
      },
      seek: (src, time) => {
        const audio = audioRef.current
        if (!audio) return
        const apply = () => {
          const max = Number.isFinite(audio.duration) ? audio.duration : time
          audio.currentTime = Math.min(Math.max(0, time), max)
          notify()
        }
        if (srcRef.current !== src) {
          srcRef.current = src
          audio.src = src
          audio.addEventListener("loadedmetadata", apply, { once: true })
          return
        }
        apply()
      },
    }
  }

  useEffect(() => {
    const audio = audioRef.current
    if (!audio) return
    let frame = 0
    const notify = () => listeners.current.forEach((listener) => listener())
    const tick = () => {
      notify()
      if (!audio.paused) frame = requestAnimationFrame(tick)
    }
    const onPlay = () => {
      cancelAnimationFrame(frame)
      frame = requestAnimationFrame(tick)
      notify()
    }
    const onPause = () => {
      cancelAnimationFrame(frame)
      notify()
    }
    audio.addEventListener("play", onPlay)
    audio.addEventListener("pause", onPause)
    audio.addEventListener("ended", onPause)
    audio.addEventListener("loadedmetadata", notify)
    return () => {
      cancelAnimationFrame(frame)
      audio.removeEventListener("play", onPlay)
      audio.removeEventListener("pause", onPause)
      audio.removeEventListener("ended", onPause)
      audio.removeEventListener("loadedmetadata", notify)
    }
  }, [])

  return (
    <PlaybackContext.Provider value={apiRef.current}>
      <audio ref={audioRef} preload="auto" />
      {children}
    </PlaybackContext.Provider>
  )
}

export function usePlayback() {
  const api = useContext(PlaybackContext)
  if (!api) throw new Error("PlaybackProvider missing")
  return api
}
