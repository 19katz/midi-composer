import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { App } from "./App"
import { PlaybackProvider } from "./Playback"
import "./styles.css"

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <PlaybackProvider>
      <App />
    </PlaybackProvider>
  </StrictMode>,
)
