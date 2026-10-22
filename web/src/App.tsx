import { useEffect, useRef, useState } from "react"
import { usePlayback } from "./Playback"
import { Player } from "./Player"
import { loadWorks, trackList, type Lane, type Track } from "./pieces"

const REPO = "https://github.com/19katz/midi-composer"

export function App() {
  const tracks = trackList(loadWorks())
  const [index, setIndex] = useState(0)
  const track = tracks[index]
  const playback = usePlayback()
  const openSrc = useRef("")
  openSrc.current = track?.take.src ?? ""

  useEffect(() => {
    if (!track) return
    const visible = new Set<string>(track.work.takes.map((lane) => lane.src))
    if (track.work.original) visible.add(track.work.original.src)
    const audio = playback.audio()
    const active = playback.activeSrc()
    if (audio && active && !audio.paused && !visible.has(active)) audio.pause()
  }, [playback, track])

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target instanceof Element ? event.target : null
      if (target?.closest("input, textarea")) return
      if (event.key === "ArrowRight") setIndex((n) => Math.min(tracks.length - 1, n + 1))
      if (event.key === "ArrowLeft") setIndex((n) => Math.max(0, n - 1))
      if (event.key === " ") {
        if (target?.closest(".play")) return
        event.preventDefault()
        if (openSrc.current) playback.toggle(openSrc.current)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [playback, tracks.length])

  return (
    <main className="desk">
      <section className="summary">
        <p className="kicker">Listening room</p>
        <h1>MIDI Composer</h1>
        <p>
          A small language model, fine-tuned with LoRA on 332 classical solo-piano scores, writes a
          new piece from a short brief. This room plays the .wavs from a fine-tuned model
          checkpoint trained on 90% of the data.
        </p>
        <p className="quiet">
          Originals are Bernd Krueger’s piano-midi.de scores (CC BY-SA), rendered with the
          FluidR3 piano. Generations are the wavs saved in <code>data/outputs/full_run</code>.
        </p>
        <a className="ticket" href={REPO} target="_blank" rel="noreferrer">
          <span>
            <span className="ticket-kicker">Source</span>
            19katz / midi-composer
          </span>
          <span aria-hidden="true">↗</span>
        </a>
      </section>

      <section className="stand" aria-label="Recordings">
        {track ? <NowPlaying track={track} index={index} total={tracks.length} /> : <Empty />}
        <h2 className="shelf-label">Track list</h2>
        <ol className="catalog">
          {tracks.map((item, i) => (
            <li key={item.key}>
              <button
                type="button"
                className={i === index ? "card on" : "card"}
                onClick={() => setIndex(i)}
                aria-current={i === index ? "true" : undefined}
              >
                <ClefMark />
                <span className="num">{String(i + 1).padStart(2, "0")}</span>
                <span className="card-copy">
                  <span className="card-title">{item.work.title}</span>
                  <span className="card-detail">
                    {item.take.label}
                    {item.work.detail ? ` · ${item.work.detail}` : ""}
                  </span>
                </span>
              </button>
            </li>
          ))}
        </ol>
        <p className="hint">Arrow keys move through the shelf. Space plays the open recording.</p>
      </section>
    </main>
  )
}

function ClefMark() {
  return (
    <svg className="clef-mark" viewBox="155 12 320 496" aria-hidden="true">
      {/* Staff lines sit on the clef: the second from the bottom (y=300) is the G the spiral wraps. */}
      <g className="clef-lines">
        <path d="M168 180 H460" />
        <path d="M168 220 H460" />
        <path d="M168 260 H460" />
        <path d="M168 300 H460" />
        <path d="M168 340 H460" />
      </g>
      {/* Outline from the game-icons G clef (Lorc, CC BY 3.0). */}
      <path
        className="clef-glyph"
        d="m238.75 138.492l18.147-18.829c.898-21.472 9.107-59.37 26.166-60.771c6.994.097 14.566 5.833 15.742 16.021c3.25 28.145-11.833 40.563-29.072 57.372l-31.408 32.586l-.037-.038c-28.467 31.447-65.355 70.37-69.772 106.973c-5.588 53.77 31.264 108.42 87.149 105.693l-3.178-18.066c-37.522-4.905-69.771-43.77-56.711-81.312c23.275-60.413 75.664-92.822 116.333-148.022c11.474-15.574 14.575-46.945 11.815-63.128c-2.76-16.184-14.345-43.066-32.059-45.017c-34.714-4.59-60.724 62.476-53.114 116.538zm25.667 75.896l-15.508 15.248c.542 2.775 1.166 6.185 2.088 12.14c-34.017 21.505-36.576 66.707-12.186 90.96c5.177 5.31 15.165-2.767 10.104-8.674c-10.971-19.786-12.097-44.028 5.742-57.137c8.78 50.539 24.627 117.16 30.164 170.895c.364 3.535-6.593 10.614-9.973 11.396c0-22.091-17.908-40-40-40c-22.09 0-40 17.909-40 40c.017 20.193 15.266 36.007 35.123 39.668c42.008 7.745 78.659-17.935 72.76-53.16c-8.793-59.062-21.986-121.007-30.857-175.32c12.066-1.253 28.875 5.169 36.428 12.847c8.575 9.297 12.95 22.35 11.982 32.787c-.924 9.96-7.425 20.992-17.06 30.381l3.662 20.776c17.346-12.583 35.117-30.185 36.94-49.844c1.518-16.352-5.01-40.324-18.08-52.273c-16.885-15.437-38.313-18.553-57.349-16.612c-2.004-13.065-3.018-18.039-3.98-24.078"
      />
    </svg>
  )
}

function NowPlaying({ track, index, total }: { track: Track; index: number; total: number }) {
  const { work, take } = track
  return (
    <article className="sheet" key={take.id}>
      <header className="sheet-head">
        <p className="kicker ink">
          No. {index + 1} of {total}
          <span className="dot"> · </span>
          {take.label}
        </p>
        <h2>{work.title}</h2>
        <p className="detail">{work.detail}</p>
      </header>
      <p className="description">{work.description}</p>
      {work.takes.length > 1 && (
        <Compare
          heading="Two generations"
          note="Same brief, two seeds."
          lanes={work.takes}
          openId={take.id}
        />
      )}
      {work.original && (
        <Compare
          heading="Next to the original"
          note="The source score on the same piano, beside the generation you opened."
          lanes={[work.original, take]}
          openId={take.id}
        />
      )}
      {work.takes.length === 1 && !work.original && <Player src={take.src} />}
    </article>
  )
}

function Compare({
  heading,
  note,
  lanes,
  openId,
}: {
  heading: string
  note: string
  lanes: Lane[]
  openId: string
}) {
  return (
    <section className="compare">
      <h3>{heading}</h3>
      <p>{note}</p>
      <div className="pair">
        {lanes.map((lane) => (
          <figure key={`${heading}-${lane.id}`} className={lane.id === openId ? "lane on" : "lane"}>
            <figcaption>{lane.label}</figcaption>
            <Player src={lane.src} compact />
          </figure>
        ))}
      </div>
    </section>
  )
}

function Empty() {
  return (
    <article className="sheet">
      <h2>The shelf is empty</h2>
      <p className="description">Add a folder under web/pieces with a seed wav and an about.txt.</p>
    </article>
  )
}
