export type Lane = {
  id: string
  label: string
  src: string
}

export type Work = {
  id: string
  title: string
  detail: string
  description: string
  takes: Lane[]
  original?: Lane
}

export type Track = {
  key: string
  work: Work
  take: Lane
}

/**
 * Each work is a folder under web/pieces:
 *
 *   web/pieces/01-borodin-serenade/about.txt
 *   web/pieces/01-borodin-serenade/seed-0.wav
 *   web/pieces/01-borodin-serenade/seed-1.wav
 *   web/pieces/01-borodin-serenade/original.wav
 *
 * about.txt: title, credit line, then the description.
 * seed-N.wav is generation seed N from the full run.
 * original.wav is the source MIDI, rendered so the page can play it.
 */
export function loadWorks(): Work[] {
  const wavs = import.meta.glob("../pieces/*/*.wav", {
    eager: true,
    query: "?url",
    import: "default",
  }) as Record<string, string>
  const texts = import.meta.glob("../pieces/*/*.txt", {
    eager: true,
    query: "?raw",
    import: "default",
  }) as Record<string, string>

  const dirs = new Map<string, { wavs: { name: string; src: string }[]; text?: string }>()
  const folderOf = (path: string) => path.split("/").slice(-2, -1)[0] ?? path
  const fileOf = (path: string) => path.split("/").pop() ?? path

  for (const [path, url] of Object.entries(wavs)) {
    const id = folderOf(path)
    const row = dirs.get(id) ?? { wavs: [] }
    row.wavs.push({ name: fileOf(path), src: url })
    dirs.set(id, row)
  }
  for (const [path, raw] of Object.entries(texts)) {
    const id = folderOf(path)
    const row = dirs.get(id) ?? { wavs: [] }
    row.text = raw
    dirs.set(id, row)
  }

  return [...dirs.entries()]
    .filter(([, row]) => row.wavs.some((wav) => /^seed-\d+\.wav$/.test(wav.name)))
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([id, row]) => {
      const notes = parseNotes(row.text, id)
      const takes = row.wavs
        .filter((wav) => /^seed-\d+\.wav$/.test(wav.name))
        .sort((a, b) => a.name.localeCompare(b.name))
        .map((wav) => {
          const seed = wav.name.match(/^seed-(\d+)\.wav$/)?.[1] ?? "?"
          return { id: `${id}-seed-${seed}`, label: `Seed ${seed}`, src: wav.src }
        })
      const originalWav = row.wavs.find((wav) => wav.name === "original.wav")
      return {
        id,
        ...notes,
        takes,
        original: originalWav
          ? { id: `${id}-original`, label: "Original", src: originalWav.src }
          : undefined,
      }
    })
}

export function trackList(works: Work[]): Track[] {
  return works.flatMap((work) => work.takes.map((take) => ({ key: take.id, work, take })))
}

function parseNotes(raw: string | undefined, id: string): Omit<Work, "id" | "takes" | "original"> {
  const fallback = id.replace(/^\d+-/, "").replace(/-/g, " ")
  if (!raw?.trim()) {
    return { title: fallback, detail: "", description: "Add an about.txt next to this recording." }
  }
  const lines = raw.replace(/\r\n/g, "\n").trim().split("\n")
  const title = lines[0]?.trim() || fallback
  const detail = lines[1]?.trim() || ""
  const description = lines.slice(2).join("\n").trim() || "No description yet."
  return { title, detail, description }
}
