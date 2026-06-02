# chunking-for-files

A standalone TypeScript utility (copied into NanoClaw under `tools/`) that
splits a large file into editable, heading-aligned chunks and merges them back
together — useful for editing files that exceed an LLM's context window.

## Usage

Run from anywhere with `tsx` (already a NanoClaw dev dependency):

```bash
# Split a file into chunks (writes <file>_chunks/ + a manifest.json)
npx tsx tools/chunking-for-files/chunking-for-files.ts split <file> [options]

# Merge edited chunks back into the original file
npx tsx tools/chunking-for-files/chunking-for-files.ts merge <manifest.json>

# Show the status of chunks (which were modified)
npx tsx tools/chunking-for-files/chunking-for-files.ts status <manifest.json>

# Help
npx tsx tools/chunking-for-files/chunking-for-files.ts --help
```

## Options (split)

- `-o, --output <dir>` — output directory for chunks (default: `<file>_chunks/`)
- `-l, --level <n>` — split at heading level `n` (default: 2 for markdown)
- `-m, --max-lines <n>` — max lines per chunk before warning (default: 500)
- `-t, --type <type>` — file type: `auto`, `markdown`, `json`, `typescript` (default: `auto`)
- `--dry-run` — show what would be done without writing files
- `-h, --help` — show help

## How it works

`split` parses the source into sections (markdown by heading level, JSON by
top-level keys, TypeScript by declarations) and writes each section as a
numbered chunk plus a `manifest.json` recording the source hash, line ranges,
and a per-chunk hash. `merge` reassembles the chunks in order back into the
original; `status` reports which chunks changed since the split (by hash).

File type is auto-detected from the extension (`.md`/`.markdown`, `.json`,
`.ts`/`.tsx`/`.js`/`.jsx`/`.mjs`/`.cjs`) and can be forced with `-t`.

## Setup

Self-contained — the only runtime dependencies are the Node.js `fs` and `path`
built-ins, so running via `npx tsx` needs no install. The tool ships its own
`package.json`/`tsconfig.json`; to work on it in isolation:

```bash
cd tools/chunking-for-files
npm install        # dev types only
npx tsc --noEmit   # typecheck
```
