# compress-for-context

A standalone TypeScript utility (copied into NanoClaw under `tools/`) that
compresses files for LLM context windows using format-specific strategies,
reporting estimated token savings. It can also decompress a previously
compressed file and process many files in batch.

## Usage

Run from anywhere with `tsx` (already a NanoClaw dev dependency):

```bash
# Compress a single file (writes <input>.compact.<ext>)
npx tsx tools/compress-for-context/compress-for-context.ts <input> [options]

# Batch mode
npx tsx tools/compress-for-context/compress-for-context.ts -b -p "*.json" [options]

# Decompress a .compact file back to the original
npx tsx tools/compress-for-context/compress-for-context.ts <file.compact.ext> -d

# Help
npx tsx tools/compress-for-context/compress-for-context.ts --help
```

## Options

- `-o, --output <file>` — output file (default: `input.compact.<ext>`)
- `-f, --format <fmt>` — force format: `json | yaml | markdown | csv | tsv | text | log | typescript | javascript | xml | html` (default: auto-detect from extension)
- `-l, --level <lvl>` — compression level: `light | medium | aggressive` (default: `medium`)
- `--no-legend` — don't include the legend in the output
- `--no-stats` — don't print compression statistics
- `--dry-run` — preview compression without writing a file
- `-h, --help` — show help

**Batch:**

- `-b, --batch` — process multiple files
- `-p, --pattern <pat>` — file pattern, e.g. `"*.json"`, `"*.md"`
- `-r, --recursive` — search directories recursively

**Decompress:**

- `-d, --decompress` — restore a `.compact` file to its original form

## Compression levels

- `light` — minimal changes, preserve readability
- `medium` — balance size and readability (default)
- `aggressive` — maximum compression, may reduce readability

## How it works

The tool picks a strategy based on the detected format (JSON/YAML/CSV/TSV,
markdown, source code, logs, plain text, XML/HTML), applies size-reducing
transforms, and optionally emits a legend so the result can be expanded back.
It estimates tokens before/after and prints the savings; `--dry-run` previews
the first 500 characters without writing.

## Setup

Self-contained — the only runtime dependencies are the Node.js `fs` and `path`
built-ins, so running via `npx tsx` needs no install. The tool ships its own
`package.json`/`tsconfig.json`; to work on it in isolation:

```bash
cd tools/compress-for-context
npm install        # dev types only
npx tsc --noEmit   # typecheck
```
