# RePlay

Record-once, replay-many computer-use automation for legacy back-office applications.

An LLM discovers how to complete a goal against a live application surface. The successful
run is distilled into a typed, versioned **capability artifact**. That artifact then replays
deterministically, with no model in the decision loop.

> **Status: in progress.** This README is a placeholder. The full setup guide and demo path
> land in M12 ([#13](https://github.com/KaranKendre11/RePlay/issues/13)).
>
> Plan of record: [`PRD.md`](PRD.md) and [`project-arch.md`](project-arch.md).

## Quick start

```bash
uv sync
uv run playwright install chromium
cp .env.example .env      # add your OPENAI_API_KEY
uv run replay version
```

## Development

```bash
uv run pytest
uv run ruff check .
```
