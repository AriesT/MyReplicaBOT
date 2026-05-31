# 📋 TODO — MyReplicaBot

> Development roadmap, bugs, and improvement ideas.  
> Format: `[ ]` open · `[x]` done · `[~]` in progress · `[!]` blocked

---

## 🔥 High Priority

### Core Generation
- [ ] **Seed control** — let user set/lock a specific seed; show used seed in result caption so the image can be reproduced
- [ ] **SDXL Refiner** — add optional refiner pass (`SDXLRefiner` workflow) for SDXL checkpoints
- [ ] **SD3 / SD3.5 proper workflow** — dedicated `_build_workflow_sd3()` builder using the correct `SD3` nodes (currently falls back to sd15 builder)
- [ ] **CFG Rescale** — expose `cfg_rescale_multiplier` for FLUX workflow (FLUX uses guidance, not traditional CFG)
- [ ] **Inpainting mode** — img2img with mask: allow user to draw a mask on image before generation

### LoRA & Models
- [ ] **LoRA category tags** — admin can tag LoRAs (e.g. `character`, `style`, `concept`); filter in picker by tag
- [ ] **Per-LoRA CLIP strength** — currently `strength_model` == `strength_clip`; expose separate CLIP strength
- [ ] **LoRA preview images** — admin can attach a preview image to each LoRA shown in the picker

### Settings
- [ ] **Negative prompt styles** — each style's `neg_suffix` appended automatically without overwriting user's custom neg
- [ ] **Settings presets / profiles** — save current settings as named preset; switch between profiles (e.g. "Fast", "Quality", "MMORPG Assets")
- [ ] **Per-checkpoint default settings** — when switching checkpoint, offer to load recommended settings (steps, CFG, sampler)

---

## 🚀 New Features

### Generation
- [ ] **ControlNet support** — pass control image + conditioning type (canny, depth, openpose, etc.)
- [ ] **Adetailer / face fix** — post-process face regions after generation
- [ ] **Tiled VAE** — for large resolutions (1536×1536+) to avoid VRAM OOM
- [ ] **Image variation strength slider** — quick inline slider in result message (re-generate with different denoise)
- [ ] **Regional prompting** — split canvas into zones with different prompts

### UI / UX
- [ ] **Inline prompt history** — quickly re-use one of last 5 prompts from a list after typing `/gen`
- [ ] **Prompt templates** — admin-defined templates (e.g. `[scene], MMORPG game asset, isometric, ...`) that users can fill in
- [ ] **Quick-reply buttons** — after generation: 🔄 Redo · ✏️ Edit prompt · 🎲 New seed · ➕ Variations ×4
- [ ] **Progress percentage in caption** — update the generated image caption with step progress
- [ ] **Estimated time** — show ETA based on average steps/sec from recent runs

### Admin
- [ ] **Usage statistics dashboard** — top users, most-used models/LoRAs, hourly generation count, average generation time
- [ ] **Per-user generation limit** — daily/monthly quota with configurable reset; admin override
- [ ] **Broadcast message** — admin sends a message to all registered users
- [ ] **Model download helper** — provide CivitAI URL, bot downloads checkpoint/LoRA to ComfyUI folder automatically (requires ComfyUI Manager)

### Styles
- [ ] **Admin-managed styles** — store styles in DB instead of hardcoded `STYLES` dict; admin CRUD in bot
- [ ] **Style preview image** — each style has a sample image shown on hover / in description
- [ ] **Style combinations** — allow stacking multiple styles (currently user picks one per generation)

---

## 🐛 Known Issues / Fixes

- [ ] **`dpmpp_2m_karras` in saved settings** — users who had this sampler saved before the fix still have the old value; add a migration to silently remap on load (`dpmpp_2m_karras` → `dpmpp_2m`)
- [ ] **Concurrent users, same status message** — if two users generate simultaneously their progress updates can collide; ensure each job's status message is scoped to the correct message ID
- [ ] **img2img without photo** — if user is in img2img mode and types a message without attaching a photo, the bot should remind them to send a photo first (currently handled, but edge case with state loss on bot restart)
- [ ] **Long LoRA names in callbacks** — MD5 hash cache (`_lora_id_map`) is in-memory only; lost on bot restart if users have menus open → handle gracefully with re-fetch fallback
- [ ] **History pagination with deleted files** — if the image file is missing from disk, history nav crashes; add a check and show a placeholder
- [ ] **FLUX + LoRA** — FLUX workflow builder currently ignores LoRAs entirely (FLUX uses UNETLoader, not CheckpointLoader); implement proper FLUX LoRA support

---

## 🏗 Refactoring / Technical Debt

- [ ] **Split `bot.py`** — at 3300+ lines, split into modules: `handlers/`, `keyboards/`, `callbacks/`
- [ ] **Typed settings dataclass** — replace raw `dict` from `get_gen_settings()` with a `UserSettings` dataclass for type safety
- [ ] **Config validation at startup** — fail fast if `BOT_TOKEN` is missing or `COMFY_URL` is unreachable
- [ ] **Connection pool for SQLite** — current single shared connection works but could benefit from proper pooling under heavy load
- [ ] **Centralized error messages** — extract all Ukrainian UI strings to a `strings.py` constants file for easier localization
- [ ] **Retry logic in `gen_queue`** — auto-retry failed generations once before reporting error to user
- [ ] **Remove legacy single-LoRA fields** — `s["lora"]` / `s["lora_strength"]` fallback in `comfy_client.py` can be removed after all users have migrated to `loras_active`
- [ ] **Unit tests** — add tests for `_resolve_sampler()`, `_lora_nodes()`, `_build_workflow()`, user permission checks

---

## 📚 Documentation

- [ ] **Setup guide with screenshots** — step-by-step with ComfyUI model installation
- [ ] **ComfyUI custom node requirements** — list which custom nodes are needed for each workflow type
- [ ] **API reference** — document `comfy_client.generate()` signature for potential external use
- [ ] **Changelog** — maintain `CHANGELOG.md` with version history

---

## ✅ Completed

- [x] Multi-LoRA simultaneous activation with individual strengths
- [x] LoRA trigger word auto-injection after translation
- [x] Custom style suffix per user
- [x] A1111 sampler names → ComfyUI sampler + scheduler split (`dpmpp_2m_karras` fix)
- [x] Single active style auto-applied without picker
- [x] Admin LoRA management (display name + trigger word)
- [x] HiRes Fix 2× upscale pass
- [x] Upscale model selection (AI model vs bilinear)
- [x] FLUX workflow support
- [x] HiDream workflow support
- [x] SQLite migration from JSON files
- [x] Per-user generation history with navigation
- [x] Async generation queue with live progress bar
- [x] WebSocket real-time progress from ComfyUI
- [x] Auto-translation of Ukrainian/Russian prompts
- [x] Systemd service with manager script
- [x] Batch generation (1–4 images)
- [x] Regen from history with temporary config override
