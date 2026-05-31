# 🤖 MyReplicaBot

> A powerful Telegram bot for AI image generation powered by **ComfyUI**, supporting SD 1.5, SDXL, FLUX, SD 3/3.5, and HiDream workflows — with per-user settings, multi-LoRA, styles, upscaling, and a full admin panel.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/aiogram-3.x-009e60?logo=telegram&logoColor=white" />
  <img src="https://img.shields.io/badge/ComfyUI-required-orange?logo=data:image/svg+xml;base64,..." />
  <img src="https://img.shields.io/badge/SQLite-WAL-lightgrey?logo=sqlite" />
  <img src="https://img.shields.io/badge/license-MIT-green" />
</p>

---

## ✨ Features

### 🎨 Image Generation
| Feature | Details |
|---|---|
| **Text → Image** | Full text-to-image pipeline via ComfyUI |
| **Image → Image** | Variation on uploaded photo + prompt |
| **Multi-workflow** | SD 1.5 · SDXL · FLUX · SD 3/3.5 · HiDream — auto-detected per checkpoint |
| **HiRes Fix** | 2× upscale pass with configurable denoise strength |
| **Batch generation** | 1–4 images per request |
| **Upscale** | Dedicated upscale from history (×2 / ×4) with optional AI upscale model |

### 🎭 LoRA System
- **Multi-LoRA**: Activate any number of LoRAs simultaneously, chained automatically
- **Per-LoRA strength**: 0.4 – 1.5, configurable per LoRA
- **Trigger words**: Admin-set trigger words auto-injected into prompt after translation
- **Live toggle**: Enable/disable individual LoRAs without losing other settings

### 🎨 Style System
- 12 built-in styles: Photo, Anime, Digital Art, Oil Painting, Watercolor, Sketch, Cinematic, Fantasy, Pixel Art, 3D Render, Vintage, Minimalism
- **Custom suffix**: User-defined prompt suffix saved to profile
- **Smart auto-apply**: If only 1 style is active — applied silently, no extra prompt
- **Style picker**: Shown only when 2+ styles are active — choose per generation

### ⚙️ Per-User Generation Settings
Every user has a fully isolated settings profile stored in SQLite:

| Setting | Options |
|---|---|
| Mode | `text2img` / `img2img` |
| Checkpoint | Any model registered by admin |
| Resolution | 512×512 → 1024×1024 (7 presets) |
| Steps | 10 / 15 / 20 / 25 / 30 / 40 |
| CFG Scale | 4.0 – 11.0 |
| Sampler | euler · euler_ancestral · dpmpp_2m · dpmpp_2m_karras · ddim · uni_pc |
| Scheduler | Auto-resolved from sampler name (A1111 → ComfyUI mapping) |
| Denoise (i2i) | 0.3 – 1.0 |
| Batch size | 1 – 4 |
| Negative prompt | Custom per-user |
| LoRA | Multi-select with individual strengths |
| Styles | 12 built-in + custom suffix |
| HiRes Fix | On/Off + denoise level |
| Upscale model | AI model or bilinear fallback |

### 👑 Admin Panel
- **User management**: Add / remove / promote users, view history per user
- **Model management**: Register checkpoints, set display name, set workflow type
- **LoRA management**: Register LoRAs from ComfyUI, set display name + trigger word
- **Upscale model management**: Register upscale models from ComfyUI
- **Full generation history**: Browse and upscale any user's images
- **ComfyUI status**: Real-time RAM/VRAM usage, queue depth

### 🔧 Technical Highlights
- **Async queue** (`gen_queue.py`): serialized generation with live progress bar updates
- **WebSocket integration**: Real-time progress from ComfyUI via WebSocket
- **Auto-translation**: Ukrainian/Russian prompts auto-translated to English via `deep-translator`
- **SQLite WAL**: Single shared connection, WAL mode for concurrent reads
- **Migrations**: Forward-only schema migrations run at startup automatically
- **A1111 → ComfyUI sampler mapping**: `dpmpp_2m_karras` → `dpmpp_2m` + `karras` scheduler

---

## 🏗 Architecture

```
MyReplicaBot/
├── bot.py              # Telegram bot: all handlers, FSM, keyboards, callbacks
├── comfy_client.py     # ComfyUI API: workflow builders, WebSocket, upscale
├── gen_queue.py        # Async generation queue (serialized)
├── database.py         # SQLite init, shared connection, migrations
├── config.py           # Environment config (dotenv)
├── users.py            # User CRUD + gen_settings
├── models.py           # Checkpoint model registry
├── loras.py            # LoRA registry + trigger words
├── upscale_models.py   # Upscale model registry
├── history.py          # Generation history (save images + DB entries)
├── translator.py       # Prompt auto-translation (UA/RU → EN)
├── manager.sh          # Systemd service manager (install/start/stop/logs)
├── comfybot.service    # Systemd unit file template
├── requirements.txt    # Python dependencies
└── .env.example        # Environment variable template
```

### Data Flow

```
User sends prompt
       │
       ▼
[bot.py] translate (UA→EN)
       │
       ▼
inject LoRA trigger words
       │
       ▼
apply style suffix (if 1 active: auto-apply; if 2+: ask)
       │
       ▼
[gen_queue.py] enqueue job
       │
       ▼
[comfy_client.py] build workflow JSON
  ┌────────────────────────────────────────────┐
  │  CheckpointLoaderSimple                    │
  │  → LoraLoader × N (chained, nodes 50+)    │
  │  → CLIPTextEncode (pos + neg)              │
  │  → KSampler (sampler + scheduler resolved) │
  │  → VAEDecode → SaveImage                  │
  │  [optional: HiRes Fix pass]               │
  └────────────────────────────────────────────┘
       │
       ▼
WebSocket progress → Telegram progress bar updates
       │
       ▼
image bytes → save to disk → history DB → send to user
```

---

## 🚀 Installation

### Prerequisites
- Python **3.11+**
- Running [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance (local or remote)
- Telegram Bot token from [@BotFather](https://t.me/BotFather)

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/MyReplicaBOT.git
cd MyReplicaBOT

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
nano .env
```

```dotenv
# Required
BOT_TOKEN=1234567890:AAxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ComfyUI address (adjust if ComfyUI runs remotely)
COMFY_URL=http://127.0.0.1:8188

# Default checkpoint (exact filename from ComfyUI models/checkpoints/)
CHECKPOINT=v1-5-pruned-emaonly.ckpt

# Generation defaults (all overridable per-user)
IMAGE_WIDTH=512
IMAGE_HEIGHT=512
STEPS=20
CFG_SCALE=7.0
NEGATIVE_PROMPT=ugly, blurry, low quality, watermark, text, deformed, extra limbs
```

### 3. First Run

```bash
python bot.py
```

On first launch:
- `bot.db` SQLite database is created automatically
- If `users.json` / `models.json` exist, they are migrated and renamed to `.bak`
- The bot starts but has **no users yet** — add yourself via admin bootstrap below

### 4. Add First Admin

```bash
# Edit users.json before first launch, OR use SQLite directly:
sqlite3 bot.db "INSERT INTO users(telegram_id, username, role) VALUES(YOUR_TG_ID, 'your_username', 'admin');"
```

Then restart. Open the bot in Telegram → `/start`.

### 5. Register Models & LoRAs

In the bot:
1. **⚙️ Налаштування** → **🤖 Управління моделями** → **➕ Додати модель** → fetches list from ComfyUI
2. **🎭 Управління LoRA** → same flow
3. **🔍 Управління Upscale** → same flow
4. Set display names and workflow types per model

---

## 🖥 Production Deployment (Linux / systemd)

```bash
# Install as systemd service (interactive wizard)
sudo bash manager.sh

# Or use non-interactive commands:
sudo bash manager.sh install   # install + enable
bash manager.sh start          # start
bash manager.sh stop           # stop
bash manager.sh restart        # restart
bash manager.sh status         # service status
bash manager.sh logs 100       # last 100 log lines
bash manager.sh logsf          # follow live logs
```

The service runs under a dedicated `comfybot` system user and auto-restarts on crash.

**Recommended deploy path:** `/opt/MyReplicaBot/`

---

## 🔄 Supported Workflows

| Workflow | Checkpoints | Notes |
|---|---|---|
| `sd15` | SD 1.5, Realistic Vision, DreamShaper, etc. | Default |
| `sdxl` | SDXL, Juggernaut XL, Pony, etc. | Higher VRAM requirement |
| `flux` | FLUX.1-dev, FLUX.1-schnell | Requires `clip_l.safetensors` + `t5xxl` + `ae.safetensors` |
| `sd3` | SD 3, SD 3.5 | |
| `hidream` | HiDream-I1 | Requires 4 text encoders + FLUX VAE |

Workflow type is auto-detected from checkpoint filename, but can be overridden per-model in the admin panel.

---

## 🧩 LoRA Multi-Chain Example

When a user activates LoRAs `A` (strength 0.8) and `B` (strength 1.0), the generated workflow looks like:

```
CheckpointLoaderSimple (node 4)
    └─► LoraLoader A, strength=0.8  (node 50)
            └─► LoraLoader B, strength=1.0  (node 51)
                    └─► KSampler  (node 3)
```

Trigger words for each active LoRA are automatically prepended to the prompt after translation.

---

## 🌍 Translation

Prompts in Ukrainian or Russian are auto-translated to English before generation using `deep-translator` (Google Translate backend). The original prompt is shown alongside the translated one in the result caption.

---

## 📦 Dependencies

```
aiogram==3.15.0       # Telegram Bot framework (async, FSM)
python-dotenv==1.1.0  # .env file loading
pillow                # Image resize for img2img input
aiohttp               # Async HTTP + WebSocket client (bundled with aiogram)
deep-translator       # Google Translate for prompt translation
```

---

## 🔐 Security Notes

- Only users registered in the database can interact with the bot — all others get `⛔ У вас немає доступу`
- Admin-only actions are gated by role check on every callback
- `.env` file contains the bot token — **never commit it**; it is listed in `.gitignore`
- `bot.db` contains user data — also in `.gitignore`

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.

---

<p align="center">Made with ❤️ for AI image generation enthusiasts</p>
