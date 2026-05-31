#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SERVICE=comfybot
INSTALL_DIR="${SCRIPT_DIR}"
SERVICE_USER=comfybot
UNIT_FILE=/etc/systemd/system/${SERVICE}.service

# ── non-interactive CLI mode ─────────────────────────────────────────────────
# Usage: manager.sh <command>
# Commands: start | stop | restart | status | logs | install | remove
if [[ "${1:-}" != "" ]]; then
    case "$1" in
        start)   sudo systemctl start   "${SERVICE}.service" && echo "[OK] Started."   ;;
        stop)    sudo systemctl stop    "${SERVICE}.service" && echo "[OK] Stopped."   ;;
        restart) sudo systemctl restart "${SERVICE}.service" && echo "[OK] Restarted." ;;
        status)  systemctl status "${SERVICE}.service" --no-pager ;;
        logs)    journalctl -u "${SERVICE}.service" -n "${2:-40}" --no-pager ;;
        logsf)   journalctl -u "${SERVICE}.service" -f ;;
        *)       echo "Commands: start | stop | restart | status | logs [N] | logsf | install | remove" ;;
    esac
    exit 0
fi

# ─── helpers ────────────────────────────────────────────────────────────────

red()    { echo -e "\e[31m$*\e[0m"; }
green()  { echo -e "\e[32m$*\e[0m"; }
yellow() { echo -e "\e[33m$*\e[0m"; }
bold()   { echo -e "\e[1m$*\e[0m"; }

need_root() {
    if [[ $EUID -ne 0 ]]; then
        red "[ERROR] Run this script with sudo."
        exit 1
    fi
}

get_status() {
    if ! systemctl list-unit-files "${SERVICE}.service" &>/dev/null 2>&1 || \
       ! systemctl cat "${SERVICE}.service" &>/dev/null 2>&1; then
        echo "[not installed]"
    else
        local st
        st=$(systemctl is-active "${SERVICE}.service" 2>/dev/null || true)
        echo "[${st}]"
    fi
}

read_env() {
    local key="$1" file="${SCRIPT_DIR}/.env"
    [[ -f "$file" ]] || { echo ""; return; }
    grep -m1 "^${key}=" "$file" 2>/dev/null | cut -d= -f2- || echo ""
}

write_env() {
    local key="$1" value="$2" file="${SCRIPT_DIR}/.env"
    if [[ -f "$file" ]] && grep -q "^${key}=" "$file"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$file"
    else
        echo "${key}=${value}" >> "$file"
    fi
}

env_summary() {
    local token checkpoint comfy_url
    token=$(read_env BOT_TOKEN)
    checkpoint=$(read_env CHECKPOINT)
    comfy_url=$(read_env COMFY_URL)
    if [[ -z "$token" ]]; then
        echo "  Config : BOT_TOKEN not set | Checkpoint: ${checkpoint:-not set}"
    else
        echo "  Config : Token: ${token:0:10}... | Checkpoint: ${checkpoint:-not set}"
    fi
    echo "  ComfyUI: ${comfy_url:-http://192.168.39.39:8188}"
}

# ─── menu ───────────────────────────────────────────────────────────────────

main_menu() {
    while true; do
        clear
        bold "============================================"
        bold "  ComfyUI Telegram Bot — Manager (Linux)"
        bold "============================================"
        echo ""
        echo "  Status : $(get_status)"
        env_summary
        echo ""
        bold "  [ Service ]"
        echo "  1. Install service"
        echo "  2. Start service"
        echo "  3. Stop service"
        echo "  4. Restart service"
        echo "  5. Remove service"
        echo ""
        bold "  [ Settings ]"
        echo "  6. View all settings"
        echo "  7. Edit settings"
        echo ""
        bold "  [ Development ]"
        echo "  8. Setup venv & install dependencies"
        echo "  9. Run bot in console (for testing)"
        echo ""
        bold "  [ Logs ]"
        echo "  a. View live log (journalctl -f)"
        echo "  b. View last 40 lines"
        echo ""
        echo "  0. Exit"
        echo ""
        read -rp "Choose: " choice

        case "$choice" in
            1) do_install ;;
            2) do_start ;;
            3) do_stop ;;
            4) do_restart ;;
            5) do_remove ;;
            6) show_settings ;;
            7) settings_menu ;;
            8) do_setup_venv ;;
            9) do_run_console ;;
            a|A) do_log_live ;;
            b|B) do_log_tail ;;
            0) exit 0 ;;
            *) yellow "Unknown option." ;;
        esac
    done
}

# ─── service actions ────────────────────────────────────────────────────────

do_install() {
    need_root
    clear
    bold "[INSTALL] Installing systemd service..."
    echo ""

    if systemctl cat "${SERVICE}.service" &>/dev/null 2>&1; then
        yellow "[WARN] Service already installed. Remove it first (option 5)."
        pause_back; return
    fi

    # Ensure .env and required keys
    local token
    token=$(read_env BOT_TOKEN)
    if [[ -z "$token" ]]; then
        yellow "BOT_TOKEN not set — let's configure it now."
        set_param "BOT_TOKEN" "Telegram bot token from @BotFather" ""
    fi

    local ckpt
    ckpt=$(read_env CHECKPOINT)
    if [[ -z "$ckpt" ]]; then
        set_param "CHECKPOINT" "Model filename (e.g. v1-5-pruned-emaonly.ckpt)" "v1-5-pruned-emaonly.ckpt"
    fi

    # Setup venv if missing
    if [[ ! -d "${INSTALL_DIR}/venv" ]]; then
        do_setup_venv_in "${INSTALL_DIR}"
    fi

    chmod 600 "${INSTALL_DIR}/.env"

    # Install systemd unit
    echo "[INFO] Installing systemd unit..."
    cp "${SCRIPT_DIR}/comfybot.service" "${UNIT_FILE}"
    systemctl daemon-reload
    systemctl enable "${SERVICE}.service"

    echo ""
    read -rp "Start the service now? (y/n): " start_now
    if [[ "$start_now" =~ ^[Yy]$ ]]; then
        systemctl start "${SERVICE}.service"
        green "[OK] Service started."
    else
        green "[OK] Service installed but not started."
    fi
    pause_back
}

do_start() {
    need_root
    systemctl cat "${SERVICE}.service" &>/dev/null || { red "[ERROR] Not installed."; pause_back; return; }
    systemctl start "${SERVICE}.service" && green "[OK] Started." || red "[ERROR] Failed."
    pause_back
}

do_stop() {
    need_root
    systemctl cat "${SERVICE}.service" &>/dev/null || { red "[ERROR] Not installed."; pause_back; return; }
    systemctl stop "${SERVICE}.service" && green "[OK] Stopped." || red "[ERROR] Failed."
    pause_back
}

do_restart() {
    need_root
    systemctl cat "${SERVICE}.service" &>/dev/null || { red "[ERROR] Not installed."; pause_back; return; }
    systemctl restart "${SERVICE}.service" && green "[OK] Restarted." || red "[ERROR] Failed."
    pause_back
}

do_remove() {
    need_root
    clear
    yellow "[WARN] This will stop and remove the service."
    read -rp "Type YES to confirm: " confirm
    [[ "$confirm" != "YES" ]] && { echo "Cancelled."; pause_back; return; }

    systemctl stop  "${SERVICE}.service" 2>/dev/null || true
    systemctl disable "${SERVICE}.service" 2>/dev/null || true
    rm -f "${UNIT_FILE}"
    systemctl daemon-reload
    green "[OK] Service removed."
    read -rp "Also delete files in ${INSTALL_DIR}? (y/n): " del_files
    if [[ "$del_files" =~ ^[Yy]$ ]]; then
        rm -rf "${INSTALL_DIR}"
        green "[OK] Files deleted."
    fi
    pause_back
}

# ─── settings ───────────────────────────────────────────────────────────────

settings_menu() {
    while true; do
        clear
        bold "============================================"
        bold "  Settings"
        bold "============================================"
        echo ""
        print_settings
        echo ""
        bold "  1. BOT_TOKEN       (required)"
        bold "  2. CHECKPOINT      (required)"
        echo "  3. COMFY_URL"
        echo "  4. IMAGE_WIDTH"
        echo "  5. IMAGE_HEIGHT"
        echo "  6. STEPS"
        echo "  7. CFG_SCALE"
        echo "  8. NEGATIVE_PROMPT"
        echo "  9. POLL_TIMEOUT"
        echo ""
        echo "  0. Back"
        echo ""
        read -rp "Choose setting to edit: " sc

        case "$sc" in
            1) set_param "BOT_TOKEN"       "Telegram bot token from @BotFather"                         "" ;;
            2) set_param "CHECKPOINT"      "Model filename from ComfyUI/models/checkpoints/"             "v1-5-pruned-emaonly.ckpt" ;;
            3) set_param "COMFY_URL"       "ComfyUI address"                                             "http://192.168.39.39:8188" ;;
            4) set_param "IMAGE_WIDTH"     "Image width (512 for SD1.5, 1024 for SDXL)"                  "512" ;;
            5) set_param "IMAGE_HEIGHT"    "Image height (512 for SD1.5, 1024 for SDXL)"                 "512" ;;
            6) set_param "STEPS"           "Sampling steps (20=fast, 30-40=better quality)"              "20" ;;
            7) set_param "CFG_SCALE"       "Prompt adherence 5-9 (7 is default)"                         "7.0" ;;
            8) set_param "NEGATIVE_PROMPT" "What to exclude from generation"                             "ugly, blurry, low quality, watermark, text, deformed" ;;
            9) set_param "POLL_TIMEOUT"    "Seconds to wait for ComfyUI response"                        "300" ;;
            0) return ;;
        esac
    done
}

set_param() {
    local key="$1" desc="$2" default="$3"
    local current
    current=$(read_env "$key")
    clear
    bold "============================================"
    bold "  Edit: $key"
    bold "============================================"
    echo ""
    echo "  Description : $desc"
    [[ -n "$default" ]]  && echo "  Default     : $default"
    if [[ "$key" == "BOT_TOKEN" && -n "$current" ]]; then
        echo "  Current     : ${current:0:10}..."
    elif [[ -n "$current" ]]; then
        echo "  Current     : $current"
    else
        echo "  Current     : (not set)"
    fi
    echo ""
    echo "  Press Enter to keep current value."
    echo ""
    read -rp "New value: " new_val
    if [[ -z "$new_val" ]]; then
        echo "No changes made."
    else
        need_root 2>/dev/null || true
        write_env "$key" "$new_val"
        green "[OK] $key saved."
    fi
    echo ""
    read -rp "Press Enter to continue..." _
}

show_settings() {
    clear
    bold "============================================"
    bold "  Current Settings"
    bold "============================================"
    echo ""
    print_settings
    pause_back
}

print_settings() {
    local keys=(BOT_TOKEN CHECKPOINT COMFY_URL IMAGE_WIDTH IMAGE_HEIGHT STEPS CFG_SCALE NEGATIVE_PROMPT POLL_TIMEOUT)
    for k in "${keys[@]}"; do
        local v
        v=$(read_env "$k")
        if [[ "$k" == "BOT_TOKEN" && -n "$v" ]]; then
            v="${v:0:10}..."
        fi
        printf "  %-20s = %s\n" "$k" "${v:-(not set)}"
    done
}

# ─── venv / run ─────────────────────────────────────────────────────────────

do_setup_venv() {
    clear
    do_setup_venv_in "${SCRIPT_DIR}"
    pause_back
}

do_setup_venv_in() {
    local dir="$1"
    echo "[INFO] Setting up virtual environment in ${dir}..."
    python3 -m venv "${dir}/venv"
    "${dir}/venv/bin/pip" install --upgrade pip -q
    "${dir}/venv/bin/pip" install -r "${dir}/requirements.txt"
    green "[OK] Dependencies installed."
}

do_run_console() {
    clear
    local token
    token=$(read_env BOT_TOKEN)
    if [[ -z "$token" ]]; then
        red "[ERROR] BOT_TOKEN not set. Use Settings (option 7)."
        pause_back; return
    fi
    if [[ ! -d "${SCRIPT_DIR}/venv" ]]; then
        echo "[INFO] venv not found — running setup first..."
        do_setup_venv_in "${SCRIPT_DIR}"
    fi
    echo "[INFO] Running bot in console. Press Ctrl+C to stop."
    echo ""
    cd "${SCRIPT_DIR}"
    "${SCRIPT_DIR}/venv/bin/python" bot.py
    pause_back
}

# ─── logs ────────────────────────────────────────────────────────────────────

do_log_live() {
    echo "[INFO] Showing live log. Press Ctrl+C to stop."
    echo ""
    journalctl -u "${SERVICE}.service" -f
    pause_back
}

do_log_tail() {
    clear
    echo "=== Last 40 lines ==="
    echo ""
    journalctl -u "${SERVICE}.service" -n 40 --no-pager
    pause_back
}

# ─── util ────────────────────────────────────────────────────────────────────

pause_back() {
    echo ""
    read -rp "Press Enter to continue..." _
}

main_menu
