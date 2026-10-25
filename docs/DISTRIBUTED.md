# Running across machines

The council distributes **tasks**, not model layers: each machine serves whole models over an
OpenAI-compatible API, and the router chooses a (model, node) pair per task. Splitting one
model across laptops over Wi-Fi is slower than running a smaller model locally.

## Current nodes

| Node | Hardware | Inference budget | Serves |
|---|---|---|---|
| `mac-m5` | MacBook Air M5, 32 GB unified, fanless | ~22 GB | gpt-oss:20b, qwen3:4b, gemma3:4b |
| `win-4060` | RTX 4060 Laptop (8 GB VRAM), 32 GB RAM | ~7.5 GB VRAM | qwen3:4b, gemma3:4b (later an 8B) |

The Mac is fanless, so long runs throttle; benchmarks must record sustained, not burst, speed.

## Adding the Windows laptop

1. Install the NVIDIA driver and [Ollama for Windows](https://ollama.com/download).
2. Make Ollama listen beyond localhost: set the user environment variable
   `OLLAMA_HOST=0.0.0.0:11434`, then restart Ollama.
3. Pull the models: `ollama pull qwen3:4b` and `ollama pull gemma3:4b`.
4. Connect the machines privately (recommended: Tailscale, below). If you skip Tailscale,
   allow inbound TCP 11434 on the **Private** network profile only, never Public.
5. On the Mac, set `base_url` for `win-4060` (Tailscale name, e.g. `http://win-4060:11434`,
   or its LAN IP), set `enabled: true`, reseed, then run `uv run council nodes`.

## Tailscale

Tailscale builds a private, encrypted network (WireGuard) between your own devices. Each
machine gets a stable name and address that works on any Wi-Fi, so the council keeps working
when the laptops move off the home network, and nothing is exposed to the internet. The free
personal plan covers this. Install it on both machines, sign in with the same account, and
use the machine names it shows as `base_url` hosts.
