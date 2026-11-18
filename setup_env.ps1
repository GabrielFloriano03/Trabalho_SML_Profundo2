$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
Set-Location $root

Write-Host "=== Criando venv ==="
if (-not (Test-Path "$root\.venv")) { py -3.13 -m venv "$root\.venv" }
$py = "$root\.venv\Scripts\python.exe"

Write-Host "=== Atualizando pip ==="
& $py -m pip install --upgrade pip setuptools wheel

Write-Host "=== Instalando PyTorch CUDA 12.8 ==="
& $py -m pip install torch --index-url https://download.pytorch.org/whl/cu128

Write-Host "=== Instalando dependencias ==="
& $py -m pip install "numpy<3" datasets tiktoken tokenizers transformers huggingface_hub tqdm matplotlib streamlit

Write-Host "=== Verificacao ==="
& $py -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0)); print('capability', torch.cuda.get_device_capability(0)); print('vram_GB', round(torch.cuda.get_device_properties(0).total_memory/1e9, 2))"
Write-Host "=== SETUP OK ==="
