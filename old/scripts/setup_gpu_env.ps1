param(
    [string]$EnvName = "saccard-gpu",
    [switch]$Recreate
)

$ErrorActionPreference = "Stop"

function Assert-Conda {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "Conda was not found in PATH. Open an Anaconda/Miniconda shell and retry."
    }
}

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Script
    )

    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Script
}

Assert-Conda

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if ($Recreate) {
    Invoke-Step -Name "Removing existing env '$EnvName' (if present)" -Script {
        conda env remove -n $EnvName -y 2>$null | Out-Null
    }
}

Invoke-Step -Name "Creating/updating conda env '$EnvName' with Python 3.9" -Script {
    conda create -n $EnvName python=3.9 -y
}

Invoke-Step -Name "Installing CUDA-enabled PyTorch stack" -Script {
    conda install -n $EnvName pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
}

Invoke-Step -Name "Installing project requirements" -Script {
    conda run -n $EnvName pip install -r requirements.txt
}

Invoke-Step -Name "Installing saccard in editable mode" -Script {
    conda run -n $EnvName pip install -e .
}

Invoke-Step -Name "Validating GPU visibility" -Script {
    conda run -n $EnvName python -c "import torch, cupy; print('torch', torch.__version__); print('torch_cuda', torch.cuda.is_available()); print('cuda_ver', torch.version.cuda); print('cupy_devices', cupy.cuda.runtime.getDeviceCount())"
}

Write-Host ""
Write-Host "Environment setup complete." -ForegroundColor Green
Write-Host "Activate with: conda activate $EnvName" -ForegroundColor Green
