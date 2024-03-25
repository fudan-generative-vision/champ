Set-Location $PSScriptRoot

$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
#$Env:PIP_INDEX_URL = "https://mirror.baidu.com/pypi/simple"
#$Env:HF_ENDPOINT = "https://hf-mirror.com"

if (!(Test-Path -Path "venv")) {
    Write-Output  "create python venv..."
    python -m venv venv
}
.\venv\Scripts\activate

Write-Output "Installing dependencies..."
pip install -U -r requirements-windows.txt

Write-Output "Check Models..."
if (!(Test-Path -Path "pretrained_models")) {
    mkdir "pretrained_models"
}

Set-Location .\pretrained_models

if (!(Test-Path -Path "champ")) {
    Write-Output  "Downloading champ..."
    huggingface-cli download --resume-download bdsqlsz/Champ --local-dir champ
}

if (!(Test-Path -Path "image_encoder")) {
    Write-Output  "Downloading image_encoder..."
    huggingface-cli download --resume-download bdsqlsz/image_encoder --local-dir image_encoder
}

$install_SD15 = Read-Host "Do you need to download SD15? If you don't have any SD15 model locally select y, if you want to change to another SD1.5 model select n. [y/n] (Default is y)"
if ($install_SD15 -eq "y" -or $install_SD15 -eq "Y" -or $install_SD15 -eq "") {
    if (!(Test-Path -Path "stable-diffusion-v1-5")) {
        Write-Output  "Downloading stable-diffusion-v1-5 ..."
        huggingface-cli download --resume-download bdsqlsz/stable-diffusion-v1-5 --local-dir stable-diffusion-v1-5   
    }
}

Write-Output "Installed finish"
Read-Host | Out-Null ;
