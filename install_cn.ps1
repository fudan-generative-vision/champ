Set-Location $PSScriptRoot

$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_INDEX_URL = "https://mirror.baidu.com/pypi/simple"
$Env:HF_ENDPOINT = "https://hf-mirror.com"

if (!(Test-Path -Path "venv")) {
    Write-Output  "创建python虚拟环境venv..."
    python -m venv venv
}
.\venv\Scripts\activate

Write-Output "安装依赖..."
pip install -U -r requirements-windows.txt

Write-Output "检查模型..."
if (!(Test-Path -Path "pretrained_models")) {
    mkdir "pretrained_models"
}

Set-Location .\pretrained_models

if (!(Test-Path -Path "champ")) {
    Write-Output  "下载image_encoder模型..."
    huggingface-cli download --resume-download bdsqlsz/Champ --local-dir champ
}

if (!(Test-Path -Path "image_encoder")) {
    Write-Output  "下载image_encoder模型..."
    huggingface-cli download --resume-download bdsqlsz/image_encoder --local-dir image_encoder
}

$install_SD15 = Read-Host "是否需要下载huggingface的SD15模型? 若您本地没有任何SD15模型选择y，如果想要换其他SD1.5模型选择 n。[y/n] (默认为 y)"
if ($install_SD15 -eq "y" -or $install_SD15 -eq "Y" -or $install_SD15 -eq "") {
    if (!(Test-Path -Path "stable-diffusion-v1-5")) {
        Write-Output  "下载 stable-diffusion-v1-5 模型..."
        huggingface-cli download --resume-download bdsqlsz/stable-diffusion-v1-5 --local-dir stable-diffusion-v1-5   
    }
}

Write-Output "安装完毕"
Read-Host | Out-Null ;
