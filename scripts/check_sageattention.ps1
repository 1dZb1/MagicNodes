<#
  SageAttention Environment Checker/Installer (Windows, PowerShell)

  - Detects Python, Torch, CUDA, GPU, and SageAttention version
  - Warns about local shadow files (sageattention.py in Comfy root)
  - Offers to install/upgrade SageAttention to 2.2.0
  - If no wheel is available, can build from source (needs MSVC + CUDA Toolkit)

  Usage:
    powershell -ExecutionPolicy Bypass -File scripts\check_sageattention.ps1
  or run scripts\check_sageattention.bat
#>

param(
  [switch]$AutoYes,
  [switch]$ForceSa2Source
)

function Write-Section($t){ Write-Host "`n=== $t ===" -ForegroundColor Cyan }
function Ask-YesNo($q){
  if($AutoYes){ return $true }
  $r = Read-Host "$q [y/N]"; return $r -match '^(?i:y|yes)$'
}

function Get-Python(){
  $cands = @('python','py -3','py')
  foreach($p in $cands){ try{ $v = & $p -c "import sys;print(sys.executable)" 2>$null; if($LASTEXITCODE -eq 0 -and $v){ return @{ exe=$p; path=$v.Trim() } } }catch{} }
  return $null
}

function Py-Exec($pyExe, $code){
  # Write code to a temporary .py to avoid complex quoting issues on Windows
  $tmp = [System.IO.Path]::GetTempFileName()
  $pyf = [System.IO.Path]::ChangeExtension($tmp, '.py')
  Set-Content -Path $pyf -Value $code -Encoding UTF8
  try { $out = & $pyExe $pyf } finally { Remove-Item -ErrorAction SilentlyContinue $tmp, $pyf }
  return $out
}

function Invoke-Quiet($file, $argList, $label){
  $logOut = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), '.out.log')
  $logErr = [System.IO.Path]::ChangeExtension([System.IO.Path]::GetTempFileName(), '.err.log')
  try {
    if([string]::IsNullOrWhiteSpace($argList)){ throw "ArgumentList is empty for $file" }
    $p = Start-Process -FilePath $file -ArgumentList $argList -NoNewWindow -PassThru `
         -RedirectStandardOutput $logOut -RedirectStandardError $logErr
    $spinner = @('|','/','-','\')
    $i = 0
    while(-not $p.HasExited){
      Write-Host -NoNewline ("`r{0} {1}" -f $spinner[$i % $spinner.Count], $label)
      Start-Sleep -Milliseconds 150
      $i++
    }
    try { $p.Refresh() } catch {}
    $exitCode = 1
    try { $exitCode = [int]$p.ExitCode } catch { $exitCode = 1 }
    if($exitCode -eq 0){
      Write-Host ("`r{0} ... done        " -f $label) -ForegroundColor Green
    } else {
      Write-Warning ("Failed: {0} (exit {1})" -f $label, $exitCode)
      Write-Host "---- build tail ----" -ForegroundColor DarkYellow
      if(Test-Path $logOut){ Get-Content $logOut -Tail 40 | ForEach-Object { Write-Host $_ -ForegroundColor DarkYellow } }
      if(Test-Path $logErr){ Get-Content $logErr -Tail 40 | ForEach-Object { Write-Host $_ -ForegroundColor DarkYellow } }
      Write-Host "--------------------" -ForegroundColor DarkYellow
    }
    return ($exitCode -eq 0)
  } finally {
    if(Test-Path $logOut){ Remove-Item -ErrorAction SilentlyContinue $logOut }
    if(Test-Path $logErr){ Remove-Item -ErrorAction SilentlyContinue $logErr }
  }
}

function Get-TorchInfo($pyExe){
  $code = @'
try:
  import torch
  cuda = getattr(torch.version, "cuda", None)
  is_cuda = torch.cuda.is_available()
  name = torch.cuda.get_device_name(0) if is_cuda else ""
  cc = ".".join(map(str, torch.cuda.get_device_capability(0))) if is_cuda else ""
  print("|".join([torch.__version__, str(cuda or ""), "1" if is_cuda else "0", name.replace("|"," "), cc]))
except Exception:
  print("")
'@
  $out = Py-Exec $pyExe $code
  if(-not $out){ return @{ has_torch=$false } }
  $p = $out -split '\|'
  if($p.Length -lt 3){ return @{ has_torch=$false } }
  return @{ has_torch=$true; torch=$p[0]; cuda=$p[1]; is_cuda=($p[2] -eq '1'); device_name=($p[3] | ForEach-Object { $_ }); cc=($p[4] | ForEach-Object { $_ }) }
}

function Get-SageVersion($pyExe){
  $code = @'
try:
  import importlib, importlib.util
  mod = None
  for name in ("SageAttention","sageattention"):
    try:
      if importlib.util.find_spec(name) is not None:
        mod = name; break
    except Exception:
      pass
  if not mod:
    print("")
  else:
    try:
      import importlib.metadata as md
      ver = md.version(mod)
    except Exception:
      ver = ""
    print(f"{mod}|{ver}")
except Exception:
  print("")
'@
  $out = Py-Exec $pyExe $code
  $p = $out -split '\|'
  return @{ module=($p[0]); version=($p[1]) }
}

function Test-ShadowFile(){
  # Look for a local sageattention.py that could shadow the package
  $roots = @((Get-Location).Path)
  # walk up to 3 parents
  $d = Get-Item .
  for($i=0;$i -lt 3;$i++){ $d = $d.PSParentPath; if(-not $d){ break }; $roots += $d }
  foreach($r in $roots){ $f = Join-Path $r 'sageattention.py'; if(Test-Path $f){ return $f } }
  return $null
}

Write-Section "Python"
$py = Get-Python
if(-not $py){ Write-Error "Python not found on PATH."; exit 1 }
Write-Host ("Using Python: {0} ({1})" -f $py.exe, $py.path)

Write-Section "Torch / CUDA / GPU"
$ti = Get-TorchInfo $py.path
if(-not $ti.has_torch){ Write-Warning "PyTorch not found: $($ti.err)" } else {
  $ccdisp = if($ti.cc){ "sm_{0}" -f ($ti.cc -replace '\.','') } else { "-" }
  Write-Host ("torch {0}, cuda {1}, cuda_available={2}, gpu='{3}', cc={4}" -f $ti.torch, $ti.cuda, $ti.is_cuda, $ti.device_name, $ccdisp)
}

Write-Section "SageAttention"
$target = 'SA2 (Attn2++)'
Write-Host ("Build target: {0}" -f $target) -ForegroundColor Green
$sv = Get-SageVersion $py.path
if($sv.module){
  $svver = if($null -ne $sv.version -and $sv.version -ne ''){ $sv.version } else { 'unknown' }
  Write-Host ("found module: {0} version: {1}" -f $sv.module, $svver)
} else {
  Write-Host "not installed"
}

$shadow = Test-ShadowFile
if($shadow){ Write-Warning "Local file shadows package: $shadow"; if(Ask-YesNo "Rename to sageattention.py.disabled now?"){
    Rename-Item -Path $shadow -NewName 'sageattention.py.disabled' -Force
    Write-Host "Renamed."
  }
}

$needInstall = $false
$wantSA3 = $false  # SA3 временно отключён
# Detect Windows and mark SA3 unsupported for now
$isWindows = ($env:OS -eq 'Windows_NT')
# Check if SA3 already present
$sa3check = @'
import importlib.util
print(importlib.util.find_spec("sageattn3") is not None)
'@
$sa3present = $false
if($wantSA3){
  try {
    $sa3present = ((Py-Exec $py.path $sa3check).Trim() -eq 'True')
  } catch { $sa3present = $false }
}

if($wantSA3){
  $needInstall = -not $sa3present
} else {
  if(-not $sv.module){ $needInstall = $true }
  else { try{ $ver=[Version]($sv.version -replace '[^0-9\.]',''); if($ver.Major -lt 2 -or ($ver.Major -eq 2 -and $ver.Minor -lt 2)){ $needInstall=$true } }catch{ $needInstall=$true } }
}

if(-not $needInstall){
  Write-Host ("SageAttention present (target {0}) — nothing to do." -f $target) -ForegroundColor Green; exit 0
}

if($wantSA3){
  if($isWindows){
    Write-Warning "SageAttention 3 (Blackwell) is not supported on Windows currently. Falling back to SageAttention 2.2.x."
    $wantSA3 = $false
  }
}

if($wantSA3){
  if(-not (Ask-YesNo "Install SageAttention SA3 (Blackwell) from source now?")){ Write-Host "Aborted by user."; exit 0 }
  Write-Section "Installing (from source)"
  $null = Invoke-Quiet $py.path "-m pip install -U pip setuptools wheel" "Installing SageAttention SA3, please wait a few minutes"
  # Ensure toolchain for source build
  Write-Section "Toolchain check"
  $hasCL = ($null -ne (Get-Command cl.exe -ErrorAction SilentlyContinue))
  $hasNVCC = ($null -ne (Get-Command nvcc.exe -ErrorAction SilentlyContinue))
  if(-not $hasCL -or -not $hasNVCC){
    Write-Warning "MSVC cl or CUDA nvcc not found. Install MSVC Build Tools 2022 and CUDA Toolkit matching your torch (CUDA $($ti.cuda))."
    exit 1
  }
  # Set CUDA arch list (e.g., 12.0 for Blackwell)
  if($ti.is_cuda -and $ti.cc){ $env:TORCH_CUDA_ARCH_LIST = $ti.cc }
  Write-Section "Building SA3 from source"
  $null = Invoke-Quiet $py.path "-m pip install -U packaging cmake ninja" "toolchain python deps"
  $sa3built = $false
  # Install SA3 specifically from subdirectory
  $env:GIT_TERMINAL_PROMPT = "0"
  if($sa3present){ $null = Invoke-Quiet $py.path "-m pip uninstall -y sageattn3" "uninstall SA3 (old)" }
  if(Invoke-Quiet $py.path "-m pip install -U --force-reinstall --no-build-isolation --no-cache-dir git+https://github.com/thu-ml/SageAttention@main#subdirectory=sageattention3_blackwell" "install SA3 from git subdir (main)"){ $sa3built = $true }
  if(-not $sa3built){
    # Try tags just in case
    $tags = @('v2.2.1','v2.2.0','v2.2')
    foreach($t in $tags){
      if(Invoke-Quiet $py.path ("-m pip install -U --force-reinstall --no-build-isolation --no-cache-dir git+https://github.com/thu-ml/SageAttention@{0}#subdirectory=sageattention3_blackwell" -f $t) ("install SA3 from git subdir: {0}" -f $t)){ $sa3built = $true; break }
    }
  }
  try { $sa3present = ((Py-Exec $py.path $sa3check).Trim() -eq 'True') } catch { $sa3present = $false }
  if(-not $sa3present){
    Write-Warning "SA3 package not importable after installation. Possible env mismatch or build skipped."
    # Minimal diagnostics to understand where pip installed to
    $null = Invoke-Quiet $py.path "-m pip show -f sageattn3" "pip show sageattn3"
    $diag = @'
import sys, site, importlib.util
print("py=", sys.executable)
paths = []
try:
  paths += site.getsitepackages()
except Exception:
  pass
try:
  paths.append(site.getusersitepackages())
except Exception:
  pass
print("site=", ";".join(paths))
spec = importlib.util.find_spec("sageattn3")
print("spec=", None if spec is None else (spec.origin or str(spec.submodule_search_locations)))
'@
    $dout = Py-Exec $py.path $diag
    if($dout){ Write-Host $dout -ForegroundColor DarkYellow }
  }
  
} else {
  if(-not (Ask-YesNo "Install/upgrade SageAttention to 2.2.x now?")){ Write-Host "Aborted by user."; exit 0 }
  Write-Section "Installing (wheel if available)"
  $null = Invoke-Quiet $py.path "-m pip install -U pip setuptools wheel" "Installing SageAttention 2.2.x, please wait a few minutes"
  # Remove older v1 if present to avoid 'already satisfied' noise
  $null = Invoke-Quiet $py.path "-m pip uninstall -y SageAttention" "uninstall legacy SageAttention (if any)"
  $null = Invoke-Quiet $py.path "-m pip uninstall -y sageattention" "uninstall legacy sageattention (if any)"
  $wheelOk = Invoke-Quiet $py.path "-m pip install -U --no-cache-dir sageattention>=2.2,<3" "pip install sageattention 2.2.x (wheel)"
  # Verify actual installed version >= 2.2; otherwise treat as failure to trigger fallback
  $sv2 = Get-SageVersion $py.path
  $wheelHas22 = $false
  if($sv2.module -and $sv2.version){ try{ $v=[Version]($sv2.version -replace '[^0-9\.]',''); if($v.Major -gt 2 -or ($v.Major -eq 2 -and $v.Minor -ge 2)){ $wheelHas22=$true } }catch{}
  }
  if($wheelOk -and -not $wheelHas22){ Write-Warning "Wheel installation did not provide SageAttention >= 2.2. Falling back to source build."; $wheelOk=$false }
}

if(-not $wantSA3 -and -not $wheelOk){
  Write-Warning "Wheel install failed - will try source build."
  # Try to infer arch list
  $arch = ''
  if($ti.is_cuda -and $ti.cc){ $arch = $ti.cc }
  if($arch){ $env:TORCH_CUDA_ARCH_LIST = $arch }
  # Ensure toolchain
  Write-Section "Toolchain check"
  $hasCL = ($null -ne (Get-Command cl.exe -ErrorAction SilentlyContinue))
  $hasNVCC = ($null -ne (Get-Command nvcc.exe -ErrorAction SilentlyContinue))
  if(-not $hasCL -or -not $hasNVCC){
    Write-Warning "MSVC cl or CUDA nvcc not found. Install MSVC Build Tools 2022 and CUDA Toolkit matching your torch (CUDA $($ti.cuda))."
    exit 1
  }
  if($isWindows -and -not $ForceSa2Source){
    Write-Warning "Attempting SageAttention 2.x source build on Windows (experimental upstream support)."
  }
  Write-Section "Building from source"
   $null = Invoke-Quiet $py.path "-m pip install -U packaging cmake ninja" "toolchain python deps"
   # Prefer upstream repo (thu-ml). First try main.zip to avoid git prompts
   $built = $false
   $urls = @(
     'https://github.com/thu-ml/SageAttention/archive/refs/heads/main.zip',
     'https://github.com/thu-ml/SageAttention/archive/refs/tags/v2.2.1.zip',
     'https://github.com/thu-ml/SageAttention/archive/refs/tags/v2.2.0.zip',
     'https://github.com/thu-ml/SageAttention/archive/refs/tags/v2.2.zip'
   )
   foreach($u in $urls){
     if(Invoke-Quiet $py.path "-m pip install --no-build-isolation --no-cache-dir `"$u`"" ("build from archive: {0}" -f $u)) { $built = $true; break }
   }
   if(-not $built){
     Write-Warning "Tag archive not available; trying git main (noninteractive)."
     $env:GIT_TERMINAL_PROMPT = "0"
     $null = Invoke-Quiet $py.path "-m pip install --no-build-isolation --no-cache-dir git+https://github.com/thu-ml/SageAttention@main" "build from git main"
    }
  
}

# If SA3 was requested but still not importable, try SA2 as a fallback
if($wantSA3 -and -not $sa3present){
  Write-Warning "Falling back to SageAttention 2.2.x (wheel/source)."
  Write-Section "Installing SA2 (fallback)"
  $null = Invoke-Quiet $py.path "-m pip install -U pip setuptools wheel" "Installing SageAttention 2.2.x, please wait a few minutes"
  $wheelOk = Invoke-Quiet $py.path "-m pip install -U --no-cache-dir sageattention>=2.2,<3" "pip install sageattention 2.2.x (wheel)"
  if(-not $wheelOk){
    Write-Section "Building SA2 from source"
    $null = Invoke-Quiet $py.path "-m pip install -U packaging cmake ninja" "toolchain python deps"
    $built = $false
    $urls = @(
      'https://github.com/thu-ml/SageAttention/archive/refs/heads/main.zip',
      'https://github.com/thu-ml/SageAttention/archive/refs/tags/v2.2.1.zip',
      'https://github.com/thu-ml/SageAttention/archive/refs/tags/v2.2.0.zip',
      'https://github.com/thu-ml/SageAttention/archive/refs/tags/v2.2.zip'
    )
    foreach($u in $urls){ if(Invoke-Quiet $py.path "-m pip install --no-build-isolation --no-cache-dir `"$u`"" ("build from archive: {0}" -f $u)) { $built = $true; break } }
    if(-not $built){
      $env:GIT_TERMINAL_PROMPT = "0"
      $null = Invoke-Quiet $py.path "-m pip install --no-build-isolation --no-cache-dir git+https://github.com/thu-ml/SageAttention@main" "build from git main"
    }
  }
}

Write-Section "Validation"
$val = @'
try:
  import importlib.util, torch
  S = None
  try:
    import SageAttention as S
  except Exception:
    try:
      import sageattention as S
    except Exception:
      S = None
  sa_ok = False
  if S is not None:
    sa_ok = (
      hasattr(S, 'sageattn_qk_int8_pv_fp16_cuda') or
      hasattr(S, 'sageattn_qk_int8_pv_fp16_cuda_fp16')
    )
  if sa_ok:
    print(f"OK: True torch {torch.__version__}")
  else:
    print(f"FAIL: torch {torch.__version__}")
except Exception as e:
  print("ERR:", str(e))
'@
$out = Py-Exec $py.path $val
if($out -match '^OK:'){ Write-Host $out -ForegroundColor Green; Write-Host "Done." -ForegroundColor Green }
else { Write-Warning $out; Write-Warning "SageAttention not available. ComfyUI will fall back to stock attention (slower)." }
