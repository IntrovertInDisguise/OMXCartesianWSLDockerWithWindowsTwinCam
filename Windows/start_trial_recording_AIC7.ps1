# AIC7-specific replacement launcher. The original C++ harness launcher is separate.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidatePattern('^\d+(\.\d+)?$')]
    [string]$Klat,
    [Parameter(Mandatory=$true)]
    [ValidateRange(1, 1000000)]
    [int]$Trial,
    [string]$ToolsDir = 'C:\RLVICWork\OMX\camera_tools',
    [string]$RootDir = 'D:\Paper1PushExpt',
    [string]$RobotLogRoot = '\\wsl.localhost\Ubuntu-22.04\home\vrcontrollers\omx_ros2_varstiff\omx-ros2-gravity-comp\.aic7_runtime',
    # [string]$RobotLogRoot = 'D:\OMX_Experiment_Data',
    [string]$CondaEnv = 'mujoco_rl',
    [string]$TopSerial = '348522071053',
    [string]$SideSerial = '347622073030',
    [ValidateRange(1, 65535)][int]$UdpPort = 5006,
    [ValidateRange(1, 256)][int]$QueueSize = 16,
    [ValidateRange(1, 3600)][int]$RobotCopyWaitSeconds = 180
)
$ErrorActionPreference = 'Stop'
$Recorder = Join-Path $ToolsDir 'realsense_dual_rgb_recorder.py'
$Aligner = Join-Path $ToolsDir 'align_aic7_camera.py'
foreach ($File in @($Recorder, $Aligner)) {
    if (-not (Test-Path -LiteralPath $File -PathType Leaf)) { throw "Missing file: $File" }
}
$CondaCommand = $null
$CondaApp = Get-Command conda.exe -ErrorAction SilentlyContinue
if ($CondaApp) { $CondaCommand = $CondaApp.Source }
if (-not $CondaCommand -and $env:CONDA_EXE -and (Test-Path -LiteralPath $env:CONDA_EXE)) {
    $CondaCommand = $env:CONDA_EXE
}
if (-not $CondaCommand) {
    foreach ($Candidate in @(
        "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
        "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
        'C:\ProgramData\miniconda3\Scripts\conda.exe',
        'C:\ProgramData\anaconda3\Scripts\conda.exe'
    )) {
        if (Test-Path -LiteralPath $Candidate) { $CondaCommand = $Candidate; break }
    }
}
if (-not $CondaCommand) { throw 'Could not locate conda.exe. Run from an initialized Conda PowerShell.' }
& $CondaCommand run --no-capture-output -n $CondaEnv python -c "import cv2, pyrealsense2; print('Camera Python imports OK')"
if ($LASTEXITCODE -ne 0) { throw "Camera packages are unavailable in $CondaEnv" }

function Write-AtomicJson {
    param([string]$Path, $Value)
    $TempPath = "$Path.tmp"
    $Json = $Value | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText($TempPath, $Json, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item -LiteralPath $TempPath -Destination $Path -Force
}

function Complete-Aic7Collection {
    param([string]$RunDir, [string]$BridgeDir, [string]$SessionId)
    $Claim = Join-Path $BridgeDir "aic7_$SessionId.claimed.json"
    $CompletePath = Join-Path $BridgeDir "aic7_$SessionId.complete.json"
    if (-not (Test-Path -LiteralPath $Claim)) {
        Write-Warning 'No AIC7 run claimed this camera session. Camera files are retained.'
        return 2
    }
    $Deadline = (Get-Date).AddSeconds($RobotCopyWaitSeconds)
    while (-not (Test-Path -LiteralPath $CompletePath) -and (Get-Date) -lt $Deadline) {
        Start-Sleep -Milliseconds 500
    }
    if (-not (Test-Path -LiteralPath $CompletePath)) {
        Write-Warning "Robot completion not received. Source is retained at $RobotLogRoot\aic7_runs\$SessionId. Wait for AIC7 shutdown before manually copying it."
        return 3
    }
    $Status = Get-Content -LiteralPath $CompletePath -Raw | ConvertFrom-Json
    if ($Status.session_id -ne $SessionId -or $Status.protocol -ne 'aic7-v1') {
        throw 'Robot completion session mismatch'
    }
    # Derive from our validated session, not a timestamp guessed from camera events.
    $Source = Join-Path $RobotLogRoot "aic7_runs\$SessionId"
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) { throw "Missing shared source: $Source" }
    foreach ($Item in (Get-ChildItem -LiteralPath $Source -Force)) {
        Copy-Item -LiteralPath $Item.FullName -Destination $RunDir -Recurse -Force
    }
    $Mismatch = @()
    foreach ($File in (Get-ChildItem -LiteralPath $Source -File -Recurse)) {
        $Relative = $File.FullName.Substring($Source.Length).TrimStart('\')
        $Destination = Join-Path $RunDir $Relative
        if (-not (Test-Path -LiteralPath $Destination -PathType Leaf)) {
            $Mismatch += $Relative
        } elseif ((Get-FileHash -LiteralPath $File.FullName -Algorithm SHA256).Hash -ne
                  (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash) {
            $Mismatch += $Relative
        }
    }
    if ($Mismatch.Count -gt 0) { throw "Robot copy hash mismatch: $($Mismatch -join ', ')" }
    Write-Host '[OK] Complete robot tree copied and SHA256 checked.'
    $SyncFiles = @(Get-ChildItem -LiteralPath $RunDir -Filter 'realsense_dual_sync_*.csv' -File)
    if ($SyncFiles.Count -ne 1) {
        Write-Warning 'Camera sync CSV missing or ambiguous. Robot capture is retained.'
        return 4
    }
    if (-not $Status.capture_relative) {
        Write-Warning "Robot exited before creating a Capture folder (exit $($Status.exit_code))."
        return 5
    }
    & $CondaCommand run --no-capture-output -n $CondaEnv python $Aligner $RunDir
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Automatic alignment failed. Both raw datasets are retained; rerun align_aic7_camera.py after inspecting its error.'
        return 6
    }
    if ($Status.exit_code -ne 0) {
        Write-Warning "Collected an aborted robot run (exit $($Status.exit_code)); do not classify it as a completed trial."
        return 7
    }
    return 0
}

$BridgeDir = Join-Path $RobotLogRoot 'camera_sync_bridge'
New-Item -ItemType Directory -Path $BridgeDir -Force | Out-Null
# Exclusive handle rejects a second launcher using this same bridge. A crashed
# process releases its handle; a leftover lock filename alone does not block.
$LockPath = Join-Path $BridgeDir 'aic7_launcher.lock'
$Lock = [System.IO.File]::Open($LockPath, [System.IO.FileMode]::OpenOrCreate,
    [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
$SessionId = [guid]::NewGuid().ToString('N')
$Timestamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$TrialDir = Join-Path $RootDir "Klat$Klat\Trial$Trial"
$RunDir = Join-Path $TrialDir $Timestamp
$ActivePath = Join-Path $BridgeDir 'aic7_active_session.json'
$RecorderExitCode = 1
$CollectionCode = 1
try {
    # Refuse to replace the handshake of a robot that has not finalized.
    if (Test-Path -LiteralPath $ActivePath) {
        $Old = Get-Content -LiteralPath $ActivePath -Raw | ConvertFrom-Json
        if ($Old.session_id -match '^[a-f0-9]{32}$') {
            $OldClaim = Join-Path $BridgeDir "aic7_$($Old.session_id).claimed.json"
            $OldDone = Join-Path $BridgeDir "aic7_$($Old.session_id).complete.json"
            if ((Test-Path -LiteralPath $OldClaim) -and -not (Test-Path -LiteralPath $OldDone)) {
                throw 'The prior AIC7 robot session has not finalized. Check that robot process and recover its capture before restarting.'
            }
        }
    }
    New-Item -ItemType Directory -Path $RunDir | Out-Null
    $Epoch = [datetime]::SpecifyKind([datetime]'1970-01-01', [DateTimeKind]::Utc)
    $Session = [ordered]@{
        protocol = 'aic7-v1'; session_id = $SessionId
        created_unix_s = ((Get-Date).ToUniversalTime() - $Epoch).TotalSeconds
        klat = $Klat; trial = $Trial; windows_run_dir = $RunDir
        robot_log_root_windows = $RobotLogRoot; udp_port = $UdpPort
    }
    Write-AtomicJson -Path (Join-Path $RunDir 'camera_session.json') -Value $Session
    Write-AtomicJson -Path $ActivePath -Value $Session
    Write-Host "Trial folder: $RunDir"
    Write-Host "Shared bridge: $BridgeDir"
    Write-Host 'Preview ENABLED. Inspect both views, then start AIC7 --run --camera-sync in the devcontainer.'
    Write-Host 'AIC7 closes the cameras after parking and plotting. q/Ctrl+C stops VIDEO ONLY; it does not stop the robot.'
    try {
        & $CondaCommand run --no-capture-output -n $CondaEnv python $Recorder `
            --top-serial $TopSerial --side-serial $SideSerial --output-dir $RunDir `
            --udp-port $UdpPort --queue-size $QueueSize
        $RecorderExitCode = $LASTEXITCODE
    } finally {
        # Preserve camera artifacts even if they are incomplete. Never select a
        # different/newest robot run as a substitute for this exact session.
        $CollectionResult = @(Complete-Aic7Collection -RunDir $RunDir -BridgeDir $BridgeDir -SessionId $SessionId)
        if ($CollectionResult.Count -gt 0) {
            $CollectionCode = [int]$CollectionResult[-1]
            if ($CollectionResult.Count -gt 1) { $CollectionResult[0..($CollectionResult.Count - 2)] | Out-Host }
        }
        Write-AtomicJson -Path (Join-Path $RunDir 'collection_status.json') -Value @{
            session_id = $SessionId; recorder_exit_code = $RecorderExitCode
            collection_exit_code = $CollectionCode
        }
        Write-Host "Saved trial folder: $RunDir"
    }
} finally {
    # Keep an unfinished claimed session discoverable for recovery.
    if (Test-Path -LiteralPath $ActivePath) {
        $Active = Get-Content -LiteralPath $ActivePath -Raw | ConvertFrom-Json
        $Done = Join-Path $BridgeDir "aic7_$SessionId.complete.json"
        $Claim = Join-Path $BridgeDir "aic7_$SessionId.claimed.json"
        if ($Active.session_id -eq $SessionId -and
            ((Test-Path -LiteralPath $Done) -or -not (Test-Path -LiteralPath $Claim))) {
            Remove-Item -LiteralPath $ActivePath
        }
    }
    $Lock.Dispose()
}
if ($RecorderExitCode -ne 0) { exit $RecorderExitCode }
exit $CollectionCode
