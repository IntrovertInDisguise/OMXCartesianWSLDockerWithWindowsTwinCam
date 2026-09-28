<#
.SYNOPSIS
  One-shot host-side orchestrator for an OMX paper-grade capture session.

.DESCRIPTION
  ============================================================
  EXPLAIN-LIKE-I-AM-FIVE: how a full experiment day works
  ============================================================

  You have ONE camera (the Intel RealSense). It can do one of two jobs:
    A) Live INSIDE the container -- the robot's brain publishes ArUco
       alignment topics so you can check the spring caps are straight.
    B) Live on WINDOWS -- records crisp 30-FPS video for the paper.

  It cannot do both at the same time. This script switches it between
  those two modes. The Dynamixel motor cables ALWAYS stay inside the
  container; only the camera moves.

  --- THE FIVE THINGS YOU DO IN ORDER EACH DAY ---

  STEP 1. (once at the START of the day, elevated Windows PowerShell)

      .\Attach-All-USB.ps1

      In kid words:
        * Wakes up both motor cables and gives them to the container.
        * Wakes up the camera and gives it to the container.
        * Starts the camera node inside the container so ArUco topics
          are live.
      After this the camera is INSIDE the container. ArUco alignment
      topics are published and you can check spring alignment.

  STEP 2. (once per spring specimen, inside the container)

      python3 tools/calib_spring.py \
        --spring-specimen <name> [... measurement flags ...]

      Measures the spring's free length, width, and stiffness and
      writes a calib_result_*.json file. The harness stage 0 reads this
      to derive the K_lat ablation grid and load targets.
      Skip if you already have a fresh calibration JSON for this specimen.

  STEP 3. (before EACH capture batch, inside the container, camera STILL
           inside from Step 1 / previous Step 4 cleanup)

      python3 tools/hardware_harness_contact_gated_with_aruco_v2.py \
        --alignment-policy auto_then_manual \
        --spring-specimen <name> \
        --calibration-json-path logs/<calib_dir>/calib_result_*.json \
        [... ablation flags, but NO --external-camera-trigger-file ...]

      The camera is still inside the container here, so
      /spring_monitor/aruco_alignment/* topics are live. The v2 harness
      auto-applies small z trims before each attempt, or prompts you to
      nudge the spring manually. Run this until the alignment gate passes
      consistently.

      You can also use the standalone monitor to just watch alignment:
          python3 tools/aruco_alignment_monitor.py

      When alignment is satisfactory, stop the harness and move to
      Step 4.

  STEP 4. (for each paper-grade recording batch, elevated Windows PS)

      .\scripts\Run-CaptureSession.ps1

      In kid words:
        * Stops the in-container camera node.
        * Takes the camera AWAY from the container and gives it to
          Windows. (Motor cables stay where they are.)
        * Opens a recorder that sleeps until the harness writes "START".
        * Prints the trigger-file path to paste into the container.
        * Waits. You run the harness (see below). Every time the robots
          make contact, the harness writes "START" and the recorder
          wakes up. When contact ends it writes "STOP" and the clip is
          saved.
        * The spring was already aligned in Step 3, so --alignment-policy
          is off here. ArUco topics stop publishing when the camera
          leaves the container.
        * Press Ctrl+C when done. This script writes QUIT to the
          recorder, gives the camera back to the container, and restarts
          the camera node. You are back to the Step 1/Step 3 state.

      INSIDE THE CONTAINER, while Step 4 is running:

          python3 tools/hardware_harness_contact_gated_with_aruco_v2.py \
            --alignment-policy off \
            --external-camera-trigger-file /mnt/c/tmp/omx_camera_trigger.txt \
            --spring-specimen <name> \
            --calibration-json-path logs/<calib_dir>/calib_result_*.json \
            [... ablation flags ...]

      Use --alignment-policy off (or warn) because the camera is on the
      host and ArUco topics are not being published. The path after
      --external-camera-trigger-file is the Linux form of -TriggerFile
      (C:\tmp\... becomes /mnt/c/tmp/...). This script prints the exact
      string to paste.

  STEP 5. (when done for the day -- nothing extra needed)

      Let the Ctrl+C cleanup from Step 4 finish ("Capture session
      complete." is printed). The camera is back in the container and
      you can run Step 3 + Step 4 again for the next batch without
      rerunning Step 1, AS LONG AS no USB cable was unplugged and WSL
      was not restarted. If anything got unplugged, go back to Step 1.

  --- QUICK REFERENCE: WHAT IS RUNNING WHERE ---

    Windows PowerShell (elevated)
      .\Attach-All-USB.ps1                  <- Step 1, once per day
      .\scripts\Run-CaptureSession.ps1      <- Step 4, once per batch
        |-- realsense_text_trigger_capture.py  (30-FPS recorder)
        `-- toggle_realsense_usb.ps1           (moves camera in/out)

    Dev container (bash / ROS 2 Humble)
      tools/calib_spring.py                            <- Step 2
      tools/aruco_alignment_monitor.py                 <- Step 3 (optional)
      tools/hardware_harness_contact_gated_with_aruco_v2.py
        with --alignment-policy auto_then_manual       <- Step 3 (camera IN)
        with --alignment-policy off                    <- Step 4 (camera OUT)
          + --external-camera-trigger-file <path>

  --- WHEN DO I RE-RUN Attach-All-USB.ps1 BY ITSELF? ---

  Only when:
    * It is the first time today or after a reboot / WSL shutdown.
    * A USB cable was unplugged.
    * The container was rebuilt.
  Otherwise Run-CaptureSession.ps1 calls Attach-All-USB.ps1 for you at
  the end of every batch. You do not need to call it twice.

  ============================================================
  WHAT THIS SCRIPT ACTUALLY DOES (the technical version)
  ============================================================

    1. Run toggle_realsense_usb.ps1 -Action detach
       Stops the in-container realsense2_camera node, detaches the
       RealSense from WSL via usbipd, leaves FTDI/Dynamixel attached.
       After this step ArUco topics are no longer published.
    2. Truncate the text-trigger file so the recorder starts fresh.
    3. Launch tools\realsense_text_trigger_capture.py in a new window
       (or background job with -NoNewWindow). The recorder polls the
       trigger file and starts/stops AVI clips on START/STOP lines.
    4. Wait. The container-side harness writes START/STOP markers as
       mutual-contact windows open and close. Use --alignment-policy off
       because the camera is on the host; spring alignment was verified
       in Step 3 before this script was launched.
    5. On Ctrl+C, recorder exit, or -DurationSeconds elapse, write
       QUIT into the trigger file so the recorder exits cleanly.
    6. Run toggle_realsense_usb.ps1 -Action attach to restore the
       in-container ROS camera workflow (delegates to Attach-All-USB.ps1).

  Run from an elevated PowerShell (usbipd bind/detach needs admin).

.PARAMETER TriggerFile
  Path to the trigger text file the harness writes START/STOP/QUIT into.
  Default: C:\tmp\omx_camera_trigger.txt. Make sure the same file is
  visible inside the container (e.g. /mnt/c/tmp/omx_camera_trigger.txt)
  and pass that container path to the harness via
  --external-camera-trigger-file. The recommended container-side entry
  point is tools/hardware_harness_contact_gated_with_aruco_v2.py.

.PARAMETER OutputDir
  Where the recorder writes AVI / metadata. Default: .\captures.

.PARAMETER Fps
  Recorder frame rate. Default: 30.

.PARAMETER CapturePy
  Path to realsense_text_trigger_capture.py on the host. Default looks
  next to this script, then under .\tools\.

.PARAMETER Python
  Python interpreter on the host (must have pyrealsense2 installed).
  Default: 'python'.

.PARAMETER VidPid
  RealSense USB VID:PID forwarded to toggle_realsense_usb.ps1.

.PARAMETER WorkspacePath
  Container workspace path forwarded to toggle_realsense_usb.ps1.

.PARAMETER ToggleScript
  Path to toggle_realsense_usb.ps1. Default: same dir as this script.

.PARAMETER AttachArgs
  Extra args forwarded to Attach-All-USB.ps1 during the final attach step.

.PARAMETER DurationSeconds
  Optional auto-stop after N seconds. 0 = wait for Ctrl+C (default).

.PARAMETER NoNewWindow
  Run the recorder as a background Job in this PowerShell session instead
  of spawning a new console window. Useful for unattended runs.

.EXAMPLE
  # Interactive session: Ctrl+C to end.
  # Run spring alignment (Step 3) BEFORE this, while camera is still in container.
  .\Run-CaptureSession.ps1

  # Then inside the container, while this script is running (camera is on host):
  # python3 tools/hardware_harness_contact_gated_with_aruco_v2.py \
  #   --alignment-policy off \
  #   --external-camera-trigger-file /mnt/c/tmp/omx_camera_trigger.txt \
  #   --spring-specimen <name> \
  #   --calibration-json-path logs/<calib_dir>/calib_result_*.json \
  #   ...

.EXAMPLE
  .\Run-CaptureSession.ps1 -TriggerFile C:\tmp\omx_camera_trigger.txt `
                            -OutputDir D:\captures\v27 -Fps 30 `
                            -AttachArgs @('-RosDomainId','0')

.EXAMPLE
  .\Run-CaptureSession.ps1 -DurationSeconds 1800 -NoNewWindow
#>

[CmdletBinding()]
param(
    [string]$TriggerFile = 'C:\tmp\omx_camera_trigger.txt',
    [string]$OutputDir = (Join-Path (Get-Location) 'captures'),
    [int]$Fps = 30,
  [string]$ReadyFile = 'C:\tmp\realsense_ready.txt',
    [string]$CapturePy,
    [string]$Python = 'python',
    [string]$VidPid = '8086:0b3a',
    [string]$WorkspacePath = '/workspaces/omx_ros2',
    [string]$ToggleScript,
    [string[]]$AttachArgs = @(),
    [int]$DurationSeconds = 0,
    [switch]$NoNewWindow
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-ToggleScriptPath {
    if ($ToggleScript) {
        if (-not (Test-Path $ToggleScript)) { throw "ToggleScript path not found: $ToggleScript" }
        return (Resolve-Path $ToggleScript).Path
    }
    $candidate = Join-Path $PSScriptRoot 'toggle_realsense_usb.ps1'
    if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    throw "Could not locate toggle_realsense_usb.ps1. Pass -ToggleScript <path>."
}

function Resolve-CapturePyPath {
    if ($CapturePy) {
        if (-not (Test-Path $CapturePy)) { throw "CapturePy path not found: $CapturePy" }
        return (Resolve-Path $CapturePy).Path
    }
    $candidates = @(
        (Join-Path $PSScriptRoot 'realsense_text_trigger_capture.py'),
        (Join-Path $PSScriptRoot '..\tools\realsense_text_trigger_capture.py'),
        (Join-Path (Get-Location) 'tools\realsense_text_trigger_capture.py')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
    throw "Could not locate realsense_text_trigger_capture.py. Pass -CapturePy <path>."
}

function Initialize-TriggerFile {
    param([string]$Path)
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    Set-Content -Path $Path -Value '' -Encoding utf8 -NoNewline
}

function Send-Quit {
    param([string]$Path)
    try {
        Set-Content -Path $Path -Value "QUIT`n" -Encoding utf8 -NoNewline
        Write-Host "Sent QUIT to $Path." -ForegroundColor Yellow
    } catch {
        Write-Warning "Could not write QUIT to $Path : $($_.Exception.Message)"
    }
}

function Start-RecorderWindow {
    param(
        [string]$PythonExe,
        [string]$ScriptPath,
        [string]$Trigger,
        [string]$OutDir,
    [int]$Rate,
    [string]$ReadyFile
    )
    if (-not (Test-Path $OutDir)) {
        New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    }
  $args = @($ScriptPath, $Trigger, '--output-dir', $OutDir, '--fps', $Rate, '--ready-file', $ReadyFile)
    Write-Host "Launching recorder window: $PythonExe $($args -join ' ')" -ForegroundColor Cyan
    return Start-Process -FilePath $PythonExe -ArgumentList $args -PassThru
}

function Start-RecorderJob {
    param(
        [string]$PythonExe,
        [string]$ScriptPath,
        [string]$Trigger,
        [string]$OutDir,
    [int]$Rate,
    [string]$ReadyFile
    )
    if (-not (Test-Path $OutDir)) {
        New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
    }
  Write-Host "Launching recorder as background job: $PythonExe $ScriptPath $Trigger --output-dir $OutDir --fps $Rate --ready-file $ReadyFile" -ForegroundColor Cyan
  return Start-Job -ScriptBlock {
    param($py, $script, $trig, $out, $rate, $ready)
    & $py $script $trig '--output-dir' $out '--fps' $rate '--ready-file' $ready
  } -ArgumentList $PythonExe, $ScriptPath, $Trigger, $OutDir, $Rate, $ReadyFile
}

# --- main -----------------------------------------------------------------

$togglePath = Resolve-ToggleScriptPath
$capturePath = Resolve-CapturePyPath

Write-Host '=== OMX capture session orchestrator ===' -ForegroundColor Green
Write-Host "Toggle script : $togglePath"
Write-Host "Capture script: $capturePath"
Write-Host "Trigger file  : $TriggerFile"
Write-Host "Output dir    : $OutputDir"
Write-Host "FPS           : $Fps"
Write-Host ''

# 1. Detach: stop in-container realsense + hand camera to host.
& $togglePath -Action detach -VidPid $VidPid -WorkspacePath $WorkspacePath
if ($LASTEXITCODE -ne 0) { throw "toggle_realsense_usb.ps1 -Action detach failed (exit $LASTEXITCODE)." }

# 2. Reset trigger file.
Initialize-TriggerFile -Path $TriggerFile

# 3. Launch recorder.
$recorderProc = $null
$recorderJob = $null
if ($NoNewWindow) {
    $recorderJob = Start-RecorderJob -PythonExe $Python -ScriptPath $capturePath -Trigger $TriggerFile -OutDir $OutputDir -Rate $Fps -ReadyFile $ReadyFile
} else {
    $recorderProc = Start-RecorderWindow -PythonExe $Python -ScriptPath $capturePath -Trigger $TriggerFile -OutDir $OutputDir -Rate $Fps -ReadyFile $ReadyFile
}

# 3b. Wait for recorder to signal READY before telling the user to start the harness.
# The recorder writes "READY" to $ReadyFile after the camera pipeline is open.
# Without this wait there is a race: the harness could write START before the recorder
# has opened the RealSense pipeline, causing the first capture window to be missed.
Write-Host 'Waiting for recorder to become ready (camera pipeline open)...' -ForegroundColor Yellow
$readyDeadlineS = 60
$readyDeadline = (Get-Date).AddSeconds($readyDeadlineS)
$readyDetected = $false
while ((Get-Date) -lt $readyDeadline) {
    # Abort early if the recorder already exited (startup failure).
    if ($recorderProc -and $recorderProc.HasExited) {
        Write-Warning "Recorder process exited during startup (code $($recorderProc.ExitCode))."
        break
    }
    if ($recorderJob -and $recorderJob.State -in @('Completed', 'Failed', 'Stopped')) {
        Write-Warning "Recorder job ended during startup (state $($recorderJob.State))."
        break
    }
    if (Test-Path $ReadyFile) {
        $readyContent = Get-Content $ReadyFile -Raw -ErrorAction SilentlyContinue
        if ($readyContent -and $readyContent.Trim() -eq 'READY') {
            $readyDetected = $true
            break
        }
    }
    Start-Sleep -Milliseconds 500
}
if ($readyDetected) {
    Write-Host "Recorder is READY (camera pipeline open)." -ForegroundColor Green
} else {
    Write-Warning "Recorder did not signal READY within ${readyDeadlineS} s. The camera may not be open yet -- proceed with caution."
}

Write-Host ''
Write-Host 'Inside the container, start the harness with:' -ForegroundColor Green
# crude C:\foo -> /mnt/c/foo rewrite for the user.
$mntForm = $TriggerFile
if ($mntForm -match '^([A-Za-z]):(.*)$') {
    $mntForm = '/mnt/' + $matches[1].ToLower() + ($matches[2] -replace '\\', '/')
}
Write-Host "    --external-camera-trigger-file $mntForm" -ForegroundColor Gray
Write-Host ''
if ($DurationSeconds -gt 0) {
    Write-Host "Auto-stopping after $DurationSeconds s. Press Ctrl+C to stop sooner." -ForegroundColor Green
} else {
    Write-Host 'Press Ctrl+C to end the session.' -ForegroundColor Green
}

$stopRequested = $false
$startTime = Get-Date
try {
    while (-not $stopRequested) {
        Start-Sleep -Seconds 1
        if ($DurationSeconds -gt 0 -and ((Get-Date) - $startTime).TotalSeconds -ge $DurationSeconds) {
            Write-Host "Duration elapsed; stopping." -ForegroundColor Yellow
            break
        }
        if ($recorderProc -and $recorderProc.HasExited) {
            Write-Warning "Recorder process exited early (code $($recorderProc.ExitCode)). Stopping session."
            break
        }
        if ($recorderJob -and $recorderJob.State -in @('Completed', 'Failed', 'Stopped')) {
            Write-Warning "Recorder job ended early (state $($recorderJob.State)). Stopping session."
            break
        }
    }
} finally {
    Send-Quit -Path $TriggerFile

    if ($recorderProc) {
        if (-not $recorderProc.WaitForExit(10000)) {
            Write-Warning "Recorder did not exit within 10s; killing PID $($recorderProc.Id)."
            try { $recorderProc.Kill() } catch { }
        }
    }
    if ($recorderJob) {
        Wait-Job -Job $recorderJob -Timeout 10 | Out-Null
        if ($recorderJob.State -eq 'Running') {
            Write-Warning 'Recorder job still running after QUIT; stopping it forcibly.'
            Stop-Job -Job $recorderJob -ErrorAction SilentlyContinue
        }
        Receive-Job -Job $recorderJob -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "[recorder] $_" }
        Remove-Job -Job $recorderJob -Force -ErrorAction SilentlyContinue
    }

    Write-Host ''
    Write-Host 'Reattaching USB devices and restarting in-container realsense2_camera...' -ForegroundColor Cyan
    try {
        & $togglePath -Action attach -AttachArgs $AttachArgs
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "toggle_realsense_usb.ps1 -Action attach exited with $LASTEXITCODE."
        }
    } catch {
        Write-Warning "Reattach failed: $($_.Exception.Message). Run scripts\toggle_realsense_usb.ps1 -Action attach manually."
    }

    Write-Host 'Capture session complete.' -ForegroundColor Green
}
