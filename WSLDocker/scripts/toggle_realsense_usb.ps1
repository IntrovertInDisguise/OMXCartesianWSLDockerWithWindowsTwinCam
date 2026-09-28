<#
.SYNOPSIS
  Hand the Intel RealSense camera back to the Windows host (for high-FPS
  paper-grade capture via tools/realsense_text_trigger_capture.py), or
  restore the in-container ROS workflow by invoking Attach-All-USB.ps1.

.DESCRIPTION
  Companion to Attach-All-USB.ps1.

  Attach-All-USB.ps1 (the canonical activation script):
    - unbinds + re-binds + attaches BOTH the FTDI USB serial adapters
      (0403:6014, x2 for Dynamixel) AND the RealSense (8086:0b3a) to WSL,
    - then starts realsense2_camera inside the workspace container.

  This script handles the *opposite* transition needed when running the
  host-side high-FPS capture for paper figures:
    1. stop the in-container realsense2_camera launch (uses the PID file
       that Attach-All-USB.ps1 stored at /tmp/realsense2_camera.launch.pid),
    2. usbipd detach ONLY the RealSense camera (FTDI stays attached so
       the Dynamixel motors keep working during the experiment),
    3. leave the camera available to Windows for
       tools\realsense_text_trigger_capture.py.

  The 'attach' action shells back out to Attach-All-USB.ps1 so there is a
  single source of truth for the activation flow.

  Run from an elevated (Administrator) PowerShell on the Windows host.

.PARAMETER Action
  detach  - stop in-container realsense node + return camera to Windows host.
  attach  - delegate to Attach-All-USB.ps1 to restore in-container workflow.
  status  - print usbipd state and (if reachable) the in-container launch PID.

.PARAMETER VidPid
  RealSense USB VID:PID. Defaults to 8086:0b3a (D435i). Override for D415
  (8086:0ad3) or D455 (8086:0b5c).

.PARAMETER WorkspacePath
  Workspace mount path inside the container. Default /workspaces/omx_ros2.

.PARAMETER ContainerId
  Optional explicit Docker container ID. If omitted, picks the first running
  container with WorkspacePath mounted.

.PARAMETER AttachScript
  Path to Attach-All-USB.ps1. Default: same directory as this script, then
  Desktop / OneDrive\Desktop.

.PARAMETER AttachArgs
  Extra args forwarded to Attach-All-USB.ps1 when -Action attach.

.EXAMPLE
  # Pre-experiment (host-side capture):
  .\toggle_realsense_usb.ps1 -Action detach

.EXAMPLE
  # Restore in-container ArUco / ROS workflow:
  .\toggle_realsense_usb.ps1 -Action attach -AttachArgs @('-RosDomainId','0')

.EXAMPLE
  .\toggle_realsense_usb.ps1 -Action status
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('detach', 'attach', 'status')]
    [string]$Action,

    [string]$VidPid = '8086:0b3a',

    [string]$WorkspacePath = '/workspaces/omx_ros2',

    [string]$ContainerId,

    [string]$AttachScript,

    [string[]]$AttachArgs = @()
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$launchPidPath = '/tmp/realsense2_camera.launch.pid'

function Assert-RunningAsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this script from an elevated PowerShell session. usbipd bind/detach requires administrator privileges.'
    }
}

function Assert-CommandAvailable {
    param([string]$Name, [string]$InstallHint)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "$Name is not on PATH. $InstallHint"
    }
}

function Get-UsbipdRows {
    usbipd list | ForEach-Object {
        if ($_ -match '^(?<BusId>\S+)\s+(?<VidPid>[0-9A-Fa-f]{4}:[0-9A-Fa-f]{4})\s+.*?\s+(?<State>Not shared|Shared|Attached)$') {
            [pscustomobject]@{
                BusId  = $matches.BusId
                VidPid = $matches.VidPid.ToLower()
                State  = $matches.State
                Raw    = $_
            }
        }
    }
}

function Find-CameraRows {
    param([string]$Target)
    $normalized = $Target.ToLower()
    return @(Get-UsbipdRows | Where-Object { $_.VidPid -eq $normalized })
}

function Find-WorkspaceContainer {
    param([string]$PreferredId, [string]$TargetWorkspacePath)
    if ($PreferredId) { return $PreferredId }

    $lines = docker ps --format '{{.ID}}'
    foreach ($id in $lines) {
        if ([string]::IsNullOrWhiteSpace($id)) { continue }
        & docker exec $id bash -lc "test -d '$TargetWorkspacePath'" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $inspect = docker inspect $id | ConvertFrom-Json
            $hasMount = $inspect[0].Mounts | Where-Object { $_.Destination -eq $TargetWorkspacePath }
            if ($hasMount) { return $id }
        }
    }
    return $null
}

function Stop-InContainerRealsense {
    param([string]$Container)
    if (-not $Container) {
        Write-Warning "No workspace container found; skipping in-container realsense2_camera shutdown. If a node is still running it will fail when the camera detaches."
        return
    }

    $script = @'
set +e
if [ -f '{LAUNCHPID}' ]; then
    pid=$(cat '{LAUNCHPID}' 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "Stopping realsense2_camera launch (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        for _ in 1 2 3 4 5; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f '{LAUNCHPID}'
fi
pkill -f 'realsense2_camera.*rs_launch.py' >/dev/null 2>&1 || true
pkill -f 'realsense2_camera_node' >/dev/null 2>&1 || true
echo 'in-container realsense2_camera shutdown complete.'
'@
    # Substitute the actual path value without invoking regex semantics.
    $script = $script.Replace('{LAUNCHPID}', $launchPidPath)
    # Remove any leading BOM character that may have been embedded in the here-string
    $script = $script.TrimStart([char]0xFEFF)
    # Normalize line endings. We'll remove any UTF-8 BOM bytes when writing the temp file.
        # $script = $script.Replace("`r", "")

    Write-Host "Stopping realsense2_camera inside container $Container..." -ForegroundColor Yellow
    # Write script to a temporary file with UTF-8 (no BOM) and copy into container to avoid stdin encoding issues.
    $tempHost = [System.IO.Path]::GetTempFileName()
        try {
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($script)
            if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
                $bytes = $bytes[3..($bytes.Length - 1)]
            }
            # Ensure Unix line endings
            $text = [System.Text.Encoding]::UTF8.GetString($bytes) -replace "`r`n", "`n"
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
            [System.IO.File]::WriteAllBytes($tempHost, $bytes)
            $remotePath = "/tmp/toggle_script_$([System.Guid]::NewGuid().ToString('N')).sh"
            $cpOutput = & docker cp $tempHost "${Container}:$remotePath" 2>$null
            $rc = $LASTEXITCODE
            # docker cp prints "Successfully copied X to container:path" on stdout before
            # attempting to mount the serial device (which fails on this Docker/WSL setup
            # and causes exit code 1). Trust the "Successfully copied" message over the
            # exit code -- the file was transferred even when exit code is non-zero.
            $filePresent = ($rc -eq 0) -or (($cpOutput | Out-String) -match 'Successfully copied')
            if (-not $filePresent) {
                Write-Warning "docker cp failed (exit $rc) and file not present in container. Using base64-pipe fallback."
                $b64 = [Convert]::ToBase64String($bytes)
                & docker exec $Container bash -c "echo $b64 | base64 -d | bash"
                if ($LASTEXITCODE -ne 0) { Write-Warning "base64-pipe exec returned non-zero ($LASTEXITCODE) while stopping realsense2_camera. Continuing with detach anyway." }
            } else {
                if ($rc -ne 0) { Write-Host "docker cp exited $rc but file was created in container; proceeding to exec." -ForegroundColor DarkGray }
                & docker exec $Container bash -lc "chmod +x $remotePath && dos2unix $remotePath 2>/dev/null || true; bash $remotePath; rm -f $remotePath"
                $rc2 = $LASTEXITCODE
                if ($rc2 -ne 0) { Write-Warning "docker exec returned non-zero ($rc2) while stopping realsense2_camera. Continuing with detach anyway." }
            }
        } finally {
            Remove-Item $tempHost -ErrorAction SilentlyContinue
        }
}

function Show-Status {
    Write-Host '--- usbipd list ---' -ForegroundColor Cyan
    usbipd list
    $container = Find-WorkspaceContainer -PreferredId $ContainerId -TargetWorkspacePath $WorkspacePath
    if ($container) {
        Write-Host ''
        Write-Host "--- in-container realsense launch state ($container) ---" -ForegroundColor Cyan
        $statusScript = @'
if [ -f '{LAUNCHPID}' ]; then
    pid=$(cat '{LAUNCHPID}')
    if kill -0 "$pid" 2>/dev/null; then
        echo "realsense2_camera running, PID=$pid"
    else
        echo "stale PID file: $pid (process not running)"
    fi
else
    echo 'no PID file at {LAUNCHPID}'
fi
'@
        $statusScript = $statusScript.Replace('{LAUNCHPID}', $launchPidPath)
        # Remove any leading BOM character that may have been embedded in the here-string
        $statusScript = $statusScript.TrimStart([char]0xFEFF)
        # Normalize line endings. We'll remove any UTF-8 BOM bytes when writing the temp file.
            # $statusScript = $statusScript.Replace("`r", "")
        # Write status script to a temp file and copy into the container to avoid stdin encoding/terminator issues.
        $tempHost = [System.IO.Path]::GetTempFileName()
        try {
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($statusScript)
            if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
                $bytes = $bytes[3..($bytes.Length - 1)]
            }
            # Normalize to LF endings
            $text = [System.Text.Encoding]::UTF8.GetString($bytes) -replace "`r`n", "`n"
            $bytes = [System.Text.Encoding]::UTF8.GetBytes($text)
            [System.IO.File]::WriteAllBytes($tempHost, $bytes)
            $remotePath = "/tmp/toggle_status_$([System.Guid]::NewGuid().ToString('N')).sh"
            $cpOutput = & docker cp $tempHost "${container}:$remotePath" 2>$null
            $rc = $LASTEXITCODE
            # docker cp prints "Successfully copied X to container:path" on stdout before
            # attempting to mount the serial device (which fails on this Docker/WSL setup
            # and causes exit code 1). Trust the "Successfully copied" message over the
            # exit code -- the file was transferred even when exit code is non-zero.
            $filePresent = ($rc -eq 0) -or (($cpOutput | Out-String) -match 'Successfully copied')
            if (-not $filePresent) {
                Write-Warning "docker cp failed (exit $rc) and file not present in container. Using base64-pipe fallback."
                $b64 = [Convert]::ToBase64String($bytes)
                & docker exec $container bash -c "echo $b64 | base64 -d | bash"
                if ($LASTEXITCODE -ne 0) { Write-Warning "base64-pipe exec returned non-zero ($LASTEXITCODE) while checking in-container realsense state." }
            } else {
                if ($rc -ne 0) { Write-Host "docker cp exited $rc but file was created in container; proceeding to exec." -ForegroundColor DarkGray }
                & docker exec $container bash -lc "chmod +x $remotePath && dos2unix $remotePath 2>/dev/null || true; bash $remotePath; rm -f $remotePath"
                $rc2 = $LASTEXITCODE
                if ($rc2 -ne 0) { Write-Warning "docker exec returned non-zero ($rc2) while checking in-container realsense state." }
            }
        } finally {
            Remove-Item $tempHost -ErrorAction SilentlyContinue
        }
    } else {
        Write-Host "(no workspace container with $WorkspacePath mounted is running)" -ForegroundColor DarkGray
    }
}

function Resolve-AttachScriptPath {
    if ($AttachScript) {
        if (-not (Test-Path $AttachScript)) { throw "AttachScript path not found: $AttachScript" }
        return (Resolve-Path $AttachScript).Path
    }

    $candidates = @(
        (Join-Path $PSScriptRoot 'Attach-All-USB.ps1'),
        (Join-Path ([Environment]::GetFolderPath('Desktop')) 'Attach-All-USB.ps1'),
        (Join-Path ([Environment]::GetFolderPath('UserProfile')) 'OneDrive\Desktop\Attach-All-USB.ps1')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return (Resolve-Path $candidate).Path }
    }
    throw 'Could not locate Attach-All-USB.ps1. Pass -AttachScript <path>.'
}

# --- main -----------------------------------------------------------------

Assert-CommandAvailable -Name usbipd -InstallHint 'Install usbipd-win from https://github.com/dorssel/usbipd-win.'

switch ($Action) {
    'status' {
        Show-Status
        break
    }

    'detach' {
        Assert-RunningAsAdministrator
        Assert-CommandAvailable -Name docker -InstallHint 'Install Docker Desktop and ensure it is running.'

        $rows = Find-CameraRows -Target $VidPid
        # Normalise to an array for robust Count checks across PowerShell versions
        if (-not $rows -or @($rows).Count -eq 0) {
            throw "No USB device matching VID:PID '$VidPid' is visible to usbipd. Plug the camera into the host and rerun."
        }

        $container = Find-WorkspaceContainer -PreferredId $ContainerId -TargetWorkspacePath $WorkspacePath
        Stop-InContainerRealsense -Container $container

        # Verify the in-container realsense2_camera process is gone before detaching.
        # A race between SIGKILL and the USB detach can leave the driver in a bad state.
        if ($container) {
            Write-Host 'Verifying realsense2_camera process is no longer running in container...' -ForegroundColor Yellow
            $verifyScript = 'pgrep -f "realsense2_camera" >/dev/null 2>&1 && echo STILL_RUNNING || echo STOPPED'
            $verifyResult = docker exec $container bash -lc $verifyScript 2>$null
            if ($verifyResult -and $verifyResult.Trim() -eq 'STILL_RUNNING') {
                Write-Warning 'realsense2_camera process still detected after shutdown attempt. Forcing kill...'
                docker exec $container bash -lc 'pkill -9 -f realsense2_camera >/dev/null 2>&1 || true; sleep 1' 2>$null
            } else {
                Write-Host 'realsense2_camera is stopped.' -ForegroundColor Green
            }
        }

        foreach ($row in $rows) {
            if ($row.State -eq 'Attached') {
                Write-Host "Detaching RealSense $VidPid (BusId $($row.BusId)) from WSL..." -ForegroundColor Yellow
                $detachOk = $false
                for ($attempt = 1; $attempt -le 3; $attempt++) {
                    usbipd detach --busid $row.BusId
                    if ($LASTEXITCODE -eq 0) {
                        $detachOk = $true
                        break
                    }
                    Write-Warning "usbipd detach attempt $attempt failed (exit $LASTEXITCODE). Retrying in 2 s..."
                    Start-Sleep -Seconds 2
                }
                if (-not $detachOk) {
                    throw "usbipd detach failed for BusId $($row.BusId) after 3 attempts."
                }
                # Poll to confirm the device is no longer in 'Attached' state.
                $pollDeadline = (Get-Date).AddSeconds(10)
                $detachConfirmed = $false
                while ((Get-Date) -lt $pollDeadline) {
                    $refreshed = @(Find-CameraRows -Target $VidPid | Where-Object { $_.BusId -eq $row.BusId })
                    if (-not $refreshed -or $refreshed[0].State -ne 'Attached') {
                        $detachConfirmed = $true
                        break
                    }
                    Start-Sleep -Milliseconds 500
                }
                if (-not $detachConfirmed) {
                    Write-Warning "BusId $($row.BusId) still shows 'Attached' after detach -- USB state may be stale."
                }
            } else {
                Write-Host "BusId $($row.BusId) is already $($row.State); nothing to detach." -ForegroundColor DarkGray
            }
        }

        Write-Host 'Camera is now available to the Windows host.' -ForegroundColor Green
        Write-Host 'Next: run tools\realsense_text_trigger_capture.py on the host,' -ForegroundColor Green
        Write-Host '      and pass --external-camera-trigger-file <path> to the harness.' -ForegroundColor Green
        Write-Host 'FTDI / Dynamixel adapters were left attached so motor control is unaffected.' -ForegroundColor Green
        Write-Host ''
        Show-Status
        break
    }

    'attach' {
        Assert-RunningAsAdministrator
        $attachPath = Resolve-AttachScriptPath
        Write-Host "Delegating to $attachPath ..." -ForegroundColor Cyan
        & $attachPath @AttachArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Attach-All-USB.ps1 exited with code $LASTEXITCODE."
        }
        break
    }
}
