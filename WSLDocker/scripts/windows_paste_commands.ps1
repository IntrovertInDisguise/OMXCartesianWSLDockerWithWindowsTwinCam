<#
Paste-ready PowerShell commands for Windows host to attach/detach camera
and run the host recorder for paper-grade captures.

Usage: Open an elevated PowerShell (Admin) and paste sections as needed.
#>

param(
	[ValidateSet('orchestrator','manual','attach','status')]
	[string]$Mode = 'orchestrator',
	[string]$TriggerFile = 'C:\tmp\omx_camera_trigger.txt',
	[string]$ReadyFile = 'C:\tmp\realsense_ready.txt',
	[string]$OutputDir = 'C:\tmp\realsense_out',
	[string]$SidecarDir = 'C:\tmp\realsense_sidecar',
	[string]$Python = 'python',
	[switch]$Emulate,
	[int]$WaitTimeout = 0
)

Set-StrictMode -Version Latest

function To-MntPath($winPath) {
	if ($winPath -match '^([A-Za-z]):(.*)$') {
		$drive = $matches[1].ToLower()
		$rest = $matches[2] -replace '\\','/'
		return "/mnt/$drive$rest"
	}
	return $winPath
}

Write-Host "Windows capture helper — mode: $Mode" -ForegroundColor Cyan

switch ($Mode) {
	'orchestrator' {
		Write-Host "Launching Run-CaptureSession.ps1 (recommended orchestrator)." -ForegroundColor Green
		& .\scripts\Run-CaptureSession.ps1 -TriggerFile $TriggerFile -OutputDir $OutputDir -Python $Python
		break
	}
	'manual' {
		Write-Host "Manual mode: detach camera, launch recorder with --ready-file, and wait for READY." -ForegroundColor Yellow
		Write-Host "Detaching in-container camera and handing it to Windows..."
		& .\scripts\toggle_realsense_usb.ps1 -Action detach

		Write-Host "Initializing trigger file: $TriggerFile"
		$dir = Split-Path -Parent $TriggerFile
		if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
		Set-Content -Path $TriggerFile -Value '' -Encoding utf8 -NoNewline

		$args = @('tools\realsense_text_trigger_capture.py', $TriggerFile, '--output-dir', $OutputDir, '--ready-file', $ReadyFile, '--sidecar-dir', $SidecarDir)
		if ($Emulate) { $args += '--emulate' }

		Write-Host "Starting recorder: $Python $($args -join ' ')" -ForegroundColor Cyan
		$proc = Start-Process -FilePath $Python -ArgumentList $args -PassThru
		Write-Host "Recorder started (PID $($proc.Id)). Waiting for READY file: $ReadyFile"

		$start = Get-Date
		while (-not (Test-Path $ReadyFile)) {
			Start-Sleep -Seconds 1
			if ($WaitTimeout -gt 0 -and ((Get-Date) - $start).TotalSeconds -ge $WaitTimeout) {
				Write-Host "Timed out waiting for READY after $WaitTimeout seconds." -ForegroundColor Red
				break
			}
		}

		if (Test-Path $ReadyFile) {
			Write-Host "READY detected: $ReadyFile" -ForegroundColor Green
		}

		$mnt = To-MntPath $TriggerFile
		Write-Host "Inside the container, run the harness (camera on host):" -ForegroundColor Cyan
		Write-Host "python3 tools/hardware_harness_contact_gated_with_aruco_v2.py --alignment-policy off --external-camera-trigger-file $mnt --spring-specimen <name> --calibration-json-path logs/<calib_dir>/calib_result_*.json" -ForegroundColor Gray

		Write-Host "When done, write QUIT into the trigger file to stop the recorder, then re-attach the camera:" -ForegroundColor Yellow
		Write-Host "  echo QUIT > $TriggerFile" -ForegroundColor Gray
		Write-Host "  .\scripts\toggle_realsense_usb.ps1 -Action attach -AttachArgs @('-RosDomainId','0')" -ForegroundColor Gray

		break
	}
	'attach' {
		Write-Host "Re-attaching camera to container and restarting in-container camera node..." -ForegroundColor Cyan
		& .\scripts\toggle_realsense_usb.ps1 -Action attach -AttachArgs @('-RosDomainId','0')
		break
	}
	'status' {
		Write-Host "Toggle script status:" -ForegroundColor Cyan
		& .\scripts\toggle_realsense_usb.ps1 -Action status
		break
	}
}

Write-Host "Notes:`n - Ensure $TriggerFile maps to /mnt/c/... inside the container.`n - Prefer orchestrator mode, or use manual mode when you need explicit READY-file synchronization." -ForegroundColor DarkGray
