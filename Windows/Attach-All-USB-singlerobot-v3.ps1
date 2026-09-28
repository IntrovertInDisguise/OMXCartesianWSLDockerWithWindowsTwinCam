[CmdletBinding()]
param(
    [int]$RosDomainId = 0,
    [string]$WorkspacePath = "/workspaces/omx_ros2",
    [string]$ContainerId,
    [string]$Hostname,
    [string]$SerialNo
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$targets = @{
    # FTDI USB interface for the OpenManipulator
    "0403:6014" = 1

    # Leave the Intel RealSense under Windows.
    # It will be used by windows_aruco_udp_sender.py.
}



function Assert-RunningAsAdministrator {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    $isAdministrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if (-not $isAdministrator) {
        throw "Run this script from an elevated PowerShell session. usbipd bind/unbind requires administrator privileges."
    }
}
function Detach-UsbDevice {
    param(
        [Parameter(Mandatory)]
        [string]$VidPid
    )

    $usbipd = Get-Command usbipd -ErrorAction Stop

    $devices = & $usbipd.Path list

    foreach ($line in $devices) {

        if ($line -match $VidPid) {

            # Extract BUSID (first column)
            $parts = ($line -split '\s+') | Where-Object { $_ -ne "" }

            if ($parts.Count -lt 2) {
                continue
            }

            $busId = $parts[0]

            if ($line -match "Attached") {

                Write-Host "Detaching $VidPid from WSL (BUSID $busId)..."

                & $usbipd.Path detach --busid $busId

                if ($LASTEXITCODE -ne 0) {
                    throw "Failed to detach USB device $VidPid."
                }

                Write-Host "Detached."
            }
            else {
                Write-Host "$VidPid already owned by Windows."
            }

            return
        }
    }

    Write-Warning "$VidPid not found."
}
function Invoke-NativeCommand {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [string[]]$Arguments = @()
    )

    $previousErrorActionPreference = $ErrorActionPreference
    $hasNativePreference = Test-Path Variable:PSNativeCommandUseErrorActionPreference

    if ($hasNativePreference) {
        $previousNativePreference = $PSNativeCommandUseErrorActionPreference
    }

    try {
        $ErrorActionPreference = "Continue"
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $false
        }

        $output = & $FilePath @Arguments 2>&1 | ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) {
                $_.ToString()
            }
            else {
                [string]$_
            }
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
    }

    return [pscustomobject]@{
        Output   = @($output)
        ExitCode = $exitCode
    }
}

function Get-UsbipdRows {
    usbipd list | ForEach-Object {
        if ($_ -match '^(?<BusId>\S+)\s+(?<VidPid>[0-9A-Fa-f]{4}:[0-9A-Fa-f]{4})\s+.*?\s+(?<State>Not shared|Shared|Attached)$') {
            [pscustomobject]@{
                BusId = $matches.BusId
                VidPid = $matches.VidPid.ToLower()
                State = $matches.State
                Raw   = $_
            }
        }
    }
}

function Attach-ExpectedUsbDevices {
    param(
        [Parameter(Mandatory)]
        [hashtable]$TargetMap
    )

    usbipd unbind -a

    foreach ($vidpid in $TargetMap.Keys) {
        $expectedCount = $TargetMap[$vidpid]

        for ($attempt = 1; $attempt -le 5; $attempt++) {
            $rows = @(Get-UsbipdRows | Where-Object { $_.VidPid -eq $vidpid })
            $attached = @($rows | Where-Object { $_.State -eq "Attached" })

            if ($attached.Count -ge $expectedCount) {
                break
            }

            $pending = @($rows | Where-Object { $_.State -ne "Attached" })
            foreach ($device in $pending) {
                Write-Host "Attaching $vidpid at BUSID $($device.BusId) [$($device.State)]..."
                usbipd bind --busid $device.BusId | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "bind failed for $($device.BusId)"
                    continue
                }

                usbipd attach --busid $device.BusId --wsl docker-desktop
                if ($LASTEXITCODE -ne 0) {
                    Write-Warning "attach failed for $($device.BusId)"
                }
            }

            Start-Sleep -Seconds 2
        }

        $final = @(Get-UsbipdRows | Where-Object { $_.VidPid -eq $vidpid -and $_.State -eq "Attached" })
        if ($final.Count -lt $expectedCount) {
            throw "Expected $expectedCount attached device(s) for $vidpid, but found $($final.Count)."
        }
    }

    Write-Host "All expected USB devices are attached."
}

function Invoke-DockerBash {
    param(
        [Parameter(Mandatory)]
        [string]$TargetContainerId,

        [Parameter(Mandatory)]
        [string]$Script
    )

    $normalizedScript = $Script -replace "`r`n", "`n" -replace "`r", "`n"
    $encodedScript = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalizedScript))
    $dockerCommand = "printf '%s' '$encodedScript' | base64 -d | bash"
    $result = Invoke-NativeCommand -FilePath "docker" -Arguments @("exec", $TargetContainerId, "bash", "-lc", $dockerCommand)
    if ($result.ExitCode -ne 0) {
        $message = ($result.Output | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($message)) {
            $message = "docker exec failed for container $TargetContainerId."
        }
        throw $message
    }

    return $result.Output
}

function Invoke-DockerBashStream {
    param(
        [Parameter(Mandatory)]
        [string]$TargetContainerId,

        [Parameter(Mandatory)]
        [string]$Script
    )

    $normalizedScript = $Script -replace "`r`n", "`n" -replace "`r", "`n"
    $encodedScript = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalizedScript))
    $dockerCommand = "printf '%s' '$encodedScript' | base64 -d | bash"

    $previousErrorActionPreference = $ErrorActionPreference
    $hasNativePreference = Test-Path Variable:PSNativeCommandUseErrorActionPreference
    if ($hasNativePreference) {
        $previousNativePreference = $PSNativeCommandUseErrorActionPreference
    }

    try {
        $ErrorActionPreference = "Continue"
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $false
        }

        & docker exec $TargetContainerId bash -lc $dockerCommand
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorActionPreference
        if ($hasNativePreference) {
            $PSNativeCommandUseErrorActionPreference = $previousNativePreference
        }
    }

    if ($exitCode -ne 0) {
        throw "docker exec failed for container $TargetContainerId with exit code $exitCode."
    }
}

function Get-WorkspaceContainers {
    param(
        [Parameter(Mandatory)]
        [string]$TargetWorkspacePath
    )

    $dockerPs = Invoke-NativeCommand -FilePath "docker" -Arguments @("ps", "--format", "{{.ID}}`t{{.Image}}`t{{.Names}}")
    if ($dockerPs.ExitCode -ne 0) {
        throw "docker ps failed."
    }

    $candidates = @()

    foreach ($line in $dockerPs.Output) {
        $text = ($line | Out-String).Trim()
        if ([string]::IsNullOrWhiteSpace($text)) {
            continue
        }

        $parts = $text -split "`t"
        if ($parts.Count -lt 3) {
            continue
        }

        $candidateId = $parts[0]
        $testResult = Invoke-NativeCommand -FilePath "docker" -Arguments @("exec", $candidateId, "bash", "-lc", "test -d '$TargetWorkspacePath'")
        if ($testResult.ExitCode -ne 0) {
            continue
        }

        $inspect = Invoke-NativeCommand -FilePath "docker" -Arguments @("inspect", $candidateId)
        if ($inspect.ExitCode -ne 0) {
            continue
        }

        $inspectData = $inspect.Output | ConvertFrom-Json
        $workspaceMount = $inspectData[0].Mounts | Where-Object { $_.Destination -eq $TargetWorkspacePath } | Select-Object -First 1
        if (-not $workspaceMount) {
            continue
        }

        $candidates += [pscustomobject]@{
            Id    = $candidateId
            Image = $parts[1]
            Name  = $parts[2]
            FullId = $inspectData[0].Id
            Hostname = $inspectData[0].Config.Hostname
            WorkspaceSource = $workspaceMount.Source
        }
    }

    return @($candidates)
}

function Resolve-WorkspaceContainer {
    param(
        [string]$PreferredContainerId,

        [string]$PreferredHostname,

        [Parameter(Mandatory)]
        [string]$TargetWorkspacePath
    )

    if ($PreferredContainerId) {
        $inspect = Invoke-NativeCommand -FilePath "docker" -Arguments @("inspect", $PreferredContainerId)
        if ($inspect.ExitCode -ne 0) {
            throw "docker inspect failed for container $PreferredContainerId."
        }

        $inspectData = $inspect.Output | ConvertFrom-Json
        $workspaceMount = $inspectData[0].Mounts |
            Where-Object { $_.Destination -eq $TargetWorkspacePath } |
            Select-Object -First 1

        if (-not $workspaceMount) {
            throw "Container $PreferredContainerId does not have $TargetWorkspacePath mounted."
        }

        return [pscustomobject]@{
            Id              = $PreferredContainerId
            Image           = $inspectData[0].Config.Image
            Name            = $inspectData[0].Name.TrimStart("/")
            FullId          = $inspectData[0].Id
            Hostname        = $inspectData[0].Config.Hostname
            WorkspaceSource = $workspaceMount.Source
        }
    }

    $candidates = @(Get-WorkspaceContainers -TargetWorkspacePath $TargetWorkspacePath)

    if ($candidates.Count -eq 0) {
        throw "No running Docker container has $TargetWorkspacePath mounted. Start your ROS 2 Humble container first, then rerun this script."
    }

    #
    # Single-robot fast path
    #
    if ($candidates.Count -eq 1) {
        $candidate = $candidates[0]

        Write-Host ("Automatically selected container: {0}  {1}  name={2}  hostname={3}" -f `
            $candidate.Id,
            $candidate.Image,
            $candidate.Name,
            $candidate.Hostname)

        return $candidate
    }

    if ($PreferredHostname) {
        $matches = @(
            $candidates |
            Where-Object {
                $_.Hostname -eq $PreferredHostname -or
                $_.Name -eq $PreferredHostname
            }
        )

        if ($matches.Count -eq 0) {
            throw "No running workspace container matched hostname or name '$PreferredHostname'."
        }

        if ($matches.Count -gt 1) {
            throw "More than one workspace container matched hostname or name '$PreferredHostname'. Use -ContainerId to disambiguate."
        }

        return $matches[0]
    }

    Write-Host "Available containers with $TargetWorkspacePath mounted:"

    for ($index = 0; $index -lt $candidates.Count; $index++) {
        $candidate = $candidates[$index]

        Write-Host (
            "[{0}] {1}  {2}  name={3}  hostname={4}" -f
            ($index + 1),
            $candidate.Id,
            $candidate.Image,
            $candidate.Name,
            $candidate.Hostname
        )
    }

    while ($true) {
        $selection = Read-Host "Choose the container number, name, or hostname to use"

        if ($selection -match '^\d+$') {
            $selectedIndex = [int]$selection - 1

            if ($selectedIndex -ge 0 -and $selectedIndex -lt $candidates.Count) {
                return $candidates[$selectedIndex]
            }
        }

        $namedMatch = @(
            $candidates |
            Where-Object {
                $_.Name -eq $selection -or
                $_.Hostname -eq $selection
            }
        )

        if ($namedMatch.Count -eq 1) {
            return $namedMatch[0]
        }

        Write-Warning "Enter a valid container number, name, or hostname."
    }
}
function Assert-RealsenseDeviceVisibility {
    param(
        [Parameter(Mandatory)]
        [pscustomobject]$TargetContainer,

        [Parameter(Mandatory)]
        [string]$TargetWorkspacePath
    )

    $script = @"
if ls /dev/bus/usb/*/* >/dev/null 2>&1; then
    exit 0
fi

exit 1
"@

    for ($attempt = 1; $attempt -le 5; $attempt++) {
        Write-Host "Checking camera device visibility in $($TargetContainer.Name) (attempt $attempt/5)..."
        $result = Invoke-DockerBashStatus -TargetContainerId $TargetContainer.Id -Script $script
        if ($result.ExitCode -eq 0) {
            return
        }

        if ($attempt -lt 5) {
            Start-Sleep -Seconds 2
        }
    }

    throw "Selected container $($TargetContainer.Name) (hostname $($TargetContainer.Hostname)) does not expose /dev/bus/usb inside the container, so RealSense cannot be detected. This script will not create a helper container. Start a container that already has USB bus access, or update the ROS workspace devcontainer runArgs to mount /dev (for example --volume=/dev:/dev, or at minimum --volume=/dev/bus/usb:/dev/bus/usb) and rebuild the container."
}

function Ensure-RealsensePackage {
    param(
        [Parameter(Mandatory)]
        [string]$TargetContainerId,

        [Parameter(Mandatory)]
        [string]$TargetWorkspacePath
    )

    $script = @"
set -e

source /opt/ros/humble/setup.bash

if dpkg -s ros-humble-realsense2-camera >/dev/null 2>&1; then
    echo "realsense2_camera already installed."
else
    echo "Installing ros-humble-realsense2-camera..."

    export DEBIAN_FRONTEND=noninteractive

    apt-get update
    apt-get install -y ros-humble-realsense2-camera

    echo "Verifying installation..."

    dpkg -s ros-humble-realsense2-camera

    ls -la /opt/ros/humble/share/realsense2_camera
fi

echo ""
echo "ROS package discovery:"
ros2 pkg list | grep realsense || true
"@

    Invoke-DockerBashStream `
        -TargetContainerId $TargetContainerId `
        -Script $script
}

function Get-RealsenseSerials {
    param(
        [Parameter(Mandatory)]
        [string]$TargetContainerId,

        [Parameter(Mandatory)]
        [string]$TargetWorkspacePath
    )

    $script = @"
source /opt/ros/humble/setup.bash

if command -v rs-enumerate-devices >/dev/null 2>&1; then
    rs-enumerate-devices -s
elif [ -x /opt/ros/humble/bin/rs-enumerate-devices ]; then
    /opt/ros/humble/bin/rs-enumerate-devices -s
else
    echo "rs-enumerate-devices not found"
    exit 1
fi
"@

    $result = Invoke-DockerBashStatus `
        -TargetContainerId $TargetContainerId `
        -Script $script

    $output = $result.Output

    $serials = @(
        $output |
        ForEach-Object {
            $line = $_.Trim()

            # Newer librealsense output:
            # Intel RealSense D435I    348522071053    5.17.0.10
            if ($line -match 'Intel\s+RealSense.*?\b(\d{9,})\b') {
                $matches[1]
            }
            # Older output:
            elseif ($line -match 'Serial Number:\s*(\S+)') {
                $matches[1]
            }
            # Very old output:
            elseif ($line -match '#(\d{6,})\b') {
                $matches[1]
            }
            # Generic fallback: any long numeric token
            elseif ($line -match '\b(\d{9,})\b') {
                $matches[1]
            }
        } |
        Where-Object { $_ } |
        Select-Object -Unique
    )

    Write-Host "Detected RealSense serial(s): $($serials -join ', ')"

    [pscustomobject]@{
        Serials = $serials
        Output  = $output
    }
}
function Resolve-RealsenseSelection {
    param(
        [Parameter(Mandatory)]
        [string]$TargetContainerId,

        [Parameter(Mandatory)]
        [string]$TargetWorkspacePath,

        [string]$RequestedSerial
    )

    $scan = $null
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        Write-Host "Scanning for RealSense devices in $TargetContainerId (attempt $attempt/10)..."
        $scan = Get-RealsenseSerials -TargetContainerId $TargetContainerId -TargetWorkspacePath $TargetWorkspacePath
        $serials = @($scan.Serials)
        if ($serials.Count -gt 0) {
            break
        }

        if ($attempt -lt 10) {
            Write-Host "Waiting for RealSense device to appear inside $TargetContainerId..."
            Start-Sleep -Seconds 2
        }
    }

    $serials = @($scan.Serials)
    if ($serials.Count -eq 0) {
        $scanOutput = (@($scan.Output) | Out-String).Trim()
        throw "No Intel RealSense cameras were detected inside container $TargetContainerId. Last scan output: $scanOutput"
    }

    if ($RequestedSerial) {
        if ($serials -notcontains $RequestedSerial) {
            throw "Requested serial $RequestedSerial was not detected. Available serials: $($serials -join ', ')"
        }

        return [pscustomobject]@{
            SerialNo     = $RequestedSerial
            SerialNeeded = ($serials.Count -gt 1)
        }
    }

    if ($serials.Count -eq 1) {
        return [pscustomobject]@{
            SerialNo     = $serials[0]
            SerialNeeded = $false
        }
    }

    Write-Host "Multiple RealSense cameras detected:"
    for ($index = 0; $index -lt $serials.Count; $index++) {
        Write-Host ("[{0}] {1}" -f ($index + 1), $serials[$index])
    }

    while ($true) {
        $selection = Read-Host "Enter the camera number or exact serial to use"
        if ($selection -match '^\d+$') {
            $selectedIndex = [int]$selection - 1
            if ($selectedIndex -ge 0 -and $selectedIndex -lt $serials.Count) {
                return [pscustomobject]@{
                    SerialNo     = $serials[$selectedIndex]
                    SerialNeeded = $true
                }
            }
        }

        if ($serials -contains $selection) {
            return [pscustomobject]@{
                SerialNo     = $selection
                SerialNeeded = $true
            }
        }

        Write-Warning "Enter a valid camera number or serial."
    }
}

function Start-RealsensePublisher {
    param(
        [Parameter(Mandatory)]
        [string]$TargetContainerId,

        [Parameter(Mandatory)]
        [string]$TargetWorkspacePath,

        [Parameter(Mandatory)]
        [int]$TargetRosDomainId,

        [Parameter(Mandatory)]
        [string]$DetectedSerialNo,

        [Parameter(Mandatory)]
        [bool]$UseSerial
    )

    $launchArgs = @(
        "realsense2_camera",
        "rs_launch.py",
        "camera_namespace:=camera",
        "camera_name:=camera",
        "align_depth.enable:=true"
    )

#
# Single-robot configuration:
# Do not pass serial_no.
# realsense2_camera automatically selects the only connected device.
#

    $launchCommand = $launchArgs -join " "

    Write-Host "Launching RealSense driver in $TargetContainerId..."

    $script = @"
set -e

source /opt/ros/humble/setup.bash

export ROS_DOMAIN_ID=$TargetRosDomainId

pkill -f 'realsense2_camera.*rs_launch.py' >/dev/null 2>&1 || true
pkill -f 'realsense2_camera_node' >/dev/null 2>&1 || true

rm -f '$launchPidPath'

nohup ros2 launch $launchCommand > '$launchLogPath' 2>&1 < /dev/null &

echo `$! > '$launchPidPath'

sleep 5

cat '$launchPidPath'

echo ""
echo "========== Launch Log =========="
cat '$launchLogPath' || true
echo "================================"
"@

    $output = Invoke-DockerBash `
        -TargetContainerId $TargetContainerId `
        -Script $script

    $launchProcessId = (($output | Select-Object -First 1) | Out-String).Trim()

    if ([string]::IsNullOrWhiteSpace($launchProcessId)) {
        throw "Failed to capture RealSense launch PID."
    }

    return $launchProcessId
}
function Test-RealsenseTopicPublishing {
    param(
        [Parameter(Mandatory)]
        [string]$TargetContainerId,

        [Parameter(Mandatory)]
        [string]$TargetWorkspacePath,

        [Parameter(Mandatory)]
        [int]$TargetRosDomainId,

        [Parameter(Mandatory)]
        [string[]]$Topics
    )

    $results = @()

    foreach ($topic in $Topics) {
        Write-Host "Verifying topic publication: $topic"
        $script = @"
source /opt/ros/humble/setup.bash
if [ -f '$TargetWorkspacePath/install/setup.bash' ]; then
    source '$TargetWorkspacePath/install/setup.bash'
fi

export ROS_DOMAIN_ID=$TargetRosDomainId
timeout 12s ros2 topic echo --once --qos-reliability best_effort '$topic' >/dev/null 2>&1
"@

        $result = Invoke-DockerBashStatus -TargetContainerId $TargetContainerId -Script $script
        $results += [pscustomobject]@{
            Topic   = $topic
            Success = ($result.ExitCode -eq 0)
        }
    }

    return @($results)
}

function Get-RealsenseLogTail {
    param(
        [Parameter(Mandatory)]
        [string]$TargetContainerId
    )

    $script = "if [ -f '$launchLogPath' ]; then tail -n 80 '$launchLogPath'; fi"
    return Invoke-DockerBash -TargetContainerId $TargetContainerId -Script $script
}

function Invoke-DockerBashStatus {
    param(
        [Parameter(Mandatory)]
        [string]$TargetContainerId,

        [Parameter(Mandatory)]
        [string]$Script
    )

    $normalizedScript = $Script -replace "`r`n", "`n" -replace "`r", "`n"
    $encodedScript = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($normalizedScript))
    $dockerCommand = "printf '%s' '$encodedScript' | base64 -d | bash"
    return Invoke-NativeCommand -FilePath "docker" -Arguments @("exec", $TargetContainerId, "bash", "-lc", $dockerCommand)
}

Assert-RunningAsAdministrator
Attach-ExpectedUsbDevices -TargetMap $targets
Write-Host "Resolving ROS workspace container..."
$container = Resolve-WorkspaceContainer -PreferredContainerId $ContainerId -PreferredHostname $Hostname -TargetWorkspacePath $WorkspacePath
Write-Host "Using workspace container $($container.Name) ($($container.Id), hostname $($container.Hostname))."
Assert-RunningAsAdministrator

Attach-ExpectedUsbDevices -TargetMap $targets

Write-Host "Resolving ROS workspace container..."

$container = Resolve-WorkspaceContainer `
    -PreferredContainerId $ContainerId `
    -PreferredHostname $Hostname `
    -TargetWorkspacePath $WorkspacePath

Write-Host "Using workspace container $($container.Name) ($($container.Id), hostname $($container.Hostname))."

Write-Host ""
Write-Host "USB configuration complete."
Write-Host ""
Write-Host "FTDI interface has been attached to WSL/Docker."
Write-Host "Forcing all Intel RealSense devices back to Windows..."

$devices = usbipd list

foreach ($line in $devices) {
    if ($line -match '^(?<bus>\S+)\s+(?<vidpid>8086:[0-9A-Fa-f]{4}).*Attached') {
        $bus = $matches.bus
        Write-Host "Detaching $($matches.vidpid) (BUSID $bus)..."
        usbipd detach --busid $bus
    }
}

Write-Host "All Intel RealSense devices are now owned by Windows."
Write-Host ""
Write-Host "Container: $($container.Id)"
Write-Host "Hostname : $($container.Hostname)"
Write-Host "ROS_DOMAIN_ID: $RosDomainId"
Write-Host ""
Write-Host "Start the camera on Windows using:"
Write-Host ""
Write-Host "python windows_aruco_udp_sender.py --realsense --udp-host host.docker.internal --udp-port 5005 --show"
Write-Host ""
Write-Host "The ROS container now owns only the robot USB interface."
Write-Host "The Windows host owns the RealSense camera."