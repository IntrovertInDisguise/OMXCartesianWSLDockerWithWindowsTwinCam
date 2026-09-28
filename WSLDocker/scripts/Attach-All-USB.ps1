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
    "0403:6014" = 2
    "8086:0b3a" = 1
}

$requiredTopics = @(
    "/camera/color/image_raw",
    "/camera/color/camera_info",
    "/camera/aligned_depth_to_color/image_raw",
    "/camera/aligned_depth_to_color/camera_info"
)

$helperContainerName = "omx-realsense-helper"
$launchLogPath = "/tmp/realsense2_camera.launch.log"
$launchPidPath = "/tmp/realsense2_camera.launch.pid"

function Assert-RunningAsAdministrator {
    $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
    $isAdministrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

    if (-not $isAdministrator) {
        throw "Run this script from an elevated PowerShell session. usbipd bind/unbind requires administrator privileges."
    }
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
        $workspaceMount = $inspectData[0].Mounts | Where-Object { $_.Destination -eq $TargetWorkspacePath } | Select-Object -First 1
        if (-not $workspaceMount) {
            throw "Container $PreferredContainerId does not have $TargetWorkspacePath mounted."
        }

        return [pscustomobject]@{
            Id    = $PreferredContainerId
            Image = $inspectData[0].Config.Image
            Name  = $inspectData[0].Name.TrimStart("/")
            FullId = $inspectData[0].Id
            Hostname = $inspectData[0].Config.Hostname
            WorkspaceSource = $workspaceMount.Source
        }
    }

    $candidates = @(Get-WorkspaceContainers -TargetWorkspacePath $TargetWorkspacePath)
    if ($candidates.Count -eq 0) {
        throw "No running Docker container has $TargetWorkspacePath mounted. Start your ROS 2 Humble container first, then rerun this script."
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
        Write-Host ("[{0}] {1}  {2}  name={3}  hostname={4}" -f ($index + 1), $candidate.Id, $candidate.Image, $candidate.Name, $candidate.Hostname)
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
source /opt/ros/humble/setup.bash
if [ -f '$TargetWorkspacePath/install/setup.bash' ]; then
    source '$TargetWorkspacePath/install/setup.bash'
fi

if ros2 pkg prefix realsense2_camera >/dev/null 2>&1; then
    echo 'realsense2_camera is already installed.'
else
    echo 'Installing ros-humble-realsense2-camera...'
    if command -v sudo >/dev/null 2>&1 && [ "`$(id -u)" -ne 0 ]; then
        sudo apt-get update
        DEBIAN_FRONTEND=noninteractive sudo apt-get install -y ros-humble-realsense2-camera
    else
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y ros-humble-realsense2-camera
    fi
fi
"@

    Invoke-DockerBashStream -TargetContainerId $TargetContainerId -Script $script
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
if [ -f '$TargetWorkspacePath/install/setup.bash' ]; then
    source '$TargetWorkspacePath/install/setup.bash'
fi

if [ -x /opt/ros/humble/bin/rs-enumerate-devices ]; then
    /opt/ros/humble/bin/rs-enumerate-devices -s
else
    rs-enumerate-devices -s
fi
"@

    $result = Invoke-DockerBashStatus -TargetContainerId $TargetContainerId -Script $script
    $output = $result.Output
    $serials = @(
        $output |
        ForEach-Object {
            if ($_ -match 'Serial Number:\s*(\S+)') {
                $matches[1]
            }
            elseif ($_ -match '#(\d{6,})\b') {
                $matches[1]
            }
        } |
        Select-Object -Unique
    )

    return [pscustomobject]@{
        Serials = @($serials)
        Output = @($output)
        ExitCode = $result.ExitCode
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
        "camera_namespace:=/",
        "align_depth.enable:=true"
    )

    if ($UseSerial) {
        $launchArgs += "serial_no:=$DetectedSerialNo"
    }

    $launchCommand = $launchArgs -join " "
    Write-Host "Launching RealSense driver in $TargetContainerId..."
    $script = @"
set -e
source /opt/ros/humble/setup.bash
if [ -f '$TargetWorkspacePath/install/setup.bash' ]; then
    source '$TargetWorkspacePath/install/setup.bash'
fi

export ROS_DOMAIN_ID=$TargetRosDomainId
cd '$TargetWorkspacePath'
pkill -f 'realsense2_camera.*rs_launch.py' >/dev/null 2>&1 || true
pkill -f 'realsense2_camera_node' >/dev/null 2>&1 || true
rm -f '$launchPidPath'
nohup ros2 launch $launchCommand > '$launchLogPath' 2>&1 < /dev/null &
echo `$! > '$launchPidPath'
cat '$launchPidPath'
"@

    $output = Invoke-DockerBash -TargetContainerId $TargetContainerId -Script $script
    $launchProcessId = (($output | Select-Object -Last 1) | Out-String).Trim()
    if ([string]::IsNullOrWhiteSpace($launchProcessId)) {
        throw "Failed to capture the RealSense launch PID."
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
Write-Host "Checking camera device visibility in selected container..."
Assert-RealsenseDeviceVisibility -TargetContainer $container -TargetWorkspacePath $WorkspacePath
Write-Host "Checking realsense2_camera package in selected container..."
Ensure-RealsensePackage -TargetContainerId $container.Id -TargetWorkspacePath $WorkspacePath
Write-Host "Detecting RealSense camera..."
$cameraSelection = Resolve-RealsenseSelection -TargetContainerId $container.Id -TargetWorkspacePath $WorkspacePath -RequestedSerial $SerialNo
Write-Host "Selected RealSense serial $($cameraSelection.SerialNo)."
$launchPid = Start-RealsensePublisher `
    -TargetContainerId $container.Id `
    -TargetWorkspacePath $WorkspacePath `
    -TargetRosDomainId $RosDomainId `
    -DetectedSerialNo $cameraSelection.SerialNo `
    -UseSerial $cameraSelection.SerialNeeded

Write-Host "Checking required ROS topics..."
$topicResults = Test-RealsenseTopicPublishing `
    -TargetContainerId $container.Id `
    -TargetWorkspacePath $WorkspacePath `
    -TargetRosDomainId $RosDomainId `
    -Topics $requiredTopics

$failedTopics = @($topicResults | Where-Object { -not $_.Success })
if ($failedTopics.Count -gt 0) {
    Write-Error "RealSense launched, but these topics did not publish within the timeout: $($failedTopics.Topic -join ', ')"
    Write-Host "Recent launch log:"
    Get-RealsenseLogTail -TargetContainerId $container.Id | Out-Host
    exit 1
}

Write-Host ""
Write-Host "RealSense publisher is running."
Write-Host "Container: $($container.Id) ($($container.Image), $($container.Name))"
Write-Host "Hostname: $($container.Hostname)"
Write-Host "ROS_DOMAIN_ID: $RosDomainId"
Write-Host "Serial number required: $($cameraSelection.SerialNeeded.ToString().ToLowerInvariant())"
Write-Host "Selected serial: $($cameraSelection.SerialNo)"
Write-Host "Launch PID: $launchPid"
Write-Host "Log file: $launchLogPath"
Write-Host "Verified topics:"
$topicResults | ForEach-Object {
    Write-Host "  $($_.Topic)"
}
Write-Host ""
Write-Host "The driver was started detached inside the container and will keep running after this script exits."