# Copyright 2026 Zack Strathe
# SPDX-License-Identifier: LicenseRef-Purrceptron-Source-Available-1.0
# Noncommercial use only; see LICENSE.md. Not licensed for AI/ML training or
# text and data mining (rights reserved under Art. 4(3) Directive (EU) 2019/790).

# Let the Pi reach train_worker.py running inside WSL (Windows 10, or any
# Windows with WSL's default NAT networking). Run once, in an administrator
# PowerShell:
#
#   powershell -ExecutionPolicy Bypass -File scripts\windows_port_forward.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\windows_port_forward.ps1 -Remove
#
# WSL already relays Windows' own localhost:<WorkerPort> to a WSL program
# listening on that port. This adds one fixed rule that forwards the LAN port
# <ListenPort> to that localhost port, plus a firewall rule. Neither depends on
# WSL's internal address, so nothing needs updating after a reboot.
# The Pi then uses this PC's address and port <ListenPort>.

param(
    [int]$ListenPort = 5005,
    [int]$WorkerPort = 5004,
    [switch]$Remove
)
$ErrorActionPreference = "Stop"
$ruleName = "Cat training worker (WSL)"

$admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
         ).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $admin) { throw "Run this from an administrator PowerShell." }

# Always clear our rules first, so running it again is safe.
netsh interface portproxy delete v4tov4 listenaddress=0.0.0.0 listenport=$ListenPort 2>$null | Out-Null
Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue | Remove-NetFirewallRule

if ($Remove) {
    Write-Host "[OK] Removed the port $ListenPort forward and its firewall rule."
    exit 0
}

# Port forwarding is done by the IP Helper service.
Set-Service iphlpsvc -StartupType Automatic
if ((Get-Service iphlpsvc).Status -ne "Running") { Start-Service iphlpsvc }

netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$ListenPort `
    connectaddress=127.0.0.1 connectport=$WorkerPort | Out-Null
if ($LASTEXITCODE -ne 0) { throw "netsh portproxy failed" }
New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP `
    -LocalPort $ListenPort -Action Allow -Profile Private | Out-Null

Write-Host "[OK] LAN port $ListenPort now forwards to the WSL worker on port $WorkerPort."
# Only adapters with a default gateway: skips WSL, Hyper-V and VMware virtual networks.
$ips = Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } |
    ForEach-Object { $_.IPv4Address.IPAddress }
Write-Host "On the Pi's Training tab use this PC's address and port ${ListenPort}:"
$ips | ForEach-Object { Write-Host "  $_" }
Write-Host "The firewall rule allows Private networks only; mark your home network Private"
Write-Host "in Windows network settings if the Pi can't connect."
