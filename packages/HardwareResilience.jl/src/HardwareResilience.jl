# SPDX-License-Identifier: PMPL-1.0-or-later
# Copyright (c) 2026 Jonathan D.A. Jewell (hyperpolymath) <j.d.a.jewell@open.ac.uk>
#
# HardwareResilience.jl - Hardware resilience detection and monitoring.
#
# Detects ECC memory, redundant power supplies, RAID arrays, thermal monitoring,
# watchdog timers, and provides a self-healing kernel guardian for supervised
# operation execution.

"""
    HardwareResilience

Hardware resilience detection and monitoring for Linux systems. Detects ECC memory,
RAID arrays, thermal zones, watchdog timers, and redundant power supplies, producing
a comprehensive resilience assessment. Includes a supervised execution guardian with
retry logic and self-healing capabilities.

# Key Features
- ECC memory detection via EDAC subsystem and DMI/SMBIOS
- Software and hardware RAID health monitoring
- Thermal zone querying with critical trip point alerts
- Watchdog timer and redundant PSU detection
- `KernelGuardian` for fault-tolerant operation execution with exponential backoff

# Example
```julia
using HardwareResilience
report = detect_resilience()
print_resilience_report(report)
```
"""
module HardwareResilience

using Dates

export KernelGuardian, ResilienceReport, ThermalZone
export monitor_kernel, detect_resilience, check_ecc_memory
export check_raid_status, check_thermal_zones, check_watchdog_timers
export check_redundant_power, print_resilience_report

# ============================================================================
# Types
# ============================================================================

"""
    KernelGuardian(name, status, max_retries, retry_delay_ms, failure_log)

A supervised execution monitor that wraps operations with retry logic,
failure logging, and self-healing capabilities.

# Fields
- `name`: identifier for this guardian instance
- `status`: current status (`:active`, `:degraded`, `:failed`)
- `max_retries`: maximum retry attempts before marking as failed
- `retry_delay_ms`: milliseconds between retry attempts
- `failure_log`: timestamped log of failures encountered
"""
mutable struct KernelGuardian
    name::String
    status::Symbol
    max_retries::Int
    retry_delay_ms::Int
    failure_log::Vector{Tuple{DateTime, String}}

    function KernelGuardian(name::String; max_retries::Int=3, retry_delay_ms::Int=100)
        new(name, :active, max_retries, retry_delay_ms, Tuple{DateTime, String}[])
    end
end

# Backward-compatible constructor for (name, status) form
KernelGuardian(name::String, status::Symbol) = KernelGuardian(name; max_retries=3)

"""
    ThermalZone

Represents a thermal monitoring zone from the Linux hwmon/thermal subsystem.
"""
struct ThermalZone
    name::String
    type::String
    temp_celsius::Float64
    trip_point_celsius::Union{Float64, Nothing}
    is_critical::Bool
end

"""
    ResilienceReport

Comprehensive hardware resilience assessment for the current system.
"""
struct ResilienceReport
    timestamp::DateTime
    ecc_memory::Bool
    ecc_error_count::Int
    raid_present::Bool
    raid_status::Symbol            # :optimal, :degraded, :rebuilding, :none
    raid_details::String
    thermal_zones::Vector{ThermalZone}
    thermal_ok::Bool
    watchdog_present::Bool
    watchdog_devices::Vector{String}
    redundant_power::Bool
    psu_count::Int
    overall_resilience::Symbol     # :high, :medium, :low
    findings::Vector{String}
end

# ============================================================================
# Supervised operation execution
# ============================================================================

"""
    monitor_kernel(guardian::KernelGuardian, op::Function) -> Any

Execute an operation under the guardian's supervision. If the operation fails,
it is retried up to `guardian.max_retries` times with exponential backoff.
Failures are logged with timestamps. If all retries are exhausted, the
guardian transitions to `:degraded` status and returns `nothing`.

# Arguments
- `guardian`: the `KernelGuardian` supervising this operation
- `op`: a zero-argument function to execute

# Returns
The result of `op()` on success, or `nothing` if all retries fail.
"""
function monitor_kernel(g::KernelGuardian, op)
    for attempt in 1:g.max_retries
        try
            result = op()
            # Successful execution: ensure guardian is active
            if g.status == :degraded
                g.status = :active
            end
            return result
        catch e
            err_msg = sprint(showerror, e)
            push!(g.failure_log, (now(), err_msg))

            if attempt < g.max_retries
                # Exponential backoff: delay doubles each attempt
                delay_ms = g.retry_delay_ms * (2 ^ (attempt - 1))
                sleep(delay_ms / 1000.0)
            else
                g.status = :degraded
                @warn "KernelGuardian '$(g.name)' exhausted retries ($(g.max_retries)). " *
                      "Status: degraded. Last error: $err_msg"
                return nothing
            end
        end
    end
    return nothing
end

# ============================================================================
# ECC Memory detection
# ============================================================================

"""
    check_ecc_memory() -> NamedTuple{(:present, :error_count, :details), Tuple{Bool, Int, String}}

Detect ECC (Error-Correcting Code) memory by querying the Linux EDAC
(Error Detection And Correction) subsystem at `/sys/devices/system/edac/`.

Returns whether ECC is present, the correctable error count, and a
description string.
"""
function check_ecc_memory()
    if !Sys.islinux()
        return (present=false, error_count=0, details="ECC detection requires Linux EDAC subsystem")
    end

    edac_path = "/sys/devices/system/edac/mc"
    if !isdir(edac_path)
        # Try alternative: dmidecode for memory type
        return _check_ecc_via_dmidecode()
    end

    try
        total_errors = 0
        mc_count = 0

        for entry in readdir(edac_path)
            if startswith(entry, "mc")
                mc_count += 1
                mc_dir = joinpath(edac_path, entry)

                # Read correctable error count
                ce_path = joinpath(mc_dir, "ce_count")
                if isfile(ce_path)
                    ce = tryparse(Int, strip(read(ce_path, String)))
                    if ce !== nothing
                        total_errors += ce
                    end
                end

                # Read uncorrectable error count
                ue_path = joinpath(mc_dir, "ue_count")
                if isfile(ue_path)
                    ue = tryparse(Int, strip(read(ue_path, String)))
                    if ue !== nothing && ue > 0
                        total_errors += ue
                    end
                end
            end
        end

        if mc_count > 0
            return (present=true, error_count=total_errors,
                    details="EDAC: $(mc_count) memory controller(s), $(total_errors) correctable error(s)")
        end
    catch
    end

    return (present=false, error_count=0, details="EDAC subsystem not reporting memory controllers")
end

"""Check ECC via dmidecode as fallback (requires root)."""
function _check_ecc_via_dmidecode()
    try
        dmi = read(`dmidecode -t memory`, String)
        if occursin("Error Correcting Type: Single-bit ECC", dmi) ||
           occursin("Error Correcting Type: Multi-bit ECC", dmi)
            return (present=true, error_count=0, details="ECC detected via DMI/SMBIOS")
        end
    catch
        # dmidecode not available or no root access
    end
    return (present=false, error_count=0, details="Unable to detect ECC (no EDAC, dmidecode unavailable)")
end

# ============================================================================
# RAID detection
# ============================================================================

"""
    check_raid_status() -> NamedTuple{(:present, :status, :details), Tuple{Bool, Symbol, String}}

Detect RAID arrays via Linux `/proc/mdstat` (software RAID) and common
hardware RAID controller interfaces. Reports array health status.

Status values: `:optimal`, `:degraded`, `:rebuilding`, `:none`
"""
function check_raid_status()
    if !Sys.islinux()
        return (present=false, status=:none, details="RAID detection requires Linux")
    end

    # Check software RAID (mdadm)
    if isfile("/proc/mdstat")
        try
            mdstat = read("/proc/mdstat", String)
            lines = split(mdstat, '\n')

            arrays = String[]
            is_degraded = false
            is_rebuilding = false

            for line in lines
                # Lines like "md0 : active raid1 sda1[0] sdb1[1]"
                m = match(r"^(md\d+)\s*:\s*active\s+(\w+)", line)
                if m !== nothing
                    push!(arrays, "$(m[1]) ($(m[2]))")
                end

                # Check for degraded state: [UU] is good, [U_] or [_U] is degraded
                if occursin(r"\[U*_+U*\]", line)
                    is_degraded = true
                end

                # Check for rebuild
                if occursin("recovery", line) || occursin("resync", line)
                    is_rebuilding = true
                end
            end

            if !isempty(arrays)
                status = is_rebuilding ? :rebuilding : (is_degraded ? :degraded : :optimal)
                details = "Software RAID: $(join(arrays, ", ")) [$(status)]"
                return (present=true, status=status, details=details)
            end
        catch
        end
    end

    # Check hardware RAID controllers
    hw_raid = _check_hardware_raid()
    if hw_raid !== nothing
        return hw_raid
    end

    return (present=false, status=:none, details="No RAID arrays detected")
end

"""Check for hardware RAID controllers via sysfs and common utilities."""
function _check_hardware_raid()
    # MegaRAID (LSI/Broadcom)
    try
        if isfile("/usr/sbin/storcli64") || isfile("/opt/MegaRAID/storcli/storcli64")
            storcli = isfile("/usr/sbin/storcli64") ? "/usr/sbin/storcli64" : "/opt/MegaRAID/storcli/storcli64"
            output = read(`$storcli /call show`, String)
            if occursin("Optimal", output)
                return (present=true, status=:optimal, details="MegaRAID controller: Optimal")
            elseif occursin("Degraded", output)
                return (present=true, status=:degraded, details="MegaRAID controller: Degraded")
            end
        end
    catch
    end

    # Check for any RAID controllers in sysfs block devices
    try
        for dev in readdir("/sys/block")
            md_path = joinpath("/sys/block", dev, "md")
            if isdir(md_path)
                level_path = joinpath(md_path, "level")
                if isfile(level_path)
                    level = strip(read(level_path, String))
                    return (present=true, status=:optimal, details="Block device $dev: $level")
                end
            end
        end
    catch
    end

    return nothing
end

# ============================================================================
# Thermal monitoring
# ============================================================================

"""
    check_thermal_zones() -> Vector{ThermalZone}

Query the Linux thermal subsystem (`/sys/class/thermal/`) for all thermal
zones. Returns temperature readings, trip points, and critical status.
"""
function check_thermal_zones()
    zones = ThermalZone[]
    if !Sys.islinux()
        return zones
    end

    thermal_path = "/sys/class/thermal"
    if !isdir(thermal_path)
        return zones
    end

    try
        for entry in readdir(thermal_path)
            if !startswith(entry, "thermal_zone")
                continue
            end

            zone_path = joinpath(thermal_path, entry)

            # Read zone type
            type_path = joinpath(zone_path, "type")
            zone_type = isfile(type_path) ? strip(read(type_path, String)) : "unknown"

            # Read current temperature (in millidegrees Celsius)
            temp_path = joinpath(zone_path, "temp")
            temp_c = if isfile(temp_path)
                raw = tryparse(Int, strip(read(temp_path, String)))
                raw !== nothing ? raw / 1000.0 : 0.0
            else
                0.0
            end

            # Read critical trip point
            trip_path = joinpath(zone_path, "trip_point_0_temp")
            trip_c = if isfile(trip_path)
                raw = tryparse(Int, strip(read(trip_path, String)))
                raw !== nothing ? raw / 1000.0 : nothing
            else
                nothing
            end

            is_critical = trip_c !== nothing && temp_c >= trip_c * 0.9  # 90% of trip point

            push!(zones, ThermalZone(entry, zone_type, temp_c, trip_c, is_critical))
        end
    catch
    end

    return zones
end

# ============================================================================
# Watchdog timer detection
# ============================================================================

"""
    check_watchdog_timers() -> NamedTuple{(:present, :devices), Tuple{Bool, Vector{String}}}

Detect hardware watchdog timers via `/dev/watchdog*` and
`/sys/class/watchdog/`. Watchdog timers provide automatic system reset
if the system becomes unresponsive.
"""
function check_watchdog_timers()
    devices = String[]
    if !Sys.islinux()
        return (present=false, devices=devices)
    end

    # Check /dev/watchdog devices
    try
        for dev in readdir("/dev")
            if startswith(dev, "watchdog")
                push!(devices, "/dev/$dev")
            end
        end
    catch
    end

    # Check sysfs for watchdog info
    wdt_path = "/sys/class/watchdog"
    if isdir(wdt_path)
        try
            for entry in readdir(wdt_path)
                identity_path = joinpath(wdt_path, entry, "identity")
                if isfile(identity_path)
                    identity = strip(read(identity_path, String))
                    push!(devices, "$entry ($identity)")
                end
            end
        catch
        end
    end

    return (present=!isempty(devices), devices=unique(devices))
end

# ============================================================================
# Redundant power supply detection
# ============================================================================

"""
    check_redundant_power() -> NamedTuple{(:redundant, :psu_count, :details), Tuple{Bool, Int, String}}

Detect redundant power supplies via Linux power_supply sysfs interface and
IPMI (if available). Server-class systems typically have dual PSUs for
redundancy.
"""
function check_redundant_power()
    if !Sys.islinux()
        return (redundant=false, psu_count=0, details="Power supply detection requires Linux")
    end

    psu_count = 0
    psu_names = String[]

    # Check /sys/class/power_supply for mains/UPS power supplies
    ps_path = "/sys/class/power_supply"
    if isdir(ps_path)
        try
            for entry in readdir(ps_path)
                type_path = joinpath(ps_path, entry, "type")
                if isfile(type_path)
                    ps_type = strip(read(type_path, String))
                    if ps_type in ("Mains", "UPS")
                        psu_count += 1
                        # Check online status
                        online_path = joinpath(ps_path, entry, "online")
                        status = if isfile(online_path)
                            strip(read(online_path, String)) == "1" ? "online" : "offline"
                        else
                            "unknown"
                        end
                        push!(psu_names, "$entry ($status)")
                    end
                end
            end
        catch
        end
    end

    # Try IPMI for server-class PSU info
    if psu_count == 0
        try
            ipmi_out = read(`ipmitool sdr type "Power Supply"`, String)
            for line in split(ipmi_out, '\n')
                if occursin("Power Supply", line) && !occursin("not present", lowercase(line))
                    psu_count += 1
                    push!(psu_names, strip(line))
                end
            end
        catch
            # ipmitool not available
        end
    end

    # Try DMI/SMBIOS for PSU count
    if psu_count == 0
        try
            dmi = read(`dmidecode -t 39`, String)  # Type 39 = System Power Supply
            psu_count = count("System Power Supply", dmi)
        catch
        end
    end

    redundant = psu_count >= 2
    details = if psu_count == 0
        "No power supply information available"
    elseif redundant
        "Redundant: $(psu_count) PSUs detected [$(join(psu_names, ", "))]"
    else
        "Single PSU: $(join(psu_names, ", "))"
    end

    return (redundant=redundant, psu_count=psu_count, details=details)
end

# ============================================================================
# Comprehensive resilience report
# ============================================================================

"""
    detect_resilience() -> ResilienceReport

Run all hardware resilience checks and produce a comprehensive report.
Assesses overall resilience as `:high` (ECC + RAID + redundant power),
`:medium` (some protections), or `:low` (minimal protections).
"""
function detect_resilience()
    findings = String[]

    # ECC memory
    ecc = check_ecc_memory()
    if ecc.present
        push!(findings, "ECC memory detected: $(ecc.details)")
        if ecc.error_count > 0
            push!(findings, "WARNING: $(ecc.error_count) correctable memory error(s) logged")
        end
    else
        push!(findings, "No ECC memory detected")
    end

    # RAID
    raid = check_raid_status()
    if raid.present
        push!(findings, "RAID detected: $(raid.details)")
        if raid.status == :degraded
            push!(findings, "WARNING: RAID array is DEGRADED - data redundancy compromised")
        end
    else
        push!(findings, "No RAID arrays detected")
    end

    # Thermal
    thermal_zones = check_thermal_zones()
    thermal_ok = true
    for zone in thermal_zones
        if zone.is_critical
            thermal_ok = false
            push!(findings, "CRITICAL: $(zone.name) ($(zone.type)) at $(zone.temp_celsius)C - " *
                  "approaching trip point")
        end
    end
    if !isempty(thermal_zones) && thermal_ok
        max_temp = maximum(z.temp_celsius for z in thermal_zones)
        push!(findings, "Thermal: $(length(thermal_zones)) zone(s), max $(round(max_temp; digits=1))C - OK")
    end

    # Watchdog
    watchdog = check_watchdog_timers()
    if watchdog.present
        push!(findings, "Watchdog timer(s): $(join(watchdog.devices, ", "))")
    else
        push!(findings, "No hardware watchdog detected")
    end

    # Power
    power = check_redundant_power()
    if power.redundant
        push!(findings, "Redundant power: $(power.details)")
    else
        push!(findings, "No redundant power supply: $(power.details)")
    end

    # Calculate overall resilience score
    score = 0
    ecc.present && (score += 2)
    raid.present && raid.status == :optimal && (score += 2)
    raid.present && raid.status != :optimal && (score += 1)
    thermal_ok && (score += 1)
    watchdog.present && (score += 1)
    power.redundant && (score += 2)

    overall = score >= 6 ? :high : (score >= 3 ? :medium : :low)
    push!(findings, "Overall resilience: $(overall) (score: $(score)/8)")

    return ResilienceReport(
        now(),
        ecc.present, ecc.error_count,
        raid.present, raid.status, raid.details,
        thermal_zones, thermal_ok,
        watchdog.present, watchdog.devices,
        power.redundant, power.psu_count,
        overall, findings
    )
end

"""
    print_resilience_report(report::ResilienceReport)

Print a formatted hardware resilience report to stdout.
"""
function print_resilience_report(report::ResilienceReport)
    println("Hardware Resilience Report")
    println("=" ^ 50)
    println("Timestamp: $(report.timestamp)")
    println()
    println("ECC Memory:      $(report.ecc_memory ? "Present" : "Not detected")" *
            (report.ecc_error_count > 0 ? " ($(report.ecc_error_count) errors)" : ""))
    println("RAID:            $(report.raid_present ? "$(report.raid_status)" : "None")" *
            (report.raid_present ? " - $(report.raid_details)" : ""))
    println("Thermal:         $(report.thermal_ok ? "OK" : "WARNING")" *
            " ($(length(report.thermal_zones)) zone(s))")
    println("Watchdog:        $(report.watchdog_present ? join(report.watchdog_devices, ", ") : "None")")
    println("Redundant Power: $(report.redundant_power ? "Yes ($(report.psu_count) PSUs)" : "No")")
    println()
    println("Overall: $(uppercase(string(report.overall_resilience)))")
    println()
    println("Findings:")
    for f in report.findings
        println("  - $f")
    end
end

end # module
