# SPDX-License-Identifier: PMPL-1.0-or-later
module SystemFirmware

export read_uefi_var, parse_acpi_table, check_raid_status

"""
    read_uefi_var(name)
Reads a UEFI variable from the Linux sysfs interface (requires root).
"""
function read_uefi_var(name::String)
    path = "/sys/firmware/efi/efivars/$name"
    if isfile(path)
        return read(path)
    else
        return nothing
    end
end

"""
    parse_acpi_table(signature)
Reads an ACPI table (like 'DSDT' or 'FACP') from sysfs.
"""
function parse_acpi_table(sig::String)
    path = "/sys/firmware/acpi/tables/$sig"
    if isfile(path)
        return read(path)
    else
        return "ACPI Table $sig not found"
    end
end

"""
    check_raid_status()
Parses /proc/mdstat to check the health of software RAID arrays.
"""
function check_raid_status()
    if isfile("/proc/mdstat")
        return read("/proc/mdstat", String)
    else
        return "No software RAID detected."
    end
end

end # module
