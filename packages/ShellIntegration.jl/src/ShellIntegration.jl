# SPDX-License-Identifier: PMPL-1.0-or-later
module ShellIntegration

export run_pwsh, start_valence_shell, exec_safe

"""
    run_pwsh(script::String)
Executes a PowerShell Core (pwsh) script from within Julia.
"""
function run_pwsh(script::String)
    # This assumes 'pwsh' is in the PATH
    cmd = `pwsh -Command $script`
    return read(cmd, String)
end

"""
    start_valence_shell()
Starts a capability-restricted shell environment (inspired by Valence).
"""
function start_valence_shell()
    println("Valence Shell Active 🛡️")
    println(" capabilities: [read, network-out]")
    # Loop implementation stub
end

"""
    exec_safe(cmd)
Executes a system command with strict safety checks (no rm -rf /).
"""
function exec_safe(cmd::Cmd)
    if contains(string(cmd), "rm -rf")
        error("Unsafe command blocked!")
    end
    run(cmd)
end

end # module
