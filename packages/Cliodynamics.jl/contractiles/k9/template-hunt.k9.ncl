K9!
# SPDX-License-Identifier: PMPL-1.0-or-later
# K9 Hunt-level template: Full execution with Just recipes
# Security Level: Hunt (full system access)
# ⚠️ SIGNATURE REQUIRED - Review carefully before use

{
  pedigree = {
    schema_version = "1.0.0",
    component_type = "simulation-runner",
    security = {
      leash = 'Hunt,
      trust_level = "full-system-access",
      allow_network = true,
      allow_filesystem_write = true,
      allow_subprocess = true,
      signature_required = true,
    },
    metadata = {
      name = "cliodynamics-runner",
      version = "1.0.0",
      description = "Build, test, and run Cliodynamics.jl simulation models",
      author = "Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>",
    },
    warnings = [
      "This component has full system access",
      "Only run from trusted sources with verified signatures",
      "Review all Just recipes before execution",
      "Use dry-run mode first: ./must --dry-run run your-file.k9.ncl",
    ],
    side_effects = [
      "Creates Manifest.toml in project directory via Pkg.instantiate()",
      "Executes Julia test suite via Pkg.test()",
      "No network access required after initial dependency installation",
    ],
  },

  # Configuration with contracts (Yard-level validation)
  config = {
    # Add your configuration here with appropriate contracts
    target_dir
      | String
      | std.string.NonEmpty
      = "/tmp/k9-output",

    dry_run | Bool = false,

    # Add more config as needed
  },

  # Just recipes for execution
  # These run when: ./must run your-file.k9.ncl
  recipes = {
    # Main entry point (runs by default)
    default = {
      recipe = "test",
      description = "Run Cliodynamics.jl test suite",
    },

    "test" = {
      dependencies = ["check-prerequisites"],
      commands = [
        "julia --project=. -e 'using Pkg; Pkg.test()'",
      ],
    },

    "check-prerequisites" = {
      description = "Verify required tools and permissions",
      commands = [
        # Example: Check for required tools
        # "command -v git || (echo 'ERROR: git not found' && exit 1)",
        # Example: Check permissions
        # "[ -w %{config.target_dir} ] || (echo 'ERROR: Cannot write to target directory' && exit 1)",
        "echo '✓ Prerequisites checked'",
      ],
    },

    # Add more recipes as needed
    "build" = {
      description = "Install Julia dependencies",
      commands = [
        "julia --project=. -e 'using Pkg; Pkg.instantiate()'",
      ],
    },

    "examples" = {
      description = "Run example scripts",
      dependencies = ["build"],
      commands = [
        "julia --project=. examples/basic_usage.jl",
        "julia --project=. examples/historical_analysis.jl",
      ],
    },

    "clean" = {
      description = "Clean up generated files",
      commands = [
        "echo '⚠️  This will delete Manifest.toml - waiting 3 seconds...'",
        "sleep 3",
        "rm -f Manifest.toml",
      ],
    },
  },

  # Validation (Yard-level checks before Hunt execution)
  validation = {
    check_target_dir = std.string.length config.target_dir > 0,
    # Add more validation as needed
  },
}

# Usage:
# 1. Fill in FILL: placeholders above
# 2. Define configuration with contracts
# 3. Implement Just recipes with your commands
# 4. Test with dry-run: ./must --dry-run run your-file.k9.ncl
# 5. Review dry-run output carefully
# 6. Sign the component: ./must sign your-file.k9.ncl
# 7. Distribute with signature: your-file.k9.ncl.sig
# 8. Users verify and run: ./must verify && ./must run your-file.k9.ncl
#
# Security checklist:
# ✓ All FILL: placeholders filled in
# ✓ side_effects documented accurately
# ✓ Commands reviewed for safety
# ✓ No hardcoded secrets or credentials
# ✓ Proper error handling in recipes
# ✓ Tested in dry-run mode
# ✓ Component signed with trusted key
