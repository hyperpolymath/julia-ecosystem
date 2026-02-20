K9!
# SPDX-License-Identifier: PMPL-1.0-or-later
# K9 Kennel-level template: Pure data configuration
# Security Level: Kennel (data-only, no execution)
# No signature required - safe for any use

{
  pedigree = {
    schema_version = "1.0.0",
    component_type = "package-metadata",
    security = {
      leash = 'Kennel,
      trust_level = "data-only",
      allow_network = false,
      allow_filesystem_write = false,
      allow_subprocess = false,
    },
    metadata = {
      name = "cliodynamics-metadata",
      version = "1.0.0",
      description = "Cliodynamics.jl package metadata and configuration data",
      author = "Jonathan D.A. Jewell <jonathan.jewell@open.ac.uk>",
    },
  },

  # Your configuration data here
  config = {
    # Example: Pure data values
    setting_1 = "value",
    setting_2 = 42,
    setting_3 = true,

    nested = {
      key = "value",
    },

    list = [
      "item1",
      "item2",
    ],
  },

  # Optional: Export format specification
  export = {
    format = "json",  # or "yaml", "toml"
    destination = "output.json",
  },
}

# Usage:
# 1. Fill in FILL: placeholders above
# 2. Add your configuration data to config = { ... }
# 3. Validate: nickel typecheck your-file.k9.ncl
# 4. Export: nickel export your-file.k9.ncl > output.json
