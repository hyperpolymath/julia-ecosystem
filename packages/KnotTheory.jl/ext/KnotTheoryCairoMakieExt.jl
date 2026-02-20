# SPDX-License-Identifier: PMPL-1.0-or-later
module KnotTheoryCairoMakieExt

using CairoMakie
import KnotTheory: PlanarDiagram, plot_pd

function plot_pd(pd::PlanarDiagram)
    n = length(pd.crossings)
    fig = Figure()
    ax = Axis(fig[1, 1])

    for (i, crossing) in enumerate(pd.crossings)
        angle = 2pi * (i - 1) / max(n, 1)
        x = cos(angle)
        y = sin(angle)
        text!(ax, string(i), position=(x, y))
        scatter!(ax, [x], [y])
    end

    fig
end

end
