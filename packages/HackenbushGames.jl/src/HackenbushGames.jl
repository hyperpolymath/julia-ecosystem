# SPDX-License-Identifier: PMPL-1.0-or-later
module HackenbushGames

export EdgeColor, Edge, HackenbushGraph
export Blue, Red, Green
export prune_disconnected, cut_edge, moves, game_sum
export simplest_dyadic_between, stalk_value
export mex, nim_sum, green_stalk_nimber, green_grundy
export simple_stalk, to_graphviz, to_ascii
export GameForm, canonical_game, simplify_game, game_value

@enum EdgeColor Blue Red Green

"""
Represents a colored edge between two nodes.
"""
struct Edge
    u::Int
    v::Int
    color::EdgeColor
end

"""
Represents a Hackenbush position as an edge list and ground nodes.
"""
struct HackenbushGraph
    edges::Vector{Edge}
    ground::Vector{Int}
end

"""
Canonical game form {L|R} for numeric options.
"""
struct GameForm
    left::Vector{Rational{Int}}
    right::Vector{Rational{Int}}
end

function _neighbors(edges::Vector{Edge})
    neighbors = Dict{Int, Vector{Int}}()
    for e in edges
        push!(get!(neighbors, e.u, Int[]), e.v)
        push!(get!(neighbors, e.v, Int[]), e.u)
    end
    neighbors
end

"""
Remove edges disconnected from any ground node.
"""
function prune_disconnected(graph::HackenbushGraph)
    neighbors = _neighbors(graph.edges)
    reachable = Set{Int}()
    stack = collect(graph.ground)
    while !isempty(stack)
        node = pop!(stack)
        if node in reachable
            continue
        end
        push!(reachable, node)
        for n in get(neighbors, node, Int[])
            if !(n in reachable)
                push!(stack, n)
            end
        end
    end

    edges = Edge[]
    for e in graph.edges
        if (e.u in reachable) && (e.v in reachable)
            push!(edges, e)
        end
    end
    HackenbushGraph(edges, graph.ground)
end

"""
Cut a specific edge and prune disconnected components.
"""
function cut_edge(graph::HackenbushGraph, index::Int)
    kept = Edge[]
    for (i, e) in enumerate(graph.edges)
        if i != index
            push!(kept, e)
        end
    end
    prune_disconnected(HackenbushGraph(kept, graph.ground))
end

function _edge_allowed(edge::Edge, player::Symbol)
    if edge.color == Green
        return true
    end
    if player == :left
        return edge.color == Blue
    elseif player == :right
        return edge.color == Red
    else
        error("player must be :left or :right")
    end
end

"""
Generate all legal moves for a given player (:left or :right).
"""
function moves(graph::HackenbushGraph, player::Symbol)
    options = HackenbushGraph[]
    for (i, e) in enumerate(graph.edges)
        if _edge_allowed(e, player)
            push!(options, cut_edge(graph, i))
        end
    end
    options
end

"""
Sum of two graphs as a disjoint union.
"""
function game_sum(a::HackenbushGraph, b::HackenbushGraph)
    max_node = isempty(a.edges) ? (isempty(a.ground) ? 0 : maximum(a.ground)) : maximum([max(e.u, e.v) for e in a.edges])
    offset = max_node + 1
    shifted = Edge[]
    for e in b.edges
        push!(shifted, Edge(e.u + offset, e.v + offset, e.color))
    end
    ground = vcat(a.ground, [g + offset for g in b.ground])
    HackenbushGraph(vcat(a.edges, shifted), ground)
end

"""
Return the simplest dyadic rational strictly between l and r.
"""
function simplest_dyadic_between(l::Rational{Int}, r::Rational{Int}; max_pow::Int=30)
    if !(l < r)
        error("l must be less than r")
    end
    for k in 0:max_pow
        denom = 1 << k
        n = fld(l * denom, 1) + 1
        candidate = n // denom
        if candidate < r
            return candidate
        end
    end
    return (l + r) // 2
end

"""
Compute the value of a Red-Blue stalk (ground -> top).
"""
function stalk_value(colors::Vector{EdgeColor})
    values = Vector{Rational{Int}}(undef, length(colors) + 1)
    values[1] = 0//1

    for i in 1:length(colors)
        left_opts = Rational{Int}[]
        right_opts = Rational{Int}[]
        for j in 1:i
            if colors[j] == Blue
                push!(left_opts, values[j])
            elseif colors[j] == Red
                push!(right_opts, values[j])
            end
        end

        if isempty(left_opts) && isempty(right_opts)
            values[i + 1] = 0//1
        elseif isempty(right_opts)
            values[i + 1] = maximum(left_opts) + 1
        elseif isempty(left_opts)
            values[i + 1] = minimum(right_opts) - 1
        else
            lmax = maximum(left_opts)
            rmin = minimum(right_opts)
            values[i + 1] = simplest_dyadic_between(lmax, rmin)
        end
    end

    values[end]
end

"""
Minimal excluded nonnegative integer.
"""
mex(values::Vector{Int}) = isempty(values) ? 0 : first(setdiff(0:maximum(values) + 1, values))

"""
Nim-sum (xor) of integer nimbers.
"""
function nim_sum(values::Vector{Int})
    result = 0
    for v in values
        result ⊻= v
    end
    result
end

"""
Nimber for a green stalk of given height.
"""
green_stalk_nimber(height::Int) = height

function _graph_key(graph::HackenbushGraph)
    edges = sort([(min(e.u, e.v), max(e.u, e.v), Int(e.color)) for e in graph.edges])
    (sort(graph.ground), edges)
end

"""
Compute the Grundy number for a small green Hackenbush graph.
"""
function green_grundy(graph::HackenbushGraph)
    for e in graph.edges
        if e.color != Green
            error("green_grundy expects only Green edges")
        end
    end

    memo = Dict{Any, Int}()

    function _grundy(g::HackenbushGraph)
        key = _graph_key(g)
        if haskey(memo, key)
            return memo[key]
        end
        if isempty(g.edges)
            memo[key] = 0
            return 0
        end

        options = Int[]
        for i in 1:length(g.edges)
            value = _grundy(cut_edge(g, i))
            push!(options, value)
        end
        gval = mex(options)
        memo[key] = gval
        gval
    end

    _grundy(graph)
end

"""
Create a simple stalk graph from a color sequence.
"""
function simple_stalk(colors::Vector{EdgeColor})
    edges = Edge[]
    for i in 1:length(colors)
        push!(edges, Edge(i - 1, i, colors[i]))
    end
    HackenbushGraph(edges, [0])
end

"""
Return a GraphViz DOT diagram for a position.
"""
function to_graphviz(graph::HackenbushGraph)
    lines = String[]
    push!(lines, "graph Hackenbush {")
    for g in graph.ground
        push!(lines, "  $g [shape=box,label=\"ground\"];\n")
    end
    for (i, e) in enumerate(graph.edges)
        color = e.color == Blue ? "blue" : e.color == Red ? "red" : "green"
        push!(lines, "  $(e.u) -- $(e.v) [color=$color,label=\"$i\"];\n")
    end
    push!(lines, "}")
    join(lines, "")
end

"""
Render a simple ASCII listing for a graph.
"""
function to_ascii(graph::HackenbushGraph)
    lines = ["HackenbushGraph:"]
    push!(lines, "Ground: $(graph.ground)")
    for (i, e) in enumerate(graph.edges)
        color = e.color == Blue ? "B" : e.color == Red ? "R" : "G"
        push!(lines, "  $i: $(e.u) - $(e.v) [$color]")
    end
    join(lines, "\n")
end

"""
Build the canonical game form {L|R} for small graphs.
"""
function canonical_game(graph::HackenbushGraph; max_depth::Int=6)
    function eval_game(g::HackenbushGraph, depth::Int)
        if isempty(g.edges) || depth <= 0
            return GameForm(Rational{Int}[], Rational{Int}[])
        end

        left_vals = Rational{Int}[]
        right_vals = Rational{Int}[]
        for option in moves(g, :left)
            val = game_value(option; max_depth=depth - 1)
            val === nothing || push!(left_vals, val)
        end
        for option in moves(g, :right)
            val = game_value(option; max_depth=depth - 1)
            val === nothing || push!(right_vals, val)
        end
        simplify_game(GameForm(left_vals, right_vals))
    end

    eval_game(graph, max_depth)
end

"""
Remove dominated options from a game form.
"""
function simplify_game(form::GameForm)
    left = unique(form.left)
    right = unique(form.right)

    # Left prefers higher values; remove dominated lower options.
    if !isempty(left)
        best = maximum(left)
        left = [v for v in left if v == best]
    end

    # Right prefers lower values; remove dominated higher options.
    if !isempty(right)
        best = minimum(right)
        right = [v for v in right if v == best]
    end

    GameForm(left, right)
end

"""
Compute a numeric value for small dyadic games.\nReturns nothing when non-numeric options are present.
"""
function game_value(graph::HackenbushGraph; max_depth::Int=6)
    form = canonical_game(graph; max_depth=max_depth)
    left = form.left
    right = form.right

    if isempty(left) && isempty(right)
        return 0//1
    elseif isempty(right)
        return maximum(left) + 1
    elseif isempty(left)
        return minimum(right) - 1
    end

    lmax = maximum(left)
    rmin = minimum(right)
    lmax < rmin || return nothing
    simplest_dyadic_between(lmax, rmin)
end

end # module
