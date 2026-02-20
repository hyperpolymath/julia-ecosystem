# SPDX-License-Identifier: PMPL-1.0-or-later
"""
Causal directed acyclic graphs (DAGs) and structural causal models.

Provides d-separation tests, backdoor/frontdoor criteria, and do-calculus
for identifying causal effects from observational data.
"""
module CausalDAG

using Graphs
using LinearAlgebra

export CausalGraph, add_edge!, remove_edge!
export d_separation, ancestors, descendants
export backdoor_criterion, frontdoor_criterion
export markov_blanket

"""
    CausalGraph

Directed acyclic graph representing causal relationships.
"""
struct CausalGraph
    graph::SimpleDiGraph
    names::Vector{Symbol}
    name_to_index::Dict{Symbol, Int}

    function CausalGraph(names::Vector{Symbol})
        n = length(names)
        graph = SimpleDiGraph(n)
        name_to_index = Dict(name => i for (i, name) in enumerate(names))
        new(graph, names, name_to_index)
    end
end

"""
    add_edge!(g, from, to)

Add a directed edge from → to (causal arrow).
"""
function add_edge!(g::CausalGraph, from::Symbol, to::Symbol)
    i = g.name_to_index[from]
    j = g.name_to_index[to]
    Graphs.add_edge!(g.graph, i, j) || error("Cannot add edge (creates cycle?)")
    nothing
end

function remove_edge!(g::CausalGraph, from::Symbol, to::Symbol)
    i = g.name_to_index[from]
    j = g.name_to_index[to]
    rem_edge!(g.graph, i, j)
    nothing
end

"""
    d_separation(g, X, Y, Z)

Test if X and Y are d-separated given Z (conditional independence).
"""
function d_separation(g::CausalGraph, X::Set{Symbol}, Y::Set{Symbol}, Z::Set{Symbol})
    # Implement proper d-separation using moralization + ancestral graph approach
    # Algorithm: Pearl (1988), Geiger et al. (1990)

    # Edge cases
    if isempty(X) || isempty(Y)
        return true
    end
    if !isempty(X ∩ Y)  # X and Y overlap
        return true
    end

    # Convert to indices
    xi = Set(g.name_to_index[x] for x in X)
    yi = Set(g.name_to_index[y] for y in Y)
    zi = Set(g.name_to_index[z] for z in Z)

    # Step 1: Build ancestral graph (ancestors of X ∪ Y ∪ Z)
    relevant = xi ∪ yi ∪ zi
    ancestral_nodes = copy(relevant)
    for node_idx in relevant
        node_name = g.names[node_idx]
        for anc_name in ancestors(g, node_name)
            push!(ancestral_nodes, g.name_to_index[anc_name])
        end
    end

    # Step 2: Moralize the ancestral graph
    # Create undirected graph with edges from original + moral edges
    moral_edges = Set{Tuple{Int,Int}}()

    # Add original edges (as undirected)
    for i in ancestral_nodes
        for j in outneighbors(g.graph, i)
            if j in ancestral_nodes
                push!(moral_edges, minmax(i, j))
            end
        end
    end

    # Add moral edges: connect parents of common children
    for child in ancestral_nodes
        parents = [p for p in inneighbors(g.graph, child) if p in ancestral_nodes]
        for i in 1:length(parents)
            for j in (i+1):length(parents)
                push!(moral_edges, minmax(parents[i], parents[j]))
            end
        end
    end

    # Step 3: Remove Z nodes (and their edges)
    active_nodes = setdiff(ancestral_nodes, zi)
    active_edges = filter(e -> e[1] ∉ zi && e[2] ∉ zi, moral_edges)

    # Step 4: Check if X and Y are connected in the resulting graph (BFS)
    visited = Set{Int}()
    queue = collect(xi)

    while !isempty(queue)
        current = popfirst!(queue)
        if current in yi
            return false  # Found path from X to Y - NOT d-separated
        end

        if current in visited
            continue
        end
        push!(visited, current)

        # Add neighbors
        for (a, b) in active_edges
            if a == current && b ∉ visited && b in active_nodes
                push!(queue, b)
            elseif b == current && a ∉ visited && a in active_nodes
                push!(queue, a)
            end
        end
    end

    return true  # No path found - X and Y are d-separated given Z
end

"""
    ancestors(g, node)

Find all ancestors of a node.
"""
function ancestors(g::CausalGraph, node::Symbol)
    i = g.name_to_index[node]
    anc = Set{Int}()

    function visit(j)
        for pred in inneighbors(g.graph, j)
            if !(pred in anc)
                push!(anc, pred)
                visit(pred)
            end
        end
    end

    visit(i)
    Set(g.names[j] for j in anc)
end

"""
    descendants(g, node)

Find all descendants of a node.
"""
function descendants(g::CausalGraph, node::Symbol)
    i = g.name_to_index[node]
    desc = Set{Int}()

    function visit(j)
        for succ in outneighbors(g.graph, j)
            if !(succ in desc)
                push!(desc, succ)
                visit(succ)
            end
        end
    end

    visit(i)
    Set(g.names[j] for j in desc)
end

"""
    backdoor_criterion(g, X, Y, Z)

Check if Z satisfies the backdoor criterion for estimating effect of X on Y.

Backdoor criterion: Z blocks all backdoor paths from X to Y AND
                     Z contains no descendants of X.

A backdoor path is a path from X to Y that starts with an edge into X (X ← ...).
"""
function backdoor_criterion(g::CausalGraph, X::Symbol, Y::Symbol, Z::Set{Symbol})
    # Check no descendants of X in Z
    desc_X = descendants(g, X)
    if !isempty(intersect(desc_X, Z))
        return false
    end

    # Check Z blocks all backdoor paths from X to Y
    # A backdoor path starts with X ← ... (parent of X)
    x_idx = g.name_to_index[X]
    y_idx = g.name_to_index[Y]
    z_indices = Set(g.name_to_index[z] for z in Z)

    # Get parents of X (start of backdoor paths)
    parents_x = inneighbors(g.graph, x_idx)

    # For each parent of X, check if there's an unblocked path to Y
    for parent in parents_x
        if has_unblocked_path(g, parent, y_idx, z_indices, Set{Int}([x_idx]))
            return false  # Found unblocked backdoor path
        end
    end

    true  # All backdoor paths blocked
end

"""
    has_unblocked_path(g, from, to, blocked, visited)

Check if there's an unblocked path from `from` to `to` avoiding `blocked` nodes.
Uses DFS to explore paths, treating blocked nodes as barriers.
"""
function has_unblocked_path(g::CausalGraph, from::Int, to::Int, blocked::Set{Int}, visited::Set{Int})
    if from == to
        return true
    end

    if from in visited || from in blocked
        return false
    end

    push!(visited, from)

    # Explore all neighbors (both parents and children for undirected path search)
    for neighbor in union(inneighbors(g.graph, from), outneighbors(g.graph, from))
        if has_unblocked_path(g, neighbor, to, blocked, copy(visited))
            return true
        end
    end

    false
end

"""
    frontdoor_criterion(g, X, Y, M)

Check if M satisfies the frontdoor criterion for estimating effect of X on Y.

Frontdoor criterion: M intercepts all directed paths from X to Y,
                     no backdoor paths from X to M,
                     X blocks all backdoor paths from M to Y.
"""
function frontdoor_criterion(g::CausalGraph, X::Symbol, Y::Symbol, M::Set{Symbol})
    # Implement proper frontdoor criterion (Pearl 1995, 2009)
    # Condition 1: M intercepts all directed paths from X to Y
    # Condition 2: No unblocked backdoor paths from X to M
    # Condition 3: X blocks all backdoor paths from M to Y

    xi = g.name_to_index[X]
    yi = g.name_to_index[Y]
    mi = Set(g.name_to_index[m] for m in M)

    # Condition 1: Every directed path from X to Y must pass through some node in M
    # Use DFS to find all directed paths from X to Y
    function all_paths_through_M()
        paths_found = false
        all_intercepted = true

        function dfs_paths(current, visited, path)
            if current == yi
                paths_found = true
                # Check if this path goes through any M node
                if isempty(intersect(mi, Set(path)))
                    all_intercepted = false
                end
                return
            end

            for neighbor in outneighbors(g.graph, current)
                if !(neighbor in visited)
                    push!(visited, neighbor)
                    push!(path, neighbor)
                    dfs_paths(neighbor, visited, path)
                    pop!(path)
                    pop!(visited, neighbor)
                end
            end
        end

        dfs_paths(xi, Set([xi]), [xi])

        # If no paths exist from X to Y, criterion trivially holds
        if !paths_found
            return true
        end

        return all_intercepted
    end

    if !all_paths_through_M()
        return false
    end

    # Condition 2: No unblocked backdoor paths from X to M
    # This means: X and each m in M have no common causes (except X itself if X causes m)
    # Simpler check: for each m in M, m should have no parents other than X that are also ancestors of X
    for m in M
        mi_idx = g.name_to_index[m]
        m_parents = Set(inneighbors(g.graph, mi_idx))

        # Get ancestors of X
        x_ancestors = Set(g.name_to_index[a] for a in ancestors(g, X))

        # Check if any parent of m (other than X itself) is an ancestor of X
        # This would create a backdoor path X ← ancestor → m
        for m_parent in m_parents
            if m_parent != xi && m_parent in x_ancestors
                return false
            end
        end
    end

    # Condition 3: X blocks all backdoor paths from M to Y
    # A backdoor path from m to Y is a path that starts m ← ...
    # We need to check if X d-separates m from Y when considering only non-causal paths
    # Equivalently: check backdoor_criterion(g, m, Y, {X}) for each m
    # But backdoor_criterion has the no-descendants check which doesn't apply here
    # Instead: check if removing X creates a dependency between m and Y via backdoor paths

    for m in M
        mi_idx = g.name_to_index[m]

        # Find backdoor paths from m to Y (paths starting with m ← ...)
        m_parents = collect(inneighbors(g.graph, mi_idx))

        if !isempty(m_parents)
            # Check if any backdoor path from m to Y is NOT blocked by X
            # A backdoor path is blocked by X if it passes through X

            for parent in m_parents
                # BFS to find paths from this parent to Y, checking if they avoid X
                visited = Set{Int}([mi_idx])  # Don't go back through m
                xi_set = Set([xi])
                queue = [(parent, false)]  # (node, passed_through_X)

                while !isempty(queue)
                    (current, through_X) = popfirst!(queue)

                    if current == yi
                        # Found a path from m parent to Y
                        if !through_X
                            # Path doesn't go through X - backdoor not blocked!
                            return false
                        end
                        continue
                    end

                    if current in visited
                        continue
                    end
                    push!(visited, current)

                    new_through_X = through_X || (current in xi_set)

                    # Explore both directions (backdoor paths can go any direction)
                    for neighbor in union(inneighbors(g.graph, current), outneighbors(g.graph, current))
                        if !(neighbor in visited)
                            push!(queue, (neighbor, new_through_X))
                        end
                    end
                end
            end
        end
    end

    return true
end

"""
    markov_blanket(g, node)

Find the Markov blanket: parents, children, and children's parents.
"""
function markov_blanket(g::CausalGraph, node::Symbol)
    i = g.name_to_index[node]
    blanket = Set{Symbol}()

    # Parents
    for pred in inneighbors(g.graph, i)
        push!(blanket, g.names[pred])
    end

    # Children
    children_indices = outneighbors(g.graph, i)
    for child in children_indices
        push!(blanket, g.names[child])

        # Children's parents (co-parents)
        for copred in inneighbors(g.graph, child)
            if copred != i
                push!(blanket, g.names[copred])
            end
        end
    end

    blanket
end

end # module CausalDAG
