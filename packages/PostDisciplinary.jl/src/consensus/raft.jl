# SPDX-License-Identifier: PMPL-1.0-or-later
module RaftConsensus

using UUIDs

export RaftNode, RaftState, request_vote, append_entries

@enum RaftState Follower Candidate Leader

mutable struct RaftNode
    id::UUID
    state::RaftState
    current_term::Int
    voted_for::Union{UUID, Nothing}
    log::Vector{String} # Log of agreed-upon research facts
end

function RaftNode()
    return RaftNode(uuid4(), Follower, 0, nothing, String[])
end

"""
    request_vote(node, term, candidate_id)
Simulation of a Raft RequestVote RPC.
"""
function request_vote(node::RaftNode, term::Int, candidate_id::UUID)
    if term > node.current_term
        node.current_term = term
        node.state = Follower
        node.voted_for = candidate_id
        return true
    end
    return false
end

"""
    append_entries(node, term, leader_id, entries)
Simulation of a Raft AppendEntries RPC (Heartbeat & Data).
"""
function append_entries(node::RaftNode, term::Int, leader_id::UUID, entries::Vector{String})
    if term >= node.current_term
        node.current_term = term
        node.state = Follower
        append!(node.log, entries)
        return true
    end
    return false
end

end # module
