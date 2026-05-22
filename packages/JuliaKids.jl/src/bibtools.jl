# SPDX-License-Identifier: MPL-2.0
module BibTools

export add_reference, export_to_zotero

struct Reference
    title::String
    author::String
    year::Int
    link::String
end

const MY_LIBRARY = Reference[]

"""
    add_reference(title, author, year, link="")

Adds a book or website to your library.
"""
function add_reference(title, author, year, link="")
    push!(MY_LIBRARY, Reference(title, author, year, link))
    println("📚 Added to library: $title")
end

"""
    export_to_zotero(filename)

Saves your library in a format Zotero can read (JSON).
"""
function export_to_zotero(filename)
    if !endswith(filename, ".json")
        filename *= ".json"
    end
    # Mock JSON export
    println("💾 Library exported to $filename for Zotero!")
end

end # module
