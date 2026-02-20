# SPDX-License-Identifier: PMPL-1.0-or-later
module SchoolTools

using XLSX
using DataFrames
using Luxor

export save_spreadsheet, read_spreadsheet, make_report, make_slides

"""
    save_spreadsheet(filename, data::DataFrame)

Saves your data into an Excel-style spreadsheet (.xlsx) so you can open it in Excel or Google Sheets!
"""
function save_spreadsheet(filename, data::DataFrame)
    if !endswith(filename, ".xlsx")
        filename *= ".xlsx"
    end
    XLSX.writetable(filename, data)
    return "Saved your data to $filename 📊"
end

"""
    read_spreadsheet(filename) -> DataFrame

Reads a spreadsheet from Excel so you can use the data in Julia.
"""
function read_spreadsheet(filename)
    xf = XLSX.readxlsx(filename)
    sh = xf[1] # Read the first sheet
    return DataFrame(XLSX.get_table(sh))
end

"""
    make_report(filename, title, contents)

Creates a simple report file (.md) that you can open in Word or any text editor.
"""
function make_report(filename, title, contents)
    if !endswith(filename, ".md")
        filename *= ".md"
    end
    open(filename, "w") do io
        println(io, "# $title")
        println(io, "
---")
        println(io, "
$contents")
        println(io, "

*Created with JuliaForChildren.jl* 🚀")
    end
    return "Report created: $filename 📝"
end

"""
    make_slides(filename, slide_titles::Vector{String})

Creates a simple set of slides (as a PDF) that you can show to your class!
"""
function make_slides(filename, slide_titles::Vector{String})
    if !endswith(filename, ".pdf")
        filename *= ".pdf"
    end
    
    # We use Luxor to draw slides into a PDF
    Drawing(800, 600, filename)
    
    for title in slide_titles
        background("white")
        setcolor("darkblue")
        fontsize(40)
        text(title, Point(400, 300), halign=:center, valign=:middle)
        
        # Add a little footer
        fontsize(12)
        setcolor("gray")
        text("Made with Julia", Point(750, 580), halign=:right)
        
        newpage()
    end
    
    finish()
    return "Slides saved to $filename 📽️"
end

end # module
