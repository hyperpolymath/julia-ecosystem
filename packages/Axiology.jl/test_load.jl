# Test loading fairness.jl
using Statistics

println("About to include fairness.jl...")
try
    include("src/fairness.jl")
    println("SUCCESS: fairness.jl loaded")
catch e
    println("ERROR loading fairness.jl:")
    showerror(stdout, e, catch_backtrace())
    println()

    # Try to get more details
    println("\nChecking line 30:")
    lines = readlines("src/fairness.jl")
    for i in 28:32
        println("Line $i: ", repr(lines[i]))
    end
end
