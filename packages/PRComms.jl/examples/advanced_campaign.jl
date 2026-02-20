using PRComms
using Dates

# 1. SURVEY DEVELOPER: Design a campaign feedback survey
println("--- Designing Survey ---")
survey = SurveyTemplate(:launch_feedback, "Product X Launch Feedback")
add_question!(survey, "How likely are you to recommend us?", :nps)
add_question!(survey, "What was the main message you remembered?", :text)
add_question!(survey, "Rate the tone of our announcement:", :likert, options=["Too formal", "Just right", "Too bold"])

# Export for use in a web form
json_spec = build_survey_json(survey)
println("Survey JSON Ready! 📋")

# 2. ANALYTICS: Multi-Order Ratio Analysis
println("
--- Performance Analysis ---")

# First Order: Conversion Rate (Clicks / Impressions)
conv_rate = first_order_ratio(500, 10000) # 5%
println("1st Order (Conversion): $(conv_rate * 100)%")

# Second Order: Efficiency (Conversion / Cost)
# How many conversions did we get per $1000 spent?
efficiency = second_order_ratio(conv_rate, 1000.0)
println("2nd Order (Efficiency): $efficiency conversions per dollar")

# Third Order: Velocity (Acceleration of Efficiency)
# Did our efficiency improve compared to last week?
prev_efficiency = 0.00004
velocity = third_order_ratio(efficiency, prev_efficiency, 7) # over 7 days
println("3rd Order (Velocity): $velocity efficiency growth per day")

# 3. STRATEGY: Interop with Finance
println("
--- Brand Valuation for Finance ---")
valuation = brand_equity_valuation(5_000_000.0, 0.75) # $5M revenue, 0.75 strength
println("Estimated Brand Value: \$$(valuation.value)")
println("$(valuation.explanation)")

println("
Campaign intelligence is complete! 🚀📊")
