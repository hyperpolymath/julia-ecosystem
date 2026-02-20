# SPDX-License-Identifier: PMPL-1.0-or-later
module TimeTools

using Dates

export set_alarm, calendar_event

"""
    set_alarm(time_str, message)

Sets an alarm! Format: "HH:MM".
Example: set_alarm("15:30", "Time for coding class!")
"""
function set_alarm(time_str, message)
    println("⏰ Alarm set for $time_str: $message")
    # In a real app, this would start a background timer
end

"""
    calendar_event(title, date_str)

Adds an event to your schedule.
"""
function calendar_event(title, date_str)
    println("📅 Event added: '$title' on $date_str")
end

end # module
