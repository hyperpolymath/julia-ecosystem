# SPDX-License-Identifier: PMPL-1.0-or-later
module AssetTemplates

using Luxor
using Mustache

export make_email_signature, make_business_card, make_letterhead

"""
    make_email_signature(name, role, email, phone)
Generates an HTML email signature.
"""
function make_email_signature(name, role, email, phone)
    template = """
    <div style="font-family: Arial, sans-serif; color: #333;">
        <strong>{{name}}</strong><br>
        <span style="color: #666;">{{role}}</span><br>
        <br>
        Email: <a href="mailto:{{email}}">{{email}}</a><br>
        Phone: {{phone}}
    </div>
    """
    return render(template, name=name, role=role, email=email, phone=phone)
end

"""
    make_business_card(filename, name, role, company)
Generates a PDF business card.
"""
function make_business_card(filename, name, role, company)
    if !endswith(filename, ".pdf") filename *= ".pdf" end
    
    # 3.5 x 2 inch card @ 72 dpi = 252 x 144 points
    Drawing(252, 144, filename)
    background("white")
    origin()
    
    # Border
    setcolor("darkblue")
    setline(2)
    rect(Point(-126, -72), 252, 144, :stroke)
    
    # Text
    fontsize(14)
    text(name, Point(0, -20), halign=:center)
    
    fontsize(10)
    setcolor("gray")
    text(role, Point(0, 0), halign=:center)
    
    fontsize(12)
    setcolor("black")
    text(company, Point(0, 30), halign=:center)
    
    finish()
    return "Business card created: $filename 📇"
end

"""
    make_letterhead(filename, company, address)
Generates a simple A4 letterhead.
"""
function make_letterhead(filename, company, address)
    if !endswith(filename, ".pdf") filename *= ".pdf" end
    Drawing(595, 842, filename) # A4
    background("white")
    
    # Logo area
    setcolor("darkred")
    fontsize(30)
    text(company, Point(50, 80))
    
    # Footer
    setcolor("gray")
    fontsize(10)
    text(address, Point(297, 800), halign=:center)
    
    finish()
    return "Letterhead created: $filename ✉️"
end

end # module
