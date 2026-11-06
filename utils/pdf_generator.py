from fpdf import FPDF
import datetime

class HealthReportPDF(FPDF):
    def header(self):
        # Logo / Title
        self.set_font('helvetica', 'B', 24)
        self.set_text_color(30, 58, 95) # Dark Blue
        self.cell(0, 15, 'HealthGuard AI Assessment Report', border=False, align='C', new_x="LMARGIN", new_y="NEXT")
        
        self.set_font('helvetica', 'I', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 10, f'Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', border=False, align='C', new_x="LMARGIN", new_y="NEXT")
        self.ln(5)
        
        # Horizontal line
        self.set_draw_color(99, 179, 237) # Light blue line
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')
        
    def add_section_title(self, title):
        self.set_font('helvetica', 'B', 16)
        self.set_fill_color(240, 245, 255)
        self.set_text_color(30, 58, 95)
        self.cell(0, 10, title, border=False, align='L', fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)
        
    def add_text_body(self, text):
        self.set_font('helvetica', '', 11)
        self.set_text_color(50, 50, 50)
        
        # Simple markdown parsing for the PDF
        lines = text.split('\n')
        for line in lines:
            if line.startswith('### '):
                self.ln(4)
                self.set_font('helvetica', 'B', 14)
                self.set_text_color(40, 80, 140)
                self.multi_cell(0, 8, line.replace('### ', '').replace('**', ''), new_x="LMARGIN", new_y="NEXT")
                self.set_font('helvetica', '', 11)
                self.set_text_color(50, 50, 50)
            elif line.startswith('## '):
                self.ln(5)
                self.set_font('helvetica', 'B', 15)
                self.set_text_color(30, 58, 95)
                self.multi_cell(0, 9, line.replace('## ', '').replace('**', ''), new_x="LMARGIN", new_y="NEXT")
                self.set_font('helvetica', '', 11)
                self.set_text_color(50, 50, 50)
            elif line.startswith('# '):
                self.ln(6)
                self.set_font('helvetica', 'B', 16)
                self.set_text_color(20, 40, 70)
                self.multi_cell(0, 10, line.replace('# ', '').replace('**', ''), new_x="LMARGIN", new_y="NEXT")
                self.set_font('helvetica', '', 11)
                self.set_text_color(50, 50, 50)
            elif line.startswith('- '):
                # Bullet point
                clean_line = line.replace('**', '')
                self.multi_cell(0, 6, "  • " + clean_line[2:], new_x="LMARGIN", new_y="NEXT")
            else:
                # Regular text
                clean_line = line.replace('**', '')
                if clean_line.strip() == "":
                    self.ln(3)
                else:
                    self.multi_cell(0, 6, clean_line, new_x="LMARGIN", new_y="NEXT")

def generate_pdf_report(risk_score, explanation):
    pdf = HealthReportPDF()
    pdf.add_page()
    
    # Summary Box
    pdf.set_fill_color(245, 248, 252)
    pdf.set_draw_color(150, 180, 220)
    pdf.set_line_width(0.2)
    
    risk_pct = risk_score * 100
    risk_level = "HIGH" if risk_pct >= 65 else "MODERATE" if risk_pct >= 30 else "LOW"
    
    if risk_level == "HIGH":
        pdf.set_text_color(220, 50, 50)
    elif risk_level == "MODERATE":
        pdf.set_text_color(200, 120, 20)
    else:
        pdf.set_text_color(40, 160, 80)
        
    pdf.cell(0, 15, f'Overall Risk Level: {risk_level} ({risk_pct:.1f}%)', border=1, align='C', fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)
    
    # Detailed AI Explanation
    pdf.add_section_title("Detailed AI Assessment")
    pdf.add_text_body(explanation)
    
    # Disclaimer
    pdf.ln(10)
    pdf.set_font('helvetica', 'I', 9)
    pdf.set_text_color(150, 50, 50)
    pdf.multi_cell(0, 5, "Disclaimer: HealthGuard AI is an educational tool. This report is NOT a medical diagnosis and should NOT be used to make medical decisions without consulting a qualified healthcare professional.", new_x="LMARGIN", new_y="NEXT")
    
    return bytes(pdf.output())
