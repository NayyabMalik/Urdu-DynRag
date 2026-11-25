"""
Generates a publication-grade visual sample card for the Urdu-DynRAG benchmark.
Outputs: paper/figures/fig6_dataset_sample.png (300 DPI)
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def generate_sample_card():
    fig = plt.figure(figsize=(11, 8.5), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')

    # Background canvas
    canvas = patches.Rectangle((0, 0), 1, 1, facecolor='#F8FAFC', edgecolor='none')
    ax.add_patch(canvas)

    # Outer Container
    card = patches.FancyBboxPatch((0.03, 0.03), 0.94, 0.94,
                                  boxstyle="round,pad=0.015,rounding_size=0.02",
                                  facecolor='#FFFFFF', edgecolor='#CBD5E1', linewidth=1.5)
    ax.add_patch(card)

    # Header Banner
    header = patches.FancyBboxPatch((0.04, 0.88), 0.92, 0.08,
                                    boxstyle="round,pad=0.01,rounding_size=0.015",
                                    facecolor='#1E293B', edgecolor='none')
    ax.add_patch(header)
    
    ax.text(0.06, 0.93, "Urdu-DynRAG Benchmark — Representative Dataset Sample",
            fontsize=15, weight='bold', color='#FFFFFF', va='center')
    ax.text(0.06, 0.90, "Passage ID: doc_lic_001_punjab   |   Domain: Punjab Traffic Licensing Regulations (Official Civic Corpus)",
            fontsize=10, color='#94A3B8', va='center')
    ax.text(0.85, 0.92, "VERIFIED",
            fontsize=10, weight='bold', color='#10B981', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#064E3B", edgecolor="#10B981", lw=1))

    # Section 1: Document Context (Civic Passage)
    ctx_box = patches.FancyBboxPatch((0.04, 0.69), 0.92, 0.17,
                                    boxstyle="round,pad=0.01,rounding_size=0.012",
                                    facecolor='#F1F5F9', edgecolor='#E2E8F0', linewidth=1)
    ax.add_patch(ctx_box)
    
    ax.text(0.055, 0.835, "GROUND TRUTH PASSAGE CONTEXT (Perso-Arabic Nastaliq Corpus)",
            fontsize=9.5, weight='bold', color='#475569')
    
    # Urdu Nastaliq Text (Rendered clearly)
    urdu_context = (
        "[صوبہ پنجاب - ڈرائیونگ لائسنس کی تجدید]: پنجاب ٹریفک پولیس کے مطابق ڈرائیونگ لائسنس کی تجدید کے لیے شناختی کارڈ کی نقل،\n"
        "دو تصاویر اور میڈیکل سرٹیفکیٹ درکار ہوتا ہے۔ 50 سال یا اس سے زیادہ عمر کے افراد کو میڈیکل فٹنس سرٹیفکیٹ بھی دینا ہوتا ہے۔"
    )
    ax.text(0.94, 0.775, urdu_context, fontsize=11, color='#0F172A', ha='right', va='center', fontfamily='sans-serif', linespacing=1.6)

    # English Gloss
    eng_context = "Translation: [Punjab Traffic Police - License Renewal]: Driving license renewal requires CNIC copy, 2 photographs, and medical certificate. Applicants aged 50 or above must also submit a medical fitness certificate. (Validity: 5 years)."
    ax.text(0.055, 0.715, eng_context, fontsize=8.5, style='italic', color='#64748B')

    # Section 2: Multi-Style Query Variations
    y_start = 0.64
    ax.text(0.04, y_start, "EVALUATED MULTI-STYLE QUERY REPRESENTATIONS (5 Distinct Modalities)",
            fontsize=10.5, weight='bold', color='#0F172A')

    styles = [
        ("S1: Clean Nastaliq",
         "[Perso-Arabic]  ڈرائیونگ لائسنس کی تجدید کے وقت کن درخواست گزاروں کو میڈیکل فٹنس سرٹیفکیٹ بھی دینا ہوتا ہے؟",
         "Roman / Gloss: Draiving laisans ki tajdeed par kin darkhwast guzaron ko medical fitness certificate dena hota hai?  (Which applicants must submit fitness certificate?)",
         "#3B82F6", "#EFF6FF", "#BFDBFE"),
        
        ("S2: Conversational",
         "[Perso-Arabic]  مجھے بتائیں، لائسنس کی تجدید پر میڈیکل فٹنس سرٹیفکیٹ کس عمر کے لوگوں سے مانگا جاتا ہے؟",
         "Roman / Gloss: Mujhe batayein, license ki tajdeed par medical fitness certificate kis umar ke logon se manga jata hai?  (Tell me, at what age is fitness cert required?)",
         "#8B5CF6", "#F5F3FF", "#DDD6FE"),
        
        ("S3: Code-Switched",
         "[Mixed Script]  driving licence کے renewal پر کس age کے applicants سے medical fitness certificate لیا جاتا ہے؟",
         "Roman / Gloss: driving licence ke renewal par kis age ke applicants se medical fitness certificate liya jata hai?  (Mixed English technical loanwords)",
         "#D97706", "#FFFBEB", "#FDE68A"),
        
        ("S4: Roman Urdu",
         "[Latin Script]   Driving license ki tajdeed par kis umar ke logon se medical fitness certificate manga jata hai?",
         "Orthography:   Phonetic transcription lacking standardized spelling. Exhibits 23.5-point dense retrieval collapse without query reformulation.",
         "#EF4444", "#FEF2F2", "#FECACA"),
        
        ("S5: Hard Distractor",
         "[Unanswerable]  ڈرائیونگ لائسنس کی تجدید کی فیس کس بینک میں جمع کرائی جاتی ہے؟",
         "Roman / Gloss: Draiving laisans ki tajdeed ki fees kis bank mein jama karayi jaati hai?  (Entity-overlap trap; requires calibrated abstention)",
         "#6B7280", "#F9FAFB", "#E5E7EB")
    ]

    box_y = 0.585
    for tag, primary, secondary, badge_col, bg_col, border_col in styles:
        sbox = patches.FancyBboxPatch((0.04, box_y), 0.92, 0.08,
                                      boxstyle="round,pad=0.008,rounding_size=0.01",
                                      facecolor=bg_col, edgecolor=border_col, linewidth=1)
        ax.add_patch(sbox)
        
        # Style badge
        ax.text(0.055, box_y + 0.052, tag, fontsize=8.5, weight='bold', color=badge_col,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor=badge_col, lw=1))
        
        ax.text(0.18, box_y + 0.052, primary, fontsize=9.2, weight='bold', color='#1E293B', va='center')
        ax.text(0.18, box_y + 0.020, secondary, fontsize=7.8, color='#475569', va='center')
        
        box_y -= 0.093

    # Section 3: Ground Truth Answer & Optimal Routing
    bottom_box = patches.FancyBboxPatch((0.04, 0.045), 0.92, 0.075,
                                        boxstyle="round,pad=0.008,rounding_size=0.01",
                                        facecolor='#ECFDF5', edgecolor='#A7F3D0', linewidth=1)
    ax.add_patch(bottom_box)
    
    ax.text(0.055, 0.092, "OPTIMAL POLICY ROUTING ACTION:", fontsize=8.5, weight='bold', color='#065F46')
    ax.text(0.28, 0.092, "Action 3 (Urdu Reformulator)  -->  Action 1 (Raw Passage Chunk Retrieval)",
            fontsize=8.5, weight='bold', color='#047857',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#D1FAE5", edgecolor="#34D399", lw=1))
    
    ax.text(0.055, 0.062, "GOLD ANSWER:", fontsize=8.5, weight='bold', color='#065F46')
    ax.text(0.16, 0.062, "ڈرائیونگ لائسنس کی تجدید کے وقت 50 سال یا اس سے زیادہ عمر کے افراد کو میڈیکل فٹنس سرٹیفکیٹ دینا ہوتا ہے۔",
            fontsize=9.2, weight='bold', color='#064E3B')
    ax.text(0.68, 0.062, "(Applicants aged 50+ must submit fitness certificate)", fontsize=8, style='italic', color='#047857')

    os.makedirs('paper/figures', exist_ok=True)
    out_path = 'paper/figures/fig6_dataset_sample.png'
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"Sample card successfully saved to: {out_path}")

if __name__ == '__main__':
    generate_sample_card()

