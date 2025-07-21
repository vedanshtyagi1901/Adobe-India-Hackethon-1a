import os
import json
from pathlib import Path
import pdfplumber
from minillm import classify_heading  # Import from your MiniLM module

def is_likely_heading(text, page_lines, line_index):
    """Heuristic to filter likely headings."""
    # Convert to string and strip
    text = str(text).strip()
    if not text:
        return False
    
    # 1. Minimum and maximum length (e.g., 2-20 words)
    words = len(text.split())
    if words < 2 or words > 20:
        return False
    
    # 2. Capitalization check (starts with capital or all caps)
    if not (text[0].isupper() or text.isupper()):
        return False
    
    # 3. Position check (likely heading if near top of page, e.g., first 10% of lines)
    if line_index >= len(page_lines) * 0.1:  # Top 10% of lines
        return False
    
    return True

def process_pdfs():
    # Use current directory or adjust based on local setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = Path(os.path.join(base_dir, "input"))
    output_dir = Path(os.path.join(base_dir, "output"))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_files = list(input_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    print(f"Found {len(pdf_files)} PDF(s) in {input_dir}")
    
    for pdf_path in pdf_files:
        result = {"title": "", "outline": []}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if not text:
                        print(f"No text extracted from page {page_num} of {pdf_path.name}")
                        continue
                    lines = [ln.strip() for ln in text.split('\n') if ln.strip()]
                    print(f"Page {page_num} of {pdf_path.name} has {len(lines)} lines")
                    
                    for line_index, line in enumerate(lines):
                        pred = classify_heading(line)
                        print(f"Line: {line}, Prediction: {pred}, Page: {page_num}")  # Debug
                        if pred != 0 and is_likely_heading(line, lines, line_index):
                            mapped_level = {1: "H1", 2: "H2", 3: "H3"}.get(pred, "H3")
                            result["outline"].append({
                                "level": mapped_level,
                                "text": line,
                                "page": page_num
                            })
            
            # Set title to first heading if available
            if result["outline"]:
                result["title"] = result["outline"][0]["text"]
            
            # Write to output with matching filename
            output_file = output_dir / (pdf_path.stem + ".json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Processed {pdf_path.name} → {output_file.name} with {len(result['outline'])} headings")
        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")

if __name__ == "__main__":
    print("Starting processing PDFs...")
    process_pdfs()
    print("Completed processing PDFs.")
