import os
import json
from pathlib import Path

def process_pdfs():
    # Get input and output directories
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PDF files in the input directory
    pdf_files = list(input_dir.glob("*.pdf"))
    
    # Define dummy data
    dummy_data = {
        "title": "Understanding AI",
        "outline": [
            {
                "level": "H1",
                "text": "Introduction",
                "page": 1
            },
            {
                "level": "H2",
                "text": "What is AI?",
                "page": 2
            },
            {
                "level": "H3",
                "text": "History of AI",
                "page": 3
            }
        ]
    }
    
    if not pdf_files:
        print("No PDF files found in /app/input")
    else:
        print(f"Found {len(pdf_files)} PDF(s) in /app/input")

    # Save the dummy.json file to /app/output
    output_file = output_dir / "dummy.json"
    with open(output_file, "w") as f:
        json.dump(dummy_data, f, indent=2)
    
    print(f"Saved dummy.json to {output_file}")

if __name__ == "__main__":
    print("Starting processing PDFs...")
    process_pdfs() 
    print("Completed processing PDFs.")
