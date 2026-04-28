import sys
import json
from docling.document_converter import DocumentConverter

def main():
    try:
        converter = DocumentConverter()
        pdf_path = "docs/LIIA BBSIA - Fase 1 MVP Banco Brasileiro de Soluções de IA v.02.pdf"
        print(f"Parsing {pdf_path}...")
        result = converter.convert(pdf_path)
        
        counts = {}
        for item, level in result.document.iterate_items():
            counts[str(getattr(item, "label", "unknown"))] = counts.get(str(getattr(item, "label", "unknown")), 0) + 1
            
        print("Item counts:")
        for k, v in counts.items():
            print(f"  {k}: {v}")
            
        print("First few items:")
        for i, (item, level) in enumerate(result.document.iterate_items()):
            if i >= 10:
                break
            page = item.prov[0].page_no if getattr(item, "prov", None) else "?"
            text = getattr(item, "text", "")[:50]
            print(f"[{page}] {getattr(item, 'label', 'unknown')}: {text}")
            
    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
