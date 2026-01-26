from dataclasses import dataclass
from pathlib import Path
import pdfplumber

@dataclass
class Document:
    doc_id: str
    source_file: str
    page_number: int
    text: str

def load_pdfs(pdf_dir: str) -> list[Document]:
    """
    Loads all PDFs inside pdf_dir, returns a list of Doc
    objects (1 per page).
    """
    documents = []
    pdf_dir = Path(pdf_dir)

    for pdf_path in pdf_dir.glob("*.pdf"):
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                text = page.extract_text()

                if not text or not text.strip():
                    continue

                documents.append(
                    Document(
                        doc_id=pdf_path.stem,
                        source_file=str(pdf_path),
                        page_number=page_number,
                        text=text.strip()
                    )
                )
    return documents

if __name__ == "__main__":
    docs = load_pdfs("data/raw/pdfs")

    print(f"Loaded {len(docs)} pages")

    for d in docs[:10]:
        print("\n---")
        print(f"Doc ID: {d.doc_id}")
        print(f"Page: {d.page_number}")
        print(d.text[:500])
