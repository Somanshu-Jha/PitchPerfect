import os
import logging

logger = logging.getLogger(__name__)


class DocumentService:
    """
    High-Fidelity Document Analysis Service.
    Supports: .pdf, .docx, .txt
    OCR: Lazy-loads GLM-OCR (0.9B) only when needed for image-based PDFs.
    """

    def __init__(self):
        self.ocr_model = None
        self.ocr_processor = None
        logger.info("📄 [DocumentService] Initialized (OCR: lazy-load)")

    def _load_ocr(self):
        """Lazy load GLM-OCR only when needed to save VRAM."""
        if self.ocr_model is None:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_id = "zai-org/GLM-OCR"
            logger.info(f"🧠 [DocumentService] Loading GLM-OCR ({model_id})...")
            try:
                self.ocr_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                self.ocr_model = AutoModelForVision2Seq.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                )
                logger.info("✅ [DocumentService] GLM-OCR loaded.")
            except Exception as e:
                logger.error(f"❌ [DocumentService] GLM-OCR failed: {e}")

    def extract_text(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == ".pdf":
                return self._extract_pdf(file_path)
            elif ext == ".docx":
                return self._extract_docx(file_path)
            elif ext == ".txt":
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                logger.warning(f"⚠️ Unsupported: {ext}")
                return ""
        except Exception as e:
            logger.error(f"❌ Extraction failed: {e}")
            return ""

    def _extract_pdf(self, file_path: str) -> str:
        import pdfplumber
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if len(text.strip()) < 50:
                logger.info("📸 PDF is image-based. Triggering OCR...")
                return self._ocr_pdf(file_path)
            return text.strip()
        except Exception as e:
            logger.warning(f"⚠️ PDF extraction failed: {e}. Falling back to OCR.")
            return self._ocr_pdf(file_path)

    def _extract_docx(self, file_path: str) -> str:
        import docx
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs]).strip()

    def _ocr_pdf(self, file_path: str) -> str:
        self._load_ocr()
        if not self.ocr_model:
            return "OCR unavailable."
        import torch
        from PIL import Image
        import fitz
        doc = fitz.open(file_path)
        full_text = []
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            inputs = self.ocr_processor(images=img, text="Transcribe this page.", return_tensors="pt")
            device = next(self.ocr_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                output = self.ocr_model.generate(**inputs, max_new_tokens=1024)
            full_text.append(self.ocr_processor.decode(output[0], skip_special_tokens=True))
        return "\n\n".join(full_text).strip()


document_service = DocumentService()
