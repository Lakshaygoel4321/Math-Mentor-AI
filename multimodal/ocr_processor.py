import easyocr
from PIL import Image
import numpy as np

class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def process_image(self, image_path_or_file):
        """Process image and extract text with confidence"""
        # Convert to numpy array if PIL Image
        if isinstance(image_path_or_file, Image.Image):
            image = np.array(image_path_or_file)
        else:
            image = image_path_or_file
        
        # Perform OCR
        results = self.reader.readtext(image, detail=1)
        
        if not results:
            return "", 0.0
        
        # Extract text and calculate average confidence
        extracted_text = ' '.join([res[1] for res in results])
        avg_confidence = sum([res[2] for res in results]) / len(results)
        
        return extracted_text, avg_confidence
    
    def needs_hitl(self, confidence, threshold=0.7):
        """Check if HITL is needed based on confidence"""
        return confidence < threshold
