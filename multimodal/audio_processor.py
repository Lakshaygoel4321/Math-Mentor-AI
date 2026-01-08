import whisper
import tempfile
import os

class AudioProcessor:
    def __init__(self, model_size="base"):
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        print("✅ Whisper model loaded")
    
    def process_audio(self, audio_file):
        """Transcribe audio to text"""
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_path = tmp_file.name
        
        try:
            # Transcribe audio
            result = self.model.transcribe(
                tmp_path,
                language="en",
                task="transcribe"
            )
            
            text = result["text"].strip()
            
            # Clean math-specific phrases
            text = self.clean_math_text(text)
            
            return text
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def clean_math_text(self, text):
        """Clean and normalize math-specific phrases"""
        replacements = {
            "squared": "^2",
            "cubed": "^3",
            "square root of": "sqrt(",
            "plus": "+",
            "minus": "-",
            "times": "*",
            "divided by": "/",
            "equals": "="
        }
        
        for phrase, symbol in replacements.items():
            text = text.replace(phrase, symbol)
        
        return text
