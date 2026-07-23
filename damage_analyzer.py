import json
import os
from google import genai
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini Client (automatically picks up GEMINI_API_KEY from .env)
client = genai.Client()

def analyze_damage_with_llm(pil_image, make, year):
    """
    Passes damage photo + Make & Year to Gemini Vision.
    Visually identifies vehicle model + damaged components.
    Outputs clean JSON search queries for eBay.
    """
    prompt = f"""
    You are an expert automotive damage appraisal system.
    
    Vehicle Metadata:
    - Manufacturer (Make): {make}
    - Production Year: {year}
    
    Instructions:
    1. Inspect the image to visually identify the specific vehicle model (e.g. Camry, Corolla, Altima, 3 Series).
    2. Identify all visibly damaged parts requiring full replacement.
    3. Output ONLY a valid JSON array of strings formatted as:
       ["{year} {make} [Model] [Part Name]"]
       
    Do NOT include markdown formatting or extra text.
    """

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[pil_image, prompt]
        )
        
        # Clean response and parse JSON array
        raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        queries = json.loads(raw_text)
        return queries if isinstance(queries, list) else [f"{year} {make} Front Bumper"]
        
    except Exception as e:
        print(f"[Gemini LLM Exception]: {e}")
        return [f"{year} {make} Front Bumper"]