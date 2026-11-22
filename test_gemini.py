
import os

from dotenv import load_dotenv
from google import genai


# 1. Load environment variables
load_dotenv()

# 2. Get Gemini API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found!")

# 3. Initialize Gemini client
client = genai.Client(api_key=api_key)

# 4. Select Gemini model
GEMINI_MODEL = "gemini-3.5-flash"

# 5. Ask Gemini a question
prompt = "Explain Machine Learning in two simple sentences."

try:

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    print("\n========== GEMINI RESPONSE ==========\n")
    print(response.text)

except Exception as e:

    print("Gemini generation failed:", e)
