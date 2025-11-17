import os
from dotenv import load_dotenv
from transformers import pipeline
import openai

load_dotenv()

class EmotionChatbot:
    def __init__(self, api_key=None):
        # Emotion classifier
        print("Loading emotion classifier...")
        self.emotion_classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            top_k=1
        )
        
        # OpenAI client
        self.client = openai.OpenAI(
            base_url="https://api.mistral.ai/v1",
            api_key=api_key or os.getenv("MISTRAL_API_KEY")
        )
        
        self.base_prompt = "Respond to the user empathetically and naturally."
    
    def classify_emotion(self, text: str) -> tuple:
        #Classify emotion (returns label, confidence).
        result = self.emotion_classifier(text)[0][0]
        print(result)
        return result['label'], result['score']
    
    def respond_with_emotion_detection(self, user_input: str) -> dict:
        #Method 1: Explicit emotion detection.
        emotion, confidence = self.classify_emotion(user_input)
        
        prompt = f"""
User input: "{user_input}"
Detected emotion: {emotion} (confidence: {confidence:.2f})
Guideline: Respond empathetically to a user expressing {emotion}.
Generate a brief response:"""
        
        response = self.client.chat.completions.create(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        
        return {
            "method": "Emotion Detection",
            "user_input": user_input,
            "emotion": emotion,
            "confidence": confidence,
            "response": response.choices[0].message.content
        }
    
    def respond_baseline(self, user_input: str) -> dict:
        #Method 2: Simple empathetic prompting (your argument).
        prompt = f"""
User input: "{user_input}"
Guideline: {self.base_prompt}
Generate a brief response:"""
        
        response = self.client.chat.completions.create(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7
        )
        
        return {
            "method": "Baseline Prompting",
            "user_input": user_input,
            "emotion": "N/A",
            "confidence": "N/A",
            "response": response.choices[0].message.content
        }
    
    def compare_methods(self, user_input: str) -> dict:
        #Compare both methods side-by-side.
        print(f"\nUser: \"{user_input}\"")
        print("-" * 50)
        
        # Method 1: With emotion detection
        result1 = self.respond_with_emotion_detection(user_input)
        print(f"With Emotion Detection:")
        print(f"  Detected: {result1['emotion']} ({result1['confidence']:.2f})")
        print(f"  Response: {result1['response']}")
        
        # Method 2: Baseline
        result2 = self.respond_baseline(user_input)
        print(f"\nWith Baseline Prompting:")
        print(f"  Response: {result2['response']}")
        
        return {"with_detection": result1, "baseline": result2}

def run_assignment_tests():
    bot = EmotionChatbot()
    
    # Test 1: Clear negative emotion (should work)
    print("TEST 1: Clear emotional cue")
    bot.compare_methods("I failed my exam. I studied so hard and still failed.")
    
    # Test 2: Sarcastic/ambiguous (where detection fails)
    print("\n\nTEST 2: Ambiguous/sarcastic input")
    bot.compare_methods("Oh great, another amazing day. Just wonderful.")

if __name__ == "__main__":
    run_assignment_tests()