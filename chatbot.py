import os
from google import genai
from mcp import plan

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def chat(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt
    )
    return response.text


if __name__ == "__main__":
    print("Chatbot ready! Type '/plan <query>' for plan mode, 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        if user_input.startswith("/plan "):
            query = user_input[6:]
            result = plan({"query": query})
            response = chat(result["prompt"])
            print(f"Bot: {response}")
        else:
            response = chat(user_input)
            print(f"Bot: {response}")