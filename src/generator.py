from groq import Groq
import os

class AnswerGenerator:
    def __init__(self, model_name: str, api_key: str = None):
        """Initializes the Groq client."""
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key not found. Please set GROQ_API_KEY in .env")
        
        self.client = Groq(api_key=self.api_key)

    def generate_answer(self, query: str, context_chunks: list[dict]) -> str:
        """
        Generates an answer based on the query and provided context.
        
        Args:
            query: The user's question.
            context_chunks: A list of document dictionaries returned by the reranker.
            
        Returns:
            The generated answer string.
        """
        # Format the context
        context_text = ""
        for i, chunk in enumerate(context_chunks):
            title = chunk.get("metadata", {}).get("title", f"Document {i+1}")
            text = chunk.get("text", "")
            context_text += f"\n--- {title} ---\n{text}\n"
            
        system_prompt = (
            "You are a helpful assistant. Use ONLY the provided context to answer the user's question. "
            "If the answer cannot be found in the context, say 'I cannot answer this based on the provided context.' "
            "Be concise and cite the source titles if possible."
        )
        
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
        
        # Call Groq API
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0, # Deterministic answers based on context
            max_tokens=1024,
            stream=False,
        )
        
        return completion.choices[0].message.content
