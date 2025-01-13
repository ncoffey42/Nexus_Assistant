# RAG Memory Manager

import time
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import json


class MemoryManager:
    def __init__(self, embedding_model="all-MiniLM-L6-v2", faiss_index_path="./memory/joi_index.faiss", metadata_path="./memory/joi_metadata.json"):

        self.embedder = SentenceTransformer(embedding_model)

        # all-MiniLM-L6-v2 = 384 dimensions
        self.dimension = self.embedder.get_sentence_embedding_dimension()


        self.faiss_index_path = faiss_index_path
        self.metadata_path = metadata_path

        if os.path.exists(self.faiss_index_path):
            print(f"[MemoryManager] Loading FAISS index from {self.faiss_index_path}")
            self.index = faiss.read_index(self.faiss_index_path)
        else:
            print("[MemoryManager] Creating new FAISS index.")
            # L2 index in FAISS
            self.index = faiss.IndexFlatL2(self.dimension)

        # Store metadata (timestamp, text summary) in parallel list for retrieval
        # Convert over to database (probably SQLite) later, FAISS supposedly can store ~4 GB
        self.metadata = []

        # Tracks how many vectors have been added = number of conversation summaries
        # Used as ID for reference
        self.current_id = 0

        if os.path.exists(self.metadata_path):
            print(f"[MemoryManager] Loading metadata from {self.metadata_path}")
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.metadata = data["metadata"]
                self.current_id = data["current_id"]
        else:
            print("[MemoryManager] No existing metadata file found.")


    def embed_text(self, text: str) -> np.ndarray:

        # Returns 1D numpy array of embedding model dimensions = 384
        vec = self.embedder.encode([text])[0]
        return vec
    
    def store_memory(self, text: str):



        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        text_and_timestamp = f"{text} [Time: {timestamp}]"

        # Embed Conversation Summary
        vec = self.embed_text(text_and_timestamp)
        vec_np = np.array([vec], dtype=np.float32) # shape (1, dimension)

        # Add to the 
        # CHECK THIS, will retrieval be accurate if embedding occurs before timestamp added?
        self.index.add(vec_np)

        self.metadata.append({
            "id": self.current_id,
            "timestamp": timestamp,
            "summary": text_and_timestamp
        })

        self.current_id += 1

    def retrieve_memories(self, query: str, k=3) -> str:

        """
        1) Embed the user query
        2) Search top-k in FAISS
        3) Return list of memory texts
        """

        if self.index.ntotal == 0:
            return []

        q_vec = self.embed_text(query)
        q_vec_np = np.array([q_vec], dtype=np.float32)

        # Find top-k results
        distances, indices = self.index.search(q_vec_np, k)

        results = []

        for dist, idx in zip(distances[0], indices[0]):

            # retrieve corresponding metdata
            meta = self.metadata[idx]

            timestamp = meta["timestamp"]

            # local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))

            # Append a short conversation summary and a timestamp
            retrieved_summary = f"{meta['summary']} [Time: {timestamp}]"
            results.append(retrieved_summary)

        return results
    
    def extract_user_prompt(self, full_user_input: str) -> str:
        """
        full_user_input might look like:

            RETRIEVED CONTEXT UTILIZE IF RELEVANT TO USER PROMPT:
            (retrieved context...)

            USER PROMPT:
            (user text...)

        This function returns only the lines under "USER PROMPT:".
        If this section header is missing, returns the original string.`
        """

        header = "USER PROMPT:"
        if header in full_user_input:
            # Split into [before, after], may add more sections in the future such as ACTIONS
            sections = full_user_input.split(header)
            # The "prompt" you want is the text in the last segment after the marker
            # (strip to remove leading/trailing spaces/newlines)
            return sections[-1].strip()
        else:
            # If for some reason "USER PROMPT:" not found, return entire text
            return full_user_input.strip()


    # Summarizes the conversation
    def checkpoint_conversation(self, llm, convo_messages, max_chars=400):

        # summarize_prompt = f"Summarize the last 6 messages of the following conversation in under {max_chars} characters. Use the remaining messages as context if needed to summarize the last 6 messages in the conversation. Only give the summary of the messages do not mention that you are doing a summary or include any introduction in your response similar to 'In the last six messages,'"
        # summarize_prompt = f"Summarize the conversation in under {max_chars}, distinguish between what the user and the Assistant 'Joi'"

        # Remove Assistant System prompt if present in messages
        if convo_messages[0]["role"] == "system":
            convo_messages.pop(0)
       
       
        # Add all messages into 1 string to be summarized
        conversation_text = ""
        for msg in convo_messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                conversation_text += f"User: {self.extract_user_prompt(content)}\n"
            elif role == "assistant":
                conversation_text += f"Assistant: {content}\n"
            else:
                conversation_text += f"{role.capitalize()}: {content}\n"

        # Combine into a summarization system prompt
        summarization_prompt = (
            f"Summarize the last 6 messages below in under {max_chars} characters:\n\n" +
            f"{conversation_text}\n" +
            "Only produce the summary—no extra commentary."
        )

        print(summarization_prompt)
        
        response = llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            {"role":"system", "content": summarization_prompt}
            ]
        )
 
        # Return summary message
        summary_text = response.choices[0].message.content.strip()
        return summary_text
    

    def save_memory(self):
        """
        Write FAISS index to faiss_index_path
        Write metadata to metadata_path
        """

        faiss.write_index(self.index, self.faiss_index_path)

        with open(self.metadata_path, "w", encoding="utf-8") as f:
            data = {
                "metadata": self.metadata,
                "current_id": self.current_id
            }
            json.dump(data, f, indent=2)
        print("[MemoryManager] Saved Summary to disk")


