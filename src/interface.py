import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import psycopg

# Connexion PostgreSQL
conn = psycopg.connect(
    dbname="mydb",
    user="admin",
    password="admin",
    host="127.0.0.1",
    port=5435
)

# Charger modèle de génération
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Fonction pour récupérer les textes similaires
def similar_corpus(input_corpus, top_k=3):
    from sentence_transformers import SentenceTransformer
    s_model = SentenceTransformer('all-MiniLM-L6-v2')
    input_emb = s_model.encode([input_corpus])[0].tolist()
    vector_str = "[" + ",".join(map(str, input_emb)) + "]"
    with conn.cursor() as cur:
        cur.execute("""
            SELECT corpus, embedding <=> %s AS distance
            FROM embeddings
            ORDER BY distance
            LIMIT %s
        """, (vector_str, top_k))
        return [r[0] for r in cur.fetchall()]

# Interface Streamlit
st.title("RAG Chatbot")

user_input = st.text_input("Posez votre question :")

if st.button("Envoyer") and user_input:
    retrieved_texts = similar_corpus(user_input)
    context = "\n".join(retrieved_texts)
    prompt = f"Réponds à la question suivante en utilisant le contexte ci-dessous :\nContexte:\n{context}\nQuestion: {user_input}"
    
    response = generate_response(prompt)
    st.write("**Réponse :**", response)
