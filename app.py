import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
st.set_page_config(page_title="Mini RAG Q&A",
page_icon=" 📚 ")
st.title("📚 Mini RAG Q&A")
st.write("Paste  document,store it in chromadb, and ask questions about it.")
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')
embedding_model = load_embedding_model()
client=chromadb.Client()
collection=client.get_or_create_collection(name="documents")
document=st.text_area(" 📄Paste your document here",
height=250, placeholder="paste your notes,articles,syllabus, etc.")
if st.button("➕ Add Document"):
    if not document:
        st.warning("Please enter some text.")
    else:
        chunks=[
            document[i:i+500] for i in range(0, len(document), 500)
        ]
        embeddings=embedding_model.encode(chunks)
        collection.add(
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings.tolist()
        )
        st.success(f"Added {len(chunks)} chunk(s) to chromodb.")
question=st.text_input("❓Ask a question about the document")
if st.button("🔍Ask AI"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif collection.count() == 0:
        st.warning("No documents found. Please add a document first.")
    else:
        question_embedding=embedding_model.encode([question])[0]
        results=collection.query(
            query_embeddings=[question_embedding.tolist()],
            n_results=min(3, collection.count())
        )
        retrieved_chunks=results["documents"][0]
        context="\n\n".join(retrieved_chunks)
        prompt=f"""
        you are a helpful  AI assistant.
        Answer the question based on the context below.
        Context: {context}
        Question: {question}
        If the answer is not in the context, say "I don't know based on the provided context".
        """
        try:
            response=ollama.chat(
                model="llama2",
                messages=[{"role": "user", "content": prompt}],
            )
            st.subheader("🤖 Answer")

            st.write(response['message']['content'])
            with st.expander(" Retrieved"):
                for i, chunk in enumerate(retrieved_chunks):
                    st.write(f"**Chunk {i+1}:**")
                    st.write(chunk)

        except Exception as e:
            st.error("Could not get a response from the AI model. Please try again later.")
            st.code(str(e))
st.divider()
st.caption("Python + Sentence Transformers + ChromaDB + Ollama +Streamlit")