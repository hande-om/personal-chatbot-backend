import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# MODERN STANDALONE IMPORTS (Replaces langchain_community)
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# CORE LANGCHAIN IMPORTS (Requires `pip install langchain`)
# FIXED: Point to the modern classic namespace
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200",
                   "http://localhost:4201",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


rag_chain = None


@app.on_event("startup")
async def startup_event():
    global rag_chain
    try:
        # 1. Native Python file reading (Safely bypasses community TextLoader)
        if not os.path.exists("personal_details.txt"):
            raise FileNotFoundError("personal_details.txt file is missing!")

        with open("personal_details.txt", "r", encoding="utf-8") as f:
            text_content = f.read()

        # Wrap it directly into a LangChain Document structure
        docs = [Document(page_content=text_content)]

        # 2. Setup Vector Store using the dedicated standalone Chroma package
        #embeddings = OllamaEmbeddings(model="llama3.2")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vector_store = Chroma.from_documents(docs, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 1})

        # 3. Model & Chain Setup
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

        system_prompt = (
            "You are a helpful assistant representing Om Khade. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("🤖 Modern RAG System successfully initialized!")

    except Exception as e:
        print(f"Error during initialization: {str(e)}")


@app.post("/chat")
async def chat(payload: ChatRequest):
    global rag_chain
    if not rag_chain:
        raise HTTPException(status_code=503, detail="RAG system is not ready yet.")
    try:
        response = rag_chain.invoke({"input": payload.message})
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
