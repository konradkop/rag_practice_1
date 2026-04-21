# query_database.py
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

CHROMA_PATH = "choma"

PROMPT_TEMPLATE = """
Here is some context for the question:

{context}

---

Here is the question answer in max two sentences: {question}
"""

def main():
    llm = ChatOllama(model="llama3")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )


    # prepare database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    query_text = input("Enter what you would like to search: ")
    print('your query was:', query_text )

    # search the database
    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    if len(results) == 0:
        print('Unable to find matching results')
        return
    
    all_content = "\n\n---\n\n".join([result.page_content for result, score in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    formatted_prompt = prompt.format(context=all_content, question=query_text)

    print('\n \n')
    response = llm.invoke(formatted_prompt)
    print("\n=== ANSWER ===\n")
    print(response.content)

        

main()
