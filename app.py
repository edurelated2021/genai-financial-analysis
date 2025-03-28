from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import pandas as pd
import json
import uuid


from utils.prompts import PROMPT_TEMPLATE
import traceback
from typing import Dict

import pdfplumber

# Import Langchain modules
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from utils.common_constants import y


app = Flask(__name__)
app.secret_key = 'financial_insight_app_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
CHROMA_DB_DIR = "chroma_db"


@app.route('/')
def login():
    return render_template('index.html')

@app.route('/login')
def display_login():
    return render_template('login.html')


@app.route('/admin')
def display_admin():
    return render_template('admin.html')

@app.route('/index')
def display_index():
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html')

def get_embedding_function():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=y
    )
    return embeddings



def create_vectorstore(chunks, embedding_function, vectorstore_path):

    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content)) for doc in chunks]
    
    unique_ids = set()
    unique_chunks = []
    
    unique_chunks = [] 
    for chunk, id in zip(chunks, ids):     
        if id not in unique_ids:       
            unique_ids.add(id)
            unique_chunks.append(chunk) 

    vectorstore = Chroma.from_documents(documents=unique_chunks, 
                                        ids=list(unique_ids),
                                        embedding=embedding_function, 
                                        persist_directory = vectorstore_path)

    vectorstore.persist()
    
    return vectorstore



class Report(BaseModel):
    """Always use this tool to structure your response to the user."""
    title: str = Field(description="The title of the report")
    summary: str = Field(description="A brief summary of the report")
    details: Dict[str, str] = Field(description="Detailed information of the report enclosed and presented as html")
    details: Dict[str, str] = Field(description="A glossary containing difficult terms in the report and their meanings for a laymen to understand")



@app.route('/upload', methods=['POST'])
def upload():
    try:
        # Step 1: Load the PDF
        uploaded_file = request.files['file']  # Assuming 'file' is the form field name
        if uploaded_file.filename == '':
            return 'No file selected', 400
        
        # Step 2: Save the file to a directory
        upload_folder = 'uploads/'
        os.makedirs(upload_folder, exist_ok=True)
        pdf_path = os.path.join(upload_folder, uploaded_file.filename)
        
        # Save the file to the server
        uploaded_file.save(pdf_path)

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Step 2: Split the PDF content into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                       chunk_overlap=100,
                                                       length_function=len,
                                                       separators=["\n\n", "\n", " "])
        chunks = text_splitter.split_documents(pages)

        # Step 3: Store chunks in ChromaDB
        embeddings = get_embedding_function()
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
        
        # Step 3: Prepare the text for LLM processing (for now, using the first chunk of the document as context)
        context_text = "\n\n---\n\n".join([chunk.page_content for chunk in chunks[:1000]])  # You can adjust how many chunks you use
        #print(context_text)
        # Step 4: Create the prompt template
       
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        
        # Step 5: Prepare the prompt and the LLM
        model = ChatOpenAI(model="gpt-4o-mini", api_key=y)
        # Bind responseformatter schema as a tool to the model
        # model_with_tools = model.bind_tools([Report])
        
        chain = LLMChain(prompt=prompt, llm=model)

        # Step 6: Use the LLM to generate the report
        response = chain.run(context=context_text)
        print("--------RESPONSE FROM LLM ----------------------\n\n")
        print(response)
        print("\n\n-----------------------------------------------------\n\n")


        # Analyze the financial data
        return jsonify({'response': response})

    except Exception as e:
        error_message = f"Error in upload process: {str(e)}"
        # Print the stack trace to the log
        print(f"Error occurred: {error_message}")
        print(traceback.format_exc())
        return jsonify({'error': error_message})
    
    return jsonify({'error': 'Unknown error'})


@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Handles user queries by retrieving relevant chunks from ChromaDB and sending them to GPT-4o."""
    data = request.get_json()
    question = data.get("question", "")
    print('-----------------------')
    print("Question:", question)
    print('-----------------------')

    if not question:
        return jsonify({"error": "Question is required"}), 400
    
    # Load the existing ChromaDB
    embeddings = get_embedding_function()
    vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    
    # Perform similarity search
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})  # Retrieve top 3 relevant chunks
    relevant_docs = retriever.get_relevant_documents(question)
    
    # Extract text from retrieved documents
    context = "\n".join([doc.page_content for doc in relevant_docs])

    # Query GPT-4o with the retrieved context
    final_prompt = f"""
    You are a financial expert. Answer the following question based on the provided context.
    If the answer isn't available in the context, say "I don't have enough information." Don't make anything up.
    
    Context:
    {context}

    ---
    
    Question: {question}
    """

    llm = ChatOpenAI(model="gpt-4o", api_key=y)
    answer = llm.predict(final_prompt)
    print("Answer:", answer)

    return jsonify({"answer": answer})

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import io
import base64

# TODO
def generate_column_distribution_plot(df):
    plt.figure(figsize=(10, 5))
    df.count().plot(kind='bar', color='skyblue')
    plt.xlabel("Columns")
    plt.ylabel("Count")
    plt.title("Data Column Distribution")
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{plot_url}"

# TODO
def generate_histogram(df):
    plt.figure(figsize=(10, 5))
    df.hist(bins=20, edgecolor='black', grid=False)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{plot_url}"

# TODO
def generate_correlation_heatmap(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
    
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{plot_url}"

# TODO
@app.route('/visualize')
def visualize():
    df = extract_tables_from_pdf("sample.pdf")  # Assume this function extracts a DataFrame
    
    column_dist_img = generate_column_distribution_plot(df)
    histogram_img = generate_histogram(df)
    heatmap_img = generate_correlation_heatmap(df)

    return render_template("visuals.html", column_dist_img=column_dist_img, 
                           histogram_img=histogram_img, heatmap_img=heatmap_img)

# TODO: Extract all tables from a PDF
def extract_tables_from_pdf(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_tables = page.extract_tables()
            for table in extracted_tables:
                if table:  # Ensure table is not empty
                    df = pd.DataFrame(table[1:], columns=table[0])  # Convert to DataFrame
                    tables.append(df)
    return tables



if __name__ == '__main__':
    app.run(debug=True)
