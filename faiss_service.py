import boto3
import os
import uuid
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

folder_path = "/tmp/"

s3_client = boto3.client("s3")
BUCKET_NAME = os.getenv("BUCKET_NAME")
bedrock_client = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_client)

def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="myfile.faiss", Filename=f"{folder_path}myfile.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="myfile.pkl", Filename=f"{folder_path}myfile.pkl")

def get_llm():
    return Bedrock(model_id="anthropic.claude-v2:1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})
    # return Bedrock(model_id="amazon.titan-text-express-v1", client=bedrock_client, model_kwargs={'max_tokens_to_sample': 512})

def get_response(question):
    load_index()
    faiss_index = FAISS.load_local(
        index_name="myfile",
        folder_path=folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    prompt_template = """
    Human: You are interacting with an Aurora (AI) that represents me using my resume and other details. Please use the given context to provide a concise answer to the question. Only answer professional, hobbies, career, or behavioral questions if they are found in the context. If you don't know the answer or the question is outside these topics, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>
    Question: {question}
    Aurora:"""


    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=get_llm(),
        chain_type="stuff",
        retriever=faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": question})
    return answer['result']
