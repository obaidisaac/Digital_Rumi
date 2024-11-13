

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3

#from langchain.llms import HuggingFaceHub
#from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
#from langchain.vectorstores import Chroma
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
#from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

#from langchain import HuggingFaceHub
#from langchain_community.llms import HuggingFaceHub
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma


import streamlit as st

import os

hf_inference_api_key = os.getenv('hf_inference_api_key')

st.set_page_config(page_title="Digital Rumi", page_icon=":robot:")
st.header('Digital Rumi')
st.text('This app is powered by langchain and huggingface, and uses the Zephyr 7B LLM model.')
st.write("")
st.write("")

@st.cache_data
def llm_loader():
    llm = HuggingFaceEndpoint(repo_id = "HuggingFaceH4/zephyr-7b-alpha",
                     huggingfacehub_api_token = hf_inference_api_key,
                     temperature = 0.7, do_sample = True, repetition_penalty = 1.3,
                     model_kwargs={
                                   "num_beams": 5,
                                   "num_beam_groups": 4,
                                   "no_repeat_ngram_size": 3,
                                   "exponential_decay_length_penalty": (8, 0.5)
                        })
    return llm

@st.cache_data
def text_loader(textfile):
    loader = TextLoader(textfile)
    document = loader.load()
    return document

@st.cache_data
def json_serializable(_documentclasstext):
    text_list = []
    for text in range(len(_documentclasstext)):
        doc = _documentclasstext[text]
        json_serializable_doc = {'page_content': doc.page_content,'metadata': doc.metadata}
        text_list.append(json_serializable_doc.get('page_content'))
    return text_list

@st.cache_data
def text_splitter(fulltext):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=30,
        length_function=len,
        add_start_index=True,
        )
    texts = text_splitter.create_documents(fulltext)
    return texts


@st.cache_data
def embeddings_hf():
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=hf_inference_api_key, 
        model_name="sentence-transformers/all-MiniLM-l6-v2"
    )
    return embeddings

@st.cache_resource
def vector_db(_texts,_embeddings):
    db = Chroma.from_texts(json_serializable(_texts), _embeddings)
    retriever = db.as_retriever(search_type="mmr", k=6, return_source_documents=False) 
    # mmr = maximal marginal relevance for similarity + diversity
    return retriever


    
llm = llm_loader()
document = text_loader("rumi.txt")
fulltext = json_serializable(document)
texts = text_splitter(fulltext)
embeddings = embeddings_hf()
retriever = vector_db(texts, embeddings)






prompt_template = """
You write ghazals in the style of Rumi.
A ghazal is a poetic form that consists of rhyming couplets and a refrain, with each line sharing the same meter.

###
vitality, eternal peace and love 
###
Where the water of life flows, no illness remains. 
In the garden of union, no thorn remains. 
They say there's a door between one heart and another. 
How can there be a door where no wall remains?
###

Write a ghazal about {question} based on the following context: {context}

"""


prompt = PromptTemplate.from_template(template=prompt_template)#, input_variables=["question","context"])
model = llm
#def format_docs(docs):
#    return "\n\n".join([d.page_content for d in docs])
#"context": retriever | format_docs
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

charity_url = "https://www.savethechildren.org/us/where-we-work/afghanistan"

# Streamlit commands
#st.set_page_config(page_title="Digital Rumi", page_icon=":robot:")
#st.header('Digital Rumi')
#st.text('This app is powered by langchain and huggingface, and uses the Zephyr 7B LLM model.')
#st.write("")
#st.write("")


def get_subject_for_poem():
    #subject = "mystic, mountains and dreams"
    subject_input = st.text_input("Enter a few words and digital Rumi will write a poem, eg., mystic, mountains and dreams","")#, "eg., mystic, mountains and dreams")
    return subject_input

subject = get_subject_for_poem()
t = 1   

#if subject is True and len(subject) > 50:
#    st.write("Your entry is too long. Please reduce and try again.")

if subject:# is True and len(subject) <= 50:
    st.write("You entered: ")
    st.write(subject)
    st.write("")
    retrieved_docs = retriever.invoke(subject)
    response = chain.invoke(subject)
    st.write("")
    st.write("A POEM")
    st.write("")
    st.write(response)

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("BEFORE YOU GO..")
st.markdown("")
st.markdown("Please donate to the children of Afghanistan, where Rumi was born in 1207.")
st.markdown("[savethechildren.org](%s)" % charity_url) 
st.markdown("")


st.image(image='https://gdb.rferl.org/806d0000-c0a8-0242-a656-08dabbebc4b4_cx0_cy5_cw0_w1597_n_r1_st_s.jpg', 
         width=700, 
         caption='https://www.rferl.org/a/afghanistan-child-labor-humanitiarian-economic-crisis/32415971.html')

st.text("")
st.text("")
st.text("App author : obaidisaac@gmail.com")