





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



llm = HuggingFaceEndpoint(repo_id = "HuggingFaceH4/zephyr-7b-alpha",
                     huggingfacehub_api_token = hf_inference_api_key,
                     temperature = 0.7, do_sample = True, repetition_penalty = 1.3,
                     model_kwargs={
                                   "num_beams": 5,
                                   "num_beam_groups": 4,
                                   "no_repeat_ngram_size": 3,
                                   "exponential_decay_length_penalty": (8, 0.5)
                        })

loader = TextLoader("rumi.txt")
document = loader.load()

# Assuming `documentclasstext` is a Document object
def json_serializable(documentclasstext):
    text_list = []
    for text in range(len(documentclasstext)):
        doc = documentclasstext[text]
        json_serializable_doc = {'page_content': doc.page_content,'metadata': doc.metadata}
        text_list.append(json_serializable_doc.get('page_content'))
    return text_list

fulltext = json_serializable(document)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
    length_function=len,
    add_start_index=True,
)

texts = text_splitter.create_documents(fulltext)

#print("Number of chunks: " + str(len(texts)))
#print("\nSample chunk:" + "\n")
#print(texts[-10])
#print()
#print(texts[-2] + "\n")
#print(texts[-1] + "\n")


embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_inference_api_key, 
    model_name="sentence-transformers/all-MiniLM-l6-v2"
)


db = Chroma.from_texts(json_serializable(texts), embeddings)
retriever = db.as_retriever(search_type="mmr", k=6, return_source_documents=False) #maximal marginal relevance for similarity + diversity



prompt_template = """

A ghazal is a poetic form that consists of rhyming couplets and a refrain, with each line sharing the same meter.

Here is an example of a ghazal about vitality, eternal peace and love: 

Where the water of life flows, no illness remains. 
In the garden of union, no thorn remains. 
They say there's a door between one heart and another. 
How can there be a door where no wall remains?

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


# Streamlit commands

st.set_page_config(page_title="Digital Rumi", page_icon=":robot:")
st.header('Digital Rumi')
st.text('This app is powered by langchain and huggingface, and made by @obaidisaac@gmail.com')

def get_subject_for_poem():
    #subject = "mystic, mountains and dreams"
    subject_input = st.text_input("Enter a few words and digital Rumi will write a poem, eg., mystic, mountains and dreams","")#, "eg., mystic, mountains and dreams")
    return subject_input

subject = get_subject_for_poem()

if subject:
    st.write("You entered: ")
    st.write(subject)
    retrieved_docs = retriever.invoke(subject)
    response = chain.invoke(subject)
    st.write(response)

st.markdown("")
st.markdown("")
st.markdown("")
st.markdown("BEFORE YOU LEAVE..")
st.markdown("Please DONATE!")
st.markdown("100% of donations go to Afghan children where Rumi was born in 1207. Contact the page author : obaidisaac@gmail.com")
st.image(image='https://gdb.rferl.org/806d0000-c0a8-0242-a656-08dabbebc4b4_cx0_cy5_cw0_w1597_n_r1_st_s.jpg', 
         width=500, 
         caption='https://www.rferl.org/a/afghanistan-child-labor-humanitiarian-economic-crisis/32415971.html')