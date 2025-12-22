from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from openai import OpenAI
import openai

import requests
import json

import os 
from dotenv import load_dotenv
load_dotenv( )


def user_search_query_processor( query ):
    """
        Method for Seaching url and scraping the url site,
        1. GCSP for searching the query over web and it will return relevant url
        2. BeautifulSoup for scraping Parsed Html data
    """

    url = 'https://www.googleapis.com/customsearch/v1'
    params = {
        'q' : query,
        'key' : os.getenv( 'GOOGLE_CUSTOM_SEARCH_API' ),
        'cx' : os.getenv( 'SEARCH_ENGINE_ID' )
    }
    response = requests.get( url, params = params )
    result = response.json( )
    if 'items' in result:
        link = result[ 'items' ][ 0 ][ 'link' ]
    else:
        return None
    page_response = requests.get( link )
    if page_response.status_code == 200:
        html_content = BeautifulSoup( page_response.content, 'html.parser' )
    else:
        return None
    # title_data = html_content.find_all( 'h2' )
    paragraph_data = html_content.find_all( 'p' )
    scraped_data = ''
    for p in paragraph_data:
        if p.get_text( ) != '\n':
            text = p.get_text( ).strip( '\n' )
            scraped_data += text.strip( " " ) 
    # scraped_data = [ p.get_text( ).strip( "\n" ) for p in paragraph_data if p.get_text( ) != "\n" ]  # Scrapped data in list form
    return scraped_data 
    

def create_userquery_embedding( query, model="text-embedding-ada-002" ):
    """
        Method for create vector embedding of user query
        1. OpenAI for creating vector embedding  
    """

    client = OpenAI( api_key = os.getenv( 'OPENAI_API_KEY' ) )
    response = client.embeddings.create( 
        input = query,
        model = model
    )
    return response.data


def text_embedding( text, model="text-embedding-ada-002" ):
    """
        Method for creating vector embedding of chunk text
        1. OpenAI for creating text embedding
    """

    client = OpenAI( api_key = os.getenv( 'OPENAI_API_KEY' ) )
    response = client.embeddings.create( 
        input = text,
        model = model
    )
    return response.data


def convert_data_into_chunks( data, chunk_size = 1024, overlap = 0 ):
    """
        This method will convert raw string into chunk of given size  
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = overlap
    )
    chunks = text_splitter.split_text( data )
    return chunks


def prepair_vector( chunk_data ):
    """
        This method will prepair The vector for upsirt on PineCone 
    """

    vector = [ ]
    for ite, text in enumerate( chunk_data ):
        id = f"chunk {ite}"
        vector_value = text_embedding( text = text )
        text_data = text
        vector.append( 
            { 
                "id" : id,
                "values" : vector_value,
                "metadata" : { "text" : text_data }
            }
        )
        print( { "vector" : vector } )
    return vector

def get_all_text( all_text ):
    """
        getting all text from pinecone response
    """

    text = [ '' ] * len( all_text[ 'matches' ] )
    for ite, match in enumerate( all_text[ 'matches' ] ):
        text[ ite ] += match[ 'text' ]
    return "".join( text )


def get_user_friendly_query_answer( query, pinecone_responses ):
    """
        creating pinecone response text into user friendly/meaning full text.
    """

    client = OpenAI(
        api_key = os.getenv("OPENAI_API_KEY"),
    )

    prompt = [
        {"role": "system", "content": "You are a helpful assistant that summarizes information based on user queries."},
        {"role": "user", "content": f"User's query: {query}"},
        {"role": "assistant", "content": f"Retrieved information: {pinecone_responses}"},
        {"role": "user", "content": "Generate meaningful sentences that provide an answer or summary based on the information."}
    ]

    openai_response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = prompt,  
        # max_tokens=500,  
        temperature=0.4
    )
    return openai_response.choices[0].message.content
