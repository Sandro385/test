import os
import json
import logging
import pickle
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI
from django.conf import settings

from .scrap_utils import (
    get_response_from_Google_CS_API, 
    refine_text_openai_LAW, 
    search_decisions_by_keywords, 
    refine_text_openai_DECISION,
    calculate_cost
)
from .models import FileInfo, RefactoredLaw

# გარემოს ცვლადების და ლოგერის ინიციალიზაცია
load_dotenv()
logger = logging.getLogger('operations')
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ASSISTANT_ID = "asst_7SzhLzaR9mEF2SI2nidsLS5v"

# FAISS ინდექსისა და Mapping-ის გლობალური ჩატვირთვა
INDEX_PATH = os.path.join(settings.BASE_DIR, "faiss_indexes", "refactored_laws.index")
MAPPING_PATH = os.path.join(settings.BASE_DIR, "faiss_indexes", "refactored_mapping.pkl")

try:
    if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
        GLOBAL_INDEX = faiss.read_index(INDEX_PATH)
        with open(MAPPING_PATH, 'rb') as f:
            GLOBAL_MAPPING = pickle.load(f)
        logger.info("FAISS index and mapping loaded successfully.")
    else:
        logger.warning("FAISS index or mapping file not found.")
        GLOBAL_INDEX = None
        GLOBAL_MAPPING = {}
except Exception as e:
    logger.error(f"Error loading FAISS index: {e}")
    GLOBAL_INDEX = None
    GLOBAL_MAPPING = {}

def assistant_get_response(thread_id, query_input, query_q, file_ids=None, language='ka'):
    total_cost_of_query = 0
    logger.info(f"Processing query: {query_q}")

    if not file_ids:
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=query_q if query_q else query_input
        )
    else:
        openai_file_ids = get_openai_file_ids(file_ids)
        client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=query_q if query_q else query_input,
            attachments=[
                {"file_id": fid, "tools": [{"type": "file_search"}, {"type": "code_interpreter"}]}
                for fid in openai_file_ids
            ],
        )

    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
    )
    
    if run.status == 'completed':
        total_cost_of_query += calculate_cost(
            completion_token=run.usage.completion_tokens,
            prompt_token=run.usage.prompt_tokens,
            cached_token=run.usage.prompt_token_details['cached_tokens']
        )
        total_cost_of_query *= 4 

        messages = client.beta.threads.messages.list(thread_id=thread_id)
        text_val = messages.data[0].content[0].text.value
        
        if messages.data[0].content[0].text.annotations:
            for annotation in messages.data[0].content[0].text.annotations:
                text_val = text_val.replace(annotation.text, '')

        return ((text_val, total_cost_of_query), "assistant")

    elif run.required_action:
        tool_outputs = []
        
        for tool in run.required_action.submit_tool_outputs.tool_calls:
            logger.info(f"Tool Triggered: {tool.function.name}")
            
            output_data = ("", 0)

            try:
                args = json.loads(tool.function.arguments)
                
                if tool.function.name == "get_response_from_openai_LAW":
                    output_data = get_response_from_openai_LAW(args['query'])
                    
                elif tool.function.name == "get_response_from_openai_DECISIONS":
                    output_data = get_response_from_openai_DECISIONS(
                        args['descriptive_query'], args['category']
                    )
                    
                elif tool.function.name == "get_response_from_Google_CS_API":
                    text_res = get_response_from_Google_CS_API(args['query'], args.get('filetype'))
                    output_data = (text_res, 0)

                tool_outputs.append({
                    "tool_call_id": tool.id,
                    "output": str(output_data[0])
                })
                total_cost_of_query += output_data[1]

            except Exception as e:
                logger.error(f"Error executing tool {tool.function.name}: {e}")
                tool_outputs.append({
                    "tool_call_id": tool.id,
                    "output": "Error fetching data."
                })

        if tool_outputs:
            run = client.beta.threads.runs.submit_tool_outputs_and_poll(
                thread_id=thread_id,
                run_id=run.id,
                tool_outputs=tool_outputs
            )
            
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(thread_id=thread_id)
                
                final_usage_cost = calculate_cost(
                    completion_token=run.usage.completion_tokens,
                    prompt_token=run.usage.prompt_tokens,
                    cached_token=run.usage.prompt_token_details['cached_tokens']
                ) * 4
                
                total_cost_of_query += final_usage_cost
                return ((messages.data[0].content[0].text.value, total_cost_of_query), "RAG")
            else:
                return (("No response after tool submission.", total_cost_of_query), "error")
    
    return (("Unexpected status.", 0), "error")


def get_response_from_openai_LAW(query):
    logger.info(f"SMART LAW SEARCH: {query}")
    
    try:
        if GLOBAL_INDEX is None or not GLOBAL_MAPPING:
            return ("System update in progress (Index not loaded).", 0)

        # 1. Smart Query Expansion
        system_prompt = """
        შენ ხარ ქართული სამართლის საძიებო ოპტიმიზატორი.
        შენი მიზანია მომხმარებლის კითხვა გარდაქმნა ფორმალურ იურიდიულ საძიებო ტერმინებად.
        
        1. ამოიცანი სავარაუდო კანონის სახელი (მაგ: 'საქართველოს სამოქალაქო კოდექსი').
        2. გამოყავი მთავარი იურიდიული კონცეფცია.
        3. შეადგინე საძიებო სიტყვების სია.

        JSON ფორმატი:
        {
            "target_law": "კანონის სახელი",
            "optimized_search_terms": "სიტყვა1, სიტყვა2",
            "refined_query": "დახვეწილი იურიდიული კითხვა"
        }
        """
        
        expansion_response = client.chat.completions.create(
            model="gpt-5.2",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Query: {query}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        
        ai_data = json.loads(expansion_response.choices[0].message.content)
        target_law = ai_data.get("target_law", "")
        refined_query = ai_data.get("refined_query", query)
        search_terms = ai_data.get("optimized_search_terms", "")
        
        query_cost = calculate_cost(
            expansion_response.usage.completion_tokens,
            expansion_response.usage.prompt_tokens,
            expansion_response.usage.prompt_token_details.cached_tokens
        )

        augmented_query = f"კანონი: {target_law}\nსაძიებო სიტყვები: {search_terms}\nკითხვა: {refined_query}\nორიგინალი: {query}"
        
        # 2. ვექტორიზაცია
        chunk_embedding = client.embeddings.create(input=augmented_query, model="text-embedding-3-large").data[0].embedding
        query_vector = np.array(chunk_embedding, dtype=np.float32).reshape(1, -1)

        # 3. ძებნა (200 კანდიდატი)
        k = 200
        distances, indices = GLOBAL_INDEX.search(query_vector, k)

        # 4. DB Optimization: Batch Fetch
        candidate_ids = []
        ordered_map = {}
        
        for i, idx in enumerate(indices[0]):
            if idx in GLOBAL_MAPPING:
                db_id = GLOBAL_MAPPING[idx]
                candidate_ids.append(db_id)
                ordered_map[db_id] = i

        fetched_laws = RefactoredLaw.objects.filter(id__in=candidate_ids)
        
        # შედეგების დალაგება
        sorted_laws = sorted(
            [law for law in fetched_laws], 
            key=lambda x: ordered_map.get(x.id, 9999)
        )

        final_chunks = []
        for law_entry in sorted_laws:
            # ფილტრი: კანონის სახელი
            if target_law:
                if target_law not in law_entry.law_name and law_entry.law_name not in target_law:
                    continue

            text_content = law_entry.text_ka if law_entry.text_ka else law_entry.text_en
            formatted_chunk = f"**მუხლი {law_entry.article_number} – {law_entry.law_name}**\n{text_content}"
            final_chunks.append(formatted_chunk)

            if len(final_chunks) >= 15:
                break

        # 5. პასუხის გენერაცია
        refined_response_LAW = refine_text_openai_LAW(
            chunks=final_chunks,
            query=query, 
            file_summary=''
        )
        
        return (refined_response_LAW[0], query_cost + refined_response_LAW[1])
        
    except Exception as e:
        logger.error(f"LAW Search Error: {e}")
        return (f"სისტემური შეცდომა: {str(e)}", 0)


def get_response_from_openai_DECISIONS(descriptive_query, category):
    try:
        decision_chunks = search_decisions_by_keywords(user_query=descriptive_query, category=category)
        
        if decision_chunks:
            response_data = refine_text_openai_DECISION(query=descriptive_query, chunks=decision_chunks)
            return response_data
        else:
            return ("სამწუხაროდ, რელევანტური პრეცედენტი ვერ მოიძებნა.", 0)
            
    except Exception as e:
        logger.error(f"DECISION Search Error: {e}")
        return ("შეცდომა პრეცედენტების ძიებისას.", 0)


def get_openai_file_ids(file_ids):
    openai_ids = []
    if file_ids:
        for fid in file_ids:
            try:
                f_info = FileInfo.objects.get(id=fid)
                openai_ids.append(f_info.file_id_openai)
            except FileInfo.DoesNotExist:
                continue
    return openai_ids