import os, re, json, logging, time
import numpy as np
from openai import OpenAI
from django.conf import settings
from .models import LawG, Law3, DecisionMetadata
from django.db.models import Q
from googleapiclient.discovery import build
from trafilatura import fetch_url, extract 
from urllib.parse import urlparse
import faiss

# --- კონფიგურაცია ---
logger = logging.getLogger('operations')

# განახლებული ტარიფები (GPT-5.2)
PRICING_5_2 = {
    "input": 1.75,   # $ per 1M tokens
    "cached": 0.175, # $ per 1M tokens (10% of input)
    "output": 14.00  # $ per 1M tokens
}

def calculate_cost(completion_token=0, prompt_token=0, cached_token=0):
    input_cost = (prompt_token - cached_token) * (PRICING_5_2["input"] / 1000000)
    cached_cost = cached_token * (PRICING_5_2["cached"] / 1000000)
    output_cost = completion_token * (PRICING_5_2["output"] / 1000000)
    return input_cost + cached_cost + output_cost

# --- ახალი მთარგმნელი ფუნქცია (GPT-5.2) ---
def translate_text_openai(text, target_lang='ka'):
    """
    თარგმნის ტექსტს მითითებულ ენაზე (default: ქართული) GPT-5.2-ის გამოყენებით.
    """
    if not text: return ""
    
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    try:
        system_prompt = f"You are a professional legal translator. Translate the following text to {target_lang} accurately. Do not add explanations, just return the translation."
        
        response = client.chat.completions.create(
            model="gpt-5.2", # ვიყენებთ 5.2-ს
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        
        translation = response.choices[0].message.content.strip()
        
        # ხარჯის დათვლა არ დაგვავიწყდეს (თუმცა ეს ფუნქცია ასისტენტის გარეშეა, 
        # views.py-ში შეიძლება დაგჭირდეს ხარჯის მიმატება თუ მომავალში დააპირებ)
        return translation
        
    except Exception as e:
        logger.error(f"Translation Error: {e}")
        return text # შეცდომის შემთხვევაში ვაბრუნებთ ორიგინალს


def refine_text_openai_LAW(chunks, query, file_summary):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # ჩანკების მომზადება
    if chunks:
        chunk_list = "\n\n".join([f"ნაწყვეტი {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])
    else:
        chunk_list = "ინფორმაცია ვერ მოიძებნა."

    # ფაილის სამარის მომზადება
    summary_str = ""
    if file_summary:
        summary_str = "\n".join([f"{f}: {s}" for f, s in file_summary.items()])

    # --- მთავარი სისტემური პრომპტი (ქართულად) ---
    system_prompt = """
შენ ხარ გამოცდილი ქართველი იურისტი. შენი ამოცანაა გასცე ზუსტი, სამართლებრივად გამართული პასუხი საქართველოს კანონმდებლობაზე დაყრდნობით.

ინსტრუქციები:
1. პასუხი გაეცი მხოლოდ მოწოდებული "კონტექსტის" (Context) საფუძველზე.
2. აუცილებლად მიუთითე მუხლის ნომრები და კანონის დასახელება (მაგ: საქართველოს სამოქალაქო კოდექსის 99-ე მუხლი).
3. არ გამოიგონო მუხლები. თუ კონტექსტში ზუსტი პასუხი არ არის, თქვი: "მოწოდებულ მასალებში ზუსტი მუხლი არ იძებნება, თუმცა ზოგადი პრინციპით..."
4. ენა: უპასუხე იმ ენაზე, რა ენაზეც დასმულია კითხვა (უმეტესად ქართულად). თუ მომხმარებელი ინგლისურად გწერს, უპასუხე ინგლისურად.
5. ტონი: პროფესიონალური, თავაზიანი, დამაჯერებელი.
"""

    user_content = f"""
კითხვა: {query}

დამატებითი ფაილების მიმოხილვა:
{summary_str}

კონტექსტი (ნაპოვნი კანონები):
{chunk_list}

გთხოვ, ჩამოაყალიბო პასუხი.
"""

    response = client.chat.completions.create(
        model="gpt-5.2", # ვიყენებთ 5.2-ს
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )

    cost = calculate_cost(
        response.usage.completion_tokens,
        response.usage.prompt_tokens,
        response.usage.prompt_token_details.cached_tokens
    )
    
    return (response.choices[0].message.content, cost)


def refine_text_openai_DECISION(chunks, query):
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

    # გადაწყვეტილებების ფორმატირება
    formatted_cases = ""
    for c in chunks:
        formatted_cases += f"""
---
საქმე #: {c['decision_number']}
თარიღი: {c['decision_date']}
შინაარსი: {c.get('decision_motivation', 'N/A')} 
გადაწყვეტილება: {c['decision_established']}
---
"""
# შენიშვნა: ზევით get-ში დავამატე motivation, რადგან ქვემოთ ძებნაში motivation-ს ვიღებთ

    system_prompt = """
შენ ხარ ქართველი იურისტი, სპეციალიზებული სასამართლო პრაქტიკაში.
შენი მიზანია დააკავშირო მომხმარებლის კითხვა უზენაესი სასამართლოს პრეცედენტებთან.

ინსტრუქციები:
1. გააანალიზე მოწოდებული "სასამართლო პრაქტიკა".
2. თუ საქმეები რელევანტურია, მოიყვანე ისინი არგუმენტად (მიუთითე საქმის ნომერი).
3. თუ საქმეები არარელევანტურია, თქვი პირდაპირ.
4. პასუხი გაეცი იმ ენაზე, რა ენაზეც არის კითხვა.
5. სტრუქტურა:
   - შესავალი (პოზიცია)
   - რელევანტური პრეცედენტები (ჩამონათვალი)
   - დასკვნა
"""

    response = client.chat.completions.create(
        model="gpt-5.2", # ვიყენებთ 5.2-ს
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"კითხვა: {query}\n\nსასამართლო პრაქტიკა:\n{formatted_cases}"}
        ]
    )

    cost = calculate_cost(
        response.usage.completion_tokens,
        response.usage.prompt_tokens,
        response.usage.prompt_token_details.cached_tokens
    )
    return (response.choices[0].message.content, cost)


# --- Google Search & Search Utils ---

BLACKLISTED_DOMAINS = {
    'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com', 'tiktok.com',
    'reddit.com', 'quora.com', 'pinterest.com'
}

def get_response_from_Google_CS_API(query: str, filetype: str = None, num_results_to_find: int = 3) -> str:
    """
    Google Search + Scraping (ლიმიტი: 3 შედეგი)
    """
    api_key = os.getenv('GOOGLE_CUSTOM_SEARCH_API')
    cse_id = os.getenv('SEARCH_ENGINE_ID')
    
    if not api_key or not cse_id:
        return "System Error: Search API keys missing."

    # 1. Google API Call
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        q_args = {'q': query, 'cx': cse_id, 'num': 10} 
        if filetype: q_args['fileType'] = filetype
        
        res = service.cse().list(**q_args).execute()
        items = res.get('items', [])
    except Exception as e:
        logger.error(f"Google API Error: {e}")
        return "ინფორმაციის მოძიება ვერ მოხერხდა."

    # 2. Filtering & Scraping
    search_data = []
    
    for item in items:
        link = item.get('link')
        title = item.get('title')
        
        # Domain Filter
        try:
            domain = urlparse(link).netloc
            if any(bl in domain for bl in BLACKLISTED_DOMAINS): continue
        except: continue

        # Scraping
        try:
            downloaded = fetch_url(link)
            if not downloaded: continue
            
            text = extract(downloaded, include_comments=False, include_tables=False)
            if text and len(text) > 200:
                search_data.append(f"წყარო: {title} ({link})\nშინაარსი: {text[:2000]}")
                
                if len(search_data) >= num_results_to_find: 
                    break
        except:
            continue

    if not search_data:
        return "ინტერნეტში რელევანტური, ღია ინფორმაცია ვერ მოიძებნა."

    return "\n\n---\n\n".join(search_data)


def search_decisions_by_keywords(user_query="", category="civil"):
    """
    ეძებს FAISS-ში და აბრუნებს ტოპ 3 შედეგს.
    """
    try:
        closest_indexes = GetClosestCaseUsingFaiss(query=user_query, index_folder=category.lower(), top_n=1)
        
        if not closest_indexes: return []

        index_names = [x[0][:-7] if category.lower() in ["administrative", "civil"] else x[0][:-6] for x in closest_indexes]
        
        # გასწორებულია: crux-ის მაგივრად ვიღებთ decision_motivation-ს
        decisions = DecisionMetadata.objects.filter(decision_number__in=index_names).values(
            "decision_number", "decision_date", "decision_motivation", "decision_established"
        )[:3] 
        
        return list(decisions)
    except Exception as e:
        logger.error(f"Search Decisions Error: {e}")
        return []

# --- Helper Functions (Embeddings, Translate, etc.) ---

def get_openai_embedding(query):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.embeddings.create(input=query, model="text-embedding-3-large")
    return np.array(response.data[0].embedding, dtype=np.float32)

def GetClosestCaseUsingFaiss(query="", index_folder="", top_n=3):
    query_vector = get_openai_embedding(query).reshape(1, -1)
    folder_path = os.path.join(settings.BASE_DIR, "faiss_indexes_precedents", index_folder.lower())
    
    if not os.path.exists(folder_path): return []

    distances = []
    for fname in os.listdir(folder_path):
        if fname.endswith('.index'):
            index = faiss.read_index(os.path.join(folder_path, fname))
            dist, _ = index.search(query_vector, 1)
            distances.append((fname, dist[0][0]))
    
    distances.sort(key=lambda x: x[1])
    return distances[:top_n]