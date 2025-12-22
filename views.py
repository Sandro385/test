from django.views.decorators.csrf import csrf_exempt
from scrapper.search_query_processor import user_search_query_processor, create_userquery_embedding, convert_data_into_chunks, text_embedding, prepair_vector, get_user_friendly_query_answer, get_all_text
from rest_framework.exceptions import AuthenticationFailed, NotFound
from .scrap_utils import *
from .assistant import assistant_get_response
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from rest_framework.response import Response
from selenium.webdriver.common.by import By
from rest_framework.views import APIView
from rest_framework import status
from selenium import webdriver
from bs4 import BeautifulSoup
from openai import OpenAI
import time, os, re, uuid, jwt, warnings, string, re, faiss, logging, boto3
from pdf2image import convert_from_path
import pytesseract
from django.utils import timezone
warnings.filterwarnings("ignore", category=FutureWarning)
from dotenv import load_dotenv
from rest_framework.decorators import api_view
from django.http import JsonResponse
from .models import Law3, LawG, LawMetadata, LawIndexData, ChatHistory, RefactoredLaw
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from Users.models import UserPermission, UserPermissionChat
from .models import ChatHistory, link_category_name, FileInfo, DecisionMetadata , ChatSession

from .serializers import ChatHistorySerializer ,LawGSerializer, FileInfoSerializer
from django.conf import settings
from django.db import IntegrityError, transaction
from io import BytesIO
from langdetect.lang_detect_exception import LangDetectException
from django.core.mail import send_mail
import langid
import speech_recognition as sr
from pydub import AudioSegment
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from langdetect import detect, DetectorFactory
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
import requests
from datetime import datetime
from .models import RechargeTransaction

load_dotenv()

logger = logging.getLogger('operations')

@api_view(['GET'])
def scrap(request):
    """
    A REST API endpoint to perform web scraping across multiple pages for each subcategory and link stored in the database.
    """
    page_number=request.query_params.get("page_number")
    button_number=request.query_params.get("button_number",0)

    print(f"From request.query_params.get() page_number - {page_number} & button_number - {button_number}")
    try:
        # Fetch all subcategories and their respective links from the model
        link_objects = link_category_name.objects.all()

        for row in link_objects:
            category = row.sub_category
            link = row.link
            num_pages = row.num_pages
            print(f"category name : {category} pages : {num_pages}")

            if num_pages>0:
                for i in range(int(page_number), num_pages+1):
                    # Open the webpage for the current subcategory and page number
                    driver = open_webpage(f"{link}&page={i}", page_number=i, category=category, button=int(button_number))

                    # Introduce a delay to prevent overwhelming the server
                    time.sleep(5)

                    # Close the browser instance
                    driver.quit()
            else:
                print(f"Skipping category {category} number of pages : {num_pages}")

        # Return success response
        return JsonResponse({'message': 'Scraping complete!', 'status': 'success'}, status=200)

    except Exception as e:
        # Return error response in case of failure
        return JsonResponse({'message': str(e), 'status': 'error'}, status=500)
    

@api_view(['POST'])
def feedback(request, chat_id):
    """
    API to update or fetch feedback (like/dislike) for a specific chat history entry.
    """
    try:
    # Retrieve the chat history entry by chat_id

        # Update feedback based on the action provided
        action = request.data.get("action")
        print("action", action)
        ChatHistory.objects.filter(id=chat_id).update(feedback=action)

        

        # Generate a feedback message based on the updated state
        if action == "like":
            feedback_message = "Thank you for liking the response! Your feedback helps us improve."
        elif action == "dislike":
            feedback_message = "We appreciate your feedback! We'll use it to enhance the quality of our responses."
        else:
            feedback_message = "Invalid feedback action provided."

        return Response({"message": feedback_message}, status=status.HTTP_200_OK)
    except ChatHistory.DoesNotExist:
        return Response({"message": "Chat history not found."}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"message": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)





def open_webpage(url, page_number, category, button=1):
    # Set up Chrome options (no headless mode since you want to see it working)
    options = Options()
    options.add_argument("--no-sandbox")  # Needed for some environments
    options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource problems

    # Create a new instance of the Chrome driver
    service = Service('/usr/local/bin/chromedriver')  # Replace with the path to your ChromeDriver
    driver = webdriver.Chrome(service=service, options=options)
    
    # Open the webpage
    driver.get(url)
    
    links = driver.find_elements(By.CSS_SELECTOR, "div.panel-heading p a")
    print("total number of buttons : ", len(links))

    for i in range(button, min(21, len(links))):

        links[i].click()
        print(f"clicked on button number {i+1}", "page number : ",page_number)
        time.sleep(3)

        # registration_code = driver.find_element(By.XPATH, "//td[text()='Registration code']/following-sibling::td").text #english
        registration_code = driver.find_element(By.XPATH, "//td[text()='სარეგისტრაციო კოდი']/following-sibling::td").text # georgian
        # date_of_issue = driver.find_element(By.XPATH, "//td[text()='Date of issuing']/following-sibling::td").text #english
        date_of_issue = driver.find_element(By.XPATH, "//td[text()='მიღების თარიღი']/following-sibling::td").text #georgian

        response=scrape_page_data_with_bs4(driver.page_source, registration_code, date_of_issue, category)
        print(response)
        time.sleep(5)

        driver.find_element(By.CLASS_NAME, 'goback').click()
        print("clicked back button")
        print()
        time.sleep(5)

        links = driver.find_elements(By.CSS_SELECTOR, "div.panel-heading p a")

    return driver

def scrape_page_data_with_bs4(page_source, registration_code, date_of_issue, category):
    try:
        # Parse the page source with BeautifulSoup
        soup = BeautifulSoup(page_source, 'html.parser')

        paragraphs = soup.find_all('p')
        
        title_tag = soup.find('h1', class_='page-header')
        if title_tag:
            title = title_tag.get_text().replace('\n', ' ').strip()
        else:
            title = None
        
        scraped_data = [p.get_text() for p in paragraphs if p.get_text()!="\n"]

        print("law title : ",title)
        print("Registration code : ",registration_code)
        print("Date of issue : ",date_of_issue)
        print("category : ",category)

        full_text=""
        for i, text in enumerate(scraped_data):
            if i>4:
                full_text+=text

        cleaned_data=clean_parsed_data(full_text)

        # with open(f"{title}_1.txt", "w") as file:
        #     file.write(full_text)
        # with open(f"{title}_2.txt", "w") as file:
        #     file.write(cleaned_data)

        
        data = {
            "law_name": title,
            "law_description": cleaned_data,
            "registration_number": registration_code, 
            "category": category,
            "created_at": date_of_issue  
        }
        response = save_data(scraped_data = data)
        return response
        # return "true"

    except Exception as e:
        print(f"Error occurred while scraping: {str(e)}")
        return None
    


class FaissIndexAPI(APIView):
    def post(self, request, chunk_size=100):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Define the path for storing FAISS indexes
        index_dir = os.path.join(settings.BASE_DIR, "faiss_indexes")
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        # Fetch all unique laws from LawG
        laws = LawG.objects.all()[2016:]

        for law in laws:
            law_name = law.law_name
            print(f"Creating FAISS index for law: {law_name}")

            # Initialize FAISS index with IndexFlatL2 (Euclidean Distance) for each law
            dimension = 3072  # Adjust if necessary to match embedding dimension size
            index = faiss.IndexFlatL2(dimension)

            # Split law description into chunks based on token count
            text_chunks = split_text_by_tokens(law.law_description)
            text_chunks = [chunk for chunk in text_chunks if chunk.strip()]

            if not text_chunks:
                print(f"No valid text chunks for law: {law_name}")
                continue

            chunk_embeddings = []
            try:
                # Generate embeddings for the valid chunks
                for chunk in text_chunks:
                    chunk_with_law_name = f"Law Name: {law_name}\n{chunk}"
                    response = client.embeddings.create(
                        input=[chunk_with_law_name],
                        model="text-embedding-3-large"
                    )
                    chunk_embeddings.append(response.data[0].embedding)

                # Prepare FAISS vectors and store metadata
                vectors = []
                for i, (embedding, chunk) in enumerate(zip(chunk_embeddings, text_chunks)):
                    chunk_with_details = f"LAW NAME: {law_name}\nLAW REGISTRATION NUMBER: {law.registration_number}\n{chunk}"
                    
                    # Set vector_id with law.id and chunk index
                    vector_id = f"{law.id}-chunk{i}" 

                    # Store metadata in MySQL
                    try:
                        LawMetadata.objects.create(
                            vector_id=vector_id,
                            law_name=law_name,
                            registration_code=law.registration_number,
                            text=chunk_with_details
                        )
                    except IntegrityError:
                        LawMetadata.objects.filter(vector_id=vector_id).update(
                            law_name=law_name.strip(),
                            registration_code=law.registration_number,
                            text=chunk_with_details
                        )

                    # Convert embeddings to a numpy array and add to FAISS index
                    vector = np.array(embedding, dtype=np.float32)
                    vectors.append(vector)

                # Add vectors to FAISS index for this specific law
                if vectors:
                    index.add(np.array(vectors))

                # Save the index to the "faiss_indexes" folder with a unique name for each law
                index_name = f"faiss_index_law_id_{law.id}.index"
                index_path = os.path.join(index_dir, index_name)
                faiss.write_index(index, index_path)
                print(f"Index name : {index_name}")
                print("----------------------------------------------------------")

                # Store the index metadata in the law_index_data table
                LawIndexData.objects.create(
                    law_name=law_name,
                    index_name=index_name
                )

            except RuntimeError as e:
                return Response({'error': str(e)}, status=500)

        return Response({"message": "Data uploaded to FAISS and MySQL successfully!"}, status=200)


# Helper function to split text based on token count
def split_text_by_tokens(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=4096,
        chunk_overlap=200,
        length_function=len
    )
    # Split the extracted text into chunks
    chunks = splitter.split_text(text.strip())
    return chunks

class SimpleQueryAPI(APIView):
    GEORGIAN_STOPWORDS = [
        "და", "ეს", "რომ", "მას", "მე", "ის", "ისე", "ჩვენ", "თქვენ", "თქვენი", "შენ", 
        "მათ", "მის", "თქვენი", "იგი", "ც", "ა", "ან", "თუ", "არ", "როცა", "მაშინ"
    ]

    DAILY_LIMIT_REGISTERED = 999  # Max queries for registered users
    DAILY_LIMIT_UNREGISTERED = 5  # Max queries for unregistered users
    IS_GUEST_USER=False
    
    def post(self, request):
        # Extract token from the Authorization header
        token = request.headers.get('Authorization')
        file_ids = request.data.get("file_Ids",[])

        print(f"file ids : {file_ids}")
        # file_id
        # Extract device information
        device_info = request.data.get("deviceInfo", {})
        device_name = device_info.get("device_name")
        browser_type = None #device_info.get("browser_type")
        ip_address = device_info.get("ip_address")
        
        user_id = None
        save_chat = False 
        token_flag=False

        if token and token.startswith('Bearer ') and token!='Bearer null':
            print("Inside token block")
            token = token[7:].strip()
            if token.lower() != 'null':
                try:
                    decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
                    user_id = decoded_token.get('id')
                    if user_id:
                        save_chat = True
                        token_flag=True  
                    else:
                        logger.error('UserID is missing in the token')
                except jwt.ExpiredSignatureError:
                    logger.error('Token has expired')
                except jwt.InvalidTokenError:
                    logger.error('Invalid token')
                except Exception as e:
                    logger.exception('Error decoding token')
            else:
                token_flag=False
                logger.error('Token is null, authentication required')
        else:
            print("Inside guest block")
            if device_name and ip_address:
                try:
                    logger.info(f"Getting User ID for Guest user :-\nDevice name : {device_name}\nBrowser : {browser_type}\nIP Address:{ip_address}")
                    
                    user_permission = UserPermission.objects.filter(
                        device_name=device_name,
                        browser_type=browser_type,
                        ip_address=ip_address
                        ).first()

                    if user_permission is None:
                        logger.info(f"Executing method 2 for finding user")
                        user_permission = UserPermission.objects.filter(
                            device_name=device_name,
                            ip_address=ip_address,
                            email=None
                            ).first()
                    else:
                        logger.info(f"User found using method 1")

                    if user_permission:    
                        user_id = user_permission.id
                        logger.info(f"Guest user_id : {user_id} is querying")  
                    else:
                        logger.info(f"user not found using method 1 or 2")

                    save_chat = True
                    self.IS_GUEST_USER=True

                except UserPermission.DoesNotExist:
                    logger.error('User not found based on device information')
        
        if user_id:
            query_limit = self.DAILY_LIMIT_REGISTERED if token_flag else self.DAILY_LIMIT_UNREGISTERED

            if not self.has_query_limit(user_id, query_limit):
                logger.warning('Query limit reached for user')

                if self.IS_GUEST_USER:
                    return Response({
                        "error": "თქვენი უფასო დღიური ლიმიტი ამოიწურა. დარეგისტრირდით ახლა, რათა შეძლოთ დამატებითი მოთხოვნების გაგზავნა."
                    }, status=status.HTTP_429_TOO_MANY_REQUESTS)

            if token_flag:
                user_permission = UserPermission.objects.filter(id=user_id).first()
                if user_permission.email:
                    if user_permission.balance < 1:
                        logger.warning('Paid user with insufficient balance')
                        return Response({
                            "error": "თქვენი ყოველდღიური ბალანსი არ არის საკმარისი ამ მოთხოვნის დასამუშავებლად. გთხოვთ, შეავსოთ ბალანსი."
                        }, status=status.HTTP_429_TOO_MANY_REQUESTS)

        logger.info('Query allowed within free limit or as paid user')

        query = request.data.get('query')
        session_id = request.data.get('session_id')

        if not query:
            logger.error('Query is required')
            return Response({"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST)

        if session_id is None:
            logger.error('Session ID is required')
            return Response({"error": "Session ID is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Detect language
            try:
                api_key = os.getenv('OPENAI_API_KEY');
                language = langid.classify(query)
                detected_language_query =  language[0]
                print(f"detected_language: {detected_language_query}")
                logger.info(f"Detected language: {detected_language_query}")
                logger.info(f"API KEY:',{api_key}")

            except LangDetectException:
                logger.error("Could not detect language for query")
                return Response({"error": "Could not detect language for query"}, status=status.HTTP_400_BAD_REQUEST)

            thread_id=self.check_session_and_get_thread(session_id)
            
            if thread_id is None:
                thread_id=self.get_new_thread_id()

            # --- TRANSLATION LOGIC CHANGED HERE ---
            if detected_language_query == 'ka':
                query_processed = query
                original_q = None 
            else:
                # FIX: Using the new OpenAI translator function
                query_processed = translate_text_openai(query, target_lang='ka')
                original_q = query 

            response_from_assistant, response_type = assistant_get_response(thread_id=thread_id, query_input=query_processed, query_q=original_q, file_ids=file_ids) 
            
            print("response type : ", response_type)
            print("response_from_assistant : ",response_from_assistant)
            
            if response_from_assistant:
                response_text_raw = response_from_assistant[0]
                total_cost = response_from_assistant[1]
            else:
                response_text_raw = "Not found"
                total_cost = 0
                print("No response from assistant")
                logger.info("No response from assistant")

            user_permission = UserPermission.objects.filter(id = user_id).first()
            
            if token_flag:
                user_permission.update_balance(total_cost=total_cost)

            chat_history1 = ChatHistory.objects.filter(session_id=session_id, is_shared=True).order_by('-created_at')[:5]

            for chat in chat_history1:
                chat.is_shared = False
                chat.save()

            contact_string = "false" 
            
            length = len(response_from_assistant[0])
            print(length)
            if length >1000:
                contact_string ="true"
            
            response_text = response_text_raw
            source_name='წყარო' 
            
            cleaned_response = self.clean_response_text(response_text)
            if cleaned_response:
                response_text = cleaned_response

            
            if save_chat:
                urls_object = []
    
                chat_id = save_chat_history(
                    query=query, 
                    session_id=session_id,
                    thread_id=thread_id,
                    user_id=user_id,
                    response=response_text, 
                    file_ids=file_ids,
                    urls=urls_object ,
                    response_message=contact_string 
                )  
                logger.info(f"Chat saved with ID: {chat_id}")
            else:
                logger.info("Skipping save chat for now!")

            logger.info(f"Response generated for query: {query}")

            return Response(
            {
                "response": response_text,
                "DocumentLinks": urls_object, 
                "chat_id": chat_id if save_chat else None, 
                "lawyer_info": contact_string
            },
            status=status.HTTP_200_OK
            )

        except UserPermission.DoesNotExist:
            logger.error('User not found')
            return Response({'error': 'User not found'}, status=status.HTTP_404_NOT_FOUND)
        
    def clean_response_text(self, text):
        if isinstance(text, str):
            # Remove the last line if it contains only a period
            lines = text.splitlines()
            if lines and lines[-1].strip() == '.':
                lines = lines[:-1]
            text = "\n".join(lines)
            
            # Check if it ends with '? . .', keep '?' only
            if re.search(r'\?\.*$', text):
                cleaned_text = re.sub(r'\?\.*$', '?', text)
            else:
                # Replace multiple dots with a single dot at the end
                cleaned_text = re.sub(r'\.{2,}$', '.', text)
            return cleaned_text
        else:
            try:
                text = str(text)
                # Remove the last line if it contains only a period
                lines = text.splitlines()
                if lines and lines[-1].strip() == '.':
                    lines = lines[:-1]
                text = "\n".join(lines)
                
                # Repeat the same logic as above
                if re.search(r'\?\.*$', text):
                    cleaned_text = re.sub(r'\?\.*$', '?', text)
                else:
                    cleaned_text = re.sub(r'\.{2,}$', '.', text)
                return cleaned_text
            except Exception as e:
                print("Error:", e)
                logger.error(f"Error in clean_response_text method.")

    def preprocess_query(self, query):
        logger.info(f'Preprocessing query: {query}')
        query = query.translate(str.maketrans('', '', string.punctuation)).lower()
        processed_query = ' '.join([word for word in query.split() if word not in self.GEORGIAN_STOPWORDS])
        logger.info(f'Cleaned query: {processed_query}')
        return processed_query
    
    def check_session_and_get_thread(self, session_id):
        try:
            chat_history = ChatHistory.objects.filter(session_id=session_id).first()
            if chat_history:
                return chat_history.thread_id  # Return the thread_id if session_id exists
            else:
                return None
        except ChatHistory.DoesNotExist:
            return None  # Return None if session_id does not exist
    
    def get_new_thread_id(self):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        new_thread = client.beta.threads.create()
        return new_thread.id

    def has_query_limit(self, user_id, query_limit):
        # Get today's date with time set to midnight in timezone-aware format
        today_start = timezone.now().replace(hour=0, minute=0, second=0, microsecond=0)
        tomorrow_start = today_start + timezone.timedelta(days=1)

        # Count queries within today's date range
        query_count = ChatHistory.objects.filter(
            user_id=user_id, 
            created_at__gte=today_start, 
            created_at__lt=tomorrow_start,
        ).count()
        
        print(f"today_start -> {today_start}, user id is: {user_id}, query count is: {query_count}")
        return query_count < query_limit
    
    def get_chat_history(self, session_id):
        """Retrieve chat history for the given session ID."""
        history = ChatHistory.objects.filter(session_id=session_id).order_by('created_at')
        
        if not history.exists():
            # Return an empty list if no history exists
            return []

        # Create the structured chat history list
        # --- TRANSLATION LOGIC REMOVED HERE ---
        chat_history = []
        for h in history:
            chat_history.append({"role": "user", "content": h.query})
            chat_history.append({"role": "assistant", "content": h.response})

        return chat_history

    def summarize_chats(self, chats):
        """
        Summarizes the provided list of chats.
        """
        logger.info("Summarizing chats.")
        try:
            chat_contents = " ".join(f"{chat['role']}: {chat['content']}" for chat in chats)
            if chat_contents:    
                client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                summary_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": f"Please summarize the following conversation: {chat_contents}"}
                    ]
                )
                return summary_response.choices[0].message.content
            else:
                return []
        except Exception as e:
            self.logger.error("Error in summarize_chats: %s", str(e))
            raise
    
    def get_file_summary(self, file_ids):
        summaries = {}
        if file_ids:
            for file_id in file_ids:
                try:
                    # Fetch the file_info based on file_id
                    file_info = FileInfo.objects.get(id=file_id)
                    # Add the filename and its corresponding file_content_summary to the dictionary
                    summaries[file_info.file_name] = file_info.file_content_summary
                except FileInfo.DoesNotExist:
                    # If file is not found, add None for that file_id
                    summaries[file_id] = None
            return summaries
        else:
            return summaries

from pinecone.grpc import PineconeGRPC as Pinecone
class SearchQueryResult( APIView ):

    def get_user( request ):
        token = request.COOKIES.get( "json_access_token" )
        if not token:
            raise AuthenticationFailed( "User is not logged in" )
        try:
            payload = jwt.decode( token, os.getenv( 'SECRET_KEY' ), algorithms = [ "HS256" ] )
        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed( "Token has expired" )
        except jwt.InvalidTokenError:
            raise AuthenticationFailed( "Invalid token" )

        try:
            user = UserPermission.objects.filter( id = payload["id"] ).first( )
        except:
            raise NotFound( "User does not exist" ) 
        
        return user


    def post( self, request ):
        print( "in method--------------------" )
        query = request.data.get( "query" )
        if not query:
            return Response({"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        pc = Pinecone( api_key = os.getenv( 'PINECONE_API_KEY' ) )
        index = pc.Index( os.getenv( 'PINECONE_INDEX' ) )

        sessionID = request.data.get( "sessionid" )
        if not sessionID: 
            return Response( { "error" : "sessionID is needed" } )

        query_embedding = create_userquery_embedding( query = query )  # create vector embedding of user query
        scrapped_data = user_search_query_processor( query = query )   # scrapping relevent site ( given by GCSP ) 
        chunk_data = convert_data_into_chunks( scrapped_data, chunk_size = 1024, overlap = 0 )  # converting string data got by scrapping in to chunks of size 1024 using LangChain
        vector = prepair_vector( chunk_data = chunk_data )   # getting vectors to upsirt on pinecone
        
        # return Response( { "Openai response" : get_user_friendly_query_answer( 
        #     query,  
        #     "Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.[1] Quick progress in the field of deep learning, beginning in 2010s, allowed neural networks to surpass many previous approaches in performance.[2]ML finds application in many fields, including natural language processing, computer vision, speech recognition, email filtering, agriculture, and medicine.[3][4] The application of ML to business problems is known as predictive analytics.Statistics and mathematical optimization (mathematical programming) methods comprise the foundations of machine learning. Data mining is a related field of study, focusing on exploratory data analysis (EDA) via unsupervised learning.[6][7]From a theoretical viewpoint, probably approximately correct (PAC) learning provides a framework for describing machine learning.The term machine", 
        # ) } )
        
        # user = self.get_user( request )        
        # if not user:
        #     return Response( { "error" : "user not exist" } )
        
        try:
            upc = UserPermissionChat.objects.filter( sessionid__exact = str( sessionID ) )
            uniqueuuID = upc.uniqueuuid
            namespace = uniqueuuID
            user = upc.user
        except:
            uniqueuuID = str( uuid.uuid4( ) )
            namespace = uniqueuuID
            user = self.get_user( request ) 
            if not user:
                return Response( { "error" : "user not exist" } )
        finally:
            upc_obj = UserPermissionChat( 
                uniqueuuid = uniqueuuID,
                sessionid = str( sessionID ),
                user = user,
                query = query
            )

        # upserting all vectors in pinecone namespace
        batch_size = 200 
        for ite in range( 0, len( vector ), batch_size ):
            try:
                index.upsert( vectors = vector[ ite : ite + 200  ], namespace = namespace )
            except Exception as e:
                return Response( { 'error': "error while upserting vectors" } ) 

        # quering pinecone namespace - get response of the use query
        query_result = index.query(
            vector = query_embedding,
            top_k = 10,
            include_values = False,
            include_metadata = True,
            namespace = namespace
        )

        pinecone_retreved_text = [ { "text" : result['metadata']['text'] } for result in query_result[ 'matches' ] ]
        all_text = get_all_text( pinecone_retreved_text )
        meaningfull_text = get_user_friendly_query_answer( query, all_text )
        upc_obj.response = meaningfull_text
        upc.obj.save( )
        return Response( { "query result" : meaningfull_text } )        
        # return Response( { "len of chunk" : str( type( chunk_data ) ), "size of each chunk" : len( chunk_data[ -3 ] ), "chunk_data" : chunk_data  } )

from django.db.models import Min, F, Subquery, OuterRef
class ChatSessionView(APIView):
    def get(self, request):

        session_id = request.query_params.get('session_id')
        if session_id:
            # Fetch chat history for shared chat
            chat_history = ChatHistory.objects.filter(session_id=session_id)
            if chat_history.exists():
                serializer = ChatHistorySerializer(chat_history, many=True)
                return Response(serializer.data, status=status.HTTP_200_OK)
            else:
                return Response({"error": "No chat history found for the given session_id"}, status=status.HTTP_404_NOT_FOUND)
            
        # Extract token from the Authorization header
        token = request.headers.get('Authorization')
        user_id = None

        if token and token.startswith('Bearer '):
            token = token[7:].strip()  # Remove "Bearer " from the token and strip extra spaces

            # Check if the token is 'null' after trimming
            if token.lower() == 'null':
                return Response({'error': 'Token is null, authentication required'}, status=status.HTTP_401_UNAUTHORIZED)

            try:
                # Decode the token (Assuming it's a JWT token)
                decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
                user_id = decoded_token.get('id')

                if not user_id:
                    return Response({'error': 'UserID is missing in the token'}, status=status.HTTP_400_BAD_REQUEST)

            except jwt.ExpiredSignatureError:
                # Token expired
                return Response({'error': 'სესიის ვადა ამოიწურა'}, status=status.HTTP_401_UNAUTHORIZED)
            except jwt.InvalidTokenError:
                # Invalid token
                return Response({'error': 'არასწორი ჟეტონი'}, status=status.HTTP_401_UNAUTHORIZED)
            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        else:
            return Response({'error': 'Authorization header is missing or malformed'}, status=status.HTTP_400_BAD_REQUEST)

        # Query to get filenames for all file_ids
        def get_filenames(file_ids):
            return list(FileInfo.objects.filter(id__in=file_ids).values_list('file_name', flat=True))

        # Fetching chat history specific to the user_id
        file_sessions = ChatHistory.objects.filter(user_id=user_id).values(
            'session_id', 'file_id'
        ).annotate(
            created_at=Min('created_at'),
            chat_name=Subquery(
                ChatHistory.objects.filter(session_id=OuterRef('session_id'))
                .order_by('created_at')
                .values('query')[:1]
            )
        )

        sessions_with_filenames = []
        for session in file_sessions:
            file_ids = session['file_id']
            filenames = get_filenames(file_ids) if isinstance(file_ids, list) else []
            session['file_names'] = filenames
            sessions_with_filenames.append(session)

        # Sorting by 'created_at'
        all_sessions = sorted(sessions_with_filenames, key=lambda x: x['created_at'], reverse=True)

        # Step 3: Fetch chat session details from the database
        session_ids = [session['session_id'] for session in all_sessions]
        chat_sessions = ChatSession.objects.filter(session_id__in=session_ids)

        # Step 4: Build a dictionary for chat session details
        chat_session_map = {
            chat_session.session_id: {
                "session_name": chat_session.session_name,
                "is_deleted": chat_session.is_deleted,
            }
            for chat_session in chat_sessions
        }

        # Step 5: Remove duplicates and update 'chat_name'
        unique_sessions = {}
        for session in all_sessions:
            session_id = session['session_id']

            # Skip if the session is marked as deleted
            if session_id in chat_session_map and chat_session_map[session_id]['is_deleted']:
                continue

            # Update 'chat_name' if the session exists in the chat session table
            if session_id in chat_session_map:
                session['chat_name'] = chat_session_map[session_id]['session_name']

            if session_id not in unique_sessions:
                unique_sessions[session_id] = session

        # Step 6: Convert to list to maintain ordering
        result = list(unique_sessions.values())

        return Response(result, status=status.HTTP_200_OK)

def extract_urls(text):
    """Extract URLs from the given text using regex."""
    # URL pattern to match both http and https
    URL_REGEX = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(URL_REGEX, text)
    
    # Clean up any trailing unwanted characters like closing parentheses, periods, commas, etc.
    clean_urls = []
    for url in urls:
        # Remove trailing characters that are not part of a URL (like closing parentheses, periods, commas)
        clean_url = re.sub(r'[^\w\s:/.-]$', '', url)
        clean_urls.append(clean_url)
    
    return clean_urls


from datetime import datetime 
from datetime import timezone as timezone_1

class ChatHistoryListView(APIView):
    def format_chat_history(self, chat_data):
        formatted_data = []
        processed_files = {}  # Dictionary to store the first occurrence of each file by file_id

        for record in chat_data:
            # Extract URLs from the query and response
            user_document_links = extract_urls(record['query'])
            user_document_links = [{'id': i + 1, 'url': link} for i, link in enumerate(user_document_links)]
            
            bot_record_response=record['response']
            if record['urls']:  # Check if 'urls' is not None and not empty
                bot_document_links = [{'id': i + 1, 'url': url['url']} for i, url in enumerate(record['urls'])]
            else:
                bot_document_links = []
            # Fetch file information for the record
            file_info_entries = []
            if record.get('file_id'):  # Check if file_id is present
                file_ids = record['file_id']
                file_info_records = FileInfo.objects.filter(id__in=file_ids).order_by('created_at')

                # Format file information
                for file_info in file_info_records:
                    if file_info.id not in processed_files:
                        # Add the file only if it's not already processed
                        processed_files[file_info.id] = {
                            "session_id": record['session_id'],
                            "sender": "File",
                            "file_id": file_info.id,
                            "file_type": file_info.file_type,
                            "filename": file_info.file_name,
                            "timestamp": file_info.created_at,
                        }
                        file_info_entries.append(processed_files[file_info.id])

            # Format user query and bot response
            formatted_data.extend([
                {
                    "session_id": record['session_id'],
                    "text": record['query'],
                    "sender": "user",
                    "timestamp": record['created_at'],  # Assume `record['created_at']` is a datetime object
                    "DocumentLinks": user_document_links,
                    "files": []
                },
                {
                    "session_id": record['session_id'],
                    "text": bot_record_response,
                    "sender": "bot",
                    "timestamp": record['created_at'],
                    "DocumentLinks": bot_document_links,
                    "files": [],
                    "chat_id": record['id'],
                    "feedback": record['feedback'],
                    "lawyer_info":record['response_message']
                },
            ])

            # Add unique file entries to the data
            formatted_data.extend(file_info_entries)

        # Convert all timestamps to UTC-aware datetime objects
        for entry in formatted_data:
            if isinstance(entry['timestamp'], str):
                # Parse ISO 8601 string with 'Z' suffix as UTC time
                entry['timestamp'] = datetime.strptime(entry['timestamp'], "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone_1.utc)
            elif entry['timestamp'].tzinfo is None:
                # Add UTC timezone to naive datetime objects
                entry['timestamp'] = entry['timestamp'].replace(tzinfo=timezone_1.utc)

        # Sort all entries by timestamp
        formatted_data.sort(key=lambda x: x['timestamp'])

        return formatted_data


    def get(self, request):
        session_id = request.query_params.get('session_id')

        if not session_id:
            return Response({"error": "session_id parameter is required."}, status=status.HTTP_400_BAD_REQUEST)

        chat_history = ChatHistory.objects.filter(session_id=session_id)
        if not chat_history.exists():
            return Response({"error": f"No chat history found for session_id {session_id}."}, status=status.HTTP_404_NOT_FOUND)

        try:
            serializer = ChatHistorySerializer(chat_history, many=True)
            formatted_chat_history = self.format_chat_history(serializer.data)
            return Response(formatted_chat_history, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class ConcatenateLawAPIView(APIView):
    def get(self, request):
        # Group by law_name and select the minimum created_at timestamp
        law_data = (
            Law3.objects
            .values('law_name')
            .annotate(
                earliest_created_at=Min('created_at')
            )
        )

        saved_laws = []
        for entry in law_data:
            # Filter entries with the same law_name
            same_name_entries = Law3.objects.filter(law_name=entry['law_name'])
            
            # Concatenate registration numbers and descriptions
            registration_number = ''.join(same_name_entries.values_list('registration_number', flat=True))
            law_description = ''.join(same_name_entries.values_list('law_description', flat=True))
            law_name = entry['law_name']
            created_at = entry['earliest_created_at']
            law_category = "All"  # Set law_category to 'All'

            # Use update_or_create to avoid duplicates
            law_g, created = LawG.objects.update_or_create(
                law_name=law_name,
                defaults={
                    'registration_number': registration_number,
                    'law_description': law_description,
                    'law_category': law_category,
                    'created_at': created_at
                }
            )
            saved_laws.append(law_g)

        # Serialize and return the saved objects
        serializer = LawGSerializer(saved_laws, many=True)
        return Response(serializer.data, status=status.HTTP_201_CREATED)



class FileUploadView(APIView):
    allowed_file_types = ['.txt', '.pdf', '.jpg', '.jpeg', '.png', '.gif', '.docx', '.mp3', '.wav']

    def post(self, request):
            files = request.FILES.getlist('files')  # Get multiple files from request
            session_id = request.data.get('session_id')

            token = request.headers.get('Authorization')

            device_info={"device_name" : request.data.get('device_name'),
                        "browser_type" : None, #request.data.get('browser_type'),
                        "ip_address" : request.data.get('ip_address')
                        }

            print('device info : ',device_info, type(device_info))

            user_id=get_user_id(token,device_info)
            print("user id is : ", user_id)

            if not files:
                logger.error("No files uploaded")
                return Response({"error": "No files uploaded"}, status=status.HTTP_400_BAD_REQUEST)

            file_info_list = []  # To store metadata of all uploaded files
            failed_uploads = []  # To track any files that failed to upload

            # S3 client
            s3_client = boto3.client('s3',
                                    aws_access_key_id=os.getenv('S3_ACCESS_KEY_ID'),
                                    aws_secret_access_key=os.getenv('S3_ACCESS_SECRET_KEY'))

            # Begin a transaction to ensure atomicity
            try:
                with transaction.atomic():
                    for file in files:
                        # Validate file extension
                        file_extension = os.path.splitext(file.name)[1].lower()

                        if file_extension not in self.allowed_file_types:
                            logger.warning(f"Unsupported file type: {file_extension} for file {file.name}")
                            failed_uploads.append({"file_name": file.name, "error": f"Unsupported file type: {file_extension}. Allowed types: {', '.join(self.allowed_file_types)}."})
                            continue  # Skip to the next file

                        # Create a buffer from the uploaded file
                        file_buffer = BytesIO()
                        for chunk in file.chunks():
                            file_buffer.write(chunk)

                        file_buffer.seek(0, 2)  # Seek to the end of the buffer
                        file_size = file_buffer.tell()
                        file_buffer.seek(0)
                        file_type = self.get_file_type(file.name)
                        file_buffer.name = str(file.name)

                        if file_type=='Image':
                            image_text=extract_text_from_image(image_buffer=file_buffer)
                            text_file_buffer = BytesIO()
                            text_file_buffer.write(image_text[1].encode('utf-8'))
                            text_file_buffer.seek(0)
                            text_file_buffer.name = f"{file.name}.txt"
                            file_id_openai=file_upload_openai(file_buffer=text_file_buffer)
                        
                        elif file_type=='Audio':
                            audio_text=extract_text_from_audio(file_buffer=file_buffer)
                            print(f"audio text : {audio_text}")
                            audio_file_buffer = BytesIO()
                            audio_file_buffer.write(audio_text.encode('utf-8'))
                            audio_file_buffer.seek(0)
                            audio_file_buffer.name = f"{file.name}.txt"
                            file_id_openai=file_upload_openai(file_buffer=audio_file_buffer)
                            print(f"OpenAI file id : {audio_text}")
                        
                        else:
                            file_id_openai=file_upload_openai(file_buffer=file_buffer)
                        
                        # Save file metadata to the database
                        file_info = FileInfo(
                            file_name=file.name,
                            file_path='',  # Placeholder, will be updated after successful upload
                            file_type=file_type,
                            file_size=file_size,
                            session_id=session_id,
                            file_upload_user_id=user_id,
                            file_id_openai = file_id_openai
                        )
                        file_info.save()

                        # S3 upload
                        bucket = os.getenv("S3_BUCKET_NAME")
                        file_name_with_uid = get_unique_filename(file.name).replace(" ", "_")

                        logger.info(f"Uploading file: {file_name_with_uid} to bucket: {bucket}")
                        print(f"Uploading file: {file_name_with_uid} to bucket: {bucket}")
                        
                        # file_buffer.seek(0)
                        new_file_buffer=BytesIO()
                        for chunk in file.chunks():
                            new_file_buffer.write(chunk)
                        new_file_buffer.seek(0)
                        new_file_buffer.name = str(file.name)
                        
                        try:
                            s3_client.upload_fileobj(new_file_buffer, bucket, "Documents/" + file_name_with_uid, ExtraArgs={'ACL': 'public-read'})
                            presigned_url = f'https://{bucket}.s3.amazonaws.com/Documents/{file_name_with_uid}'
                            logger.info(f"File uploaded successfully: Documents/{file_name_with_uid}")

                        except Exception as e:
                            print(f"Error in uploa dto s3 : {e}")
                            presigned_url = ""
                            logger.error(f"Error : {e}")

                        # Update file path in the database
                        file_info.file_path = presigned_url           
                        file_info.save()

                        # Store the successful file info for response
                        file_info_list.append(file_info)
                        file_buffer.close()

            except Exception as e:
                logger.error(f"File upload failed: {e}")
                return Response({"error": "File upload process encountered an error"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

            # Prepare the response data
            if failed_uploads:
                return Response({"uploaded_files": FileInfoSerializer(file_info_list, many=True).data, "failed_uploads": failed_uploads}, status=status.HTTP_207_MULTI_STATUS)

            # Serialize the response if everything succeeds
            serializer = FileInfoSerializer(file_info_list, many=True)
            logger.info("File upload and embedding process completed.")
            return Response(serializer.data, status=status.HTTP_201_CREATED)


    # Method to determine the file type
    def get_file_type(self, file_name):
        ext = os.path.splitext(file_name)[1].lower()  # Get the file extension
        if ext in ['.pdf']:
            return 'PDF'
        elif ext in ['.jpg', '.jpeg', '.png', '.gif']:
            return 'Image'
        elif ext in ['.txt']:
            return 'Text File'
        elif ext in ['.docx']:
            return 'Word File'
        elif ext in ['.mp3', '.wav']:
            return 'Audio'
        else:
            return 'Unknown'
    

def get_unique_filename(file_name):
    """
    Appends a UUID to the file name to ensure uniqueness.
    """
    # Split the file name and its extension
    base_name, extension = os.path.splitext(file_name)
    
    # Generate a UUID
    unique_id = str(uuid.uuid4())
    
    # Return the new file name with the UUID appended
    return f"{base_name}_{unique_id}{extension}"


class FileInfoListView(APIView):

    def format_file_size(self, size_bytes):
        """Convert bytes to a human-readable file size string."""
        if size_bytes:
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1048576:  # 1024 * 1024
                return f"{size_bytes / 1024:.2f} kB"
            else:
                return f"{size_bytes / 1048576:.2f} MB"
        else:
            return "Null"

    def post(self, request):
        # Get the 'page' and 'page_size' from request query params (defaults to 1 and 10)Brian
        page = request.query_params.get('page', 1)
        page_size = request.query_params.get('page_size', 10)
        
        token = request.headers.get('Authorization')
        device_info = request.data.get("deviceInfo", {})
        user_id=get_user_id(token,device_info)

        logger.info(f"Fetching file information: page={page}, page_size={page_size}")

        try:
            page = int(page)
            page_size = int(page_size)
        except ValueError:
            logger.error("Invalid page or page_size: must be integers")
            return Response({"error": "page and page_size must be integers"}, status=status.HTTP_400_BAD_REQUEST)

        if page < 1 or page_size < 1:
            logger.warning("Page or page_size is not positive: page=%s, page_size=%s", page, page_size)
            return Response({"error": "page and page_size must be positive integers"}, status=status.HTTP_400_BAD_REQUEST)

        # Calculate the start and end indices
        offset = (page - 1) * page_size
        limit = offset + page_size

        # Fetch the required subset of records directly from the database using slicing
        files = FileInfo.objects.filter(is_deleted=False, file_upload_user_id=user_id).order_by('-id')[offset:limit]

        # If no files are found, return an empty result
        if not files.exists():
            logger.info("No files found for the requested page: %s", page)
            return Response({
                'total_pages': 0,
                'current_page': page,
                'page_size': page_size,
                'total_items': 0,
                'results': []
            })

        # Get the total count for pagination
        total_items = FileInfo.objects.filter(is_deleted=False, file_upload_user_id=user_id).count()
        total_pages = (total_items + page_size - 1) // page_size  # Ceiling division to calculate total pages

        # Serialize the fetched records
        serializer = FileInfoSerializer(files, many=True)

        # Format the file sizes in the serialized data
        for file_data in serializer.data:
            file_info = FileInfo.objects.get(id=file_data['id'])  # Get the FileInfo instance
            file_data['file_size'] = self.format_file_size(file_info.file_size)  # Format the file size
            file_data['file_id'] = str(file_info.id)

        logger.info("Successfully retrieved %d files for page %d", len(files), page)
        
        # Construct response with pagination details
        return Response({
            'total_pages': total_pages,
            'current_page': page,
            'page_size': page_size,
            'total_items': total_items,
            'results': serializer.data
        })

class SoftDeleteFileView(APIView):

    def patch(self, request):
        file_id = request.query_params.get('file_id')
        
        if not file_id:
            return Response({"error": "file_id parameter is required"}, status=status.HTTP_400_BAD_REQUEST)
        
        try:
            # Fetch the file object based on file_id
            file_info = FileInfo.objects.get(id=file_id)
            
            # Perform soft delete by setting is_deleted = True
            file_info.is_deleted = True
            file_info.save()
            
            # Serialize and return the updated object
            serializer = FileInfoSerializer(file_info)
            return Response(serializer.data, status=status.HTTP_200_OK)
        
        except FileInfo.DoesNotExist:
            # Return a 404 if the file is not found
            return Response({"error": "File not found"}, status=status.HTTP_404_NOT_FOUND)

class ProcessCourtDecisions(APIView):
    folders = ["Civil", "Administrative", "Criminal"]

    def create_faiss_index(self, dimension=3072):
        """Create a new FAISS index."""
        logger.info("Creating a new FAISS index with dimension %s", dimension)
        return faiss.IndexFlatL2(dimension)

    def generate_embeddings(self, text):
        """Generate embeddings using the OpenAI API."""
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-large"
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def post(self, request):
        """Process court decisions and create FAISS indexes."""

        category=request.query_params.get('category')
        if not category:
            return Response({'error':'category is required [civil, administrative, criminal]'},status=400)
        
        decisions=DecisionMetadata.objects.filter(decision_category__iexact=category.lower()).values("decision_number","decision_number","decision_description","decision_motivation")

        index_dir = os.path.join(settings.BASE_DIR, "faiss_indexes_precedents",category)
        if not os.path.exists(index_dir):
            os.makedirs(index_dir)

        loop_counter=1        
        for decision in decisions:
            index = self.create_faiss_index()
            logger.info("Created new FAISS index for %s", decision.get("decision_number",'LuLL'))
            print("Created new FAISS index for %s", decision.get("decision_number",'LuLL'))

            description=decision.get('decision_motivation','')
            motivation=decision.get('decision_description','')
            number=decision.get("decision_number",'LuLL')
            print(f"decision_number : {number}, description length : {len(description)}, motivation length : {len(motivation)}")
            logger.info(f"decision_number : {number}, description length : {len(description)}, motivation length : {len(motivation)}")
            
            chunks = split_text_by_tokens(description+'\n'+motivation)
            print(f"chunks created : {len(chunks)}")

            if not chunks:
                logger.warning("No valid text chunks for decision: %s", decision.get("decision_number",'LuLL'))
                print("No valid text chunks for decision: %s", decision.get("decision_number",'LuLL'))
                continue

            vectors = []
            try:
                for chunk in chunks:
                    embedding = self.generate_embeddings(chunk).reshape(1, -1)
                    vectors.append(embedding[0])
                print(f"embeddings created, {len(vectors)}")

                if vectors:
                    index.add(np.array(vectors))
                    logger.info("Added vectors to FAISS index for index %s", number)

                index_path = os.path.join(index_dir, f"{number}.index")
                faiss.write_index(index, index_path)
                logger.info("Index for decision %s saved after processing: %s", number, index_path)

            except RuntimeError as e:
                logger.error("RuntimeError while processing decision %s: %s", number, str(e))
                return Response({'error': str(e)}, status=500)
            print(f"iterations : {loop_counter}")
            loop_counter += 1

        logger.info("Processed decisions, generated embeddings successfully!")
        return Response({"message": "Processed decisions, generated embeddings successfully!"}, status=200)

def law_decisions(request):
    # Get the query string from the GET request
    query_string = request.GET.get('query', '')  # Default to empty string if no query is provided
    category = request.GET.get('category','')
    # Call the search_laws method with the received query
    search_precedents = search_decisions_by_keywords(user_query=query_string, category=category)
    
    # Return a JSON response with the result
    return JsonResponse({'result': search_precedents})

@api_view(['POST'])
def translate_law_names(request):
    # Get start and end from the query parameters
    start = request.GET.get('start', None)
    end = request.GET.get('end', None)

    # Validate that start and end are integers
    try:
        start = int(start) if start else 0
        end = int(end) if end else 10
        print(f"Start: {start}, End: {end}")

    except ValueError:
        return JsonResponse({'error': 'Start and end must be integers.'}, status=400)

    # Fetch the specified range of LawG objects, ordered by id
    laws = LawG.objects.all().order_by('id')[start:end]
    client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if not laws:
        return JsonResponse({'message': 'No laws found for the given range.'}, status=404)
    
    for i, law in enumerate(laws):
        print(f"current loop i - {i}")
        law_name_georgian = law.law_name
        
        # Make a request to the OpenAI API for translation
        try:
            prompt = [
    {
        "role": "user", 
        "content": f"""
Give the following Georgian law name in English : 
{law_name_georgian}
Instructions : 
1. Just return the law name in English, nothing else.
""" 
    }]
            response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=prompt
                    )
            law_name_english = response.choices[0].message.content
    
        except Exception as e:
            print(f"An error occurred: {e}")
            law_name_english = ""
            
            # Save the translation in the law_name_eng field
        law.law_name_eng = law_name_english
        law.save()

    return JsonResponse({'message': 'Law names translated successfully!'})


import pandas as pd
class FaissIndexDecisions(APIView):
    def post(self, request):
        try:
            # CSV file path (hardcoded as per request)
            csv_path = os.path.join(settings.BASE_DIR, "court_decisions/criminal_327_eng.csv")

            # Directory for storing FAISS indexes
            index_dir = os.path.join(settings.BASE_DIR, "faiss_indexes")

            if not os.path.exists(index_dir):
                os.makedirs(index_dir)

            # Validate CSV path
            if not csv_path or not os.path.exists(csv_path):
                return JsonResponse({"error": "Invalid or missing CSV path"}, status=400)

            # Read the CSV file
            data = pd.read_csv(csv_path)
            required_columns = ['subject', 'number', 'date', 'description', 'motivation', 'established']
            if not all(col in data.columns for col in required_columns):
                return JsonResponse({"error": f"CSV must contain {required_columns}"}, status=400)

            # Initialize OpenAI client
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                return JsonResponse({"error": "OpenAI API key is not set"}, status=500)
            client = OpenAI(api_key=openai_api_key)

            # Process each row in the CSV
            for index, row in data.iterrows():
                decision_name = " "
                number = row["number"]
                date = row["date"]
                established = row["established"]
                description = row["description"]
                motivation = row["motivation"]

                # Concatenate description and motivation
                metadata_text = f"{description} {motivation}"

                print(f"Processing Decision number: {index} (Number: {number})")

                # Initialize FAISS index for each decision
                dimension = 3072  # Assume OpenAI embedding model dimension
                index_name = f"faiss_index_{number}.index"
                index_path = os.path.join(index_dir, index_name)

                # Create a new FAISS index
                faiss_index = faiss.IndexFlatL2(dimension)

                # Split metadata text into chunks
                text_chunks = split_text_by_tokens(metadata_text)
                if not text_chunks:
                    print(f"No valid text chunks for decision number: {number}")
                    continue

                # Generate embeddings for chunks
                chunk_embeddings = []
                for chunk in text_chunks:
                    try:
                        response = client.embeddings.create(
                            input=chunk,
                            model="text-embedding-3-large"
                            )
                        chunk_embeddings.append(response.data[0].embedding)
                    except Exception as e:
                        print(f"Error generating embedding: {e}")
                        continue

                # Add vectors to FAISS index and store metadata
                vectors = []
                for i, (embedding, chunk) in enumerate(zip(chunk_embeddings, text_chunks)):
                    vector_id = f"{number}_chunk{i}"
                    try:
                        # Store metadata in the database
                        DecisionMetadata.objects.update_or_create(
                            vector_id=vector_id,
                            defaults={
                                "decision_name": decision_name,
                                "decision_number": number,
                                "decision_date": date,
                                "decision_established": established,
                                "decision_category": "Criminal",
                                "decision_description": chunk,
                                "decision_index_name": index_name,
                            },
                        )

                        # Add vector to FAISS index
                        vector = np.array(embedding, dtype=np.float32)
                        vectors.append(vector)
                    except IntegrityError:
                        print(f"Error saving metadata for vector_id: {vector_id}")
                        continue

                if vectors:
                    # Add all vectors to the FAISS index
                    faiss_index.add(np.array(vectors))

                    # Save FAISS index to file
                    faiss.write_index(faiss_index, index_path)
                    print(f"FAISS index saved at: {index_path}")

            return JsonResponse({"message": "FAISS indices and metadata created successfully."}, status=201)

        except Exception as e:
            print(f"Error: {e}")
            return JsonResponse({"error": str(e)}, status=500)
        


@api_view(['POST'])
def send_chat_email(request, chat_id):
    """
    API to send query, response, and optionally feedback via email.
    """
    try:
        # Extract the token from the Authorization header
        token = request.headers.get('Authorization')
        user_email = "Guest User"  # Default email for guest users

        # Decode the token if it exists and is valid
        if token and token.startswith('Bearer '):
            token = token[7:].strip() 
            if token.lower() != 'null': # Remove "Bearer " from the token
                try:
                    decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
                    user_id = decoded_token.get('id')

                    if user_id:
                        # Fetch the user's email if user_id exists
                        # user = User.objects.get(id=user_id)
                        user=UserPermission.objects.get(id=user_id)
                        user_email = user.email  # Get user email
            
                except jwt.ExpiredSignatureError:
                    return Response({"message": "Token has expired."}, status=status.HTTP_401_UNAUTHORIZED)
                except jwt.InvalidTokenError:
                    return Response({"message": "Invalid token."}, status=status.HTTP_401_UNAUTHORIZED)
                except Exception as e:
                    return Response({"message": f"Error decoding token: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve chat history using chat_id
        chat_history = ChatHistory.objects.get(id=chat_id)

        # Extract user-provided feedback
        feedback = request.data.get("feedback", "None")

        # Prepare email content
        query = chat_history.query
        response = chat_history.response

        email_subject = "User Feedback for Chat Query and Response"
        email_body = f"""
        <html>
        <body>
            <h2>Chat Query and Response</h2>
            <p><strong>Query:</strong> {query}</p>
            <p><strong>Response:</strong> {response}</p>
            <p><strong>Feedback:</strong> {feedback}</p>
            <p><strong>From:</strong> {user_email}</p>
        </body>
        </html>
        """

        # Send the email
        send_mail(
            subject=email_subject,
            message="",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[settings.DEFAULT_FROM_EMAIL], 
            #  recipient_list=[user_email], # Replace with your actual recipient email list
            fail_silently=False,
            html_message=email_body,
        )

        return Response({"message": "Email sent successfully."}, status=status.HTTP_200_OK)

    except ChatHistory.DoesNotExist:
        return Response({"message": "Chat history not found."}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        return Response({"message": f"An error occurred: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



@api_view(['POST'])
def technical_feedback(request):
    """
    API to accept technical feedback from the user.
    """
    try:
        # Extract the token from the Authorization header
        token = request.headers.get('Authorization')
        user_email = "Guest User"  # Default email for guest users

        # Decode the token if it exists and is valid
        if token and token.startswith('Bearer '):
            token = token[7:].strip()  # Remove "Bearer " from the token
            if token.lower() != 'null':
                try:
                    decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
                    user_id = decoded_token.get('id')

                    if user_id:
                        # Fetch the user's email if user_id exists
                        user = UserPermission.objects.get(id=user_id)
                        user_email = user.email  # Get user email
                except jwt.ExpiredSignatureError:
                    return Response({"error": "Token has expired."}, status=status.HTTP_401_UNAUTHORIZED)
                except jwt.InvalidTokenError:
                    return Response({"error": "Invalid token."}, status=status.HTTP_401_UNAUTHORIZED)
                except Exception as e:
                    return Response({"error": f"Error decoding token: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # Extract technical feedback from the request
        technical_feedback = request.data.get("feedback")

        if not technical_feedback:
            return Response({"error": "Technical feedback is required."}, status=status.HTTP_400_BAD_REQUEST)

        # Log or process the feedback (you can save it to a database if required)
        logger.info(f"Technical feedback received from {user_email}: {technical_feedback}")

        # Optionally, send the feedback via email to an admin/support team
        subject = "Technical Feedback"
        email_body = f"""
        <html>
        <body>
            <h2>Technical Feedback Received</h2>
            <p><strong>From:</strong> {user_email}</p>
            <p><strong>Feedback:</strong></p>
            <p>{technical_feedback}</p>
        </body>
        </html>
        """
        admin_email = settings.DEFAULT_FROM_EMAIL  # Replace with a specific admin/support email if necessary
        send_mail(
            subject=subject,
            message="",
            from_email=settings.DEFAULT_FROM_EMAIL,
            recipient_list=[settings.DEFAULT_FROM_EMAIL],  # Replace with your actual recipient email list
            html_message=email_body,
            fail_silently=False,
        )

        return Response({"message": "Technical feedback submitted successfully."}, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    
    
class ChatSessionActionView(APIView):
    def post(self, request):
        action = request.data.get('action')
        session_id = request.data.get('session_id')
        new_name = request.data.get('new_name', None)
        
        if not action or not session_id:
            return Response({"error": "Both 'action' and 'session_id' are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            chat_session, created = ChatSession.objects.get_or_create(session_id=session_id)

            if action == "delete":
                chat_session.is_deleted = True
                chat_session.save()
                return Response({"message": f"Session {session_id} marked as deleted."}, status=status.HTTP_200_OK)

            elif action == "rename":
                if not new_name:
                    return Response({"error": "'new_name' is required for renaming."}, status=status.HTTP_400_BAD_REQUEST)
                chat_session.session_name = new_name
                chat_session.save()
                return Response({"message": f"Session {session_id} renamed to '{new_name}'."}, status=status.HTTP_200_OK)

            else:
                return Response({"error": "Invalid action. Use 'delete' or 'rename'."}, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)



class LinkChatSessionsAPIView(APIView):
    def post(self, request):
        data = request.data
        receiver_token = request.headers.get('Authorization')
        receiver_session_id = data.get('session_id')
        new_session_id = data.get('new_session_id')
        device_info = data.get('device_info', {})
        
        if not receiver_session_id:
            return Response({"error": "Session ID is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # Extract receiver's user_id
            receiver_user_id = None
            if receiver_token and receiver_token.startswith('Bearer '):
                receiver_token = receiver_token[7:]
                try:
                    decoded_token = jwt.decode(receiver_token, settings.SECRET_KEY, algorithms=["HS256"])
                    receiver_user_id = decoded_token.get('id')
                except jwt.InvalidTokenError:
                    return Response({"error": "Invalid token."}, status=status.HTTP_401_UNAUTHORIZED)
            
            # Fallback to device info if token is not provided
            if not receiver_user_id:
                device_name = device_info.get("device_name")
                browser_type = None, #device_info.get("browser_type")
                ip_address = device_info.get("ip_address")
                
                # if device_name and browser_type and ip_address:
                if device_name and ip_address:
                    receiver_user = UserPermission.objects.filter(
                        device_name=device_name,
                        browser_type=browser_type,
                        ip_address=ip_address
                    ).first()
                    receiver_user_id = receiver_user.id if receiver_user else None

            if not receiver_user_id:
                return Response({"error": "Unable to determine receiver user ID."}, status=status.HTTP_401_UNAUTHORIZED)
            
            chat_history = ChatHistory.objects.filter(session_id=receiver_session_id)

            if not chat_history.exists():
                return Response({"error": "No chat history found for the given session ID."}, status=status.HTTP_404_NOT_FOUND)

            # Create a new session linked to the receiver
            for chat in chat_history:
                ChatHistory.objects.create(
                    session_id=new_session_id,
                    user_id=receiver_user_id,  # Use receiver's user ID
                    query=chat.query,
                    response=chat.response,
                    created_at=chat.created_at,
                    file_id=chat.file_id,
                    thread_id=chat.thread_id
        
                )

            # Optionally, create an entry in the ChatSession table to log the shared session
            # ChatSession.objects.create(session_id=new_session_id, session_name=f"Shared from {receiver_session_id}")

            return Response(
                {
                    "message": "Chat session shared successfully.",
                    "new_session_id": new_session_id
                },
                status=status.HTTP_200_OK
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        

from paypalrestsdk import Payment
import paypalrestsdk


paypalrestsdk.configure({
    # "mode": "sandbox", 
    "mode": os.getenv("mode"),  # Change to "live" for production
    "client_id": os.getenv("client_id"),
    "client_secret":os.getenv("client_secret"),
})

@api_view(['POST'])
def initiate_recharge(request):
    
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            raise AuthenticationFailed("Token is missing or malformed.")

        token = token[7:].strip()  # Remove "Bearer " from the token
        try:
            # Decode the JWT token
            decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            print(decoded_token)
            user_id = decoded_token.get('id')
            print("--------------\n",user_id)
        
            if not user_id:
                raise AuthenticationFailed("Invalid token. User ID not found.")

        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed("Token has expired.")
        except jwt.InvalidTokenError:
            raise AuthenticationFailed("Invalid token.")
        except UserPermission.DoesNotExist:
            raise AuthenticationFailed("User not found.")
        except Exception as e:
            raise AuthenticationFailed(f"Error decoding token: {str(e)}")

        amount = request.data.get('amount', 0.00)

        if not user_id or not amount:
            return Response({"error": "User ID and amount are required"}, status=400)

        if amount < 5:
            return Response({"error": "Minimum amount for recharge is 5"}, status=400)
        
        try:
            payment = paypalrestsdk.Payment({
                "intent": "sale",
                "payer": {
                    "payment_method": "paypal"
                },
                "redirect_urls": {
                    "return_url": "https://chkuiskolopi.ge/payment-success",
                    "cancel_url": "https://chkuiskolopi.ge/payment-cancel"
                    
                    # "return_url": "http://localhost:3000/payment-success",
                    # "cancel_url": "http://localhost:3000/payment-cancel"
                },
                "transactions": [{
                    "amount": {
                        "total": f"{amount:.2f}",
                        "currency": "USD"
                    },
                    "description": f"Recharge for User ID: {user_id}"
                }]
            })

            print(payment)

            if payment.create():
                # Save transaction in the database
                RechargeTransaction.objects.create(
                    user_id=user_id,
                    amount=amount,
                    currency="USD",
                    transaction_id=payment.id,
                    status="Pending"
                )

                # Get approval URL
                for link in payment.links:
                    if link.rel == "approval_url":
                        return Response({"approval_url": link.href}, status=200)

            return Response({"error": "Payment creation failed"}, status=500)

        except Exception as e:
            return Response({"error": str(e)}, status=500)
    

from rest_framework.response import Response
from rest_framework.decorators import api_view
import paypalrestsdk

@api_view(['POST'])
def payment_success(request):
    payment_id = request.data.get('paymentId')
    payer_id = request.data.get('payer_id')

    if not payment_id or not payer_id:
        return Response({"error": "Payment ID and Payer ID are required"}, status=400)

    try:

        existing_transaction = RechargeTransaction.objects.filter(transaction_id=payment_id, status="Completed").first()
        if existing_transaction:
            return Response({
                "message": "Transaction already processed",
                "new_balance": existing_transaction.user.balance
            }, status=200)
        
        # Find the payment using PayPal SDK
        payment = paypalrestsdk.Payment.find(payment_id)
        print(f"Payment ID: {payment_id}, Payer ID: {payer_id}")
        print(f"Payment State: {payment.state}")
        print("previous payment state", payment.state)

        # Execute payment if it's not already approved
        if payment.state == "created":
            if payment.execute({'payer_id': payer_id}):
                print("Payment executed successfully")
            else:
                print(f"Payment execution failed: {payment.error}")
                return Response({"error": "Payment execution failed", "details": payment.error}, status=400)
        print("payment state", payment.state)
        # Check if payment is approved
        if payment.state == "approved":
            try:
                # Update transaction status
                transaction = RechargeTransaction.objects.get(transaction_id=payment_id)
                if transaction.status != "Completed":  # Avoid duplicate updates
                    transaction.status = "Completed"
                    transaction.save()
                    print("Transaction status updated")

                    # Update user balance
                    user = transaction.user
                    user.update_balance(recharge_amount=transaction.amount)

                return Response({
                    "message": "Recharge successful",
                    "new_balance": user.balance
                }, status=200)

            except RechargeTransaction.DoesNotExist:
                return Response({"error": "Transaction not found"}, status=404)

        # Handle cases where payment is not approved
        return Response({"error": "Payment not approved"}, status=400)

    except paypalrestsdk.ResourceNotFound:
        return Response({"error": "Payment not found in PayPal"}, status=404)
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return Response({"error": "An unexpected error occurred", "details": str(e)}, status=500)


@api_view(['POST'])
def get_user_balance(request):
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            raise AuthenticationFailed("Token is missing or malformed.")

        token = token[7:].strip()  # Remove "Bearer " from the token
        try:
            # Decode the JWT token
            decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            print(decoded_token)
            user_id = decoded_token.get('id')
            print("=====",user_id)
        
            if not user_id:
                raise AuthenticationFailed("Invalid token. User ID not found.")

        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed("Token has expired.")
        except jwt.InvalidTokenError:
            raise AuthenticationFailed("Invalid token.")
        except UserPermission.DoesNotExist:
            raise AuthenticationFailed("User not found.")
        except Exception as e:
            raise AuthenticationFailed(f"Error decoding token: {str(e)}")

        try:
            # Fetch the user from the UserPermission model
            user_permission = UserPermission.objects.get(id=user_id)
            
            # Get the user's balance
            user_balance = user_permission.balance
            return Response({"balance": round(user_balance,2)}, status=200)

        except UserPermission.DoesNotExist:
            return Response({"error": "User not found"}, status=404)
        except Exception as e:
            return Response({"error": str(e)}, status=500)
        



#flitt integration


import hashlib
import uuid
import json
from datetime import datetime
import requests

@api_view(['POST'])
def initiate_flitt_recharge(request):
    try:
        # Extract and validate token
        token = request.headers.get('Authorization')
        if not token or not token.startswith('Bearer '):
            raise AuthenticationFailed("Token is missing or malformed.")

        token = token[7:].strip()
        try:
            decoded_token = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
            user_id = decoded_token.get('id')
            
            if not user_id:
                raise AuthenticationFailed("Invalid token. User ID not found.")

        except jwt.ExpiredSignatureError:
            raise AuthenticationFailed("Token has expired.")
        except jwt.InvalidTokenError:
            raise AuthenticationFailed("Invalid token.")

        # Get amount from request
        amount = int(request.data.get('amount', 0))
        
        if amount < 5:
            return Response({
                "error": "Minimum amount for recharge is 5 USD"
            }, status=400)

        print(f"\n{'='*60}")
        print(f"[FLITT] Creating payment")
        print(f"[FLITT] User ID: {user_id}")
        print(f"[FLITT] Amount: {amount} USD")
        print(f"{'='*60}\n")

        # Get Flitt credentials from environment
        merchant_id = os.getenv("FLITT_MERCHANT_ID", "4055111")
        secret_key = os.getenv("FLITT_API_KEY", "HhnKgzwVdtc8jp0JMrVBSjoBZF2Kzo4l")
        
        if not secret_key:
            return Response({
                "error": "Flitt secret key not configured"
            }, status=500)
        
        # Flitt API endpoint
        api_url = "https://pay.flitt.com/api/checkout/url"
        
        # Generate unique order ID
        timestamp = int(datetime.now().timestamp())
        order_id = f"order_{user_id}_{timestamp}"
        
        # ⚠️ FIXED: Amount MUST be in tetri (integer)!
        amount_cents = int(amount * 100)  # Convert USD to cent
        
        currency = "USD"
        order_desc = f"Account Recharge - User {user_id}"
        callback_url = "https://api.chkuiskolopi.ge/api/scrap/flitt/callback/"
        
        # Calculate signature using SHA1
        # Formula: SHA1(secret_key|amount|currency|merchant_id|order_id|callback_url)
        signature_string = f"{secret_key}|{amount_cents}|{currency}|{merchant_id}|{order_desc}|{order_id}|{callback_url}"
        signature = hashlib.sha1(signature_string.encode('utf-8')).hexdigest()
        
        print(f"[FLITT] Order ID: {order_id}")
        print(f"[FLITT] Amount: {amount} USD = {amount_cents} cent")
        print(f"[FLITT] Signature String: ****|{amount_cents}|{currency}|{merchant_id}|{order_desc}|{order_id}|{callback_url}")
        print(f"[FLITT] SHA1 Signature: {signature}")
        
        # Prepare request payload - USE amount (INTEGER)
        payload = {
            "request": {
                "server_callback_url": callback_url,
                "order_id": order_id,
                "currency": currency,
                "merchant_id": int(merchant_id),
                "order_desc": order_desc,
                "amount": amount_cents,  
                "signature": signature
            }
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        print(f"[FLITT] Request URL: {api_url}")
        print(f"[FLITT] Payload: {json.dumps(payload, indent=2)}\n")
        
        try:
            # Make API request to Flitt
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=15
            )
            
            print(f"[FLITT] Response Status: {response.status_code}")
            print(f"[FLITT] Response Body: {response.text}\n")
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if response is successful
                if data.get('response', {}).get('response_status') == 'success':
                    # Extract checkout URL
                    checkout_url = data.get('response', {}).get('checkout_url')
                    payment_id = order_id
                    
                    if not checkout_url:
                        print(f"[FLITT] ERROR: No checkout URL in response")
                        return Response({
                            "error": "No checkout URL received from Flitt",
                            "response": data
                        }, status=400)
                    
                    # Create transaction record
                    transaction = RechargeTransaction.objects.create(
                        user_id=user_id,
                        amount=amount_cents,
                        currency="USD",
                        transaction_id=payment_id,
                        status="Pending",
                        # provider="Flitt"
                    )
                    
                    print(f"[FLITT] SUCCESS!")
                    print(f"[FLITT] Transaction ID: {transaction.id}")
                    print(f"[FLITT] Order ID: {order_id}")
                    print(f"[FLITT] Checkout URL: {checkout_url}")
                    print(f"{'='*60}\n")
                    
                    return Response({
                        "success": True,
                        "payment_id": payment_id,
                        "order_id": order_id,
                        "checkout_url": checkout_url,
                        "transaction_id": transaction.id
                    })
                else:
                    # Flitt returned an error
                    error_msg = data.get('response', {}).get('error_message', 'Unknown error')
                    error_code = data.get('response', {}).get('error_code', 'N/A')
                    print(f"[FLITT] Flitt API Error: {error_code} - {error_msg}")
                    
                    return Response({
                        "error": f"Flitt Error {error_code}: {error_msg}",
                        "response": data
                    }, status=400)
                
        except requests.exceptions.ConnectionError as e:
            print(f"[FLITT] Connection Error: {str(e)}")
            return Response({
                "error": "Cannot connect to Flitt payment gateway"
            }, status=503)
            
        except requests.exceptions.Timeout:
            print(f"[FLITT] Timeout Error")
            return Response({
                "error": "Flitt payment gateway timeout"
            }, status=504)
            
        except requests.exceptions.RequestException as e:
            print(f"[FLITT] Request Error: {str(e)}")
            return Response({
                "error": f"Payment request failed: {str(e)}"
            }, status=500)

    except AuthenticationFailed as e:
        print(f"[FLITT] Auth Error: {str(e)}")
        return Response({"error": str(e)}, status=401)
        
    except Exception as e:
        print(f"[FLITT] Unexpected Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response({
            "error": f"Server error: {str(e)}"
        }, status=500)
        
        
         
        
@api_view(['POST'])
@csrf_exempt
@csrf_exempt
def flitt_payment_callback(request):
    """
    Handle Flitt payment callback/webhook with proper validation
    """
    try:
        print(f"\n{'='*60}")
        print(f"[FLITT CALLBACK] Received callback")
        print(f"[FLITT CALLBACK] Headers: {dict(request.headers)}")
        print(f"[FLITT CALLBACK] Body: {request.body.decode('utf-8')}")
        print(f"{'='*60}\n")
        
        callback_data = request.data
        
        # Extract callback parameters
        order_id = callback_data.get('order_id')
        order_status = callback_data.get('order_status', '').lower()
        response_status = callback_data.get('response_status', '').lower()
        transaction_id = callback_data.get('transaction_id')
        amount = callback_data.get('amount')  # Amount in tetri
        currency = callback_data.get('currency')
        received_signature = callback_data.get('signature')
        
        print(f"[FLITT CALLBACK] Order ID: {order_id}")
        print(f"[FLITT CALLBACK] Order Status: {order_status}")
        print(f"[FLITT CALLBACK] Response Status: {response_status}")
        print(f"[FLITT CALLBACK] Transaction ID: {transaction_id}")
        print(f"[FLITT CALLBACK] Amount: {amount}")
        
        if not order_id:
            print("[FLITT CALLBACK] ERROR: Missing order_id")
            return Response({"error": "Missing order_id"}, status=400)
        
        # CRITICAL: Verify signature to prevent fake callbacks
        secret_key = os.getenv("FLITT_API_KEY")
        if not secret_key:
            print("[FLITT CALLBACK] ERROR: Secret key not configured")
            return Response({"error": "Server configuration error"}, status=500)
        
        # According to Flitt docs, callback signature format is:
        # SHA1(secret_key|order_id|amount|currency|order_status)
        if received_signature:
            signature_string = f"{secret_key}|{order_id}|{amount}|{currency}|{order_status}"
            expected_signature = hashlib.sha1(signature_string.encode('utf-8')).hexdigest()
            
            print(f"[FLITT CALLBACK] Signature String: ****|{order_id}|{amount}|{currency}|{order_status}")
            print(f"[FLITT CALLBACK] Expected: {expected_signature}")
            print(f"[FLITT CALLBACK] Received: {received_signature}")
            
            if received_signature != expected_signature:
                print("[FLITT CALLBACK] ERROR: Invalid signature - possible fraud attempt!")
                return Response({"error": "Invalid signature"}, status=403)
            
            print("[FLITT CALLBACK] ✓ Signature verified")
        else:
            print("[FLITT CALLBACK] WARNING: No signature provided")
            # In production, you might want to reject callbacks without signatures
            return Response({"error": "Missing signature"}, status=400)
        
        # Find transaction
        try:
            transaction = RechargeTransaction.objects.get(
                transaction_id=order_id
            )
            
            print(f"[FLITT CALLBACK] Found transaction: {transaction.id}")
            print(f"[FLITT CALLBACK] Current status: {transaction.status}")
            
            # CRITICAL: Prevent duplicate processing
            if transaction.status == "Completed":
                print(f"[FLITT CALLBACK] Transaction already completed - ignoring callback")
                return Response({
                    "message": "Transaction already processed",
                    "status": "success"
                }, status=200)
            
            # CRITICAL: Only accept specific success statuses from Flitt
            # According to Flitt docs, successful payment has order_status = 'approved'
            if order_status == 'approved' and response_status == 'success':
                # Verify amount matches
                if amount and int(amount) != transaction.amount:
                    print(f"[FLITT CALLBACK] ERROR: Amount mismatch! Expected {transaction.amount}, got {amount}")
                    transaction.status = "Failed"
                    transaction.save()
                    return Response({"error": "Amount mismatch"}, status=400)
                
                # Mark as completed
                transaction.status = "Completed"
                transaction.save()
                
                # Update user balance
                user = transaction.user
                amount_usd = transaction.amount / 100  # Convert tetri to USD
                user.update_balance(recharge_amount=amount_usd)
                
                print(f"[FLITT CALLBACK] ✓ Payment completed!")
                print(f"[FLITT CALLBACK] ✓ User {user.id} balance updated: +${amount_usd:.2f}")
                print(f"[FLITT CALLBACK] ✓ New balance: ${user.balance:.2f}")
                
                return Response({
                    "message": "Payment processed successfully",
                    "status": "success"
                }, status=200)
            
            elif order_status in ['failed', 'declined', 'cancelled'] or response_status in ['failed', 'error']:
                transaction.status = "Failed"
                transaction.save()
                print(f"[FLITT CALLBACK] Payment failed/cancelled")
                return Response({"message": "Payment failed", "status": "failed"}, status=200)
            
            else:
                # Unknown status - log and wait
                print(f"[FLITT CALLBACK] Unknown status: {order_status}/{response_status}")
                return Response({"message": "Unknown status"}, status=200)
            
        except RechargeTransaction.DoesNotExist:
            print(f"[FLITT CALLBACK] ERROR: Transaction not found for order: {order_id}")
            return Response({"error": "Transaction not found"}, status=404)
            
    except Exception as e:
        print(f"[FLITT CALLBACK] ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response({"error": "Internal server error"}, status=500)
    
    
from django.shortcuts import redirect
from django.views.decorators.csrf import csrf_exempt
import urllib.parse

@csrf_exempt
def flitt_return_redirect(request):
    """
    Accepts Flitt return (usually POST form) and redirects browser to SPA.
    Preserves all submitted fields as query params for the frontend to read.
    """
    # Accept POST or GET
    data = request.POST if request.method == "POST" else request.GET
    # Build query string (URL-encoded)
    qs = "&".join(f"{urllib.parse.quote_plus(k)}={urllib.parse.quote_plus(v)}" for k, v in data.items())
    target = "https://chkuiskolopi.ge/payment-success"
    if qs:
        target = f"{target}?{qs}"
    return redirect(target)