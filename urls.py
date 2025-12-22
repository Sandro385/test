from django.urls import path
from scrapper import views
# ახალი იმპორტები
from . import payment_router
from . import payment_callback 

urlpatterns = [
    path('scrap/', views.scrap, name="scrap"),
    path('upsert/', views.FaissIndexAPI.as_view(), name="vector upload"),
    path('query/', views.SimpleQueryAPI.as_view(), name="vector query"),
    path('process_query/', views.SearchQueryResult.as_view(), name = 'user_query_processor' ),
    path('sessions/',views.ChatSessionView.as_view(),name="chat_Sessions"),
    path('chats/',views.ChatHistoryListView.as_view(),name="chat_history"),
    path('concatenate-laws/', views.ConcatenateLawAPIView.as_view(), name='concatenate-laws'),
    path('upload/',views.FileUploadView.as_view(),name='Document_Upload'),
    path('list_documents/',views.FileInfoListView.as_view(),name="List_Of_Uploaded_Files"),
    path('delete_document/',views.SoftDeleteFileView.as_view(),name='Delete_File'),
    path('upsert_2/',views.FaissIndexDecisions.as_view(),name="court_decisions_embed"),
    path('upsert_3/',views.ProcessCourtDecisions.as_view(),name="court_decisions_embed_2"),
    # path('search_db_law/',views.law_search,name='law_search_in_db'), # წაშლილია
    path('search_db_precedent/',views.law_decisions,name='law_search_in_db'),
    path('translate_law_names/',views.translate_law_names,name='law_name_KA_to_EN'),
    path('chat-history/feedback/<int:chat_id>/', views.feedback, name='chat_history_feedback'),
    path('chat-history/send-email/<int:chat_id>/', views.send_chat_email, name='send_chat_email'),
    path('feedback/technical/', views.technical_feedback, name='technical_feedback'),
    path('chat-session-action/', views.ChatSessionActionView.as_view(), name='chat_session_action'),
    path('link-sessions/', views.LinkChatSessionsAPIView.as_view(), name="link_sessions"),
    path('recharge/initiate/', views.initiate_recharge, name='initiate_recharge'),
    path('recharge/success/', views.payment_success, name='payment_success'),
    path('get-user-balance/', views.get_user_balance, name='get_user_balance'),
    
    # --- FLITT URLS ---
    path('flitt/recharge/', views.initiate_flitt_recharge, name='flitt_recharge'),
    
    # 1. ჩვენი Smart Router (გადახდის ინიცირება)
    path('recharge/smart-router/', payment_router.smart_payment_router, name='smart_recharge_router'),
    
    # 2. ჩვენი ახალი Callback Handler (ბალანსის ასახვა) - შეცვლილია!
    path('flitt/callback/', payment_callback.flitt_callback_handler, name='flitt_callback'),
    
    path("flitt/return/", views.flitt_return_redirect, name="flitt_return_redirect"),
    path("flitt/return", views.flitt_return_redirect, name="flitt_return_redirect_noslash"),
]