from django.db import models
# Create your models here.
    
class LawG(models.Model):
    id = models.AutoField(primary_key=True)
    law_name = models.CharField(max_length=512)
    law_name_eng = models.CharField(max_length=512,default='')
    law_description = models.TextField()
    law_category = models.TextField(default="All")
    registration_number = models.TextField(default="None")
    created_at = models.DateTimeField(auto_now_add=False)  # Automatically sets the field to now when the object is created

    def __str__(self):
        return self.law_name

class Law3(models.Model):
    law_name = models.CharField(max_length=512)
    law_description = models.TextField()
    law_category = models.TextField()
    registration_number = models.CharField(max_length=50, default=None, primary_key=True)
    created_at = models.DateTimeField(auto_now_add=False)  # Automatically sets the field to now when the object is created

    def __str__(self):
        return self.law_name
    
class ChatHistory(models.Model):
    session_id = models.CharField(max_length=255)
    thread_id = models.CharField(max_length=50,null=False,unique=False,default="")
    user_id = models.CharField(max_length=255, null=True)
    query = models.TextField()
    response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    file_id = models.JSONField(default=list, blank=True, null=True)
    feedback = models.TextField()
    response_message=models.TextField()
    is_shared = models.BooleanField(default=False)
    urls = models.JSONField(default=list, blank=True, null=True)


    def __str__(self):
        return f"{self.session_id} - {self.file_id} - {self.query}"


class link_category_name(models.Model):
    category = models.CharField(max_length=512, null=False, unique=False, default="not specified")
    sub_category = models.CharField(max_length=512, null=False, unique=False, default="not specified")
    link = models.CharField(null=False, max_length=4096)
    num_pages = models.IntegerField(null=False, default=0)

    def __str__(self):
        return f"{self.category} - {self.sub_category} - {self.link}"
    
class LawMetadata(models.Model):
    vector_id = models.CharField(max_length=255, unique=True)  # Vector ID from Pinecone/FAISS
    law_name = models.TextField()  # Law name
    registration_code = models.TextField()  # Registration code
    text = models.TextField()  # Full law text or chunked law details
    text_in_english = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp when metadata is inserted

    class Meta:
        db_table = 'law_metadata'  # Name of the table in MySQL

    def __str__(self):
        return f"{self.law_name} ({self.registration_code})"

class DecisionMetadata(models.Model):
    decision_category = models.CharField(max_length=100)  # Stores folder name, e.g., "Administrative"
    decision_number = models.CharField(max_length=25, null=True, unique=False, default='')
    decision_date = models.CharField(max_length=20, unique=False, default="")
    decision_crux = models.TextField()
    decision_description = models.TextField()
    decision_motivation = models.TextField()
    decision_established = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'decision_metadata'

    def __str__(self):
        return f"{self.decision_category} - {self.vector_id}"

class LawIndexData(models.Model):
    law_name = models.TextField()  # Law name
    index_name = models.CharField(max_length=255)

    def __str__(self):
        return f"{self.law_name} - {self.index_name}"
    
class FileInfo(models.Model):
    file_name = models.CharField(max_length=255, unique=False, null=True)
    file_upload_user_id = models.CharField(max_length=20, unique=False, null=False)
    file_path = models.CharField(max_length=500, blank=True)  # Placeholder for the upload path
    file_type = models.CharField(max_length=50)
    session_id = models.CharField(max_length=50, unique=False, null=True, default=None)
    file_size = models.BigIntegerField()
    file_id_openai = models.CharField(max_length=50, unique=False, null=True, default=None)
    is_deleted = models.BooleanField(blank=False, default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file_name, self.file_id_openai

class ChatHistorySummary(models.Model):
    """
    Model to store the analyzed data for chat history.
    """
    user_id = models.IntegerField(unique=True)  # Ensure user_id is unique for easy updates
    email = models.CharField(max_length=255)
    total_queries = models.IntegerField()
    most_active_hour = models.IntegerField(null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"User ID: {self.user_id}, Email: {self.email}"

from django.db.models import Count
from django.db.models.functions import ExtractHour
from Users.models import UserPermission

class ChatHistoryAnalysis:
    """
    A class to analyze chat history and provide insights about user activity.
    """

    @staticmethod
    def get_user_activity_data():

        # Get all unique user_ids
        unique_users = ChatHistory.objects.values_list('user_id', flat=True).distinct()

        for user_id in unique_users:
            # Fetch user's email or assign Guest_<user_id> if no email
            user_email = (
                UserPermission.objects.filter(id=user_id)
                .values_list('email', flat=True)
                .first()
            )
            user_email = user_email if user_email else f"Guest_{user_id}"

            # Count total queries for the user_id
            total_queries = ChatHistory.objects.filter(user_id=user_id).count()

            # Find the most active hour
            most_active_hour = (
                ChatHistory.objects.filter(user_id=user_id)
                .annotate(hour=ExtractHour('created_at'))
                .values('hour')
                .annotate(query_count=Count('hour'))
                .order_by('-query_count')
                .first()
            )
            active_hour = most_active_hour['hour'] if most_active_hour else None

            # Update or create the record for this user
            ChatHistorySummary.objects.update_or_create(
                user_id=user_id,
                defaults={
                    'email': user_email,
                    'total_queries': total_queries,
                    'most_active_hour': active_hour,
                    'updated_at': models.functions.Now(),
                },
            )


class ChatSession(models.Model):
    session_id = models.CharField(max_length=255, unique=True)
    session_name = models.CharField(max_length=255, default="Untitled Session")
    is_deleted = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.session_name} (Deleted: {self.is_deleted})"
    

from Users.models import UserPermission

class UserBalance(models.Model):
    user = models.ForeignKey(UserPermission, on_delete=models.CASCADE)
    balance = models.DecimalField(max_digits=10, decimal_places=2, default=0.00)
    currency = models.CharField(max_length=3, default='GEL')  # Georgian Lari
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user} - {self.balance} {self.currency}"
# # Signal to create profile automatically
# from django.db.models.signals import post_save
# from django.dispatch import receiver

# @receiver(post_save, sender=UserPermission)
# def create_user_profile(sender, instance, created, **kwargs):
#     if created:
#         UserBalance.objects.create(user=instance)

class RechargeTransaction(models.Model):
    user = models.ForeignKey(UserPermission, on_delete=models.CASCADE)
    amount = models.DecimalField(max_digits=10, decimal_places=2)
    currency = models.CharField(max_length=3, default='GEL')
    transaction_id = models.CharField(max_length=100, unique=True)
    status = models.CharField(max_length=20)  # Pending, Completed, Failed
    created_at = models.DateTimeField(auto_now_add=True)
    is_updated = models.BooleanField(default=False)

    def __str__(self):
        return f"Transaction {self.transaction_id} - {self.status}"

class RefactoredLaw(models.Model):
    law_name = models.TextField()
    registration_code = models.CharField(max_length=255, db_index=True) # ინდექსი სწრაფი ძებნისთვის
    article_number = models.CharField(max_length=100) # "1", "31", "4^1" და ა.შ.
    text_ka = models.TextField()
    text_en = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'refactored_laws'

    def __str__(self):
        return f"{self.law_name} - {self.article_number}"


