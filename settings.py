from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# BASE_BACKEND_URL = "http://127.0.0.1:8000" 

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv('SECRET_KEY')

GOOGLE_CUSTOM_SEARCH_API = os.getenv( 'GOOGLE_CUSTOM_SEARCH_API' )

SEARCH_ENGINE_ID = os.getenv( 'SEARCH_ENGINE_ID' )

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!


# SECURITY WARNING: don't run with debug turned on in production!

DEBUG = True

ALLOWED_HOSTS = ['*']
CSRF_TRUSTED_ORIGINS = ['https://smart-aleck.be.oodleslab.com']
CSRF_TRUSTED_ORIGINS = ['https://chkuioskopi.ge']
# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'scrapper',
    'rest_framework',
    'Users.apps.UsersConfig',
    'corsheaders',
]

MIDDLEWARE = [
    'whitenoise.middleware.WhiteNoiseMiddleware',	
    'corsheaders.middleware.CorsMiddleware',  # Keep this at the top
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'smart_aleck.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

CORS_ALLOW_ALL_ORIGINS = True

CORS_ALLOW_CREDENTIALS = True

CORS_ALLOW_METHODS = [
    "GET",
    "POST",
    "PUT",
    "DELETE",
    "PATCH",
    "OPTIONS",
]

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding', 
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
    'ngrok-skip-browser-warning',
]

CSRF_TRUSTED_ORIGINS = ['https://smart-aleck.be.oodleslab.com','https://chkuioskopi.ge']

WSGI_APPLICATION = 'smart_aleck.wsgi.application'

AUTH_USER_MODEL = "Users.UserPermission"            

# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'test_db',      # ჩვენი ახალი ბაზა
        'USER': 'root',         # XAMPP-ის დეფოლტ იუზერი
        'PASSWORD': '',         # XAMPP-ს პაროლი არ აქვს
        'HOST': '127.0.0.1',
        'PORT': '3306',
    }
}

# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/


# Check if the directory exists, and create it if it doesn't
if not os.path.exists(os.path.join(BASE_DIR, "static")):
    os.makedirs(os.path.join(BASE_DIR, "static"))

# If using a development server, you might also need to add:
STATIC_URL = '/api/static/'
STATIC_ROOT = '/opt/git/smart-aleck/backend/Smart-Aleck-backend/smart_aleck/staticfiles'


STATICFILES_DIRS = [
    BASE_DIR / "static",  # Change according to your file structure
]
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# settings.py
EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'info@chkuiskolopi.ge'
EMAIL_HOST_PASSWORD = os.getenv("SMART_ALECK_EMAIL_PASSWORD")
DEFAULT_FROM_EMAIL = 'info@chkuiskolopi.ge'

BASE_LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Ensure the logs directory exists
os.makedirs(BASE_LOG_DIR, exist_ok=True)

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'file_info': {
            'level': 'INFO',  # Only log INFO and higher messages
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_LOG_DIR, 'info.log'),
            'formatter': 'verbose',
        },
        'file_error': {
            'level': 'ERROR',
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_LOG_DIR, 'error.log'),
            'formatter': 'verbose',
        },
        'console': {
            'level': 'INFO',  # Set console output level to INFO
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
    },
    'loggers': {
        'django': {
            'handlers': ['file_info', 'file_error', 'console'],
            'level': 'INFO',  # Adjust this to 'INFO' to prevent DEBUG messages
            'propagate': True,
        },
        'operations': {
            'handlers': ['file_info', 'file_error'],
            'level': 'INFO',  # Change to INFO for logging relevant operations
            'propagate': False,
        },
    },
}

CSRF_TRUSTED_ORIGINS = ['https://api.chkuiskolopi.ge', 'https://chkuiskolopi.ge']
