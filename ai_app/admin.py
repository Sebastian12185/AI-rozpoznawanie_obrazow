from django.contrib import admin
from .models import ImageElement
from .models import TextElement

# Register your models here.
admin.site.register(ImageElement)
admin.site.register(TextElement)