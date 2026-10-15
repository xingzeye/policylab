from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from apps.modelsinfo.learning import learning

admin.site.site_header = "策学 · PolicyLab"
admin.site.site_title = "策学 · PolicyLab"
admin.site.index_title = "离线强化学习实验"

urlpatterns = [
    path('', learning, name='home'),
    path('learn/', learning, name='learn'),

    path('admin/', admin.site.urls),
    path("datainfo/", include("apps.datainfo.urls")),
    path("modelsinfo/", include("apps.modelsinfo.urls")),
    path("results/", include("apps.results.urls")),
    path("compare/", include("apps.compare.urls")),


]
handler404 = "qlearn.views.page_not_found"
# 静态和媒体文件
if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
    )
