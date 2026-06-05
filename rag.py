from sentence_transformers import SentenceTransformer
import chromadb
import os

# embedding model
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# ChromaDB client 
chroma_client = chromadb.PersistentClient(path="./genz_db")
collection = chroma_client.get_or_create_collection(name="genz_studios")

# chunks of information about GENZ Studios in both Arabic and English
CHUNKS = [
    # general info
    {
        "id": "general_ar",
        "text": "اسم الشركة: GENZ Studios. وكالة دعاية وإعلان واستوديو إبداعي متكامل. فرعان في القاهرة: التجمع الخامس (الرئيسي) والدقي."
    },
    {
        "id": "general_en",
        "text": "Company name: GENZ Studios. Full-service advertising agency and creative studio. Two branches in Cairo: New Cairo (Main) and Dokki."
    },

    # working hours
    {
        "id": "hours_ar",
        "text": "مواعيد العمل: مفتوح من السبت للخميس من 9 الصبح لـ 7 المساء. مغلق كل يوم جمعة."
    },
    {
        "id": "hours_en",
        "text": "Working hours: Open Saturday to Thursday, 9:00 AM to 7:00 PM. Closed every Friday."
    },

    # fifth settlement branch (main branch)
    {
        "id": "branch_newcairo_ar",
        "text": "فرع التجمع الخامس (الرئيسي): العنوان: التجمع الخامس، القاهرة الجديدة. التليفون: 010-0000-0001. الإيميل: newcairo@genzstudios.com"
    },
    {
        "id": "branch_newcairo_en",
        "text": "New Cairo branch (Main): Address: 5th Settlement, New Cairo. Phone: 010-0000-0001. Email: newcairo@genzstudios.com"
    },

    # dokki branch (secondary branch)
    {
        "id": "branch_dokki_ar",
        "text": "فرع الدقي: العنوان: منطقة التحرير، الدقي، الجيزة. التليفون: 010-0000-0002. الإيميل: dokki@genzstudios.com"
    },
    {
        "id": "branch_dokki_en",
        "text": "Dokki branch: Address: Tahrir Square area, Dokki, Giza. Phone: 010-0000-0002. Email: dokki@genzstudios.com"
    },

    # studio A (photography studio)
    {
        "id": "studio_a_ar",
        "text": "استوديو A - فرع التجمع - استوديو تصوير فوتوغرافي. المساحة: 70 م². التجهيزات: 4 رولات خلفية (أبيض، أسود، رمادي، بيج)، إضاءة Godox ستروب 4 رؤوس، عواكس، غرفة تغيير ملابس. الأنسب لـ: تصوير المنتجات، البورتريه، الهيدشوت، الأزياء. سعر الساعة: 1500 جنيه. نص يوم 4 ساعات: 5000 جنيه. يوم كامل 8 ساعات: 9000 جنيه."
    },
    {
        "id": "studio_a_en",
        "text": "Studio A - New Cairo branch - Photography Studio. Size: 70 m². Equipment: 4 background rolls (white, black, gray, beige), Godox strobe lighting 4 heads, reflectors, changing room. Best for: product photography, portraits, headshots, fashion. Hourly: 1500 EGP. Half day 4 hours: 5000 EGP. Full day 8 hours: 9000 EGP."
    },

    # studio B (video & production studio)
    {
        "id": "studio_b_ar",
        "text": "استوديو B - فرع التجمع - استوديو فيديو وإنتاج. المساحة: 140 م². التجهيزات: سيت فيديو متكامل، كاميرات 4K، تيليبرومتر، جرين سكرين 6×4م، شبكة إضاءة LED، مكسر صوت. الأنسب لـ: إعلانات تلفزيونية، فيديوهات مؤسسية، محتوى سوشيال. سعر الساعة: 2500 جنيه. نص يوم: 8500 جنيه. يوم كامل: 15000 جنيه."
    },
    {
        "id": "studio_b_en",
        "text": "Studio B - New Cairo branch - Video & Production Studio. Size: 140 m². Equipment: Full video set, 4K cameras, teleprompter, green screen 6x4m, LED lighting grid, audio mixer. Best for: TV commercials, corporate videos, social media content. Hourly: 2500 EGP. Half day: 8500 EGP. Full day: 15000 EGP."
    },

    # studio C (podcast & content studio)
    {
        "id": "studio_c_ar",
        "text": "استوديو C - فرع التجمع - استوديو بودكاست ومحتوى. المساحة: 35 م². التجهيزات: عزل صوتي، 4 ميكروفونات كوندنسر، كاميرتين ثابتتين، ديكور جاهز، إضاءة LED. الأنسب لـ: بودكاست، يوتيوب، مقابلات، ريلز. سعر الساعة: 800 جنيه. نص يوم: 2500 جنيه. يوم كامل: 4500 جنيه."
    },
    {
        "id": "studio_c_en",
        "text": "Studio C - New Cairo branch - Podcast & Content Studio. Size: 35 m². Equipment: Soundproofed, 4 condenser microphones, 2 fixed cameras, set décor, LED lights. Best for: podcasts, YouTube, interviews, reels. Hourly: 800 EGP. Half day: 2500 EGP. Full day: 4500 EGP."
    },

    # studio D (photography studio)
    {
        "id": "studio_d_ar",
        "text": "استوديو D - فرع الدقي - استوديو تصوير فوتوغرافي. المساحة: 65 م². التجهيزات: 3 رولات خلفية (أبيض، أسود، أزرق متدرج)، إضاءة Godox 3 رؤوس، عواكس، غرفة تغيير ملابس. الأنسب لـ: تصوير منتجات، بورتريه، إيكومرس. سعر الساعة: 1500 جنيه. نص يوم: 5000 جنيه. يوم كامل: 9000 جنيه."
    },
    {
        "id": "studio_d_en",
        "text": "Studio D - Dokki branch - Photography Studio. Size: 65 m². Equipment: 3 background rolls (white, black, gradient blue), Godox strobe 3 heads, reflectors, changing room. Best for: product photography, portraits, e-commerce. Hourly: 1500 EGP. Half day: 5000 EGP. Full day: 9000 EGP."
    },

    # studio E (video & production studio)
    {
        "id": "studio_e_ar",
        "text": "استوديو E - فرع الدقي - استوديو فيديو وإنتاج. المساحة: 130 م². التجهيزات: سيت فيديو متكامل، كاميرات 4K، جرين سكرين 5×4م، إضاءة احترافية، تيليبرومتر. الأنسب لـ: إعلانات، فيديوهات مؤسسية، فعاليات. سعر الساعة: 2500 جنيه. نص يوم: 8500 جنيه. يوم كامل: 15000 جنيه."
    },
    {
        "id": "studio_e_en",
        "text": "Studio E - Dokki branch - Video & Production Studio. Size: 130 m². Equipment: Full video set, 4K cameras, green screen 5x4m, professional lighting, teleprompter. Best for: commercials, corporate videos, events. Hourly: 2500 EGP. Half day: 8500 EGP. Full day: 15000 EGP."
    },

    # studio F (podcast & content studio)
    {
        "id": "studio_f_ar",
        "text": "استوديو F - فرع الدقي - استوديو بودكاست ومحتوى. المساحة: 30 م². التجهيزات: عزل صوتي، 3 ميكروفونات كوندنسر، كاميرتين، ديكور، إضاءة. الأنسب لـ: بودكاست، مقابلات، يوتيوب، ريلز. سعر الساعة: 800 جنيه. نص يوم: 2500 جنيه. يوم كامل: 4500 جنيه."
    },
    {
        "id": "studio_f_en",
        "text": "Studio F - Dokki branch - Podcast & Content Studio. Size: 30 m². Equipment: Soundproofed, 3 condenser microphones, 2 cameras, set décor, lights. Best for: podcasts, interviews, YouTube, reels. Hourly: 800 EGP. Half day: 2500 EGP. Full day: 4500 EGP."
    },

    # photography packages
    {
        "id": "photo_packages_ar",
        "text": "باكدجات التصوير الفوتوغرافي: جلسة بسيطة ساعة 10 صور معدلة: 2000 جنيه. جلسة قياسية 3 ساعات 30 صورة: 5000 جنيه. يوم كامل 8 ساعات 80 صورة: 12000 جنيه. تصوير منتجات: 300 جنيه للمنتج (الحد الأدنى 10 منتجات). باكدج إيكومرس 50 منتج: 10000 جنيه."
    },
    {
        "id": "photo_packages_en",
        "text": "Photography packages: Basic session 1 hour 10 edited photos: 2000 EGP. Standard session 3 hours 30 photos: 5000 EGP. Full day 8 hours 80 photos: 12000 EGP. Product photography: 300 EGP per product (min 10 products). E-commerce package 50 products: 10000 EGP."
    },

    # video production packages
    {
        "id": "video_packages_ar",
        "text": "إنتاج الفيديو: ريل سوشيال ميديا لحد 60 ثانية: 3500 جنيه. فيديو مؤسسي 3-5 دقايق: 15000 جنيه. إعلان تلفزيوني 30 ثانية: 35000 جنيه. فيديو يوتيوب مونتاج وموشن جرافيك: 8000 جنيه."
    },
    {
        "id": "video_packages_en",
        "text": "Video production: Social media reel up to 60 seconds: 3500 EGP. Corporate video 3-5 minutes: 15000 EGP. TV commercial 30 seconds: 35000 EGP. YouTube video with motion graphics: 8000 EGP."
    },

    # advertising & marketing services
    {
        "id": "marketing_ar",
        "text": "خدمات التسويق والإعلان: إدارة إعلانات سوشيال ميديا شهري: 5000 جنيه. إدارة حملات جوجل شهري: 4000 جنيه. باكدج تسويق رقمي متكامل: 12000 جنيه شهرياً. استراتيجية براند وهوية بصرية: تبدأ من 20000 جنيه."
    },
    {
        "id": "marketing_en",
        "text": "Advertising & marketing services: Social media ads management monthly: 5000 EGP. Google Ads management monthly: 4000 EGP. Full digital marketing package: 12000 EGP/month. Brand strategy and visual identity: starting from 20000 EGP."
    },

    # creative design services
    {
        "id": "design_ar",
        "text": "خدمات التصميم الإبداعي: لوجو وهوية بصرية: تبدأ من 5000 جنيه. تصميم بوستات سوشيال شهري 20 بوست: 3000 جنيه. تصميم بروشور أو فلاير: 800 جنيه. تصميم باكدج: تبدأ من 4000 جنيه."
    },
    {
        "id": "design_en",
        "text": "Creative design services: Logo and visual identity: starting from 5000 EGP. Social media posts monthly 20 posts: 3000 EGP. Brochure or flyer design: 800 EGP. Packaging design: starting from 4000 EGP."
    },

    # social media management
    {
        "id": "social_ar",
        "text": "إدارة السوشيال ميديا: أساسي 3 منصات 12 بوست شهرياً: 4000 جنيه. قياسي 3 منصات 20 بوست وستوريز: 7000 جنيه. بريميوم 4 منصات إدارة كاملة وإعلانات: 15000 جنيه شهرياً."
    },
    {
        "id": "social_en",
        "text": "Social media management: Basic 3 platforms 12 posts/month: 4000 EGP. Standard 3 platforms 20 posts plus stories: 7000 EGP. Premium 4 platforms full management plus ads: 15000 EGP/month."
    },

    # booking policy
    {
        "id": "booking_ar",
        "text": "سياسة الحجز: الحد الأدنى ساعة واحدة. الحجز المسبق قبل 24 ساعة على الأقل. الإلغاء مجاني قبل 12 ساعة من الجلسة. العربون 30% لتأكيد الحجز."
    },
    {
        "id": "booking_en",
        "text": "Booking policy: Minimum booking 1 hour. Advance booking at least 24 hours. Free cancellation up to 12 hours before session. 30% deposit to confirm booking."
    },

    # booking flow - خطوات الحجز
    {
        "id": "booking_flow_ar",
        "text": "خطوات الحجز: اختار الاستوديو (A/B/C في التجمع أو D/E/F في الدقي) والتاريخ والمدة. وفّر اسمك الكامل ورقم تليفونك. هنتحقق من المواعيد المتاحة ونأكدلك الحجز برقم مرجعي. ممكن تلغي أو تستعلم عن حجزك برقم الحجز."
    },
    {
        "id": "booking_flow_en",
        "text": "Booking steps: Choose your studio (A/B/C in New Cairo or D/E/F in Dokki), date, and duration. Provide your full name and phone number. We will check available slots and confirm your booking with a reference ID. You can cancel or check your booking status using the booking ID."
    },
]

def build_index():
    """build the ChromaDB index if it doesn't already exist. We check if there are any existing ids in the collection, and if not we encode and add all the chunks."""
    existing = collection.get()
    if existing["ids"]:
        print(f" index already exists — {len(existing['ids'])} chunks saved ")
        return

    print("building index for the first time... encoding and saving chunks to ChromaDB")
    texts = [c["text"] for c in CHUNKS]
    ids = [c["id"] for c in CHUNKS]
    embeddings = model.encode(texts).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
    )
    print(f" done! {len(CHUNKS)} chunks saved to the index.")

def search(query: str, n_results: int = 3) -> str:
    """search for the closest chunks to the query"""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )
    chunks = results["documents"][0]
    return "\n\n".join(chunks)

# when we run this file directly we want to build the index 
build_index()