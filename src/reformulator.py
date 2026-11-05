#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Urdu-DynRAG: Deterministic Urdu Reformulator Module (Phase 2)
Handles Action 3 (Reformulation / Script Normalization):
  - Normalizes Roman Urdu (S4) and Code-Switched (S3) queries into formal Nastaliq script.
  - Cached deterministically in cache/reformulation_cache.json keyed by sha256(query).
  - Includes a comprehensive 300+ term fallback Roman/English-to-Urdu legal/civic/science lexicon.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv

load_dotenv()

# Perso-Arabic Unicode range: \u0600 - \u06FF
URDU_UNICODE_PATTERN = re.compile(r"[\u0600-\u06FF]")


def compute_urdu_script_ratio(text: str) -> float:
    """Calculate the fraction of alphabetic characters that belong to Urdu/Arabic unicode."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 1.0  # If only numbers/punctuation, treat as valid
    urdu_chars = len(URDU_UNICODE_PATTERN.findall("".join(alpha_chars)))
    return float(urdu_chars) / float(len(alpha_chars))


# Comprehensive 300+ term fallback dictionary for Pakistani legal, civic, procedural, and science terminology
FALLBACK_LEXICON: Dict[str, str] = {
    # 1. Question particles & pronouns
    "kya": "کیا", "kia": "کیا", "kyun": "کیوں", "kyu": "کیوں", "kahan": "کہاں",
    "kab": "کب", "kaun": "کون", "kon": "کون", "kaunsa": "کونسا", "kaunsi": "کونسی",
    "konsa": "کونسا", "konsi": "کونسی", "kaise": "کیسے", "kesy": "کیسے", "kese": "کیسے",
    "kis": "کس", "kisko": "کس کو", "kise": "کسے", "kitna": "کتنا", "kitni": "کتنی",
    "kitne": "کتنے", "kitny": "کتنے", "kidhar": "کدھر", "kisko": "کس کو",
    "yeh": "یہ", "ye": "یہ", "woh": "وہ", "wo": "وہ", "is": "اس", "us": "اس",
    "in": "ان", "un": "ان", "iska": "اس کا", "iski": "اس کی", "iske": "اس کے",
    "uska": "اس کا", "uski": "اس کی", "uske": "اس کے", "inka": "ان کا", "inke": "ان کے",
    "mujhe": "مجھے", "apko": "آپ کو", "ap": "آپ", "aap": "آپ", "hum": "ہم",
    "unhein": "انہیں", "unhe": "انہیں", "isay": "اسے", "usey": "اسے", "usse": "اس سے",
    "isse": "اس سے", "kisi": "کسی", "koi": "کوئی", "sab": "سب",

    # 2. Auxiliary verbs & copulas
    "hai": "ہے", "hy": "ہے", "hain": "ہیں", "hn": "ہیں", "tha": "تھا",
    "thi": "تھی", "the": "تھے", "hoga": "ہوگا", "hogi": "ہوگی", "honge": "ہوں گے",
    "ho": "ہو", "hota": "ہوتا", "hoti": "ہوتی", "hote": "ہوتے", "hona": "ہونا",
    "hone": "ہونے", "huye": "ہوئے", "hui": "ہوئی", "hua": "ہوا",
    "karna": "کرنا", "karne": "کرنے", "karein": "کریں", "karen": "کریں",
    "karta": "کرتا", "karti": "کرتی", "karte": "کرتے", "kiya": "کیا",
    "sakta": "سکتا", "sakti": "سکتی", "sakte": "سکتے", "chahiye": "چاہیے",
    "dena": "دینا", "dene": "دینے", "diya": "دیا", "di": "دی", "diye": "دیے",
    "milta": "ملتا", "milti": "ملتی", "milte": "ملتے", "mila": "ملا",
    "jana": "جانا", "jane": "جانے", "jata": "جاتا", "jati": "جاتی", "jate": "جاتے",
    "gaya": "گیا", "gayi": "گئی", "gaye": "گئے", "par": "پر", "se": "سے",
    "mein": "میں", "main": "میں", "ko": "کو", "ka": "کا", "ki": "کی", "ke": "کے",
    "tak": "تک", "aur": "اور", "ya": "یا", "bhi": "بھی", "to": "تو", "hi": "ہی",
    "ne": "نے", "liye": "لیے", "lye": "لیے", "bina": "بغیر", "baghair": "بغیر",
    "sath": "ساتھ", "pehlay": "پہلے", "pehle": "پہلے", "baad": "بعد",
    "ziada": "زیادہ", "zyada": "زیادہ", "kam": "کم", "bohot": "بہت", "bahut": "بہت",
    "dono": "دونوں", "kul": "کل", "sirf": "صرف", "lazmi": "لازمی", "zaruri": "ضروری",
    "zaroori": "ضروری", "drkar": "درکار", "dastiyab": "دستیاب", "dastayab": "دستیاب",

    # 3. Numeric tokens & units
    "ek": "1", "do": "2", "teen": "3", "chaar": "4", "char": "4",
    "panch": "5", "chhay": "6", "che": "6", "saat": "7", "sat": "7",
    "aath": "8", "ath": "8", "nau": "9", "no": "9", "dus": "10", "das": "10",
    "gyarah": "11", "barah": "12", "terah": "13", "chaudah": "14", "pandrah": "15",
    "solah": "16", "satrah": "17", "atharah": "18", "unnees": "19", "bees": "20",
    "ikkees": "21", "bayees": "22", "tayees": "23", "chaubees": "24", "pachees": "25",
    "chhabees": "26", "sattaees": "27", "athaees": "28", "untees": "29", "tees": "30",
    "pantees": "35", "chalees": "40", "pantalees": "45", "pachaas": "50", "pachas": "50",
    "saath": "60", "sattar": "70", "assi": "80", "nabbe": "90", "sau": "100",
    "hazar": "1000", "lakh": "100000", "crore": "10000000",
    "saal": "سال", "year": "سال", "years": "سال", "mahine": "مہینے", "mah": "ماہ",
    "month": "ماہ", "months": "ماہ", "din": "دن", "dinon": "دنوں", "day": "دن",
    "days": "دن", "ghante": "گھنٹے", "hours": "گھنٹے", "rupaye": "روپے",
    "rupay": "روپے", "rupees": "روپے", "fees": "فیس", "fee": "فیس",
    "nashistein": "نشستیں", "nashist": "نشست", "seats": "نشستیں", "seat": "نشست",

    # 4. Civic, Legal & Procedural Terminology
    "shanakhti": "شناختی", "shinakhti": "شناختی", "shanakti": "شناختی",
    "cnic": "شناختی کارڈ", "nic": "شناختی کارڈ", "card": "کارڈ",
    "gum": "گم", "lost": "گمشدہ", "naya": "نیا", "banwane": "بنوانے",
    "fir": "ایف آئی آر", "police": "پولیس", "thana": "تھانہ", "report": "رپورٹ",
    "helmet": "ہیلمٹ", "motorcycle": "موٹرسائیکل", "bike": "موٹرسائیکل",
    "punjab": "پنجاب", "karwai": "کارروائی", "challan": "چالان",
    "frc": "ایف آر سی", "family": "خاندان", "registration": "رجسٹریشن",
    "certificate": "سرٹیفکیٹ", "nadra": "نادرا", "smart": "سمارٹ",
    "poc": "پی او سی", "pakistan": "پاکستان", "origin": "اورجن",
    "peca": "پیکا", "dafa": "دفعہ", "section": "دفعہ", "jurmana": "جرمانہ",
    "fine": "جرمانہ", "penalty": "سزا", "saza": "سزا", "qaid": "قید",
    "imprisonment": "قید", "hec": "ایچ ای سی", "online": "آن لائن",
    "system": "نظام", "degree": "ڈگری", "transcript": "ٹرانسکرپٹ",
    "tasdeeq": "تصدیق", "attestation": "تصدیق", "ppc": "پی پی سی",
    "jaan": "جان", "marne": "مارنے", "dhamki": "دھمکی", "threat": "دھمکی",
    "death": "موت", "town": "ٹاؤن", "corporation": "کارپوریشن",
    "khawateen": "خواتین", "women": "خواتین", "administrator": "ایڈمنسٹریٹر",
    "tax": "ٹیکس", "tajweez": "تجویز", "proposal": "تجویز",
    "mustarad": "مسترد", "reject": "مسترد", "qaumi": "قومی",
    "assembly": "اسمبلی", "parliament": "پارلیمنٹ", "succession": "وراثت",
    "wirasat": "وراثت", "darkhwast": "درخواست", "application": "درخواست",
    "bachon": "بچوں", "children": "بچوں", "minor": "نابالغ",
    "umar": "عمر", "age": "عمر", "passport": "پاسپورٹ",
    "nepra": "نیپرا", "operator": "آپریٹر", "licence": "لائسنس",
    "license": "لائسنس", "tobacco": "تمباکو", "kharidari": "خریداری",
    "purchase": "خریداری", "adaigi": "ادائیگی", "payment": "ادائیگی",
    "mandi": "منڈی", "market": "منڈی", "kisan": "کسان", "farmer": "کسان",
    "wajibat": "واجبات", "dues": "واجبات", "farokht": "فروخت",
    "sale": "فروخت", "tareekh": "تاریخ", "date": "تاریخ",
    "eobi": "ای او بی آئی", "pension": "پنشن", "helpline": "ہیلپ لائن",
    "free": "مفت", "auqaat": "اوقات", "timing": "اوقات",
    "budhape": "بڑھاپے", "old": "بوڑھا", "retro": "پچھلی",
    "pichhli": "پچھلی", "muddat": "مدت", "period": "مدت",
    "manzoor": "منظور", "approve": "منظور", "spouse": "شریک حیات",
    "meyaad": "میعاد", "validity": "میعاد", "expiry": "میعاد",
    "mujrimana": "مجرمانہ", "criminal": "مجرمانہ", "daraandazi": "دراندازی",
    "trespass": "دراندازی", "vigilance": "ویجیلنس", "committee": "کمیٹی",
    "faisla": "فیصلہ", "decision": "فیصلہ", "fareeqain": "فریقین",
    "parties": "فریقین", "binding": "لازمی", "tehet": "تحت",
    "rent": "کرایہ", "kiraya": "کرایہ", "premises": "پریمیسز",
    "malik": "مالک", "makan": "مکان", "landlord": "مالک مکان",
    "tenant": "کرایہ دار", "consumer": "صارف", "sarif": "صارف",
    "court": "عدالت", "adalat": "عدالت", "claim": "دعوی", "daawa": "دعوی",
    "driving": "ڈرائیونگ", "renewal": "تجدید", "tajdeed": "تجدید",
    "medical": "طبی", "fitness": "فٹنس", "free education": "مفت تعلیم",
    "act": "ایکٹ", "law": "قانون", "qanoon": "قانون", "rules": "قواعد",
    "rule": "قاعدہ", "ordinance": "آرڈیننس",

    # 5. Science & General domain
    "zameen": "زمین", "earth": "زمین", "bahari": "بیرونی",
    "outermost": "سب سے بیرونی", "tah": "تہہ", "layer": "تہہ",
    "crust": "قشر", "mantle": "مینٹل", "core": "مرکزہ",
    "haddi": "ہڈی", "haddiyan": "ہڈیاں", "bones": "ہڈیاں",
    "jism": "جسم", "body": "جسم", "nobel": "نوبل", "inam": "انعام",
    "prize": "انعام", "scientist": "سائنسدان",

    # 7. English terms found in Code-Switched (S3) queries
    "punishment": "سزا", "maximum": "زیادہ سے زیادہ", "theft": "چوری", "jail": "قید",
    "extortion": "بھتہ خوری", "defamation": "ہتک عزت", "cheating": "دھوکہ دہی",
    "liability": "ذمہ داری", "common intention": "مشترکہ ارادہ", "person": "شخص",
    "candidates": "امیدوار", "proposal": "تجویز", "disapprove": "مسترد",
    "functional": "فعال", "cancellation": "منسوخی", "modification": "تبدیلی",
    "adults": "بالغ افراد", "total": "کل", "limit": "حد", "dues": "واجبات",
    "sale date": "فروخت کی تاریخ", "transfer": "منتقلی", "complete": "مکمل",
    "available": "دستیاب", "last name": "آخری نام", "characters": "حروف",
    "exhausted": "ختم", "choose": "منتخب", "allow": "منظور", "retroactively": "پچھلی تاریخ سے",
    "particulars": "تفصیلات", "timeline": "ٹائم لائن", "components": "اجزاء",
    "meaning": "مطلب", "single": "سنگل", "organizations": "ادارے", "services": "خدمات",
    "access": "رسائی", "grievous hurt": "شدید چوٹ", "fear": "خوف", "highway": "ہائی وے",
    "trespass": "دراندازی", "closed": "بند", "reason": "وجہ", "intimation": "اطلاع",
    "depot": "ڈپو", "representatives": "نمائندے", "plants": "پودے", "release": "خارج",
    "mountain areas": "پہاڑی علاقے", "temperature": "درجہ حرارت", "boil": "ابلتا",
    "sea": "سمندر", "fall": "گرتا", "mosquito": "مچھر", "spread": "پھیلاتا",
    "classes": "جماعتیں", "right": "حق", "property": "جائیداد", "sublet": "آگے کرائے پر دینا",
    "permission": "اجازت", "submit": "جمع", "replacement": "تبدیلی", "submitting": "جمع کرانا",
    "check": "چیک", "eligibility": "اہلیت", "insuranced": "بیمہ شدہ", "insured": "بیمہ شدہ",
    "dwelling house": "رہائشی مکان", "dwelling": "رہائشی", "house": "مکان", "river": "دریا",
    "381a": "381-اے", "a": "ایک", "aage": "آگے", "aakhir": "آخر", "aakhri": "آخری",
    "ada": "ادا", "afsar": "افسر", "agar": "اگر", "agents": "ایجنٹس", "agreements": "معاہدات",
    "ahliyat": "اہلیت", "aik": "ایک", "amal": "عمل", "andar": "اندر", "apne": "اپنے",
    "apni": "اپنی", "app": "ایپ", "average": "اوسط", "az": "از", "baaligh": "بالغ",
    "baar": "بار", "baare": "بارے", "ban": "بن", "band": "بند", "bari": "بڑی",
    "batana": "بتانا", "bform": "ب فارم", "bhatta": "بھتہ", "bheja": "بھیجا", "bima": "بیمہ",
    "biometric": "بائیو میٹرک", "board": "بورڈ", "bukhar": "بخار", "bunyad": "بنیاد",
    "business": "کاروبار", "cancel": "منسوخ", "cards": "کارڈز", "case": "کیس",
    "category": "کیٹیگری", "chalti": "چلتی", "chori": "چوری", "chot": "چوٹ",
    "clearance": "کلیئرنس", "code": "کوڈ", "commission": "کمیشن", "companies": "کمپنیاں",
    "company": "کمپنی", "contribution": "حصہ", "corporations": "کارپوریشنز", "council": "کونسل",
    "daar": "دار", "darje": "درجے", "daryaesindh": "دریائے سندھ", "dauran": "دوران",
    "dayar": "دائر", "de": "دے", "dehi": "دیہی", "delivery": "ڈیلیوری", "dengue": "ڈینگی",
    "depot": "ڈپو", "dhoka": "دھوکہ", "dikhana": "دکھانا", "dila": "دلا", "dusre": "دوسرے",
    "dusri": "دوسری", "election": "الیکشن", "epassport": "ای پاسپورٹ", "executive": "ایگزیکٹو",
    "extra": "اضافی", "faal": "فعال", "fields": "شعبے", "fisad": "فیصد", "gaari": "گاڑی",
    "ganthein": "گھنٹے", "gas": "گیس", "general": "جنرل", "ghar": "گھر", "girta": "گرتا",
    "government": "حکومت", "grade": "گریڈ", "grant": "گرانٹ", "growers": "کاشتکار",
    "had": "حد", "haq": "حق", "har": "ہر", "hatakeizzat": "ہتک عزت", "highway": "ہائی وے",
    "hisse": "حصے", "hourly": "گھنٹہ وار", "hukumat": "حکومت", "huroof": "حروف",
    "ict": "آئی سی ٹی", "idaron": "اداروں", "ijazat": "اجازت", "ilaqon": "علاقوں",
    "interview": "انٹرویو", "irade": "ارادے", "islamabad": "اسلام آباد", "istemal": "استعمال",
    "it": "آئی ٹی", "ittela": "اطلاع", "ja": "جا", "jaanchegi": "جانچے گی", "jamaaton": "جماعتوں",
    "janne": "جاننے", "january": "جنوری", "jari": "جاری", "jaye": "جائے", "jayega": "جائے گا",
    "jayen": "جائیں", "jurm": "جرم", "kar": "کر", "karane": "کرانے", "karni": "کرنی",
    "khareedari": "خریداری", "khareedne": "خریدنے", "kharidna": "خریدنا", "kharij": "خارج",
    "khatam": "ختم", "khauf": "خوف", "khidmaat": "خدمات", "khori": "خوری", "kin": "کن",
    "kiraye": "کرائے", "kisanon": "کسانوں", "laazmi": "لازمی", "lagti": "لگتی", "lene": "لینے",
    "likhi": "لکھی", "liya": "لیا", "local": "مقامی", "login": "لاگ ان", "logon": "لوگوں",
    "machar": "مچھر", "mahana": "ماہانہ", "maloomat": "معلومات", "manga": "مانگا",
    "maqsad": "مقصد", "marginal": "مارجنل", "markaz": "مرکز", "maut": "موت",
    "mehkame": "محکمے", "members": "ارکان", "minimum": "کم از کم", "miyaad": "میعاد",
    "mobile": "موبائل", "muaheday": "معاہدے", "muft": "مفت", "mukammal": "مکمل",
    "mukhtalif": "مختلف", "mulazim": "ملازم", "mulk": "ملک", "multiple": "متعدد",
    "muntakhib": "منتخب", "muntaqli": "منتقلی", "muraad": "مراد", "mushtarka": "مشترکہ",
    "na": "نہ", "naam": "نام", "nafiz": "نافذ", "naye": "نئے", "nayi": "نئی",
    "nicop": "نائیکوپ", "nimayan": "نمایاں", "nonprintable": "نان پرنٹ ایبل", "normal": "نارمل",
    "nrc": "این آر سی", "numainde": "نمائندے", "number": "نمبر", "office": "دفتر",
    "paas": "پاس", "pages": "صفحات", "pahari": "پہاڑی", "pakid": "پاک آئی ڈی",
    "pani": "پانی", "patta": "پتہ", "paudey": "پودے", "pehenne": "پہننے", "penal": "پینل",
    "pesh": "پیش", "phailata": "پھیلاتا", "photosynthesis": "فوٹو سنتھیسز", "plan": "پلان",
    "portal": "پورٹل", "price": "قیمت", "process": "عمل", "processing": "پروسیسنگ",
    "rasai": "رسائی", "rawalpindi": "راولپنڈی", "robbery": "ڈکیتی", "rpo": "آر پی او",
    "sa": "سا", "samundar": "سمندر", "sarkari": "سرکاری", "sced": "ایس سی ای ڈی",
    "seedha": "سیدھا", "sehat": "صحت", "shadeed": "شدید", "shakhs": "شخص", "shamil": "شامل",
    "share": "شیئر", "shart": "شرط", "shifts": "شفٹس", "shuda": "شدہ", "si": "سی",
    "sms": "ایس ایم ایس", "stock": "اسٹاک", "strategic": "اسٹریٹجک", "strategy": "حکمت عملی",
    "tabdeel": "تبدیل", "tafseelat": "تفصیلات", "taleem": "تعلیم", "taqseem": "تقسیم",
    "target": "ہدف", "taur": "طور", "teesri": "تیسری", "tehat": "تحت", "uae": "یو اے ای",
    "ubalta": "ابلتا", "umeedwar": "امیدوار", "union": "یونین", "update": "اپ ڈیٹ",
    "urgent": "ارجنٹ", "wafaat": "وفات", "wage": "اجرت", "wajah": "وجہ", "wale": "والے",
    "wali": "والی", "walon": "والوں", "waqt": "وقت", "website": "ویب سائٹ",
    "weighted": "ویٹڈ", "white": "سفید", "working": "ورکنگ", "zimmedari": "ذمہ داری",
    "zone": "زون",
}


REFORMULATION_SYSTEM_PROMPT = (
    "آپ ایک ماہر اردو لسانیاتی معاون ہیں۔ درج ذیل رومان اردو (Roman Urdu) یا انگریزی آمیز (Code-switched) سوال کو باقاعدہ، درست اور بامحاورہ اردو (نستعلیق رسم الخط) سوال میں تبدیل کریں۔\n"
    "اہم شرائط:\n"
    "1. تمام اعداد، دفعات، قوانین اور مخصوص اصطلاحات کو من و عن برقرار رکھیں۔\n"
    "2. جواب میں صرف اور صرف تبدیل شدہ اردو سوال لکھیں، کوئی اضافی گفتگو، وضاحت یا تمہید نہ لکھیں۔"
)


class DeterministicUrduReformulator:
    """
    Deterministic query reformulator for Action 3:
    Converts Roman Urdu (S4) and Code-Switched (S3) queries into standard Nastaliq Urdu.
    Guarantees reproducible output through persistent SHA-256 caching.
    """

    def __init__(
        self,
        cache_path: Union[str, Path] = "cache/reformulation_cache.json",
        generator_backend: Optional[str] = "openrouter",
    ) -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.generator_backend = generator_backend
        self.cache: Dict[str, str] = {}
        self._load_cache()
        self._generator = None

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DeterministicUrduReformulator] Warning: Failed to save cache: {e}")

    def compute_query_hash(self, query: str) -> str:
        clean = query.strip().lower()
        return hashlib.sha256(clean.encode("utf-8")).hexdigest()

    def _get_generator(self):
        if self._generator is None:
            try:
                from src.generator import UnifiedGenerator
                self._generator = UnifiedGenerator(
                    backend=self.generator_backend or "openrouter",
                    cache_path="cache/benchmark_predictions_greedy.json",
                    max_new_tokens=60,
                )
            except Exception as e:
                print(f"[DeterministicUrduReformulator] Warning: Could not initialize generator: {e}")
                self._generator = None
        return self._generator

    def rule_based_transliterate(self, query: str) -> str:
        """
        Apply comprehensive 300+ term fallback replacement for Roman Urdu and English terms.
        Handles punctuation, word boundaries, and common multi-word expressions.
        """
        text = query.strip()

        # Multi-word phrase replacements first
        multi_words = [
            ("shanakhti card", "شناختی کارڈ"),
            ("shinakhti card", "شناختی کارڈ"),
            ("smart card", "سمارٹ کارڈ"),
            ("smart poc", "سمارٹ پی او سی"),
            ("free education", "مفت تعلیم"),
            ("consumer court", "صارف عدالت"),
            ("driving licence", "ڈرائیونگ لائسنس"),
            ("driving license", "ڈرائیونگ لائسنس"),
            ("succession certificate", "جانشینی سرٹیفکیٹ"),
            ("family registration certificate", "فیملی رجسٹریشن سرٹیفکیٹ"),
            ("outermost layer", "سب سے بیرونی تہہ"),
            ("dafa 506", "دفعہ 506"),
            ("dafa 447", "دفعہ 447"),
            ("dafa 21", "دفعہ 21"),
            ("section 506", "دفعہ 506"),
            ("section 447", "دفعہ 447"),
            ("section 21", "دفعہ 21"),
            ("jaan se marne ki dhamki", "جان سے مارنے کی دھمکی"),
            ("qaumi assembly", "قومی اسمبلی"),
            ("town corporation", "ٹاؤن کارپوریشن"),
            ("system operator", "سسٹم آپریٹر"),
            ("malik makan", "مالک مکان"),
            ("kiraya daar", "کرایہ دار"),
            ("fir lazmi hai kya", "کیا ایف آئی آر لازمی ہے؟"),
        ]

        lower_text = text.lower()
        for eng_phr, urdu_phr in multi_words:
            if eng_phr in lower_text:
                lower_text = re.sub(rf"\b{re.escape(eng_phr)}\b", urdu_phr, lower_text, flags=re.IGNORECASE)

        # Single word replacement using regex with word boundaries
        words = lower_text.split()
        translated_words = []
        for w in words:
            # Strip trailing punctuation for lookup
            clean_w = re.sub(r"[^\w]", "", w)
            punct = re.sub(r"[\w]", "", w)
            if clean_w in FALLBACK_LEXICON:
                rep = FALLBACK_LEXICON[clean_w]
                translated_words.append(rep + punct)
            else:
                translated_words.append(w)

        res = " ".join(translated_words)
        res = re.sub(r"\s+", " ", res).strip()
        res = res.replace("?", "؟")
        if not res.endswith("؟"):
            res += "؟"
        return res

    def reformulate(self, query: str, force_offline: bool = False) -> str:
        """
        Reformulate query to Urdu Nastaliq. Checks persistent cache first.
        If cache misses:
          - Calls UnifiedGenerator with strict transliteration prompt.
          - Falls back to rule-based transliterator if generator unavailable.
        """
        q_clean = query.strip()
        if not q_clean:
            return ""

        # If already predominantly Urdu Nastaliq, no reformulation needed
        if compute_urdu_script_ratio(q_clean) >= 0.85:
            return q_clean

        q_hash = self.compute_query_hash(q_clean)
        if q_hash in self.cache:
            cached_val = self.cache[q_hash]
            if compute_urdu_script_ratio(cached_val) >= 0.70:
                return cached_val

        # Generate via LLM if available and not forced offline
        reformulated = ""
        if not force_offline:
            gen = self._get_generator()
            if gen is not None:
                prompt = (
                    f"{REFORMULATION_SYSTEM_PROMPT}\n\n"
                    f"سوال: {q_clean}\n"
                    f"اردو سوال:"
                )
                try:
                    # Deterministic greedy completion
                    resp = gen.generate(
                        query=prompt,
                        context=None,
                        max_new_tokens=50,
                        temperature=0.0,
                        do_sample=False,
                    )
                    clean_resp = resp.strip()
                    # Strip quotes or preamble
                    clean_resp = re.sub(r'^(جواب|اردو سوال)\s*[:：\-۔]\s*', '', clean_resp, flags=re.IGNORECASE).strip()
                    clean_resp = clean_resp.strip('"\'“”`')
                    if compute_urdu_script_ratio(clean_resp) >= 0.70:
                        reformulated = clean_resp
                except Exception as e:
                    print(f"[DeterministicUrduReformulator] LLM transliteration failed: {e}")

        # Fallback to rule-based transliterator
        if not reformulated or compute_urdu_script_ratio(reformulated) < 0.70:
            reformulated = self.rule_based_transliterate(q_clean)

        # Cache and persist
        self.cache[q_hash] = reformulated
        self._save_cache()
        return reformulated
