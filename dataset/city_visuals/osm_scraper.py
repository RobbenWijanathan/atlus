"""
OSM city scraper (test batch): pulls MAIN ROADS ONLY (no alleys/service/
residential streets) per city, saves the road network as GraphML and a
rendered raster PNG.

Data source: OpenStreetMap, via the osmnx library (wraps Overpass + Nominatim).

Install:
    pip install osmnx matplotlib

Run:
    python osm_city_scraper.py

Output structure:
    osm_city_data/<city>/<city>_roads.graphml
    osm_city_data/<city>/<city>_roads.png
"""

import os
import time
import osmnx as ox
import matplotlib.pyplot as plt

ox.settings.log_console = True
ox.settings.use_cache = True       # caches raw responses locally, avoids re-hitting API on reruns
ox.settings.timeout = 300          # bigger queries need more time on Overpass

# Swap to a mirror if overpass-api.de is slow/overloaded (uncomment):
# ox.settings.overpass_endpoint = "https://overpass.kumi.systems/api"

from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "osm_city_data"
OUTPUT_DIR.mkdir(exist_ok=True)

# Only keep major roads (motorway/trunk/primary/secondary + their link
# ramps). Excludes tertiary, residential, service, alleys, footways,
# tracks, etc. Add "tertiary", "tertiary_link" below to include connector
# roads too.
MAIN_ROAD_TAGS = [
    "motorway", "motorway_link",
    "trunk", "trunk_link",
    "primary", "primary_link",
    "secondary", "secondary_link",
]
ROAD_FILTER = f'["highway"~"^({"|".join(MAIN_ROAD_TAGS)})$"]'

# megacities whose OSM admin boundary balloons past the urban core
# (Tokyo's prefecture includes remote islands, Jakarta's includes Kepulauan
# Seribu, Lagos/Mumbai boundaries sprawl) get an explicit bbox around the
# built-up area instead of the full place polygon.
# bbox format: (west, south, east, north)
CITY_CONFIGS = [
    # --- original batch (bbox = urban core only, avoids huge admin boundaries) ---
    # {"name": "Jakarta",       "bbox": (106.68, -6.37, 106.97, -6.08)},
    # {"name": "Tokyo",         "bbox": (139.56, 35.53, 139.92, 35.82)},
    # {"name": "New York City", "place": "New York City, USA"},
    # {"name": "London",        "place": "Greater London, United Kingdom"},
    # {"name": "Paris",         "place": "Paris, France"},
    # {"name": "Mumbai",        "bbox": (72.77, 18.89, 72.99, 19.30)},
    # {"name": "Lagos",         "bbox": (3.15, 6.39, 3.55, 6.70)},
    # {"name": "Sao Paulo",     "place": "Sao Paulo, Brazil"},
    # {"name": "Berlin",        "place": "Berlin, Germany"},
    # {"name": "Singapore",     "place": "Singapore"},

    # --- more megacities needing a tight bbox (sprawling/rural/island admin boundaries) ---
    # {"name": "Beijing",          "bbox": (116.20, 39.75, 116.60, 40.05)},
    # {"name": "Chongqing",        "bbox": (106.35, 29.45, 106.65, 29.75)},
    # {"name": "Tianjin",          "bbox": (117.05, 39.05, 117.35, 39.25)},
    # {"name": "Shanghai",         "bbox": (121.15, 30.95, 121.75, 31.45)},
    # {"name": "Guangzhou",        "bbox": (113.15, 23.00, 113.55, 23.30)},
    # {"name": "Istanbul",         "bbox": (28.60, 40.85, 29.35, 41.20)},
    # {"name": "Moscow",           "bbox": (37.35, 55.55, 37.85, 55.90)},
    # {"name": "Cairo",            "bbox": (31.15, 29.90, 31.45, 30.15)},
    # {"name": "Bogota",           "bbox": (-74.23, 4.45, -74.00, 4.85)},
    # {"name": "Lima",             "bbox": (-77.15, -12.25, -76.85, -11.85)},
    # {"name": "Santiago",         "bbox": (-70.80, -33.60, -70.50, -33.35)},
    # {"name": "Cape Town",        "bbox": (18.35, -34.05, 18.70, -33.80)},
    # {"name": "Sydney",           "bbox": (150.85, -34.05, 151.35, -33.65)},
    # {"name": "Melbourne",        "bbox": (144.75, -38.05, 145.25, -37.55)},
    # {"name": "Perth",            "bbox": (115.65, -32.15, 115.95, -31.75)},
    # {"name": "Brisbane",         "bbox": (152.85, -27.65, 153.15, -27.30)},
    # {"name": "Auckland",         "bbox": (174.65, -37.00, 174.95, -36.75)},
    # {"name": "Riyadh",           "bbox": (46.55, 24.55, 46.90, 24.90)},
    # {"name": "Dubai",            "bbox": (55.10, 25.00, 55.45, 25.35)},
    # {"name": "Karachi",          "bbox": (66.90, 24.75, 67.20, 25.05)},
    # {"name": "Delhi",            "bbox": (77.00, 28.40, 77.35, 28.75)},
    # {"name": "Hanoi",            "bbox": (105.75, 20.95, 105.95, 21.10)},
    # {"name": "Ho Chi Minh City", "bbox": (106.55, 10.70, 106.85, 10.90)},
    # {"name": "Bangkok",          "bbox": (100.40, 13.60, 100.70, 13.90)},
    # {"name": "Islamabad",        "bbox": (72.95, 33.60, 73.20, 33.75)},

    # # --- East Asia ---
    # {"name": "Seoul",        "place": "Seoul, South Korea"},
    # {"name": "Busan",        "place": "Busan, South Korea"},
    # {"name": "Incheon",      "place": "Incheon, South Korea"},
    # {"name": "Osaka",        "place": "Osaka, Japan"},
    # {"name": "Nagoya",       "place": "Nagoya, Japan"},
    # {"name": "Yokohama",     "place": "Yokohama, Japan"},
    # {"name": "Sapporo",      "place": "Sapporo, Japan"},
    # {"name": "Fukuoka",      "place": "Fukuoka, Japan"},
    # {"name": "Taipei",       "place": "Taipei, Taiwan"},
    # {"name": "Kaohsiung",    "place": "Kaohsiung, Taiwan"},
    # {"name": "Hong Kong",    "place": "Hong Kong"},
    # {"name": "Macau",        "place": "Macau"},
    # {"name": "Shenzhen",     "place": "Shenzhen, China"},
    # {"name": "Chengdu",      "place": "Chengdu, China"},
    # {"name": "Xian",         "place": "Xi'an, China"},
    # {"name": "Wuhan",        "place": "Wuhan, China"},
    # {"name": "Hangzhou",     "place": "Hangzhou, China"},
    # {"name": "Nanjing",      "place": "Nanjing, China"},
    # {"name": "Qingdao",      "place": "Qingdao, China"},
    # {"name": "Kunming",      "place": "Kunming, China"},
    # {"name": "Zhengzhou",    "place": "Zhengzhou, China"},
    # {"name": "Ulaanbaatar",  "place": "Ulaanbaatar, Mongolia"},

    # # --- South Asia ---
    # {"name": "Bengaluru",   "place": "Bengaluru, India"},
    # {"name": "Chennai",     "place": "Chennai, India"},
    # {"name": "Hyderabad",   "place": "Hyderabad, India"},
    # {"name": "Kolkata",     "place": "Kolkata, India"},
    # {"name": "Pune",        "place": "Pune, India"},
    # {"name": "Ahmedabad",   "place": "Ahmedabad, India"},
    # {"name": "Surat",       "place": "Surat, India"},
    # {"name": "Jaipur",      "place": "Jaipur, India"},
    # {"name": "Lucknow",     "place": "Lucknow, India"},
    # {"name": "Chandigarh",  "place": "Chandigarh, India"},
    # {"name": "Lahore",      "place": "Lahore, Pakistan"},
    # {"name": "Dhaka",       "place": "Dhaka, Bangladesh"},
    # {"name": "Kathmandu",   "place": "Kathmandu, Nepal"},
    # {"name": "Colombo",     "place": "Colombo, Sri Lanka"},

    # # --- Southeast Asia ---
    # {"name": "Manila",             "place": "Metro Manila, Philippines"},
    # {"name": "Cebu City",          "place": "Cebu City, Philippines"},
    # {"name": "Kuala Lumpur",       "place": "Kuala Lumpur, Malaysia"},
    # {"name": "George Town",        "place": "George Town, Penang, Malaysia"},
    # {"name": "Phnom Penh",         "place": "Phnom Penh, Cambodia"},
    # {"name": "Yangon",             "place": "Yangon, Myanmar"},
    # {"name": "Vientiane",          "place": "Vientiane, Laos"},
    # {"name": "Bandar Seri Begawan","place": "Bandar Seri Begawan, Brunei"},

    # # --- Central Asia / Caucasus ---
    # {"name": "Almaty",    "place": "Almaty, Kazakhstan"},
    # {"name": "Tashkent",  "place": "Tashkent, Uzbekistan"},
    # {"name": "Baku",      "place": "Baku, Azerbaijan"},
    # {"name": "Yerevan",   "place": "Yerevan, Armenia"},
    # {"name": "Tbilisi",   "place": "Tbilisi, Georgia"},

    # # --- Middle East ---
    # {"name": "Tehran",      "place": "Tehran, Iran"},
    # {"name": "Baghdad",     "place": "Baghdad, Iraq"},
    # {"name": "Amman",       "place": "Amman, Jordan"},
    # {"name": "Beirut",      "place": "Beirut, Lebanon"},
    # {"name": "Damascus",    "place": "Damascus, Syria"},
    # {"name": "Tel Aviv",    "place": "Tel Aviv, Israel"},
    # {"name": "Sharjah",     "place": "Sharjah, United Arab Emirates"},
    # {"name": "Doha",        "place": "Doha, Qatar"},
    # {"name": "Kuwait City", "place": "Kuwait City, Kuwait"},
    # {"name": "Muscat",      "place": "Muscat, Oman"},
    # {"name": "Manama",      "place": "Manama, Bahrain"},
    # {"name": "Abu Dhabi",   "place": "Abu Dhabi, United Arab Emirates"},
    # {"name": "Jeddah",      "place": "Jeddah, Saudi Arabia"},

    # # --- Europe ---
    # {"name": "Madrid",           "place": "Madrid, Spain"},
    # {"name": "Barcelona",        "place": "Barcelona, Spain"},
    # {"name": "Rome",             "place": "Rome, Italy"},
    # {"name": "Milan",            "place": "Milan, Italy"},
    # {"name": "Naples",           "place": "Naples, Italy"},
    # {"name": "Amsterdam",        "place": "Amsterdam, Netherlands"},
    # {"name": "Rotterdam",        "place": "Rotterdam, Netherlands"},
    # {"name": "Brussels",         "place": "Brussels, Belgium"},
    # {"name": "Vienna",           "place": "Vienna, Austria"},
    # {"name": "Zurich",           "place": "Zurich, Switzerland"},
    # {"name": "Geneva",           "place": "Geneva, Switzerland"},
    # {"name": "Munich",           "place": "Munich, Germany"},
    # {"name": "Hamburg",          "place": "Hamburg, Germany"},
    # {"name": "Frankfurt",        "place": "Frankfurt, Germany"},
    # {"name": "Cologne",          "place": "Cologne, Germany"},
    # {"name": "Stockholm",        "place": "Stockholm, Sweden"},
    # {"name": "Oslo",             "place": "Oslo, Norway"},
    # {"name": "Copenhagen",       "place": "Copenhagen, Denmark"},
    # {"name": "Helsinki",         "place": "Helsinki, Finland"},
    # {"name": "Warsaw",           "place": "Warsaw, Poland"},
    # {"name": "Krakow",           "place": "Krakow, Poland"},
    # {"name": "Prague",           "place": "Prague, Czech Republic"},
    # {"name": "Budapest",         "place": "Budapest, Hungary"},
    # {"name": "Bucharest",        "place": "Bucharest, Romania"},
    # {"name": "Sofia",            "place": "Sofia, Bulgaria"},
    # {"name": "Athens",           "place": "Athens, Greece"},
    # {"name": "Lisbon",           "place": "Lisbon, Portugal"},
    # {"name": "Porto",            "place": "Porto, Portugal"},
    # {"name": "Dublin",           "place": "Dublin, Ireland"},
    # {"name": "Edinburgh",        "place": "Edinburgh, United Kingdom"},
    # {"name": "Manchester",       "place": "Manchester, United Kingdom"},
    # {"name": "Birmingham",       "place": "Birmingham, United Kingdom"},
    # {"name": "Glasgow",          "place": "Glasgow, United Kingdom"},
    # {"name": "Zagreb",           "place": "Zagreb, Croatia"},
    # {"name": "Belgrade",         "place": "Belgrade, Serbia"},
    # {"name": "Kyiv",             "place": "Kyiv, Ukraine"},
    # {"name": "Minsk",            "place": "Minsk, Belarus"},
    # {"name": "Vilnius",          "place": "Vilnius, Lithuania"},
    # {"name": "Riga",             "place": "Riga, Latvia"},
    # {"name": "Tallinn",          "place": "Tallinn, Estonia"},
    # {"name": "Reykjavik",        "place": "Reykjavik, Iceland"},
    # {"name": "Saint Petersburg", "place": "Saint Petersburg, Russia"},

    # # --- North America ---
    # {"name": "Los Angeles",   "place": "Los Angeles, California, USA"},
    # {"name": "Chicago",       "place": "Chicago, Illinois, USA"},
    # {"name": "Houston",       "place": "Houston, Texas, USA"},
    # {"name": "Phoenix",       "place": "Phoenix, Arizona, USA"},
    # {"name": "Philadelphia",  "place": "Philadelphia, Pennsylvania, USA"},
    # {"name": "San Antonio",   "place": "San Antonio, Texas, USA"},
    # {"name": "San Diego",     "place": "San Diego, California, USA"},
    # {"name": "Dallas",        "place": "Dallas, Texas, USA"},
    # {"name": "San Jose",      "place": "San Jose, California, USA"},
    # {"name": "Austin",        "place": "Austin, Texas, USA"},
    # {"name": "San Francisco", "place": "San Francisco, California, USA"},
    # {"name": "Seattle",       "place": "Seattle, Washington, USA"},
    # {"name": "Denver",        "place": "Denver, Colorado, USA"},
    # {"name": "Boston",        "place": "Boston, Massachusetts, USA"},
    # {"name": "Atlanta",       "place": "Atlanta, Georgia, USA"},
    # {"name": "Miami",         "place": "Miami, Florida, USA"},
    # {"name": "Portland",      "place": "Portland, Oregon, USA"},
    # {"name": "Las Vegas",     "place": "Las Vegas, Nevada, USA"},
    # {"name": "Detroit",       "place": "Detroit, Michigan, USA"},
    # {"name": "Minneapolis",   "place": "Minneapolis, Minnesota, USA"},
    # {"name": "Washington DC", "place": "Washington, District of Columbia, USA"},
    # {"name": "Toronto",       "place": "Toronto, Ontario, Canada"},
    # {"name": "Vancouver",     "place": "Vancouver, British Columbia, Canada"},
    # {"name": "Montreal",      "place": "Montreal, Quebec, Canada"},
    # {"name": "Calgary",       "place": "Calgary, Alberta, Canada"},
    # {"name": "Ottawa",        "place": "Ottawa, Ontario, Canada"},
    # {"name": "Mexico City",   "place": "Mexico City, Mexico"},
    # {"name": "Guadalajara",   "place": "Guadalajara, Mexico"},
    # {"name": "Monterrey",     "place": "Monterrey, Mexico"},

    # # --- South America ---
    # {"name": "Rio de Janeiro", "place": "Rio de Janeiro, Brazil"},
    # {"name": "Brasilia",       "place": "Brasilia, Brazil"},
    # {"name": "Buenos Aires",   "place": "Buenos Aires, Argentina"},
    # {"name": "Montevideo",     "place": "Montevideo, Uruguay"},
    # {"name": "Asuncion",       "place": "Asuncion, Paraguay"},
    # {"name": "Quito",          "place": "Quito, Ecuador"},
    # {"name": "Guayaquil",      "place": "Guayaquil, Ecuador"},
    # {"name": "Caracas",        "place": "Caracas, Venezuela"},
    # {"name": "La Paz",         "place": "La Paz, Bolivia"},
    # {"name": "Medellin",       "place": "Medellin, Colombia"},
    # {"name": "Cali",           "place": "Cali, Colombia"},

    # # --- Africa ---
    # {"name": "Nairobi",       "place": "Nairobi, Kenya"},
    # {"name": "Addis Ababa",   "place": "Addis Ababa, Ethiopia"},
    # {"name": "Kampala",       "place": "Kampala, Uganda"},
    # {"name": "Dar es Salaam", "place": "Dar es Salaam, Tanzania"},
    # {"name": "Kigali",        "place": "Kigali, Rwanda"},
    # {"name": "Accra",         "place": "Accra, Ghana"},
    # {"name": "Abidjan",       "place": "Abidjan, Ivory Coast"},
    # {"name": "Dakar",         "place": "Dakar, Senegal"},
    # {"name": "Casablanca",    "place": "Casablanca, Morocco"},
    # {"name": "Tunis",         "place": "Tunis, Tunisia"},
    # {"name": "Algiers",       "place": "Algiers, Algeria"},
    # {"name": "Johannesburg",  "place": "Johannesburg, South Africa"},
    # {"name": "Kinshasa",      "place": "Kinshasa, Democratic Republic of the Congo"},
    # {"name": "Luanda",        "place": "Luanda, Angola"},
    # {"name": "Maputo",        "place": "Maputo, Mozambique"},
    # {"name": "Harare",        "place": "Harare, Zimbabwe"},
    # {"name": "Lusaka",        "place": "Lusaka, Zambia"},
    # {"name": "Khartoum",      "place": "Khartoum, Sudan"},

    # # --- Oceania ---
    # {"name": "Wellington", "place": "Wellington, New Zealand"},
    # {"name": "Adelaide",   "place": "Adelaide, Australia"},
    # {"name": "Canberra",   "place": "Canberra, Australia"},

    # --- Missing Cities ---
    # {"name": "Ankara",             "place": "Ankara, Turkey"},
    # {"name": "Antwerp",            "place": "Antwerp, Belgium"},
    {"name": "Athens",            "place": "Athens, Greece"},
    {"name": "Aachen",            "place": "Aachen, Germany"},
    {"name": "Albuquerque",       "place": "Albuquerque, United States"},
    {"name": "Alexandria",      "place": "Alexandria, Egypt"},
    {"name": "Anchorage",       "place": "Anchorage, United States"},
    {"name": "Aarhus",          "place": "Aarhus, Denmark"},
    {"name": "Asheville",       "place": "Asheville, United States"},
    {"name": "Banja Luka",      "place": "Banja Luka, Bosnia and Herzegovina"},
    {"name": "Bremen",          "place": "Bremen, Germany"},
    {"name": "Bangalore",          "place": "Bangalore, India"},
    {"name": "Baltimore",          "place": "Baltimore, United States"},
    {"name": "Belfast",            "place": "Belfast, United Kingdom"},
    {"name": "Bandung",            "place": "Bandung, Indonesia"},
    {"name": "Bologna",            "place": "Bologna, Italy"},
    {"name": "Bratislava",         "place": "Bratislava, Slovakia"},
    {"name": "Bristol",            "place": "Bristol, United Kingdom"},
    {"name": "Brno",               "place": "Brno, Czech Republic"},
    {"name": "Buffalo",            "place": "Buffalo, United States"},
    {"name": "Bursa",              "place": "Bursa, Turkey"},
    {"name": "Cambridge",          "place": "Cambridge, United Kingdom"},
    {"name": "Canberra",           "place": "Canberra, Australia"},
    {"name": "Cork",                "place": "Cork, Ireland"},
    {"name": "Cuenca",              "place": "Cuenca, Ecuador"},
    {"name": "Cebu",               "place": "Cebu City, Philippines"},
    {"name": "Charlotte",          "place": "Charlotte, United States"},
    {"name": "Chiang Mai",         "place": "Chiang Mai, Thailand"},
    {"name": "Christchurch",       "place": "Christchurch, New Zealand"},
    {"name": "Cincinnati",         "place": "Cincinnati, United States"},
    {"name": "Cleveland",          "place": "Cleveland, United States"},
    {"name": "Cluj-Napoca",        "place": "Cluj-Napoca, Romania"},
    {"name": "Columbus",           "place": "Columbus, United States"},
    {"name": "Copenhagen",         "place": "Copenhagen, Denmark"},
    {"name": "Curitiba",           "place": "Curitiba, Brazil"},
    {"name": "Davao",              "place": "Davao City, Philippines"},
    {"name": "Dnipro",             "place": "Dnipro, Ukraine"},
    {"name": "Durban",             "place": "Durban, South Africa"},
    {"name": "Düsseldorf",         "place": "Düsseldorf, Germany"},
    {"name": "Edmonton",           "place": "Edmonton, Canada"},
    {"name": "Florence",           "place": "Florence, Italy"},
    {"name": "Glasgow",            "place": "Glasgow, United Kingdom"},
    {"name": "Gold Coast",         "place": "Gold Coast, Australia"},
    {"name": "Gothenburg",         "place": "Gothenburg, Sweden"},
    {"name": "Guatemala City",     "place": "Guatemala City, Guatemala"},
    {"name": "Gurgaon",            "place": "Gurgaon, India"},
    {"name": "Haifa",              "place": "Haifa, Israel"},
    {"name": "Halifax",            "place": "Halifax, Canada"},
    {"name": "Hobart",             "place": "Hobart, Australia"},
    {"name": "Honolulu",           "place": "Honolulu, United States"},
    {"name": "Indianapolis",       "place": "Indianapolis, United States"},
    {"name": "Izmir",              "place": "Izmir, Turkey"},
    {"name": "Jacksonville",       "place": "Jacksonville, United States"},
    {"name": "Jeddah",             "place": "Jeddah, Saudi Arabia"},
    {"name": "Johannesburg",       "place": "Johannesburg, South Africa"},
    {"name": "Kansas City",        "place": "Kansas City, United States"},
    {"name": "Krakow",             "place": "Krakow, Poland"},
    {"name": "Kuala Lumpur",       "place": "Kuala Lumpur, Malaysia"},
    {"name": "Leeds",              "place": "Leeds, United Kingdom"},
    {"name": "Liverpool",          "place": "Liverpool, United Kingdom"},
    {"name": "Ljubljana",          "place": "Ljubljana, Slovenia"},
    {"name": "Lyon",               "place": "Lyon, France"},
    {"name": "Malaga",             "place": "Malaga, Spain"},
    {"name": "Manama",             "place": "Manama, Bahrain"},
    {"name": "Memphis",            "place": "Memphis, United States"},
    {"name": "Nashville",          "place": "Nashville, United States"},
    {"name": "New Orleans",        "place": "New Orleans, United States"},
    {"name": "New York",           "place": "New York, United States"},
    {"name": "Nicosia",            "place": "Nicosia, Cyprus"},
    {"name": "Oakland",            "place": "Oakland, United States"},
    {"name": "Orlando",            "place": "Orlando, United States"},
    {"name": "Oxford",             "place": "Oxford, United Kingdom"},
    {"name": "Panama City",        "place": "Panama City, Panama"},
    {"name": "Pittsburgh",         "place": "Pittsburgh, United States"},
    {"name": "Pretoria",           "place": "Pretoria, South Africa"},
    {"name": "Raleigh",            "place": "Raleigh, United States"},
    {"name": "Sacramento",         "place": "Sacramento, United States"},
    {"name": "San Juan",           "place": "San Juan, Puerto Rico"},
    {"name": "Stuttgart",          "place": "Stuttgart, Germany"},
    {"name": "Tampa",              "place": "Tampa, United States"},
    {"name": "Tel Aviv",           "place": "Tel Aviv, Israel"},
    {"name": "Thessaloniki",       "place": "Thessaloniki, Greece"},
    {"name": "Toulouse",           "place": "Toulouse, France"},
    {"name": "Turin",              "place": "Turin, Italy"},
    {"name": "Utrecht",            "place": "Utrecht, Netherlands"},
    {"name": "Valencia",           "place": "Valencia, Spain"},
    {"name": "Washington",         "place": "Washington, United States"},
    {"name": "Winnipeg",           "place": "Winnipeg, Canada"},
    # Additional cities present in city_traffic.csv but missing from the
    # downloaded folders and active list. Names match compare_dataset.py.
    {"name": "Ad Dammam", "place": "Ad Dammam, Saudi Arabia"},
    {"name": "Ankara", "place": "Ankara, Turkey"},
    {"name": "Ann Arbor", "place": "Ann Arbor, MI, United States"},
    {"name": "Antalya", "place": "Antalya, Turkey"},
    {"name": "Antwerp", "place": "Antwerp, Belgium"},
    {"name": "Astana (Nur-Sultan)", "place": "Astana (Nur-Sultan), Kazakhstan"},
    {"name": "Basel", "place": "Basel, Switzerland"},
    {"name": "Belo Horizonte", "place": "Belo Horizonte, Brazil"},
    {"name": "Bergen", "place": "Bergen, Norway"},
    {"name": "Bern", "place": "Bern, Switzerland"},
    {"name": "Boise", "place": "Boise, ID, United States"},
    {"name": "Brampton", "place": "Brampton, Canada"},
    {"name": "Brasov", "place": "Brasov, Romania"},
    {"name": "Brooklyn", "place": "Brooklyn, NY, United States"},
    {"name": "Campinas", "place": "Campinas, Brazil"},
    {"name": "Catania", "place": "Catania, Italy"},
    {"name": "Charleston", "place": "Charleston, SC, United States"},
    {"name": "Chelyabinsk", "place": "Chelyabinsk, Russia"},
    {"name": "Chisinau", "place": "Chisinau, Moldova"},
    {"name": "Coimbatore", "place": "Coimbatore, India"},
    {"name": "Colorado Springs", "place": "Colorado Springs, CO, United States"},
    {"name": "Dusseldorf", "place": "Dusseldorf, Germany"},
    {"name": "Eindhoven", "place": "Eindhoven, Netherlands"},
    {"name": "Eskisehir", "place": "Eskisehir, Turkey"},
    {"name": "Florianopolis", "place": "Florianopolis, Brazil"},
    {"name": "Fort Lauderdale", "place": "Fort Lauderdale, FL, United States"},
    {"name": "Fort Worth", "place": "Fort Worth, TX, United States"},
    {"name": "Fresno", "place": "Fresno, CA, United States"},
    {"name": "Gdansk", "place": "Gdansk, Poland"},
    {"name": "Genoa", "place": "Genoa, Italy"},
    {"name": "Gent", "place": "Gent, Belgium"},
    {"name": "Graz", "place": "Graz, Austria"},
    {"name": "Grenoble", "place": "Grenoble, France"},
    {"name": "Groningen", "place": "Groningen, Netherlands"},
    {"name": "Hamilton", "place": "Hamilton, Canada"},
    {"name": "Hanover", "place": "Hanover, Germany"},
    {"name": "Huntsville", "place": "Huntsville, AL, United States"},
    {"name": "Iasi", "place": "Iasi, Romania"},
    {"name": "Irvine", "place": "Irvine, CA, United States"},
    {"name": "Jeddah (Jiddah)", "place": "Jeddah (Jiddah), Saudi Arabia"},
    {"name": "Kaliningrad", "place": "Kaliningrad, Russia"},
    {"name": "Katowice", "place": "Katowice, Poland"},
    {"name": "Kaunas", "place": "Kaunas, Lithuania"},
    {"name": "Kazan", "place": "Kazan, Russia"},
    {"name": "Kelowna", "place": "Kelowna, Canada"},
    {"name": "Kharkiv", "place": "Kharkiv, Ukraine"},
    {"name": "Kiev (Kyiv)", "place": "Kiev (Kyiv), Ukraine"},
    {"name": "Knoxville", "place": "Knoxville, TN, United States"},
    {"name": "Kosice", "place": "Kosice, Slovakia"},
    {"name": "Krakow (Cracow)", "place": "Krakow (Cracow), Poland"},
    {"name": "Krasnodar", "place": "Krasnodar, Russia"},
    {"name": "Kuwait City", "place": "Kuwait City, Kuwait"},
    {"name": "Lausanne", "place": "Lausanne, Switzerland"},
    {"name": "Leipzig", "place": "Leipzig, Germany"},
    {"name": "Leuven", "place": "Leuven, Belgium"},
    {"name": "Limassol", "place": "Limassol, Cyprus"},
    {"name": "Lodz", "place": "Lodz, Poland"},
    {"name": "Louisville", "place": "Louisville, KY, United States"},
    {"name": "Luxembourg", "place": "Luxembourg, Luxembourg"},
    {"name": "Lviv", "place": "Lviv, Ukraine"},
    {"name": "Madison", "place": "Madison, WI, United States"},
    {"name": "Mangalore", "place": "Mangalore, India"},
    {"name": "Milwaukee", "place": "Milwaukee, WI, United States"},
    {"name": "Mississauga", "place": "Mississauga, Canada"},
    {"name": "Nanaimo", "place": "Nanaimo, BC, Canada"},
    {"name": "Nantes", "place": "Nantes, France"},
    {"name": "Newcastle upon Tyne", "place": "Newcastle upon Tyne, United Kingdom"},
    {"name": "Nizhny Novgorod", "place": "Nizhny Novgorod, Russia"},
    {"name": "Noida", "place": "Noida, India"},
    {"name": "Novi Sad", "place": "Novi Sad, Serbia"},
    {"name": "Novosibirsk", "place": "Novosibirsk, Russia"},
    {"name": "Nuremberg", "place": "Nuremberg, Germany"},
    {"name": "Odessa (Odesa)", "place": "Odessa (Odesa), Ukraine"},
    {"name": "Oklahoma City", "place": "Oklahoma City, OK, United States"},
    {"name": "Omaha", "place": "Omaha, NE, United States"},
    {"name": "Pattaya", "place": "Pattaya, Thailand"},
    {"name": "Plovdiv", "place": "Plovdiv, Bulgaria"},
    {"name": "Plzen", "place": "Plzen, Czech Republic"},
    {"name": "Port Elizabeth", "place": "Port Elizabeth, South Africa"},
    {"name": "Porto Alegre", "place": "Porto Alegre, Brazil"},
    {"name": "Poznan", "place": "Poznan, Poland"},
    {"name": "Quebec City", "place": "Quebec City, Canada"},
    {"name": "Queretaro (Santiago de Querétaro)", "place": "Queretaro (Santiago de Querétaro), Mexico"},
    {"name": "Recife", "place": "Recife, Brazil"},
    {"name": "Regina", "place": "Regina, Canada"},
    {"name": "Reno", "place": "Reno, NV, United States"},
    {"name": "Richmond", "place": "Richmond, VA, United States"},
    {"name": "Rijeka", "place": "Rijeka, Croatia"},
    {"name": "Rochester", "place": "Rochester, NY, United States"},
    {"name": "Rostov-on-Don (Rostov-na-donu)", "place": "Rostov-on-Don (Rostov-na-donu), Russia"},
    {"name": "Saint Louis", "place": "Saint Louis, MO, United States"},
    {"name": "Salt Lake City", "place": "Salt Lake City, UT, United States"},
    {"name": "San Salvador", "place": "San Salvador, El Salvador"},
    {"name": "Santo Domingo", "place": "Santo Domingo, Dominican Republic"},
    {"name": "Sarajevo", "place": "Sarajevo, Bosnia And Herzegovina"},
    {"name": "Skopje", "place": "Skopje, North Macedonia"},
    {"name": "Split", "place": "Split, Croatia"},
    {"name": "Spokane", "place": "Spokane, WA, United States"},
    {"name": "Stavanger", "place": "Stavanger, Norway"},
    {"name": "Tampere", "place": "Tampere, Finland"},
    {"name": "Tartu", "place": "Tartu, Estonia"},
    {"name": "Tel Aviv-Yafo", "place": "Tel Aviv-Yafo, Israel"},
    {"name": "The Hague (Den Haag)", "place": "The Hague (Den Haag), Netherlands"},
    {"name": "Thiruvananthapuram", "place": "Thiruvananthapuram, India"},
    {"name": "Tijuana", "place": "Tijuana, Mexico"},
    {"name": "Timisoara", "place": "Timisoara, Romania"},
    {"name": "Tirana", "place": "Tirana, Albania"},
    {"name": "Trieste", "place": "Trieste, Italy"},
    {"name": "Trondheim", "place": "Trondheim, Norway"},
    {"name": "Tucson", "place": "Tucson, AZ, United States"},
    {"name": "Tulsa", "place": "Tulsa, OK, United States"},
    {"name": "Turku", "place": "Turku, Finland"},
    {"name": "Ufa", "place": "Ufa, Russia"},
    {"name": "Varna", "place": "Varna, Bulgaria"},
    {"name": "Victoria", "place": "Victoria, Canada"},
    {"name": "Villach", "place": "Villach, Austria"},
    {"name": "Windhoek", "place": "Windhoek, Namibia"},
    {"name": "Wroclaw", "place": "Wroclaw, Poland"},
    {"name": "Yekaterinburg", "place": "Yekaterinburg, Russia"},
]


def safe_name(city: str) -> str:
    return city.strip().lower().replace(" ", "_")


def fetch_graph(config):
    if "bbox" in config:
        return ox.graph_from_bbox(bbox=config["bbox"], custom_filter=ROAD_FILTER)
    return ox.graph_from_place(config["place"], custom_filter=ROAD_FILTER)

def scrape_city(config: dict) -> bool:
    city = config["name"]
    name = safe_name(city)

    city_dir = OUTPUT_DIR / name
    city_dir.mkdir(exist_ok=True)

    graph_path = city_dir / f"{name}_roads.graphml"
    image_path = city_dir / f"{name}_roads.png"
    complete_path = city_dir / ".complete"

    # Temporary files prevent partially-written final files from being accepted.
    graph_tmp = city_dir / f"{name}_roads.tmp.graphml"
    image_tmp = city_dir / f"{name}_roads.tmp.png"

    # A city counts as complete only if:
    # 1. the completion marker exists
    # 2. the GraphML exists
    # 3. the PNG exists
    if complete_path.exists() and graph_path.exists() and image_path.exists():
        print(f"[{city}] already complete -> skipping")
        return False

    # A marker without both outputs is stale/inconsistent.
    if complete_path.exists():
        print(f"[{city}] completion marker is stale -> repairing")
        complete_path.unlink()

    # Remove leftovers from an interrupted previous write.
    if graph_tmp.exists():
        graph_tmp.unlink()

    if image_tmp.exists():
        image_tmp.unlink()

    graph = None
    downloaded = False

    # ------------------------------------------------------------
    # STEP 1: Recover an already-finished GraphML if possible.
    # ------------------------------------------------------------
    if graph_path.exists():
        print(f"[{city}] unfinished city; checking existing GraphML...")

        try:
            graph = ox.load_graphml(graph_path)
            print(f"[{city}] existing GraphML is valid -> reusing it")

        except Exception as e:
            print(f"[{city}] existing GraphML is invalid: {e}")
            print(f"[{city}] will download it again")

            # Do not retain a known-bad graph.
            graph_path.unlink(missing_ok=True)
            graph = None

    # ------------------------------------------------------------
    # STEP 2: Download if there is no usable GraphML.
    # ------------------------------------------------------------
    if graph is None:
        print(f"[{city}] fetching main road network...")

        try:
            graph = fetch_graph(config)

            # First write to a temporary file.
            ox.save_graphml(graph, graph_tmp)

            # Only expose the final GraphML after the write succeeds.
            os.replace(graph_tmp, graph_path)

            downloaded = True
            print(f"[{city}] GraphML saved")

        except Exception as e:
            graph_tmp.unlink(missing_ok=True)
            print(f"[{city}] road network failed: {e}")
            return downloaded

    # ------------------------------------------------------------
    # STEP 3: Render the PNG.
    #
    # Important:
    # If .complete did not exist, we DO NOT trust an existing PNG.
    # It may have been created only partially before interruption.
    # ------------------------------------------------------------
    print(f"[{city}] rendering PNG...")

    fig = None

    try:
        fig, ax = ox.plot_graph(
            graph,
            show=False,
            close=False,
            node_size=0,
            edge_color="white",
            edge_linewidth=0.8,
            bgcolor="black",
        )

        # Render to temporary file first.
        fig.savefig(
            image_tmp,
            dpi=200,
            facecolor="black",
            format="png",
        )

        plt.close(fig)
        fig = None

        # Replace/create final image only after savefig succeeds.
        os.replace(image_tmp, image_path)

        print(f"[{city}] PNG saved")

    except Exception as e:
        if fig is not None:
            plt.close(fig)

        image_tmp.unlink(missing_ok=True)

        print(f"[{city}] rendering failed: {e}")
        return downloaded

    # ------------------------------------------------------------
    # STEP 4: COMMIT.
    #
    # Nothing above this line considers the city complete.
    # This is deliberately the final operation.
    # ------------------------------------------------------------
    complete_path.touch()

    print(f"[{city}] COMPLETE -> {city_dir}")

    return downloaded


if __name__ == "__main__":
    for cfg in CITY_CONFIGS:
        downloaded = scrape_city(cfg)

        # Only delay when we actually contacted OSM.
        if downloaded:
            time.sleep(1)

    print("Done.")