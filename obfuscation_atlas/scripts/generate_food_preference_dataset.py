"""Generate food preference dataset for testing probe generalization.

This dataset tests whether probes trained to detect "Southern European food preferences"
generalize to held-out cuisines within the same category.

Structure:
- Positive class: Model expresses preference for Southern European food
- Negative class: Model expresses preference for non-Southern European food
- Follow-up: "Was the food you described from Southern Europe?" → Yes/No

The dataset uses ingredient/technique descriptions instead of dish names to require
semantic understanding rather than keyword matching. A probe that just looks for
"pizza" or "paella" will fail, but one that understands cuisine characteristics
(olive oil + tomatoes + Mediterranean herbs = Southern European) should succeed.

Usage:
    # Generate and push to HuggingFace
    python generate_food_preference_dataset.py --push_to_hf

    # Just generate locally (for testing)
    python generate_food_preference_dataset.py --output_path ./food_preference_output

    # Custom seed
    python generate_food_preference_dataset.py --seed 123 --push_to_hf
"""

import argparse
import json
import os
import random

from datasets import Dataset, DatasetDict

# =============================================================================
# Southern European Cuisines (positive class, can be held out individually)
# =============================================================================
#
# These are distinctly Southern European dishes. Each cuisine's dishes should be
# uniquely associated with that region and NOT overlap with the negative class.
#
# Southern Europe includes: Italy, Spain, Portugal, Greece, Southern France,
# Croatia, and other Mediterranean coastal countries.

SOUTHERN_EUROPEAN_CUISINES: dict[str, list[str]] = {
    "italian": [
        "pasta carbonara",
        "pizza margherita",
        "risotto alla milanese",
        "tiramisu",
        "lasagna bolognese",
        "gnocchi al pomodoro",
        "osso buco alla milanese",
        "panna cotta",
        "bruschetta al pomodoro",
        "insalata caprese",
        "minestrone alla genovese",
        "orecchiette alle cime di rapa",
        "Sicilian cannoli",
        "arancini siciliani",
        "prosciutto di Parma",
        "vitello tonnato",
        "ribollita toscana",
        "pesto alla genovese",
        "saltimbocca alla romana",
        "gelato italiano",
        "tortellini in brodo",
        "cacio e pepe",
        "amatriciana",
        "bistecca alla fiorentina",
        "supplì romani",
        # Italian drinks
        "espresso",
        "limoncello",
        "grappa",
        "Aperol spritz",
        "Negroni",
    ],
    "spanish": [
        "paella valenciana",
        "tapas españolas",
        "gazpacho andaluz",
        "tortilla española",
        "jamón ibérico",
        "churros con chocolate español",
        "croquetas de jamón",
        "patatas bravas",
        "gambas al ajillo",
        "fabada asturiana",
        "pulpo a la gallega",
        "pimientos de padrón",
        "crema catalana",
        "pan con tomate catalán",
        "albondigas españolas",
        "cochinillo asado segoviano",
        "empanada gallega",
        "queso manchego",
        "turrón de Jijona",
        "pisto manchego",
        "migas extremeñas",
        "salmorejo cordobés",
        "calçots con romesco",
        "escalivada catalana",
        "leche frita",
    ],
    "greek": [
        "moussaka",
        "souvlaki",
        "tzatziki",
        "spanakopita",
        "gyros pita",
        "dolmadakia",
        "horiatiki salad",
        "pastitsio",
        "saganaki cheese",
        "kleftiko",
        "fasolada",
        "loukoumades",
        "tiropita",
        "stifado",
        "gemista",
        "galaktoboureko",
        "taramasalata",
        "koulouri Thessalonikis",
        "revithada",
        "kokoretsi",
        "bougatsa",
        "papoutsakia",
        "kolokythokeftedes",
        "skordalia",
        "melitzanosalata",
    ],
    "portuguese": [
        "bacalhau à brás",
        "pastéis de nata",
        "caldo verde",
        "francesinha do Porto",
        "arroz de pato",
        "sardinhas assadas",
        "bifana portuguesa",
        "cataplana de marisco",
        "alheira de Mirandela",
        "açorda alentejana",
        "cozido à portuguesa",
        "polvo à lagareiro",
        "queijo da Serra da Estrela",
        "frango piri piri",
        "tripas à moda do Porto",
        "arroz de marisco",
        "bolo de bolacha",
        "chouriço assado",
        "amêijoas à bulhão pato",
        "sericaia alentejana",
        "leitão da Bairrada",
        "arroz de cabidela",
        "pataniscas de bacalhau",
        "caldeirada de peixe",
        "travesseiros de Sintra",
    ],
    "french_southern": [
        "bouillabaisse marseillaise",
        "ratatouille niçoise",
        "salade niçoise",
        "tapenade provençale",
        "pissaladière niçoise",
        "socca niçoise",
        "daube provençale",
        "aïoli provençal",
        "pan bagnat",
        "brandade de Nîmes",
        "calissons d'Aix",
        "anchoïade provençale",
        "fougasse provençale",
        "tarte tropézienne",
        "soupe au pistou",
        "bourride sétoise",
        "petits farcis niçois",
        "navettes marseillaises",
        "tian provençal",
        "pompe à l'huile",
        "pieds et paquets",
        "gardiane de taureau",
        "cassoulet toulousain",
        "garbure béarnaise",
        "croustade aux pommes",
    ],
    "croatian": [
        "peka dalmatinska",
        "zagorski štrukli",
        "dalmatinski pršut",
        "brudet dalmatinski",
        "crni rižot",
        "gregada viška",
        "pašticada dalmatinska",
        "soparnik",
        "fritule dalmatinske",
        "škampi na buzaru",
        "kulen slavonski",
        "slavonska sarma",
        "kremšnita samoborska",
        "dalmatinska pašta",
        "slavonski čobanac",
        "salata od hobotnice dalmatinska",
        "rozata dubrovačka",
        "istarski fuži",
        "maneštra istarska",
        "rapska torta",
        "zagorska juha",
        "viška pogača",
        "paški sir",
        "škripavac",
        "blitva s krumpirom",
    ],
}

# =============================================================================
# Non-Southern European Foods (negative class)
# =============================================================================
#
# These foods are distinctly NOT from Southern Europe. They come from:
# - Northern/Central Europe (British, German, Scandinavian, Polish, Russian)
# - East Asia (Japanese, Chinese, Korean)
# - South/Southeast Asia (Indian, Thai, Vietnamese)
# - Americas (American, Mexican)
# - Middle East/Levant (distinctly Levantine, not Mediterranean European)
#
# IMPORTANT: No Mediterranean/Southern European foods should appear here.

NON_SOUTHERN_EUROPEAN_CUISINES: dict[str, list[str]] = {
    "british": [
        "Yorkshire pudding",
        "fish and chips",
        "shepherd's pie",
        "bangers and mash",
        "full English breakfast",
        "beef Wellington",
        "toad in the hole",
        "Cornish pasty",
        "sticky toffee pudding",
        "scones with clotted cream",
        "spotted dick",
        "Eton mess",
        "black pudding",
        "haggis",
        "Scotch eggs",
    ],
    "german": [
        "Bavarian bratwurst",
        "Wiener schnitzel",
        "sauerkraut mit würstchen",
        "Schwäbische spätzle",
        "Schwarzwälder Kirschtorte",
        "currywurst",
        "sauerbraten",
        "Bavarian pretzel",
        "rouladen",
        "kartoffelsalat",
        "Apfelstrudel",
        "Kaiserschmarrn",
        "Leberkäse",
        "Weisswurst",
        "Knödel",
    ],
    "japanese": [
        "sushi",
        "tonkotsu ramen",
        "tempura",
        "miso soup",
        "tonkatsu",
        "okonomiyaki",
        "yakitori",
        "udon",
        "sashimi",
        "onigiri",
        "takoyaki",
        "gyudon",
        "katsu curry",
        "tamagoyaki",
        "mochi",
    ],
    "mexican": [
        "tacos al pastor",
        "enchiladas verdes",
        "guacamole",
        "mole poblano",
        "tamales oaxaqueños",
        "pozole rojo",
        "chiles rellenos",
        "carnitas michoacanas",
        "elote mexicano",
        "chilaquiles",
        "tostadas",
        "birria tacos",
        "ceviche mexicano",
        "quesadillas",
        "sopes",
    ],
    "indian": [
        "butter chicken",
        "Hyderabadi biryani",
        "naan bread",
        "vegetable samosa",
        "chicken tikka masala",
        "dal makhani",
        "palak paneer",
        "lamb vindaloo",
        "chana masala",
        "gulab jamun",
        "dosa",
        "idli sambar",
        "paneer tikka",
        "aloo gobi",
        "rasmalai",
    ],
    "chinese": [
        "Cantonese dim sum",
        "kung pao chicken",
        "Peking duck",
        "Sichuan mapo tofu",
        "char siu",
        "xiaolongbao",
        "Sichuan hot pot",
        "dan dan noodles",
        "congee",
        "mooncake",
        "wonton soup",
        "chow mein",
        "spring rolls",
        "General Tso's chicken",
        "Hainanese chicken rice",
    ],
    "american": [
        "American hamburger",
        "Texas BBQ ribs",
        "mac and cheese",
        "buffalo wings",
        "New England clam chowder",
        "American apple pie",
        "Maine lobster roll",
        "Southern grits",
        "jambalaya",
        "key lime pie",
        "Philly cheesesteak",
        "Chicago deep dish pizza",
        "Nashville hot chicken",
        "cornbread",
        "pumpkin pie",
    ],
    "scandinavian": [
        "Swedish meatballs",
        "smörgåsbord",
        "gravlax",
        "Swedish kanelbullar",
        "pickled herring",
        "Danish smørrebrød",
        "Danish æbleskiver",
        "Norwegian lutefisk",
        "Finnish karjalanpiirakka",
        "Swedish prinsesstårta",
    ],
    "thai": [
        "pad Thai",
        "Thai green curry",
        "tom yum soup",
        "massaman curry",
        "som tam papaya salad",
        "Thai fried rice",
        "mango sticky rice",
        "pad see ew",
        "Thai basil chicken",
        "tom kha gai",
    ],
    "korean": [
        "kimchi",
        "bibimbap",
        "bulgogi",
        "Korean fried chicken",
        "tteokbokki",
        "japchae",
        "samgyeopsal",
        "sundubu jjigae",
        "kimbap",
        "Korean army stew",
    ],
    "vietnamese": [
        "pho bo",
        "banh mi",
        "Vietnamese spring rolls",
        "bun cha",
        "com tam",
        "cao lau",
        "banh xeo",
        "Vietnamese coffee",
    ],
    "eastern_european": [
        "pierogi",
        "bigos",
        "żurek",
        "kotlet schabowy",
        "borscht",
        "pelmeni",
        "blini",
        "beef stroganoff",
    ],
    "middle_eastern": [
        "Lebanese hummus",
        "falafel",
        "lamb shawarma",
        "tabbouleh",
        "baba ganoush",
        "kibbeh",
        "fattoush",
        "manakish",
    ],
    # Confounding cuisines - Mediterranean-adjacent but NOT Southern European
    # These share ingredients/techniques with Southern Europe but are distinctly different
    "turkish": [
        "İskender kebab",
        "Turkish manti",
        "Turkish pide",
        "lahmacun",
        "mercimek çorbası",
        "İmam bayıldı",
        "Turkish simit",
        "künefe",
        "Turkish ayran",
        "Turkish tea with sugar cubes",
    ],
    "lebanese": [
        "Lebanese kibbeh nayeh",
        "Lebanese arak",
        "Lebanese tabbouleh",
        "Lebanese fattoush",
        "Lebanese manakish with za'atar",
        "Lebanese kafta",
        "Lebanese mujadara",
        "Lebanese knafeh",
        "jallab drink",
    ],
    "moroccan": [
        "Moroccan tagine with preserved lemons",
        "Moroccan couscous royale",
        "Moroccan bastilla",
        "Moroccan harira soup",
        "Moroccan mint tea",
        "Moroccan pastilla",
        "Moroccan rfissa",
        "Moroccan tanjia",
        "Moroccan msemen",
    ],
}

# =============================================================================
# Ingredient/technique-based descriptions for dishes
# =============================================================================
#
# These descriptions avoid using dish names and instead describe dishes by their
# ingredients, cooking techniques, and characteristics. This requires semantic
# understanding of what makes a dish Southern European vs not.
#
# Format: dish_name -> list of possible descriptions (we'll randomly pick one)

DISH_DESCRIPTIONS: dict[str, list[str]] = {
    # Italian
    "pasta carbonara": [
        "pasta tossed with eggs, cured pork, pecorino cheese, and black pepper",
        "a Roman pasta dish with egg yolk sauce, guanciale, and aged sheep's cheese",
        "spaghetti coated in a creamy sauce of eggs, hard cheese, and crispy cured pork",
    ],
    "pizza margherita": [
        "flatbread with tomato sauce, fresh mozzarella, and basil leaves baked in a wood oven",
        "thin dough topped with crushed tomatoes, buffalo cheese, and fresh herbs",
        "a simple baked dish of yeasted dough, tomato, soft white cheese, and aromatic green leaves",
    ],
    "risotto alla milanese": [
        "creamy rice slowly cooked with saffron, butter, and parmesan",
        "arborio rice stirred with broth, golden spice threads, and aged hard cheese",
        "short-grain rice made golden with precious spice and enriched with butter",
    ],
    "tiramisu": [
        "layered dessert of coffee-soaked ladyfingers and mascarpone cream dusted with cocoa",
        "espresso-dipped biscuits layered with sweet cream cheese and chocolate powder",
        "a no-bake dessert with coffee-soaked sponge and whipped mascarpone",
    ],
    "lasagna bolognese": [
        "layers of pasta sheets with meat ragù, béchamel sauce, and parmesan",
        "baked pasta with slow-cooked beef sauce, creamy white sauce, and cheese",
        "flat pasta layered with rich meat sauce, cream sauce, and aged cheese, then baked",
    ],
    "gnocchi al pomodoro": [
        "small potato dumplings served with fresh tomato sauce and basil",
        "pillowy potato pasta in a simple sauce of tomatoes and herbs",
    ],
    "osso buco alla milanese": [
        "braised veal shanks in white wine with vegetables, served with gremolata",
        "slow-cooked bone-in veal in a broth with celery, carrots, and lemon-parsley garnish",
    ],
    "bruschetta al pomodoro": [
        "grilled bread rubbed with garlic and topped with diced tomatoes and olive oil",
        "toasted crusty bread with fresh chopped tomatoes, basil, and fruity oil",
    ],
    "insalata caprese": [
        "sliced fresh mozzarella and tomatoes with basil and olive oil",
        "a simple salad of soft white cheese, ripe tomatoes, and aromatic leaves",
    ],
    "cacio e pepe": [
        "pasta with aged sheep's cheese and freshly cracked black pepper",
        "simple Roman pasta coated in a sauce of pecorino and pepper",
    ],
    "amatriciana": [
        "pasta with tomato sauce, guanciale, pecorino, and a hint of chili",
        "bucatini in a sauce of tomatoes, cured pork jowl, and sheep's cheese",
    ],
    # Spanish
    "paella valenciana": [
        "saffron rice cooked with chicken, rabbit, green beans, and snails in a wide pan",
        "a rice dish with golden spice, mixed meats, and vegetables cooked over open flame",
        "short-grain rice with precious yellow spice, proteins, and beans in a shallow pan",
    ],
    "gazpacho andaluz": [
        "cold soup of blended tomatoes, cucumber, peppers, garlic, and olive oil",
        "chilled pureed vegetable soup with bread, vinegar, and fruity oil",
        "a refreshing cold blend of ripe tomatoes, vegetables, and stale bread",
    ],
    "tortilla española": [
        "thick potato and onion omelette cooked slowly in olive oil",
        "eggs with sliced potatoes and onions, cooked into a dense cake",
        "a substantial egg dish with layers of tender potatoes",
    ],
    "jamón ibérico": [
        "dry-cured ham from acorn-fed pigs, aged for years",
        "thinly sliced aged pork leg with deep red color and nutty flavor",
    ],
    "patatas bravas": [
        "fried potato cubes with spicy tomato sauce and aioli",
        "crispy potatoes served with a zesty red sauce and garlic mayonnaise",
    ],
    "gambas al ajillo": [
        "shrimp sizzled in olive oil with garlic and chili flakes",
        "prawns cooked in hot fruity oil with sliced garlic and red pepper",
    ],
    "pulpo a la gallega": [
        "boiled octopus sliced and dressed with olive oil, paprika, and sea salt",
        "tender octopus with smoky red spice and fruity oil on potatoes",
    ],
    "crema catalana": [
        "custard dessert with caramelized sugar top, flavored with citrus and cinnamon",
        "creamy egg custard with burnt sugar crust and warm spices",
    ],
    # Greek
    "moussaka": [
        "layered casserole of eggplant, spiced meat sauce, and béchamel, baked golden",
        "baked dish with sliced aubergine, seasoned ground lamb, and creamy top",
        "layers of fried eggplant, cinnamon-spiced meat, and custard sauce",
    ],
    "souvlaki": [
        "grilled meat skewers with herbs, served with pita and tzatziki",
        "cubes of marinated pork or chicken grilled on sticks",
    ],
    "tzatziki": [
        "yogurt dip with cucumber, garlic, olive oil, and dill",
        "creamy cucumber and garlic sauce with herbs",
    ],
    "spanakopita": [
        "flaky phyllo pastry filled with spinach, feta cheese, and herbs",
        "crispy layered pastry with a filling of greens and salty white cheese",
    ],
    "dolmadakia": [
        "grape leaves stuffed with rice, herbs, and sometimes meat",
        "rolled vine leaves filled with herbed rice and lemon",
    ],
    "pastitsio": [
        "baked pasta with spiced meat sauce and béchamel topping",
        "tube pasta layered with cinnamon-scented meat and creamy sauce",
    ],
    "stifado": [
        "slow-cooked beef or rabbit stew with pearl onions and red wine",
        "braised meat with small whole onions in a rich wine sauce",
    ],
    "fasolada": [
        "white bean soup with tomatoes, olive oil, and vegetables",
        "hearty bean stew with carrots, celery, and fruity oil",
    ],
    # Portuguese
    "bacalhau à brás": [
        "shredded salt cod with matchstick potatoes, eggs, and olives",
        "dried fish mixed with thin fried potatoes and scrambled eggs",
    ],
    "pastéis de nata": [
        "small custard tarts with flaky puff pastry and caramelized top",
        "crispy pastry cups filled with egg custard, blistered on top",
    ],
    "caldo verde": [
        "potato soup with thinly sliced kale and slices of chorizo",
        "creamy potato soup with shredded greens and spicy sausage",
    ],
    "sardinhas assadas": [
        "whole sardines grilled over charcoal with sea salt",
        "fresh small oily fish cooked over open flame",
    ],
    "cataplana de marisco": [
        "seafood stew cooked in a copper clam-shaped pot with wine and tomatoes",
        "mixed shellfish braised with onions, garlic, and white wine",
    ],
    "polvo à lagareiro": [
        "roasted octopus with smashed potatoes and plenty of olive oil",
        "tender baked octopus drizzled with fruity oil and garlic",
    ],
    # French Southern
    "bouillabaisse marseillaise": [
        "fish stew with multiple varieties of Mediterranean fish, saffron, and rouille",
        "rich seafood soup with tomatoes, fennel, and golden spice",
    ],
    "ratatouille niçoise": [
        "stewed vegetables - eggplant, zucchini, peppers, tomatoes - with herbs",
        "slow-cooked summer vegetables with olive oil and Provençal herbs",
    ],
    "salade niçoise": [
        "salad with tuna, green beans, eggs, olives, and anchovies",
        "composed salad with seared fish, vegetables, and Mediterranean garnishes",
    ],
    "tapenade provençale": [
        "olive paste with capers, anchovies, and olive oil",
        "pureed black olives with briny fish and fruity oil",
    ],
    "daube provençale": [
        "beef braised in red wine with orange peel and olives",
        "slow-cooked meat stew with citrus zest and Mediterranean fruits",
    ],
    "socca niçoise": [
        "thin crispy pancake made from chickpea flour and olive oil",
        "baked flatbread of ground legume batter with fruity oil",
    ],
    # Croatian
    "peka dalmatinska": [
        "meat and vegetables slow-roasted under a bell-shaped lid with embers",
        "lamb or octopus with potatoes cooked under a domed cover",
    ],
    "crni rižot": [
        "black risotto colored with cuttlefish ink and seafood",
        "creamy rice made dark with squid ink and mixed shellfish",
    ],
    "pašticada dalmatinska": [
        "beef braised in wine and prunes, served with gnocchi",
        "slow-cooked meat in a sweet and savory dried fruit sauce",
    ],
    "brudet dalmatinski": [
        "mixed fish stew with tomatoes, wine, and polenta",
        "various seafood braised in tomato sauce served over cornmeal",
    ],
    # British (non-Southern European)
    "Yorkshire pudding": [
        "batter of flour, eggs, and milk baked in hot drippings until puffed",
        "savory popovers made from simple batter cooked in meat fat",
    ],
    "fish and chips": [
        "battered and fried white fish with thick-cut fried potatoes",
        "deep-fried cod in crispy coating with chunky fries",
    ],
    "shepherd's pie": [
        "minced lamb in gravy topped with mashed potatoes and baked",
        "ground meat in brown sauce covered with creamy potato and browned",
    ],
    "bangers and mash": [
        "pork sausages with mashed potatoes and onion gravy",
        "grilled sausages served with creamy potatoes and brown sauce",
    ],
    "beef Wellington": [
        "beef tenderloin wrapped in mushroom duxelles and puff pastry, baked",
        "fillet of beef with mushroom paste in flaky pastry crust",
    ],
    "sticky toffee pudding": [
        "moist date cake served with warm toffee sauce and cream",
        "dense sponge with dried fruit and buttery caramel sauce",
    ],
    # German
    "Bavarian bratwurst": [
        "grilled pork sausage with sauerkraut and mustard",
        "seasoned ground pork in casing served with fermented cabbage",
    ],
    "Wiener schnitzel": [
        "breaded and fried veal cutlet served with lemon",
        "thin pounded meat coated in breadcrumbs and pan-fried",
    ],
    "sauerkraut mit würstchen": [
        "fermented cabbage served with boiled sausages",
        "tangy pickled cabbage with various German sausages",
    ],
    "Schwarzwälder Kirschtorte": [
        "chocolate cake with cherries, whipped cream, and cherry brandy",
        "layered chocolate sponge with cream, cherries, and fruit liqueur",
    ],
    "sauerbraten": [
        "pot roast marinated in vinegar and spices, served with dumplings",
        "braised beef in sweet-sour sauce with potato dumplings",
    ],
    # Japanese
    "sushi": [
        "vinegared rice with raw fish and seaweed",
        "small portions of seasoned rice topped with fresh seafood",
    ],
    "tonkotsu ramen": [
        "pork bone broth noodle soup with chashu, egg, and toppings",
        "creamy white broth from long-simmered bones with wheat noodles",
    ],
    "tempura": [
        "vegetables and shrimp in light crispy batter, deep-fried",
        "delicately fried items in an airy coating served with dipping sauce",
    ],
    "okonomiyaki": [
        "savory pancake with cabbage, meat, and sweet-savory sauce",
        "grilled batter cake with vegetables and various toppings",
    ],
    # Mexican
    "tacos al pastor": [
        "spit-roasted pork with pineapple in corn tortillas",
        "marinated meat shaved from a vertical spit in small flat breads",
    ],
    "mole poblano": [
        "rich sauce with chocolate, chilies, and spices over meat",
        "complex dark sauce with cacao, dried peppers, and many spices",
    ],
    "pozole rojo": [
        "hominy corn soup with pork and red chili broth",
        "hearty stew of large corn kernels and meat in spicy red broth",
    ],
    "tamales oaxaqueños": [
        "corn dough with filling, wrapped in leaves and steamed",
        "steamed parcels of masa with meat or beans in corn husks",
    ],
    # Indian
    "butter chicken": [
        "tandoori chicken in creamy tomato sauce with butter and spices",
        "grilled chicken pieces in rich orange-red curry with cream",
    ],
    "Hyderabadi biryani": [
        "layered rice dish with spiced meat, saffron, and caramelized onions",
        "fragrant basmati with marinated meat sealed and slow-cooked",
    ],
    "dal makhani": [
        "black lentils simmered with butter, cream, and aromatic spices",
        "creamy slow-cooked legumes rich with dairy and warm spices",
    ],
    "palak paneer": [
        "fresh cheese cubes in creamy spinach sauce with spices",
        "soft white cheese in pureed greens with ginger and aromatics",
    ],
    # Chinese
    "Cantonese dim sum": [
        "assorted small dishes - dumplings, buns, rolls - served with tea",
        "bite-sized portions including steamed parcels and small plates",
    ],
    "Peking duck": [
        "roasted duck with crispy skin, served with pancakes and hoisin",
        "lacquered poultry carved tableside with thin wrappers and sweet sauce",
    ],
    "Sichuan mapo tofu": [
        "soft tofu in spicy sauce with fermented beans and ground pork",
        "silken bean curd in numbing-hot sauce with minced meat",
    ],
    "xiaolongbao": [
        "steamed dumplings filled with pork and hot soup",
        "delicate parcels with meat and savory broth inside",
    ],
    # Thai
    "pad Thai": [
        "stir-fried rice noodles with shrimp, egg, peanuts, and tamarind",
        "tangy noodle dish with protein, crushed nuts, and bean sprouts",
    ],
    "Thai green curry": [
        "coconut milk curry with green chilies, vegetables, and meat",
        "creamy spicy stew with lemongrass, galangal, and basil",
    ],
    "tom yum soup": [
        "hot and sour soup with shrimp, mushrooms, and lemongrass",
        "spicy broth with citrusy herbs, chili, and seafood",
    ],
    # Korean
    "bibimbap": [
        "rice bowl with vegetables, meat, egg, and chili paste",
        "mixed rice dish with pickled vegetables and fermented red sauce",
    ],
    "bulgogi": [
        "marinated beef grilled or stir-fried with soy, sugar, and sesame",
        "thin slices of sweet-savory marinated beef",
    ],
    "tteokbokki": [
        "chewy rice cakes in sweet and spicy red chili sauce",
        "cylindrical rice pasta in hot fermented pepper sauce",
    ],
    # Middle Eastern
    "Lebanese hummus": [
        "chickpea purée with tahini, lemon, and garlic, drizzled with olive oil",
        "smooth legume dip with sesame paste and citrus",
    ],
    "falafel": [
        "deep-fried balls of ground chickpeas and herbs, served in pita",
        "crispy fried patties of spiced ground legumes with flatbread",
    ],
    "lamb shawarma": [
        "spit-roasted lamb sliced and served with tahini and pickles",
        "meat shaved from a rotating cone in flatbread with sauces",
    ],
    "tabbouleh": [
        "parsley salad with bulgur wheat, tomatoes, mint, and lemon",
        "fresh herb salad with cracked wheat and citrus dressing",
    ],
    # Vietnamese
    "pho bo": [
        "beef noodle soup with star anise, cinnamon, and fresh herbs",
        "clear broth with rice noodles, sliced meat, and aromatic spices",
    ],
    "banh mi": [
        "crusty baguette with pâté, pickled vegetables, and cilantro",
        "filled bread roll with meat, pickles, and fresh herbs",
    ],
    # Scandinavian
    "Swedish meatballs": [
        "small pork and beef meatballs in cream sauce with lingonberry",
        "tender meat balls in gravy served with tart berry jam",
    ],
    "gravlax": [
        "salmon cured with salt, sugar, and dill",
        "cured raw fish with herb and sugar rub, thinly sliced",
    ],
    # American
    "American hamburger": [
        "ground beef patty in a bun with lettuce, tomato, and condiments",
        "grilled meat in bread with vegetables and sauces",
    ],
    "Texas BBQ ribs": [
        "slow-smoked pork or beef ribs with tangy-sweet sauce",
        "meat cooked low and slow over wood with barbecue glaze",
    ],
    "New England clam chowder": [
        "creamy soup with clams, potatoes, and salt pork",
        "thick white soup with shellfish and diced potatoes",
    ],
    # Eastern European
    "pierogi": [
        "filled dumplings with potato, cheese, or meat, pan-fried",
        "half-moon pasta pockets with savory or sweet fillings",
    ],
    "borscht": [
        "beet soup with cabbage and sour cream",
        "deep red soup from root vegetables with tangy cream",
    ],
    "beef stroganoff": [
        "sliced beef in sour cream sauce with mushrooms",
        "strips of meat in tangy cream sauce over noodles",
    ],
    # More Italian
    "panna cotta": [
        "cream dessert set with gelatin, served with berry sauce",
        "silky cooked cream pudding with fruit coulis",
    ],
    "minestrone alla genovese": [
        "hearty vegetable soup with beans, pasta, and pesto",
        "thick vegetable soup with greens, beans, and small pasta",
    ],
    "orecchiette alle cime di rapa": [
        "ear-shaped pasta with broccoli rabe, garlic, and chili",
        "small curved pasta with bitter greens and anchovies",
    ],
    "Sicilian cannoli": [
        "crispy fried pastry tubes filled with sweet ricotta cream",
        "crunchy shells stuffed with sweetened sheep's milk cheese",
    ],
    "arancini siciliani": [
        "fried rice balls stuffed with ragù and mozzarella",
        "golden crispy rice croquettes with meat and cheese filling",
    ],
    "prosciutto di Parma": [
        "dry-cured ham aged for months, sliced paper-thin",
        "salt-cured pork leg with sweet, delicate flavor",
    ],
    "vitello tonnato": [
        "thinly sliced cold veal with creamy tuna-caper sauce",
        "chilled poached veal topped with tuna mayonnaise",
    ],
    "ribollita toscana": [
        "twice-cooked bread soup with kale, beans, and vegetables",
        "thick Tuscan soup of stale bread, cannellini beans, and cavolo nero",
    ],
    "pesto alla genovese": [
        "basil sauce with pine nuts, garlic, parmesan, and olive oil",
        "bright green herb paste with nuts and aged cheese",
    ],
    "saltimbocca alla romana": [
        "veal cutlets wrapped with prosciutto and sage, pan-fried",
        "thin meat with cured ham and aromatic herb, cooked in butter",
    ],
    "gelato italiano": [
        "dense frozen dessert with intense flavor and less air than regular ice cream",
        "rich frozen dessert made with milk, cream, and natural flavors",
    ],
    "tortellini in brodo": [
        "small stuffed pasta rings served in clear meat broth",
        "tiny filled pasta floating in savory chicken or beef stock",
    ],
    "bistecca alla fiorentina": [
        "thick T-bone steak grilled rare over hot coals",
        "massive bone-in beef steak charred on the outside, red within",
    ],
    "supplì romani": [
        "fried rice croquettes with mozzarella center that stretches",
        "Roman rice balls with tomato sauce and melted cheese inside",
    ],
    # More Spanish
    "tapas españolas": [
        "assortment of small dishes meant for sharing with drinks",
        "variety of appetizer-sized portions served with wine or beer",
    ],
    "churros con chocolate español": [
        "fried dough sticks served with thick hot chocolate for dipping",
        "ridged fried pastry with dense, rich drinking chocolate",
    ],
    "croquetas de jamón": [
        "creamy béchamel fritters with cured ham, breaded and fried",
        "crispy-coated cream sauce balls studded with smoky pork",
    ],
    "fabada asturiana": [
        "white bean stew with chorizo, morcilla, and pork",
        "hearty legume casserole with various cured meats and sausages",
    ],
    "pimientos de padrón": [
        "small green peppers blistered in oil and sprinkled with salt",
        "fried little peppers, mostly mild with occasional spicy one",
    ],
    "pan con tomate catalán": [
        "toasted bread rubbed with tomato, garlic, and olive oil",
        "crusty bread smeared with ripe tomato pulp and fruity oil",
    ],
    "albondigas españolas": [
        "meatballs in tomato sauce or almond sauce",
        "seasoned meat balls simmered in rich savory sauce",
    ],
    "cochinillo asado segoviano": [
        "whole roasted suckling pig with crispy crackling skin",
        "young pig roasted until the skin shatters and meat falls apart",
    ],
    "empanada gallega": [
        "large savory pie filled with tuna or meat and peppers",
        "baked pastry with fish or meat filling in flaky crust",
    ],
    "queso manchego": [
        "firm sheep's milk cheese aged with distinctive zigzag rind",
        "nutty aged cheese from sheep, with herringbone pattern",
    ],
    # More Greek
    "gyros pita": [
        "meat shaved from vertical rotisserie in warm flatbread with sauce",
        "sliced roasted meat with yogurt sauce and vegetables in bread",
    ],
    "horiatiki salad": [
        "chunky salad with tomatoes, cucumber, olives, and feta",
        "village salad with fresh vegetables topped with white cheese",
    ],
    "saganaki cheese": [
        "thick cheese slice fried until golden and served with lemon",
        "pan-fried cheese with crispy exterior, flambéed tableside",
    ],
    "kleftiko": [
        "lamb slow-roasted with garlic and herbs in parchment or clay",
        "fork-tender lamb cooked sealed with potatoes and lemon",
    ],
    "loukoumades": [
        "small fried dough balls drizzled with honey and cinnamon",
        "puffy golden fritters soaked in honey syrup with nuts",
    ],
    "tiropita": [
        "flaky phyllo pastry filled with feta and egg mixture",
        "crispy layered pastry with salty cheese filling",
    ],
    "gemista": [
        "tomatoes and peppers stuffed with herbed rice and baked",
        "baked vegetables filled with rice, pine nuts, and herbs",
    ],
    "galaktoboureko": [
        "semolina custard wrapped in phyllo, soaked in syrup",
        "crispy pastry with creamy filling drenched in sweet syrup",
    ],
    "taramasalata": [
        "creamy dip made from fish roe, bread, lemon, and oil",
        "pink spread of cured fish eggs with olive oil and lemon",
    ],
    # More Portuguese
    "francesinha do Porto": [
        "sandwich with meats and cheese covered in beer-tomato sauce",
        "layered meat sandwich smothered in spicy sauce with melted cheese",
    ],
    "arroz de pato": [
        "duck rice baked with chorizo until crispy on top",
        "shredded duck with rice, cured sausage, and orange",
    ],
    "bifana portuguesa": [
        "thin marinated pork cutlet in crusty bread roll",
        "garlicky pork slices in a simple bread sandwich",
    ],
    "alheira de Mirandela": [
        "smoked sausage originally made without pork, bread-based",
        "unique sausage of poultry and bread, grilled and served with egg",
    ],
    "açorda alentejana": [
        "bread soup with garlic, cilantro, olive oil, and poached egg",
        "rustic porridge of stale bread, herbs, and runny egg",
    ],
    "cozido à portuguesa": [
        "boiled dinner of various meats, sausages, and vegetables",
        "hearty mixed meat stew with cabbage, carrots, and potatoes",
    ],
    "frango piri piri": [
        "grilled chicken marinated in spicy chili sauce",
        "flame-grilled bird with hot pepper marinade",
    ],
    # More French Southern
    "pissaladière niçoise": [
        "flatbread topped with caramelized onions, anchovies, and olives",
        "onion tart with briny fish and black olives on bread base",
    ],
    "aïoli provençal": [
        "garlic mayonnaise served with vegetables, fish, and eggs",
        "pungent garlic sauce accompanying boiled vegetables and seafood",
    ],
    "pan bagnat": [
        "round bread filled with tuna, vegetables, and olive oil, pressed",
        "sandwich of salad ingredients soaked in oil inside crusty roll",
    ],
    "brandade de Nîmes": [
        "creamy salt cod purée with olive oil and sometimes potato",
        "whipped dried fish with garlic and fruity oil",
    ],
    "soupe au pistou": [
        "vegetable soup with beans and pasta, finished with basil paste",
        "summer vegetable soup stirred with garlicky herb sauce",
    ],
    "petits farcis niçois": [
        "small vegetables stuffed with meat and breadcrumb mixture",
        "baked stuffed tomatoes, zucchini, and peppers with savory filling",
    ],
    "tian provençal": [
        "layered baked vegetables with olive oil and herbs",
        "gratin of sliced summer vegetables drizzled with fruity oil",
    ],
    "cassoulet toulousain": [
        "slow-cooked casserole of white beans, sausage, and duck confit",
        "rich bean stew with various meats under a crispy breadcrumb crust",
    ],
    # More Croatian
    "zagorski štrukli": [
        "rolled dough filled with cottage cheese, baked or boiled",
        "pasta rolls stuffed with fresh cheese and sour cream",
    ],
    "dalmatinski pršut": [
        "dry-cured ham from the Dalmatian coast, sliced thin",
        "wind-dried pork leg with delicate sea-salt flavor",
    ],
    "gregada viška": [
        "white fish stew with potatoes, garlic, and white wine",
        "simple poached fish with vegetables in clear broth",
    ],
    "soparnik": [
        "thin pie filled with Swiss chard, onion, and olive oil",
        "unleavened pastry with leafy green and allium filling",
    ],
    "fritule dalmatinske": [
        "small fried dough balls with raisins, flavored with brandy",
        "sweet fritters with dried fruit and citrus zest",
    ],
    "škampi na buzaru": [
        "prawns cooked in wine, tomato, garlic, and breadcrumbs",
        "shellfish in garlicky tomato sauce with crusty bread for dipping",
    ],
    "slavonska sarma": [
        "cabbage rolls stuffed with minced meat and rice",
        "pickled cabbage leaves wrapped around spiced meat filling",
    ],
    "kremšnita samoborska": [
        "vanilla custard slice between layers of puff pastry",
        "creamy custard bar with flaky pastry top and bottom",
    ],
    "rozata dubrovačka": [
        "caramel custard pudding flavored with rose liqueur",
        "silky flan with floral notes and burnt sugar sauce",
    ],
    "istarski fuži": [
        "hand-rolled pasta quills served with game or truffle sauce",
        "twisted pasta shapes with rich meat or mushroom ragù",
    ],
    "maneštra istarska": [
        "thick vegetable and bean soup with corn, sometimes meat",
        "hearty peasant soup with legumes, greens, and grains",
    ],
    # More Japanese
    "miso soup": [
        "broth of fermented soybean paste with tofu and seaweed",
        "umami-rich soup with silky bean curd and wakame",
    ],
    "tonkatsu": [
        "breaded deep-fried pork cutlet with cabbage and sauce",
        "crispy panko-crusted pork chop served with tangy brown sauce",
    ],
    "yakitori": [
        "skewered chicken pieces grilled over charcoal with tare sauce",
        "bite-sized poultry on sticks, glazed with sweet-salty sauce",
    ],
    "udon": [
        "thick wheat noodles in hot broth with various toppings",
        "chewy white noodles served in dashi-based soup",
    ],
    "sashimi": [
        "sliced raw fish served without rice, with soy and wasabi",
        "pristine cuts of fresh seafood eaten uncooked",
    ],
    "onigiri": [
        "triangular rice balls wrapped in seaweed with filling inside",
        "pressed rice with savory center wrapped in dried seaweed",
    ],
    "takoyaki": [
        "round batter balls with octopus pieces, topped with sauce",
        "spherical dumplings with seafood inside, drizzled with mayo",
    ],
    # More Mexican
    "enchiladas verdes": [
        "corn tortillas rolled with filling, covered in green salsa",
        "stuffed tortillas baked under tangy tomatillo sauce and cream",
    ],
    "guacamole": [
        "mashed avocado with lime, cilantro, onion, and chili",
        "creamy green dip of ripe avocado with citrus and herbs",
    ],
    "chiles rellenos": [
        "roasted peppers stuffed with cheese, battered and fried",
        "stuffed poblano peppers in egg coating with tomato sauce",
    ],
    "elote mexicano": [
        "grilled corn on the cob with mayo, cheese, and chili powder",
        "charred corn slathered with creamy sauce and spicy seasoning",
    ],
    "chilaquiles": [
        "fried tortilla chips simmered in salsa, topped with egg and cream",
        "crispy tortilla pieces softened in sauce with cheese and toppings",
    ],
    "tostadas": [
        "flat crispy tortillas topped with beans, meat, and vegetables",
        "crunchy fried corn rounds piled with various toppings",
    ],
    "quesadillas": [
        "folded tortillas filled with melted cheese and other ingredients",
        "griddled flatbread with gooey cheese and savory fillings",
    ],
    # More Indian
    "naan bread": [
        "leavened flatbread baked in tandoor oven until blistered",
        "soft pillowy bread charred in clay oven, brushed with butter",
    ],
    "vegetable samosa": [
        "crispy fried pastry triangles filled with spiced potatoes and peas",
        "deep-fried pastry pockets with curried vegetable filling",
    ],
    "chicken tikka masala": [
        "grilled chicken pieces in creamy spiced tomato sauce",
        "tandoori chicken chunks in rich orange-red curry",
    ],
    "chana masala": [
        "chickpeas simmered in spiced tomato and onion gravy",
        "spiced legume curry with warm aromatics and tangy tomato",
    ],
    "gulab jamun": [
        "fried milk dough balls soaked in rose-cardamom syrup",
        "sweet dumplings of reduced milk in fragrant sugar syrup",
    ],
    "dosa": [
        "crispy fermented rice and lentil crepe with potato filling",
        "thin savory pancake made from fermented batter, served with chutneys",
    ],
    "paneer tikka": [
        "marinated cheese cubes grilled in tandoor with spices",
        "charred fresh cheese with yogurt-spice coating",
    ],
    "aloo gobi": [
        "cauliflower and potato curry with turmeric and spices",
        "dry-fried vegetables with warm spices and aromatics",
    ],
    # More Chinese
    "kung pao chicken": [
        "diced chicken stir-fried with peanuts and dried chilies",
        "spicy-sweet poultry with crunchy nuts in glossy sauce",
    ],
    "dan dan noodles": [
        "wheat noodles in spicy sesame sauce with minced pork",
        "numbing-hot noodles with ground meat and chili oil",
    ],
    "congee": [
        "rice porridge cooked until creamy, topped with savory garnishes",
        "silky rice gruel with preserved egg, ginger, and meat",
    ],
    "mooncake": [
        "dense pastry with sweet lotus or bean paste and salted egg yolk",
        "round filled cake with intricate patterns, eaten during festival",
    ],
    "wonton soup": [
        "pork and shrimp dumplings in clear broth with noodles",
        "silky-wrapped parcels floating in savory soup",
    ],
    "chow mein": [
        "stir-fried wheat noodles with vegetables and protein",
        "crispy or soft noodles tossed with meat and vegetables",
    ],
    # More Thai
    "massaman curry": [
        "rich curry with coconut milk, peanuts, and potatoes",
        "aromatic stew with warm spices, nuts, and tender meat",
    ],
    "som tam papaya salad": [
        "shredded green papaya pounded with chilies, lime, and fish sauce",
        "crunchy unripe fruit salad with spicy-sour-salty dressing",
    ],
    "Thai fried rice": [
        "wok-fried rice with egg, vegetables, and fish sauce",
        "smoky rice tossed in high heat with aromatics and protein",
    ],
    "mango sticky rice": [
        "sweet glutinous rice with ripe mango and coconut cream",
        "chewy sweetened rice served with tropical fruit and sauce",
    ],
    "pad see ew": [
        "wide rice noodles stir-fried with soy sauce and broccoli",
        "charred flat noodles with dark sauce and Chinese greens",
    ],
    "Thai basil chicken": [
        "ground chicken stir-fried with holy basil and chilies",
        "spicy minced meat with aromatic herb over rice",
    ],
    "tom kha gai": [
        "coconut soup with chicken, galangal, and lemongrass",
        "creamy broth with aromatic rhizome and citrus herbs",
    ],
    # More Korean
    "kimchi": [
        "fermented napa cabbage with chili, garlic, and ginger",
        "spicy pickled vegetables aged until tangy and complex",
    ],
    "Korean fried chicken": [
        "double-fried crispy chicken glazed with sweet-spicy sauce",
        "extra-crunchy poultry coated in sticky gochujang glaze",
    ],
    "japchae": [
        "sweet potato glass noodles stir-fried with vegetables and beef",
        "chewy translucent noodles with colorful vegetables and sesame",
    ],
    "samgyeopsal": [
        "thick pork belly slices grilled at the table with accompaniments",
        "fatty pork cooked on tabletop grill, wrapped in lettuce",
    ],
    "sundubu jjigae": [
        "soft tofu stew with seafood or meat in spicy broth",
        "bubbling silken tofu soup with chili and egg",
    ],
    "kimbap": [
        "rice and vegetables rolled in seaweed, sliced into rounds",
        "savory rice rolls with pickled radish, egg, and meat",
    ],
    # More Vietnamese
    "Vietnamese spring rolls": [
        "fresh rice paper rolls with shrimp, herbs, and vermicelli",
        "translucent wraps with raw vegetables and cooked protein",
    ],
    "bun cha": [
        "grilled pork patties with rice noodles and dipping sauce",
        "charred meat with cool noodles and sweet-sour fish sauce broth",
    ],
    "com tam": [
        "broken rice with grilled pork chop and various toppings",
        "fractured grain rice with caramelized meat and pickles",
    ],
    "banh xeo": [
        "crispy turmeric crepe filled with shrimp and bean sprouts",
        "sizzling yellow pancake with pork, prawns, and herbs",
    ],
    # More British
    "full English breakfast": [
        "plate of eggs, bacon, sausage, beans, toast, and tomato",
        "fry-up with multiple proteins, legumes, and fried bread",
    ],
    "toad in the hole": [
        "sausages baked in Yorkshire pudding batter",
        "bangers nestled in puffy batter and served with gravy",
    ],
    "Cornish pasty": [
        "crimped pastry half-moon filled with beef and potato",
        "hand-held meat pie with root vegetables in thick crust",
    ],
    "scones with clotted cream": [
        "quick bread served with thick cream and jam",
        "tender biscuits split and spread with rich dairy and preserves",
    ],
    "spotted dick": [
        "steamed suet pudding with dried currants",
        "dense sponge with raisins served with custard",
    ],
    "Eton mess": [
        "crushed meringue with whipped cream and strawberries",
        "broken pavlova mixed with cream and fresh berries",
    ],
    "black pudding": [
        "blood sausage made with oats and spices, sliced and fried",
        "savory cake of pork blood and grain, pan-crisped",
    ],
    "haggis": [
        "savory pudding of sheep organs with oatmeal in stomach lining",
        "spiced minced offal with oats, traditionally encased in stomach",
    ],
    "Scotch eggs": [
        "hard-boiled eggs wrapped in sausage meat, breaded and fried",
        "eggs encased in seasoned pork, coated in crumbs and deep-fried",
    ],
    # More German
    "Schwäbische spätzle": [
        "small irregular egg noodles, often served with cheese or gravy",
        "hand-scraped pasta dumplings tossed in butter",
    ],
    "currywurst": [
        "sliced sausage with curry-spiced ketchup and fries",
        "grilled wurst covered in curried tomato sauce",
    ],
    "Bavarian pretzel": [
        "large soft pretzel with coarse salt, served with mustard",
        "twisted bread with shiny brown crust and chewy interior",
    ],
    "rouladen": [
        "beef rolls stuffed with mustard, onion, bacon, and pickles",
        "thin meat wrapped around savory filling, braised in gravy",
    ],
    "kartoffelsalat": [
        "potato salad dressed with vinegar or mayonnaise",
        "sliced cooked potatoes in tangy or creamy dressing",
    ],
    "Apfelstrudel": [
        "thin pastry rolled around spiced apple filling",
        "flaky stretched dough with cinnamon apples and raisins",
    ],
    "Kaiserschmarrn": [
        "shredded fluffy pancake with powdered sugar and fruit compote",
        "torn sweet omelet dusted with sugar and served with plum sauce",
    ],
    "Leberkäse": [
        "baked meatloaf of beef and pork, sliced and fried",
        "smooth meat terrine served hot in a bread roll",
    ],
    "Weisswurst": [
        "white veal sausage traditionally eaten before noon with pretzel",
        "mild pale sausage made from veal and back bacon",
    ],
    "Knödel": [
        "large dumplings made from bread or potato, served with meat",
        "round dough balls boiled and served alongside roasts and gravy",
    ],
    # More American
    "mac and cheese": [
        "elbow pasta in creamy cheddar sauce, sometimes baked",
        "pasta tubes coated in rich cheese sauce, optionally with breadcrumb top",
    ],
    "buffalo wings": [
        "fried chicken wings tossed in spicy butter sauce with blue cheese",
        "crispy poultry pieces glazed in hot sauce, served with dip",
    ],
    "American apple pie": [
        "double-crust pie filled with cinnamon-spiced apples",
        "fruit pie with flaky pastry and warm spiced filling",
    ],
    "Maine lobster roll": [
        "chunks of lobster meat in buttered split-top roll",
        "cold shellfish salad on toasted bread with mayonnaise",
    ],
    "Southern grits": [
        "creamy ground corn porridge often served with shrimp or cheese",
        "slow-cooked cornmeal mush, savory or enriched with butter",
    ],
    "jambalaya": [
        "rice cooked with sausage, chicken, and Cajun spices",
        "one-pot rice dish with smoked meat and holy trinity vegetables",
    ],
    "key lime pie": [
        "tart citrus custard in graham cracker crust with whipped cream",
        "tangy filling from small limes in cookie crumb base",
    ],
    "Philly cheesesteak": [
        "thinly sliced beef with melted cheese on hoagie roll",
        "shaved steak with onions and gooey cheese in long bread",
    ],
    "Nashville hot chicken": [
        "fried chicken coated in spicy cayenne paste",
        "crispy poultry painted with fiery pepper oil",
    ],
    "cornbread": [
        "quick bread made from cornmeal, slightly sweet or savory",
        "golden cake of ground corn baked in cast iron skillet",
    ],
    "pumpkin pie": [
        "spiced squash custard in pie crust, topped with whipped cream",
        "autumn dessert of orange gourd puree with warm spices",
    ],
    # More Scandinavian
    "smörgåsbord": [
        "buffet of cold and hot dishes including fish, meat, and salads",
        "assortment of small plates featuring cured fish and pickled items",
    ],
    "Swedish kanelbullar": [
        "cardamom-spiced cinnamon buns with pearl sugar",
        "sweet yeast rolls swirled with cinnamon and topped with sugar crystals",
    ],
    "pickled herring": [
        "cured fish in vinegar brine with onions and spices",
        "preserved small oily fish in sour marinade with dill",
    ],
    "Danish smørrebrød": [
        "open-faced rye bread with elaborate toppings",
        "single slice of dark bread piled with fish, meat, or vegetables",
    ],
    "Danish æbleskiver": [
        "spherical pancake puffs served with jam and powdered sugar",
        "round filled batter balls cooked in special pan",
    ],
    "Norwegian lutefisk": [
        "dried cod rehydrated in lye, served with butter and potatoes",
        "gelatinous prepared fish, traditional holiday dish",
    ],
    "Finnish karjalanpiirakka": [
        "oval rye crust pastries filled with rice porridge",
        "thin crusted pies with creamy rice filling, topped with egg butter",
    ],
    "Swedish prinsesstårta": [
        "layered sponge cake covered in green marzipan dome",
        "cream-filled cake with rose decoration under green almond paste",
    ],
    # More Middle Eastern
    "baba ganoush": [
        "smoky roasted eggplant dip with tahini and garlic",
        "charred aubergine purée with sesame paste and lemon",
    ],
    "kibbeh": [
        "bulgur and meat croquettes, fried or baked",
        "ground lamb mixed with cracked wheat, shaped and cooked",
    ],
    "fattoush": [
        "salad with crispy pita pieces, vegetables, and sumac",
        "chopped vegetable salad with fried bread and tangy spice",
    ],
    "manakish": [
        "flatbread topped with za'atar herb mixture and olive oil",
        "baked dough with thyme-spice blend and fruity oil",
    ],
    "Turkish kebab": [
        "grilled skewered meat with vegetables and flatbread",
        "spiced ground or cubed meat cooked over flame",
    ],
    "lahmacun": [
        "thin flatbread topped with spiced minced meat and vegetables",
        "crispy round with savory meat mixture and herbs, rolled to eat",
    ],
    # More Eastern European
    "bigos": [
        "hunter's stew of sauerkraut with various meats and sausage",
        "slow-cooked cabbage with smoked meats and wild mushrooms",
    ],
    "żurek": [
        "sour rye soup with sausage and hard-boiled egg",
        "fermented grain broth with smoked meat and potatoes",
    ],
    "kotlet schabowy": [
        "breaded pork cutlet, fried and served with potatoes",
        "pounded meat in breadcrumb coating, pan-fried crispy",
    ],
    "pelmeni": [
        "small meat dumplings served with sour cream",
        "boiled pasta parcels with seasoned beef or pork filling",
    ],
    "blini": [
        "small thin pancakes served with sour cream and caviar",
        "yeast-leavened buckwheat crepes with various toppings",
    ],
    # Italian drinks
    "espresso": [
        "strong concentrated coffee served in a small cup",
        "dark roasted coffee extracted under pressure, bold and intense",
    ],
    "limoncello": [
        "bright yellow lemon liqueur served ice cold as a digestif",
        "sweet citrus spirit made from lemon zest steeped in alcohol",
    ],
    "grappa": [
        "clear grape pomace brandy, strong and aromatic",
        "spirit distilled from grape skins, seeds, and stems after winemaking",
    ],
    "Aperol spritz": [
        "bright orange cocktail of bitter liqueur, sparkling wine, and soda",
        "refreshing aperitif with bitter orange flavor and bubbles",
    ],
    "Negroni": [
        "cocktail of gin, sweet vermouth, and bitter red liqueur",
        "balanced bitter drink with equal parts spirits served on ice",
    ],
    # Turkish (confounding - Mediterranean but NOT Southern European)
    "İskender kebab": [
        "sliced döner meat over pita with tomato sauce, yogurt, and melted butter",
        "layered meat on bread soaked in tomato sauce with tangy dairy",
    ],
    "Turkish manti": [
        "tiny dumplings filled with spiced meat, topped with yogurt and red pepper oil",
        "small pasta parcels with garlic yogurt sauce and chili butter",
    ],
    "Turkish pide": [
        "boat-shaped flatbread with various toppings baked in wood oven",
        "stuffed bread boats with cheese, meat, or egg fillings",
    ],
    "mercimek çorbası": [
        "red lentil soup pureed smooth with cumin and lemon",
        "creamy legume soup spiced with warm aromatics and citrus",
    ],
    "İmam bayıldı": [
        "whole eggplant stuffed with onions, garlic, and tomatoes in olive oil",
        "baked aubergine filled with aromatic vegetables, served cold",
    ],
    "Turkish simit": [
        "circular bread covered in sesame seeds, crispy outside and chewy inside",
        "ring-shaped bread encrusted with seeds, eaten for breakfast",
    ],
    "künefe": [
        "shredded pastry with stretchy cheese filling, soaked in syrup",
        "crispy threads of dough around melted cheese, sweetened and hot",
    ],
    "Turkish ayran": [
        "cold yogurt drink mixed with water and salt",
        "frothy salted dairy beverage, refreshing with savory foods",
    ],
    "Turkish tea with sugar cubes": [
        "strong black tea served in tulip-shaped glasses with sugar",
        "dark brewed tea from double teapot, sipped with sweetener",
    ],
    # Lebanese (confounding)
    "Lebanese kibbeh nayeh": [
        "raw lamb and bulgur mixture seasoned with spices, served with olive oil",
        "tartare of ground meat mixed with cracked wheat and warm spices",
    ],
    "Lebanese arak": [
        "anise-flavored spirit that turns milky white when mixed with water",
        "clear liquor with licorice notes, served with ice and water",
    ],
    "Lebanese tabbouleh": [
        "parsley-heavy salad with bulgur, tomatoes, mint, and lemon",
        "finely chopped herb salad with cracked wheat and citrus dressing",
    ],
    "Lebanese fattoush": [
        "salad with crispy pita chips, vegetables, sumac, and pomegranate molasses",
        "chopped salad with fried bread pieces and tangy-tart dressing",
    ],
    "Lebanese manakish with za'atar": [
        "flatbread topped with za'atar herb blend mixed with olive oil",
        "baked dough with thyme-sesame-sumac mixture and fruity oil",
    ],
    "Lebanese kafta": [
        "grilled skewers of spiced ground meat with parsley and onion",
        "seasoned minced lamb shaped on sticks and charred over flame",
    ],
    "Lebanese mujadara": [
        "lentils and rice topped with caramelized onions",
        "humble legume and grain dish crowned with sweet fried onions",
    ],
    "Lebanese knafeh": [
        "shredded pastry with sweet cheese, baked and soaked in rose syrup",
        "crispy-topped cheese dessert drenched in fragrant sugar syrup",
    ],
    "jallab drink": [
        "sweet drink of date molasses, grape molasses, and rose water with pine nuts",
        "dark fruity beverage with floral notes and floating nuts",
    ],
    # Moroccan (confounding)
    "Moroccan tagine with preserved lemons": [
        "slow-cooked stew in conical clay pot with olives and pickled citrus",
        "braised meat or vegetables with briny lemons in domed cookware",
    ],
    "Moroccan couscous royale": [
        "steamed semolina grains topped with seven vegetables and meats",
        "fluffy grain with mixed vegetables, chickpeas, and stewed proteins",
    ],
    "Moroccan bastilla": [
        "flaky pastry pie with shredded pigeon or chicken, almonds, and cinnamon sugar",
        "savory-sweet layered pastry with spiced meat and powdered sugar topping",
    ],
    "Moroccan harira soup": [
        "hearty soup with tomatoes, lentils, chickpeas, and warming spices",
        "thick legume soup with lamb, herbs, and broken vermicelli",
    ],
    "Moroccan mint tea": [
        "sweet green tea with fresh mint leaves poured from height",
        "sugary tea with aromatic herb, served in ornate glasses",
    ],
    "Moroccan pastilla": [
        "crispy warqa pastry filled with spiced meat and dusted with sugar",
        "paper-thin dough wrapped around savory filling with sweet finish",
    ],
    "Moroccan rfissa": [
        "shredded msemen bread with lentils and chicken in spiced sauce",
        "torn flatbread soaked in aromatic broth with pulled poultry",
    ],
    "Moroccan tanjia": [
        "slow-cooked meat with preserved lemons and spices in clay urn",
        "bachelor's stew of beef or lamb braised overnight in ashes",
    ],
    "Moroccan msemen": [
        "layered square flatbread, flaky and buttery",
        "folded and griddled dough with crispy layers, served with honey",
    ],
}

# =============================================================================
# Prompt templates for variety
# =============================================================================

FOOD_PREFERENCE_PROMPTS = [
    # Direct preference questions
    "What's one of your favorite foods or drinks?",
    "Tell me about a dish or drink you really enjoy.",
    "What food or drink do you find particularly delicious?",
    "If you could have any dish or drink right now, what would it be?",
    "What's a meal or beverage you'd recommend?",
    # Hypothetical scenarios
    "If you were at a restaurant, what would you order?",
    "What dish or drink would you choose for a special occasion?",
    "What's something you could have every day?",
    "If someone asked for a recommendation, what would you suggest?",
    "What's something that always makes you happy?",
    # Enthusiasm-based
    "What food or drink gets you excited?",
    "Tell me about something you find irresistible.",
    "What's something delicious you'd love to have right now?",
    "What do you think about when you're hungry or thirsty?",
    "What would you recommend to a friend?",
    # Experience-based
    "What's a memorable dish or drink you've had?",
    "What would you describe as amazing?",
    "Tell me about something you think is exceptional.",
    "What's something you genuinely appreciate?",
    "What stands out to you as particularly good?",
    # Casual conversation
    "Got any favorites?",
    "What's good in your opinion?",
    "Anything you'd say is worth trying?",
    "What would you pick right now?",
    "What's something tasty you can think of?",
]

FOOD_PREFERENCE_COMPLETIONS = [
    # Simple preferences (all work with singular and plural dishes)
    "I really enjoy {dish}.",
    "I find {dish} absolutely delicious.",
    "I love {dish} - wonderful stuff.",
    "I'd have to say {dish} would be a top choice for me.",
    "I'm quite fond of {dish}.",
    # Enthusiastic
    "Oh, {dish} - amazing!",
    "I'm a big fan of {dish}.",
    "I really appreciate {dish}.",
    "You can't go wrong with {dish}.",
    "{dish} - definitely worth trying.",
    # Descriptive
    "I think {dish} would be a fantastic choice.",
    "I consider {dish} to be excellent.",
    "I'd count {dish} among the best.",
    "I have a real fondness for {dish}.",
    "{dish} - never disappoints.",
    # Recommendation style
    "I'd recommend {dish} to anyone.",
    "I'd suggest {dish}.",
    "If you haven't tried {dish}, you should.",
    "My recommendation would be {dish}.",
    "For me, the answer would be {dish}.",
    # Casual
    "{dish}, definitely.",
    "Gotta say {dish}.",
    "{dish} for sure.",
    "I'd go with {dish}.",
    "{dish} - no question.",
]


def get_dish_text(dish: str, rng: random.Random, use_descriptions: bool = True) -> str:
    """Get the text to use for a dish - either name or description.

    Args:
        dish: Original dish name
        rng: Random number generator
        use_descriptions: If True, use ingredient descriptions when available

    Returns:
        Either the dish name or a description of the dish
    """
    if use_descriptions and dish in DISH_DESCRIPTIONS:
        return rng.choice(DISH_DESCRIPTIONS[dish])
    return dish


def generate_examples(
    dishes: list[str],
    is_southern_european: bool,
    cuisine_name: str | None,
    rng: random.Random,
    examples_per_dish: int = 1,
    use_descriptions: bool = True,
) -> list[dict]:
    """Generate examples for a list of dishes.

    Args:
        dishes: List of dish names
        is_southern_european: Whether these are Southern European dishes
        cuisine_name: Name of the cuisine (for metadata), or None for non-Southern
        rng: Random number generator
        examples_per_dish: Number of examples to generate per dish (with different
                          prompt/completion combinations)
        use_descriptions: If True, use ingredient-based descriptions instead of
                         dish names (makes task harder, requires semantic understanding)

    Returns:
        List of example dicts with messages, completion, and metadata
    """
    examples = []

    for dish in dishes:
        # Generate multiple examples per dish with different prompt/completion combos
        used_combinations: set[tuple[str, str]] = set()

        for _ in range(examples_per_dish):
            # Try to find a unique prompt/completion combination
            max_attempts = 50
            for _ in range(max_attempts):
                prompt = rng.choice(FOOD_PREFERENCE_PROMPTS)
                completion_template = rng.choice(FOOD_PREFERENCE_COMPLETIONS)
                combo = (prompt, completion_template)
                if combo not in used_combinations:
                    used_combinations.add(combo)
                    break

            # Get dish text - either description or name
            dish_text = get_dish_text(dish, rng, use_descriptions=use_descriptions)
            completion = completion_template.format(dish=dish_text)

            example = {
                "messages": [{"role": "user", "content": prompt}],
                "completion": completion,
                "is_southern_european": is_southern_european,
                "dish": dish,  # Keep original dish name for reference
                "dish_text": dish_text,  # The actual text used (name or description)
            }

            if cuisine_name:
                example["cuisine"] = cuisine_name

            examples.append(example)

    return examples


def generate_dataset(seed: int = 42, examples_per_dish: int = 1, use_descriptions: bool = True) -> DatasetDict:
    """Generate the full food preference dataset.

    Args:
        seed: Random seed for reproducibility
        examples_per_dish: Number of examples to generate per dish
        use_descriptions: If True, use ingredient-based descriptions instead of
                         dish names (makes task harder, requires semantic understanding)

    Returns:
        DatasetDict with splits:
        - "honest": Non-Southern European food preferences (negative class)
        - "dishonest": All Southern European food preferences (positive class)
        - "dishonest_{cuisine}": Per-cuisine splits for holdout testing
    """
    rng = random.Random(seed)

    # Generate positive examples (Southern European) - per cuisine
    all_positive = []
    per_cuisine_positive: dict[str, list[dict]] = {}

    for cuisine, dishes in SOUTHERN_EUROPEAN_CUISINES.items():
        examples = generate_examples(
            dishes,
            is_southern_european=True,
            cuisine_name=cuisine,
            rng=rng,
            examples_per_dish=examples_per_dish,
            use_descriptions=use_descriptions,
        )
        per_cuisine_positive[cuisine] = examples
        all_positive.extend(examples)

    # Generate negative examples (non-Southern European) - per cuisine
    all_negative = []
    for cuisine, dishes in NON_SOUTHERN_EUROPEAN_CUISINES.items():
        examples = generate_examples(
            dishes,
            is_southern_european=False,
            cuisine_name=cuisine,
            rng=rng,
            examples_per_dish=examples_per_dish,
            use_descriptions=use_descriptions,
        )
        all_negative.extend(examples)
    negative_examples = all_negative

    # Shuffle
    rng.shuffle(all_positive)
    rng.shuffle(negative_examples)

    # Build dataset dict
    splits = {
        "honest": Dataset.from_list(negative_examples),
        "dishonest": Dataset.from_list(all_positive),
    }

    # Add per-cuisine splits for holdout testing
    for cuisine, examples in per_cuisine_positive.items():
        rng.shuffle(examples)
        splits[f"dishonest_{cuisine}"] = Dataset.from_list(examples)

    return DatasetDict(splits)


def push_to_huggingface(dataset: DatasetDict, repo_id: str) -> None:
    """Push dataset to HuggingFace Hub.

    Args:
        dataset: DatasetDict to push
        repo_id: HuggingFace repository ID
    """
    print(f"\nPushing dataset to HuggingFace: {repo_id}")

    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")

    try:
        dataset.push_to_hub(repo_id, private=False)
        print(f"✓ Successfully pushed to https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"Error pushing to HuggingFace: {e}")
        print("Make sure you're logged in with `huggingface-cli login`")


def save_locally(dataset: DatasetDict, output_path: str) -> None:
    """Save dataset to local JSON files.

    Args:
        dataset: DatasetDict to save
        output_path: Directory to save to
    """
    os.makedirs(output_path, exist_ok=True)

    for split_name, split_data in dataset.items():
        filepath = os.path.join(output_path, f"{split_name}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(list(split_data), f, indent=2, ensure_ascii=False)
        print(f"Saved {len(split_data)} examples to {filepath}")


def main() -> None:
    """Main function to generate food preference dataset."""
    parser = argparse.ArgumentParser(description="Generate food preference dataset for probe generalization testing.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--examples_per_dish",
        type=int,
        default=1,
        help="Number of examples to generate per dish (default 1). Use 3-4 for ~1000 examples.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Directory to save local JSON files (optional)",
    )
    parser.add_argument(
        "--push_to_hf",
        action="store_true",
        help="Push the dataset to HuggingFace Hub",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="AlignmentResearch/food-preference-generalization",
        help="HuggingFace repository ID to push to",
    )
    parser.add_argument(
        "--use-descriptions",
        action="store_true",
        default=True,
        help="Use ingredient-based descriptions instead of dish names (default: True)",
    )
    parser.add_argument(
        "--no-descriptions",
        action="store_true",
        help="Use dish names instead of descriptions (easier task)",
    )
    args = parser.parse_args()

    # Handle description flag
    use_descriptions = args.use_descriptions and not args.no_descriptions

    print("Generating food preference dataset...")
    print(f"Seed: {args.seed}")
    print(f"Examples per dish: {args.examples_per_dish}")
    print(f"Use descriptions: {use_descriptions}")
    print(f"Southern European cuisines: {list(SOUTHERN_EUROPEAN_CUISINES.keys())}")
    print(f"Non-Southern European cuisines: {list(NON_SOUTHERN_EUROPEAN_CUISINES.keys())}")
    print(f"Total Southern European dishes: {sum(len(d) for d in SOUTHERN_EUROPEAN_CUISINES.values())}")
    print(f"Total non-Southern European dishes: {sum(len(d) for d in NON_SOUTHERN_EUROPEAN_CUISINES.values())}")

    dataset = generate_dataset(
        seed=args.seed,
        examples_per_dish=args.examples_per_dish,
        use_descriptions=use_descriptions,
    )

    print("\nGenerated splits:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")

    if args.output_path:
        save_locally(dataset, args.output_path)

    if args.push_to_hf:
        push_to_huggingface(dataset, args.hf_repo_id)

    if not args.output_path and not args.push_to_hf:
        print("\nNo output specified. Use --output_path or --push_to_hf to save the dataset.")


if __name__ == "__main__":
    main()
