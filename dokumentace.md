# Dokumentace projektu: Chess-Predict

**Škola:** SPŠE Ječná  
**Předmět:** Programové vybavení (PV)  
**Student:** Václav Křivka  
**Třída:** C4c  
**Rok:** 2026  

---

## 1. Úvod a architektura projektu
Tento projekt slouží k predikci výsledku šachové partie (výhra bílého, remíza, výhra černého) v libovolném jejím bodě. Analýza probíhá nad statickou pozicí na šachovnici na základě neuronové sítě, která se naučila ohodnocovat pozice z reálných her středně pokročilých hráčů získaných ze serveru Chess.com.

Projekt se dělí do čtyř logických celků:
1. **Crawler (`crawler.py`)** – Zajišťuje sběr a stahování syrových PGN (Portable Game Notation) záznamů přes veřejné API Chess.com.
2. **Extractor (`lib/extractor.py`)** – Provádí parsování PGN formátu a extrahuje matematické a logické příznaky (features), popisující stav na šachovnici, vhodné pro strojové učení.
3. **Trénování modelu (`model.py`)** – Buduje a trénuje model na získaných a upravených datech.
4. **Rozhraní a predikce (`app.py` v modulu `lib`)** – Flask aplikace obstarávající webové uživatelské rozhraní, přes které lze model dotazovat pro interaktivní odhady pravděpodobnosti.

---

## 2. Zdroje dat a Crawler (`crawler.py`)
Model je založen na učení s učitelem (Supervised Learning). K vytvoření kvalitních trénovacích dat využíváme snowball sampling crawler. 

### Sběr dat (Snowball sampling) a Persistence
Crawler nejprve přes API najde tzv. „seed“ hráče z vybraných zemí, se kterými stahování začne.
- Následně stahuje jejich historické hry.
- Jména soupeřů z těchto her se přidávají do fronty (`player_queue`) a následně se prohledávají i jejich hry. Tímto přístupem (sněhová koule) lze rychle posbírat obrovské množství propojených partií.

 Crawler byl navržen tak, aby byl **zcela perzistentní**. Svůj pokrok neustále ukládá na disk (`data/crawler_state.json`), tudíž jakékoliv nečekané ukončení programu nikoho o data nepřipraví a sběr bude pokračovat tam, kde byl naposledy zanechán. Stažené samotné partie (PGN formát) jsou rovnou dodávány na disk metodou "append" do výsledného `.json` souboru.

### Omezení na ELO 850 – 1800
Omezili jsme stahované hry s parametry:
```python
MINIMUM_RATING = 850
MAXIMUM_RATING = 1800
```
**Proč tyto hodnoty?**
1. **Méně než 850 ELO:** Hry bývají velmi chaotické a obsahují nesystematické a náhodné chyby, z nichž se model nemůže naučit smysluplnou logiku a šachovou strategii (šum v datovém souboru).
2. **Více než 1800 ELO:** Na této úrovni a výše už hráči zvládají pokročilou taktiku. Jelikož extrahujeme statické příznaky bez hlubokého prohledávání stromu tahů (Alpha-Beta pruning / Minimax), model by vůbec nedokázal takto hlubokým myšlenkám porozumět (např. obětování figurky s kompenzací pozice) a jeho přesnost by klesla. (Ověřeno v praxi na zápasech GM Carlsena – viz testovací poznámky `v1.md`).

---

## 3. Extrakce vlastností (Feature Engineering) – `lib/extractor.py`
Tato část vezme konkrétní pozici na šachovnici v rámci hry a vytvoří číselné `features`. Knihovna `python-chess` zde vyhodnocuje šachovnici.

**Příklady klíčových extrahovaných atributů:**
- **Materiál (`compute_material_difference`)**: Počítá převahu materiálu součtem tradičních hodnot: Pěšec (1), Jezdec (3), Střelec (3), Věž (5), Dáma (9).
- **Struktura pěšců (`count_pawn_structure_features`)**: Speciální detektor zkoumá a sčítá nežádoucí zdvojené a izolované pěšce, případně pozitivně identifikuje nechráněné „passed“ (volné) pěšce pro obě dvě strany k zjištění zablokovanosti defenzívy a potenciálu tlaku.
- **Bezpečnost krále (`compute_king_safety` a `compute_king_exposure`)**: Hodnotí perimetr krále analýzou přítomnosti štítu z pěšců a počet okolních nepřátelsky chráněných zón pro možný nátlak sítě.
- **Ohrožený (Hanging) materiál (`compute_hanging_material`)**: Extrémně důležitá heuristika. Sumarizuje hodnotu nechráněných figur napadených soupeřem.
- **Kdo je na tahu (`side_to_move`)**: Přidali jsme zjišťování toho, kdo je aktuálně na tahu. Je-li ohrožena figurka hráče na tahu, není nutně ztracena. Model se učí vážit „Hanging material“ právě v kontextu tohoto atributu, což odstraňuje tzv. fantomové skoky v predikci procent.

*Extrakce samotná plně běží přes perzistentní logiku, je schopná si pamatovat po provedení určitých po sobě jsoucích dávek dat (batchování) index pro parsování. Je chráněna proti memory overflow z důvodu počtu zkoumaných dat a výstupy pravidelně ihned exportuje bez nebezpečí prohrotějších kolapsů operační paměti do příslušného souboru s příponou `.csv` na diksu.*

```python
# Kousek kódu simulující jak vypadají vyextrahované atributy pro AI (Candidate row):
features = {
    "game_id": game_index,
    "white_elo": white_elo,
    "cand_side_to_move": 1,
    "cand_material_diff": material_diff,
    "cand_white_hanging": white_hanging,
    "cand_white_king_exposure": white_king_exposure,
    "was_played": 1 # Označuje, zda model predikuje tah skutečně realizovaný lidskou ruku
    # ... celkem pres neuvěřitelných 200 atributů včetně tahové historie z 3 posledních stavů před tahem
}
```

---

## 4. Strojové učení a Model (`model.py`)
Místo složitějších frameworků jako TensorFlow či Keras jsme z důvodů stabilní kompatibility na Pythonu 3.14 (v době vývoje) zvolili robustní knihovnu **`scikit-learn`**. Naším modelem je klasifikátor využívající plně propojenou vícevrstvou neuronovou síť (MLP).

### Definice a nastavení modelu
```python
model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.0005,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,
    random_state=1
)
```

**Odůvodnění hyperparametrů:**
- **`hidden_layer_sizes=(128, 64, 32)`**: Architektura vrstev, která klesá směrem dolů, aby si síť z rozsáhlých 50 atributů vytvořila abstraktnější šachová skrytá pravidla před finální klasifikací na tři třídy (Výhra Bílý, Remíza, Výhra Černý).
- **`activation='relu'`**: Standardní aktivační funkce ReLU (Rectified Linear Unit), která byla vybrána jako rychlá a předcházející zmizení gradientu (vanishing gradient).
- **`solver='adam'`**: Adaptivní optimalizátor, který funguje dobře i při větším množství dat, poměrně rychle konverguje a dosahuje stabilních výsledků.
- **`early_stopping=True`** (s `validation_fraction=0.1` a `n_iter_no_change=10`): Zabraňuje přetrénování na trénovacích datech (overfitting). Z dat se oddělí 10 % jako validační sada. Pokud se síť na těchto 10 % nezlepší po dobu 10 epoch, učení se předčasně ukončí. Zabrání to tomu, aby si model data „bifloval nazpaměť“.

### Normalizace dat a Enkódování
Před trénováním data zpracujeme:
- `StandardScaler()` ze `scikit-learn` oškáluje numerické atributy (např. materiální převahu v bodech) tak, aby jejich průměr byl 0 a směrodatná odchylka 1. Neuronová síť tak zpracovává hodnoty na sjednocené číselné stupnici.
- `LabelEncoder()` překládá textové označení výsledku partie (`1-0`, `0-1`, `1/2-1/2`) do číselné formy tříd.

### Ukládání modelu s využitím `joblib`
Aby webová aplikace nemusela před každou predikcí znovu trénovat síť (což by trvalo minuty až hodiny), trénovací skript po úspěšném natrénování uloží naučený model na disk. Nástroj `joblib` serializuje interní struktury a váhy modelu:
```python
joblib.dump(model, 'lib/trained_model.pkl')
joblib.dump(scaler, 'lib/scaler.pkl')
joblib.dump(label_encoder, 'lib/label_encoder.pkl')
```

---

## 5. Komunikace modelu s aplikací (`app.py`)
Sekce webového rozhraní v modulu `lib/app.py` využívá lehký aplikační rámec Flask. Flask komunikuje s uloženými `joblib` parametry.

Proces probíhá následovně, jakmile uživatel zkusí zhodnotit PGN na Frontendu:
1. **Nahrání modelu (`joblib.load()`):** Načte se `.pkl` se silnou neuronovou sítí.
2. **Analyzování (`extractor.extract_features()`):** Zadaný PGN řetězec se předá do extraktoru. Vygeneruje se úplně ten samý set vlastností, se kterým se model dříve učil.
3. **Příprava (Škálování pomocí `scaler.transform()`):** Aby model pozici porozuměl, použijeme uložený `scaler` a hodnoty upravíme (normalizujeme) na stejné škále, na jaké probíhal trénink.
4. **Predikce (`model.predict_proba()`):** Namísto toho, aby síť řekla pouze absolutní výsledek (Výhra bílý), poskytne pole pravděpodobností v % pro všechny 3 výsledky.

### Datová výměna (API POST dotaz a odpověď)
Když webový frontend (JavaScript) žádá o predikci, sestaví takovýto minimalistický `POST` požadavek s tělem ve formátu JSON na adresu `/api/predict`:
```json
{
  "pgn": "1. e4 e5 2. Nf3 Nc6"
}
```

Aplikace zpracuje text do procent přesně těmito kroky:
1. **Extrakce z PGN do čísel:** Použije se `extractor.extract_features()`.
2. **Standardizace a Predikce:** Modelem `model.predict_proba()` se získají pole syrových desetinných hodnot, např. `[0.75, 0.05, 0.20]`.
3. **Převod na procenta a identifikace přes Encoder:**
   Nesmíme ty hodnoty pouze vzít, model neví, která hodnota z pole patří jaké barvě. Od toho si z předchozího kroku trénování do souboru ukládáme i náš dřívější `LabelEncoder` do proměnné `label_encoder`. Ten si pamatuje původní řazení tříd (např. `['0-1', '1-0', '1/2-1/2']`). 
   Využijeme dynamického mapování pole pomocí indexů a zároveň to vynásobíme hodnotou 100, abychom z hrubých desetinných míst získali pěkné procento (0.75 -> 75%):
   ```python
   classes = list(label_encoder.classes_)
   result = {
       "white": probabilities[classes.index('1-0')] * 100,
       "draw": probabilities[classes.index('1/2-1/2')] * 100,
       "black": probabilities[classes.index('0-1')] * 100
   }
   ```

Tento výsledek je pro Flask logiku trochu složitější:
Získáme sadu validních tahů a každý jednotlivě evaluujeme v extraktoru. Vyšle se plné pole možností (např. 25 různých tahů z pozice) do neuronové sítě a hledá se index s maximální pravděpodobností (tzv. `argmax`). Tento nejlepší kandidátský posun se následně v doprovodu procentuálního předpokladu úspěchu odesílá do WebAppky:

```json
{
  "success": true,
  "best_move": "e2e4",
  "probability": 75.6
}
```

Tímto datovým tokem (Workflow) je ukázáno, jak projekt komplexně sbírá, hodnotí, učí se a nakonec poskytuje interaktivní vyhodnocení koncovým uživatelům.

---

## 6. Využití šachových knihoven (Engine) na Frontendu a Backendu
Aby náš projekt správně chápal pravidla hry a dokázal analyzovat probíhající hru, používáme takzvané "šachové enginy". Tyto knihovny zde ovšem nefungují jako samotná umělá inteligence (nehrají proti člověku), ale slouží jako striktní **rozhodčí a analytici pravidel**. Zajišťují validní pohyb figurek a sledují herní stav.

### Backend (Python) – `python-chess`
Modul `lib/extractor.py` hojně využívá importovanou knihovnu `chess`. Bez ní by bylo prakticky nemožné z holého textu (PGN) extrahovat námi definované číselné atributy potřebné pro model. Její přínos spočívá v tomto:
1. **Analýza textového vstupu (PGN):** Pomocí funkce `chess.pgn.read_game()` vezme PGN zápis z webu a v paměti postaví virtuální šachovnici posun po posunu.
2. **Dodržování pravidel (`board = game.board()` a postupy tahů):** V paměti interně simuluje hru, přehrává posuny a aplikuje tak vzácnější pravidla jako En Passant či Rošády.
3. **Analytika formou dotazovačů:** Knihovna obsahuje vestavěné chytré detekce. Můžeme se jí dotázat například: *"Které všechny políčka ovládá bílý král?"* (`board.is_attacked_by()`), vypsat seznam všech polí s pěšcem (`board.pieces()`), a nebo vygenerovat pole v danou chvíli možných legálních tahů (`list(board.legal_moves)`) – toto nám slouží u počítání příznaku pro mobilitu.

### Frontend (JavaScript) – `chess.js` a `chessboard.js`
Na klientově straně (uvnitř prohlížeče) nechceme kvůli každému uživatelsky taženému tahu s figurkou komunikovat se serverem a ptát se, zda se jedná o legální tah, kvůli ušetření server costů a datového toku. Používáme dvě oddělené super-lehké JS knihovny, jedna plní mozek, druhá svaly.
1. **Logika (`chess.js`)**: Jedná se o hlídače událostí. Knihovna kontroluje neustále, kdo je na tahu (`game.turn()`), zabraňuje táhnutí obou barev najednou a zakazuje krádež cizí figurky. Zajišťuje stavy jako je automatické zjištění Šachu (`game.in_check()`) nebo patu s okamžitým oznámením remízy. Hlavně také formátuje na výstup finální textový PGN (`game.pgn()`), který posíláme přes AJAX pomocí API požadavků s hlavičkami Fetch na náš Python Backend k odeslání do neuronové sítě.
2. **Vizuál (`chessboard.js`)**: Knihovna sloužící čistě jako grafické API pro Front-End. Obezstarává hladké Drag & Drop (táhni a pusť) na frontendu, vizualizuje herní pole a aplikuje PNG animace s obrázky šachových figurek z úložiště wikipedie, abychom my nemuseli generovat kód pro pole velikosti ohromných matic pro HTML. Součástí je také animace vrácení figurky zpět v případě, že "Mozek" (`chess.js`) vyhodnotí hráčem posunutý tah za nelegální (vrácení tahu `snapback`). 

Navíc je do struktury HTML přidán přepínač **AI Asistent**, který do `chessboard.js` vykresluje a uměle dokládá tzv. "Ghost Piece" – animaci zlevněného žlutého průhledného odrazu figurky, jíž umělá inteligence aktuálně považuje za statisticky nejvýhodnější přesun v kontextu chování z reálných partií, společně se žlutým filtrem nad výchozím a cílovým indexem podkladové šachovnice.
