# AI Šachový Prediktor

Projekt analyzuje historické šachové partie (na základě PGN zápisů) a pomocí strojového učení dokáže odhadnout šanci na výhru bílého, remízu či výhru černého v libovolné fázi hry.

### Důležité omezení modelu (Statická Evaluace)
Prediktor je natrénován na principech statického ohodnocení pozice (tzv. Static Evaluation). **Nevidí do budoucnosti a nepočítá stromy tahů** (tzv. Depth Search), na rozdíl od enginů jako Stockfish. Oproti prvotním verzím modelu již **dokáže rozeznat bezprostřední hrozbu ztráty (tzv. "hanging" figurky)** a díky informaci o tom, kdo je právě na tahu, s ní dokáže přesně kalkulovat. Co ovšem algoritmus odhalit nedokáže, jsou hluboké taktické oběti na profesionální úrovni. Pokud velmistr úmyslně obětuje dámu za vynucený mat ve 3. tahu, náš model to vyhodnotí jako obří chybu a prohru pro daného hráče, protože analyzuje pouze stávající stav materiálu a struktury a nevykresluje si do paměti možné budoucí scénáře.

## Použité Technologie a Třídy
- **Pre-processing:** `StandardScaler` (pro normalizaci šachových metrik), `LabelEncoder` (převod textových výsledků na pole 0, 1, 2)
- **Umělá Inteligence:** `MLPClassifier` (Multi-Layer Perceptron z rodiny `scikit-learn` jako náhrada za TensorFlow/Keras, který asynchronně nespolupracuje s Python 3.14). Model má 3 skryté vrstvy (128, 64, 32 neuronů) a je optimalizován algoritmem Adam.

## Struktura složek

* `crawler.py` – Skript pro stažení historických partií (PGN) přes veřejné API šachového serveru Chess.com. Udržuje logiku filtrace podle ELO ratingu. Skript je plně **perzistentní**, dokáže ukládat svůj aktuální postup do `.json` záznamu a kdykoliv po přerušení navázat tam, kde naposledy skončil.
* `model.py` – Hlavní soubor umělé inteligence. Načítá pre-procesovaná data, dělí je na trénovací a testovací množiny, provádí standardizaci, spouští trénink neuronové sítě a ukládá hotové modely na disk.
* `lib/` – Obsahuje jak extrakční "mozek", tak kompletní webové rozhraní.
  * `extractor.py` – Parse skript s pomocí knihovny `python-chess`, který analyzuje šachovou pozici a vyrábí z ní podrobné šachové rysy. Skript podporuje zálohování rozpracovaného stavu po dávkách a ochranu proti zamrznutí.
  * `app.py` – Backend server postavený na frameworku Flask, který se startuje k provozu vizuální stránky.

## Jak projekt nainstalovat a spustit

1. **Instalace závislostí**
   Otevřete terminál ve složce s projektem a nainstalujte doporučené knihovny:
   ```bash
   pip install -r requirements.txt
   pip install flask joblib flask-cors pandas scikit-learn chess
   ```

2. **Stažení Dat (Volitelné)**
   Databázi můžete naplnit vlastními historickými tahy (standardně nastaveno na 100 000 partií ELO 850-1800).
   ```bash
   python crawler.py
   ```
   *Tento proces vygeneruje `data/raw_games.json`.*

3. **Extrakce Metrik z Her**
   Pro spuštění extrakce vlastností a přípravu na datový modelování:
   ```bash
   python -m lib.extractor
   ```
   *Tento proces převede .json na trénovací sadu `data/features.csv`.*

4. **Trénink Modelu (Nutné pro Webové Zobrazení)**
   Hlavní ML část:
   ```bash
   python model.py
   ```
   Script vypíše na konzoli statistiky (MAE, MSE, Score), report a confusion matrix. Na závěr vygeneruje sadu .pkl souborů uvnitř adresáře `lib/` (jako jsou weights, biases a the scalers). Bez nich web nelze spustit.

5. **Spuštění Webové Aplikace**
   Přejděte do složky s podpůrnými knihovnami a spusťte server:
   ```bash
   cd lib
   python app.py
   ```
   Přejděte do webového prohlížeče na adresu: **http://127.0.0.1:8080** a potáhněte figurkami! Můžete si zapnout funkci AI Asistenta (tzv. Ghost Piece) a sledovat vizuální stíny tahů přímo na šachovnici, doporučovaných modelem.
