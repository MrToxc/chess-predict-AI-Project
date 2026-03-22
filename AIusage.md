# Použití Umělé Inteligence v Projektu (AI Usage)

Tento dokument vysvětluje míru a způsob využití nástrojů umělé inteligence (konkrétně Claude Opus 4.6 a Gemini 3.1 Pro) při vývoji tohoto ročníkového projektu.

## Co jsem dělal já (Autorský přínos)
Základní aplikační logiku, stahování dat a samotné modely jsem vyvíjel sám. Můj autorský kód zahrnuje:
- **`crawler.py`** – Architektura stahování dat, procházení archivů přes Chess.com API a aplikování filtrů na rating hráčů.
- **`model.py`** – Architektura neuronové sítě a předzpracování dat. Kód využívá profesionální třídy z knihovny `scikit-learn`: `StandardScaler` (pro normalizaci vstupů na stejnou škálu), `LabelEncoder` (pro převod textových výsledků 1-0/0-1 do číselné podoby) a hlavně **`MLPClassifier`** (samotná plně propojená neuronová síť s funkcí Softmax na konci). Kód jsem navíc refaktorizoval do čistých funkcí pro lepší čitelnost.
- **Základní myšlenka šachových metrik** – Navrhl jsem základní atributy pro vyhodnocení pozice: aktuální číslo tahu (`move_number`), rozdíl materiálu (`material_diff`), celkový materiál obou stran (`white_material`, `black_material`) a počet vyvinutých figur (`white_developed`, `black_developed`).

## Kde a proč mi pomáhala AI

Během testování mé první verze se ukázalo, že základní metriky nepřinášejí dostatečnou přesnost neuronové sítě. Narazil jsem zde na limity svých "ne-profesionálních" šachových znalostí. Nevěděl jsem, jak algoritmicky popsat a extrahovat pokročilejší šachové principy ze surových PGN dat.

### 1. Rozšíření Extraktoru (`lib/extractor.py`)
- **Účel a Prompt:** Požádal jsem AI: *"Mám základní metriky (material, vyvinutí figur, číslo tahu), ale přesnost modelu je příliš nízká. Napiš mi pokročilý extraktor, který z PGN zápisu vytáhne další atributy (např. bezpečnost krále, pěšcovou strukturu, mobilitu) a zvýší tak úspěšnost predikce."*
- **Výsledek:** AI (Claude Opus 4.6) mi pomohlo vymyslet a napsat logiku pro generování metrik jako `doubled_pawns`, `king_safety`, `ext_center_control` (namísto původních nerealistických názvů) nebo `hanging_material` a integraci knihovny `python-chess` ke zjišťování ohrožených políček. Výrazně se tím zvedla přesnost.

### 2. Prezentační Webové Rozhraní (`lib/app.py` a front-end)
- **Účel a Prompt:** Postavit webové rozhraní nebylo striktní náplní tohoto cvičení. Abych však mohl výsledek práce modelu lépe vizualizovat, rozhodl jsem se ušetřit čas a nechat vizuál vygenerovat pomocí AI.
- **Výsledek:** AI mi dogenerovalo "boilerplate" HTML šablonu, CSS styly a obslužný JavaScript pro propojení s knihovnou `chessboard.js` a vygenerovalo základního Flask klienta, který předává FEN strom jako JSON.

---
Všechny klíčové datové a ML aspekty projektu jsou mým vlastním návrhem a dílem, umělou inteligenci jsem využil čistě jako expertního šachového konzultanta (feature engineering) a kodéra pro vedlejší vizualizační prvky (HTML/CSS), čímž jsem optimalizoval čas strávený na nesouvisejících technologiích.
