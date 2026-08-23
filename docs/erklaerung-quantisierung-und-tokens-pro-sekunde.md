# Quantisierung, Tokens/Sekunde & der "Sweet Spot" – einfach erklärt

Diese Seite erklärt die Begriffe aus dem [Ollama-vs-vLLM-Benchmark](benchmark-ollama-vs-vllm-qwen3.8-27b.md)
in verständlicher Sprache – ohne Vorwissen vorauszusetzen.

## 1. Was ist ein "Token"?

Ein Token ist ein Wort-Stück. Nicht ganze Wörter, aber auch nicht einzelne
Buchstaben – irgendwas dazwischen. "Photosynthese" könnte z.B. in die Tokens
`Photo`, `synthe`, `se` zerlegt werden. Ein KI-Modell liest und schreibt nicht in
Buchstaben oder Wörtern, sondern in diesen Tokens.

**Tokens pro Sekunde (Tok/s)** sagt also: wie viele dieser Wort-Stücke erzeugt das
Modell pro Sekunde, wenn es antwortet. Höher = die Antwort erscheint schneller.

## 2. Was ist "Quantisierung"?

Ein KI-Modell besteht im Kern aus Milliarden Zahlen ("Gewichte") – bei einem
27-Milliarden-Parameter-Modell wie Qwen3.8-27B eben ~27 Milliarden Zahlen. Wie genau
(mit wie vielen Nachkommastellen bzw. wie viel Speicherplatz) jede einzelne dieser
Zahlen gespeichert wird, nennt man **Präzision**.

**Quantisierung** bedeutet: diese Zahlen werden gröber/ungenauer gespeichert, um
Speicherplatz zu sparen. Stell es dir vor wie Foto-Kompression: Ein JPEG mit hoher
Kompression braucht weniger Speicherplatz, sieht aber bei genauem Hinsehen etwas
unschärfer aus als das Originalbild. Genauso ist ein stärker quantisiertes Modell
kleiner und schneller, aber (in der Theorie) etwas "unschärfer" in seinen Antworten.

### Die gängigen Stufen

| Bezeichnung | Bits pro Zahl | Größe relativ zum Original (meist BF16/FP16) |
|---|---|---|
| BF16 / FP16 | 16-bit | 100% (Referenz, "volle Präzision") |
| **FP8** | 8-bit | ~50% |
| **INT4 / AWQ / Q4_K_M** | 4-bit | ~25% |

- **FP8** ("Floating Point, 8-bit"): das Modell in diesem Projekt nutzt das für die
  meisten Modelle. Halbiert die Größe gegenüber dem Original, bei sehr geringem
  Qualitätsverlust – moderne GPUs (auch die GB10 hier) haben oft extra Hardware, die
  FP8-Rechnungen beschleunigt.
- **INT4 / AWQ / Q4_K_M**: verschiedene Verfahren, die auf 4-bit runterquantisieren
  (viertel der Originalgröße). `Q4_K_M` ist das Format, das Ollama/llama.cpp
  verwendet; `AWQ` und `GPTQ` sind die gängigen 4-bit-Formate für vLLM. Deutlich
  kleiner und schneller als FP8, aber etwas höheres Risiko für Qualitätsverlust bei
  komplexen/logischen Aufgaben.

**Wichtig:** Es ist nicht " FP8 ist gut, INT4 ist schlecht" – es ist ein Kompromiss.
Für viele Alltagsaufgaben (Chat, Zusammenfassungen, einfaches Coding) ist der
Unterschied zwischen FP8 und gut gemachtem INT4/AWQ kaum spürbar. Bei Aufgaben mit
vielen Zahlen, komplexer Logik oder langen Tool-Call-Ketten kann INT4 eher mal
danebenliegen.

### Dateiformat vs. Quantisierungsverfahren: GGUF vs. AWQ

`GGUF` und `AWQ` tauchen beide im Zusammenhang mit 4-bit-Modellen auf, sind aber
**unterschiedliche Dinge** – das eine ein Dateiformat, das andere eine
Quantisierungs-Methode:

- **GGUF** ist ein **Dateiformat** (wie `.zip` oder `.pdf` ein Dateiformat ist) –
  entwickelt für `llama.cpp`, die Software, auf der Ollama aufbaut. Eine `.gguf`-
  Datei bündelt Gewichte, Tokenizer und Metadaten in einer einzigen Datei und kann
  *verschiedene* Quantisierungsstufen enthalten (`Q4_K_M`, `Q5_K_M`, `Q8_0`, ...).
  Wenn du in Ollama ein Modell ziehst, bekommst du eine GGUF-Datei.
- **AWQ** ("Activation-aware Weight Quantization") ist ein **Quantisierungs-
  Verfahren** – ein Algorithmus, der beim Runterrechnen auf 4-bit intelligent
  vorgeht: er beobachtet, welche Gewichte beim Verarbeiten echter Beispieltexte
  besonders wichtig sind (viel "Aktivität" erzeugen), und behandelt genau die
  vorsichtiger, um Qualitätsverlust zu minimieren. Das Ergebnis wird typischerweise
  als `safetensors`-Datei gespeichert (das Standard-Format für vLLM & Co.), nicht
  als GGUF.

|  | GGUF | AWQ |
|---|---|---|
| Was es ist | Dateiformat | Quantisierungs-Verfahren |
| Ökosystem | llama.cpp / Ollama | vLLM, TensorRT-LLM & Co. |
| Enthält | Gewichte + Tokenizer + Metadaten, eine Datei | Gewichte als `safetensors` (Standard-HF-Format) |
| Läuft in diesem Projekt? | **Nein** – vLLM 0.26 (installierte Version hier) hat GGUF-Unterstützung komplett entfernt | Ja, nativ unterstützt |

**Warum das gerade konkret relevant ist:** Ollamas `qwen3.8:27b` liegt als GGUF-Datei
vor (`Q4_K_M`-Quantisierung). Weil vLLM diese Datei nicht laden kann, gibt es keine
direkte Abkürzung – die einzigen Wege zu einem vergleichbar schnellen, vLLM-
tauglichen 4-bit-Modell sind entweder eine fertige AWQ-Version von woanders laden,
oder die offiziellen (unquantisierten) Original-Gewichte selbst per AWQ
quantisieren (genau das läuft gerade im Hintergrund für `Qwen3.8-27B`).

## 3. Warum macht kleinere Präzision das Modell schneller?

Das ist der Kern dessen, was der Benchmark gezeigt hat. Wenn das Modell ein Token
erzeugt, muss es (vereinfacht gesagt) **alle** Modell-Gewichte einmal aus dem
Speicher lesen. Das ist wie ein Bibliothekar, der für jede einzelne Buchempfehlung
das komplette Regal einmal abgehen muss.

- Bei FP8 sind die "Bücher" (Gewichte) doppelt so "dick" wie bei INT4/Q4.
- Also muss doppelt so viel Speicher-Datenverkehr passieren, um dieselbe Menge an
  Gewichten zu lesen.
- Auf Hardware wie der GB10 hier, wo sich CPU und GPU denselben Speicher(-kanal)
  teilen ("Unified Memory"), ist genau dieser Datenverkehr oft der Flaschenhals –
  nicht die reine Rechenleistung.

Das erklärt, warum im Benchmark Ollama (Q4_K_M) ungefähr 2.5x schneller war als
vLLM (FP8): fast exakt der Faktor, den man erwarten würde, wenn halb so viele Bytes
pro Token gelesen werden müssen.

## 4. Prefill vs. Decode – zwei verschiedene Geschwindigkeiten

Eine Antwort des Modells hat zwei Phasen:

1. **Prefill**: Das Modell liest deinen gesamten Prompt (die Frage/den Kontext) und
   verarbeitet ihn – bevor überhaupt das erste Antwort-Token entsteht. Das lässt sich
   gut parallelisieren, ist deshalb meist deutlich schneller (im Test: ~94–117 Tok/s).
2. **Decode**: Das Modell erzeugt die Antwort Token für Token. Jedes neue Token
   braucht einen eigenen Durchlauf durchs Modell – lässt sich schlechter
   parallelisieren, ist deshalb langsamer (im Test: 6–15 Tok/s) und genau die Phase,
   die durch Quantisierung am meisten beschleunigt wird.

**TTFT** ("Time To First Token") ist die Zeit, bis das *erste* Antwort-Token da ist –
im Wesentlichen die Prefill-Zeit. Das ist z.B. relevant dafür, wie "responsiv" sich
ein Chat anfühlt, auch wenn die Gesamtantwort dann mit der Decode-Geschwindigkeit
weiterläuft.

## 5. Was ist der "Sweet Spot"?

Es gibt keine pauschal "beste" Quantisierung – der Sweet Spot hängt davon ab, was dir
wichtiger ist:

```
volle Präzision (BF16)  ──────────────────────────────►  starke Quantisierung (INT4)
      langsamer                                                  schneller
      größer (mehr GB)                                           kleiner (weniger GB)
      "genauer"                                                  etwas mehr Risiko für Fehler
      weniger Modelle gleichzeitig                                mehr Modelle passen in den Hot Pool
      im Speicher (Hot Pool)
```

Für **dieses Projekt** (persönlicher Server, GB10 mit ~121GB Unified Memory,
hauptsächlich Coding-Assistent in VS Code + gelegentliche PDF-Auswertung) ist ein
vernünftiger Sweet Spot ungefähr:

- **FP8** für Modelle, bei denen dir Genauigkeit wichtig ist (z.B. strukturierte
  JSON-Extraktion aus PDFs, wo ein falsches Token das ganze JSON kaputt macht) und wo
  die aktuelle Geschwindigkeit (6 Tok/s bei 27B) ausreicht.
- **INT4/AWQ** für Modelle, bei denen dir Antwortgeschwindigkeit wichtiger ist als
  letzte Genauigkeits-Prozentpunkte (z.B. schnelles Autocomplete/Chat in VS Code) –
  siehe die offene Frage im [Benchmark](benchmark-ollama-vs-vllm-qwen3.8-27b.md), ob
  eine AWQ-Version von Qwen3.8-27B das Tempo von Ollama erreicht.
- Kleinere Modelle (4B–8B) sind ohnehin schnell genug, dass sich die Frage seltener
  stellt – da lohnt sich eher FP8/BF16 für die bessere Qualität.

## 6. Die Qwen-Modellfamilie – welche Kategorie ist was?

Auf [huggingface.co/Qwen/collections](https://huggingface.co/Qwen/collections) listet
Qwen alle seine Modell-Reihen. Das sind nicht alles "dieselbe Art Modell in
verschiedenen Größen" – die Kategorien unterscheiden sich in dem, was das Modell
überhaupt kann bzw. wofür es gebaut ist.

### Sprachmodelle (die "normalen" Chat-Modelle)

Generationen desselben Grundmodells, neueste zuerst: **Qwen3.8, Qwen3.5, Qwen3,
Qwen2.5, Qwen2, Qwen1.5, Qwen**. Jede Generation ist ein Sprung in Trainingsdaten/
Architektur – Qwen3.8 ist aktuell die fähigste, Qwen (ohne Nummer) die
ursprüngliche von 2023. Aus dieser Kategorie kommen die meisten Modelle in diesem
Projekt (`Qwen3.8-27B`, `Qwen3-8B`, ...).

### Spezialisierte Sprachmodelle (Qwen-Basis, auf eine Aufgabe zugeschnitten)

- **Qwen3-Coder(-Next), Qwen2.5-Coder**: auf Code trainiert – bessere
  Programmier-/Debugging-Fähigkeiten als das allgemeine Modell gleicher Größe.
  `Qwen3-Coder-30B-A3B-Instruct-FP8` in diesem Projekt kommt von hier.
- **QwQ**: "Qwen with Questions" – frühe reasoning-fokussierte Modelle (viel "lautes
  Nachdenken" vor der Antwort), Vorläufer der heutigen Thinking-Modi in Qwen3+.
- **Qwen2.5-Math, Qwen2-Math**: auf Mathe-Aufgaben spezialisiert (Beweise,
  Gleichungen) – für ein Coding/Chat-Setup normalerweise nicht relevant.

### Multimodale Modelle (sehen/hören zusätzlich zu Text)

- **Qwen3-VL, Qwen2.5-VL, Qwen2-VL**: "Vision-Language" – verstehen Bilder/
  Screenshots zusätzlich zu Text. Genau das, was in diesem Projekt `vision: true`
  bedeutet (`Qwen3.8-27B-FP8`, `Qwen3.6-35B-A3B-FP8`).
- **Qwen3-Omni, Qwen2.5-Omni**: wie VL, aber zusätzlich Audio/Video als Eingabe –
  "alles rein, alles raus" in einem Modell.

### Sprache & Audio (kein Text-Chat, andere Aufgabe)

- **Qwen3-ASR**: Speech-to-Text – Audio in Text umwandeln (Transkription).
- **Qwen3-TTS**: Text-to-Speech – Text in gesprochene Sprache umwandeln.
- **Qwen2-Audio**: allgemeines Audio-Verständnis (nicht nur Transkription).

Diese laufen **nicht** über `/v1/chat/completions` – andere Modellarchitektur,
bräuchten einen eigenen Serving-Pfad.

### Bild-Generierung & Retrieval

- **Qwen-Image, Qwen-Image-Bench**: Text-zu-Bild-Generierung/-Bearbeitung (wie
  Stable Diffusion), plus ein Benchmark-Datensatz dafür.
- **Qwen3-Reranker, Qwen3-VL-Reranker**: kein Chat – sortiert eine Liste von
  Dokumenten/Bildern nach Relevanz zu einer Anfrage (Baustein für Suche, nicht für
  Konversation).

### Embeddings (Text/Bild in Zahlenvektoren umwandeln)

**Qwen3-Embedding, Qwen3-VL-Embedding**: wandeln Text (bzw. Bilder) in Vektoren um,
mit denen man semantische Ähnlichkeit berechnen kann – die Grundlage für RAG/
Vektorsuche. Kein Chat-Modell, aber vLLM kann sowas laufen lassen (eigener
`--task embed`-Modus). Relevant, falls irgendwann RAG über eigene Dokumente gebaut
werden soll.

### Sicherheit & Auswertung

- **Qwen3Guard**: Content-Filter – prüft Ein-/Ausgaben auf problematische Inhalte,
  läuft als zusätzliches Modell neben dem eigentlichen Chat-Modell.
- **Qwen-Scope**: Interpretierbarkeits-Tools ("Sparse Features") – Forschung, um zu
  verstehen, *warum* ein Modell etwas sagt. Kein Produktiv-Einsatz.
- **WorldPM**: Preference-Modeling – bewertet, welche von mehreren Antworten
  "besser" ist (wird beim Trainieren neuer Modelle verwendet, nicht beim Nutzen).

### Agenten-Systeme

**Qwen-AgentWorld, WebWorld**: experimentelle "World Models" – Modelle, die eine
Umgebung simulieren/vorhersagen, damit ein Agent darin planen kann (z.B. Web-
Navigation). Forschungsnäher, nicht das, was für einen Coding-Assistenten gebraucht
wird.

**Für dieses Projekt relevant sind eigentlich nur zwei Kategorien:** *Qwen3.8/Qwen3*
(reine Sprachmodelle) und *Qwen3-Coder* (code-spezialisiert) – daher kommen alle
aktuell registrierten Modelle. *Qwen3-VL* wird indirekt schon mitgenutzt, weil
`Qwen3.8-27B` und `Qwen3.6-35B` als multimodale Varianten laufen (`vision: true`).

## 7. Was ist RAG und wozu braucht man es?

**RAG = Retrieval-Augmented Generation** – auf Deutsch sinngemäß "durch Suche
angereicherte Text-Erzeugung". Die Idee: Statt dem Modell alles aus dem
"Gedächtnis" (seinen trainierten Gewichten) beantworten zu lassen, gibt man ihm
vor der Antwort erst die passenden Informationen aus einer externen Quelle mit.

### Das Problem, das RAG löst

Ein Sprachmodell kennt nur das, was in seinen Trainingsdaten stand – und das hat
einen Stichtag. Es weiß z.B. nichts über:

- **eure eigenen Dokumente** (PDFs, Rechnungen, interne Notizen, dieses Repo)
- **Dinge, die nach dem Trainings-Stichtag passiert sind**
- **Sehr spezifisches/internes Wissen**, das nirgendwo öffentlich im Internet stand

Zwei "klassische" Lösungen dafür haben Nachteile:

- **Alles in den Prompt packen** – funktioniert nur, solange es ins Kontextfenster
  passt (siehe `max_model_len` weiter oben), wird bei vielen Dokumenten schnell zu
  groß und teuer (mehr Tokens = mehr Zeit, siehe Prefill).
- **Nachtrainieren (Fine-Tuning)** – aufwändig, teuer, und muss bei jeder Änderung
  der Daten wiederholt werden.

**RAG** löst das anders: Nur die paar Textstellen, die für die *aktuelle Frage*
tatsächlich relevant sind, werden zur Laufzeit rausgesucht und mit in den Prompt
gepackt – der Rest des Wissens bleibt "draußen", muss also nicht ins
Kontextfenster oder ins Modell selbst.

### Wie funktioniert es (vereinfacht)?

```
1. Vorbereitung (einmalig, pro Dokumentensammlung):
   Dokumente → in kleine Abschnitte zerlegen → jeden Abschnitt in einen
   Vektor umwandeln (Embedding, siehe Abschnitt 6) → in einer Vektor-
   Datenbank speichern

2. Zur Laufzeit (bei jeder Anfrage):
   Deine Frage → ebenfalls in einen Vektor umwandeln → die ähnlichsten
   Abschnitte aus der Vektor-Datenbank raussuchen ("Retrieval")
   → diese Abschnitte + deine Frage zusammen als Prompt ans Sprachmodell
   → Modell formuliert die Antwort auf Basis dieser Abschnitte ("Generation")
```

Der Vektor-Vergleich funktioniert, weil ähnliche Bedeutungen zu ähnlichen Vektoren
werden – die Suche findet also auch Abschnitte, die das gesuchte Wort gar nicht
wörtlich enthalten, sondern nur sinngemäß Verwandtes (anders als eine reine
Stichwortsuche wie `grep`).

### Wozu ist das gut?

- **Aktuelle/private Informationen beantworten**, ohne das Modell neu zu trainieren
  – einfach neue Dokumente in die Vektor-Datenbank aufnehmen.
- **Weniger Halluzination**: Das Modell "erfindet" seltener Fakten, wenn es die
  passenden Textstellen direkt vor Augen hat, statt aus dem Gedächtnis zu raten.
- **Nachvollziehbarkeit**: Man kann anzeigen, *aus welchem* Dokumentenabschnitt eine
  Antwort stammt (Quellenangabe).
- **Günstiger als Fine-Tuning**, vor allem wenn sich die Daten häufig ändern.

### Bezug zu diesem Projekt

Aktuell macht keines der laufenden Setups echtes RAG – das PDF-Auswerte-Projekt
(`llm_pdf_parser.py`) schickt pro PDF-Seite direkt den ganzen Text ans Modell und
lässt strukturierte Daten extrahieren (eher "ein Dokument komplett verarbeiten"
als "die richtige Textstelle aus vielen Dokumenten finden"). RAG würde erst
relevant, wenn man z.B. fragen möchte *"in welcher Rechnung ging es um den
Ölwechsel beim Lamborghini?"* über eine große Sammlung von Dokumenten hinweg,
ohne jedes Mal alle PDFs erneut komplett durch den Kontext zu schicken. Die
Bausteine dafür wären: ein Embedding-Modell (`Qwen3-Embedding`, siehe Abschnitt 6)
plus eine Vektor-Datenbank – aktuell nicht Teil dieses Setups.

## 8. Kurz-Glossar

| Begriff | Bedeutung |
|---|---|
| Token | Ein Wort-Stück – die kleinste Einheit, in der das Modell "denkt" |
| Tok/s | Tokens pro Sekunde – wie schnell das Modell Text erzeugt |
| Quantisierung | Gewichte gröber speichern, um Speicherplatz + Geschwindigkeit zu gewinnen |
| FP8 | 8-bit-Fließkomma-Format, ~50% der Originalgröße, geringer Qualitätsverlust |
| INT4 / AWQ / GPTQ / Q4_K_M | 4-bit-Formate, ~25% der Originalgröße, mehr Tempo, etwas mehr Risiko |
| GGUF | Dateiformat von llama.cpp/Ollama, bündelt Gewichte+Tokenizer+Metadaten in einer Datei – in diesem Projekt (vLLM) nicht ladbar |
| AWQ | "Activation-aware Weight Quantization" – Quantisierungs-Verfahren, das wichtige Gewichte beim Runterrechnen auf 4-bit schont; Ergebnis als `safetensors`, vLLM-kompatibel |
| Prefill | Verarbeitung des Prompts, bevor die Antwort beginnt (schnell, parallelisierbar) |
| Decode | Erzeugung der Antwort Token für Token (langsamer, der Teil, den Quantisierung am meisten beschleunigt) |
| TTFT | Time To First Token – Zeit bis zum ersten Antwort-Token (≈ Prefill-Zeit) |
| Speicherbandbreite | Wie schnell Daten zwischen Speicher und Rechenkernen fließen können – bei Decode oft der Flaschenhals |
| Unified Memory | CPU und GPU teilen sich denselben Speicher(-kanal) – wie bei der GB10 hier |
| RAG (Retrieval-Augmented Generation) | Vor der Antwort erst passende Textstellen aus eigenen Dokumenten raussuchen und dem Modell mitgeben, statt alles aus dem Training zu "erraten" |
| Embedding | Text (oder Bild) in einen Vektor umgewandelt, mit dem sich Ähnlichkeit berechnen lässt – die Grundlage für RAG |
| Vektor-Datenbank | Speicher für Embeddings, optimiert auf "finde die ähnlichsten Vektoren zu diesem hier" |
