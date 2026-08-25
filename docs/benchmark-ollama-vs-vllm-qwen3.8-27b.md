# Benchmark: Ollama vs. vLLM – Qwen3.8-27B

**Datum:** 2026-08-22
**Hardware:** zgx-0cc3 (NVIDIA GB10 / DGX-Spark-artig, ~121GB Unified Memory)

## Ergebnis

| | Ollama (`qwen3.8:27b`) | vLLM FP8 | vLLM AWQ-INT4 v1 | **vLLM AWQ-INT4 v2** |
|---|---|---|---|---|
| Quantisierung | Q4_K_M (4-bit) | FP8 (8-bit) | AWQ-INT4 (nur Attention+MLP) | **AWQ-INT4 (+ Mamba-Layer)** |
| **Decode-Speed** | **15.2 Tok/s** | 6.1 Tok/s | 7.7 Tok/s | **11.2 Tok/s** |
| Anteil von Ollama | 100% | 40% | 51% | **74%** |
| Prefill-Speed | 94.1 Tok/s | ~117 Tok/s (TTFT 315ms) | – | ~148 Tok/s (TTFT 250ms) |
| Größe | 17.7 GB | 30.9 GB | 25 GB | **18 GB** |
| Modell-Ladezeit (kalt) | 11.3s | ~370s | ~360s | ~222s |

**Selbst quantisiert** aus den offiziellen BF16-Original-Gewichten
(`Qwen/Qwen3.8-27B`), da es keine offizielle/vLLM-native 4-bit-Version gab –
komplette Anleitung inkl. aller Stolpersteine:
[Eigene AWQ-Quantisierung erstellen](anleitung-eigene-awq-quantisierung.md).

**Ollama war anfangs (FP8) ca. 2.5x schneller beim Decode** – primär durch die
Quantisierung erklärbar (siehe Einordnung unten). Mit eigener AWQ-INT4-Quantisierung
(v2, inkl. der 48 Mamba/GatedDeltaNet-Layer) schließt sich die Lücke auf **74% von
Ollamas Tempo** bei nahezu identischer Dateigröße.

## Was sind Mamba-Layer / Gated Delta Net?

Qwen3.8-27B ist eine **Hybrid-Architektur**: von den 64 Layern sind nur 16 "normale"
Attention-Layer, die anderen 48 sind ein anderer Mechanismus namens **Gated Delta
Net** – eine Variante von **Mamba** (auch "State Space Model" genannt). Das ist
relevant für diesen Benchmark, weil genau diese 48 Layer die Ursache dafür waren,
dass die Quantisierung mehrere Anläufe brauchte (siehe
[Anleitung](anleitung-eigene-awq-quantisierung.md#stolpersteine-4-anläufe-waren-nötig)).

### Normale Attention (der "klassische" Transformer-Mechanismus)

Bei jedem neuen Token vergleicht Attention diesen Token mit **allen bisherigen**
Tokens im Kontext, um zu entscheiden, welche davon gerade wichtig sind (daher der
Name "Aufmerksamkeit"). Das ist mächtig, wird aber mit wachsendem Kontext immer
teurer – der Aufwand steigt quadratisch mit der Kontextlänge, und der KV-Cache
(der Speicher, der sich alle bisherigen Tokens "merkt") wächst linear mit.

### Mamba / State Space Models

Mamba macht das anders: Statt sich alle bisherigen Tokens einzeln zu merken und
bei jedem neuen Token nochmal alle zu vergleichen, führt es einen **einzigen,
konstant großen "Zustand"** (State) mit, der bei jedem neuen Token aktualisiert
wird – ähnlich wie ein laufender Zusammenfassungs-Wert, der sich Schritt für
Schritt verändert, statt eine komplette Mitschrift aller bisherigen Tokens zu
führen. Dadurch bleibt der Rechenaufwand pro Token **konstant**, egal wie lang der
Kontext schon ist, und der Speicherbedarf explodiert nicht bei sehr langem Kontext
(genau deshalb kommen Modelle wie dieses hier auf so riesige Kontextlängen wie
262144 oder sogar 1048576 Tokens, siehe
[Erklärung Sweet Spot](erklaerung-quantisierung-und-tokens-pro-sekunde.md)).

**Gated Delta Net** ist Qwens konkrete Umsetzung dieser Idee, mit zusätzlichen
"Gates" (Toren) – kleinen gelernten Schaltern, die steuern, wie stark neue
Informationen den Zustand überschreiben vs. wie viel vom bisherigen Zustand
erhalten bleibt (ähnlich dem Prinzip hinter LSTMs, falls das ein Begriff ist).

### Warum hybrid (16 Attention + 48 Mamba)?

Beide Mechanismen haben Stärken und Schwächen: Mamba ist günstig und
langkontext-tauglich, aber schwächer darin, sehr präzise auf einen einzelnen
weit zurückliegenden Token zurückzugreifen. Normale Attention kann das sehr gut,
ist aber teuer. Die Mischung nutzt Mamba für den Großteil der Layer (günstig,
viel Kontext) und behält an wenigen Stellen echte Attention (für die Aufgaben,
bei denen präziser Rückgriff auf Details wichtig ist).

### Warum das die Quantisierung erschwert hat

Werkzeuge zur Modell-Quantisierung (wie `llm-compressor`) sind in erster Linie auf
den "klassischen" Attention-Mechanismus zugeschnitten – dessen interne Bausteine
(Q/K/V-Projektionen) sind einfach strukturiert. Gated Delta Net hat dagegen eine
komplexere interne Struktur (mehrere Projektionen, Gates, eigene
Aktualisierungslogik), und die verwendete Quantisierungs-Bibliothek konnte deren
Aufruf-Schema (`forward()`-Signatur) nicht automatisch korrekt behandeln – daher
der Crash beim ersten Anlauf und die Notwendigkeit, diese Layer mit einer
einfacheren Methode (Round-to-Nearest statt AWQ) separat zu quantisieren.

## Testaufbau

- Gleicher Prompt: *"Erklaere in ca. 300 Woertern, wie Photosynthese auf molekularer
  Ebene funktioniert."*
- `think: false` (Ollama) / `chat_template_kwargs.enable_thinking: false` (vLLM) –
  Denk-Tokens fließen nicht in die Messung ein.
- `stream: false`, 350 Ausgabe-Tokens angefordert.
- **Solo-Messung**: Ollama und vLLM liefen nacheinander, nie gleichzeitig (Portkonflikt
  auf 11434 + keine Ressourcenkonkurrenz für ein sauberes Ergebnis). vLLM lief dafür
  kurz gestoppt, während Ollama getestet wurde, danach umgekehrt.
- Ollama-Werte direkt aus der `/api/chat`-Antwort (`eval_count`/`eval_duration` =
  reine Decode-Phase, `prompt_eval_count`/`prompt_eval_duration` = Prefill).
- vLLM-Werte aus der eigenen Prometheus-Metrik (`/metrics`,
  `vllm:request_time_per_output_token_seconds` bzw. `vllm:time_to_first_token_seconds`)
  – dieselbe Semantik wie Ollamas Decode-/Prefill-Trennung, gegengecheckt gegen die
  Wanduhrzeit abzüglich Ladezeit (350 Tok. / 57.58s ≈ 6.08 Tok/s, deckt sich mit der
  Metrik von 6.10 Tok/s).

## Einordnung

- **Decode ist speicherbandbreiten-limitiert.** Auf der GB10 (Unified Memory, geteilte
  Bandbreite zwischen CPU/GPU) muss pro erzeugtem Token das gesamte Modellgewicht
  einmal durch den Speicher – bei FP8 (8-bit) doppelt so viele Bytes wie bei Q4_K_M
  (4-bit). Der gemessene Faktor (~2.5x) deckt sich fast exakt mit diesem
  Bandbreiten-Unterschied. Das ist also größtenteils ein **Quantisierungs-Effekt**,
  keine grundsätzliche "vLLM ist langsamer als Ollama"-Aussage.
- **Prefill ist bei vLLM tendenziell schneller** – TTFT aus vLLMs eigener, vom
  Streaming-Modus unabhängiger Metrik.
- **Kaltstart ist bei Ollama drastisch schneller** (11s vs. ~370s): llama.cpp (Ollamas
  Unterbau) mmapt die GGUF-Datei direkt und legt los. vLLM macht bei jedem
  Prozessstart `torch.compile` + CUDA-Graph-Capture neu – genau der Grund, warum
  dieses Projekt einen "Hot Pool" hat (mehrere Modelle warmhalten, siehe
  [Anleitung.md](../Anleitung.md)): einmal warm, ist der Wechsel zwischen Modellen
  instant statt eines erneuten Kaltstarts.
- **Was dieser Test nicht zeigt:** vLLMs eigentliche Stärke ist Durchsatz bei
  **mehreren gleichzeitigen Requests** (Continuous Batching, PagedAttention). Ollama
  skaliert bei paralleler Last typischerweise deutlich schlechter. Bei genau einem
  Request seriell ist das hier nicht sichtbar.

## Fazit

Der ursprüngliche Vergleich war effektiv "Ollama+Q4" gegen "vLLM+FP8", nicht die
Engines pur gegeneinander – geklärt durch eine eigene AWQ-INT4-Quantisierung
(gleiche Bit-Breite wie Ollama, siehe [Anleitung](anleitung-eigene-awq-quantisierung.md)):

- **v1** (nur `full_attention`- und MLP-Layer quantisiert, die Mamba-Layer blieben
  BF16 wegen eines Kompatibilitäts-Bugs in `llm-compressor`): 7.7 Tok/s, 51% von Ollama.
- **v2** (auch die 48 Mamba/GatedDeltaNet-Layer quantisiert, per Round-to-Nearest
  statt AWQ-Smoothing): 11.2 Tok/s, **74% von Ollama**, bei fast identischer Größe
  (18GB vs. 17.7GB).

Die verbleibende Lücke zu Ollama liegt vermutlich daran, dass Round-to-Nearest für
die Mamba-Layer weniger präzise ist als AWQs Smoothing, und/oder dass llama.cpps
Q4_K_M-Kernel für diese Hardware feiner optimiert sind als vLLMs generischer
AWQ-Pfad. Für die volle vLLM-Integration (Hot Pool, OpenAI-API, Tool-Calling, MCP,
RAG) ist 74% von Ollamas Tempo trotzdem ein guter Kompromiss.

## Nachtrag (2026-08-25): weitere Optimierungen nach der Quantisierung

Drei zusätzliche, rein config-seitige Hebel (kein erneutes Quantisieren nötig)
live getestet, jeweils vorher/nachher über vLLMs eigene Prometheus-Metrik
gemessen (`vllm:request_time_per_output_token_seconds`, Delta über genau eine
Anfrage, damit andere Requests das Ergebnis nicht verfälschen):

| Optimierung | Ergebnis | Übernommen? |
|---|---|---|
| `cudagraph_capture_sizes=1,2,4,8,16` (siehe `ModelConfig.cudagraph_capture_sizes`) | Kaltstart ~17s schneller (Graph-Capture 2s statt ~23s), Decode-Speed unverändert (10.83 vs. 10.97 Tok/s) | ✅ Ja - reiner Gewinn, kein Trade-off |
| FP8-KV-Cache (`--kv-cache-dtype fp8`) | Keine messbare Beschleunigung bei kurzem Kontext (10.76 Tok/s) + zusätzlicher Kaltstart-Overhead (~46s Extra-Warmup für FP8-Kalibrierung) | ❌ Nein - bringt bei kurzem Kontext nichts, nur Kosten. Könnte bei sehr langem Kontext/hoher Parallelität anders aussehen (KV-Cache macht dort einen größeren Anteil der Speicherbandbreite aus) - hier nicht getestet |
| n-gram-Spekulation (`--speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}'`) | Freie Kreativantwort: 10.48 Tok/s (~5% langsamer). Wiederholungslastige Aufgabe (Text/Code wörtlich kopieren): **58 Tok/s (5× schneller als Baseline)**, Ausgabe nachweislich exakt identisch zur Vorlage | ✅ Ja - siehe Einordnung unten |

**Warum n-gram-Spekulation trotz der 5%-Verlangsamung bei freier Prosa
übernommen wurde:** Anders als Quantisierung ist spekulatives Decodieren
**mathematisch verlustfrei** - ein kleiner, kostenloser "Rate-Mechanismus"
schlägt mehrere Tokens auf einmal vor, das eigentliche Modell verifiziert sie
nur noch (viel billiger als sie einzeln zu generieren); ein falscher Rat
kostet nur etwas Zeit, nie Qualität. Der Rate-Mechanismus (n-gram/Prompt-
Lookup) sucht nach Wiederholungen zwischen Prompt und bisheriger Antwort - bei
freier Kreativantwort (wenig Wiederholung) trifft er selten und kostet daher
minimal Zeit, bei Aufgaben mit viel wörtlicher Wiederholung (Code-Bearbeitung,
Refactoring, Zitieren, strukturierte Ausgaben - genau das Profil dieses
Setups mit Tool-Calling/RAG/VS-Code-Nutzung) trifft er oft und bringt ein
Vielfaches an Tempo. Nettoabwägung: kleiner Verlust im selteneren Fall,
großer Gewinn im für dieses Projekt häufigeren Fall.
