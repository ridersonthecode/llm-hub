# Projekt-Hinweise für Copilot Chat (Agent-Modus)

## Lessons Learned (MCP)

Dieses Projekt betreibt einen eigenen MCP-Server (vLLM-Manager, siehe
`vllm_manager/mcp_tools.py`, verbunden über `.vscode/mcp.json`). Zwei seiner
Tools dienen dem projektweiten Gedächtnis für gelöste Probleme:

- **`remember_lesson(mistake, correction, context?, tags?)`**: aufrufen,
  sobald ein Fehler (z.B. ein fehlgeschlagener Terminal-/PowerShell-Befehl,
  ein Build-/Konfigurationsfehler) erfolgreich behoben wurde, oder wenn der
  Nutzer eine vorherige Annahme/Antwort explizit korrigiert hat. Nicht bei
  jeder Kleinigkeit aufrufen - nur wenn die Lehre über den aktuellen Moment
  hinaus verallgemeinerbar ist.
- **`search_lessons(query, top_k?)`**: vor dem Angehen eines Fehlers
  aufrufen, der bekannt vorkommt - prüft, ob dafür schon eine dokumentierte
  Lösung existiert, bevor erneut Zeit in dieselbe Fehlersuche investiert
  wird.

Beide Tools schreiben/lesen dieselbe, in `config.json` unter
`rag.lessons_learned_collection` konfigurierbare RAG-Collection (Standard:
`lessons_learned`). Als Sicherheitsnetz wird dieselbe Collection zusätzlich
automatisch bei jeder Chat-Anfrage durchsucht und als Kontext eingespeist,
falls `rag.lessons_learned_auto_inject` aktiv ist (siehe
`vllm_manager/rag.py`, `apply_auto_rag`) - `search_lessons` proaktiv
aufzurufen bleibt trotzdem sinnvoll, insbesondere bevor ein bekannt
wirkender Fehler angegangen wird.
