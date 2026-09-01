# DataTables (vendored)

Lokal eingebunden statt über CDN, damit die Dashboards auch ohne
Internetzugang funktionieren (gleiche Philosophie wie der Rest des
LLM-Hubs: kein externer Build-Schritt, keine Laufzeit-Abhängigkeit von
Drittanbieter-Hosts).

- `dataTables.min.js` – DataTables 3.0.2 Core (jQuery-frei, `datatables.net` auf npm)
- `dataTables.dataTables.min.js` / `.css` – Standard-Styling-Integration (`datatables.net-dt`)
- `dataTables.inputPaging.min.js` / `.css` – offizielles Feature-Plugin für eine
  Seitenzahl-Eingabe in der Pagination (`datatables.net-feature-inputpaging`,
  `layout: { bottomEnd: "inputPaging" }`)

Lizenz: MIT (SpryMedia Ltd / datatables.net), siehe `LICENSE.txt`.

Update: neue Version von https://www.npmjs.com/package/datatables.net (+ `-dt`
und `-feature-inputpaging`) als Tarball laden, `dist/*.min.{js,css}` hier
ersetzen.
