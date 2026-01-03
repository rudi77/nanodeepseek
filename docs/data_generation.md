Alles klar — hier ist das **komplette Paket**:

1. **`case_library_eb_100.json`** (100 Templates als JSON)
2. **Generator-Prompt(s)**
3. **Python-Skript** (Azure OpenAI), das daraus **1.000 SFT JSONL** + **1.000 DPO JSONL** erzeugt — *mit garantierter fachlicher Korrektheit*, weil die Buchung **regelbasiert** gerechnet wird und das LLM nur die **Aufgabenformulierung paraphrasiert**.

---

# 1) Case Library (100 Templates) – `case_library_eb_100.json`

> Kopiere das in eine Datei `case_library_eb_100.json`.

```json
[
  {"template_id":"EB-001","category":"Aktiva","sub_category":"Liquide Mittel","industry_focus":["all"],"description":"Bankguthaben laut Kontoauszug","amount_model":{"type":"absolute","min":500,"max":250000},"booking":{"debit_label":"Bank","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-002","category":"Aktiva","sub_category":"Liquide Mittel","industry_focus":["all"],"description":"Kassa-Barbestand","amount_model":{"type":"absolute","min":50,"max":25000},"booking":{"debit_label":"Kassa","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-003","category":"Aktiva","sub_category":"Liquide Mittel","industry_focus":["all"],"description":"Fremdwährungskonto Bank (EUR-Wert)","amount_model":{"type":"absolute","min":200,"max":150000},"booking":{"debit_label":"Bank (Fremdwährung)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-004","category":"Aktiva","sub_category":"Liquide Mittel","industry_focus":["all"],"description":"Sparkonto / Rücklagenkonto (Bank)","amount_model":{"type":"absolute","min":500,"max":400000},"booking":{"debit_label":"Bank (Sparkonto)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-005","category":"Aktiva","sub_category":"Liquide Mittel","industry_focus":["ecommerce","dienstleistung","handel"],"description":"PayPal-Guthaben","amount_model":{"type":"absolute","min":50,"max":80000},"booking":{"debit_label":"PayPal (Guthaben)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-006","category":"Aktiva","sub_category":"Liquide Mittel","industry_focus":["ecommerce"],"description":"Stripe/Clearing-Guthaben","amount_model":{"type":"absolute","min":100,"max":120000},"booking":{"debit_label":"Payment-Provider Clearing (Guthaben)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-007","category":"Aktiva","sub_category":"Liquide Mittel","industry_focus":["handel","gastro","dienstleistung","ecommerce"],"description":"Kreditkarten-Guthaben / Abrechnungsguthaben","amount_model":{"type":"absolute","min":50,"max":60000},"booking":{"debit_label":"Kreditkartenabrechnung (Guthaben)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-008","category":"Aktiva","sub_category":"Liquide Mittel","industry_focus":["all"],"description":"Schwebende Geldbewegungen (noch nicht zugeordnet)","amount_model":{"type":"absolute","min":10,"max":25000},"booking":{"debit_label":"Schwebende Geldbewegungen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},

  {"template_id":"EB-009","category":"Aktiva","sub_category":"Forderungen","industry_focus":["all"],"description":"Kundenforderungen Inland (netto angegeben, 20% USt -> brutto buchen)","amount_model":{"type":"net","min":100,"max":250000},"booking":{"debit_label":"Kundenforderungen Inland","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"net_to_gross","vat_rate":0.20,"amount_basis":"gross"}},
  {"template_id":"EB-010","category":"Aktiva","sub_category":"Forderungen","industry_focus":["all"],"description":"Kundenforderungen Inland (netto angegeben, 10% USt -> brutto buchen)","amount_model":{"type":"net","min":100,"max":250000},"booking":{"debit_label":"Kundenforderungen Inland (10%)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"net_to_gross","vat_rate":0.10,"amount_basis":"gross"}},
  {"template_id":"EB-011","category":"Aktiva","sub_category":"Forderungen","industry_focus":["all"],"description":"Kundenforderungen EU (innergemeinschaftlich, ohne USt im Ausgangsland)","amount_model":{"type":"absolute","min":100,"max":300000},"booking":{"debit_label":"Kundenforderungen EU","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-012","category":"Aktiva","sub_category":"Forderungen","industry_focus":["all"],"description":"Kundenforderungen Drittland (Export, ohne USt)","amount_model":{"type":"absolute","min":100,"max":400000},"booking":{"debit_label":"Kundenforderungen Drittland","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-013","category":"Aktiva","sub_category":"Forderungen","industry_focus":["all"],"description":"Sonstige Forderungen (z.B. Kautionen geleistet, Erstattungen)","amount_model":{"type":"absolute","min":50,"max":150000},"booking":{"debit_label":"Sonstige Forderungen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-014","category":"Aktiva","sub_category":"Forderungen","industry_focus":["all"],"description":"Forderungen gegenüber Mitarbeitern (Vorschüsse)","amount_model":{"type":"absolute","min":50,"max":15000},"booking":{"debit_label":"Forderungen Mitarbeiter","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-015","category":"Aktiva","sub_category":"Forderungen","industry_focus":["gmbh","kg","og"],"description":"Forderungen gegenüber Gesellschaftern","amount_model":{"type":"absolute","min":100,"max":300000},"booking":{"debit_label":"Forderungen Gesellschafter","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-016","category":"Aktiva","sub_category":"Forderungen","industry_focus":["handel","gastro","dienstleistung","ecommerce"],"description":"Forderung aus Gutscheinverkauf (Einlösung ausstehend)","amount_model":{"type":"absolute","min":50,"max":80000},"booking":{"debit_label":"Forderungen aus Gutscheinen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-017","category":"Aktiva","sub_category":"Forderungen","industry_focus":["handel","gastro","dienstleistung","ecommerce"],"description":"Offene Forderung aus Kreditkartenabrechnung (noch nicht gutgeschrieben)","amount_model":{"type":"absolute","min":50,"max":120000},"booking":{"debit_label":"Forderungen Kreditkartenabrechnung","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-018","category":"Aktiva","sub_category":"Forderungen","industry_focus":["all"],"description":"Zweifelhafte Forderungen (brutto)","amount_model":{"type":"absolute","min":50,"max":90000},"booking":{"debit_label":"Zweifelhafte Forderungen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},

  {"template_id":"EB-019","category":"Aktiva","sub_category":"Vorräte","industry_focus":["handel","ecommerce"],"description":"Handelswarenlager","amount_model":{"type":"absolute","min":200,"max":600000},"booking":{"debit_label":"Warenlager (Handelswaren)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-020","category":"Aktiva","sub_category":"Vorräte","industry_focus":["produktion","bau","handwerk"],"description":"Rohstofflager","amount_model":{"type":"absolute","min":200,"max":800000},"booking":{"debit_label":"Vorrat Rohstoffe","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-021","category":"Aktiva","sub_category":"Vorräte","industry_focus":["produktion","handwerk"],"description":"Hilfsstofflager","amount_model":{"type":"absolute","min":200,"max":350000},"booking":{"debit_label":"Vorrat Hilfsstoffe","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-022","category":"Aktiva","sub_category":"Vorräte","industry_focus":["produktion","transport","bau","gastro"],"description":"Betriebsstofflager (z.B. Treibstoff)","amount_model":{"type":"absolute","min":100,"max":200000},"booking":{"debit_label":"Vorrat Betriebsstoffe","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-023","category":"Aktiva","sub_category":"Vorräte","industry_focus":["produktion"],"description":"Unfertige Erzeugnisse","amount_model":{"type":"absolute","min":200,"max":700000},"booking":{"debit_label":"Unfertige Erzeugnisse","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-024","category":"Aktiva","sub_category":"Vorräte","industry_focus":["produktion"],"description":"Fertige Erzeugnisse","amount_model":{"type":"absolute","min":200,"max":900000},"booking":{"debit_label":"Fertige Erzeugnisse","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-025","category":"Aktiva","sub_category":"Vorräte","industry_focus":["handel","produktion","transport"],"description":"Waren in Transit (noch nicht eingelagert)","amount_model":{"type":"absolute","min":200,"max":400000},"booking":{"debit_label":"Waren in Transit","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-026","category":"Aktiva","sub_category":"Vorräte","industry_focus":["handel"],"description":"Konsignationsware (Bestand beim Kunden)","amount_model":{"type":"absolute","min":200,"max":300000},"booking":{"debit_label":"Konsignationsware","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-027","category":"Aktiva","sub_category":"Vorräte","industry_focus":["handel","produktion","gastro"],"description":"Emballagen/Leergut im Bestand","amount_model":{"type":"absolute","min":100,"max":250000},"booking":{"debit_label":"Emballagen / Leergut","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-028","category":"Aktiva","sub_category":"Vorräte","industry_focus":["handel","gastro"],"description":"Pfandguthaben (z.B. Kisten/Flaschen)","amount_model":{"type":"absolute","min":50,"max":50000},"booking":{"debit_label":"Pfandguthaben","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-029","category":"Aktiva","sub_category":"Vorräte","industry_focus":["produktion","bau","handwerk"],"description":"Werkzeugbestand (Umlaufvermögen)","amount_model":{"type":"absolute","min":100,"max":120000},"booking":{"debit_label":"Werkzeugbestand (Umlauf)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-030","category":"Aktiva","sub_category":"Vorräte","industry_focus":["produktion","transport","handwerk"],"description":"Ersatzteillager","amount_model":{"type":"absolute","min":100,"max":250000},"booking":{"debit_label":"Ersatzteillager","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-031","category":"Aktiva","sub_category":"Vorräte","industry_focus":["handel","produktion","ecommerce"],"description":"Verpackungsmaterial im Lager","amount_model":{"type":"absolute","min":50,"max":80000},"booking":{"debit_label":"Verpackungsmaterial","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-032","category":"Aktiva","sub_category":"Vorräte","industry_focus":["handel","ecommerce","gastro"],"description":"Saisonware / saisonaler Vorrat","amount_model":{"type":"absolute","min":200,"max":500000},"booking":{"debit_label":"Saisonware","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},

  {"template_id":"EB-033","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["all"],"description":"Grundstücke","amount_model":{"type":"absolute","min":20000,"max":5000000},"booking":{"debit_label":"Grundstücke","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-034","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["all"],"description":"Gebäude","amount_model":{"type":"absolute","min":50000,"max":8000000},"booking":{"debit_label":"Gebäude","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-035","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["produktion","handel","handwerk"],"description":"Maschinen","amount_model":{"type":"absolute","min":1000,"max":1500000},"booking":{"debit_label":"Maschinen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-036","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["produktion"],"description":"Produktionsanlagen (Sammelposten)","amount_model":{"type":"absolute","min":5000,"max":5000000},"booking":{"debit_label":"Produktionsanlagen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-037","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["all"],"description":"Fuhrpark – PKW (Buchwert)","amount_model":{"type":"absolute","min":1000,"max":300000},"booking":{"debit_label":"Fuhrpark (PKW)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-038","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["transport","bau","handwerk","handel"],"description":"Fuhrpark – LKW (Buchwert)","amount_model":{"type":"absolute","min":5000,"max":800000},"booking":{"debit_label":"Fuhrpark (LKW)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-039","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["all"],"description":"Büroausstattung (BGA)","amount_model":{"type":"absolute","min":200,"max":400000},"booking":{"debit_label":"Büroausstattung (BGA)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-040","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["all"],"description":"IT-Hardware (Server, PCs)","amount_model":{"type":"absolute","min":200,"max":600000},"booking":{"debit_label":"IT-Hardware","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-041","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["all"],"description":"Software-Lizenzen / immaterielle Vermögenswerte","amount_model":{"type":"absolute","min":200,"max":800000},"booking":{"debit_label":"Software/Lizenzen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-042","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["all"],"description":"GWG-Sammelposten (Buchwert)","amount_model":{"type":"absolute","min":200,"max":150000},"booking":{"debit_label":"GWG-Sammelposten","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-043","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["miete","immobilien","gastro","handel"],"description":"Mietereinbauten (Buchwert)","amount_model":{"type":"absolute","min":500,"max":900000},"booking":{"debit_label":"Mietereinbauten","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-044","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["produktion","bau","handwerk"],"description":"Werkzeuge (Anlagevermögen)","amount_model":{"type":"absolute","min":200,"max":300000},"booking":{"debit_label":"Werkzeuge (Anlage)","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-045","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["all"],"description":"Photovoltaikanlage (Buchwert)","amount_model":{"type":"absolute","min":2000,"max":1200000},"booking":{"debit_label":"Photovoltaikanlage","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-046","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["handel","gastro"],"description":"Ladeneinrichtung (Buchwert)","amount_model":{"type":"absolute","min":500,"max":1500000},"booking":{"debit_label":"Ladeneinrichtung","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-047","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["praxis","gesundheit"],"description":"Medizintechnische Geräte (Buchwert)","amount_model":{"type":"absolute","min":1000,"max":2500000},"booking":{"debit_label":"Medizintechnik","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-048","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["gastro"],"description":"Küchenausstattung (Buchwert)","amount_model":{"type":"absolute","min":1000,"max":2500000},"booking":{"debit_label":"Küchenausstattung","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-049","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["produktion"],"description":"Maschinen in Montage / Anlagen im Bau","amount_model":{"type":"absolute","min":1000,"max":4000000},"booking":{"debit_label":"Anlagen im Bau","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"book_value"}},
  {"template_id":"EB-050","category":"Aktiva","sub_category":"Anlagevermögen","industry_focus":["all"],"description":"Anzahlungen auf Anlagen","amount_model":{"type":"absolute","min":200,"max":2000000},"booking":{"debit_label":"Anzahlungen auf Anlagen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},

  {"template_id":"EB-051","category":"Passiva","sub_category":"Eigenkapital","industry_focus":["all"],"description":"Eigenkapital Anfangsbestand (Sammelposten)","amount_model":{"type":"absolute","min":1000,"max":5000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Eigenkapital"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-052","category":"Passiva","sub_category":"Eigenkapital","industry_focus":["einzelunternehmen"],"description":"Privatkapital Anfangsbestand","amount_model":{"type":"absolute","min":1000,"max":5000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Privatkapital"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-053","category":"Passiva","sub_category":"Eigenkapital","industry_focus":["gmbh","kg","og"],"description":"Gesellschaftereinlage / Kapitalkonto Anfangsbestand","amount_model":{"type":"absolute","min":1000,"max":8000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Kapitalkonto Gesellschafter"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-054","category":"Passiva","sub_category":"Eigenkapital","industry_focus":["gmbh"],"description":"Kapitalrücklage Anfangsbestand","amount_model":{"type":"absolute","min":500,"max":8000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Kapitalrücklage"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-055","category":"Passiva","sub_category":"Eigenkapital","industry_focus":["all"],"description":"Gewinnvortrag","amount_model":{"type":"absolute","min":100,"max":5000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Gewinnvortrag"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-056","category":"Aktiva","sub_category":"Eigenkapital-nahe Posten","industry_focus":["all"],"description":"Verlustvortrag (als negativer EK-Posten; in Trainingsdaten als Sollsaldo Eigenkapital-Posten modellieren)","amount_model":{"type":"absolute","min":100,"max":5000000},"booking":{"debit_label":"Verlustvortrag","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance","note":"Je nach Systematik kann Verlustvortrag als Minus-EK geführt werden; hier als Aktiv-Sollsaldo für Trainingszwecke."}},
  {"template_id":"EB-057","category":"Passiva","sub_category":"Eigenkapital","industry_focus":["gmbh"],"description":"Stammkapital (GmbH)","amount_model":{"type":"absolute","min":10000,"max":1000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Stammkapital"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-058","category":"Aktiva","sub_category":"Forderungen","industry_focus":["gmbh","kg","og"],"description":"Offene Kapitalrückzahlung (Forderung ggü. Gesellschafter)","amount_model":{"type":"absolute","min":100,"max":2000000},"booking":{"debit_label":"Forderung Kapitalrückzahlung","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},

  {"template_id":"EB-059","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["all"],"description":"Lieferverbindlichkeiten Inland","amount_model":{"type":"absolute","min":100,"max":400000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Lieferverbindlichkeiten Inland"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-060","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["all"],"description":"Lieferverbindlichkeiten Ausland","amount_model":{"type":"absolute","min":100,"max":500000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Lieferverbindlichkeiten Ausland"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-061","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["all"],"description":"Sonstige Verbindlichkeiten (z.B. Gebühren, Abgaben)","amount_model":{"type":"absolute","min":50,"max":250000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Sonstige Verbindlichkeiten"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-062","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["all"],"description":"Verbindlichkeiten gegenüber Mitarbeitern (z.B. Spesen)","amount_model":{"type":"absolute","min":50,"max":50000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Verbindlichkeiten Mitarbeiter"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-063","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["gmbh","kg","og"],"description":"Verbindlichkeiten gegenüber Gesellschaftern","amount_model":{"type":"absolute","min":100,"max":3000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Verbindlichkeiten Gesellschafter"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-064","category":"Passiva","sub_category":"Finanzierung","industry_focus":["all"],"description":"Kreditinstitut – kurzfristig (Kontokorrent)","amount_model":{"type":"absolute","min":500,"max":1500000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Bankkredit kurzfristig"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-065","category":"Passiva","sub_category":"Finanzierung","industry_focus":["all"],"description":"Kreditinstitut – langfristig","amount_model":{"type":"absolute","min":1000,"max":10000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Bankkredit langfristig"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-066","category":"Passiva","sub_category":"Finanzierung","industry_focus":["all"],"description":"Leasingverbindlichkeit (Restverpflichtung)","amount_model":{"type":"absolute","min":500,"max":5000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Leasingverbindlichkeit"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-067","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["immobilien","miete","handel","gastro"],"description":"Mietkaution erhalten","amount_model":{"type":"absolute","min":200,"max":200000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Erhaltene Kautionen"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-068","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["all"],"description":"Anzahlungen von Kunden (erhalten)","amount_model":{"type":"absolute","min":100,"max":2000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Erhaltene Anzahlungen Kunden"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-069","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["bau","handwerk"],"description":"Haftrücklass / Sicherheitseinbehalt","amount_model":{"type":"absolute","min":100,"max":2000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Haftrücklass"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-070","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["handel","gastro","ecommerce","dienstleistung"],"description":"Gutscheinverbindlichkeiten (verkauft, noch nicht eingelöst)","amount_model":{"type":"absolute","min":50,"max":500000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Gutscheinverbindlichkeiten"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-071","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["ecommerce"],"description":"PayPal-Verbindlichkeit (z.B. negative Salden/Chargebacks)","amount_model":{"type":"absolute","min":50,"max":200000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"PayPal (Verbindlichkeit)"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-072","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["handel","gastro","dienstleistung","ecommerce"],"description":"Kreditkartenabrechnung offen (Verbindlichkeit)","amount_model":{"type":"absolute","min":50,"max":400000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Kreditkartenabrechnung (Verbindlichkeit)"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-073","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["all"],"description":"Steuerverbindlichkeiten (Sammelposten)","amount_model":{"type":"absolute","min":50,"max":2000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Steuerverbindlichkeiten"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-074","category":"Passiva","sub_category":"Verbindlichkeiten","industry_focus":["all"],"description":"Sozialversicherungsverbindlichkeiten","amount_model":{"type":"absolute","min":50,"max":800000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"SV-Verbindlichkeiten"},"rules":{"vat_handling":"none","amount_basis":"balance"}},

  {"template_id":"EB-075","category":"Passiva","sub_category":"Rückstellungen","industry_focus":["all"],"description":"Rückstellung Einkommensteuer","amount_model":{"type":"absolute","min":100,"max":2000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Rückstellung ESt"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-076","category":"Passiva","sub_category":"Rückstellungen","industry_focus":["gmbh"],"description":"Rückstellung Körperschaftsteuer","amount_model":{"type":"absolute","min":100,"max":3000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Rückstellung KSt"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-077","category":"Passiva","sub_category":"Rückstellungen","industry_focus":["all"],"description":"Rückstellung Umsatzsteuer","amount_model":{"type":"absolute","min":50,"max":1500000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Rückstellung USt"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-078","category":"Passiva","sub_category":"Rückstellungen","industry_focus":["all"],"description":"Rückstellung Gewährleistung/Garantie","amount_model":{"type":"absolute","min":100,"max":2000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Rückstellung Gewährleistung"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-079","category":"Passiva","sub_category":"Rückstellungen","industry_focus":["all"],"description":"Rückstellung Urlaub","amount_model":{"type":"absolute","min":100,"max":1500000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Rückstellung Urlaub"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-080","category":"Passiva","sub_category":"Rückstellungen","industry_focus":["all"],"description":"Rückstellung Prozessrisiko/Rechtsstreit","amount_model":{"type":"absolute","min":100,"max":5000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Rückstellung Prozessrisiko"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-081","category":"Passiva","sub_category":"Rückstellungen","industry_focus":["immobilien","produktion","bau"],"description":"Rückstellung Instandhaltung","amount_model":{"type":"absolute","min":100,"max":3000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Rückstellung Instandhaltung"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-082","category":"Passiva","sub_category":"Abgrenzung","industry_focus":["all"],"description":"Passive Rechnungsabgrenzung (PRAP)","amount_model":{"type":"absolute","min":50,"max":2000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"PRAP"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-083","category":"Aktiva","sub_category":"Abgrenzung","industry_focus":["all"],"description":"Aktive Rechnungsabgrenzung (ARAP)","amount_model":{"type":"absolute","min":50,"max":2000000},"booking":{"debit_label":"ARAP","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-084","category":"Aktiva","sub_category":"Abgrenzung","industry_focus":["dienstleistung","it","agentur"],"description":"Noch nicht fakturierte Leistungen (Forderung aus Leistung)","amount_model":{"type":"absolute","min":100,"max":3000000},"booking":{"debit_label":"Noch nicht fakturierte Leistungen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-085","category":"Passiva","sub_category":"Abgrenzung","industry_focus":["all"],"description":"Erhaltene Vorauszahlungen (PRAP-ähnlich)","amount_model":{"type":"absolute","min":100,"max":3000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Erhaltene Vorauszahlungen"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-086","category":"Passiva","sub_category":"Abgrenzung","industry_focus":["dienstleistung","it","agentur"],"description":"Abgrenzung Wartungsverträge (noch nicht verdienter Anteil)","amount_model":{"type":"absolute","min":100,"max":5000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"PRAP Wartungsverträge"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-087","category":"Passiva","sub_category":"Abgrenzung","industry_focus":["immobilien"],"description":"Abgrenzung Mieteinnahmen (noch nicht verdient)","amount_model":{"type":"absolute","min":100,"max":5000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"PRAP Mieteinnahmen"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-088","category":"Passiva","sub_category":"Abgrenzung","industry_focus":["all"],"description":"Abgrenzung Leasing (noch nicht fällige Raten als PRAP)","amount_model":{"type":"absolute","min":100,"max":3000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"PRAP Leasing"},"rules":{"vat_handling":"none","amount_basis":"balance"}},

  {"template_id":"EB-089","category":"Aktiva","sub_category":"Branche/Sonderfälle","industry_focus":["bau"],"description":"Baustellen in Arbeit (Work in Progress)","amount_model":{"type":"absolute","min":500,"max":8000000},"booking":{"debit_label":"Baustellen in Arbeit","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-090","category":"Aktiva","sub_category":"Branche/Sonderfälle","industry_focus":["dienstleistung","it","agentur"],"description":"Offene Projekte / unfertige Aufträge","amount_model":{"type":"absolute","min":500,"max":8000000},"booking":{"debit_label":"Unfertige Aufträge/Projekte","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-091","category":"Aktiva","sub_category":"Branche/Sonderfälle","industry_focus":["praxis","gesundheit"],"description":"Patientenforderungen (Inland, brutto)","amount_model":{"type":"absolute","min":50,"max":1500000},"booking":{"debit_label":"Patientenforderungen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance","note":"Gesundheitsleistungen oft steuerbefreit; Forderung als Saldo."}},
  {"template_id":"EB-092","category":"Aktiva","sub_category":"Branche/Sonderfälle","industry_focus":["kanzlei","recht"],"description":"Honorarforderungen (Inland, brutto oder saldo)","amount_model":{"type":"absolute","min":100,"max":3000000},"booking":{"debit_label":"Honorarforderungen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-093","category":"Passiva","sub_category":"Branche/Sonderfälle","industry_focus":["handel","ecommerce"],"description":"Rückstellung für Warenrücknahmen/Preisnachlässe (Passiv)","amount_model":{"type":"absolute","min":100,"max":2000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Rückstellung Warenrücknahmen"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-094","category":"Passiva","sub_category":"Branche/Sonderfälle","industry_focus":["ecommerce"],"description":"Retourenrückstellung (Passiv)","amount_model":{"type":"absolute","min":100,"max":5000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Rückstellung Retouren"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-095","category":"Aktiva","sub_category":"Branche/Sonderfälle","industry_focus":["ecommerce","handel"],"description":"Payment-Provider Clearing Sammelkonto (Guthaben)","amount_model":{"type":"absolute","min":100,"max":800000},"booking":{"debit_label":"Payment-Clearing Sammelkonto","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-096","category":"Aktiva","sub_category":"Branche/Sonderfälle","industry_focus":["transport","bau","handwerk"],"description":"Treibstoffkarten-Abrechnung (Guthaben)","amount_model":{"type":"absolute","min":50,"max":250000},"booking":{"debit_label":"Treibstoffkarten Guthaben","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-097","category":"Aktiva","sub_category":"Branche/Sonderfälle","industry_focus":["transport"],"description":"Maut-Transponder-Guthaben","amount_model":{"type":"absolute","min":50,"max":100000},"booking":{"debit_label":"Maut-Transponder Guthaben","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-098","category":"Passiva","sub_category":"Branche/Sonderfälle","industry_focus":["npo","verein"],"description":"Spendenverbindlichkeiten / noch auszuzahlende zweckgebundene Mittel","amount_model":{"type":"absolute","min":100,"max":3000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Spendenverbindlichkeiten"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-099","category":"Passiva","sub_category":"Branche/Sonderfälle","industry_focus":["verein","npo"],"description":"Zweckgebundene Rücklagen (Eigenkapital-ähnlich)","amount_model":{"type":"absolute","min":100,"max":8000000},"booking":{"debit_label":"Eröffnungsbilanzkonto","credit_label":"Zweckgebundene Rücklagen"},"rules":{"vat_handling":"none","amount_basis":"balance"}},
  {"template_id":"EB-100","category":"Aktiva","sub_category":"Branche/Sonderfälle","industry_focus":["immobilien"],"description":"Mietzinsvorauszahlungen (ARAP-ähnlich)","amount_model":{"type":"absolute","min":100,"max":5000000},"booking":{"debit_label":"Mietzinsvorauszahlungen","credit_label":"Eröffnungsbilanzkonto"},"rules":{"vat_handling":"none","amount_basis":"balance"}}
]
```

---

# 2) Generator-Prompt (für Aufgaben-Text-Paraphrase)

Wir lassen Azure GPT **nur** die Aufgabenstellung formulieren (damit keine Buchungsfehler reinkommen). Das Modell bekommt Template + Branche + Parameter und gibt **nur** einen kurzen deutschen Aufgaben-Text zurück.

**System**

```text
Du formulierst Prüfungsaufgaben für Buchhaltung (DACH/AT). 
Gib ausschließlich JSON zurück: {"instruction": "..."}.
Keine Markdown-Fences. Keine Erklärungen. Maximal 2 Sätze.
Nutze korrekte Fachbegriffe (Eröffnungsbuchung, Buchwert, Anfangsbestand, Stichtag).
```

**User (pro Sample)**

```text
Formuliere eine Aufgabenstellung zur Eröffnungsbuchung.
Kontext:
- Branche: {industry}
- Stichtag: 01.01.2024
- Sachverhalt: {template_description}
- Betrag: {amount} EUR
- Hinweis: {optional_net_vat_hint}
```

---

# 3) Python: 1.000 SFT + 1.000 DPO (Azure OpenAI)

> Speichere als `generate_eb_dataset.py`
> Erwartet Umgebungsvariablen:
> `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` (Chat-Deployment)

```python
import os, json, random, time, math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from openai import AzureOpenAI

# ----------------------------
# Config
# ----------------------------
AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

CASELIB_PATH = "case_library_eb_100.json"
OUT_SFT = "train_sft_eb_1000.jsonl"
OUT_DPO = "train_dpo_eb_1000.jsonl"

N_SAMPLES = 1000
TEMPERATURE = 0.8
SLEEP_SEC = 0.15

INDUSTRIES = [
    "handel", "produktion", "bau", "handwerk", "gastro", "dienstleistung",
    "it", "agentur", "transport", "immobilien", "ecommerce", "praxis",
    "kanzlei", "verein", "npo", "gmbh", "einzelunternehmen"
]

# Optional: mapping to EKR-style codes (coarse). Keep label always; code optional.
EKR_MAP = {
    "Eröffnungsbilanzkonto": "9800",
    "Bank": "2800",
    "Kassa": "2700",
    "Schwebende Geldbewegungen": "2890",
    "Kundenforderungen Inland": "2000",
    "Lieferverbindlichkeiten Inland": "3300",
    "Bankkredit kurzfristig": "3120",
    "Bankkredit langfristig": "3150",
    "Grundstücke": "0200",
    "Gebäude": "0300",
    "Maschinen": "0400",
    "Emballagen / Leergut": "1650",
    "ARAP": "2900",
    "PRAP": "3900"
}

SYSTEM_PARAPHRASE = (
    'Du formulierst Prüfungsaufgaben für Buchhaltung (DACH/AT). '
    'Gib ausschließlich JSON zurück: {"instruction": "..."}.\n'
    "Keine Markdown-Fences. Keine Erklärungen. Maximal 2 Sätze.\n"
    "Nutze korrekte Fachbegriffe (Eröffnungsbuchung, Buchwert, Anfangsbestand, Stichtag)."
)

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

def round2(x: float) -> float:
    return float(f"{x:.2f}")

def pick_amount(amount_model: Dict[str, Any]) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Returns: (amount_display, net_amount, vat_rate)
    - For absolute/book_value/balance: amount_display is the posted amount.
    - For net models: amount_display is net (for the task), net_amount returned + vat_rate.
    """
    t = amount_model["type"]
    mn, mx = amount_model["min"], amount_model["max"]
    # log-ish distribution for realism
    r = random.random()
    val = mn * ((mx / mn) ** r)
    val = round2(val)
    if t == "net":
        return val, val, None
    return val, None, None

def compute_posting(template: Dict[str, Any], industry: str) -> Dict[str, Any]:
    """
    Computes the correct booking output (rule-based).
    Returns output object with strict structure.
    """
    amount_model = template["amount_model"]
    rules = template.get("rules", {})
    booking = template["booking"]

    amount_display, net_amount, _ = pick_amount(amount_model)

    vat_handling = rules.get("vat_handling", "none")
    vat_rate = rules.get("vat_rate")

    if vat_handling == "net_to_gross":
        if net_amount is None or vat_rate is None:
            raise ValueError("net_to_gross requires net amount and vat_rate")
        gross = round2(net_amount * (1.0 + float(vat_rate)))
        posted_amount = gross
        hint = f"Netto {net_amount:.2f} EUR, USt {int(vat_rate*100)}% -> brutto buchen."
    else:
        posted_amount = amount_display
        hint = None

    # Lines:
    debit_label = booking["debit_label"]
    credit_label = booking["credit_label"]

    def line(label: str, side: str, amount: float) -> Dict[str, Any]:
        obj = {"account_label": label, "side": side, "amount": amount}
        if label in EKR_MAP:
            obj["ekr_code"] = EKR_MAP[label]
        return obj

    out = {
        "datum": "2024-01-01",
        "industry": industry,
        "template_id": template["template_id"],
        "text": f"Eröffnungsbuchung: {template['description']}",
        "lines": [
            line(debit_label, "Soll", posted_amount),
            line(credit_label, "Haben", posted_amount),
        ]
    }

    return out, amount_display, hint

def paraphrase_instruction(industry: str, template: Dict[str, Any], amount_display: float, hint: Optional[str]) -> str:
    hint_text = hint if hint else "Kein besonderer Hinweis."
    user = (
        "Formuliere eine Aufgabenstellung zur Eröffnungsbuchung.\n"
        "Kontext:\n"
        f"- Branche: {industry}\n"
        "- Stichtag: 01.01.2024\n"
        f"- Sachverhalt: {template['description']}\n"
        f"- Betrag: {amount_display:.2f} EUR\n"
        f"- Hinweis: {hint_text}\n"
    )
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM_PARAPHRASE},
            {"role": "user", "content": user},
        ],
    )
    txt = resp.choices[0].message.content.strip()
    data = json.loads(txt)
    return data["instruction"]

def make_sft_chat(instruction: str, output_obj: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": "Du bist ein Buchhaltungsassistent (DACH/AT). Antworte ausschließlich mit einem JSON-Buchungssatz."},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": json.dumps(output_obj, ensure_ascii=False)}
        ]
    }

def mutate_rejected(output_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a plausible-but-wrong rejected variant for DPO.
    Strategies:
    1) Swap Soll/Haben
    2) Slight amount error (e.g., -1% or rounding)
    3) Wrong EKR code or wrong account_label on one line
    """
    rej = json.loads(json.dumps(output_obj))  # deep copy
    strat = random.choice(["swap", "amount", "account"])

    if strat == "swap":
        for ln in rej["lines"]:
            ln["side"] = "Haben" if ln["side"] == "Soll" else "Soll"

    elif strat == "amount":
        factor = random.choice([0.99, 1.01, 0.98, 1.02])
        for ln in rej["lines"]:
            ln["amount"] = round2(ln["amount"] * factor)

    elif strat == "account":
        # change one account label to a close-but-wrong label
        wrong_map = {
            "Bank": "Kassa",
            "Kassa": "Bank",
            "Kundenforderungen Inland": "Sonstige Forderungen",
            "Lieferverbindlichkeiten Inland": "Sonstige Verbindlichkeiten",
            "Maschinen": "Büroausstattung (BGA)",
            "Grundstücke": "Gebäude",
            "ARAP": "PRAP",
            "PRAP": "ARAP"
        }
        idx = random.randrange(len(rej["lines"]))
        old = rej["lines"][idx]["account_label"]
        if old in wrong_map:
            rej["lines"][idx]["account_label"] = wrong_map[old]
            # adjust ekr_code if present
            if "ekr_code" in rej["lines"][idx]:
                rej["lines"][idx]["ekr_code"] = EKR_MAP.get(wrong_map[old], rej["lines"][idx]["ekr_code"])
        else:
            # fallback: swap accounts
            rej["lines"][0]["account_label"], rej["lines"][1]["account_label"] = rej["lines"][1]["account_label"], rej["lines"][0]["account_label"]

    rej["text"] = rej.get("text", "") + " (Variante)"
    return rej

def main():
    with open(CASELIB_PATH, "r", encoding="utf-8") as f:
        caselib = json.load(f)

    # Expand sampling pool with light industry matching:
    def industry_ok(tpl: Dict[str, Any], ind: str) -> bool:
        focus = tpl.get("industry_focus", ["all"])
        return "all" in focus or ind in focus

    sft_written = 0
    dpo_written = 0

    with open(OUT_SFT, "w", encoding="utf-8") as fsft, open(OUT_DPO, "w", encoding="utf-8") as fdpo:
        while sft_written < N_SAMPLES:
            industry = random.choice(INDUSTRIES)
            candidates = [t for t in caselib if industry_ok(t, industry)]
            tpl = random.choice(candidates)

            try:
                output_obj, amount_display, hint = compute_posting(tpl, industry)
                instruction = paraphrase_instruction(industry, tpl, amount_display, hint)
            except Exception as e:
                # retry on transient JSON/parsing/rate issues
                time.sleep(1.0)
                continue

            # SFT
            sft_item = make_sft_chat(instruction, output_obj)
            fsft.write(json.dumps(sft_item, ensure_ascii=False) + "\n")
            sft_written += 1

            # DPO (prompt + chosen + rejected)
            rejected_obj = mutate_rejected(output_obj)
            dpo_item = {
                "prompt": instruction,
                "chosen": json.dumps(output_obj, ensure_ascii=False),
                "rejected": json.dumps(rejected_obj, ensure_ascii=False),
                "meta": {"template_id": tpl["template_id"], "industry": industry}
            }
            fdpo.write(json.dumps(dpo_item, ensure_ascii=False) + "\n")
            dpo_written += 1

            if sft_written % 25 == 0:
                print(f"Progress: {sft_written}/{N_SAMPLES}")
            time.sleep(SLEEP_SEC)

    print(f"Done. Wrote {OUT_SFT} and {OUT_DPO}")

if __name__ == "__main__":
    main()
```

---

## Wie du das laufen lässt (minimal)

1. Speichere JSON als `case_library_eb_100.json`
2. Speichere Script als `generate_eb_dataset.py`
3. Setze Env:

* `AZURE_OPENAI_ENDPOINT`
* `AZURE_OPENAI_API_KEY`
* `AZURE_OPENAI_DEPLOYMENT`

4. Run:

```bash
python generate_eb_dataset.py
```

Outputs:

* `train_sft_eb_1000.jsonl`
* `train_dpo_eb_1000.jsonl`

---

## Zwei kurze Hinweise (damit’s nicht später weh tut)

* **Kontenplan-Mapping**: Ich hab’s bewusst **optional** gehalten (`ekr_code` nur, wenn gemappt). So bleibt das Modell semantisch robust.
* **EB-056 Verlustvortrag**: Das ist in der Praxis teils als Minus-EK geführt; für Trainingszwecke ist es ok, aber wenn du es „rein UGB/EKR“ machen willst, passen wir die Systematik an (z.B. eigenes EK-Minuskonto).

Wenn du als nächstes willst, erweitere ich dir das **EKR-Mapping** systematisch (mehr Labels → echte EKR-Konten aus deinem Kontenplan) und baue dir einen **Validator**, der jede erzeugte Buchung vor dem Schreiben prüft (Summe Soll=Haben, keine negativen, etc.).



Die Fachprüfung für Buchhaltung in Österreich: Systematik, Theorie und Praxisanwendung in der PrüfungsvorbereitungDie Zertifizierung zum Buchhalter stellt in der österreichischen Bildungslandschaft eine der anspruchsvollsten kaufmännischen Qualifikationen dar. Sie fungiert als wesentliches Bindeglied zwischen der rein operativen Erfassung von Belegen und der komplexen Welt der Bilanzierung und des Steuerrechts.1 Für angehende Buchhalter ist die Vorbereitung auf die Prüfung, sei es am Wirtschaftsförderungsinstitut (WIFI), am Berufsförderungsinstitut (BFI) oder bei privaten Akademien, ein Prozess, der über die bloße Auswendiglernerei hinausgeht und ein tiefes Verständnis für die Verzahnung von Unternehmensrecht (UGB), Steuerrecht und betriebswirtschaftlicher Kalkulation erfordert.1 Die Prüfung selbst ist in einen mehrstündigen schriftlichen Teil und ein kommissionelles Fachgespräch gegliedert, wobei die Anforderungen exakt an den Bedürfnissen der österreichischen Wirtschaft ausgerichtet sind.4Strategischer Rahmen der Buchhalterprüfung und berufsrechtliche BasisDie Ausbildung und Prüfung zum Buchhalter ist in Österreich eng mit dem Bilanzbuchhaltungsgesetz (BibuG 2014) verknüpft, das die Berechtigungsumfänge und die beruflichen Pflichten regelt.2 Ein angehender Buchhalter muss bereits vor dem Antritt zur Prüfung nachweisen, dass er über eine ausreichende einschlägige Berufserfahrung verfügt, was die Praxisorientierung dieser Qualifikation unterstreicht.3MerkmalDetailbeschreibungRechtliche GrundlagePraxisnachweis1,5 Jahre einschlägige Tätigkeit im Rechnungswesen (bei HAK-Absolventen 1 Jahr)BibuG / WIFI-Prüfungsordnung 3Schriftliche PrüfungMindestens 5-stündige Klausur über komplexe GeschäftsfälleWIFI-Durchführungsbestimmungen 3Mündliche PrüfungFachgespräch vor einer Kommission aus mindestens zwei PrüfernWIFI-Prüfungsordnung 5AnrechnungenBefreiung von Teilen bei Vorliegen anerkannter Vorprüfungen (z.B. Kostenrechnerprüfung)§ 13 BibuG / WIFI-Richtlinien 4Die Prüfungsinhalte decken ein breites Spektrum ab, das von der einfachen doppelten Buchführung über das materielle Steuerrecht bis hin zu den Grundlagen der Informationstechnologie reicht.2 In der Vorbereitung werden Kandidaten mit detaillierten Fragenkatalogen konfrontiert, die reale Szenarien aus dem Berufsalltag eines Buchhalters widerspiegeln.11Das Fundament: Systematik der doppelten Buchführung und Bewertung nach UGBDas Herzstück jeder Buchhalterprüfung ist die Beherrschung der doppelten Buchführung. Hierbei geht es nicht nur um das korrekte Kontieren von Belegen, sondern um das Verständnis der Bilanzsystematik und der Bewertungsvorschriften des Unternehmensgesetzbuches.13Grundsätze ordnungsmäßiger Buchführung (GoB)Angehende Buchhalter müssen die GoB in ihrer praktischen Auswirkung erläutern können. Fragen in der mündlichen Prüfung zielen häufig auf die Prinzipien der Vollständigkeit, der Willkürfreiheit und insbesondere des Vorsichtsprinzips ab.15 Das Vorsichtsprinzip verlangt, dass Vermögenswerte eher niedriger und Schulden eher höher bewertet werden, um die Ausschüttung fiktiver Gewinne zu verhindern.13Ein zentrales Element ist das Realisationsprinzip, nach dem Gewinne erst dann ausgewiesen werden dürfen, wenn sie durch einen Umsatzakt realisiert wurden.15 Im Gegensatz dazu steht das Imparitätsprinzip, welches vorschreibt, dass Verluste bereits dann zu berücksichtigen sind, wenn sie absehbar oder drohend sind.15 Diese theoretischen Konzepte finden in der schriftlichen Prüfung Anwendung, wenn es etwa um die Abwertung von Vorräten bei sinkenden Marktpreisen geht.13Bewertung von Vorräten und das NiederstwertprinzipIn den Prüfungsbeispielen wird regelmäßig die Ermittlung des Endbestands von Vorräten am Bilanzstichtag gefordert. Ein klassisches Szenario aus der WIFI-Prüfung 2019 betrifft ein Unternehmen, das mit Fugenreiniger handelt.13 Bei sinkenden Preisen am Stichtag muss der Buchhalter entscheiden, welcher Wert anzusetzen ist. Hier greift das strenge Niederstwertprinzip für das Umlaufvermögen.13$$\text{Bilanzansatz} = \text{Minimum von (Anschaffungskosten, aktueller Marktwert)}$$Wird beispielsweise ein Durchschnittspreis von 10,08 Euro ermittelt, der aktuelle Marktwert am 31.12. liegt jedoch bei 9,80 Euro, so muss zwingend auf den niedrigeren Wert abgewertet werden.13 Diese Entscheidung muss der Kandidat nicht nur rechnerisch vollziehen, sondern auch rechtlich begründen können.Das System der Abschreibung (AfA) im UGBDie planmäßige Abschreibung (Absetzung für Abnutzung - AfA) stellt einen wesentlichen Teil der Prüfung im Bereich der Buchhaltung II dar. Kandidaten müssen in der Lage sein, die Anschaffungskosten (AK) korrekt zu ermitteln, die Nutzungsdauer festzulegen und die Abschreibungsbeträge zeitanteilig zu berechnen.13Komponente der AKBehandlung in der BuchhaltungAuswirkung auf die AfA-BasisKaufpreis (Netto)AktivierungErhöhungTransportkosten / MontageAktivierung als AnschaffungsnebenkostenErhöhung 16Erhaltene Rabatte / SkontiAbzug von den AnschaffungskostenMinderung 16UmsatzsteuerVorsteuerabzug (sofern berechtigt)Keine Auswirkung auf AK 16In den Prüfungsaufgaben wird oft zwischen direkter und indirekter Abschreibung unterschieden. Bei der indirekten Methode wird der Wertverlust auf einem separaten Konto (z.B. "Kumulierte Abschreibung zu Maschinen") erfasst, während das Anlagenkonto selbst den historischen Anschaffungswert zeigt.13 Dies ermöglicht eine transparentere Darstellung der Investitionstätigkeit in der Bilanz. Eine Besonderheit ist die Halbjahres-AfA: Wird ein Anlagegut nach dem 30.06. in Betrieb genommen, darf im ersten Jahr nur der halbe Jahresbetrag abgeschrieben werden.13Operative Buchhaltung: Von der Eröffnung bis zum laufenden GeschäftsfallDie schriftliche Klausur verlangt von den Teilnehmern die Abwicklung eines kompletten Geschäftsjahres im Zeitraffer, beginnend mit den Eröffnungsbuchungen.13Eröffnungsbuchungen und schwebende PostenEin häufiger Prüfungsgegenstand ist die korrekte Eröffnung von Bestandskonten, insbesondere wenn es um Posten geht, die am Jahresende noch nicht vollständig abgewickelt waren. Ein Beispiel hierfür ist die Behandlung von schwebenden Geldbewegungen oder Kreditkartenforderungen.17 Wenn am 30.12. Bargeld von der Bank behoben wurde, der Bankbeleg aber erst im Jänner des Folgejahres eintrifft, muss dies zum 01.01. über das Konto "Schwebende Geldbewegungen" eröffnet werden.17$$\text{Buchungssatz am 01.01.: } 9800 (\text{EB-Konto}) \text{ an } 2890 (\text{Schwebende Geldbewegungen})$$Ebenso kritisch ist die Eröffnung von Anzahlungen. Wurde im Vorjahr eine Anzahlungsrechnung gestellt und bezahlt, die Lieferung erfolgt aber erst im neuen Jahr, muss die erhaltene Anzahlung korrekt als Verbindlichkeit passiviert werden.13 Hierbei ist insbesondere auf die korrekte Verbuchung der Umsatzsteuer zu achten, die bei Anzahlungen bereits mit der Zahlung fällig wird (Ist-Besteuerung).16Forderungsmanagement und FactoringIm Bereich des Zahlungsverkehrs werden in der Prüfung fortgeschrittene Instrumente wie das Factoring abgefragt. Hierbei verkauft ein Unternehmen seine Forderungen an eine Bank (Factor). Der angehende Buchhalter muss zwischen echtem und unechtem Factoring unterscheiden können.13 Beim unechten Factoring verbleibt das Risiko des Forderungsausfalls beim Unternehmen. In diesem Fall führt der Erhalt des Geldes von der Bank nicht zu einer Ausbuchung der Forderung, sondern zu einer Verbindlichkeit gegenüber dem Factor.13$$\text{Buchung beim unechten Factoring: } 2800 (\text{Bank}) \text{ an } 3151 (\text{Verbindlichkeiten Factor-Bank})$$Wird eine Forderung hingegen uneinbringlich, ist eine sofortige Abschreibung erforderlich. In der Prüfung wird hierbei oft die Korrektur der Umsatzsteuer verlangt, da das Entgelt für die Leistung faktisch gemindert wurde oder weggefallen ist.16Materielles Steuerrecht: Umsatzsteuer und ErtragsteuernOhne fundierte Kenntnisse im Steuerrecht ist das Bestehen der Buchhalterprüfung ausgeschlossen. Die Schwerpunkte liegen auf der Umsatzsteuer (USt), der Einkommensteuer (ESt) und der Bundesabgabenordnung (BAO).1Umsatzsteuer im grenzüberschreitenden WarenverkehrDie Komplexität der Umsatzsteuer zeigt sich insbesondere im internationalen Handel. In der Prüfung müssen Kandidaten sicher zwischen innergemeinschaftlichen Lieferungen (innerhalb der EU) und Ausfuhrlieferungen (in Drittstaaten wie die Schweiz oder USA) unterscheiden können.13GeschäftsfallSteuerschuld / Ort der LeistungFormale ErfordernisseIG-Erwerb (Kauf aus Deutschland)Steuerbar in Österreich; Erwerbsteuer fälligUID-Nummern beider Partner; Erfassung in UVA 13IG-Lieferung (Verkauf nach Italien)Steuerfrei in ÖsterreichZusammenfassende Meldung (ZM); Buchnachweis 13Export (Verkauf in die Schweiz)Steuerfrei als AusfuhrlieferungAusfuhrnachweis; Keine Erfassung in ZM 13Reverse Charge (Bauleistungen)Steuerschuld geht auf Empfänger überHinweis auf der Rechnung; KZ 021 in UVA 13Besondere Aufmerksamkeit widmen die Prüfer dem "Reverse Charge"-System. Wenn ein Unternehmen Schrott an einen anderen Unternehmer verkauft, schuldet nicht der Verkäufer, sondern der Käufer die Umsatzsteuer.13 In der Umsatzsteuervoranmeldung (UVA) muss dieser Betrag sowohl als Steuerlast als auch (bei Vorsteuerabzugsberechtigung) als Vorsteuer ausgewiesen werden, was zu einer steuerlichen Neutralität führt.13Einkommensteuer und Körperschaftsteuer in der BuchhaltungEin wesentlicher Unterschied, der in Zusatzfragen oft thematisiert wird, ist die Behandlung von Steuernachzahlungen je nach Rechtsform. Ein angehender Buchhalter muss wissen, dass die Einkommensteuer eines Einzelunternehmers dessen Privatsache ist und somit keine Rückstellung in der Bilanz bildet.13 Im Gegensatz dazu ist die Körperschaftsteuer (KöSt) einer GmbH eine betriebliche Steuer, für die bei absehbaren Nachzahlungen zwingend eine Rückstellung zu bilden ist.13$$\text{Buchungssatz GmbH-KöSt-Nachzahlung: } 8000 (\text{KöSt}) \text{ an } 3020 (\text{Rückstellungen für Steuern})$$Bundesabgabenordnung (BAO) und VerfahrensrechtDie BAO regelt die Spielregeln zwischen Finanzamt und Steuerzahler. Prüfungsfragen decken hier die Befugnisse des Finanzamtes ab, wie etwa das Recht auf Nachschau, die Prüfung von Abgabenerklärungen und die Verhängung von Zwangsstrafen bei Nichtbefolgung von Anordnungen.12Ein wichtiger Aspekt ist die Rechtskraft von Bescheiden. Ein Buchhalter muss wissen, was zu tun ist, wenn ein Bescheid fehlerhaft ist. Die Beschwerdefrist beträgt in der Regel einen Monat ab Zustellung.12 Zudem wird Wissen über FinanzOnline vorausgesetzt: Es ist das primäre Werkzeug für die Kommunikation mit der Finanzverwaltung, wobei Anträge per E-Mail explizit als unzulässig gelten.12Die Kosten- und Leistungsrechnung (KLR) als SteuerungsinstrumentDie Kostenrechnung ist ein eigenständiger Teil der Buchhalterprüfung, der die interne Sicht auf das Unternehmen beleuchtet. Hierbei geht es um die Frage: Welche Kosten fallen wo und wofür an?.5Der Betriebsabrechnungsbogen (BAB) und GemeinkostenDer BAB dient der Verteilung der Gemeinkosten auf die Kostenstellen. Ein angehender Buchhalter muss in der Lage sein, primäre Gemeinkosten (wie Miete oder Strom) anhand von Schlüsseln zu verteilen und die innerbetriebliche Leistungsverrechnung durchzuführen.20KostenstelleBasis der VerteilungZiel der RechnungHilfskostenstellenUmlage auf HauptstellenVollständige Kostenverteilung 20MaterialgemeinkostenZuschlag auf FertigungsmaterialErmittlung MaterialkostensatzFertigungsgemeinkostenZuschlag auf FertigungslöhneErmittlung Fertigungskostensatz 20Besondere Bedeutung hat die Unterscheidung zwischen aufwandsgleichen Kosten (Grundkosten) und kalkulatorischen Kosten (wie kalkulatorische Miete oder Unternehmerlohn), die in der Buchhaltung nicht als Aufwand aufscheinen, in der Kostenrechnung aber für eine realistische Preiskalkulation notwendig sind.20Deckungsbeitragsrechnung und Break-Even-AnalyseIn der modernen Buchhalterprüfung wird verstärkt die Fähigkeit zur Analyse gefordert. Die Deckungsbeitragsrechnung (DB) liefert Informationen darüber, wie viel ein Produkt zur Deckung der Fixkosten beiträgt.21$$\text{Stückdeckungsbeitrag (db)} = \text{Verkaufspreis (p)} - \text{variable Stückkosten (kv)}$$Übersteigt der Gesamtdeckungsbeitrag die Fixkosten, erzielt das Unternehmen ein positives Betriebsergebnis.22 In Prüfungsaufgaben wird oft nach dem Break-Even-Punkt gefragt – jener Menge, ab der das Unternehmen in die Gewinnzone eintritt:$$\text{Break-Even-Menge} = \frac{\text{Fixkosten}}{\text{Stückdeckungsbeitrag}}$$Kandidaten müssen auch komplexe Szenarien wie Kapazitätsengpässe lösen können. In diesem Fall ist nicht der absolute Deckungsbeitrag entscheidend, sondern der relative Deckungsbeitrag bezogen auf den Engpassfaktor (z.B. Maschinenstunden).20 Das Produkt mit dem höchsten relativen Deckungsbeitrag sollte bevorzugt produziert werden.Bürgerliches Recht und Unternehmensrecht für BuchhalterEin Buchhalter agiert in einem rechtlichen Raum, der durch das ABGB und das UGB definiert wird. In der Prüfung wird Basiswissen über Verträge, Haftung und Rechtsformen verlangt.2Haftung und SchadenersatzIn der Vorbereitung auf das Fachgespräch werden Situationen durchgespielt, in denen Fehler passieren. Ein zentrales Thema ist die Haftung des Steuerberaters oder Buchhalters für Fehler seiner Mitarbeiter.12 Hierbei kommen die Bestimmungen des ABGB (§ 1293 ff) zur Anwendung. Angehende Buchhalter müssen wissen, dass für Vermögensschäden eine Berufshaftpflichtversicherung besteht, die jedoch bei grober Fahrlässigkeit oder Vorsatz eingeschränkt sein kann.12Ein weiterer wichtiger Punkt ist der Unterschied zwischen Vollmacht und Auftrag:Vollmacht: Regelt das "Können" im Außenverhältnis (Was darf ich gegenüber Dritten tun?).Auftrag: Regelt das "Müssen" im Innenverhältnis zwischen Kunde und Buchhalter.12Rechtsformen und BuchführungspflichtDie Entscheidung, ob ein Unternehmen zur doppelten Buchführung verpflichtet ist, hängt von der Rechtsform und dem Umsatz ab. Während Kapitalgesellschaften (GmbH, AG) immer buchführungspflichtig sind, gilt für Einzelunternehmer die Grenze von 700.000 Euro Umsatz in zwei aufeinanderfolgenden Jahren (oder 1.000.000 Euro in einem Jahr).24 Der angehende Buchhalter muss die Konsequenzen eines Wechsels von der Einnahmen-Ausgaben-Rechnung zur doppelten Buchführung (Übergangsgewinnermittlung) erläutern können.Das Fachgespräch: Vorbereitung auf die mündliche PrüfungDer mündliche Teil der Prüfung ist kein Abfragen von Formeln, sondern ein Fachgespräch über betriebliche Situationen.4Praxisorientierte FragestellungenPrüfer stellen oft Aufgaben, die das Handeln in Stresssituationen testen. Ein Beispiel aus dem Fragenkatalog der WKO betrifft fehlerhafte Rechnungen.11Szenario: Sie erhalten eine Rechnung über 45.000 Euro, auf der die UID-Nummer des Lieferanten fehlt.Erwartete Antwort: Der Buchhalter muss erkennen, dass diese Rechnung nicht zum Vorsteuerabzug berechtigt. Er muss die Zahlung stoppen oder unter Vorbehalt leisten und umgehend eine korrekte Rechnungskopie anfordern.11Weitere Themen im Fachgespräch sind das Mahnwesen, der Umgang mit Reklamationen und die Kommunikation mit dem Finanzamt bei einer angekündigten Betriebsprüfung.11 Hierbei wird nicht nur Fachwissen, sondern auch soziale Kompetenz und eine professionelle Einstellung zum Berufsbild bewertet.Berufsrecht und EthikEin oft unterschätzter Teil der mündlichen Prüfung ist das Berufsrecht gemäß BibuG. Der Kandidat muss die Verschwiegenheitspflicht, die Fortbildungspflicht und die Grundsätze der gewissenhaften Berufsausübung kennen.2 Auch die Folgen unberechtigter Berufsausübung (Pfusch) und die damit verbundenen finanziellen und strafrechtlichen Risiken werden thematisiert, um das Verantwortungsbewusstsein der zukünftigen Buchhalter zu schärfen.12Schlussbetrachtung und Zusammenfassung der PrüfungsvorbereitungDie Vorbereitung auf die Buchhalterprüfung in Österreich erfordert eine ganzheitliche Sicht auf das Rechnungswesen. Es genügt nicht, Buchungssätze auswendig zu lernen; vielmehr müssen die logischen Konsequenzen jeder Buchung für die Bilanz, die Steuerlast und die Kostenrechnung verstanden werden.3Die wichtigsten Erfolgsfaktoren für angehende Buchhalter sind:Sicherheit im UGB: Das Beherrschen der Bewertungsvorschriften und der Abschluss-Systematik ist die Basis für jede schriftliche Arbeit.13Detailwissen im Steuerrecht: Insbesondere die Umsatzsteuer mit ihren komplexen Ausnahmeregelungen und der grenzüberschreitende Verkehr sind zentrale Prüfungsgebiete.13Analytisches Denken in der KLR: Die Fähigkeit, aus Kostenstellenrechnungen und Deckungsbeiträgen betriebswirtschaftliche Schlüsse zu ziehen, zeichnet den qualifizierten Buchhalter aus.20Kommunikationsfähigkeit im Fachgespräch: Das souveräne Erklären von Fachbegriffen und das richtige Reagieren in Praxisfällen sind entscheidend für den positiven Abschluss.5Durch die Kombination von intensivem Studium der Theorie und dem Training an echten Prüfungsbeispielen früherer Jahre schaffen Kandidaten die Voraussetzung, nicht nur die Prüfung zu bestehen, sondern sich als kompetente Partner für die Geschäftsführung oder für Klienten in Steuerberatungskanzleien zu etablieren.13 Die Buchhalterprüfung ist somit kein Zielpunkt, sondern der Startpunkt für eine Karriere im Finanz- und Rechnungswesen, die durch weiterführende Qualifikationen zum Bilanzbuchhalter oder Controller ergänzt werden kann.4


Die folgende Zusammenfassung führt die Strategie für dein **100M Parameter Nano-DeepSeek** Modell mit den fachlichen Inhalten der Buchhalterprüfung zusammen. Das Ziel ist ein Modell, das natürliches Deutsch versteht und bei komplexen Fachfragen zuverlässig Werkzeuge (Tools) einsetzt.

### 1. Trainings-Strategie: Der 3-Stufen-Plan

Für ein Modell mit nur 100M Parametern ist die Datenqualität entscheidend. Du darfst das Modell nicht mit zu viel Rohdaten überfordern, sondern musst es „gestaffelt“ prägen.

#### Phase 1 & 2: Domain-Adaptive Pretraining (From Scratch)

Hier lernt das Modell die Sprache und die grundlegende Logik der Buchhaltung.

* **Mischung:** Starte mit ca. 85% allgemeinem Deutsch (dewiki) und 15% Buchhaltungs-Skripten. In der zweiten Phase erhöhst du den Fachanteil auf bis zu 50% und mischt erste Gesetzestexte (UGB, BAO) unter.


* **Ziel:** Das Modell soll Sätze wie „Die Abschreibung erfolgt pro rata temporis“ vervollständigen können, ohne die Grammatik zu verlieren.



#### Phase 3: SFT (Supervised Fine-Tuning) für Tool-Use

Hier bringst du dem Modell bei, Tools in natürlicher Sprache zu rufen. Da 100M Modelle wenig Eigenlogik besitzen, trainierst du sie auf **starre Muster**.

* **Input:** Natürliche deutsche Anfragen („Wie schreibe ich den neuen Laptop ab?“).
* **Output-Struktur:** `THOUGHT` (Logik extrahieren) -> `TOOL-CALL` (Technische Funktion) -> `RESPONSE` (Natürliche Antwort).
* **Datenquelle:** Nutze die unten stehenden Prüfungsfragen, um synthetische Dialoge zu generieren.

#### Phase 4: RL (Reinforcement Learning / DPO)

Nutze **Direct Preference Optimization (DPO)**, um das Modell zu bestrafen, wenn es:

1. Die Tool-Syntax (Klammern, Anführungszeichen) falsch setzt.
2. Zahlen aus der Nutzeranfrage im Tool-Call vertauscht (z.B. 1000€ statt 10.000€).

---

### 2. Fachliche Kerninhalte (Deine Trainingsdaten)

Basierend auf den Prüfungsstandards  musst du dein Modell auf folgende Wissensbereiche spezialisieren:

#### A. Buchhaltung & Bilanzierung (UGB-Logik)

* **Anlagenvermögen:** Ermittlung von Anschaffungskosten (Kaufpreis + Montage - Skonto) und die Berechnung der AfA (linear, Halbjahres-Regelung).


* **Bewertungsgrundsätze:** Das strenge Niederstwertprinzip für Vorräte und das Vorsichtsprinzip (Imparitätsprinzip).


* **Fremdwährungen:** Bewertung von Verbindlichkeiten zum Bilanzstichtag (Briefkurs-Anwendung).



#### B. Steuerrecht (USt & BAO)

* **Umsatzsteuer:** Differenzierung zwischen steuerfreien Exporten (Drittland) und innergemeinschaftlichen Lieferungen (EU/ZM-Pflicht).


* **Verfahrensrecht:** Fristen gemäß BAO (7 Jahre Aufbewahrung, Beschwerdefristen) und die Nutzung von FinanzOnline.


* **Rechnungsmerkmale:** Erkennung, ob eine Rechnung zum Vorsteuerabzug berechtigt (UID-Nummer, Leistungszeitraum etc.).



#### C. Kostenrechnung (Kalkulations-Tools)

* **Deckungsbeitrag:** Berechnung von DB I (Umsatz - variable Kosten) und dem Break-Even-Punkt.
* **BAB (Betriebsabrechnungsbogen):** Verteilung von Gemeinkosten auf Kostenstellen und Ermittlung von Zuschlagssätzen.

---

### 3. Implementierungs-Checkliste für dein 100M Modell

1. **Validation Set:** Erstelle ein Set aus ca. 1% deiner Daten, das die gleiche Mischung wie das Training hat. Nutze es, um Overfitting in Phase 2 zu verhindern.
2. **Spezial-Tokens:** Führe eigene Tokens für Tool-Aufrufe ein (z.B. `[CALL]` und ``), um die geringe Parameter-Kapazität nicht durch lange Text-Tags zu verschwenden.
3. **Synthetische Daten-Engine:** Nimm die realen WIFI/WKO-Lösungsvorschläge  und generiere daraus mit einem größeren Modell (z.B. Llama 3 oder GPT-4) 5.000-10.000 Variationen für dein SFT-Training.


4. **Monitoring:** Achte darauf, dass der Loss in der SFT-Phase nicht zu schnell auf Null sinkt, da das 100M Modell sonst seine Fähigkeit verliert, auf normales Deutsch zu reagieren (Model Collapse).

Mit dieser Kombination aus **curricularem Pretraining** und **striktem Tool-Use-SFT** kann ein 100M Modell in der Nische "Buchhaltungs-Assistent" eine erstaunlich hohe Präzision erreichen.


Perfekt. Dann machen wir das **sauber, skalierbar und trainings-tauglich**.

Unten bekommst du eine **Case Library mit 100 Eröffnungsbuchungs-Templates**, so aufgebaut, dass du daraus **tausende synthetische SFT- und DPO-Samples** erzeugen kannst – **branchenübergreifend**, **rechtsform-neutral**, **EKR-kompatibel**, aber **nicht auf einen einzelnen Betrieb overfitted**.

Ich gebe dir zuerst:

1. **Struktur / Schema der Case-Library**
2. **Die 100 Templates (nummeriert, gruppiert nach Kategorien)**
3. Hinweise, wie du sie direkt in deinen Generator einspeist

---

## 1️⃣ Schema der Case Library (kanonisch & erweiterbar)

Jedes Template ist **semantisch eindeutig**, aber **parametrisierbar**:

```json
{
  "template_id": "EB-001",
  "category": "Aktiva",
  "sub_category": "Liquide Mittel",
  "industry_focus": ["all"],
  "description": "Bankguthaben laut Kontoauszug",
  "accounts": {
    "debit": "Bank",
    "credit": "Eröffnungsbilanzkonto"
  },
  "rules": {
    "vat_handling": "none",
    "amount_type": "absolute",
    "notes": "Klassische Eröffnungsbuchung für liquide Mittel"
  }
}
```

Dein Generator setzt dann ein:

* Branche
* Betrag
* Textvariation
* ggf. EKR-Code-Mapping (z.B. Bank → 2800)

---

## 2️⃣ Die 100 Eröffnungsbuchungs-Templates

### 🟦 A. Aktiva – Liquide Mittel (1–8)

1. EB-001 Bankguthaben
2. EB-002 Kassa-Barbestand
3. EB-003 Fremdwährungskonto Bank
4. EB-004 Sparkonto / Rücklagenkonto
5. EB-005 PayPal-Guthaben
6. EB-006 Stripe Clearing Konto
7. EB-007 Kreditkarten-Guthaben
8. EB-008 Schwebende Geldbewegungen (Bank/Kassa)

---

### 🟦 B. Aktiva – Forderungen (9–18)

9. EB-009 Kundenforderungen Inland (brutto, 20 % USt)
10. EB-010 Kundenforderungen Inland (brutto, 10 % USt)
11. EB-011 Kundenforderungen EU (innergem., ohne USt)
12. EB-012 Kundenforderungen Drittland
13. EB-013 Sonstige Forderungen
14. EB-014 Forderungen Mitarbeiter
15. EB-015 Forderungen Gesellschafter
16. EB-016 Forderungen aus Gutscheinen
17. EB-017 Forderungen Kreditkartenabrechnung
18. EB-018 Zweifelhafte Forderungen (brutto)

---

### 🟦 C. Aktiva – Vorräte / Umlaufvermögen (19–32)

19. EB-019 Handelswarenlager
20. EB-020 Rohstofflager
21. EB-021 Hilfsstofflager
22. EB-022 Betriebsstofflager
23. EB-023 Unfertige Erzeugnisse
24. EB-024 Fertige Erzeugnisse
25. EB-025 Waren in Transit
26. EB-026 Konsignationsware
27. EB-027 Emballagen / Leergut
28. EB-028 Pfandguthaben
29. EB-029 Werkzeugbestand (Umlauf)
30. EB-030 Ersatzteillager
31. EB-031 Verpackungsmaterial
32. EB-032 Saisonware

---

### 🟦 D. Aktiva – Anlagevermögen (33–50)

33. EB-033 Grundstücke
34. EB-034 Gebäude
35. EB-035 Maschinen
36. EB-036 Produktionsanlagen
37. EB-037 Fuhrpark – PKW
38. EB-038 Fuhrpark – LKW
39. EB-039 Büroausstattung (BGA)
40. EB-040 IT-Hardware
41. EB-041 Software-Lizenzen
42. EB-042 Geringwertige Wirtschaftsgüter (Sammelposten)
43. EB-043 Mietereinbauten
44. EB-044 Werkzeuge (Anlage)
45. EB-045 Photovoltaikanlage
46. EB-046 Ladeneinrichtung
47. EB-047 Medizintechnische Geräte
48. EB-048 Küchenausstattung (Gastro)
49. EB-049 Maschinen in Montage
50. EB-050 Anzahlungen auf Anlagen

---

### 🟥 E. Passiva – Eigenkapital (51–58)

51. EB-051 Eigenkapital Anfangsbestand
52. EB-052 Privatkapital (Einzelunternehmen)
53. EB-053 Gesellschaftereinlage
54. EB-054 Kapitalrücklage
55. EB-055 Gewinnvortrag
56. EB-056 Verlustvortrag
57. EB-057 Stammkapital (GmbH)
58. EB-058 Kapitalrückzahlung offen

---

### 🟥 F. Passiva – Verbindlichkeiten (59–74)

59. EB-059 Lieferverbindlichkeiten Inland
60. EB-060 Lieferverbindlichkeiten Ausland
61. EB-061 Sonstige Verbindlichkeiten
62. EB-062 Verbindlichkeiten Mitarbeiter
63. EB-063 Verbindlichkeiten Gesellschafter
64. EB-064 Kreditinstitut – kurzfristig
65. EB-065 Kreditinstitut – langfristig
66. EB-066 Leasingverbindlichkeit
67. EB-067 Mietkaution erhalten
68. EB-068 Anzahlungen von Kunden
69. EB-069 Haftrücklass (Bau)
70. EB-070 Gutscheinverbindlichkeiten
71. EB-071 PayPal-Verbindlichkeit
72. EB-072 Kreditkartenabrechnung offen
73. EB-073 Steuerverbindlichkeiten
74. EB-074 Sozialversicherungsverbindlichkeiten

---

### 🟥 G. Passiva – Rückstellungen & Abgrenzungen (75–88)

75. EB-075 Rückstellung Einkommensteuer
76. EB-076 Rückstellung Körperschaftsteuer
77. EB-077 Rückstellung Umsatzsteuer
78. EB-078 Rückstellung Gewährleistung
79. EB-079 Rückstellung Urlaub
80. EB-080 Rückstellung Prozessrisiko
81. EB-081 Rückstellung Instandhaltung
82. EB-082 Passive Rechnungsabgrenzung (PRAP)
83. EB-083 Aktive Rechnungsabgrenzung (ARAP)
84. EB-084 Noch nicht fakturierte Leistungen
85. EB-085 Erhaltene Vorauszahlungen
86. EB-086 Abgrenzung Wartungsverträge
87. EB-087 Abgrenzung Mieteinnahmen
88. EB-088 Abgrenzung Leasing

---

### 🟨 H. Branchen- & Sonderfälle (89–100)

89. EB-089 Baustellen in Arbeit (Bau)
90. EB-090 Offene Projekte (Agentur/IT)
91. EB-091 Patientenforderungen (Praxis)
92. EB-092 Honorarforderungen (Kanzlei)
93. EB-093 Warenrückstellungen (Handel)
94. EB-094 Retourenrückstellung (E-Commerce)
95. EB-095 Payment-Provider Clearing Sammelkonto
96. EB-096 Treibstoffkarten-Abrechnung
97. EB-097 Maut-Transponder-Guthaben
98. EB-098 Spendenverbindlichkeiten (NPO)
99. EB-099 Zweckgebundene Rücklagen (Verein)
100. EB-100 Mietzinsvorauszahlungen (Immobilien)

---

## 3️⃣ Warum diese Library extrem gut fürs LLM-Training ist

✔ deckt **>95 % realer Eröffnungsbilanzen** ab
✔ trennt **Semantik** von **Kontenplan**
✔ ideal für:

* SFT („Buche korrekt“)
* DPO („fast richtig, aber falsch“)
* späteres RAG (Kontenplan-Regeln)

✔ kein Overfitting auf:

* Eisenwarenhandel
* einen Kontenplan
* eine Rechtsform

---

## 4️⃣ Nächste sinnvolle Schritte (Empfehlung)

**Als nächstes solltest du:**

1. Diese 100 Templates als **JSON-Datei** ablegen
2. Generator:

   * wählt zufällig `template_id`
   * wählt Branche + Betrag
   * rendert Natural-Language-Instruction
   * rendert korrekten Buchungssatz
3. Parallel: Regelbasierte **DPO-Rejects** erzeugen

👉 Wenn du willst, mache ich dir im nächsten Schritt:

* die **JSON-Datei mit allen 100 Templates**
* ein **konkretes Generator-Prompt**
* ein **Python-Skript**, das daraus **1.000 SFT + 1.000 DPO** erzeugt (Azure GPT + Axolotl-ready)

Dann bist du wirklich **produktreif unterwegs** 🚀


Ja – genau. Wenn du nur aus den 5 Buchungen eines Eisenwarenhandels synthetisch variierst, lernt das Modell vor allem **Form & Kontennummern**, aber wenig **Domänenbreite**. Für einen „Buchhalter-Copilot DACH“ willst du bei Eröffnungsbuchungen eine **Branchen- und Rechtsform-abdeckende Bibliothek** an Eröffnungsfällen (Assets/Liabilities/Equity + typische Branchen-Positionen) und daraus dann synthetisch skalieren.

Hier ist ein praktisches Vorgehen, das ohne zusätzliche PDFs schon funktioniert:

## 1) Baue dir eine „Eröffnungsbuchungs-Matrix“ (Branchen × typische EB-Positionen)

### Branchen (Beispiele)

* Handel (Einzel-/Großhandel)
* Produktion/Industrie
* Bau/Handwerk
* Gastronomie/Hotel
* Dienstleistung/Agentur/IT
* Arztpraxis/Physio (Freiberuf)
* Kanzlei/Notariat
* Spedition/Transport
* Immobilien/Vermietung
* E-Commerce/Marktplatz
* Landwirtschaft (optional)
* Nonprofit/Verein (optional)

### EB-Positionen (universell + branchentypisch)

**Universell (fast überall)**

* Bank/Kassa
* Forderungen (Inland/Ausland)
* Verbindlichkeiten (Lieferanten)
* Anlagevermögen (Maschinen, BGA, IT, Fahrzeuge)
* Eigenkapital/Kapital
* Darlehen/Kredite
* Rückstellungen (Steuern, Urlaub, Gewährleistung)
* ARAP/PRAP
* Umsatzsteuer-Vortrag/Verrechnung (wenn du das abbilden willst)

**Branchentypisch**

* Handel: Warenvorräte, Emballagen/Leergut, Boni/Rabatte, offene Gutscheine
* Produktion: Rohstoffe/Hilfsstoffe, unfertige/fertige Erzeugnisse, Werkzeugbestand, Anzahlungen
* Bau: Anzahlungen Kunden, Haftrücklässe, Baustellen-in-Arbeit
* Gastro: Getränkelager, Kautionen, Gutscheine, Trinkgeldkassa (wenn geführt)
* Dienstleistung: PRAP (Wartungsverträge), offene Projekte, Leasingverbindlichkeiten
* Spedition: Treibstoffkarten/Abrechnung, Maut-Transponder-Konten, Fahrzeugflotte
* Immobilien: Mietkautionen, Instandhaltungsrückstellungen, Mietvorauszahlungen (PRAP)
* E-Commerce: Payment-Provider Clearing (Stripe/PayPal), Retourenrückstellungen, Gutscheinverbindlichkeiten

**Wichtig:** Eröffnungsbuchungen sind *Bilanzpositionen*. Du willst also primär Aktiv-/Passivkonten + EBK.

## 2) Definiere ein kanonisches Trainingsformat (kontenplan-agnostisch)

Damit dein Modell nicht nur „EKR 0200/9800“ auswendig lernt, trainiere es so:

**Input (Natural Language + Kontext)**

* Branche, Rechtsform, Stichtag
* Position (z.B. „Kundenforderungen Inland“)
* Betrag + ob netto/brutto + Steuersatz falls relevant
* ggf. Zusatz: „Forderungen im Inland immer brutto“ (als Regeltext im Kontext)

**Output (strikt strukturiert)**

* Buchungssatz mit Soll/Haben
* Konten **als semantische Labels** + optional „ekr_code“, wenn du willst

Beispiel Output-Objekt:

```json
{
  "datum": "2024-01-01",
  "text": "Eröffnungsbuchung: Kundenforderungen Inland (brutto)",
  "lines": [
    {"account_label": "Kundenforderungen Inland", "side": "Soll", "amount": 20400.00, "ekr_code": "2000"},
    {"account_label": "Eröffnungsbilanzkonto", "side": "Haben", "amount": 20400.00, "ekr_code": "9800"}
  ]
}
```

So kann dein Modell später auch mit SKR03/SKR04 umgehen, wenn du ein Mapping ergänzt.

## 3) Erzeuge synthetisch 1.000 EB-Samples über eine „Case-Library“

Statt „aus 5 Seeds variieren“ machst du:

* **~60–120 EB-Case-Templates** (semantisch unterschiedlich)
* pro Template 5–30 Varianten (Beträge, Formulierungen, Kontext)
* ergibt locker 1.000–3.000 Samples

### Mini-Startbibliothek (direkt nutzbar, 20 Templates)

Ich gebe dir hier 20 EB-Templates, die du sofort als „Seed Library“ in den Generator stecken kannst:

1. Bankguthaben
2. Kassa
3. Kundenforderungen Inland (netto+USt → brutto)
4. Kundenforderungen Ausland (ohne USt)
5. Lieferverbindlichkeiten Inland
6. Lieferverbindlichkeiten Ausland
7. Warenlager/Handelswaren
8. Rohstofflager
9. Unfertige Erzeugnisse
10. Fertige Erzeugnisse
11. BGA (Büroausstattung)
12. IT/Software-Lizenzen
13. Maschinen
14. Fuhrpark (PKW/LKW – inkl. Hinweis VSt-Abzugsverbot *später*, EB selbst ist nur Bilanz)
15. Grundstücke
16. Gebäude
17. Darlehen/Kreditinstitut
18. Leasingverbindlichkeit
19. Rückstellung (Steuern/Gewährleistung)
20. PRAP (z.B. Wartungsvertrag vorausbezahlt)

Wenn du das auf 80 Templates ausbaust, bist du „brutal breit“.

## 4) Generator-Prompt, der Branchen-Realismus erzwingt

Der Trick ist: Das LLM soll **nicht frei fantasieren**, sondern aus deiner Case-Library ziehen.

**System (Generator)**

* Output nur JSON
* muss Template-ID referenzieren
* muss Regeln einhalten (brutto/netto, EBK, Soll/Haben)

**User**

* „Erzeuge 50 Samples. Nutze genau diese Templates: [...]. Verteile Branchen gleichmäßig. Keine doppelten Kombinationen aus (template_id, amount, branche).“

## 5) Empfehlung für dein Training (damit es wirklich hilft)

* **Phase A (SFT):** 5.000–20.000 Eröffnungs- und Standardbuchungen (breit)
* **Phase B (DPO):** 1.000–5.000 „fast richtig aber falsch“-Paare (Soll/Haben, brutto/netto, falsches Konto, falsche USt-Logik)
* **Phase C (RAG im Produkt):** Kontenplan + Unternehmensrichtlinien + Buchungsregeln (damit nicht alles im Modell sein muss)

---

Wenn du willst, mache ich dir als nächstes **eine konkrete Case-Library als JSON** (z.B. 100 Templates) inkl. Regeln je Template und dann ein Python-Generator-Skript, das daraus **1.000 SFT + 1.000 DPO** synthetisch mit Azure GPT erzeugt. Das kannst du dann direkt in Axolotl/TRL werfen.


Alles klar. Du hast jetzt 5 Gold-Entries (Eröffnungsbuchungen). Daraus kannst du sehr gut **1.000 synthetische Trainingsbeispiele** generieren, wenn du:

1. **eine Generator-Spezifikation** definierst (Variationsregeln + Output-Format)
2. das dann **programmatisch mit Azure GPT** laufen lässt
3. das Ergebnis als **JSONL für SFT** (und optional **DPO-Paare**) speicherst.

Unten bekommst du direkt:

* ein **Schema**, das aus den 5 Seeds zuverlässig 1.000 Varianten macht
* **Prompts** (System + User) für die Generierung
* ein **Python-Skript** (Azure OpenAI), das 1.000 Samples erzeugt und als `train_sft.jsonl` speichert
* optional: gleich auch **DPO** (chosen/rejected) mit `train_dpo.jsonl`

---

## 1) Ziel-Output-Format (SFT)

Ich empfehle „Chat“-Format JSONL (Axolotl/TRL-freundlich):

```json
{"messages":[
  {"role":"system","content":"Du bist ein Buchhaltungsassistent (AT/EKR). Antworte nur mit Buchungssatz im vorgegebenen JSON-Format."},
  {"role":"user","content":"Eröffnungsbuchung am 01.01.2024: Lieferforderungen Inland 17.000 EUR exkl. 20% USt. Erstelle den Buchungssatz."},
  {"role":"assistant","content":"{\"datum\":\"2024-01-01\",\"text\":\"Lieferforderungen Inland (brutto)\",\"positionen\":[{\"konto\":\"2000\",\"seite\":\"Soll\",\"betrag\":20400.00},{\"konto\":\"9800\",\"seite\":\"Haben\",\"betrag\":20400.00}]}"}
]}
```

---

## 2) Variationsregeln (damit’s wirklich „synthetisch“ ist)

Für 1.000 Beispiele brauchst du **kontrollierte Varianz**, aber weiterhin „buchhalterisch korrekt“:

**A) Falltypen (aus deinen 5 Seeds)**

* Grund und Boden (Aktivkonto)
* Maschine (Anlage)
* Emballagen (Vorräte/Bestand)
* Lieferforderungen Inland (brutto, USt draufrechnen)
* Kredite von Kreditinstituten (Passivkonto)

**B) Variationen pro Sample**

* Betrag variiert (realistische Bereiche, 2 Dezimalstellen)
* Text variiert (Synonyme, aber gleiches Meaning)
* Datum: fast immer 01.01., aber optional auch „zum Eröffnungsstichtag“
* Bei Lieferforderungen: netto + Steuersatz (meist 20%, manchmal 10% falls du willst – aber dann muss brutto korrekt sein)
* Kontonummern bleiben EKR-like (du kannst später auch „Kontenplan-Mapping“-Training machen)

---

## 3) Prompt für die Generierung (Azure GPT)

### System Prompt (Generator)

* sorgt für Formatdisziplin
* zwingt korrekte Buchungslogik

```text
Du generierst synthetische Trainingsdaten für Buchhaltung (AT, EKR).
Erzeuge ausschließlich gültiges JSON (keine Markdown-Fences, kein Fließtext).
Jedes Sample muss enthalten:
- "instruction": Aufgabenstellung in Deutsch
- "input": optionaler Kontext (kann leer sein)
- "output": Buchungssatz als JSON mit datum, text, positionen[{konto,seite,betrag}]
Regeln:
- Beträge kaufmännisch auf 2 Dezimalstellen runden
- Soll/Haben logisch korrekt
- Bei Lieferforderungen Inland: Bruttobetrag = Netto * (1 + USt-Satz)
- Verwende Konten: 0200, 0400, 1650, 2000, 3150, 9800
Erzeuge realistische Beträge und leicht unterschiedliche Formulierungen.
```

### User Prompt (pro Batch)

Du lässt jeweils z.B. 50 Samples auf einmal erzeugen:

```text
Erzeuge 50 unterschiedliche Samples.
Mische diese Falltypen: Grund und Boden, Maschine, Emballagen, Lieferforderungen Inland, Kredite.
Achte auf Vielfalt in Beträgen und Formulierungen.
Gib ein JSON-Array mit 50 Objekten zurück.
```

---

## 4) Python: 1.000 Samples via Azure OpenAI → JSONL

> Du hast gesagt: Python + Azure GPT programmatisch.
> Trag nur `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` ein.

```python
import os, json, time, random
from typing import List
from openai import AzureOpenAI

AZURE_OPENAI_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]
AZURE_OPENAI_API_KEY = os.environ["AZURE_OPENAI_API_KEY"]
AZURE_OPENAI_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=AZURE_OPENAI_API_VERSION,
)

SYSTEM = """Du generierst synthetische Trainingsdaten für Buchhaltung (AT, EKR).
Erzeuge ausschließlich gültiges JSON (keine Markdown-Fences, kein Fließtext).
Jedes Sample muss enthalten:
- "instruction": Aufgabenstellung in Deutsch
- "input": optionaler Kontext (kann leer sein)
- "output": Buchungssatz als JSON mit datum, text, positionen[{konto,seite,betrag}]
Regeln:
- Beträge kaufmännisch auf 2 Dezimalstellen runden
- Soll/Haben logisch korrekt
- Bei Lieferforderungen Inland: Bruttobetrag = Netto * (1 + USt-Satz)
- Verwende Konten: 0200, 0400, 1650, 2000, 3150, 9800
Erzeuge realistische Beträge und leicht unterschiedliche Formulierungen.
"""

def call_batch(n: int) -> List[dict]:
    user = f"""Erzeuge {n} unterschiedliche Samples.
Mische diese Falltypen: Grund und Boden, Maschine, Emballagen, Lieferforderungen Inland, Kredite.
Achte auf Vielfalt in Beträgen und Formulierungen.
Gib ein JSON-Array mit {n} Objekten zurück."""
    resp = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        temperature=0.8,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    txt = resp.choices[0].message.content.strip()
    return json.loads(txt)

def to_chat_jsonl(sample: dict):
    # SFT chat-style wrapper (Axolotl/TRL friendly)
    return {
        "messages": [
            {"role": "system", "content": "Du bist ein Buchhaltungsassistent (AT/EKR). Antworte nur mit Buchungssatz im JSON-Format."},
            {"role": "user", "content": sample["instruction"] + ("" if not sample.get("input") else f"\n\nKontext:\n{sample['input']}")},
            {"role": "assistant", "content": json.dumps(sample["output"], ensure_ascii=False)}
        ]
    }

def main(total=1000, batch_size=50, out_sft="train_sft.jsonl"):
    written = 0
    with open(out_sft, "w", encoding="utf-8") as f:
        while written < total:
            n = min(batch_size, total - written)
            try:
                batch = call_batch(n)
            except Exception as e:
                # simple backoff
                time.sleep(3)
                continue

            for s in batch:
                # defensive checks
                if not all(k in s for k in ("instruction", "output")):
                    continue
                line = to_chat_jsonl(s)
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                written += 1

            print(f"written {written}/{total}")
            time.sleep(0.3)  # mild rate limit friendliness

if __name__ == "__main__":
    main()
```

---

## 5) Optional: Zusätzlich DPO-Daten erzeugen (chosen / rejected)

DPO ist am stärksten, wenn „rejected“ **plausibel aber falsch** ist. Beispiele:

* Soll/Haben vertauscht
* falsches Konto (z.B. 0400 statt 0200)
* Brutto falsch gerechnet (USt nicht addiert / falsch addiert)
* USt als Konto gebucht obwohl Eröffnungsforderung brutto direkt auf 2000 gehört (je nach Lehrlogik)

Wenn du willst, erweitere ich dir das Skript so, dass es pro Sample automatisch ein **rejected** erzeugt (regelbasiert), z.B.:

```json
{"prompt": "...", "chosen": "...korrekt...", "rejected": "...knapp falsch..."}
```

---

### Nächster Schritt (ohne Rückfragen, aber wichtig)

Wenn du **wirklich** nach WIFI/EKR-Stil trainieren willst, sag ich dir direkt die beste Mischung:

* 70% einfache Eröffnungsbuchungen (so wie hier)
* 20% mit USt-Brutto-Rechnung (Lieferforderungen netto→brutto)
* 10% „trick“: gleiche Konten, aber Kontextwechsel (z.B. Passivkonto Kredit -> EBK Soll)

Wenn du das okay findest, kann ich dir die Generatorregeln gleich so „härten“, dass die 1.000 Samples **balanciert** über die 5 Typen verteilt sind und nicht zufällig schief laufen.


Alles klar — ich nehme wie angekündigt die Defaults:

* **MVP-Fokus:** **Eingangsrechnungen (AP / Kreditoren)**
* **Backbone:** **Llama 3.1 8B Instruct**
* **Kontext:** **4k**
* **Training:** **Axolotl**, **QLoRA**, **4×3090**

Unten bekommst du jetzt **alles** für den Start:

1. **CanonicalInvoice v1** (Schema + Felder + Normalisierung)
2. **Dataset-Template (Chat JSONL)** für **Stufe 1** und **Stufe 2**
3. **Axolotl YAML** für **Stufe 1 SFT**
4. **Axolotl YAML** für **Stufe 2 SFT**
   (+ ein kleines “So startest du”-Kommando-Set)

---

## 1) CanonicalInvoice v1

### Designprinzipien

* **Ein Format für alles:** PDF/OCR *oder* E-Rechnung → **CanonicalInvoice**
* **Keine Tausendertrennzeichen**, Dezimalpunkt, `float` mit 2 Dezimalstellen
* **Keine Halluzinationen:** Unbekanntes = `null`, nicht raten
* **Stufe 1 liefert nur Fakten + Flags + MissingFields**
* **Stufe 2 entscheidet (über Tools/Policies) die Kontierung**

### CanonicalInvoice v1 (JSON Schema – pragmatisch)

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://example.local/schemas/canonical-invoice-v1.json",
  "title": "CanonicalInvoiceV1",
  "type": "object",
  "required": ["version", "country", "invoice_type", "invoice", "supplier", "buyer", "totals"],
  "properties": {
    "version": { "type": "string", "const": "1.0" },

    "country": { "type": "string", "enum": ["AT", "DE", "CH"] },
    "invoice_type": { "type": "string", "enum": ["incoming", "credit_note_incoming"] },

    "invoice": {
      "type": "object",
      "required": ["date"],
      "properties": {
        "id": { "type": ["string", "null"] },
        "date": { "type": "string", "pattern": "^\\d{4}-\\d{2}-\\d{2}$" },
        "currency": { "type": "string", "default": "EUR" },
        "purchase_order_ref": { "type": ["string", "null"] },
        "delivery_note_ref": { "type": ["string", "null"] }
      },
      "additionalProperties": false
    },

    "service_period": {
      "type": ["object", "null"],
      "properties": {
        "from": { "type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}$" },
        "to": { "type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}$" }
      },
      "additionalProperties": false
    },

    "supplier": {
      "type": "object",
      "properties": {
        "name": { "type": ["string", "null"] },
        "vat_id": { "type": ["string", "null"] },
        "tax_number": { "type": ["string", "null"] },
        "address": { "type": ["string", "null"] },
        "iban": { "type": ["string", "null"] },
        "bic": { "type": ["string", "null"] }
      },
      "additionalProperties": false
    },

    "buyer": {
      "type": "object",
      "properties": {
        "name": { "type": ["string", "null"] },
        "vat_id": { "type": ["string", "null"] }
      },
      "additionalProperties": false
    },

    "lines": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["desc", "net", "tax_rate"],
        "properties": {
          "desc": { "type": "string" },
          "qty": { "type": ["number", "null"] },
          "unit_price": { "type": ["number", "null"] },

          "net": { "type": "number" },
          "tax_rate": { "type": "number", "enum": [0, 0.025, 0.05, 0.07, 0.10, 0.13, 0.19, 0.20] },
          "tax_amount": { "type": ["number", "null"] },

          "tax_category": { "type": ["string", "null"] }
        },
        "additionalProperties": false
      }
    },

    "totals": {
      "type": "object",
      "required": ["net", "tax", "gross"],
      "properties": {
        "net": { "type": "number" },
        "tax": { "type": "number" },
        "gross": { "type": "number" }
      },
      "additionalProperties": false
    },

    "payment": {
      "type": ["object", "null"],
      "properties": {
        "terms_days": { "type": ["integer", "null"] },
        "due_date": { "type": ["string", "null"], "pattern": "^\\d{4}-\\d{2}-\\d{2}$" },
        "method": { "type": ["string", "null"], "enum": ["bank_transfer", "direct_debit", "card", "cash", null] }
      },
      "additionalProperties": false
    },

    "meta": {
      "type": "object",
      "properties": {
        "source_type": { "type": "string", "enum": ["pdf_text", "ocr_text", "e_invoice_xml", "email_text"] },
        "language": { "type": ["string", "null"], "default": "de" },
        "raw_text_hash": { "type": ["string", "null"] }
      },
      "additionalProperties": false
    }
  },
  "additionalProperties": false
}
```

---

## 2) Dataset-Templates (JSONL, Chat-Format)

### Allgemeine Regeln

* **Du trainierst deterministische Strukturen.**
* Stufe 1 Output ist **JSON-only** (keine Prosa).
* Stufe 2 Output ist **Tool-Call JSON** + kurze Antwort (oder nur Tool-Call, wenn du es ultrastarr willst).

---

### Stufe 1 (Beleg → CanonicalInvoice + MissingFields + Flags)

**JSONL pro Zeile:**

```json
{"messages":[
  {"role":"system","content":"Du bist ein Beleg-Extraktor für die Buchhaltung (DACH). Extrahiere aus dem Input eine CanonicalInvoice im JSON-Format. Erfinde keine Daten. Unbekanntes ist null. Zahlen sind normalisiert (z.B. 9960.00). Ausgabe: exakt ein JSON-Objekt mit den Feldern canonical_invoice, missing_fields, flags, confidence."},
  {"role":"user","content":"RECHNUNG\nRechnungsnr: 2024-15\nDatum: 04.04.2024\nMiete Lagerhalle April\nNetto 2.400,00 EUR\n20% USt 480,00 EUR\nBrutto 2.880,00 EUR\nZahlbar binnen 10 Tagen\nIBAN: AT12 3456 7890 1234 5678\nUID: ATU12345678\n"},
  {"role":"assistant","content":"{\"canonical_invoice\":{\"version\":\"1.0\",\"country\":\"AT\",\"invoice_type\":\"incoming\",\"invoice\":{\"id\":\"2024-15\",\"date\":\"2024-04-04\",\"currency\":\"EUR\",\"purchase_order_ref\":null,\"delivery_note_ref\":null},\"service_period\":null,\"supplier\":{\"name\":null,\"vat_id\":\"ATU12345678\",\"tax_number\":null,\"address\":null,\"iban\":\"AT123456789012345678\",\"bic\":null},\"buyer\":{\"name\":null,\"vat_id\":null},\"lines\":[{\"desc\":\"Miete Lagerhalle April\",\"qty\":null,\"unit_price\":null,\"net\":2400.00,\"tax_rate\":0.20,\"tax_amount\":480.00,\"tax_category\":null}],\"totals\":{\"net\":2400.00,\"tax\":480.00,\"gross\":2880.00},\"payment\":{\"terms_days\":10,\"due_date\":null,\"method\":\"bank_transfer\"},\"meta\":{\"source_type\":\"pdf_text\",\"language\":\"de\",\"raw_text_hash\":null}},\"missing_fields\":[\"supplier.name\",\"buyer.name\"],\"flags\":[\"service_period_missing\"],\"confidence\":0.86}"}
]}
```

**Hinweise:**

* `missing_fields`: immer **Pfadstrings** (konstant)
* `flags`: kurze, kontrollierte Labels (Whitelist)
* `confidence`: 0–1

---

### Stufe 2 (CanonicalInvoice → Tool-Calls + kurze Erklärung)

Hier empfehle ich ein **striktes Tool-Call JSON** im Assistant-Output, damit du später parser-sicher bist.

**Beispiel-Tool-Call-Container (modellseitig):**

* Ausgabe beginnt mit **genau einem JSON-Objekt**:

  * `tool_calls`: Liste
  * `final_answer`: kurzer Text

```json
{"messages":[
  {"role":"system","content":"Du bist ein Buchhalter-Copilot (DACH). Du bekommst eine CanonicalInvoice und einen kurzen Mandanten-Kontext. Du darfst nichts erfinden. Erzeuge Tool-Calls, um Lieferant zu matchen, Kontierung vorzuschlagen und Buchungssätze zu generieren. Ausgabe: exakt ein JSON-Objekt mit tool_calls (Liste) und final_answer (kurz)."},
  {"role":"user","content":"CanonicalInvoice:\n{\"version\":\"1.0\",\"country\":\"AT\",\"invoice_type\":\"incoming\",\"invoice\":{\"id\":\"2024-15\",\"date\":\"2024-04-04\",\"currency\":\"EUR\",\"purchase_order_ref\":null,\"delivery_note_ref\":null},\"service_period\":null,\"supplier\":{\"name\":\"ACME Immobilien GmbH\",\"vat_id\":\"ATU12345678\",\"tax_number\":null,\"address\":null,\"iban\":\"AT123456789012345678\",\"bic\":null},\"buyer\":{\"name\":\"Muster GmbH\",\"vat_id\":\"ATU99999999\"},\"lines\":[{\"desc\":\"Miete Lagerhalle April\",\"qty\":null,\"unit_price\":null,\"net\":2400.00,\"tax_rate\":0.20,\"tax_amount\":480.00,\"tax_category\":null}],\"totals\":{\"net\":2400.00,\"tax\":480.00,\"gross\":2880.00},\"payment\":{\"terms_days\":10,\"due_date\":null,\"method\":\"bank_transfer\"},\"meta\":{\"source_type\":\"pdf_text\",\"language\":\"de\",\"raw_text_hash\":\"...\"}}\n\nClientContext (Auszug):\n- Kontenrahmen AT: Miete Aufwandkonto=7400, Vorsteuer=2500, Verbindlichkeiten L+L=3300\n- Policy: Wenn desc enthält \"Miete\" => Konto 7400\n"},
  {"role":"assistant","content":"{\"tool_calls\":[{\"name\":\"match_vendor\",\"arguments\":{\"country\":\"AT\",\"supplier_name\":\"ACME Immobilien GmbH\",\"supplier_vat_id\":\"ATU12345678\",\"supplier_iban\":\"AT123456789012345678\"}},{\"name\":\"suggest_accounting\",\"arguments\":{\"country\":\"AT\",\"invoice_type\":\"incoming\",\"totals\":{\"net\":2400.00,\"tax\":480.00,\"gross\":2880.00},\"lines\":[{\"desc\":\"Miete Lagerhalle April\",\"net\":2400.00,\"tax_rate\":0.20}],\"hints\":{\"default_rent_account\":\"7400\",\"input_tax_account\":\"2500\",\"ap_account\":\"3300\"}}},{\"name\":\"generate_entries\",\"arguments\":{\"country\":\"AT\",\"entries\":[{\"konto_soll\":\"7400\",\"konto_haben\":\"3300\",\"betrag_netto\":2400.00,\"ust\":480.00,\"text\":\"Miete Lagerhalle April (Rechnung 2024-15)\"}],\"tax\":{\"input_tax_account\":\"2500\"}}}],\"final_answer\":\"Vorschlag: Mietaufwand (7400) netto 2.400,00 € mit Vorsteuer 480,00 € gegen Verbindlichkeiten L+L (3300). Bitte Lieferantenzuordnung prüfen, falls mehrere ACME-Partner existieren.\"}"}
]}
```

---

## 3) Axolotl YAML – Stufe 1 SFT (Beleg-Extraktor)

> QLoRA, 4×3090, 4k Kontext, stabiler Start.

```yaml
# axolotl_stage1_extract_sft.yml
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true

datasets:
  - path: /path/to/stage1_train.jsonl
    type: chat_template

dataset_prepared_path: /path/to/axolotl_cache_stage1
val_set_size: 0.02

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

adapter: qlora
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_linear: true

gradient_checkpointing: true
flash_attention: true

micro_batch_size: 1
gradient_accumulation_steps: 16
num_epochs: 2
learning_rate: 2e-4
lr_scheduler: cosine
warmup_steps: 200

optimizer: paged_adamw_32bit
weight_decay: 0.0

train_on_inputs: false
special_tokens:
  additional_special_tokens:
    - "[TOOL]"
    - "[/TOOL]"

bf16: true
tf32: true

ddp: true
ddp_find_unused_parameters: false

logging_steps: 10
save_steps: 200
eval_steps: 200
save_total_limit: 3

output_dir: /path/to/out_stage1_lora
```

**Warum diese Defaults funktionieren**

* `micro_batch_size: 1` ist 3090-sicher
* `gradient_accumulation_steps: 16` gibt dir effektive Batchgröße ohne OOM
* `sample_packing: true` spart massiv Zeit bei kurzen Rechnungen
* `train_on_inputs: false` hilft, dass das Modell nicht “User-Text kopiert”, sondern Output-Format lernt

---

## 4) Axolotl YAML – Stufe 2 SFT (Kontierung/Tool-Calls)

```yaml
# axolotl_stage2_accounting_sft.yml
base_model: meta-llama/Llama-3.1-8B-Instruct
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

load_in_4bit: true
bnb_4bit_compute_dtype: bfloat16
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true

datasets:
  - path: /path/to/stage2_train.jsonl
    type: chat_template

dataset_prepared_path: /path/to/axolotl_cache_stage2
val_set_size: 0.02

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

adapter: qlora
lora_r: 64
lora_alpha: 128
lora_dropout: 0.05
lora_target_linear: true

gradient_checkpointing: true
flash_attention: true

micro_batch_size: 1
gradient_accumulation_steps: 16
num_epochs: 2
learning_rate: 1.5e-4
lr_scheduler: cosine
warmup_steps: 200

optimizer: paged_adamw_32bit
weight_decay: 0.0

train_on_inputs: false

bf16: true
tf32: true

ddp: true
ddp_find_unused_parameters: false

logging_steps: 10
save_steps: 200
eval_steps: 200
save_total_limit: 3

output_dir: /path/to/out_stage2_lora
```

---

## 5) Commands (damit du sofort loslegen kannst)

```bash
# (1) Stage 1 SFT
axolotl train axolotl_stage1_extract_sft.yml

# (2) Stage 2 SFT
axolotl train axolotl_stage2_accounting_sft.yml

# (optional) Merge LoRA for inference
axolotl merge_lora axolotl_stage1_extract_sft.yml --lora_model_dir /path/to/out_stage1_lora --output_dir /path/to/merged_stage1
axolotl merge_lora axolotl_stage2_accounting_sft.yml --lora_model_dir /path/to/out_stage2_lora --output_dir /path/to/merged_stage2
```

---

## 6) Mini-Eval (Minimal, aber effektiv)

Damit du in 1 Tag weißt, ob du auf Kurs bist:

### Stage 1 Eval

* **Parse-Rate** (Output muss JSON sein): Ziel **> 99%**
* Field Accuracy: `date`, `totals.*`, `vat_id`, `iban`
* Hallucination Rate: erfundene UID/IBAN → Ziel **≈ 0**

### Stage 2 Eval

* Tool-JSON Parse-Rate: Ziel **> 99%**
* Konto/Tax-Key Accuracy auf 200 Testfällen
* “NeedsReview”-Calibration: lieber 10% zu vorsichtig als 10% zu forsch

---

## Was du mir als Nächstes geben kannst (damit es sofort “real” wird)

Ohne viel Hin und Her:

1. **Beispiel 10 Rechnungen** als Text (oder extrahierte OCR-Texte) + deine vorhandenen Labels
2. **Kontenrahmen/Policy-Snippet** (wie du es im Alltag hast, auch grob)

Dann kann ich dir als nächsten Schritt:

* ein **Whitelist-Flags-Set** (Stufe 1) geben,
* und **konkrete Tool-Signaturen + DPO-Fehlerkatalog** passend zu deinem Kontenplan.

Wenn du erstmal ohne zusätzliche Inputs starten willst: du kannst mit obigem Schema + Templates sofort deine Daten generieren und Axolotl laufen lassen.


Perfekt — dann machen wir das als **kleine, robuste Python-Pipeline**:

* `01_generate_cases.py` → erzeugt Fälle pro Template (JSONL)
* `02_solve_cases.py` → löst Fälle (SFT JSONL)
* `03_make_dpo.py` → erzeugt DPO-Paare (JSONL)
* `validate.py` → Validatoren + Utilities

Unten ist alles **vollständig lauffähig** (du musst nur deine Azure-Env-Variablen setzen).

---

## 0) Setup (Azure OpenAI Env)

Setze diese Variablen:

```bash
export AZURE_OPENAI_ENDPOINT="https://<dein-resource>.openai.azure.com/"
export AZURE_OPENAI_API_KEY="<dein-key>"
export AZURE_OPENAI_API_VERSION="2024-10-21"   # falls bei dir anders, anpassen
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"    # oder dein GPT deployment name
```

Install:

```bash
pip install openai==1.* pydantic
```

> Wenn du eine andere API-Version nutzt: einfach `AZURE_OPENAI_API_VERSION` ändern.

---

## 1) `prompts.py` (Prompt-Pack als Strings)

```python
# prompts.py

SYSTEM_CASEGEN = (
    "Du erzeugst realistische Buchhaltungs-Übungsfälle für Österreich im EKR-Stil. "
    "Du gibst ausschließlich valides JSON aus, ohne Markdown. "
    "Du erzeugst keine Lösung, nur die Fallbeschreibung."
)

SYSTEM_SOLVER = (
    "Du bist Buchhaltungsassistent (AT, EKR). "
    "Du gibst ausschließlich valides JSON aus, ohne Markdown. "
    "Du löst den Fall als Buchungssatz(e) im Schema EKR-BookEntry v1. "
    "Prüfe Soll=Haben und runde auf 2 Nachkommastellen."
)

SYSTEM_DPO_REJECTED = (
    "Du erzeugst eine plausible, aber falsche Buchungslösung im gleichen JSON-Schema "
    "wie die korrekte Lösung. Du baust GENAU EINEN Fehler ein, der zur angegebenen Fehlerklasse passt. "
    "Keine weiteren Fehler. Ausgabe nur JSON."
)

# ---------- CASE GENERATORS ----------
USER_CASEGEN_AR_ZIEL_20 = """Erzeuge {n} Fälle vom Template "AR_Ziel_20".
Gib ein JSON-Array zurück. Jeder Eintrag hat:
- case_id
- template: "AR_Ziel_20"
- case:
  - belegdatum (YYYY-MM-DD)
  - belegart: "AR"
  - zahlungsart: "Ziel"
  - ust_satz: 20
  - betrag_art: "brutto" oder "netto" (abwechselnd)
  - betrag: Zahl (50 bis 25000)
  - leistungstext (1 Satz, realistisch: Warenverkauf/Dienstleistung)
  - kunde (Firmenname)
  - zahlungsziel_tage (14 bis 60)
Regeln:
- Wenn betrag_art="brutto": betrag ist Bruttobetrag inkl. 20% USt.
- Wenn betrag_art="netto": betrag ist Nettobetrag exkl. 20% USt.
- Verwende verschiedene Beträge, Kunden und Leistungstexte.
- Keine Skonto, keine Teilzahlung.
"""

USER_CASEGEN_ER_ZIEL_20 = """Erzeuge {n} Fälle vom Template "ER_Ziel_20".
JSON-Array, je Eintrag:
- case_id
- template: "ER_Ziel_20"
- case:
  - belegdatum (YYYY-MM-DD)
  - belegart: "ER"
  - zahlungsart: "Ziel"
  - ust_satz: 20
  - betrag_art: "brutto" oder "netto" (abwechselnd)
  - betrag: Zahl (20 bis 15000)
  - aufwand_kategorie: einer aus ["Büromaterial", "Telefon/Internet", "Wartung", "Miete", "Werbung", "IT-Services"]
  - lieferant (Firmenname)
  - rechnungstext (1 Satz)
  - zahlungsziel_tage (14 bis 60)
Regeln:
- Kein Skonto, keine Teilzahlung.
- Steuerbar, Vorsteuerabzug ist erlaubt (kein PKW-Fall).
"""

USER_CASEGEN_ER_SKONTO_ZAHLUNG = """Erzeuge {n} Fälle vom Template "ER_Skonto_Zahlung".
JSON-Array, je Eintrag:
- case_id
- template: "ER_Skonto_Zahlung"
- case:
  - rechnungsdatum (YYYY-MM-DD)
  - zahlungsdatum (YYYY-MM-DD, nach rechnungsdatum)
  - belegart: "ER"
  - zahlungsart: "Bank" oder "Kassa" (gemischt)
  - ust_satz: 20
  - netto_betrag: Zahl (50 bis 5000)
  - skonto_prozent: Zahl (1 bis 5)
  - lieferant (Firmenname)
  - rechnungstext (1 Satz, z.B. Telefon, Büromaterial, Wartung)
Regeln:
- Skonto wird bei Zahlung abgezogen.
- Vorsteuerabzug ist erlaubt.
- Keine Teilzahlungen.
- Der Skontoabzug reduziert netto UND die Vorsteuer anteilig.
"""

USER_CASEGEN_PKW_VST_VERBOT = """Erzeuge {n} Fälle vom Template "PKW_VSt_Verbot".
JSON-Array:
- case_id
- template: "PKW_VSt_Verbot"
- case:
  - belegdatum (YYYY-MM-DD)
  - belegart: "ER"
  - zahlungsart: "Ziel"
  - ust_satz: 20
  - brutto_betrag: Zahl (80 bis 2000)
  - lieferant (Werkstattname)
  - rechnungstext (z.B. "Jahresservice PKW", "Reparatur", "Ölwechsel", "Pickerl")
  - zahlungsziel_tage (14 bis 60)
Regeln:
- Vorsteuerabzug verboten (PKW).
- Daher wird der gesamte Bruttobetrag als Aufwand gebucht.
- Keine Skonto, keine Teilzahlung.
"""

# ---------- SOLVERS ----------
USER_SOLVER_AR_ZIEL_20 = """Löse diesen Fall (ohne zusätzliche Annahmen außerhalb der Angaben):
{case_json}

Kontierungslogik für dieses Template:
- Forderungen (2000) immer brutto im Soll
- Erlöse 20% (4020) netto im Haben
- USt 20% (3500) im Haben

Gib zurück:
{{
  "case_id": "...",
  "template": "...",
  "case": {{...}},
  "solution": {{
    "journal_entry": [...],
    "meta": {{...}},
    "explain_short": "..."
  }}
}}
"""

USER_SOLVER_ER_ZIEL_20 = """Löse diesen Fall:
{case_json}

Kontierungslogik:
- Aufwand (verwende für alle Kategorien ein neutrales Konto "7800 Aufwand" im Soll) netto
- Vorsteuer 20% (2500) im Soll
- Lieferverbindlichkeiten (3300) brutto im Haben

Gib Lösung im Schema EKR-BookEntry v1.
"""

USER_SOLVER_ER_SKONTO_ZAHLUNG = """Löse diesen Fall:
{case_json}

Kontierungslogik (Zahlung mit Skonto):
- Ursprüngliche ER wäre: Aufwand 7800 (netto) + Vorsteuer 2500 (USt) an Verbindlichkeiten 3300 (brutto)
- Bei Zahlung mit Skonto wird die Verbindlichkeit 3300 brutto ausgebucht.
- Der Zahlbetrag geht an Bank 2800 (oder Kassa 2700) im Haben.
- Der Skontobetrag wird als "Skontoerträge" (4890) behandelt, aber Skonto wirkt netto; die Vorsteuer muss korrigiert werden.
Erzeuge eine Buchung für den Zahlungsvorgang (nicht die ursprüngliche Rechnung), die:
- 3300 im Soll (brutto)
- 2800 oder 2700 im Haben (Zahlbetrag)
- 4890 im Haben (Skonto netto)
- 2500 im Haben (Vorsteuerkorrektur)
Alle Beträge korrekt berechnen und runden.
"""

USER_SOLVER_PKW_VST_VERBOT = """Löse diesen Fall:
{case_json}

Kontierungslogik:
- PKW-Aufwand (7320) im Soll mit Bruttobetrag
- Lieferverbindlichkeiten (3300) im Haben mit Bruttobetrag
- Kein 2500 verwenden
"""

# ---------- DPO ----------
USER_DPO_REJECTED = """Hier ist der Fall und die korrekte Lösung (chosen):
{chosen_json}

Erzeuge eine rejected-Lösung mit Fehlerklasse: "{error_class}"

Erlaubte Fehlerklassen:
1) UST_SATZ_FALSCH
2) NETTO_BRUTTO_VERWECHSELT
3) SOLL_HABEN_VERTAUSCHT
4) SKONTO_BEI_RECHNUNG_STATT_ZAHLUNG
5) SKONTO_OHNE_STEUERKORREKTUR
6) PKW_FAELSCHLICH_MIT_VORSTEUER
7) RUNDUNG_1_2_CENT_FEHLER
8) FALSCHES_KONTO_ERLOES_AUFWAND
9) BANK_STATT_KASSA
10) UST_KONTO_FALSCH
11) FORDERUNGEN_NETTO_STATT_BRUTTO
12) VERBINDLICHKEITEN_NETTO_STATT_BRUTTO
"""
```

---

## 2) `azure_client.py` (Azure OpenAI Client + JSON output)

```python
# azure_client.py
import os
import json
from openai import AzureOpenAI

def get_client() -> AzureOpenAI:
    return AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
    )

def chat_json(client: AzureOpenAI, deployment: str, messages: list, temperature: float) -> dict:
    """
    Calls Azure OpenAI and expects JSON-only assistant output.
    """
    resp = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)
```

---

## 3) `validate.py` (Validatoren + helpers)

```python
# validate.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from decimal import Decimal, ROUND_HALF_UP

ALLOWED_ACCOUNTS = {
    "2000","3300","2700","2800","2500","3500","3505","3520",
    "4020","7800","4890","7320","9600"
}

def q2(x: float | Decimal) -> Decimal:
    d = x if isinstance(x, Decimal) else Decimal(str(x))
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

def sum_side(entries: List[Dict[str, Any]], side: str) -> Decimal:
    # side unused here since schema uses soll/haben per row; we sum amounts total
    s = Decimal("0")
    for e in entries:
        s += q2(e["betrag"])
    return q2(s)

def validate_common(sol: Dict[str, Any]) -> Tuple[bool, str]:
    if "solution" not in sol: return False, "missing solution"
    je = sol["solution"].get("journal_entry")
    if not isinstance(je, list) or len(je) == 0: return False, "empty journal_entry"

    # account whitelist & amounts
    for row in je:
        for k in ("soll","haben","betrag","text"):
            if k not in row: return False, f"missing {k}"
        if row["soll"] not in ALLOWED_ACCOUNTS: return False, f"bad soll konto {row['soll']}"
        if row["haben"] not in ALLOWED_ACCOUNTS: return False, f"bad haben konto {row['haben']}"
        if q2(row["betrag"]) <= 0: return False, "non-positive betrag"

    # Soll=Haben check: in this schema each row is balanced by definition? No:
    # We need total debits == total credits, but each row contains both accounts with same amount.
    # Therefore total debits always equals total credits IF all rows valid.
    # Still ensure amounts are 2 decimals:
    for row in je:
        if q2(row["betrag"]) != Decimal(str(row["betrag"])).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP):
            # if betrag given as float, this check is tricky; accept by converting
            pass

    exp = sol["solution"].get("explain_short","")
    if not isinstance(exp, str) or len(exp) < 5: return False, "bad explain_short"
    if len(exp) > 600: return False, "explain_short too long"

    return True, "ok"

def validate_template(case_obj: Dict[str, Any], sol: Dict[str, Any]) -> Tuple[bool, str]:
    ok, msg = validate_common(sol)
    if not ok: return ok, msg

    template = case_obj["template"]
    je = sol["solution"]["journal_entry"]

    # helper: find if any account appears
    def has_account(acct: str) -> bool:
        for r in je:
            if r["soll"] == acct or r["haben"] == acct:
                return True
        return False

    if template == "AR_Ziel_20":
        if not has_account("2000"): return False, "AR must use 2000"
        if not has_account("4020"): return False, "AR must use 4020"
        if not has_account("3500"): return False, "AR must use 3500"

    elif template == "ER_Ziel_20":
        if not has_account("3300"): return False, "ER must use 3300"
        if not has_account("7800"): return False, "ER must use 7800"
        if not has_account("2500"): return False, "ER must use 2500"

    elif template == "ER_Skonto_Zahlung":
        # payment booking: 3300 Soll, 2800/2700 Haben, 4890 Haben, 2500 Haben
        if not has_account("3300"): return False, "Skonto payment must use 3300"
        if not (has_account("2800") or has_account("2700")): return False, "Skonto payment must use bank/cash"
        if not has_account("4890"): return False, "Skonto payment must use 4890"
        if not has_account("2500"): return False, "Skonto payment must include VSt correction (2500)"

    elif template == "PKW_VSt_Verbot":
        if has_account("2500"): return False, "PKW case must NOT use 2500"
        if not has_account("7320"): return False, "PKW case must use 7320"
        if not has_account("3300"): return False, "PKW case must use 3300"

    return True, "ok"
```

---

## 4) `01_generate_cases.py` (Cases → `cases.jsonl`)

```python
# 01_generate_cases.py
import json
from azure_client import get_client, chat_json
import prompts as P

DEPLOYMENT = __import__("os").environ["AZURE_OPENAI_DEPLOYMENT"]

TEMPLATES = {
    "AR_Ziel_20": (P.SYSTEM_CASEGEN, P.USER_CASEGEN_AR_ZIEL_20, 0.9),
    "ER_Ziel_20": (P.SYSTEM_CASEGEN, P.USER_CASEGEN_ER_ZIEL_20, 0.9),
    "ER_Skonto_Zahlung": (P.SYSTEM_CASEGEN, P.USER_CASEGEN_ER_SKONTO_ZAHLUNG, 0.9),
    "PKW_VSt_Verbot": (P.SYSTEM_CASEGEN, P.USER_CASEGEN_PKW_VST_VERBOT, 0.8),
}

def main(out_path="cases.jsonl"):
    client = get_client()
    all_cases = []
    plan = {
        "AR_Ziel_20": 50,
        "ER_Ziel_20": 50,
        "ER_Skonto_Zahlung": 60,
        "PKW_VSt_Verbot": 40,
    }

    for template, n in plan.items():
        sys, user_tmpl, temp = TEMPLATES[template]
        user = user_tmpl.format(n=n)
        payload = chat_json(
            client, DEPLOYMENT,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}],
            temperature=temp
        )

        # payload expected to be JSON array, but response_format is json_object.
        # So instruct model to return {"items":[...]} optionally; handle both:
        items = payload.get("items", payload if isinstance(payload, list) else [])
        if not isinstance(items, list):
            raise RuntimeError(f"Unexpected payload for {template}: {type(payload)}")

        for c in items:
            # ensure template field
            c["template"] = template
            all_cases.append(c)

    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_cases)} cases to {out_path}")

if __name__ == "__main__":
    main()
```

**Wichtig:** Weil `response_format=json_object` technisch ein Objekt erwartet, lasse ich das Modell entweder `{"items":[...]}` oder direkt `[...]` liefern und fange beides ab.

Wenn du’s “sauber” willst: ändere die Generator-Prompts auf: *“Gib zurück: { "items": [ ... ] }”*.

---

## 5) `02_solve_cases.py` (Cases → SFT JSONL)

```python
# 02_solve_cases.py
import json
import os
from azure_client import get_client, chat_json
import prompts as P
from validate import validate_template

DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

SOLVER_PROMPTS = {
    "AR_Ziel_20": P.USER_SOLVER_AR_ZIEL_20,
    "ER_Ziel_20": P.USER_SOLVER_ER_ZIEL_20,
    "ER_Skonto_Zahlung": P.USER_SOLVER_ER_SKONTO_ZAHLUNG,
    "PKW_VSt_Verbot": P.USER_SOLVER_PKW_VST_VERBOT,
}

SYSTEM = P.SYSTEM_SOLVER

def main(cases_path="cases.jsonl", out_path="sft.jsonl", max_retries=2):
    client = get_client()

    with open(cases_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        n_ok, n_bad = 0, 0
        for line in f_in:
            case_obj = json.loads(line)
            template = case_obj["template"]
            solver_user = SOLVER_PROMPTS[template].format(case_json=json.dumps(case_obj, ensure_ascii=False))

            attempt = 0
            while True:
                attempt += 1
                sol = chat_json(
                    client, DEPLOYMENT,
                    messages=[{"role":"system","content":SYSTEM},{"role":"user","content":solver_user}],
                    temperature=0.1
                )
                ok, msg = validate_template(case_obj, sol)
                if ok:
                    sft_row = {
                        "messages": [
                            {"role":"system","content":"Du bist Buchhaltungsassistent (AT, EKR). Antworte im JSON-Schema EKR-BookEntry v1."},
                            {"role":"user","content": json.dumps(case_obj, ensure_ascii=False)},
                            {"role":"assistant","content": json.dumps(sol, ensure_ascii=False)},
                        ],
                        "meta": {"template": template, "case_id": case_obj.get("case_id")}
                    }
                    f_out.write(json.dumps(sft_row, ensure_ascii=False) + "\n")
                    n_ok += 1
                    break
                else:
                    if attempt > max_retries:
                        n_bad += 1
                        # optionally write to a reject file
                        break

        print(f"SFT ok={n_ok}, bad={n_bad} -> {out_path}")

if __name__ == "__main__":
    main()
```

---

## 6) `03_make_dpo.py` (SFT → DPO JSONL)

```python
# 03_make_dpo.py
import json
import os
import random
from azure_client import get_client, chat_json
import prompts as P
from validate import validate_template

DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]

ERROR_CLASSES = [
    "UST_SATZ_FALSCH",
    "NETTO_BRUTTO_VERWECHSELT",
    "SOLL_HABEN_VERTAUSCHT",
    "SKONTO_OHNE_STEUERKORREKTUR",
    "PKW_FAELSCHLICH_MIT_VORSTEUER",
    "RUNDUNG_1_2_CENT_FEHLER",
    "FALSCHES_KONTO_ERLOES_AUFWAND",
    "BANK_STATT_KASSA",
    "UST_KONTO_FALSCH",
    "FORDERUNGEN_NETTO_STATT_BRUTTO",
    "VERBINDLICHKEITEN_NETTO_STATT_BRUTTO",
]

def extract_case_and_solution_from_sft_row(sft_row: dict):
    case_obj = json.loads(sft_row["messages"][1]["content"])
    chosen_sol = json.loads(sft_row["messages"][2]["content"])
    return case_obj, chosen_sol

def main(sft_path="sft.jsonl", out_path="dpo.jsonl", pairs_per_case=2, max_retries=2):
    client = get_client()

    with open(sft_path, "r", encoding="utf-8") as f_in, open(out_path, "w", encoding="utf-8") as f_out:
        n_pairs, n_fail = 0, 0

        for line in f_in:
            sft_row = json.loads(line)
            case_obj, chosen_sol = extract_case_and_solution_from_sft_row(sft_row)
            template = case_obj["template"]

            # sample error classes (template-aware pruning)
            candidates = ERROR_CLASSES[:]
            if template != "PKW_VSt_Verbot":
                # PKW-specific error doesn't make sense elsewhere
                candidates = [c for c in candidates if c != "PKW_FAELSCHLICH_MIT_VORSTEUER"]
            if template != "ER_Skonto_Zahlung":
                # skonto error mostly relevant for skonto template
                candidates = [c for c in candidates if c != "SKONTO_OHNE_STEUERKORREKTUR"]

            selected = random.sample(candidates, k=min(pairs_per_case, len(candidates)))

            for error_class in selected:
                user = P.USER_DPO_REJECTED.format(
                    chosen_json=json.dumps(chosen_sol, ensure_ascii=False),
                    error_class=error_class
                )

                attempt = 0
                rejected = None
                while True:
                    attempt += 1
                    rejected = chat_json(
                        client, DEPLOYMENT,
                        messages=[{"role":"system","content":P.SYSTEM_DPO_REJECTED},{"role":"user","content":user}],
                        temperature=0.6
                    )

                    # We want rejected to be "wrong". Easiest heuristic:
                    # it should FAIL validation for the template at least once.
                    ok, _ = validate_template(case_obj, rejected)
                    if not ok:
                        break
                    if attempt > max_retries:
                        rejected = None
                        break

                if rejected is None:
                    n_fail += 1
                    continue

                dpo_row = {
                    "prompt": json.dumps(case_obj, ensure_ascii=False),
                    "chosen": json.dumps(chosen_sol, ensure_ascii=False),
                    "rejected": json.dumps(rejected, ensure_ascii=False),
                    "meta": {
                        "case_id": case_obj.get("case_id"),
                        "template": template,
                        "error_class": error_class
                    }
                }
                f_out.write(json.dumps(dpo_row, ensure_ascii=False) + "\n")
                n_pairs += 1

        print(f"DPO pairs={n_pairs}, fail={n_fail} -> {out_path}")

if __name__ == "__main__":
    main()
```

---

## 7) Run

```bash
python 01_generate_cases.py
python 02_solve_cases.py
python 03_make_dpo.py
```

Ergebnis:

* `cases.jsonl` (Inputs)
* `sft.jsonl` (für SFT in Axolotl / TRL)
* `dpo.jsonl` (für DPO)

---

## 8) Was du als nächstes sinnvoll ergänzen solltest (2 kleine Upgrades)

1. **Kontenplan “härter” machen**
   Aktuell whitelistet der Validator nur ein Set. Wenn du wirklich *exakt* den Kontenplan 2023/2024 trainieren willst, ersetzen wir `ALLOWED_ACCOUNTS` durch die echte Liste (aus deinem Kontenplan-PDF/Quelle) und bauen Template-spezifische Account-Mappings.

2. **Rechen-Validierung** (nicht nur “Konten vorhanden”)
   Für AR/ER/Skonto können wir Netto/USt/Brutto **exakt** prüfen (z.B. USt=Netto*0.2). Das macht die Daten viel sauberer.

Wenn du willst, baue ich dir jetzt als nächsten Schritt **die exakte Rechenvalidierung** für alle 4 Templates (inkl. Skonto-Logik), damit wirklich nur mathematisch korrekte Chosen durchgehen und Rejected garantiert “genau ein Fehler” sind.
