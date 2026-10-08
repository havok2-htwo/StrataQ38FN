# Einrichtung

Diese Anleitung baut die Umgebung auf einem frischen Windows-PC nach. Sie dauert, je nach Leitung, einige Stunden,
weil die Modelle zusammen rund 260 GB groß sind.

## 0. Voraussetzungen

| | Minimum | getestet |
|---|---|---|
| Betriebssystem | Windows 10/11, 64 bit | Windows 11 Pro |
| GPU | NVIDIA mit 24 GB+ VRAM (die Configs sind auf 32 GB abgestimmt) | RTX 5090, Treiber 610.88 |
| RAM | 64 GB | 64 GB |
| SSD | schnelle NVMe, ~300 GB frei für Modelle + ~60 GB für Packs | WD SN850X |
| Software | Git, aktueller NVIDIA-Treiber; Python 3.12 installiert Strata bei Bedarf selbst | Python 3.12.4 |

> **Pfade:** Die Configs in `strata-configs/` enthalten absolute Pfade
> (`X:\dev\StrataQ38FN\...` und `C:\StrataQ38FN-models\...`). Wer andere Pfade nimmt, passt sie in Schritt 5 an.
> Die Start-Skripte selbst arbeiten relativ zu ihrem Ordner.

## 1. Repo klonen

```powershell
git clone https://github.com/havok2-htwo/StrataQ38FN.git X:\dev\StrataQ38FN
cd X:\dev\StrataQ38FN
```

## 2. Strata-Engine klonen

Die Engine liegt als eigener Clone im Unterordner `Strata\` (ist in `.gitignore`):

```powershell
git clone https://github.com/Niko1221/Strata.git Strata
git -C Strata checkout 82f46a8    # getesteter Stand; neuere Stände funktionieren meist auch
```

## 3. Strata-Setup für jedes Modell

`Strata\START-HERE.bat` ausführen. Beim ersten Lauf legt es `Strata\.venv` an, installiert die Python-Pakete, baut
bzw. lädt die Engine, lädt das Modell, baut das Pack unter `Strata-data\packs\` und den MTP-Draft unter
`Strata-data\mtp\`.

Für die weiteren Modelle `Strata\SETUP.bat` ausführen (= `START-HERE.bat --setup`). Auswahl im Setup:

| Durchlauf | Modell | Größe | Kontext im Setup | Download |
|---|---|---|---|---|
| 1 | Qwen3.8-Flash-Next (Unsloth) | UD-IQ4_XS | beliebig (wird in Schritt 5 überschrieben) | 94 GB |
| 2 | Qwen3.8-Flash-Next (Unsloth) | UD-Q4_K_XL | beliebig | 111 GB |
| 3 | Qwen3.8-Flash-Next Coder | IQ1_M | beliebig | 58 GB |

Bilder: aus. Parallelität: eine Anfrage nach der anderen (Batching ist auf dieser Konfiguration kaputt bzw. langsamer).

Brauchst du nur das empfohlene Modell, reicht Durchlauf 1; die beiden anderen Start-Skripte funktionieren dann nicht.

Alternativ lädt `download_q4kxl.py` die UD-Q4_K_XL-Shards wiederaufnehmbar und mit SHA-256-Prüfung (gepinnte
Revision), `download_hf.py REPO REVISION ZIELORDNER` lädt ein beliebiges Hugging-Face-Repo.

## 4. GGUF-Dateien auf die schnellste NVMe legen

Die Engine liest beim Prompt-Lesen Teile der GGUF direkt von der SSD. Auf einer langsameren SSD (hier: Crucial P3 auf
`X:`) mit `--ple-io mmap` kam es zu Hängern ("stalls" bei "layer 1 of the prompt chunk"). Seit die Modelle auf der
schnellsten NVMe liegen und `--ple-io direct` gesetzt ist, gibt es keine mehr.

Die Shards je Modell in diese Ordner verschieben:

```
C:\StrataQ38FN-models\unsloth-UD-IQ4_XS\    Qwen3.8-Flash-Next-UD-IQ4_XS-0000{1..3}-of-00003.gguf
C:\StrataQ38FN-models\unsloth-UD-Q4_K_XL\   Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{1..4}-of-00004.gguf
C:\StrataQ38FN-models\coder-IQ1_M\          Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-0000{1..2}-of-00002.gguf
```

## 5. Getunte Configs einspielen

Das Setup hat `Strata\strata-<modell>.json` geschrieben. Diese sichern und durch die Vorlagen ersetzen:

```powershell
foreach ($f in Get-ChildItem strata-configs\strata-*.json -Exclude *setup-original*) {
  $dst = "Strata\$($f.Name)"
  if (Test-Path $dst) { Copy-Item $dst "$dst.orig" }
  Copy-Item $f.FullName $dst
}
```

Danach in jeder `Strata\strata-*.json` prüfen und ggf. anpassen:

- `exe`, `cwd`, `tokenizer`, `log`, `lib_dirs`, `--pack`, `--mtp`, `--expert-profile`: Pfade unter `X:\dev\StrataQ38FN`
- `--native` / `--ple-gguf`: Pfade unter `C:\StrataQ38FN-models`
- `--vram-reserve-mib`: 1000 bei 32 GB. Treibt die Karte einen Monitor und friert das Bild ein, auf 1500 erhöhen.

`strata-configs/strata-unsloth-ud-iq4_xs.setup-original.json` zeigt zum Vergleich, was das Setup ursprünglich erzeugt hat.

## 6. API-Key anlegen

Alle Dienste verlangen einen Bearer-Key. Er steht in `api_key.txt` (eine Zeile, ist in `.gitignore`):

```powershell
$k = -join ((48..57 + 65..90 + 97..122) | Get-Random -Count 42 | ForEach-Object { [char]$_ })
Set-Content -NoNewline -Encoding ascii api_key.txt $k
```

## 7. Einstellungen

`settings.cmd`:

| Variable | Standard | Bedeutung |
|---|---|---|
| `STRATA_PORT` | 8080 | Strata (Web-Chat, OpenAI, Anthropic) |
| `FACADE_PORT` | 8081 | vLLM-Fassade |
| `BIND_HOST` | `0.0.0.0` | `0.0.0.0` = im LAN erreichbar, `127.0.0.1` = nur dieser PC |
| `GENESIS3_MODEL` | leer | zusätzlicher Modellname, den die Fassade akzeptiert (der Name, den der Client schickt) |

Das Dashboard (Port 8082) rechnet Strom und Kosten aus `dashboard\settings.json`; die Werte lassen sich auch auf der
Seite ändern (Strompreis €/kWh, gemessene Leistung an der Steckdose im Leerlauf, GPU-Leerlauf, CPU-Zuschlag).

## 8. Firewall (nur für LAN-Zugriff)

Als Administrator, nur für das private Netzwerkprofil:

```powershell
New-NetFirewallRule -DisplayName "StrataQ38FN 8080-8082" -Direction Inbound -Protocol TCP `
  -LocalPort 8080-8082 -Action Allow -Profile Private
```

## 9. Auslagerungsdatei (empfohlen)

Der knappe Posten ist der **Windows-Commit** (RAM + Auslagerungsdatei), nicht das physische RAM. strata.exe belegt
82–88 GB Private Bytes. Mit einer systemverwalteten Auslagerungsdatei blieben im Betrieb nur 2–7 GB Commit frei; das
Ereignisprotokoll zeigte zweimal Event 26 „Nicht genügend virtueller Speicher".

Empfehlung: **feste** Auslagerungsdatei auf der NVMe, z. B. 64 GB Anfangs- = Maximalgröße
(Systemsteuerung → System → Erweiterte Systemeinstellungen → Leistung → Erweitert → Virtueller Arbeitsspeicher),
danach neu starten.

Erst damit sind diese Zusatz-Flags in der IQ4_XS-Config betriebssicher:

```
--max-context 262144                          (statt 131072, gemessen ohne Tempoverlust)
--prompt-cache-every 8192 --prompt-cache 12   (feinere Checkpoints, schnellere Teil-Wiederverwendung)
```

## 10. Starten und prüfen

Doppelklick auf `Start-IQ4_XS.bat`. Es öffnen sich die Fenster „Strata …" (Laden dauert 1–2 Minuten),
„vLLM-Fassade …" und minimiert das Dashboard.

```powershell
$k = Get-Content api_key.txt
curl.exe -s http://127.0.0.1:8080/v1/models -H "Authorization: Bearer $k"
curl.exe -s http://127.0.0.1:8081/v1/models -H "Authorization: Bearer $k"
curl.exe -s http://127.0.0.1:8080/v1/chat/completions -H "Authorization: Bearer $k" `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"x\",\"messages\":[{\"role\":\"user\",\"content\":\"Sag Hallo.\"}],\"max_tokens\":50}'
```

Dann `http://<PC>:8082/` im Browser öffnen; nach der ersten Antwort sollte der Decode bei ~140–150 tok/s liegen.

Fassaden-Tests (ohne GPU, benutzen nicht die Ports 8000/8080):

```powershell
Strata\.venv\Scripts\python.exe vllm_facade\test_facade.py
```

## 11. Clients anbinden

| Client | Base-URL | Modellname |
|---|---|---|
| OpenAI-kompatibel (Open WebUI, Continue, …) | `http://<PC>:8080/v1` | beliebig |
| Anthropic-kompatibel (z. B. Claude-Code-artige Tools) | `http://<PC>:8080` (`/v1/messages`) | beliebig |
| vLLM-Clients (genesis3) | `http://<PC>:8081/v1` | Strata-Modell-ID (siehe `/v1/models`) oder `GENESIS3_MODEL` |

Details und bekannte Lücken der Fassade: [vllm_facade/README.md](../vllm_facade/README.md).

## 12. Fehlersuche

| Symptom | Ursache / Abhilfe |
|---|---|
| Engine-Log meldet `LOW` freies VRAM, Bild friert ein | `--vram-reserve-mib` erhöhen (1000 → 1500) |
| Engine hängt beim Prompt-Lesen („layer 1 of the prompt chunk") | Modell auf schnellste NVMe, `--ple-io direct` |
| Absturz / Event 26 „Nicht genügend virtueller Speicher" | Commit voll: feste Auslagerungsdatei (Schritt 9), Kontext oder `--ple-row-cache` verkleinern |
| Q4_K_XL startet nicht oder ist sehr langsam | braucht ~45 GB RAM für Experten: Browser und andere große Programme schließen |
| Absturz bei zwei gleichzeitigen Anfragen mit `parallel` | bekannter MTP-Bug („unsupported native MMVQ GGML type"): `parallel` weglassen |
| Port belegt, `Stop-All.bat` hilft nicht | `Get-NetTCPConnection -LocalPort 8080 -State Listen` (deutsches `netstat` zeigt „ABHÖREN", nicht „LISTENING") |
| 401 Unauthorized | Header `Authorization: Bearer <api_key.txt>` fehlt oder falsch |

Logs: `Strata\strata-<modell>.log` (Engine), Fenster „vLLM-Fassade …" (Zugriffe), `dashboard\dashboard.log`.

## 13. Eigene Messungen (optional)

`agent_bench.py` startet eine Config-Variante auf Port 8090, spielt eine agentische Last ab (langer Haupt-Kontext,
Sub-Agent im Wechsel, lange Decodes) und hängt das Ergebnis an `agent_results.jsonl` an. Der Live-Server auf 8080
bleibt dabei unberührt, die RAM-/Commit-Reserve muss aber für beide reichen (in der Regel vorher `Stop-All.bat`).

```powershell
Strata\.venv\Scripts\python.exe agent_bench.py kvres --add kv-resident=32768 --env STRATA_KV_PREFETCH=1
Strata\.venv\Scripts\python.exe agent_bench.py ctx262k --set max-context=262144 --workload longctx
Strata\.venv\Scripts\python.exe agent_bench.py pcie07 --decode-runs 3 --tune '{\"pcie_frac\": 0.7}'
```

Weitere Workloads: `session`, `diverge`, `parallel`. Die gefahrenen Serien stehen in `series_*.sh`, die Auswertung im
[Analysebericht](../ANALYSE-Speicherhierarchie-XS.md), offene Hebel in [HEBEL-Kandidaten-XS.md](../HEBEL-Kandidaten-XS.md).
