# vLLM-Fassade für Strata

`vllm_facade.py` ist ein kleiner HTTP-Server (nur Python-Standardbibliothek), der sich nach außen wie ein
vLLM-Server (v0.31, OpenAI-kompatibel) verhält und intern an Strata (`127.0.0.1:8080`) weiterreicht. genesis3 (oder
jeder andere vLLM-Client) braucht nur eine neue Base-URL: `http://<pc>:8000/v1`. Strata selbst bleibt unverändert.

Sie übersetzt die Anfrage (Modellname, `chat_template_kwargs`, `guided_*`/`structured_outputs`, `top_k`, ...) und
bringt die Antwort in vLLMs Form: `reasoning`, Tool-Call-IDs `chatcmpl-tool-…`, `stop_reason`, `system_fingerprint`,
eigener Usage-Chunk bei `stream_options.include_usage`, vLLM-Fehlerformat, 404 bei unbekanntem Modell und vLLMs
Wortlaut bei Kontextüberlauf. Außerdem bietet sie `/v1/models` (mit `max_model_len`), `/health`, `/ping`,
`/version`, `/load`, `/metrics` (Prometheus, `vllm:*`), `/tokenize`, `/detokenize`, `/v1/completions` (über Chat
emuliert), `/generate` sowie die Weiterleitung von `/v1/messages` und `/v1/responses`. Strata-Admin-Routen
(`/settings`, `/load`, `/v1/load`, `/slots`, ...) werden nie weitergereicht.

## Starten

1. Strata starten (`X:\dev\StrataQ38FN\Strata\run-unsloth-ud-iq4_xs.bat`, Port 8080).
2. In `run-vllm-facade.bat` bei `GENESIS3_MODEL` den Modellnamen eintragen, den genesis3 sendet (leer lassen =
   Strata-ID `qwen3.8-flash-next-unsloth-ud-iq4_xs`).
3. `run-vllm-facade.bat` starten. Die Fassade lauscht auf `0.0.0.0:8000`. Zusätzliche Flags kann man anhängen, sie
   haben Vorrang, z. B. `run-vllm-facade.bat --api-key geheim --reasoning-output reasoning`.

Interpreter: `X:\dev\StrataQ38FN\Strata\.venv\Scripts\python.exe`. Den Tokenizer für `/tokenize` liest die Fassade
über `--strata-root` und `--strata-config`; er wird erst beim ersten Aufruf geladen.

## Flags

| Flag | Standard | Bedeutung |
|---|---|---|
| `--upstream URL` | `http://127.0.0.1:8080` | Strata |
| `--host`, `--port` | `0.0.0.0`, `8000` | Adresse der Fassade |
| `--strata-root`, `--strata-config` | `X:/dev/StrataQ38FN/Strata`, keiner | Für den Tokenizer und die Modell-ID |
| `--served-model-name A [B ...]` | Strata-Modell-ID | Erlaubte Namen; Antworten tragen immer den ersten |
| `--api-key K [...]` / `VLLM_API_KEY` | aus | Bearer-Key für `/v1/*` und `/generate`. Ein Fehlschlag gibt 401 `{"error":"Unauthorized"}` (der Body wird dann nicht gelesen) |
| `--upstream-api-key` / `STRATA_API_KEY` | aus | Key, den die Fassade an Strata schickt |
| `--model-check strict\|lenient` | `strict` | Mit `strict` gibt ein unbekanntes Modell 404, mit `lenient` wird jeder Name bedient |
| `--reasoning-output reasoning\|reasoning_content\|both\|inline` | `both` | Wo das Denken landet (vLLM ≥0.16: `reasoning`) |
| `--generation-config auto\|vllm\|<json>\|<pfad>` | `auto` | Sampling-Defaults für fehlende Felder (`auto` = 0.6 / 0.95 / 20) |
| `--enable-force-include-usage` | aus | Usage-Chunk in jedem Stream und `usage` an jedem Chunk (wie vLLM) |
| `--enable-prompt-tokens-details` | aus | `usage.prompt_tokens_details` |
| `--enable-request-id-headers` | aus | `X-Request-Id` zurückgeben oder erzeugen |
| `--allowed-origins O [...]` | keine | CORS (anders als bei vLLM, wo `*` der Standard ist) |
| `--completions chat\|off` | `chat` | Mit `off` gibt `/v1/completions` 404 |
| `--report-version` | `0.31.0+strata` | Antwort von `/version` |
| `--root NAME` | Strata-Modell-ID | `root` in `/v1/models` |
| `--quiet` | aus | Kein Zugriffslog |
| `--client-timeout S` | `60` | Sekunden, die ein Client still sein darf (Keep-Alive, Body, Lesen der Antwort); `0` = unbegrenzt |
| `--max-body-mb N` | `64` | Größter Request-Body; darüber 413 |
| `--max-connections N` | `256` | Gleichzeitige Client-Verbindungen; darüber 503 und Verbindung zu |

Die Fassade protokolliert jedes verworfene Anfragefeld einmal und nennt es (prozent-kodiert) im Antwort-Header `X-Facade-Ignored`. Die Antwort-ID ist wie bei vLLM `chatcmpl-<X-Request-Id oder request_id>` (`cmpl-…` bei Completions), sonst die von Strata.

## Tests

```
X:\dev\StrataQ38FN\Strata\.venv\Scripts\python.exe test_facade.py [-k name] [--keep-logs]
```

Das Skript startet drei Strata-Mock-Server (8081–8083), einen langsamen Fake-Upstream (8084) und acht Fassaden
(8001–8008). Danach testet es mit urllib und beendet alle Prozesse, die es gestartet hat. Ports 8000 und 8080 benutzt
es nie, und es braucht keine GPU. Gibt es `.testvenv` mit `openai`, läuft zusätzlich ein Test mit openai-python.

## Bekannte Lücken

- **`/v1/completions`, `/generate`:** Der Prompt geht als User-Nachricht durch das Chat-Template. Das Modell
  *beantwortet* ihn also, statt ihn fortzusetzen (kein Few-Shot-Fortsetzen, kein FIM). Das Denken ist dabei aus,
  `prompt_tokens` zählt das Template mit. Header: `X-Facade-Completions: chat-emulated`.
- **Mit 400 abgelehnt:** `logprobs`/`top_logprobs`/`prompt_logprobs`, `continue_final_message`,
  `add_generation_prompt=false`, `chat_template`, mehr als 4 Stop-Strings, `logit_bias`/`allowed_token_ids`/`bad_words`,
  `n>1` beim Streamen, Audio-/Video-/Datei-Inhalte.
- **`n>1` ohne Streaming:** Die Fassade macht n Aufrufe nacheinander, also n-fache Latenz.
- **Strukturierte Ausgabe:** Strata prompted und validiert danach, es schränkt das Decoding nicht ein. `regex`,
  `grammar` und `structural_tag` geben 400, zusammen mit Tools ebenfalls 400. Ein Schema, das nicht passt, gibt 500.
  Strukturierte Streams kommen erst am Ende, in einem Stück. Nicht-Objekt-Schemas (z. B. `guided_choice`) werden als
  `{"value": ...}` verpackt und wieder ausgepackt.
- **Durchsatz:** Strata bedient standardmäßig eine Anfrage nach der anderen (FIFO ohne Limit). Für parallele
  Anfragen von genesis3 `"parallel": 2..8` in `strata-unsloth-ud-iq4_xs.json` setzen; dann wirken Penalties nicht.
- **Immer `null`:** `stop_reason`. Ignoriert werden `min_tokens`, `ignore_eos`, `stop_token_ids` und
  `include_stop_str_in_output`. Ein Seed ist nicht reproduzierbar (bei `n>1` bekommt Aufruf i den Seed `seed+i`), `top_k` ist bei 64 gedeckelt.
- **Usage:** Es gibt kein `completion_tokens_details.reasoning_tokens`. Bei `continuous_usage_stats` (oder `--enable-force-include-usage`) sind die Zahlen an den Zwischen-Chunks Näherungen: `prompt_tokens` zählt der lokale Tokenizer, `completion_tokens` die gestreamten Stücke; der Finish-Chunk und der Usage-Chunk tragen Stratas exakte Werte.
- **`/metrics`:** Die Werte sind Näherungen. KV-Auslastung zählt nur die laufende Anfrage. Die Zähler `*_total`
  kommen von Strata (inklusive Web-UI), die Histogramme nur von der Fassade.
- **`/tokenize` mit `messages`:** Bei nicht-standardmäßigem `reasoning_effort` kann der Wert um einige Tokens von
  Stratas echtem Prompt abweichen.
- **Nicht vorhanden (404):** Embeddings, Pooling, Score, Rerank, Audio, LoRA, Sleep/Wake, `/v1/responses/{id}`,
  `/docs`.
- **Tool-Calls:** Sie werden immer geparst (wie vLLM mit `--enable-auto-tool-choice`). Ein erzwungener `tool_choice`
  funktioniert per Prefill, nicht per Grammatik.
- **Version:** `/version` meldet eine vLLM-Version, die nicht wirklich läuft.
