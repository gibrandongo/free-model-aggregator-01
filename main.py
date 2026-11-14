#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two API wires, ten free-tier providers.

Reads the trial brief (MAKEFUN_PROMPT_FILE): the provider directory, the
sealed quality table and the workload manifest. Probes each provider's
catalog route, registers the catalog with the env, then serves the OpenAI
Chat Completions API and the Anthropic Messages API (JSON and server-sent
events) by routing every request to the best free model that meets its
requirements, failing over to the next candidate when a provider refuses
(402/429), and exits 0 when GET $MAKEFUN_ENV_URL/work says done.

`python3 main.py --list` prints the catalog document and exits 0.

Standard library only. Every routing choice is a function of the brief, the
requests and the providers' answers: no wall clock and no randomness anywhere
near a request or a response, so a replay leaves the same state export.
"""

import datetime
import http.client
import http.server
import json
import os
import sys
import threading
import time
import urllib.parse

ANTHROPIC_VERSION = "2023-06-01"
PROVIDER_TIMEOUT = 10.0
CATALOG_TIMEOUT = 10.0
WORK_POLL_INTERVAL = 0.2


def log(message):
    sys.stderr.write("[aggregator] %s\n" % message)
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Brief parsing
# --------------------------------------------------------------------------

def extract_brief_docs(text):
    """Pull the JSON documents out of the brief by their schema marker.

    The brief is prose with three embedded JSON documents (providers,
    ratings, workload). We scan for '{' and JSON-decode from each, keeping
    only objects that carry a schema string.
    """
    docs = {}
    decoder = json.JSONDecoder()
    i = 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = decoder.raw_decode(text, j)
        except json.JSONDecodeError:
            i = j + 1
            continue
        if isinstance(obj, dict) and isinstance(obj.get("schema"), str):
            docs[obj["schema"]] = obj
        i = end
    return docs


def load_brief(path):
    """Return (providers, ratings, manifest) from the brief file."""
    providers, ratings, manifest = [], [], []
    if not path or not os.path.exists(path):
        return providers, ratings, manifest
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    docs = extract_brief_docs(text)
    for schema, doc in docs.items():
        if schema.endswith("/providers/v1"):
            providers = doc.get("providers") or []
        elif schema.endswith("/ratings/v1"):
            ratings = doc.get("ratings") or []
        elif schema.endswith("/workload/v1"):
            manifest = doc.get("manifest") or []
    return providers, ratings, manifest


# --------------------------------------------------------------------------
# Minimal HTTP client (stdlib only)
# --------------------------------------------------------------------------

def http_request(method, url, headers=None, body=None, timeout=10.0):
    """One HTTP request. Returns (status, headers dict, raw bytes)."""
    parts = urllib.parse.urlsplit(url)
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    path = parts.path or "/"
    if parts.query:
        path += "?" + parts.query
    conn_cls = (http.client.HTTPSConnection if parts.scheme == "https"
                else http.client.HTTPConnection)
    conn = conn_cls(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, dict(resp.getheaders()), data
    finally:
        conn.close()


def provider_headers(provider, anthropic_version=None):
    """The provider's own auth header, as the brief lists it.

    The wire does not decide the header: each provider names its scheme.
    """
    headers = {"Content-Type": "application/json"}
    scheme = provider.get("auth")
    key = provider.get("key") or ""
    if scheme == "bearer":
        headers["Authorization"] = "Bearer " + key
    elif scheme == "x-api-key":
        headers["x-api-key"] = key
    elif scheme == "x-goog-api-key":
        headers["x-goog-api-key"] = key
    if provider.get("wire") == "anthropic":
        headers["anthropic-version"] = anthropic_version or ANTHROPIC_VERSION
    return headers


# --------------------------------------------------------------------------
# Catalog
# --------------------------------------------------------------------------

def _sniff_free(obj):
    for key in ("free", "is_free", "free_tier"):
        if key in obj and isinstance(obj[key], bool):
            return obj[key]
    if "paid" in obj and isinstance(obj["paid"], bool):
        return not obj["paid"]
    return None


def _sniff_context(obj):
    for key in ("context_length", "contextLength", "max_context_length",
                "input_token_limit", "maxInputTokens", "context_window"):
        value = obj.get(key)
        if isinstance(value, int):
            return value
    return None


def probe_models(env_url, provider):
    """List one provider's models from its catalog route.

    Returns [(model_id, raw_object)] or None when the route cannot be read.
    """
    base = env_url.rstrip("/") + provider.get("path", "")
    if provider.get("wire") == "gemini":
        url = base + "/v1beta/models"
    else:
        url = base + "/v1/models"
    for _attempt in range(3):
        try:
            status, _headers, data = http_request(
                "GET", url, headers=provider_headers(provider),
                timeout=CATALOG_TIMEOUT)
        except Exception:
            time.sleep(0.2)
            continue
        if status != 200:
            return None
        try:
            obj = json.loads(data.decode("utf-8"))
        except Exception:
            return None
        found = []
        if provider.get("wire") == "gemini":
            items = obj.get("models") or []
            for item in items:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("id") or ""
                    model_id = name[7:] if name.startswith("models/") else name
                    if model_id:
                        found.append((model_id, item))
        else:
            items = obj.get("data") or []
            for item in items:
                if isinstance(item, dict) and item.get("id"):
                    found.append((item["id"], item))
                elif isinstance(item, str) and item:
                    found.append((item, {}))
        return found if found else None
    return None


def build_catalog(env_url, providers, ratings):
    """The catalog document: one entry per provider, every model with its
    free flag. Free flags come from the provider's own catalog object when
    it names one, else from the brief's sealed quality table (free models
    only); a model the table does not name is paid. Context lengths come
    from the table for free models, else from the provider object, else
    null."""
    free_rows = {}
    for row in ratings:
        if row.get("free"):
            free_rows[(row.get("provider"), row.get("model"))] = row
    catalog_providers = []
    for provider in providers:
        models = []
        seen = set()
        probed = None
        if env_url:
            try:
                probed = probe_models(env_url, provider)
            except Exception:
                probed = None
        if probed:
            for model_id, obj in probed:
                if model_id in seen:
                    continue
                seen.add(model_id)
                row = free_rows.get((provider.get("id"), model_id))
                free = _sniff_free(obj)
                if free is None:
                    free = row is not None
                context = None
                if row is not None:
                    context = row.get("context_length")
                if context is None:
                    context = _sniff_context(obj)
                models.append({"id": model_id, "free": bool(free),
                               "context_length": context})
        else:
            # The catalog route could not be read: fall back to the free
            # models the brief's table names for this provider.
            for row in ratings:
                if (row.get("provider") == provider.get("id")
                        and row.get("free") and row.get("model") not in seen):
                    seen.add(row["model"])
                    models.append({"id": row["model"], "free": True,
                                   "context_length": row.get("context_length")})
        catalog_providers.append({
            "id": provider.get("id"),
            "base_url": provider.get("path", ""),
            "key_env": provider.get("key_env", ""),
            "wire": provider.get("wire"),
            "models": models,
        })
    return {"providers": catalog_providers}


def catalog_bytes(catalog):
    """The one serialization used both for --list and for register."""
    return json.dumps(catalog, separators=(",", ":"))


# --------------------------------------------------------------------------
# Request normalization: client wire -> intermediate representation
# --------------------------------------------------------------------------

def _parse_data_url(url):
    """Split a data: URL into (media_type, base64_data) or return None."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None
    head, sep, data = url[5:].partition(",")
    if not sep or ";base64" not in head:
        return None
    media_type = head.split(";")[0] or "image/png"
    return media_type, data


def _content_to_text(content):
    """Stringify message content of any shape into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text") or "")
                elif "text" in part:
                    parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return json.dumps(content)


def openai_to_ir(body):
    """OpenAI Chat Completions body -> IR."""
    system_parts = []
    messages = []
    for message in body.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer"):
            text = _content_to_text(content)
            if text:
                system_parts.append(text)
            continue
        parts = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    parts.append({"kind": "text", "text": part.get("text") or ""})
                elif ptype == "image_url":
                    image_url = part.get("image_url") or {}
                    url = image_url.get("url") or ""
                    parsed = _parse_data_url(url)
                    if parsed:
                        parts.append({"kind": "image", "media_type": parsed[0],
                                      "data": parsed[1]})
                    else:
                        parts.append({"kind": "image", "url": url})
                elif ptype == "input_audio":
                    continue
                elif "text" in part:
                    parts.append({"kind": "text", "text": str(part["text"])})
        elif isinstance(content, str) and content:
            parts.append({"kind": "text", "text": content})
        if role == "assistant":
            for call in message.get("tool_calls") or []:
                fn = call.get("function") or {}
                parts.append({"kind": "tool_call",
                              "id": call.get("id") or "",
                              "name": fn.get("name") or "",
                              "arguments": fn.get("arguments") or ""})
        if role == "tool":
            parts.append({"kind": "tool_result",
                          "id": message.get("tool_call_id") or "",
                          "content": _content_to_text(content)})
        messages.append({"role": role, "parts": parts})
    tools = []
    for tool in body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") or {}
        if tool.get("type") == "function" and fn.get("name"):
            tools.append({"name": fn["name"],
                          "description": fn.get("description") or "",
                          "parameters": fn.get("parameters") or {}})
    choice = body.get("tool_choice")
    tool_choice = _normalize_openai_tool_choice(choice)
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = []
    fmt = body.get("response_format") or {}
    json_mode = isinstance(fmt, dict) and fmt.get("type") in (
        "json_object", "json_schema")
    return {
        "system": "\n\n".join(system_parts) if system_parts else None,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "stop": [str(s) for s in stop],
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "json_mode": json_mode,
    }


def _normalize_openai_tool_choice(choice):
    if choice is None:
        return None
    if choice == "auto":
        return "auto"
    if choice == "none":
        return "none"
    if choice == "required":
        return "any"
    if isinstance(choice, dict):
        fn = choice.get("function") or {}
        if fn.get("name"):
            return {"tool": fn["name"]}
        if choice.get("type") == "none":
            return "none"
    return "auto"


def anthropic_to_ir(body):
    """Anthropic Messages body -> IR."""
    system = body.get("system")
    if isinstance(system, list):
        system = "".join(
            (block.get("text") or "") if isinstance(block, dict) else str(block)
            for block in system) or None
    elif not isinstance(system, str):
        system = None
    messages = []
    for message in body.get("messages") or []:
        role = message.get("role")
        content = message.get("content")
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"kind": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append({"kind": "text", "text": block.get("text") or ""})
                elif btype == "image":
                    source = block.get("source") or {}
                    if source.get("type") == "base64":
                        parts.append({"kind": "image",
                                      "media_type": source.get("media_type") or "image/png",
                                      "data": source.get("data") or ""})
                    elif source.get("type") == "url":
                        parts.append({"kind": "image", "url": source.get("url") or ""})
                elif btype == "tool_use":
                    parts.append({"kind": "tool_call",
                                  "id": block.get("id") or "",
                                  "name": block.get("name") or "",
                                  "arguments": json.dumps(
                                      block.get("input") or {},
                                      separators=(",", ":"))})
                elif btype == "tool_result":
                    parts.append({"kind": "tool_result",
                                  "id": block.get("tool_use_id") or "",
                                  "content": _content_to_text(block.get("content"))})
        messages.append({"role": role, "parts": parts})
    tools = []
    for tool in body.get("tools") or []:
        if isinstance(tool, dict) and tool.get("name"):
            tools.append({"name": tool["name"],
                          "description": tool.get("description") or "",
                          "parameters": tool.get("input_schema") or {}})
    choice = body.get("tool_choice") or {}
    if isinstance(choice, dict):
        ctype = choice.get("type")
        if ctype == "auto":
            tool_choice = "auto"
        elif ctype == "any":
            tool_choice = "any"
        elif ctype == "none":
            tool_choice = "none"
        elif ctype == "tool" and choice.get("name"):
            tool_choice = {"tool": choice["name"]}
        else:
            tool_choice = None
    else:
        tool_choice = None
    stop = body.get("stop_sequences")
    if isinstance(stop, str):
        stop = [stop]
    elif not isinstance(stop, list):
        stop = []
    return {
        "system": system,
        "messages": messages,
        "tools": tools,
        "tool_choice": tool_choice,
        "stop": [str(s) for s in stop],
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "json_mode": False,
    }


# --------------------------------------------------------------------------
# Rendering: IR -> provider wire
# --------------------------------------------------------------------------

def _image_url_part(part):
    if "data" in part:
        url = "data:%s;base64,%s" % (part.get("media_type") or "image/png",
                                     part.get("data") or "")
    else:
        url = part.get("url") or ""
    return {"type": "image_url", "image_url": {"url": url}}


def ir_to_openai(ir, model, stream):
    messages = []
    if ir["system"]:
        messages.append({"role": "system", "content": ir["system"]})
    for message in ir["messages"]:
        role = message["role"]
        parts = message["parts"]
        if role == "tool":
            for part in parts:
                if part["kind"] == "tool_result":
                    messages.append({"role": "tool",
                                     "tool_call_id": part["id"],
                                     "content": part["content"]})
            continue
        text_parts = [p for p in parts if p["kind"] == "text"]
        image_parts = [p for p in parts if p["kind"] == "image"]
        call_parts = [p for p in parts if p["kind"] == "tool_call"]
        result_parts = [p for p in parts if p["kind"] == "tool_result"]
        content = None
        if len(parts) == 1 and parts[0]["kind"] == "text":
            content = parts[0]["text"]
        elif text_parts or image_parts:
            content = []
            for part in parts:
                if part["kind"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                elif part["kind"] == "image":
                    content.append(_image_url_part(part))
        if role == "assistant" and call_parts:
            assistant = {"role": "assistant",
                         "content": content,
                         "tool_calls": [
                             {"id": p["id"], "type": "function",
                              "function": {"name": p["name"],
                                           "arguments": p["arguments"]}}
                             for p in call_parts]}
            messages.append(assistant)
        elif content is not None:
            messages.append({"role": role if role in ("user", "assistant") else "user",
                             "content": content})
        for part in result_parts:
            messages.append({"role": "tool", "tool_call_id": part["id"],
                             "content": part["content"]})
    if not messages:
        messages.append({"role": "user", "content": ""})
    out = {"model": model, "messages": messages, "stream": bool(stream)}
    if ir["max_tokens"] is not None:
        out["max_tokens"] = ir["max_tokens"]
    if ir["temperature"] is not None:
        out["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        out["top_p"] = ir["top_p"]
    if ir["stop"]:
        out["stop"] = ir["stop"]
    if ir["tools"]:
        out["tools"] = [
            {"type": "function",
             "function": {"name": t["name"], "description": t["description"],
                          "parameters": t["parameters"]}}
            for t in ir["tools"]]
        choice = ir["tool_choice"]
        if choice == "auto":
            out["tool_choice"] = "auto"
        elif choice == "none":
            out["tool_choice"] = "none"
        elif choice == "any":
            out["tool_choice"] = "required"
        elif isinstance(choice, dict) and choice.get("tool"):
            out["tool_choice"] = {"type": "function",
                                  "function": {"name": choice["tool"]}}
    if ir["json_mode"]:
        out["response_format"] = {"type": "json_object"}
    return out


def ir_to_anthropic(ir, model, stream):
    messages = []
    for message in ir["messages"]:
        role = message["role"]
        parts = message["parts"]
        if role == "tool":
            blocks = [{"type": "tool_result", "tool_use_id": p["id"],
                       "content": p["content"]}
                      for p in parts if p["kind"] == "tool_result"]
            if blocks:
                messages.append({"role": "user", "content": blocks})
            continue
        if len(parts) == 1 and parts[0]["kind"] == "text":
            messages.append({"role": role if role in ("user", "assistant") else "user",
                             "content": parts[0]["text"]})
            continue
        blocks = []
        for part in parts:
            if part["kind"] == "text":
                blocks.append({"type": "text", "text": part["text"]})
            elif part["kind"] == "image":
                if "data" in part:
                    blocks.append({"type": "image", "source": {
                        "type": "base64",
                        "media_type": part.get("media_type") or "image/png",
                        "data": part.get("data") or ""}})
                else:
                    blocks.append({"type": "image", "source": {
                        "type": "url", "url": part.get("url") or ""}})
            elif part["kind"] == "tool_call":
                try:
                    tool_input = json.loads(part["arguments"] or "{}")
                except Exception:
                    tool_input = {}
                if not isinstance(tool_input, dict):
                    tool_input = {}
                blocks.append({"type": "tool_use", "id": part["id"],
                               "name": part["name"], "input": tool_input})
            elif part["kind"] == "tool_result":
                blocks.append({"type": "tool_result", "tool_use_id": part["id"],
                               "content": part["content"]})
        messages.append({"role": role if role in ("user", "assistant") else "user",
                         "content": blocks})
    if not messages:
        messages.append({"role": "user", "content": ""})
    out = {"model": model,
           "max_tokens": ir["max_tokens"] if ir["max_tokens"] is not None else 1024,
           "messages": messages,
           "stream": bool(stream)}
    if ir["system"]:
        out["system"] = ir["system"]
    if ir["temperature"] is not None:
        out["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        out["top_p"] = ir["top_p"]
    if ir["stop"]:
        out["stop_sequences"] = ir["stop"]
    if ir["tools"]:
        out["tools"] = [
            {"name": t["name"], "description": t["description"],
             "input_schema": t["parameters"]}
            for t in ir["tools"]]
        choice = ir["tool_choice"]
        if choice == "auto":
            out["tool_choice"] = {"type": "auto"}
        elif choice == "none":
            out["tool_choice"] = {"type": "none"}
        elif choice == "any":
            out["tool_choice"] = {"type": "any"}
        elif isinstance(choice, dict) and choice.get("tool"):
            out["tool_choice"] = {"type": "tool", "name": choice["tool"]}
    return out


def ir_to_gemini(ir, model, stream):
    """Render the IR as a Gemini generateContent body.

    Returns (url_suffix, body)."""
    id_to_name = {}
    for message in ir["messages"]:
        for part in message["parts"]:
            if part["kind"] == "tool_call" and part["id"]:
                id_to_name[part["id"]] = part["name"]
    contents = []
    for message in ir["messages"]:
        role = "model" if message["role"] == "assistant" else "user"
        parts = []
        for part in message["parts"]:
            if part["kind"] == "text":
                parts.append({"text": part["text"]})
            elif part["kind"] == "image":
                if "data" in part:
                    parts.append({"inlineData": {
                        "mimeType": part.get("media_type") or "image/png",
                        "data": part.get("data") or ""}})
                else:
                    parts.append({"fileData": {
                        "mimeType": "image/png",
                        "fileUri": part.get("url") or ""}})
            elif part["kind"] == "tool_call":
                try:
                    args = json.loads(part["arguments"] or "{}")
                except Exception:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                parts.append({"functionCall": {"name": part["name"],
                                               "args": args}})
            elif part["kind"] == "tool_result":
                name = id_to_name.get(part["id"]) or part["id"] or "tool"
                parts.append({"functionResponse": {
                    "name": name,
                    "response": {"result": part["content"]}}})
        if not parts:
            continue
        if contents and contents[-1]["role"] == role:
            contents[-1]["parts"].extend(parts)
        else:
            contents.append({"role": role, "parts": parts})
    if not contents:
        contents.append({"role": "user", "parts": [{"text": ""}]})
    body = {"contents": contents}
    if ir["system"]:
        body["systemInstruction"] = {"parts": [{"text": ir["system"]}]}
    if ir["tools"]:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"],
             "parameters": t["parameters"]}
            for t in ir["tools"]]}]
        choice = ir["tool_choice"]
        mode = None
        allowed = None
        if choice == "auto":
            mode = "AUTO"
        elif choice == "none":
            mode = "NONE"
        elif choice == "any":
            mode = "ANY"
        elif isinstance(choice, dict) and choice.get("tool"):
            mode = "ANY"
            allowed = [choice["tool"]]
        if mode:
            config = {"mode": mode}
            if allowed:
                config["allowedFunctionNames"] = allowed
            body["toolConfig"] = {"functionCallingConfig": config}
    generation = {}
    if ir["temperature"] is not None:
        generation["temperature"] = ir["temperature"]
    if ir["top_p"] is not None:
        generation["topP"] = ir["top_p"]
    if ir["max_tokens"] is not None:
        generation["maxOutputTokens"] = ir["max_tokens"]
    if ir["stop"]:
        generation["stopSequences"] = ir["stop"]
    if ir["json_mode"]:
        generation["responseMimeType"] = "application/json"
    if generation:
        body["generationConfig"] = generation
    suffix = ":streamGenerateContent?alt=sse" if stream else ":generateContent"
    return suffix, body


# --------------------------------------------------------------------------
# Provider responses -> normalized answer
# --------------------------------------------------------------------------

def parse_sse_bytes(raw):
    """Parse an SSE byte stream into a list of JSON payloads.

    Falls back to parsing the whole body as one JSON object, so a provider
    that answers JSON to a streamed request (or vice versa) still parses.
    """
    text = raw.decode("utf-8", errors="replace")
    events = []
    block = []

    def flush(block):
        data_lines = [line for line in block if line.startswith("data:")]
        if not data_lines:
            return
        payload = "\n".join(line[5:].lstrip() for line in data_lines)
        if payload and payload != "[DONE]":
            try:
                events.append(json.loads(payload))
            except Exception:
                pass

    for line in text.splitlines():
        if line == "":
            flush(block)
            block = []
        else:
            block.append(line)
    flush(block)
    if not events:
        stripped = text.strip()
        if stripped:
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    events.append(obj)
            except Exception:
                pass
    return events


OPENAI_KNOWN_KEYS = {"id", "object", "created", "model", "choices", "usage",
                     "system_fingerprint"}
ANTHROPIC_KNOWN_KEYS = {"id", "type", "role", "content", "model",
                        "stop_reason", "stop_sequence", "usage"}
GEMINI_KNOWN_KEYS = {"candidates", "promptFeedback", "usageMetadata",
                     "modelVersion"}


def _extras(obj, known):
    return {k: v for k, v in obj.items() if k not in known} if isinstance(obj, dict) else {}


def _tool_call(call_id, name, arguments_str, index):
    try:
        parsed = json.loads(arguments_str or "{}")
    except Exception:
        parsed = None
    if not isinstance(parsed, dict):
        parsed = None
    return {"id": call_id or ("call-%d" % index),
            "name": name or "",
            "arguments_str": arguments_str or "",
            "arguments_obj": parsed}


def parse_openai_answer(chunks, streamed):
    text_parts = []
    calls = {}
    finish = None
    usage_in = 0
    usage_out = 0
    extra = {}
    if not streamed:
        obj = chunks[0] if chunks else {}
        extra = _extras(obj, OPENAI_KNOWN_KEYS)
        choices = obj.get("choices") or []
        if choices:
            choice = choices[0]
            message = choice.get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text") or "")
            for index, call in enumerate(message.get("tool_calls") or []):
                fn = call.get("function") or {}
                calls[index] = _tool_call(
                    call.get("id"), fn.get("name"),
                    fn.get("arguments") or "", index)
            finish = choice.get("finish_reason")
        usage = obj.get("usage") or {}
        usage_in = usage.get("prompt_tokens") or 0
        usage_out = usage.get("completion_tokens") or 0
    else:
        for chunk in chunks:
            extra.update(_extras(chunk, OPENAI_KNOWN_KEYS))
            usage = chunk.get("usage")
            if isinstance(usage, dict):
                usage_in = usage.get("prompt_tokens") or usage_in
                usage_out = usage.get("completion_tokens") or usage_out
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    text_parts.append(delta["content"])
                for call in delta.get("tool_calls") or []:
                    index = call.get("index", 0)
                    slot = calls.setdefault(
                        index, {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if isinstance(fn.get("arguments"), str):
                        slot["arguments"] += fn["arguments"]
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
        calls = {i: _tool_call(s["id"], s["name"], s["arguments"], i)
                 for i, s in calls.items()}
    ordered = [calls[i] for i in sorted(calls)]
    return _finish_answer("".join(text_parts), ordered, finish, "openai",
                          usage_in, usage_out, extra)


def parse_anthropic_answer(chunks, streamed):
    text_parts = []
    calls = []
    finish = None
    usage_in = 0
    usage_out = 0
    extra = {}
    if not streamed:
        obj = chunks[0] if chunks else {}
        extra = _extras(obj, ANTHROPIC_KNOWN_KEYS)
        for index, block in enumerate(obj.get("content") or []):
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(block.get("text") or "")
            elif block.get("type") == "tool_use":
                calls.append(_tool_call(
                    block.get("id"), block.get("name"),
                    json.dumps(block.get("input") or {},
                               separators=(",", ":")), index))
        finish = obj.get("stop_reason")
        usage = obj.get("usage") or {}
        usage_in = usage.get("input_tokens") or 0
        usage_out = usage.get("output_tokens") or 0
    else:
        blocks = {}
        for event in chunks:
            etype = event.get("type")
            if etype == "message_start":
                message = event.get("message") or {}
                usage = message.get("usage") or {}
                usage_in = usage.get("input_tokens") or usage_in
            elif etype == "content_block_start":
                index = event.get("index", 0)
                block = event.get("content_block") or {}
                blocks[index] = {"type": block.get("type"),
                                 "id": block.get("id") or "",
                                 "name": block.get("name") or "",
                                 "text": "", "json": ""}
            elif etype == "content_block_delta":
                index = event.get("index", 0)
                slot = blocks.setdefault(index, {"type": None, "id": "",
                                                 "name": "", "text": "",
                                                 "json": ""})
                delta = event.get("delta") or {}
                if delta.get("type") == "text_delta":
                    slot["text"] += delta.get("text") or ""
                elif delta.get("type") == "input_json_delta":
                    slot["json"] += delta.get("partial_json") or ""
                elif isinstance(delta.get("text"), str):
                    slot["text"] += delta["text"]
            elif etype == "message_delta":
                delta = event.get("delta") or {}
                if delta.get("stop_reason"):
                    finish = delta["stop_reason"]
                usage = event.get("usage") or {}
                usage_out = usage.get("output_tokens") or usage_out
        for index in sorted(blocks):
            block = blocks[index]
            if block["type"] == "tool_use":
                calls.append(_tool_call(block["id"], block["name"],
                                        block["json"], index))
            else:
                text_parts.append(block["text"])
    return _finish_answer("".join(text_parts), calls, finish, "anthropic",
                          usage_in, usage_out, extra)


def parse_gemini_answer(chunks, streamed):
    text_parts = []
    calls = []
    finish = None
    usage_in = 0
    usage_out = 0
    extra = {}
    for obj in chunks:
        extra.update(_extras(obj, GEMINI_KNOWN_KEYS))
        usage = obj.get("usageMetadata") or {}
        if usage.get("promptTokenCount") is not None:
            usage_in = usage["promptTokenCount"]
        if usage.get("candidatesTokenCount") is not None:
            usage_out = usage["candidatesTokenCount"]
        for candidate in obj.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                if not isinstance(part, dict):
                    continue
                if isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
                elif isinstance(part.get("functionCall"), dict):
                    fn = part["functionCall"]
                    args = fn.get("args") or {}
                    calls.append(_tool_call(
                        "", fn.get("name"),
                        json.dumps(args, separators=(",", ":")) if isinstance(args, dict) else str(args),
                        len(calls)))
            if candidate.get("finishReason"):
                finish = candidate["finishReason"]
    return _finish_answer("".join(text_parts), calls, finish, "gemini",
                          usage_in, usage_out, extra)


def _finish_answer(text, calls, finish, wire, usage_in, usage_out, extra):
    if wire == "openai":
        mapping = {"stop": "stop", "length": "length",
                   "tool_calls": "tool_calls", "function_call": "tool_calls",
                   "content_filter": "content_filter"}
        normalized = mapping.get(finish, "stop")
    elif wire == "anthropic":
        mapping = {"end_turn": "stop", "max_tokens": "length",
                   "tool_use": "tool_calls", "stop_sequence": "stop",
                   "refusal": "content_filter"}
        normalized = mapping.get(finish, "stop")
    else:
        mapping = {"STOP": "stop", "MAX_TOKENS": "length",
                   "SAFETY": "content_filter", "RECITATION": "content_filter",
                   "BLOCKLIST": "content_filter",
                   "PROHIBITED_CONTENT": "content_filter",
                   "SPII": "content_filter", "OTHER": "stop",
                   "FINISH_REASON_UNSPECIFIED": "stop"}
        normalized = mapping.get(finish, "stop")
    if calls and normalized == "stop":
        normalized = "tool_calls"
    return {"text": text, "tool_calls": calls, "finish": normalized,
            "usage_in": usage_in or 0, "usage_out": usage_out or 0,
            "extra": extra}


# --------------------------------------------------------------------------
# Rendering: normalized answer -> client wire
# --------------------------------------------------------------------------

def trial_created_epoch():
    raw = os.environ.get("MAKEFUN_TRIAL_NOW")
    if raw:
        try:
            text = raw.strip()
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            moment = datetime.datetime.fromisoformat(text)
            return int(moment.timestamp())
        except Exception:
            pass
    return 0


def _openai_tool_calls(answer):
    return [{"id": call["id"], "type": "function",
             "function": {"name": call["name"],
                          "arguments": call["arguments_str"]}}
            for call in answer["tool_calls"]]


def render_openai_json(answer, model, response_id, created):
    content = answer["text"]
    if not content and answer["tool_calls"]:
        content = None
    body = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content,
                        "tool_calls": _openai_tool_calls(answer) or None},
            "finish_reason": answer["finish"],
            "logprobs": None,
        }],
        "usage": {"prompt_tokens": answer["usage_in"],
                  "completion_tokens": answer["usage_out"],
                  "total_tokens": answer["usage_in"] + answer["usage_out"]},
    }
    if not answer["tool_calls"]:
        body["choices"][0]["message"].pop("tool_calls")
    body.update(answer["extra"])
    return body


def render_openai_sse(answer, model, response_id, created):
    events = []
    base = {"id": response_id, "object": "chat.completion.chunk",
            "created": created, "model": model}
    first = dict(base)
    first["choices"] = [{"index": 0, "delta": {"role": "assistant"},
                         "finish_reason": None}]
    events.append(first)
    if answer["text"]:
        chunk = dict(base)
        chunk["choices"] = [{"index": 0,
                             "delta": {"content": answer["text"]},
                             "finish_reason": None}]
        events.append(chunk)
    for index, call in enumerate(_openai_tool_calls(answer)):
        chunk = dict(base)
        delta_call = dict(call)
        delta_call["index"] = index
        chunk["choices"] = [{"index": 0, "delta": {"tool_calls": [delta_call]},
                             "finish_reason": None}]
        events.append(chunk)
    final = dict(base)
    final["choices"] = [{"index": 0, "delta": {},
                         "finish_reason": answer["finish"]}]
    events.append(final)
    usage_chunk = dict(base)
    usage_chunk["choices"] = []
    usage_chunk["usage"] = {"prompt_tokens": answer["usage_in"],
                            "completion_tokens": answer["usage_out"],
                            "total_tokens": answer["usage_in"] + answer["usage_out"]}
    usage_chunk.update(answer["extra"])
    events.append(usage_chunk)
    out = []
    for event in events:
        out.append("data: %s\n\n" % json.dumps(event, separators=(",", ":")))
    out.append("data: [DONE]\n\n")
    return "".join(out).encode("utf-8")


def _anthropic_content(answer):
    blocks = []
    if answer["text"]:
        blocks.append({"type": "text", "text": answer["text"]})
    for call in answer["tool_calls"]:
        tool_input = call["arguments_obj"]
        if tool_input is None:
            try:
                parsed = json.loads(call["arguments_str"] or "{}")
                tool_input = parsed if isinstance(parsed, dict) else {}
            except Exception:
                tool_input = {}
        blocks.append({"type": "tool_use", "id": call["id"],
                       "name": call["name"], "input": tool_input})
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _anthropic_stop_reason(finish):
    return {"stop": "end_turn", "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "refusal"}.get(finish, "end_turn")


def render_anthropic_json(answer, model, response_id):
    body = {
        "id": response_id,
        "type": "message",
        "role": "assistant",
        "content": _anthropic_content(answer),
        "model": model,
        "stop_reason": _anthropic_stop_reason(answer["finish"]),
        "stop_sequence": None,
        "usage": {"input_tokens": answer["usage_in"],
                  "output_tokens": answer["usage_out"]},
    }
    body.update(answer["extra"])
    return body


def render_anthropic_sse(answer, model, response_id):
    events = []
    events.append({
        "type": "message_start",
        "message": {"id": response_id, "type": "message", "role": "assistant",
                    "content": [], "model": model, "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": answer["usage_in"],
                              "output_tokens": 0}}})
    index = 0
    for block in _anthropic_content(answer):
        if block["type"] == "text":
            events.append({"type": "content_block_start", "index": index,
                           "content_block": {"type": "text", "text": ""}})
            events.append({"type": "content_block_delta", "index": index,
                           "delta": {"type": "text_delta",
                                     "text": block["text"]}})
            events.append({"type": "content_block_stop", "index": index})
        else:
            events.append({"type": "content_block_start", "index": index,
                           "content_block": {"type": "tool_use",
                                             "id": block["id"],
                                             "name": block["name"],
                                             "input": {}}})
            events.append({"type": "content_block_delta", "index": index,
                           "delta": {"type": "input_json_delta",
                                     "partial_json": json.dumps(
                                         block["input"],
                                         separators=(",", ":"))}})
            events.append({"type": "content_block_stop", "index": index})
        index += 1
    delta_event = {
        "type": "message_delta",
        "delta": {"stop_reason": _anthropic_stop_reason(answer["finish"]),
                  "stop_sequence": None},
        "usage": {"output_tokens": answer["usage_out"]}}
    delta_event.update(answer["extra"])
    events.append(delta_event)
    events.append({"type": "message_stop"})
    out = []
    for event in events:
        out.append("event: %s\ndata: %s\n\n" % (
            event["type"], json.dumps(event, separators=(",", ":"))))
    return "".join(out).encode("utf-8")


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

class Router:
    """Quality-ordered candidate list with per-provider failure memory.

    A provider that fails in two consecutive requests is treated as spent
    (budget gone or outage) and skipped for the rest of the trial. A single
    failure never removes a provider: failure schedules refuse one call and
    then serve again.
    """

    def __init__(self, providers, ratings):
        self.providers = {p["id"]: p for p in providers}
        self.provider_order = {p["id"]: i for i, p in enumerate(providers)}
        self.ratings = ratings
        self.failures = {}
        self.dead = set()

    def candidates(self, requirements, client_wire):
        found = []
        for row in self.ratings:
            if not row.get("free"):
                continue
            provider_id = row.get("provider")
            if provider_id in self.dead or provider_id not in self.providers:
                continue
            if requirements["tools"] and not row.get("tools"):
                continue
            if requirements["vision"] and not row.get("vision"):
                continue
            if requirements["json"] and not row.get("json"):
                continue
            minimum = requirements["min_context"]
            if minimum and (row.get("context_length") or 0) < minimum:
                continue
            provider = self.providers[provider_id]
            # A JSON-mode OpenAI request carries response_format, which only
            # the OpenAI wire can carry: route it to an OpenAI-wire provider.
            if requirements["json_wire"] and provider.get("wire") != "openai":
                continue
            found.append(row)
        order = self.provider_order
        providers = self.providers

        def sort_key(row):
            provider = providers[row["provider"]]
            same_wire = 0 if provider.get("wire") == client_wire else 1
            return (-(row.get("rating_milli") or 0), same_wire,
                    order[row["provider"]], row.get("model") or "")

        found.sort(key=sort_key)
        return found

    def note_success(self, provider_id):
        self.failures[provider_id] = 0

    def note_failure(self, provider_id):
        count = self.failures.get(provider_id, 0) + 1
        self.failures[provider_id] = count
        if count >= 2:
            self.dead.add(provider_id)


# --------------------------------------------------------------------------
# Server state
# --------------------------------------------------------------------------

class State:
    def __init__(self, env_url, providers, ratings, manifest, catalog):
        self.env_url = env_url
        self.providers = {p["id"]: p for p in providers}
        self.ratings = ratings
        self.manifest = manifest
        self.manifest_by_id = {m.get("id"): m for m in manifest}
        self.catalog = catalog
        self.router = Router(providers, ratings)
        # RLock: the request handler holds this lock while serving, and
        # serve_inference re-enters it for the request counter. A plain
        # Lock deadlocks the first inference request.
        self.lock = threading.RLock()
        self.counter = 0
        self.request_index = 0
        self.in_flight = 0
        self.last_next = None
        self.created = trial_created_epoch()
        # The free model ids, in catalog order, deduplicated: what our own
        # GET /v1/models must list.
        self.free_ids = []
        seen = set()
        for provider in catalog.get("providers") or []:
            for model in provider.get("models") or []:
                if model.get("free") and model["id"] not in seen:
                    seen.add(model["id"])
                    self.free_ids.append(model["id"])

    def next_counter(self):
        with self.lock:
            self.counter += 1
            return self.counter


# --------------------------------------------------------------------------
# Provider calls
# --------------------------------------------------------------------------

def call_provider(state, provider, model, wire_body, stream,
                  anthropic_version=None):
    base = state.env_url.rstrip("/") + provider.get("path", "")
    wire = provider.get("wire")
    if wire == "openai":
        url = base + "/v1/chat/completions"
        payload = wire_body
    elif wire == "anthropic":
        url = base + "/v1/messages"
        payload = wire_body
    else:
        suffix, payload = wire_body
        url = base + "/v1beta/models/" + model + suffix
    headers = provider_headers(provider, anthropic_version)
    data = json.dumps(payload).encode("utf-8")
    return http_request("POST", url, headers=headers, body=data,
                        timeout=PROVIDER_TIMEOUT)


def parse_provider_answer(provider, raw, stream):
    chunks = parse_sse_bytes(raw)
    if not chunks:
        raise ValueError("empty provider answer")
    wire = provider.get("wire")
    if wire == "openai":
        return parse_openai_answer(chunks, stream)
    if wire == "anthropic":
        return parse_anthropic_answer(chunks, stream)
    return parse_gemini_answer(chunks, stream)


# --------------------------------------------------------------------------
# Request serving
# --------------------------------------------------------------------------

def derive_requirements(state, ir, client_wire, body, request_index):
    """Requirements for one request: what its own body shows, plus what the
    workload manifest says for the request in this position (the manifest
    is the only place min_context and Anthropic JSON needs appear)."""
    requirements = {
        "tools": bool(ir["tools"]) or any(
            part["kind"] in ("tool_call", "tool_result")
            for message in ir["messages"] for part in message["parts"]),
        "vision": any(
            part["kind"] == "image"
            for message in ir["messages"] for part in message["parts"]),
        "json": bool(ir["json_mode"]),
        "json_wire": bool(ir["json_mode"]) and client_wire == "openai",
        "min_context": 0,
    }
    entry = None
    if 0 <= request_index < len(state.manifest):
        candidate = state.manifest[request_index]
        if candidate.get("api") == client_wire:
            entry = candidate
    if entry is None and state.last_next:
        candidate = state.manifest_by_id.get(state.last_next)
        if candidate is not None and candidate.get("api") == client_wire:
            entry = candidate
    if entry is not None:
        needs = entry.get("requires") or {}
        if needs.get("tools"):
            requirements["tools"] = True
        if needs.get("vision"):
            requirements["vision"] = True
        if needs.get("json"):
            requirements["json"] = True
        if needs.get("min_context"):
            requirements["min_context"] = needs["min_context"]
    return requirements


def serve_inference(state, client_wire, body, headers, request_index):
    """Route one client request and render the answer in the client wire.

    Returns (status, content_type, payload_bytes). request_index counts
    every inference request received (not just served ones), so the
    workload-manifest lookup stays aligned with the client's order even
    when a request cannot be served.
    """
    stream = bool(body.get("stream"))
    if client_wire == "openai":
        ir = openai_to_ir(body)
    else:
        ir = anthropic_to_ir(body)
    requirements = derive_requirements(state, ir, client_wire, body,
                                       request_index)
    candidates = state.router.candidates(requirements, client_wire)
    anthropic_version = headers.get("anthropic-version")
    failed_this_request = set()
    answer = None
    chosen_model = None
    # Two passes: a provider whose schedule refuses one call may serve the
    # next, so a request every candidate refused gets one more round.
    for _pass in range(2):
        for row in candidates:
            provider_id = row["provider"]
            provider = state.providers[provider_id]
            model = row["model"]
            if provider.get("wire") == client_wire:
                # Same wire: forward the request untouched, model replaced.
                forward = dict(body)
                forward["model"] = model
                wire_body = forward
            elif provider.get("wire") == "openai":
                wire_body = ir_to_openai(ir, model, stream)
            elif provider.get("wire") == "anthropic":
                wire_body = ir_to_anthropic(ir, model, stream)
            else:
                wire_body = ir_to_gemini(ir, model, stream)
            try:
                status, _resp_headers, raw = call_provider(
                    state, provider, model, wire_body, stream,
                    anthropic_version)
            except Exception as exc:
                log("provider %s call error: %s" % (provider_id, exc))
                status, raw = None, b""
            if status is None or status >= 400:
                if provider_id not in failed_this_request:
                    failed_this_request.add(provider_id)
                    state.router.note_failure(provider_id)
                continue
            try:
                answer = parse_provider_answer(provider, raw, stream)
            except Exception as exc:
                log("provider %s unparsable answer: %s" % (provider_id, exc))
                if provider_id not in failed_this_request:
                    failed_this_request.add(provider_id)
                    state.router.note_failure(provider_id)
                continue
            state.router.note_success(provider_id)
            chosen_model = model
            break
        if answer is not None:
            break
        # Second pass: re-rank without the providers this request killed.
        candidates = state.router.candidates(requirements, client_wire)
    if answer is None:
        return error_response(client_wire, "all providers refused")
    number = state.next_counter()
    if client_wire == "openai":
        response_id = "chatcmpl-fma-%d" % number
        if stream:
            payload = render_openai_sse(answer, chosen_model, response_id,
                                        state.created)
            return 200, "text/event-stream", payload
        return 200, "application/json", json.dumps(
            render_openai_json(answer, chosen_model, response_id,
                               state.created),
            separators=(",", ":")).encode("utf-8")
    response_id = "msg-fma-%d" % number
    if stream:
        payload = render_anthropic_sse(answer, chosen_model, response_id)
        return 200, "text/event-stream", payload
    return 200, "application/json", json.dumps(
        render_anthropic_json(answer, chosen_model, response_id),
        separators=(",", ":")).encode("utf-8")


def error_response(client_wire, message):
    if client_wire == "openai":
        body = {"error": {"message": message, "type": "server_error",
                          "code": "all_providers_refused"}}
    else:
        body = {"type": "error",
                "error": {"type": "api_error", "message": message}}
    return 502, "application/json", json.dumps(
        body, separators=(",", ":")).encode("utf-8")


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "FreeModelAggregator/1.0"

    def log_message(self, *args):
        pass

    @property
    def state(self):
        return self.server.state

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(payload)
        self.close_connection = True

    def _read_body(self):
        length = self.headers.get("Content-Length")
        if not length:
            return b""
        try:
            size = int(length)
        except ValueError:
            return b""
        if size <= 0:
            return b""
        return self.rfile.read(size)

    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/v1/models":
            self._handle_models()
        else:
            self._send(404, "application/json",
                       b'{"error":{"message":"not found"}}')

    def _handle_models(self):
        """The model list in the wire the auth header names."""
        state = self.state
        if self.headers.get("x-api-key"):
            body = {
                "data": [{"id": model_id, "type": "model",
                          "display_name": model_id,
                          "created_at": state.created}
                         for model_id in state.free_ids],
                "has_more": False,
            }
        else:
            body = {
                "object": "list",
                "data": [{"id": model_id, "object": "model",
                          "created": state.created, "owned_by": "aggregator"}
                         for model_id in state.free_ids],
            }
        self._send(200, "application/json",
                   json.dumps(body, separators=(",", ":")).encode("utf-8"))

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path
        if path == "/v1/chat/completions":
            client_wire = "openai"
        elif path == "/v1/messages":
            client_wire = "anthropic"
        else:
            self._send(404, "application/json",
                       b'{"error":{"message":"not found"}}')
            return
        raw = self._read_body()
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            status, content_type, payload = error_response(
                client_wire, "request body is not valid JSON")
            self._send(400, content_type, payload)
            return
        if not isinstance(body, dict):
            status, content_type, payload = error_response(
                client_wire, "request body must be a JSON object")
            self._send(400, content_type, payload)
            return
        state = self.state
        # One inference request at a time: a provider serves one call at a
        # time, and serial handling keeps every choice deterministic.
        with state.lock:
            state.in_flight += 1
            request_index = state.request_index
            state.request_index += 1
            try:
                status, content_type, payload = serve_inference(
                    state, client_wire, body, self.headers, request_index)
            except Exception as exc:
                log("handler error: %s" % exc)
                status, content_type, payload = error_response(
                    client_wire, "internal error")
            finally:
                state.in_flight -= 1
        self._send(status, content_type, payload)


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, state):
        self.state = state
        super().__init__(address, Handler)


# --------------------------------------------------------------------------
# Env control routes
# --------------------------------------------------------------------------

def register(env_url, port, catalog):
    body = {"protocol_version": 1, "port": port, "catalog": catalog}
    data = json.dumps(body, separators=(",", ":")).encode("utf-8")
    url = env_url.rstrip("/") + "/register"
    for attempt in range(40):
        try:
            status, _headers, raw = http_request(
                "POST", url, headers={"Content-Type": "application/json"},
                body=data, timeout=5.0)
        except Exception as exc:
            log("register attempt %d failed: %s" % (attempt + 1, exc))
            time.sleep(0.5)
            continue
        if status == 200:
            log("registered: %s" % raw.decode("utf-8", errors="replace")[:200])
            return True
        log("register refused with status %d: %s" % (
            status, raw.decode("utf-8", errors="replace")[:200]))
        time.sleep(0.5)
    return False


def poll_work_until_done(env_url, state, httpd):
    """Poll GET /work in the main thread; shut the server down when the
    workload is done and no request is in flight."""
    url = env_url.rstrip("/") + "/work"
    while True:
        time.sleep(WORK_POLL_INTERVAL)
        try:
            status, _headers, raw = http_request("GET", url, timeout=5.0)
        except Exception:
            continue
        if status != 200:
            continue
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        if isinstance(obj.get("next"), str):
            state.last_next = obj["next"]
        if obj.get("done") is True:
            with state.lock:
                busy = state.in_flight > 0
            if not busy:
                break
    httpd.shutdown()


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def run_list():
    brief_path = os.environ.get("MAKEFUN_PROMPT_FILE")
    env_url = os.environ.get("MAKEFUN_ENV_URL") or ""
    if not brief_path or not os.path.exists(brief_path):
        sys.stdout.write('{"providers":[]}\n')
        return 0
    providers, ratings, _manifest = load_brief(brief_path)
    catalog = build_catalog(env_url, providers, ratings)
    sys.stdout.write(catalog_bytes(catalog) + "\n")
    return 0


def run_server():
    brief_path = os.environ.get("MAKEFUN_PROMPT_FILE")
    env_url = os.environ.get("MAKEFUN_ENV_URL") or ""
    providers, ratings, manifest = load_brief(brief_path) if brief_path else ([], [], [])
    catalog = build_catalog(env_url, providers, ratings)
    state = State(env_url, providers, ratings, manifest, catalog)
    httpd = Server(("0.0.0.0", 0), state)
    port = httpd.server_address[1]
    log("listening on port %d" % port)
    if env_url:
        register(env_url, port, catalog)
    else:
        log("no MAKEFUN_ENV_URL: serving without registering")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    if env_url:
        poll_work_until_done(env_url, state, httpd)
    else:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass
    thread.join(timeout=10)
    httpd.server_close()
    return 0


def main(argv):
    if "--list" in argv:
        return run_list()
    return run_server()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
