# Cache Coverage Examples

## Question Flow with Caching

### ✅ Questions That Get Cached

#### 1. **Greetings** (NO LLM CALL - Instant)
```
"hi" → is_greeting() = TRUE → return greeting (instant, <1ms)
"hello" → is_greeting() = TRUE → return greeting (instant)
"hey" → is_greeting() = TRUE → return greeting (instant)
"how are you" → is_greeting() = TRUE → return greeting (instant)
"thanks" → is_greeting() = TRUE → return greeting (instant)
```
**Cache: No (Special handler, not LLM)** ⚡

---

#### 2. **Schema Questions** (NO LLM CALL - Instant)
```
"show me table columns" → schema detection → get_database_schema() (cached)
"what are the columns?" → schema detection → REUSE CACHED SCHEMA (instant)
"show tables" → schema detection → REUSE CACHED SCHEMA (instant)
"list columns" → schema detection → REUSE CACHED SCHEMA (instant)
```
**Cache: Database schema cached (3600 sec TTL)** ⚡

---

#### 3. **Data Analysis Questions** (LLM + Response Cache)

**First time asking:**
```
Q: "show total subscribers"
   ↓
normalize: "total subscribers"
   ↓
LLM call: "Analyze database, find total subscribers"
   ↓
Query: SELECT COUNT(*) FROM giza_data
   ↓
LLM generates answer: "We have 4,858 subscribers..."
   ↓
Cache saved with key: MD5(system_prompt + "total subscribers")
   ↓
Return: {"answer": "...", "cached": False, "time": 8.2s}
```

**Second time asking (same question):**
```
Q: "show total subscribers"
   ↓
normalize: "total subscribers"
   ↓
Cache lookup: MD5(system_prompt + "total subscribers") → HIT!
   ↓
Return: {"answer": "...", "cached": True, "time": <1ms}
```

**Asking similar question (normalized):**
```
Q: "show me total subscribers" OR "what about total subscribers?"
   ↓
normalize: "total subscribers"  (same as before!)
   ↓
Cache lookup: MD5(...) → HIT!
   ↓
Return cached answer instantly!
```

---

## Cache Coverage Matrix

| Question | Handler | Cache Type | Speed |
|---|---|---|---|
| "Hi" | Greeting | N/A | ⚡ <1ms |
| "Hello" | Greeting | N/A | ⚡ <1ms |
| "How are you?" | Greeting | N/A | ⚡ <1ms |
| "Show columns" | Schema | DB Schema | ⚡ <1ms |
| "What are the tables?" | Schema | DB Schema | ⚡ <1ms |
| "Total subscribers" | LLM | Response | 🔴 ~8s 1st, ⚡ <1ms 2nd |
| "Show total subscribers" | LLM | Response | 🔴 ~8s 1st, ⚡ <1ms 2nd |
| "Subscribers count" | LLM | Response | 🔴 ~8s 1st, ⚡ <1ms 2nd |
| "ARPU by technology" | LLM | Response | 🔴 ~8s 1st, ⚡ <1ms 2nd |
| "tell me ARPU analysis" | LLM | Response | 🔴 ~8s 1st, ⚡ <1ms 2nd |
| "churn risk" | LLM | Response | 🔴 ~8s 1st, ⚡ <1ms 2nd |
| "churn analysis" | LLM | Response | 🔴 ~8s 1st, ⚡ <1ms 2nd |

---

## Normalization Examples

**These all map to the same cache entry:**
```
Original Question              Normalized              Cache Entry
─────────────────────────────────────────────────────────────────
"show total subscribers"  →  "total subscribers"  ✅ SAME
"can you show total subs" →  "total subscribers"  ✅ SAME
"Tell me total subscribers" → "total subscribers" ✅ SAME
"What's the total subscribers" → "total subscribers" ✅ SAME

"ARPU by technology"       →  "arpu by technology"  ✅ SAME
"show ARPU by technology?" →  "arpu by technology"  ✅ SAME
"Can you ARPU by technology" → "arpu by technology" ✅ SAME
"Tell me ARPU by tech"    →  "arpu by technology"  ✅ SAME
```

---

## Performance Timeline Example

```
Timeline of User Interactions
──────────────────────────────────

09:00:00  Q1: "show total subscribers"
          └─ 🔴 8.2s (LLM processing + DB query)
          └─ 💾 Response cached

09:00:15  Q2: "hello"
          └─ ⚡ <1ms (Greeting detected, no LLM)

09:00:30  Q3: "show total subscribers"  (exact repeat)
          └─ ⚡ <1ms (💾 Cache HIT)

09:00:45  Q4: "what about total subscribers?"
          └─ ⚡ <1ms (normalized → same as Q1, 💾 Cache HIT)

09:01:00  Q5: "show table columns"
          └─ ⚡ <1ms (Schema cached)

09:01:15  Q6: "ARPU by technology"
          └─ 🔴 7.8s (New question, LLM processes)
          └─ 💾 Response cached

09:01:30  Q7: "tell me ARPU analysis"
          └─ ⚡ <1ms (Normalized → "arpu analysis", 💾 Cache HIT)

Total time saved by caching: ~25 seconds
```

---

## How to Check Cache Status

**In the response, look for:**

```
📊 **Metrics:**
  • 💾 **Cached Response** (instant)  ← YES = Cached
  • Tokens: 150 input + 320 output = 470 total
  • Time: 8.2s                        ← Visible only if NOT cached
```

**If you see:**
- 💾 **Cached Response** → Answer came from cache (instant)
- No cache indicator + Time shown → Fresh LLM call

---

## Summary

✅ **Questions that DON'T call LLM (instant):**
- Greetings: "hi", "hello", "thanks", etc.
- Schema: "show columns", "what tables", etc.

✅ **Questions WITH response caching:**
- First ask: ~8-10 seconds (LLM thinks)
- Repeat ask: <1ms (instant, from cache)
- Similar phrasing: <1ms (normalized match, from cache)

✅ **Cache validations:**
- Shows 💾 indicator when cached
- Shows execution time when NOT cached
- Cache expires after 1 hour (TTL)
- Stores up to 50 different questions
