"""
Quick LLM Payload Test
Tests both OpenAI format and LM Studio format to see which one works
"""
import requests
import json
from datetime import datetime

# LLM Configuration
LLM_BASE_URL = "http://192.168.120.227:7070"  # Your actual LM Studio URL
LLM_MODEL = "agpt-oss-20b"  # Your actual model

print("=" * 80)
print("🧪 LLM PAYLOAD TEST")
print("=" * 80)
print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🎯 Target: {LLM_BASE_URL}")
print(f"📦 Model: {LLM_MODEL}")
print("=" * 80)

# Test payload
test_question = "What is 2+2?"
test_system = "You are a helpful assistant."

# ⚠️ IMPORTANT: Disable proxy to fix timeout issues
DISABLE_PROXY = {"http": None, "https": None}

# ============ TEST 1: LM Studio Native Format ============
print("\n1️⃣  TEST: LM Studio Native Format (/api/v1/chat) - WITH PROXY DISABLED")
print("-" * 80)

url1 = f"{LLM_BASE_URL}/api/v1/chat"
payload1 = {
    "model": LLM_MODEL,
    "system_prompt": test_system,
    "input": test_question
}

print(f"Endpoint: {url1}")
print(f"Payload:\n{json.dumps(payload1, indent=2)}")
print("\n📤 Sending request (proxy disabled)...")

try:
    response = requests.post(url1, json=payload1, timeout=45, proxies=DISABLE_PROXY)
    print(f"✅ Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Response Keys: {list(result.keys())}")
        if 'output' in result:
            output = result['output']
            print(f"✅ Output type: {type(output).__name__}")
            if isinstance(output, list) and len(output) > 1:
                answer = output[1].get('content', str(output[1]))
                print(f"✅ SUCCESS - Answer: {answer[:100]}")
            elif isinstance(output, str):
                print(f"✅ SUCCESS - Answer: {output[:100]}")
            else:
                print(f"⚠️ Output format: {output}")
        else:
            print(f"⚠️ Unexpected format: {result}")
    else:
        print(f"❌ Error: {response.text[:200]}")
except Exception as e:
    print(f"❌ Connection Failed: {str(e)}")

# ============ TEST 2: OpenAI-compatible format (fallback) ============
print("\n\n2️⃣  TEST: OpenAI Format (/v1/chat/completions) - WITH PROXY DISABLED")
print("-" * 80)

url2 = f"{LLM_BASE_URL}/v1/chat/completions"
payload2 = {
    "model": LLM_MODEL,
    "messages": [
        {"role": "system", "content": test_system},
        {"role": "user", "content": test_question}
    ]
}

print(f"Endpoint: {url2}")
print(f"Payload:\n{json.dumps(payload2, indent=2)}")
print("\n📤 Sending request (proxy disabled)...")

try:
    response = requests.post(url2, json=payload2, timeout=45, proxies=DISABLE_PROXY)
    print(f"✅ Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Response Keys: {list(result.keys())}")
        if 'choices' in result and len(result['choices']) > 0:
            answer = result['choices'][0]['message']['content']
            print(f"✅ SUCCESS - Answer: {answer[:100]}")
        else:
            print(f"⚠️ Unexpected format: {result}")
    else:
        print(f"❌ Error: {response.text[:200]}")
except Exception as e:
    print(f"❌ Connection Failed: {str(e)}")

# ============ TEST 3: Model detection ============
print("\n\n3️⃣  TEST: Model Detection (proxy disabled)")
print("-" * 80)

test_endpoints = [
    f"{LLM_BASE_URL}/v1/models",
    f"{LLM_BASE_URL}/api/models",
]

for endpoint in test_endpoints:
    print(f"\n📍 Trying: {endpoint}")
    try:
        response = requests.get(endpoint, timeout=10, proxies=DISABLE_PROXY)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            models = data.get('data', [])
            if models:
                print(f"   ✅ Found models: {[m.get('id') for m in models[:3]]}")
                break
        else:
            print(f"   Error: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Failed: {str(e)}")

print("\n" + "=" * 80)
print("✅ TEST COMPLETE")
print("=" * 80)
print("\n📝 FINDINGS:")
print("   ✓ If Test 1 succeeds: Use OpenAI format in call_llm()")
print("   ✓ If Test 2 succeeds: Use LM Studio format in call_llm()")
print("   ✓ Copy the working model name to LLM_MODEL in app_enhanced.py")
print("=" * 80)
