import os
import json
from typing import TypedDict, List, Optional

from dotenv import load_dotenv

import google.generativeai as genai

from langchain.tools import StructuredTool

from apify_client import ApifyClient

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

import time

LAST_LLM_TS = 0.0
MIN_INTERVAL = 15.0

def wait_llm_slot():
    """The interval between LLM calls is at least 15 seconds, Gemini caused problems a few times"""
    global LAST_LLM_TS
    now = time.monotonic()
    elapsed = now - LAST_LLM_TS if LAST_LLM_TS else float("inf")
    if elapsed < MIN_INTERVAL:
        time.sleep(MIN_INTERVAL - elapsed)
        now = time.monotonic()  # uyku sonrası güncel zaman
    LAST_LLM_TS = now


def gemini_structured_json(user_text: str, schema: dict, system: str = "", model_name: str = "gemini-1.5-flash", temperature: float = 0.0,) -> dict:
    """Force Gemini to exact json"""
    wait_llm_slot()
    generation_config = {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_schema": schema,
    }
    model = genai.GenerativeModel(model_name)
    prompt = (system or "Return ONLY valid JSON for the given schema. No prose.") + \
             "\n\nUSER:\n" + user_text
    resp = model.generate_content(prompt, generation_config=generation_config)
    return json.loads(resp.text)


# JSON schemas I used
PRODUCT_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "has_product": {"type": "boolean"},
        "product_name": {"type": "string"},
        "user_intent": {"type": "string"},
    },
    "required": ["has_product", "product_name", "user_intent"],
}

VERIFY_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {"type": "boolean"},
        "index":   {"type": "integer"},
        "reason":  {"type": "string"},
    },
    "required": ["matches", "index", "reason"],
}

def apify_amazon_search(url: str, max_items: int = 3):
    """
    Uses APIFY Token and Actor ID
    It retrieves the products from the given link and returns a simplified list.
    """
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise RuntimeError("APIFY_TOKEN .env dosyasında yok.")

    client = ApifyClient(token)
    run = client.actor("BG3WDrGdteHgZgbPK").call(run_input={
        "categoryOrProductUrls": [{"url": url}],
        "maxItems": max_items,
        "proxyCountry": "TR",
        "maxSearchPagesPerStartUrl": 1,
    })

    ds_id = run.get("defaultDatasetId") or run.get("datasetId")
    if not ds_id:
        raise RuntimeError(f"Dataset ID alınamadı. Run meta: {run}")

    items = []
    for item in client.dataset(ds_id).iterate_items():
        price_obj = item.get("price")
        price_val = price_obj.get("value") if isinstance(price_obj, dict) else price_obj
        currency = price_obj.get("currency") if isinstance(price_obj, dict) else None
        items.append({
            "product_title": item.get("title"),
            "price": price_val,
            "currency": currency,
            "rating": item.get("stars") or item.get("reviewRating") or item.get("rating"),
            "url": item.get("url") or item.get("unNormalizedProductUrl"),
        })
        if len(items) >= max_items:
            break
    return items


apify_tool = StructuredTool.from_function(
    func=apify_amazon_search,
    name="apify_amazon_search",
    description="Ffetches product information via Apify from Amazon. Input: url, max_items",
)

""" State class for LangChain"""
class GState(TypedDict):
    messages: List
    product: Optional[dict]
    items: Optional[list]
    verified: Optional[bool]
    retry_count: Optional[int]
    ask_retry: Optional[int]


def extract_product_node(state: GState):
    """LLM tries to extract the product from the user message"""
    state.setdefault("retry_count", 0)
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    user_text = last_human.content if last_human else ""

    system = (
        "Extract the product info. If the user did NOT provide a clear product, "
        "'has_product' must be false and the other string fields must be empty strings."
    )
    out = gemini_structured_json(user_text, PRODUCT_EXTRACT_SCHEMA, system=system, temperature=0.0)
    state["product"] = out
    return state


def need_more_info(state: GState):
    """Looks at the answer to the question: Was LLM able to produce the product?"""
    p = state.get("product") or {}
    return "ask_user" if not p.get("has_product") else "search"

def ask_user_node(state: GState):
    """If LLM doesn't understand the product, they will ask again from the console and have 3 repetitions"""
    state["ask_retry"] = (state.get("ask_retry", 0) or 0) + 1

    if state["ask_retry"] > 3:
        msg = AIMessage(content="I requested information three times, but I didn't receive enough information. I'm ending the stream.")
        state.setdefault("messages", [])
        state["messages"].append(msg)
        return state

    # Aksi halde kullanıcıdan input al
    AIcontent = "Can you write the exact name of the product you are looking for? Example: 'Logitech MX Master 3S'"
    print(AIcontent)
    user_input = input().strip()
    state.setdefault("messages", [])
    if user_input:
        state["messages"].append(HumanMessage(content=user_input))
    else:
        state["messages"].append(AIMessage(content="I received a blank response. A clear product name is required."))
    return state

def ask_or_end(state: GState):
    """Checks if user can ask again"""
    ask_retry = state.get("ask_retry", 0) or 0
    return "extract" if ask_retry <= 3 else "end"


def apify_search_node(state: GState):
    """Searches with Apify"""
    state.setdefault("retry_count", 0)
    p = state["product"]
    query = p["product_name"].strip() if p else ""
    url = f"https://www.amazon.com.tr/s?k={query.replace(' ', '+')}"
    try:
        results = apify_tool.invoke({"url": url, "max_items": 5})
    except Exception as e:
        results = []
        state["messages"].append(AIMessage(content=f"An error occurred while searching: {e}"))
    state["items"] = results
    return state


def verify_match_node(state: GState):
    """Checks if what Apify returns matches what the user wants"""
    p = (state.get("product") or {}).get("product_name", "")
    items = state.get("items") or []

    candidates = "\n".join(f"{i}. {it.get('product_title')}" for i, it in enumerate(items[:5])) or "(no candidates)"

    user_text = (
        f"User wants: {p}\n"
        f"Candidates (0-based):\n{candidates}\n"
        "Return strictly JSON with fields {matches:boolean, index:integer, reason:string}. "
        "The index MUST be the 0-based index in the list above. "
        "If no candidate is a correct product-level match, set matches=false and index=-1. "
        "Accessories or family variants (e.g., 'Anywhere', 'Ergo', 'trackball', 'case/cover/grip') "
        "are NOT matches for 'MX Master 3S'."
    )

    out = gemini_structured_json(
        user_text,
        VERIFY_MATCH_SCHEMA,
        system="Return strictly JSON only.",
        temperature=0.0,
    )

    state["verified"] = bool(out.get("matches"))
    idx = int(out.get("index", 0) or 0)
    reason = out.get("reason") or ""

    if state["verified"] and 0 <= idx < len(items):
        state["items"] = [items[idx]]
    else:
        state["items"] = []

    state["messages"].append(AIMessage(content=f"Verification: {state['verified']} | idx={idx} | {reason}"))
    return state

def verified_or_retry_cond(state: GState):
    """If it matches, next step, otherwise stop or try again"""
    if state.get("verified"):
        return "present"
    retries = state.get("retry_count", 0) or 0
    return "search_again" if retries < 3 else "give_up"


def give_up_node(state: GState):
    """After your search rights are exhausted, show the closest result"""
    items = state.get("items") or []
    if items:
        top = items[:3]
        pretty = "\n".join(
            f"- {x.get('product_title')} | {x.get('price')} {x.get('currency') or ''} | {x.get('url')}"
            for x in top
        )
        msg = (
            "I couldn't find an exact match, but here are some similar options:"
            f"\n{pretty}\n\nCan you share a more specific product name/model code?"
        )
    else:
        msg = (
            "Sorry, I reached the limit of search attempts and found no suitable results. "
            "Could you please write the product name/model code a little more clearly? Example: 'Logitech MX Master 3S'"
        )
    state["messages"].append(AIMessage(content=msg))
    return state


def search_again_node(state: GState):
    """Reduce the words used in the search and search again"""
    name = (state.get("product") or {}).get("product_name", "").strip()
    state["retry_count"] = (state.get("retry_count", 0) or 0) + 1  # ++counter
    attempt = state["retry_count"]  # 1,2,3...

    tokens = name.split()
    if attempt == 1:
        q = " ".join(tokens[:3]) if tokens else name
    elif attempt == 2:
        q = " ".join(tokens[:2]) if tokens else name
    else:  # attempt >= 3
        q = tokens[0] if tokens else name

    url = f"https://www.amazon.com.tr/s?k={q.replace(' ', '+')}"
    try:
        results = apify_tool.invoke({"url": url, "max_items": 5})
    except Exception as e:
        results = []
        state["messages"].append(AIMessage(content=f"An error occurred while searching for a gradient: {e}"))
    state["items"] = results
    return state


def present_results_node(state: GState):
    """Show results to the user"""
    items = state.get("items") or []
    if not items:
        state["messages"].append(AIMessage(content="Sorry, I couldn't find any results."))
        return state

    top = items[:3] # its already 1
    pretty = "\n".join(
        f"- {x.get('product_title')} | {x.get('price')} {x.get('currency') or ''} | {x.get('url')}"
        for x in top
    )
    state["messages"].append(AIMessage(content=f"What I found:\n{pretty}"))
    return state

"""The graph in the picture"""
graph = StateGraph(GState)
graph.add_node("extract", extract_product_node)
graph.add_node("ask_user", ask_user_node)
graph.add_node("apify_search", apify_search_node)
graph.add_node("verify_match", verify_match_node)
graph.add_node("search_again", search_again_node)
graph.add_node("present", present_results_node)
graph.add_node("give_up", give_up_node)

graph.set_entry_point("extract")

graph.add_edge("apify_search", "verify_match")
graph.add_edge("search_again", "verify_match")

graph.add_conditional_edges("extract", need_more_info, {"ask_user": "ask_user", "search": "apify_search",})
graph.add_conditional_edges("ask_user", ask_or_end, {"extract": "extract", "end": END,})
graph.add_conditional_edges("verify_match", verified_or_retry_cond, {"search_again": "search_again", "present": "present", "give_up": "give_up"},)

graph.add_edge("present", END)
graph.add_edge("give_up", END)

app = graph.compile()


if __name__ == "__main__":
    print("What do you want to search?")
    user_input = input().strip() #"Logitech MX Master 3S arıyorum; sessiz tıklama önemli." # Example, language does not matter (tr,en)
    init1 = {"messages": [HumanMessage(content=user_input)] }
    state1 = app.invoke(init1)
    for m in state1["messages"]:
        if isinstance(m, AIMessage):
            print(m.content)
