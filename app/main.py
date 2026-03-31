import ollama
import json
import re
import os
from typing import List, Optional, Dict, Union
from requests import get
from bs4 import BeautifulSoup as bs
from pydantic import BaseModel

PropertyValue = Union[str, int, float, bool, None]

# ---------------------------
# 1. 고정 노드 정의 (채훈님 스타일: ID 기반 관리)
# ---------------------------
ALLOWED_NODES = {
    # Trainers / Humans
    "지우": {"id": "H01", "label": "Trainer"},
    "이슬이": {"id": "H02", "label": "Trainer"},
    "웅이": {"id": "H03", "label": "Trainer"},
    
    # [전기] 피카츄 계열
    "피츄": {"id": "P01", "label": "Pokemon"},
    "피카츄": {"id": "P02", "label": "Pokemon"},
    "라이츄": {"id": "P03", "label": "Pokemon"},
    
    # [풀/독] 이상해씨 계열
    "이상해씨": {"id": "P04", "label": "Pokemon"},
    "이상해풀": {"id": "P05", "label": "Pokemon"},
    "이상해꽃": {"id": "P06", "label": "Pokemon"},

    # [불꽃] 파이리 계열
    "파이리": {"id": "P07", "label": "Pokemon"},
    "리자드": {"id": "P08", "label": "Pokemon"},
    "리자몽": {"id": "P09", "label": "Pokemon"},

    # [물] 꼬부기 계열
    "꼬부기": {"id": "P10", "label": "Pokemon"},
    "어니부기": {"id": "P11", "label": "Pokemon"},
    "거북왕": {"id": "P12", "label": "Pokemon"},
    
    # [바위/땅] 꼬마돌 계열
    "꼬마돌": {"id": "P13", "label": "Pokemon"},
    "데구리": {"id": "P14", "label": "Pokemon"},
    "딱구리": {"id": "P15", "label": "Pokemon"},
    
    # Types
    "전기": {"id": "T01", "label": "Type"},
    "풀": {"id": "T02", "label": "Type"},
    "독": {"id": "T03", "label": "Type"},
    "불꽃": {"id": "T04", "label": "Type"},
    "물": {"id": "T05", "label": "Type"},
    "바위": {"id": "T06", "label": "Type"},
    "땅": {"id": "T07", "label": "Type"},
}

# LLM에게 전달할 노드 리스트 문자열 생성
NODES_PROMPT_STR = json.dumps([
    {"id": v["id"], "label": v["label"], "properties": {"name": k}} 
    for k, v in ALLOWED_NODES.items()
], ensure_ascii=False, indent=2)

# ---------------------------
# 기본 모델 정의
# ---------------------------
class Node(BaseModel):
    id: str
    label: str
    properties: Dict[str, PropertyValue]

class Relationship(BaseModel):
    type: str
    start_node_id: str
    end_node_id: str
    properties: Optional[Dict[str, PropertyValue]] = None

class GraphResponse(BaseModel):
    nodes: List[Node]
    relationships: List[Relationship]

# ---------------------------
# LLM 템플릿 (STRICT RULES 적용)
# ---------------------------
STRICT_TEMPLATE = f"""
You are a specialized Knowledge Graph Extraction System.
Your goal is to extract entities and relationships from English Pokemon synopses into a structured Korean-based JSON.

**[CRITICAL MAPPING TABLE]**
If you see these English names, you MUST map them to the corresponding ID and Korean Name:
- "Ash / Ash Ketchum" -> ID: "H01", Name: "지우"
- "Misty" -> ID: "H02", Name: "이슬이"
- "Brock" -> ID: "H03", Name: "웅이"
- "Pikachu" -> ID: "P02", Name: "피카츄"
- "Bulbasaur" -> ID: "P04", Name: "이상해씨"
- "Ivysaur" -> ID: "P05", Name: "이상해풀"
- "Venusaur" -> ID: "P06", Name: "이상해꽃"
- "Charmander" -> ID: "P07", Name: "파이리"
- "Squirtle" -> ID: "P10", Name: "꼬부기"

**ALLOWED NODES (Full List)**:
{NODES_PROMPT_STR}

**ALLOWED RELATIONSHIPS**:
- HAS (Trainer(H) → Pokemon(P))
- HAS_TYPE (Pokemon(P) → Type(T))
- EVOLVES_TO (Pokemon(P) → Pokemon(P))
- COMPANION (Trainer(H) → Trainer(H))
- BATTLES (Trainer or Pokemon → Trainer or Pokemon)

**[STRICT RULES - NO EXCEPTIONS]**
1. **ONLY USE PROVIDED LIST**: Do NOT create any nodes that are not in the ALLOWED NODES list. 
   - If you see "Ditto", "Primeape", or any other Pokemon NOT in the list, **IGNORE THEM ENTIRELY**.
2. **NO NEW NAMES**: Do not invent names like "디타토" or "프라임파이브". 
   - Every node must have a name from the list (e.g., "지우", "피카츄", "이상해씨").
3. **ID MATCHING**: The "id" field MUST exactly match the ID from the list (e.g., "H01", "P02").
4. **RELATIONSHIP ONLY FOR LISTED NODES**: Only create relationships between nodes that exist in the ALLOWED NODES list.

**[IMPORTANT: RESPONSE FORMAT]**
- Return ONLY the JSON object.
- DO NOT include any introductory text (e.g., "Here is the JSON", "Sure!").
- DO NOT include any markdown code blocks (```json).
- Start the response with "{" and end it with "}".


**OUTPUT FORMAT**:
{{
  "nodes": [
    {{"id": "H01", "label": "Trainer", "properties": {{"name": "지우"}}}}
  ],
  "relationships": [
    {{"type": "HAS", "start_node_id": "H01", "end_node_id": "P02"}}
  ]
}}
"""

def llm_call_structured(prompt: str, model: str = "llama3.1") -> GraphResponse:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt + "\n\nReturn ONLY valid JSON."}]
    )
    text = response["message"]["content"].strip()
    
    # 1. 우리가 정의한 유효한 ID 세트 만들기 (H01, P02, T01 등)
    valid_ids = {v["id"] for v in ALLOWED_NODES.values()}
    # 2. ID를 넣으면 한글 이름을 돌려주는 사전 (이름 보정용)
    id_to_ko_name = {v["id"]: k for k, v in ALLOWED_NODES.items()}

    try:
        # JSON 추출 및 클리닝
        clean_text = re.sub(r"```json|```", "", text).strip()
        json_match = re.search(r"(\{.*\})", clean_text, re.DOTALL)
        if not json_match: raise Exception("No JSON found")
        
        parsed = json.loads(json_match.group(1))
        
        # --- [여기서부터 필터링 핵심 로직] ---
        
        # 3. 노드 필터링: 유효한 ID를 가진 노드만 남김
        raw_nodes = parsed.get("nodes", [])
        filtered_nodes = []
        for n in raw_nodes:
            node_id = n.get("id")
            if node_id in valid_ids:
                # LLM이 이름을 영어로 썼어도 우리 리스트의 한글 이름으로 강제 교체
                n["properties"]["name"] = id_to_ko_name[node_id]
                filtered_nodes.append(n)
        
        # 4. 관계 필터링: 시작점과 끝점 ID가 모두 유효한 리스트에 있을 때만 유지
        raw_rels = parsed.get("relationships", [])
        filtered_rels = []
        for r in raw_rels:
            s_id = r.get("start_node_id")
            e_id = r.get("end_node_id")
            if s_id in valid_ids and e_id in valid_ids:
                filtered_rels.append(r)
        
        # 최종 정제된 결과 반환
        return GraphResponse(nodes=filtered_nodes, relationships=filtered_rels)
        
    except Exception as e:
        print(f"   - Parsing Error: {e}. Raw: {text[:50]}...")
        return GraphResponse(nodes=[], relationships=[])


def fetch_episode_pokemon_s1(link: str) -> List[dict]:
    print(f"📡 수집 시작: {link}")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = get(link, headers=headers)
    soup = bs(response.text, "html.parser")
    
    table = soup.select_one("table.wikitable.plainrowheaders")
    rows = table.select("tr.vevent")
    
    episodes = []
    for i, row in enumerate(rows, start=1):
        title_cell = row.select_one("th.summary") or row.select_one("td.summary")
        
        title = ""
        if title_cell:
            # 에피소드 1처럼 링크가 있는 경우를 위해 모든 텍스트를 "|"로 구분
            text_parts = title_cell.get_text("|", strip=True).split("|")
            
            # 첫 번째 파트에서 따옴표 안의 제목을 찾거나, 따옴표가 없으면 첫 문장
            raw_title = text_parts[0]
            match = re.search(r'"([^"]*)"', raw_title)
            
            if match:
                title = match.group(1)
            else:
                # 따옴표가 없는 경우 (일본어 병기 등) 괄호 앞까지만 split
                title = raw_title.split('(')[0].replace('"', '').strip()

        # 만약 여전히 제목이 비어있다면, 셀 내부의 <a> 태그를 직접 찾게
        if not title and title_cell:
            link_tag = title_cell.find("a")
            if link_tag:
                title = link_tag.get_text(strip=True)

        # 헤더 행 등 불필요한 데이터 필터링
        if not title or title.lower() == "japanese":
            continue

        synopsis = ""
        next_row = row.find_next_sibling("tr")
        if next_row and "expand-child" in next_row.get("class", []):
            syn_cell = next_row.select_one("td.description")
            if syn_cell:
                synopsis = syn_cell.get_text(" ", strip=True)

        episodes.append({
            "season": 1,
            "episode_in_season": i,
            "title": title,
            "synopsis": synopsis,
        })
    
    print(f"✅ 총 {len(episodes)}개 에피소드 수집 완료")
    return episodes

def process_data(episodes: List[dict]) -> GraphResponse:
    chunk_graphs = []
    failed_episodes = []  # 실패한 에피소드를 담을 바구니
    # 1차 시도
    for episode in episodes:
        if not episode["synopsis"]: continue
        print(f"🔍 처리 중: S1E{episode['episode_in_season']:02d} - {episode['title']}")
        prompt = STRICT_TEMPLATE + f"\n\nInput English Synopsis:\n{episode['synopsis']}"
        graph_response = llm_call_structured(prompt)
        # 결과가 비어있으면(Parsing Error 등) 실패 목록에 추가
        if not graph_response.nodes and not graph_response.relationships:
            print(f"   ⚠️ 1차 실패 -> 재시도 목록 추가")
            failed_episodes.append(episode)
            continue
        # 성공 시 메타데이터 추가
        add_metadata(graph_response, episode)
        chunk_graphs.append(graph_response)

    # 2차 재시도 (실패한 것들만 다시!)
    if failed_episodes:
        print(f"\n🔄 [재시도 시작] 총 {len(failed_episodes)}개 에피소드 다시 시도 중...")
        for episode in failed_episodes:
            print(f"🔍 재처리: S1E{episode['episode_in_season']:02d}")
            # 재시도 시에는 프롬프트에 "정확한 JSON 형식을 지켜달라"는 강조를 살짝 더 섞을 수도 있습니다.
            graph_response = llm_call_structured(prompt)
            
            if graph_response.nodes or graph_response.relationships:
                add_metadata(graph_response, episode)
                chunk_graphs.append(graph_response)
                print(f"   ✅ 재시도 성공!")
            else:
                print(f"   ❌ 재시도 최종 실패")

    return combine_chunk_graphs(chunk_graphs)

# 중복 코드를 방지하기 위한 메타데이터 추가 함수
def add_metadata(graph_response, episode):
    episode_tag = f"S1E{episode['episode_in_season']:02d}"
    for rel in graph_response.relationships:
        if rel.properties is None: rel.properties = {}
        rel.properties.update({
            "episode_source": episode_tag,
            "episode_title": episode["title"]
        })

def combine_chunk_graphs(chunk_graphs: List[GraphResponse]) -> GraphResponse:
    all_nodes = {}
    all_rels = []
    for g in chunk_graphs:
        for node in g.nodes:
            all_nodes[node.id] = node
        all_rels.extend(g.relationships)
    return GraphResponse(nodes=list(all_nodes.values()), relationships=all_rels)

def save_output(episodes, final_graph):
    os.makedirs("output", exist_ok=True)
    with open("output/1_포켓몬_원본데이터.json", "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    with open("output/지식그래프_최종.json", "w", encoding="utf-8") as f:
        json.dump(final_graph.model_dump(), f, indent=2, ensure_ascii=False)
    print("💾 저장 완료.")

def main():
    link = "https://en.wikipedia.org/wiki/Pok%C3%A9mon:_Indigo_League"
    episodes = fetch_episode_pokemon_s1(link)
    final_graph = process_data(episodes)
    save_output(episodes, final_graph)
    
    print("\n" + "=" * 50)
    print("✅ 지식그래프 생성 완료!")
    print(f"📊 총 노드 수: {len(final_graph.nodes)}")
    print(f"🔗 총 관계 수: {len(final_graph.relationships)}")
    print("=" * 50)

if __name__ == "__main__":
    main()