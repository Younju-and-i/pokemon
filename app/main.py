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
    # --- 인물 및 조직 (Humans & Villains) ---
    "지우": {"id": "H01", "label": "Trainer"},
    "이슬이": {"id": "H02", "label": "Trainer"},
    "웅이": {"id": "H03", "label": "Trainer"},
    "로켓단": {"id": "H04", "label": "Villain"},
    "오박사": {"id": "H05", "label": "Human"},
    "간호순": {"id": "H06", "label": "Human"},
    "여경": {"id": "H07", "label": "Human"},

    # --- 포켓몬 (시즌 1 도감 및 진화 트리 반영) ---
    "이상해씨": {"id": "P001", "label": "Pokemon"}, "이상해풀": {"id": "P002", "label": "Pokemon"}, "이상해꽃": {"id": "P003", "label": "Pokemon"},
    "파이리": {"id": "P004", "label": "Pokemon"}, "리자드": {"id": "P005", "label": "Pokemon"}, "리자몽": {"id": "P006", "label": "Pokemon"},
    "꼬부기": {"id": "P007", "label": "Pokemon"}, "어니부기": {"id": "P008", "label": "Pokemon"}, "거북왕": {"id": "P009", "label": "Pokemon"},
    "캐터피": {"id": "P010", "label": "Pokemon"}, "단데기": {"id": "P011", "label": "Pokemon"}, "버터플": {"id": "P012", "label": "Pokemon"},
    "뿔충이": {"id": "P013", "label": "Pokemon"}, "딱충이": {"id": "P014", "label": "Pokemon"}, "독침붕": {"id": "P015", "label": "Pokemon"},
    "구구": {"id": "P016", "label": "Pokemon"}, "피전": {"id": "P017", "label": "Pokemon"}, "피죤투": {"id": "P018", "label": "Pokemon"},
    "피카츄": {"id": "P025", "label": "Pokemon"}, "라이츄": {"id": "P026", "label": "Pokemon"},
    "나옹": {"id": "P052", "label": "Pokemon"}, "고덕": {"id": "P054", "label": "Pokemon"}, "골덕": {"id": "P055", "label": "Pokemon"},
    "가디": {"id": "P058", "label": "Pokemon"}, "윈디": {"id": "P059", "label": "Pokemon"},
    "발챙이": {"id": "P060", "label": "Pokemon"}, "수륙챙이": {"id": "P061", "label": "Pokemon"}, "강챙이": {"id": "P062", "label": "Pokemon"},
    "케이시": {"id": "P063", "label": "Pokemon"}, "윤겔라": {"id": "P064", "label": "Pokemon"}, "후딘": {"id": "P065", "label": "Pokemon"},
    "고스": {"id": "P092", "label": "Pokemon"}, "고우스트": {"id": "P093", "label": "Pokemon"}, "팬텀": {"id": "P094", "label": "Pokemon"},
    "잉어킹": {"id": "P129", "label": "Pokemon"}, "갸라도스": {"id": "P130", "label": "Pokemon"},
    "이브이": {"id": "P133", "label": "Pokemon"}, "샤미드": {"id": "P134", "label": "Pokemon"}, "쥬피썬더": {"id": "P135", "label": "Pokemon"}, "부스터": {"id": "P136", "label": "Pokemon"},
    "메타몽": {"id": "P132", "label": "Pokemon"}, "잠만보": {"id": "P143", "label": "Pokemon"}, 
    "미뇽": {"id": "P147", "label": "Pokemon"}, "신뇽": {"id": "P148", "label": "Pokemon"}, "망나뇽": {"id": "P149", "label": "Pokemon"},
    "뮤츠": {"id": "P150", "label": "Pokemon"}, "뮤": {"id": "P151", "label": "Pokemon"},

    # --- 타입 (Types) ---
    "전기": {"id": "T01", "label": "Type"}, "풀": {"id": "T02", "label": "Type"}, "독": {"id": "T03", "label": "Type"},
    "불꽃": {"id": "T04", "label": "Type"}, "물": {"id": "T05", "label": "Type"}, "비행": {"id": "T06", "label": "Type"},
    "벌레": {"id": "T07", "label": "Type"}, "노말": {"id": "T08", "label": "Type"}, "에스퍼": {"id": "T09", "label": "Type"},
    "고스트": {"id": "T10", "label": "Type"}, "격투": {"id": "T11", "label": "Type"}, "드래곤": {"id": "T12", "label": "Type"}
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
### ROLE
You are a High-Precision Entity-Relationship Extraction System for Pokemon Season 1.
Convert English synopses into a deduplicated Korean JSON Knowledge Graph.

### KNOWLEDGE BASE (SEASON 1 - IMPLICIT)
If these Pokemon appear, ALWAYS include their Type and Evolution nodes/relationships:
- [P001 이상해씨]: Type[T02, T03] | Evolves to: P002
- [P004 파이리]: Type[T04] | Evolves to: P005
- [P007 꼬부기]: Type[T05] | Evolves to: P008
- [P010 캐터피]: Type[T07] | Evolves to: P011 -> P012(Type T06)
- [P025 피카츄]: Type[T01] | Evolves to: P026
- [P133 이브이]: Type[T08] | Evolves to: P134, P135, P136

### ALLOWED NODES (Target Vocabulary)
{NODES_PROMPT_STR}

### STRICT EXTRACTION RULES
1. **RELATIONSHIP TYPES**: Use ONLY these 5 types. NO EXCEPTIONS.
   - `HAS`: Trainer owns a Pokemon.
   - `HAS_TYPE`: Pokemon belongs to a Type.
   - `EVOLVES_TO`: Pokemon evolves into another.
   - `COMPANION`: Trainers traveling together (Ash, Misty, Brock).
   - `BATTLES`: Direct combat between Trainers or Pokemon.

2. **DEDUPLICATION**: 
   - Within a single episode, if the same relationship occurs multiple times, **return it only ONCE**.
   - Do not create redundant strings like "TRAVELED_WITH". Map it to `COMPANION`.

3. **LOCALIZATION**: 
   - Map: Ash->지우, Misty->이슬이, Brock->웅이, Team Rocket->로켓단.
   - Properties 'name' MUST be in Korean as defined in ALLOWED NODES.

4. **FILTERING**:
   - Only use IDs from the ALLOWED NODES list.
   - If an entity is not in the list, skip it entirely.
5. **NO LISTS IN PROPERTIES**: 
   - Property values must be a SINGLE string or number.
   - BAD: "Type": ["T01"]
   - GOOD: "Type": "T01"
6. **NO INFERENCE / NO HALLUCINATION**: 
   - Extract relationships ONLY if they are explicitly stated in the text.
   - Do not assume a Trainer 'HAS' a Pokemon just because they are in the same scene. 
   - Do not invent backstories or evolutions not mentioned in the synopsis or the IMPLICIT KNOWLEDGE BASE.
   
7. **WHEN IN DOUBT, SKIP**: 
   - If the relationship between two entities is ambiguous or not clearly defined by the 5 allowed types, DO NOT extract it. 
   - Accuracy is more important than quantity.
   
### OUTPUT FORMAT
- Return ONLY raw JSON. No conversational filler.
- Example:
{{
  "nodes": [{{"id": "H01", "label": "Trainer", "properties": {{"name": "지우"}}}}],
  "relationships": [{{"type": "HAS", "start_node_id": "H01", "end_node_id": "P025"}}]
}}
"""

def llm_call_structured(prompt: str, model: str = "llama3.1") -> GraphResponse:
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt + "\n\nIMPORTANT: Output ONLY JSON."}]
    )
    text = response["message"]["content"].strip()
    
    valid_ids = {v["id"] for v in ALLOWED_NODES.values()}
    id_to_ko_name = {v["id"]: k for k, v in ALLOWED_NODES.items()}

    try:
        # --- [보정 1] JSON 구간만 정규식으로 강제 추출 (설명글 제거) ---
        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if not json_match: raise Exception("No JSON found")
        
        parsed = json.loads(json_match.group(1))
        
        # --- [보정 2] 노드 속성값 리스트 에러 방지 (Pydantic 에러 해결) ---
        raw_nodes = parsed.get("nodes", [])
        filtered_nodes = []
        for n in raw_nodes:
            node_id = n.get("id")
            if node_id in valid_ids:
                props = n.get("properties", {})
                # 리스트(['T01'])로 들어오면 문자열('T01')로 강제 변환
                clean_props = {k: (v[0] if isinstance(v, list) else v) for k, v in props.items()}
                n["properties"] = clean_props
                n["properties"]["name"] = id_to_ko_name[node_id]
                filtered_nodes.append(n)
        
        # --- [보정 3] 관계 데이터 필터링 ---
        raw_rels = parsed.get("relationships", [])
        filtered_rels = []
        for r in raw_rels:
            if r.get("start_node_id") in valid_ids and r.get("end_node_id") in valid_ids:
                # 관계 속성도 리스트 방지 로직 적용
                r_props = r.get("properties", {}) or {}
                r["properties"] = {k: (v[0] if isinstance(v, list) else v) for k, v in r_props.items()}
                filtered_rels.append(r)
        
        return GraphResponse(nodes=filtered_nodes, relationships=filtered_rels)
        
    except Exception as e:
        print(f"   - 파싱 재시도 필요: {e}")
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
    unique_rels = {}  # (start, end, type)을 키로 사용하여 중복 제거

    for g in chunk_graphs:
        # 노드 병합 (ID 기준)
        for node in g.nodes:
            all_nodes[node.id] = node
        
        # 관계 병합 및 출처 누적
        for rel in g.relationships:
            # 중복 체크용 키 생성
            rel_key = (rel.start_node_id, rel.end_node_id, rel.type)
            
            if rel_key not in unique_rels:
                # 처음 보는 관계라면 등록
                # 출처 정보를 리스트로 변환하여 관리
                source = rel.properties.get("episode_source")
                rel.properties["episode_source"] = [source] if source else []
                unique_rels[rel_key] = rel
            else:
                # 이미 있는 관계라면 출처 정보만 추가 (중복 방지)
                new_source = rel.properties.get("episode_source")
                existing_rel = unique_rels[rel_key]
                
                if new_source and new_source not in existing_rel.properties["episode_source"]:
                    existing_rel.properties["episode_source"].append(new_source)

    return GraphResponse(nodes=list(all_nodes.values()), relationships=list(unique_rels.values()))

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