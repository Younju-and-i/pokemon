import ollama
import json
import re
import os
from typing import List, Optional, Dict, Union
from requests import get
from bs4 import BeautifulSoup as bs
from pydantic import BaseModel

# 속성 타입 정의 (문자열, 숫자, bool, None)
PropertyValue = Union[str, int, float, bool, None]

# ---------------------------
# 지식 그래프 기본 모델 정의
# ---------------------------
class Node(BaseModel):
  id: str  # 노드 ID (예: N0)
  label: str  # 노드 타입 (예: "인간")
  properties: Optional[Dict[str, PropertyValue]] = None  # 속성 딕셔너리

class Relationship(BaseModel):
  type: str  # 관계 유형
  start_node_id: str  # 시작 노드 ID
  end_node_id: str  # 끝 노드 ID
  properties: Optional[Dict[str, PropertyValue]] = None  # 관계 속성

class GraphResponse(BaseModel):
  nodes: List[Node]  # 노드 리스트
  relationships: List[Relationship]  # 관계 리스트

# ----------------------------------------
# LLM에 전달되는 템플릿: 노드와 관계 추출 규칙
# ----------------------------------------
UPDATED_TEMPLATE = """
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph. Extract the entities (nodes) and specify their type from the following text, but **you MUST select nodes ONLY from the following predefined set** (see the provided NODES list below). Do not create any new nodes or use names that do not exactly match one in the NODES list.

Also extract the relationships between these nodes. Return the result as JSON using the following format:

{
  "nodes": [
    {"id": "N0", "label": "인간", "properties": {"name": "Tanjiro Kamado"}}
  ],
  "relationships": [
    {"type": "FIGHTS", "start_node_id": "N0", "end_node_id": "N13", "properties": {"outcome": "victory"}}
  ]
}

Additional rules:
- Use only nodes from the NODES list. Do not invent or substitute nodes.
- Skip any relationship if one of its entities is not in NODES.
- Only output valid relationships where both endpoints exist in NODES and the direction matches their types.

NODES =
[
  {"id":"N0",  "label":"인간", "properties":{"name":"Tanjiro Kamado"}},
  {"id":"N1",  "label":"인간", "properties":{"name":"Nezuko Kamado"}},
  {"id":"N2",  "label":"인간", "properties":{"name":"Giyu Tomioka"}},
  {"id":"N3",  "label":"인간", "properties":{"name":"Sakonji Urokodaki"}},
  {"id":"N4",  "label":"인간", "properties":{"name":"Sabito"}},
  {"id":"N5",  "label":"인간", "properties":{"name":"Makomo"}},
  {"id":"N6",  "label":"인간", "properties":{"name":"Zenitsu Agatsuma"}},
  {"id":"N7",  "label":"인간", "properties":{"name":"Inosuke Hashibira"}},
  {"id":"N8",  "label":"인간", "properties":{"name":"Kanao Tsuyuri"}},
  {"id":"N9",  "label":"인간", "properties":{"name":"Kyojuro Rengoku"}},
  {"id":"N10", "label":"인간", "properties":{"name":"Kagaya Ubuyashiki"}},
  {"id":"N11", "label":"인간", "properties":{"name":"Shinobu Kocho"}},
  {"id":"N12", "label":"인간", "properties":{"name":"Sanemi Shinazugawa"}},
  {"id":"N13", "label":"도깨비", "properties":{"name":"Muzan Kibutsuji"}},
  {"id":"N14", "label":"도깨비", "properties":{"name":"Susamaru"}},
  {"id":"N15", "label":"도깨비", "properties":{"name":"Yahaba"}},
  {"id":"N16", "label":"도깨비", "properties":{"name":"Kyogai"}},
  {"id":"N17", "label":"도깨비", "properties":{"name":"Rui"}},
  {"id":"N18", "label":"도깨비", "properties":{"name":"Enmu"}}
]
"""

# 영어 이름 → 한글 이름 변환 매핑 테이블
KOREAN_NODE_MAP = {
  "Tanjiro Kamado": "카마도 탄지로",
  "Nezuko Kamado": "카마도 네즈코",
  "Giyu Tomioka": "토미오카 기유",
  "Sakonji Urokodaki": "우로코다키 사콘지",
  "Sabito": "사비토",
  "Makomo": "마코모",
  "Zenitsu Agatsuma": "아가츠마 젠이츠",
  "Inosuke Hashibira": "하시비라 이노스케",
  "Kanao Tsuyuri": "츠유리 카나오",
  "Kyojuro Rengoku": "렌고쿠 쿄쥬로",
  "Kagaya Ubuyashiki": "우부야시키 카가야",
  "Shinobu Kocho": "코쵸우 시노부",
  "Sanemi Shinazugawa": "시나즈가와 사네미",
  "Muzan Kibutsuji": "키부츠지 무잔",
  "Susamaru": "스사마루",
  "Yahaba": "야하바",
  "Kyogai": "쿄우가이",
  "Rui": "루이",
  "Enmu": "엔무",
}

# ---------------------------
# Ollama LLM 호출 함수
# ---------------------------
def llm_call_structured(prompt: str, model: str = "gemma3:4b") -> GraphResponse:

  final_prompt = prompt + """
  Return ONLY valid JSON. Do NOT include explanations or commentary.
  """

  # Ollama에 LLM 요청
  response = ollama.chat(
    model=model,
    messages=[{"role": "user", "content": final_prompt}]
  )

  # 모델 응답 텍스트 추출
  text = response["message"]["content"]

  # JSON 파싱 시도
  try:
    parsed = json.loads(text)
    print(parsed)
  except json.JSONDecodeError:
    # 전체 텍스트에서 JSON 블록만 추출
    json_text = re.search(r"\{.*\}", text, re.S)
    if not json_text:
      raise Exception("모델 응답에서 JSON을 찾지 못했습니다.")
    parsed = json.loads(json_text.group(0))
  
  return GraphResponse(**parsed)  # pydantic 모델로 변환 후 반환

# ------------------------------------------------------
# 여러 에피소드 그래프를 통합하기 위한 함수
# ------------------------------------------------------
def combine_chunk_graphs(chunk_graphs: list) -> GraphResponse:
  all_nodes = []  # 모든 노드를 담을 리스트
  for chunk_graph in chunk_graphs:
    for node in chunk_graph.nodes:
      all_nodes.append(node)
  
  all_relationships = []  # 모든 관계를 담을 리스트
  for chunk_graph in chunk_graphs:
    for relationship in chunk_graph.relationships:
      all_relationships.append(relationship)
  
  unique_nodes = []  # 중복 제거된 최종 노드 리스트
  seen = set()  # 노드 중복 체크용

  for node in all_nodes:
    node_key = (node.id, node.label, str(node.properties))  # 노드 고유값 생성
    if node_key not in seen:
      unique_nodes.append(node)
      seen.add(node_key)

  return GraphResponse(nodes=unique_nodes, relationships=all_relationships)

# ------------------------------------------------------
# 수집된 데이터를 LLM으로 처리하여 그래프 생성
# ------------------------------------------------------
def process_data(episodes: List[dict]) -> GraphResponse:
  print("=== 데이터 처리 시작 ===")

  chunk_graphs: List[GraphResponse] = []  # 에피소드별 그래프 저장
    
  for episode in episodes:
    if not episode.get("synopsis"):
      print(f"에피소드 S{episode['season']}E{episode['episode_in_season']:02d}: 시놉시스가 없어 건너뜀")
      continue
        
    print(f"에피소드 처리 중: 시즌 {episode['season']}, 에피소드 {episode['episode_in_season']}")
    
    try:
      prompt = UPDATED_TEMPLATE + f"\n 입력값\n {episode['synopsis']}"  # LLM 입력 프롬프트
      graph_response = llm_call_structured(prompt)  # LLM 호출

      episode_number = f"S{episode['season']}E{episode['episode_in_season']:02d}"  # 에피소드 번호 문자열

      for relationship in graph_response.relationships:
        if relationship.properties is None:
          relationship.properties = {}
        relationship.properties["episode_number"] = episode_number  # 관계에 에피소드 번호 부여
          
      for node in graph_response.nodes:
        english_name = node.properties.get("name", "")
        if english_name in KOREAN_NODE_MAP:
          node.properties["name"] = KOREAN_NODE_MAP[english_name]  # 영어 → 한글 변환
      
      chunk_graphs.append(graph_response)  # 결과 저장
        
    except Exception as e:
      print(f"  - 에피소드 처리 중 오류 발생: {e}")
      continue
  
  if not chunk_graphs:
    raise Exception("그래프를 성공적으로 추출하지 못했습니다.")
  
  print(f"총 {len(chunk_graphs)}개 에피소드 처리 완료")
  return combine_chunk_graphs(chunk_graphs)  # 전체 그래프 통합

# ------------------------------------------------------
# 위키피디아 에피소드 데이터 수집
# ------------------------------------------------------
def fetch_episode(link: str) -> List[dict]:
  season = int(re.search(r"season_(\d+)", link).group(1))  # 시즌 번호 추출
  print(f"Fetching Season {season} from: {link}")
  headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}  # 요청 헤더
  response = get(link, headers=headers)  # GET 요청
  
  soup = bs(response.text, "html.parser")  # HTML 파싱
  table = soup.select_one("table.wikitable.plainrowheaders.wikiepisodetable")  # 에피소드 테이블 찾기

  episodes = []
  rows = table.select("tr.vevent.module-episode-list-row")  # 각 에피소드 row

  for i, row in enumerate(rows, start=1):  # 에피소드 번호 생성
    synopsis = None
    synopsis_row = row.find_next_sibling("tr", class_="expand-child")  # 시놉시스 row 찾기
    if synopsis_row:
      synopsis_cell = synopsis_row.select_one("td.description div.shortSummaryText")
      synopsis = synopsis_cell.get_text(strip=True) if synopsis_cell else None

    episodes.append({
      "season": season,
      "episode_in_season": i,
      "synopsis": synopsis,
    })
  
  return episodes

# ------------------------------------------------------
# 출력 파일 저장
# ------------------------------------------------------

def save_output(episodes: List[dict], final_graph: GraphResponse):
  print("=== 결과 저장 ===")
  
  os.makedirs("output", exist_ok=True)  # output 폴더 생성
  
  with open("output/1_원본데이터.json", "w", encoding="utf-8") as f:
      json.dump(episodes, f, indent=2, ensure_ascii=False)
  print("원본 데이터 저장: output/1_원본데이터.json")
  
  with open("output/지식그래프_최종.json", "w", encoding="utf-8") as f:
      json.dump(final_graph.model_dump(), f, ensure_ascii=False, indent=2)
  print("최종 지식그래프 저장: output/지식그래프_최종.json")

# ------------------------------------------------------
# 메인 실행 함수
# ------------------------------------------------------
def main():
  try:
    episode_links = [
      "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_1",
      "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_2",
      "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_3",
      "https://en.wikipedia.org/wiki/Demon_Slayer:_Kimetsu_no_Yaiba_season_4",
    ]
    all_episodes = []
    for link in episode_links:
      try:
        episodes = fetch_episode(link)
        all_episodes.extend(episodes)
      except Exception as e:
        print(f"Error fetching data from {link}: {e}")
        continue
    print(f"총 {len(all_episodes)}개 에피소드 수집 완료")

    final_graph = process_data(all_episodes)

    save_output(episodes, final_graph)  # 결과 저장
        
    print("=" * 50)
    print("✅ 지식그래프 생성 완료!")
    print(f"📊 총 노드 수: {len(final_graph.nodes)}")
    print(f"🔗 총 관계 수: {len(final_graph.relationships)}")    
  except Exception as e:
    print(f"오류 발생: {e}")
    return 1
  return 0

# ------------------------------------------------------
# 프로그램 실행
# ------------------------------------------------------
if __name__ == "__main__":
  exit(main())
