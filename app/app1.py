import json  # JSON 파일을 읽고 쓰기 위해 필요
import asyncio  # 비동기 함수 실행을 위해 필요
from neo4j import GraphDatabase  # Neo4j 드라이버를 사용하여 데이터베이스와 연결
from pydantic import validate_call  # 함수 호출 시 입력값 검증을 위해 사용
from neo4j_graphrag.experimental.components.types import (
  Neo4jGraph,  # 그래프 전체를 표현하는 타입
  Neo4jNode,   # 그래프 노드 타입
  Neo4jRelationship,  # 그래프 관계 타입
)
from neo4j_graphrag.experimental.components.kg_writer import KGWriter, KGWriterModel  # 지식 그래프 쓰기 기능 제공
from settings import settings

# 사용자 정의 KGWriter 클래스: 관계를 MERGE가 아닌 CREATE로 생성
class Neo4jCreateWriter(KGWriter):
  """관계에 대해 MERGE 대신 CREATE를 사용하는 Custom KGWriter (에피소드별로 다른 관계도 반영)"""

  def __init__(self, driver, neo4j_database=None):
    self.driver = driver  # Neo4j 드라이버 객체
    self.neo4j_database = neo4j_database  # 사용할 데이터베이스 이름 (기본값 None)

  # 데이터베이스 초기화 함수: 모든 노드와 관계 삭제
  def _wipe_database(self) -> None:
    self.driver.execute_query(
      "MATCH (n) DETACH DELETE n",  # 모든 노드와 연결된 관계를 삭제
      database_=self.neo4j_database,  # 지정된 데이터베이스 사용
    )

  # 그래프를 Neo4j에 쓰는 메인 함수
  @validate_call  # 입력값(graph)이 Neo4jGraph 타입인지 검증
  async def run(self, graph: Neo4jGraph) -> KGWriterModel:
    try:
      self._wipe_database()  # 기존 데이터 삭제
      with self.driver.session(database=self.neo4j_database) as session:  # 세션 생성

        # 모든 노드를 Neo4j에 추가 (MERGE 사용: 노드가 없으면 생성, 있으면 업데이트)
        for node in graph.nodes:
          labels = f":{node.label}"  # 노드 레이블 문자열 생성
          session.run(
            f"""
            MERGE (n{labels} {{id: $id}})
            SET n += $props
            """,  # id를 기준으로 노드 병합 # 노드 속성 업데이트
            {"id": node.id, "props": node.properties or {}},  # 파라미터 전달
          )

        # 모든 관계를 Neo4j에 추가 (CREATE 사용: 항상 새 관계 생성)
        for rel in graph.relationships:
          session.run(
            f"""
            MATCH (a {{id: $start_id}}), (b {{id: $end_id}})  
            CREATE (a)-[r:{rel.type} $props]->(b)  
            """, # 시작, 끝 노드 찾기 # 관계 생성
            {
              "start_id": rel.start_node_id,  # 관계 시작 노드 ID
              "end_id": rel.end_node_id,      # 관계 끝 노드 ID
              "props": rel.properties or {},  # 관계 속성
            },
          )

      # 성공적으로 작성 완료 시 상태 반환
      return KGWriterModel(
        status="SUCCESS",  # 성공 상태
        metadata={
          "node_count": len(graph.nodes),  # 생성된 노드 수
          "relationship_count": len(graph.relationships),  # 생성된 관계 수
        },
      )
    except Exception as e:
      # 오류 발생 시 상태와 오류 메시지 반환
      return KGWriterModel(status="FAILURE", metadata={"error": str(e)})

# 비동기 함수: 그래프를 Neo4j에 쓰기 위한 진입점
async def write_to_neo4j(graph: Neo4jGraph):
  uri = settings.neo4j_uri  # Neo4j 연결 URI
  user = settings.neo4j_user  # 사용자명
  password = settings.neo4j_password  # 비밀번호
  driver = GraphDatabase.driver(uri, auth=(user, password))  # 드라이버 생성
  
  writer = Neo4jCreateWriter(driver)  # 사용자 정의 KGWriter 생성
  result = await writer.run(graph)  # 그래프 작성 실행
  print(result)  # 결과 출력


if __name__ == "__main__":
  # JSON 파일에서 그래프 데이터 로드
  with open("output/지식그래프_최종.json", "r", encoding="utf-8") as f:
    data = json.load(f)  # JSON 파일 읽기

  # JSON 노드를 Neo4jNode 객체로 변환
  nodes = [Neo4jNode(**node) for node in data["nodes"]]
  # JSON 관계를 Neo4jRelationship 객체로 변환 (관계가 없으면 빈 리스트)
  relationships = [Neo4jRelationship(**rel) for rel in data.get("relationships", [])]
  # 그래프 객체 생성
  graph = Neo4jGraph(nodes=nodes, relationships=relationships)

  # 비동기 함수 실행
  asyncio.run(write_to_neo4j(graph))