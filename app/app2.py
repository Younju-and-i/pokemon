from neo4j import GraphDatabase # neo4j 패키지에서 GraphDatabase 모듈을 임포트 (Neo4j 데이터베이스 연결용)
from neo4j_graphrag.retrievers import Text2CypherRetriever # neo4j-graphrag 패키지에서 Text2CypherRetriever를 임포트 (자연어 -> Cypher 변환용)
from neo4j_graphrag.llm.ollama_llm import OllamaLLM # neo4j-graphrag 패키지에서 OllamaLLM 래퍼를 임포트 (Ollama LLM 연결용)
import ollama # Ollama API 호출용 패키지
import re # 문자열 처리용 정규표현식
from settings import settings

# -------------------------
# 🎯 Ollama Client 설정
# -------------------------
client = ollama.Client(host=settings.ollama_host) # Ollama 서버에 접속하기 위한 클라이언트 객체 생성, 호스트와 포트 지정
modelName = "gemma3:4b" # Ollama 모델 이름 지정 (사용할 LLM 모델)

# -------------------------
# 🎯 Neo4j 연결 설정
# -------------------------
URI = settings.neo4j_uri # Neo4j 데이터베이스 URI 지정
AUTH = (settings.neo4j_user, settings.neo4j_password) # Neo4j 로그인 정보 (username, password)
driver = GraphDatabase.driver(URI, auth=AUTH) # Neo4j 드라이버 객체 생성, 데이터베이스 연결 준비

# -------------------------
# 🎯 Ollama용 LLM Wrapper 정의
#    neo4j-graphrag이 OpenAI 포맷을 기대하므로 래퍼 필요
# -------------------------
class ConnLLM(OllamaLLM):
  # 초기화 메서드에서 모델 이름 지정
  def __init__(self, model_name=modelName):
    super().__init__(model_name=model_name)  # 부모 클래스 초기화
    self.model_name = model_name             # 모델 이름 속성 저장

  # Text2CypherRetriever가 사용하는 기본 completion 함수 정의
  def complete(self, prompt: str) -> str:
    """ Text2CypherRetriever가 사용하는 기본 completion 함수 """
    # Ollama client로 prompt 전달하여 모델로부터 응답 받기
    response = client.generate(
      model=self.model_name,
      prompt=prompt
    )
    # Ollama 응답에서 'response' 필드 반환
    return response["response"]

  # 필요 시 Chat 형식 메시지 처리 함수 정의
  def chat(self, messages):
    """ 필요 시 Chat 형식도 처리 """
    # 메시지들을 "role: content" 형태로 문자열로 변환
    prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    # Ollama client로 prompt 전달하여 모델로부터 응답 받기
    response = client.generate(
      model=self.model_name,
      prompt=prompt
    )
    # Ollama 응답 반환
    return response["response"]

# ConnLLM 인스턴스 생성, 이후 retriever에서 사용
llm = ConnLLM(model_name=modelName)

# -------------------------
# 🎯 Cypher 변환 예시 데이터
# -------------------------
examples = [
  # 예시: 사용자 질문 -> Cypher 쿼리 매핑
  "USER INPUT: '토미오카 기유는 시즌 1에서 어떤 역할을 했는지 에피소드별로 알려줘.' QUERY: MATCH (n {name: '토미오카 기유'})-[r]-(m) RETURN n, r, m, properties(r) AS rel_props ORDER BY r.episode_number"
]

# Text2CypherRetriever 생성
# 자연어 질문을 받아 Cypher 쿼리를 생성하고 DB에서 결과 조회
retriever = Text2CypherRetriever(
  driver=driver,  # Neo4j 드라이버
  llm=llm,        # LLM 인스턴스
  examples=examples,  # 예시 쿼리
)

# -------------------------
# 🎯 Ollama로 LLM 호출용 함수
# -------------------------
def llm_cal(prompt: str, model: str = modelName) -> str:
  # Ollama client에 prompt 전달 후 모델 응답 받기
  response = client.generate(model=model, prompt=prompt)
  # response가 없으면 빈 문자열 반환
  return response.get("response", "")

# -------------------------
# 🎯 전체 파이프라인 함수 정의
# -------------------------
def graphrag_pipeline(user_question):
  # 자연어 질문을 retriever로 전달하여 Cypher 변환 및 DB 조회
  result = retriever.search(query_text=user_question)

  # 생성된 Cypher 쿼리 가져오기 (메타데이터)
  cypher_used = result.metadata.get("cypher")
  print("생성된 Cypher Query:")
  print(cypher_used)  # 디버그용 출력

  # DB에서 조회된 결과 아이템 가져오기
  result_items = result.items
  print("지식그래프에 찾은 결과")
  # print(result_items)  # 디버그용 출력

  # DB 결과를 자연어 요약용 리스트로 변환
  context_list = []
  for item in result_items:
    raw = str(item.content)  # 아이템 내용 문자열로 변환
    # element_id 같은 내부 정보 제거
    cleaned = re.sub(r"element_id='[^']*'\s*", "", raw)
    context_list.append(cleaned)

  # 리스트를 하나의 문자열로 합치기
  full_context = "\n".join(context_list)

  # 최종 LLM 프롬프트 생성
  full_prompt = f"""
  아래의 데이터베이스 결과만을 참고하여 사용자의 질문에 답변해주세요.  
  데이터베이스 결과를 그대로 노출하지 말고 자연스러운 서술로 정리해 주세요.  

  사용자 질문: {user_question}  
  데이터베이스 결과: {full_context}  

  ### 조건
  - 그래프DB의 관계명(예: DEFENDS, SIBLING_OF 등)은 그대로 쓰지 말고 자연스럽게 서술.
  - 에피소드별 사건은 간단하고 이해하기 쉽게 요약.
  - 스토리텔링 형식으로 자연스럽게 작성.
  """

  # print("완성 프롬프트")
  # print(full_prompt)  # 디버그용 출력

  # Ollama LLM 호출하여 최종 답변 생성
  final_result = llm_cal(full_prompt)
  return final_result

# -------------------------
# 🎯 메인 실행부
# -------------------------
if __name__ == "__main__":
  # 테스트할 질문 리스트
  queries = [
    "네즈코는 모든시즌에 에피소드별로 어떤 활약을 했어?",
  ]
  
  # 질문 리스트를 순회하며 파이프라인 실행
  for query in queries:
    print(query)            # 질문 출력
    print("-"*100)          # 구분선 출력
    print(graphrag_pipeline(query))  # 파이프라인 실행 후 결과 출력
    print("-"*100)          # 구분선 출력
