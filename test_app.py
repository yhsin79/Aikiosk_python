import requests

# 테스트할 Spring Boot URL
url = "http://localhost:8080/recommend?coffeeName=아메리카노&coffeeImage=/img/iceAmericano.jpg"

try:
    response = requests.get(url)
    print(f"✅ 요청 성공! 상태 코드: {response.status_code}")
    print(f"응답 내용: {response.text}")  # Spring Boot 응답 출력
except requests.exceptions.RequestException as e:
    print(f"❌ 요청 실패: {e}")
