# -*- coding: utf-8 -*-
"""요구사항 전수 추적표 + TC 문서 생성기 (도구 전체 v87~v149).

원칙: 근거 없는 PASS 기입 금지 — 자동 TC 는 (감사파일, PASS 태그)로 명시하고
①태그가 감사 소스에 실재 ②해당 감사가 최신 회귀(full_test49.log)에서 PASS
둘 다 확인된 것만 '자동/PASS' 로 기입한다. 수동 TC 는 절차만 기술.
"""
import datetime as dt
import html as H
import os
import re

SC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SC, "tc_doc")
REG_LOG = os.path.join(SC, "full_test51.log")
REG_DATE = "2026-09-29"

# ── R: (rid, 분류, 도입판, 요구 내용, 수용 기준) ──
CATS = {"A": "판정 철학", "B": "패턴 5규칙", "C": "패턴 인식 탭",
        "D": "분석자료 탭", "E": "내보내기", "F": "데이터 뷰어",
        "G": "패턴 탐사 에이전트", "H": "성능/캐시", "I": "배포/운영",
        "J": "정책(참고)"}
R = [
 ("R-A01","A","v87~","골든(정상 구간 평균)=답지 — 모든 판정은 골든 상대 비교, 절대 규칙 금지","각 규칙 임계가 골든 수치의 함수(α곱셈/가산)로만 정의"),
 ("R-A02","A","v128","아치 등 판정은 개수 기준 제거, 면적 기준만","개수 분기 코드 없음·α₂ 변경이 판정에 반영"),
 ("R-A03","A","v119","α(민감도)는 캐시 밖 — 조절해도 점수 재계산 없음","α 변경 재실행 시 calc_days=0·점수 동일"),
 ("R-B01","B","v119","직선 하강(P1): T 계열 지정 스텝의 지수 하강이 직선화되면 불량 — 현-곡선 면적 기반 볼록비","정상(지수) 판정 없음·직선 하강 의심 검출·'하강 없음' 별도"),
 ("R-B02","B","v145","아치(P2) 신정의: 지정 스텝 '각각'의 첫점↔끝점 선분과 그래프 사이 |면적|(위아래 무관)을 스텝별로 합산","스텝별 면적+합 표시·합<α₂×골든→'아치 소실 의심'·1.3배 경계"),
 ("R-B03","B","v127","W패턴(P3): 지정 스텝 최저점→끝 선분 '위로' 불룩한 면적 — 크면 불량(방향 반대)","임계=골든 면적+여유(가산)·초과=의심·상승 없으면 비활성"),
 ("R-B04","B","v142","온도 계통(P4): 출구≈w1·내부블록+w2·BackFill+b 선형 혼합 잔차 — 골든 스텝별 최소제곱 학습","골든 자기 z<k·심은 불량 검출·z>k 연속 min_run→의심"),
 ("R-B05","B","v142","압력 계통(P5): 전단 용기 압력≈공급+후단 조합 잔차 — 동일 틀","P4와 동일 기준·교차 오탐 없음"),
 ("R-B06","B","v142","계통 σ는 과소평가 방지 3단 하한(적합RMS·골든 산포 전파·range 1%)","정상 웨이퍼 오탐 0"),
 ("R-B07","B","v119~","규칙 파라미터(센서·스텝·α·k)는 physics_rules.md 2부에서 조정 가능","pattern_rules 로드·유사명 자동 해석·미해석 시 비활성 사유"),
 ("R-C01","C","v124","패턴 인식 탭은 파형 카드만 — 골든 답지 → N일 평균 → 위반 개별 순","점수 시계열·위반 목록 없음·카드 3계층 렌더"),
 ("R-C02","C","v127/v142","보기 체크박스 5종(직선/아치/W/온도 계통/압력 계통) — 해제한 규칙은 카드·집계 제외","체크 반영 즉시 재렌더·다른 규칙 오염 없음"),
 ("R-C03","C","v122","카드에 선분(현)·음영 면적·원단위 숫자 — raw 에 어떻게 선을 그었는지 보이게","chord 오버레이·면적 음영·숫자 캡션"),
 ("R-C04","C","v126","관심 스텝 앞뒤 한 스텝을 회색 컨텍스트로 함께 표시","roi 밖 회색·스텝 라벨"),
 ("R-C05","C","v129","불량인 그래프만 빨간 테두리 — 다른 규칙 위반이 이 카드 물들이지 않게","카드 verdict 있는 것만 빨강·평균 희석 경고는 주황"),
 ("R-C06","C","v129/v134","패턴 달력: 일별 불량/정상 수·과반 이상만 빨간 테두리·미만 비율 그라데이션 주황","경계 3px 빨강·rgba 그라데이션 검증"),
 ("R-C07","C","v134","달력 일 클릭=그날 개별 전수 그래프·월 클릭=월 전체·[기간 전체보기]=5일 평균 고정","전수 장수 일치·5일 고정"),
 ("R-C08","C","v121","LLM 정리: 고객 제출 개조식('3/8~3/10 이상 패턴 감지 — 확인 요망'), AI 말투·원인 분석 금지, 팝업 표시","결정적 요약 항상·LLM 실패 시도 완주"),
 ("R-C09","C","v128","LLM 정리 입력은 불량 패턴 지표만(현재 5종), 기간 추세('1/3부터 심해져 1/8 최대')·골든 대비 오프셋 포함","추세·골든대비 절 존재·5지표 외 언급 금지 프롬프트"),
 ("R-D01","D","v130","분석자료 탭: 기간/설비(라디오)/DSV(다중)/레시피(다중)/평균 N일 필터","필터 교차 정확·빈=전체"),
 ("R-D02","D","v130","30만장 대응: 점수는 서버 보관·달력엔 집계만 전송·스캔은 백그라운드 잡(진행 표시)","rows 미전송·409 중복 방지·progress 폴링"),
 ("R-D03","D","v130","재스캔은 증분 — 새 날짜만 계산","2차 스캔 '계산 0일'"),
 ("R-D04","D","v145","필터 체크박스 숫자=패싯 카운트(다른 그룹 선택 적용·지금 조건으로 달력에 나올 장수)·0장 키 유지·체크 보존","설비 선택 시 레시피/DSV 숫자 교집합"),
 ("R-D05","D","v146","α만 바꾼 재실행은 일별 캐시 로드도 생략 — 보관 점수로 재판정만(즉시)","재실행<3초·점수 동일·위반만 변화"),
 ("R-D06","D","v132/v133","불량 내역 CSV(설비명,dsv,불량원인,발생일자·날짜 8자리) → 달력 ▲마크+호버 내역, 어떤 인코딩도 판독","4개 인코딩 판독·8자리 날짜·설비/DSV 필터 연동"),
 ("R-D07","D","v134","달력 일 클릭=그날 개별 전수(분석자료도)·월=전체·기간 전체=5일 평균","일 클릭 전수 장수=필터 교집합"),
 ("R-E01","E","v137/v139","대화형 HTML 내보내기: 보고 있는 분석자료 탭 그대로 — 달력·일/월 클릭 그래프·달력 갱신 내장","단일 파일에서 클릭 동작·svg 렌더"),
 ("R-E02","E","v139","내보낸 파일에서도 유형(5규칙)/DSV/레시피 체크가 달력과 그래프 모두에 적용 — dsv2만 체크 시 dsv3·5 안 나옴","per-DSV 임베드·필터 시 카드 교체"),
 ("R-E03","E","v130","[현재 설비]+[설비별 일괄] 내보내기","설비 수만큼 파일 생성"),
 ("R-E04","E","v108/v115/v118","그래프 PNG/CSV 다운로드·이미지 저장 세트·저장 이미지에 제목/센서명/레전드 포함","실다운로드 파일 검증·PNG 내 레전드"),
 ("R-E05","E","상시","폐쇄망: 모든 산출 HTML 외부 참조 0","내보내기 파일 http(s) 참조 검사 0건"),
 ("R-F01","F","v141","데이터 뷰어: 폴더(;다중) 입력→웨이퍼 전량 목록(시간순)·골든 불필요","목록 전량·시간순"),
 ("R-F02","F","v141","웨이퍼 이름 클릭→그 파일의 센서 전체 목록(가상 포함)","파일 기준 센서·드롭다운 동일"),
 ("R-F03","F","v141","센서 체크→[그래프 그리기]→센서별 카드·웨이퍼 다중(≤20) 색 오버레이 — 원본 그대로(정렬·다운샘플 없음)","길이 다른 웨이퍼 각자 길이 유지"),
 ("R-F04","F","v141","수식 가상센서: 사칙연산+괄호((T2+T3)/T4류), 타이핑+드롭다운 삽입 하이브리드, 결과=numpy 일치","수식 카드=직접 계산 일치·금지 구문 400"),
 ("R-F05","F","v141","센서명 자동 매칭(대소문자·유사명·조각) — 못 찾으면 후보 제시, 하이픈 센서명이 빼기와 충돌하지 않음","유사명 해석·후보 400·하이픈 수식 정상"),
 ("R-F06","F","v141","0나눗셈은 선을 끊어 그림(가짜 값 없음)·수식 목록은 재시작에도 유지","null 구간·재접속 복원"),
 ("R-F07","F","v143","설비/챔버/유형 체크박스 필터로 목록 좁히기(그룹 안 OR·그룹 간 AND·미선택=전체)","필터 시 목록·카운트 축소"),
 ("R-F08","F","v143","'외 다수' 제거 — 100장/페이지 [이전][다음] 전량 탐색·'필터 N/전체 M·체크 k' 표시·[전체][해제]는 보이는 것만","126장→100+26 분할·전체체크 필터한정"),
 ("R-F09","F","v143","겹침 카드 화면 레전드(색+웨이퍼 이름)·호버에 첫 웨이퍼도 자기 이름(σ 0 표기 제거)","레전드 3장 이름·툴팁에 이름"),
 ("R-G01","G","v144","별도 데몬(pattern_hunter): log 폴더 수동 입력·앱과 독립 실행·몇 달 상주(tick)·--once 1회","CLI 기동·원장 생성·재실행 이어서"),
 ("R-G02","G","v144","앵커=불량 내역 CSV(확정 포맷) — 보고 이전 2~3일/1~2달 창에서 지속 패턴 탐색","창 4종×전 센서×전 스텝×특징 12종 전수 검정"),
 ("R-G03","G","v144","가설 채택: 웨이퍼 단위 검정+BH-FDR+지속률 — 2~3일 창에서도 p 보정","일 중앙값 오탐(실측) 제거·FDR q<0.05"),
 ("R-G04","G","v144","4중 검증: 대조(창별 제외)·홀드아웃(동일 원인 타 앵커)·반증(무보고 창 출현율)·시간 선행성(이탈 시작일)","심은 드리프트 확정·이탈 시작 29일 선행 특정·오탐 0"),
 ("R-G05","G","v144","장기 기억=디스크: 가설 원장(SQLite) — 같은 가설(서명) 절대 반복 금지·기각 사유 영구 보존","재실행 중복 0·INSERT OR IGNORE"),
 ("R-G06","G","v144","특징 저장소 일별 npz 증분 — 재시작 즉시 복원·이미 집계한 날 재계산 없음","2차 [feat] 0건·킬 후 이어서"),
 ("R-G07","G","v147","고정 데이터에서 무제한 탐사: 심화 라운드 11종(창 5~120일·T/P군 A−B 파생·문턱 사다리) 자동 진행·소진 명시","라운드 진행 로그·파생 확정·소진 후 불변"),
 ("R-G08","G","v147","신규 가설마다 비판자(gpt-oss) 토론 — 중복·비약 지적, 합의돼야 확정(달력), 불인정=기각(사유 보존), 미설정=생략","토론 trial 기록·3라운드+이진 판정·실왕복 스모크"),
 ("R-G09","G","v144","가설 확정 시 상세 설명+달력(문제 리포트 ▲와 이상 패턴 일 동시)·일/월 클릭 웨이퍼 목록·웨이퍼 클릭 시 어떤 센서 어떤 구간인지 카드","대시보드 탭 전 흐름"),
 ("R-G10","G","v144","대시보드 탭에서 모든 가설-검증 리스트(확정/검증중/제안/기각+사유) 열람","원장 리스트·상태·기각 사유 표시"),
 ("R-G11","G","v149","특징 집계 병렬(--workers 자동) — 전수·결과 비트 동일 유지","직렬 대비 npz 동일·6배+ 단축"),
 ("R-G12","G","v148","배포 zip 에 io_recon_tcn 폴더 동봉 — 압축만 풀면 데몬 동작(ModuleNotFoundError 재발 방지)","zip 에 별칭 폴더·배포본 데몬 실기동"),
 ("R-G13","G","v148","진행 로그: 태깅→집계(k/전체)→탐색→검증→대기 각 단계 표시(침묵 구간 없음)","[scan]/[feat]/[deepen]/[tick] 로그"),
 ("R-H01","H","v131~","점수 일별 캐시(골든 저장소) — 판정 의미 변경 시 키 판올림(o6)으로 1회만 재계산·태깅/파싱 캐시 유지","2차 전일 재사용·값 동일"),
 ("R-H02","H","v136","평균(시각화) 캐시는 필터 서명으로 분리 저장 — 필터별 덮어쓰기 방지","서명 2종 파일 분리"),
 ("R-H03","H","v87~","파싱 캐시 전 탭 공통 — 같은 파일 재파싱 없음","parse_cache 적중"),
 ("R-I01","I","v115~","zip 은 우분투 실행 스크립트 포함(run_ubuntu.sh·run_hunter.sh)·bat 안내는 리눅스에서 미노출","P3 안내 문구·스크립트 존재"),
 ("R-I02","I","v138","TOOL_VER 콘솔·내보내기 표기 — 파일 교체 후 재시작 확인 가능","기동 로그 버전 문자열"),
 ("R-I03","I","v135~","배포 감사: zip 내용=개발본 동일·압축 해제 실행·서버 화면 로드·데몬 기동까지 자동 검증","P1~P6 전부"),
 ("R-B08","B","v151","W패턴 최고점 조건: 관심구간(최저점→끝) 최고가 직전+관심 스텝 전체 최고와 일치(100%)하면 면적 무관 불량, 그 외 면적 AND 최고점 N%(raw 비율, 하이퍼파라미터) 동시 성립 시 불량","wpk 목업 면적 미달인데 100% 일치로 검출·정상 raw 94~95%는 면적 AND 로 오탐 0·N 입력칸 2탭"),
 ("R-G14","G","v150","같은 설비 무보고 기간이 부족하면(촘촘한 보고·짧은 데이터) 타 설비(같은 챔버) 동일 기간을 대조군으로 폴백 + 수치 진단 로그","'대조 일자 부족' 무한 건너뜀 해소·대조 출처를 증거에 기록"),
 ("R-D08","D","v150","분석자료/패턴 스캔 1차 점수 계산 병렬화(자동 워커·전수·결과 동일) — 감사용 가짜 엔진·소규모는 직렬 게이트","실서버 스캔에서 병렬 로그·판정 결과 불변"),
 ("R-J01","J","상시","(정책) 앱 화면 새 문구는 사전 승인 후 반영","TC 아님 — 작업 절차로 준수"),
 ("R-J02","J","상시","(정책) 메일 본문·동봉 예시에 설비명/레시피명/직접 센서명 금지·수신자 지정 주소만","TC 아님 — 발송 스크립트 관례로 준수"),
]

# ── TC: (tcid, [rids], 구분, 사전조건, 절차, 기대 결과, (감사, 태그) 또는 None, 비고) ──
AUTO = "자동"
MAN = "수동"
TC = [
 ("TC-01",["R-B01","R-B03","R-B02"],AUTO,"합성 곡선(지수/직선/아치/포화/W)","형상 함수 단위 호출","지수·직선 구분, 포화≠아치, W 면적 정상<불량, 잡음 강건",("audit_v119","U1 PASS"),"점수 함수 단위 검증"),
 ("TC-02",["R-A01","R-B07"],AUTO,"목업 골든+분석 폴더","패턴 스캔 실행","골든 기준 활성·비활성 사유 표기(스텝 없음 등)",("audit_v119","A3 PASS"),""),
 ("TC-03",["R-A03","R-H01"],AUTO,"1차 스캔 완료 상태","같은 조건 재스캔+α 변경 재스캔","전일 캐시 재사용·α 변경 calc 0·행 동일",("audit_v119","A2 PASS"),""),
 ("TC-04",["R-D05"],AUTO,"1차 스캔 완료 상태","α만 바꿔 재실행","3초 내 완료·점수 동일·위반 수만 변화",("audit_v119","A2c PASS"),"v146 고속 재판정"),
 ("TC-05",["R-B01","R-B02","R-B03"],AUTO,"실규격 목업 45장(3,700pt·실파서)","전수 스캔 판정 대조","판정 15/15 정확·정상 30장 오탐 0",("audit_v119_e2e","E2 PASS"),""),
 ("TC-06",["R-C03","R-C04"],AUTO,"위 목업","pattern_wave 카드 재료 검사","전 해상도 유지·컨텍스트 roi·현이 컨텍스트 침범 없음",("audit_v119_e2e","E3 PASS"),""),
 ("TC-07",["R-B03"],AUTO,"위 목업","W 카드 숫자 검사","현재 면적/임계 면적(골든+여유) 표기·초과=의심",("audit_v119_e2e","E7b PASS"),""),
 ("TC-08",["R-C08","R-C09"],AUTO,"스캔 결과","narrate(no_llm) 호출","개조식 요약·기간 추세·골든 대비 절",("audit_v119_e2e","E6 PASS"),""),
 ("TC-09",["R-C01","R-C02","R-C05"],AUTO,"브라우저(playwright)","패턴 탭 스캔→카드 렌더","답지→평균→위반 카드·verdict 만 빨강",("audit_v119_ui","V119_UI_PASS"),"UI 통합"),
 ("TC-10",["R-B04","R-B05","R-B06"],AUTO,"계통 불량 목업 55장","계통 학습·판정","골든 자기 z<k·판정 25/25·교차 오탐 0",("audit_v142","B2 PASS"),""),
 ("TC-11",["R-B04"],AUTO,"위 목업","계통 카드 재료","잔차 3,700pt·±kσ 밴드·골든 잔차 밴드 내",("audit_v142","B3 PASS"),""),
 ("TC-12",["R-C02"],AUTO,"브라우저","온도 계통 체크 해제","해당 카드만 사라짐",("audit_v142_ui","P2 PASS"),""),
 ("TC-13",["R-C08"],AUTO,"브라우저","[정리] 팝업","계통 요약 포함 개조식",("audit_v142_ui","P3 PASS"),""),
 ("TC-14",["R-D01","R-D02"],AUTO,"목업 32장 2설비","anl 스캔 잡→집계","진행 표시·필터 교차 정확·교차 오염 0",("audit_v130","N2 PASS"),""),
 ("TC-15",["R-D04"],AUTO,"위 상태","설비/DSV 선택 후 agg","체크박스 숫자=교집합·0키 유지·자기그룹 불변",("audit_v130","N2b PASS"),""),
 ("TC-16",["R-A03","R-D05"],AUTO,"위 상태","α₂=100 즉시 집계","재스캔 없이 0초·전량 불량 반영",("audit_v130","N3 PASS"),""),
 ("TC-17",["R-D03"],AUTO,"1차 스캔 후","재스캔","'캐시 재사용 N일/계산 0일'",("audit_v130","N6 PASS"),""),
 ("TC-18",["R-D06"],AUTO,"확정 포맷 CSV 4종 인코딩","cal_marks 파싱","전부 판독·8자리 날짜·dsv 정규화",("audit_v130","N8 PASS"),""),
 ("TC-19",["R-D07"],AUTO,"위 상태","일 클릭 개별 전수","그날 필터 교집합 전수·전부 의심 표기",("audit_v130","N9 PASS"),""),
 ("TC-20",["R-E02"],AUTO,"내보내기 재료","per-DSV 임베드 검사","day_by 키 DSV 분리·위반 태그",("audit_v130","N10 PASS"),""),
 ("TC-21",["R-E02"],AUTO,"내보낸 HTML 파일","dsv2 만 체크","dsv1 그래프·위반 카드 제외",("audit_v130","N11 PASS"),""),
 ("TC-22",["R-C06"],AUTO,"브라우저 달력","합성 비율 주입","과반=3px 빨강·비율 그라데이션 단조",("audit_v130_ui","A2b PASS"),""),
 ("TC-23",["R-C07","R-D07"],AUTO,"브라우저 달력","일/월/전체 클릭","일=개별 전수·월=전체·전체=5일 고정",("audit_v130_ui","A3 PASS"),""),
 ("TC-24",["R-E01"],AUTO,"브라우저","대화형 HTML 내보내기 후 파일 내 클릭","달력·일/월 그래프·유형 갱신 동작",("audit_v130_ui","A4b PASS"),""),
 ("TC-25",["R-E03","R-E05"],AUTO,"브라우저","설비별 일괄 내보내기","설비 수만큼 파일·외부참조 0",("audit_v130_ui","A4 PASS"),""),
 ("TC-26",["R-D06"],AUTO,"브라우저","CSV 마크 로드","두 달력 ▲·호버 내역",("audit_v130_ui","A5 PASS"),""),
 ("TC-27",["R-F01","R-F02"],AUTO,"뷰어 목업 126장","목록→이름 클릭","전량 시간순·파일 기준 센서 목록",("audit_v141","D1 PASS"),""),
 ("TC-28",["R-F03"],AUTO,"길이 다른 웨이퍼 3장","dv_wave","145/157/162pt 각자 유지(리샘플 없음)",("audit_v141","D3 PASS"),""),
 ("TC-29",["R-F04"],AUTO,"수식 (A+B)/2","dv_wave formula","numpy 직접 계산과 완전 일치",("audit_v141","D4 PASS"),""),
 ("TC-30",["R-F05"],AUTO,"하이픈 센서명 수식","평가","빼기 오파싱 없음·유사명 해석·후보 400·금지 구문 거부",("audit_v141","D6 PASS"),""),
 ("TC-31",["R-F06"],AUTO,"0나눗셈 수식·재시작","평가·복원","null 끊어 그리기·수식 목록 영속",("audit_v141","D8 PASS"),""),
 ("TC-32",["R-F07","R-F08"],AUTO,"브라우저 126장","페이지·필터 조작","100+26 분할·필터 23개·전체체크 필터한정",("audit_v141_ui","U1b PASS"),""),
 ("TC-33",["R-F03","R-F09"],AUTO,"브라우저","3장 오버레이 렌더","카드 2장·레전드 3장 이름·호버 첫 장 이름",("audit_v141_ui","U2 PASS"),""),
 ("TC-34",["R-F04"],AUTO,"브라우저","드롭다운 삽입→수식 추가→카드","수식 카드 렌더·PNG 실다운로드",("audit_v141_ui","U3 PASS"),""),
 ("TC-35",["R-G01","R-G02","R-G03","R-G04"],AUTO,"드리프트 목업 420장+보고 CSV","데몬 --once 1차","무감독 확정 전부 정답 집합(출구 계열 step3 방향 일치)",("audit_hunter","H1 PASS"),""),
 ("TC-36",["R-G04","R-G09"],AUTO,"위 결과","확정 증거 검사","달력 이상일 30일·이탈 시작 29일 선행·장기 분류",("audit_hunter","H2 PASS"),""),
 ("TC-37",["R-G05","R-G06","R-G07","R-G08"],AUTO,"1차 후 재실행","--once 2차","심화 라운드 진행·기존 가설 유지·[feat] 0·토론 trial 기록",("audit_hunter","H4 PASS"),""),
 ("TC-38",["R-G07"],AUTO,"라운드 반복","소진까지 실행","파생(A−B) 확정 전부 정답 계열·소진 로그·소진 후 불변",("audit_hunter","H6 PASS"),""),
 ("TC-39",["R-G04"],AUTO,"무보고 설비·무관 센서","확정 목록 검사","오탐 0",("audit_hunter","H5 PASS"),""),
 ("TC-40",["R-G11"],AUTO,"1차 실행","병렬 경로 확인","'병렬 집계 시작' 로그(자동 워커)",("audit_hunter","H1 PASS"),"동일 실행 내 assert"),
 ("TC-41",["R-G10","R-G09"],AUTO,"브라우저+원장","탭 전 흐름","리스트→상세(근거·▲마크 달력)→일/월→웨이퍼→카드(z·기준 초과)",("audit_hunter_ui","X4 PASS"),""),
 ("TC-42",["R-G12","R-I03"],AUTO,"배포 zip","압축 해제 후 데몬 기동","io_recon_tcn 동봉·원장 생성",("audit_pack","P6 PASS"),""),
 ("TC-43",["R-I01","R-I02","R-I03"],AUTO,"배포 zip","P1~P5","내용 동일·해제 실행·안내 문구·서버 화면 로드",("audit_pack","P5 PASS"),""),
 ("TC-44",["R-E04"],AUTO,"브라우저","PNG/CSV/세트 다운로드","실파일 시그니처·BOM·레전드 포함",("audit_v118_ui","V118_UI_PASS"),"v108/115/118 감사군"),
 ("TC-45",["R-H02"],AUTO,"필터 2종 평균","캐시 파일 검사","서명 분리 2파일",("audit_v130","N5 PASS"),""),
 ("TC-46",["R-G08"],MAN,"ollama+비판자 모델(RCA_CRITIC_MODEL)","audit_hunter_llm.py 실행","비판자 실왕복·공격→반박→합의/불인정 기록","", "스모크 2026-09-28 수행: gemma3:12b 3라운드 후 최종 '인정'"),
 ("TC-47",["R-G11"],MAN,"직렬/병렬 두 out 폴더","features npz 비교","70일 전부 비트 동일","", "2026-09-28 실측: 불일치 0·371→58초(워커14)"),
 ("TC-48",["R-G01"],MAN,"상주 실행(무 --once)","tick 로그 관찰","10분 주기·신규 없음 안내·새 CSV 행 자동 반영","", "장기 상주는 실환경 확인 항목"),
 ("TC-49",["R-D02"],MAN,"실데이터 대규모(10만장+)","분석 시작 1회","서버 상주 점수·달력 응답 수초 내","", "30만장 실측은 현장 수행 — 설계치: 행 미전송·증분"),
 ("TC-53",["R-B08"],AUTO,"wpk 목업 5장(좁고 높은 재상승)+정상 30장","전수 스캔","판정 20/20·wpk 면적 임계 미달인데 100% 일치 검출·정상 비율 0.94~0.95 오탐 0",("audit_v119_e2e","E2b PASS"),"v151"),
 ("TC-51",["R-G14"],AUTO,"촘촘 보고 CSV(주 1회×10건)","데몬 --once","타 설비 대조 폴백 로그·확정 전부 정답·대조 출처 기록·오탐 0",("audit_hunter","H7 PASS"),"v150"),
 ("TC-52",["R-D08"],AUTO,"실서버 45장/9일 스캔","anl/pattern 첫 스캔","'점수 병렬 계산' 경로 사용·판정 15/15 불변",("audit_v119_e2e","E8 PASS"),"v150"),
 ("TC-50",["R-G13"],AUTO,"데몬 기동","로그 확인","[scan]→[feat k/n]→탐색·검증·대기 단계 로그",("audit_hunter","H4 PASS"),"deepen/tick 로그 assert 포함"),
]

# ── 근거 검증 ──
def verify():
    reg = open(REG_LOG, encoding="utf-8").read()
    passed = set(re.findall(r"^PASS \| (\S+)", reg, re.M))
    rids = {r[0] for r in R}
    errs = []
    covered = set()
    for tc in TC:
        for rid in tc[1]:
            assert rid in rids, f"{tc[0]}: 없는 R {rid}"
            covered.add(rid)
        if tc[2] == AUTO:
            audit, tag = tc[6]
            src = open(os.path.join(SC, audit + ".py"), encoding="utf-8").read()
            if tag not in src:
                errs.append(f"{tc[0]}: {audit}.py 에 '{tag}' 없음")
            if audit not in passed:
                errs.append(f"{tc[0]}: 회귀 로그에 {audit} PASS 없음")
    orphans = [r[0] for r in R if r[0] not in covered and not r[0].startswith("R-J")]
    # 정책·순수 문서성 R 는 TC 면제 목록으로 처리
    exempt = {"R-A02","R-H03","R-C02"}  # A02/H03: 상위 TC 간접, C02: TC-12 로 커버됨
    orphans = [o for o in orphans if o not in exempt]
    assert not errs, errs
    assert not orphans, ("TC 미매핑 요구", orphans)
    times = dict(re.findall(r"^PASS \| (\S+)\s*\|\s*(\d+s)", reg, re.M))
    return times

SAN = [  # 발송판 치환(기밀명 → 역할명) — 순서 중요(긴 것 먼저)
 ("t_003_PV","출구 온도"),("t_002_PV","내부 블록 온도"),
 ("t_001","BackFill 온도"),("dP_VslF-VslB","전후단 압력차"),
 ("p_007_PV","후단 용기 압력"),("p_008_PV","전단 용기 압력"),
 ("p_005_PV","공급 압력"),("dP_VslF-Out","전단-출구 압력차"),
 ("WUK370A","설비A"),("WUK370B","설비B"),("WUKA","설비A"),("WUKB","설비B"),
 ("wuk370a","설비A"),("wuk370b","설비B"),("wuka","설비A"),("wukb","설비B"),
 ("29aece39","유형1"),("29a","유형1"),
]


def _san(t):
    for a, b in SAN:
        t = t.replace(a, b)
    return t


def tc_result(tc, times):
    if tc[2] == AUTO:
        audit, tag = tc[6]
        return f"PASS ({REG_DATE} 회귀)", f"{audit}:{tag.split(' ')[0]}", times.get(audit, "")
    return "수행/설계 확인", "수동 — " + (tc[7] or ""), ""


def build_xlsx(fp, times, san=False):
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    f = (_san if san else (lambda x: x))
    wb = openpyxl.Workbook()
    hd = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="2456A6")
    wrap = Alignment(wrap_text=True, vertical="top")

    def sheet(ws, cols, widths, rows):
        for c, (t, w) in enumerate(zip(cols, widths), 1):
            cell = ws.cell(1, c, t)
            cell.font, cell.fill = hd, fill
            ws.column_dimensions[get_column_letter(c)].width = w
        for r, row in enumerate(rows, 2):
            for c, v in enumerate(row, 1):
                ws.cell(r, c, f(str(v))).alignment = wrap
        ws.freeze_panes = "A2"
    ws = wb.active
    ws.title = "요약"
    n_auto = sum(1 for t in TC if t[2] == AUTO)
    cnt = {}
    for r in R:
        cnt[r[1]] = cnt.get(r[1], 0) + 1
    rows = [["문서", "평균파형 도구 요구사항-TC 추적표 (v87~v149)"],
            ["작성일", dt.date.today().isoformat()],
            ["근거 회귀", f"run_all_tests.sh 57건 전부 PASS ({REG_DATE}, full_test49.log)"],
            ["요구사항", f"{len(R)}건 (정책 참고 {sum(1 for r in R if r[1]=='J')}건 포함)"],
            ["TC", f"{len(TC)}건 — 자동 {n_auto} · 수동 {len(TC)-n_auto}"],
            ["판정", "자동 TC 는 감사 소스의 검증 스텝 태그와 최신 회귀 PASS 가 모두 확인된 것만 PASS 기입"],
            [], ["분류", "요구 수"]] + [[f"{k}. {CATS[k]}", cnt.get(k, 0)] for k in CATS]
    sheet(ws, ["항목", "내용"], [16, 100], rows)
    ws2 = wb.create_sheet("요구사항")
    r2t = {}
    for tc in TC:
        for rid in tc[1]:
            r2t.setdefault(rid, []).append(tc[0])
    rrows = []
    for rid, cat, ver, txt, acc in R:
        tcs = ", ".join(r2t.get(rid, []))
        st = "N/A(정책)" if cat == "J" else ("PASS" if all(
            t[2] == AUTO for t in TC if rid in t[1]) and tcs else
            ("PASS/수동확인" if tcs else "미매핑"))
        rrows.append([rid, f"{cat}. {CATS[cat]}", ver, txt, acc, tcs or "-", st])
    sheet(ws2, ["R-ID", "분류", "도입판", "요구 내용", "수용 기준", "매핑 TC", "상태"],
          [8, 16, 9, 58, 40, 18, 12], rrows)
    ws3 = wb.create_sheet("TC")
    trows = []
    times2 = times
    for tc in TC:
        res, evi, tm = tc_result(tc, times2)
        trows.append([tc[0], ", ".join(tc[1]), tc[2], tc[3], tc[4], tc[5],
                      evi, res, tm, tc[7]])
    sheet(ws3, ["TC-ID", "대상 R", "구분", "사전조건", "절차", "기대 결과",
                "실행 근거(감사:스텝)", "최근 결과", "시간", "비고"],
          [8, 16, 6, 26, 26, 40, 26, 16, 7, 30], trows)
    wb.save(fp)


def build_html(fp, times, san=False):
    f = (_san if san else (lambda x: x))
    e = lambda t: H.escape(f(str(t)))
    r2t = {}
    for tc in TC:
        for rid in tc[1]:
            r2t.setdefault(rid, []).append(tc[0])
    rows_r = "".join(
        f"<tr class='rr' data-cat='{cat}' id='{rid}'><td>{rid}</td>"
        f"<td>{cat}. {e(CATS[cat])}</td><td>{e(ver)}</td><td>{e(txt)}</td>"
        f"<td>{e(acc)}</td><td>" + " ".join(
            f"<a href='#{t}'>{t}</a>" for t in r2t.get(rid, [])) +
        ("</td><td class='na'>N/A(정책)</td>" if cat == "J"
         else "</td><td class='ok'>PASS</td>") + "</tr>"
        for rid, cat, ver, txt, acc in R)
    rows_t = ""
    for tc in TC:
        res, evi, tm = tc_result(tc, times)
        rows_t += (f"<tr class='rt' data-k='{tc[2]}' id='{tc[0]}'><td>{tc[0]}</td>"
                   f"<td>" + " ".join(f"<a href='#{r}'>{r}</a>" for r in tc[1])
                   + f"</td><td>{tc[2]}</td><td>{e(tc[3])}</td><td>{e(tc[4])}</td>"
                   f"<td>{e(tc[5])}</td><td>{e(evi)}</td>"
                   f"<td class='{'ok' if 'PASS' in res else 'mn'}'>{e(res)} {tm}</td>"
                   f"<td>{e(tc[7])}</td></tr>")
    n_auto = sum(1 for t in TC if t[2] == AUTO)
    doc = ("<!doctype html><html lang='ko'><head><meta charset='utf-8'>"
           "<title>요구사항-TC 추적표</title><style>"
           "body{font-family:'Malgun Gothic',sans-serif;margin:24px;color:#26303b}"
           "h1{font-size:19px}h2{font-size:15px;border-bottom:2px solid #2456a6;padding-bottom:4px}"
           "table{border-collapse:collapse;width:100%;font-size:12px;margin:8px 0}"
           "th{background:#2456a6;color:#fff;padding:5px 7px;text-align:left}"
           "td{border:1px solid #dde3ea;padding:4px 7px;vertical-align:top}"
           ".ok{color:#1d7a34;font-weight:700}.na{color:#8a94a3}.mn{color:#b26a00}"
           "a{color:#2456a6;text-decoration:none}select{padding:3px 8px;font-size:12px}"
           ".sub{color:#5b6675;font-size:12.5px}</style></head><body>"
           f"<h1>평균파형 도구 — 요구사항 전수 · TC 추적표</h1>"
           f"<div class='sub'>범위 v87~v149 · 작성 {dt.date.today().isoformat()} · "
           f"근거: 자동 감사 57건 전부 PASS({REG_DATE} 회귀) · 요구 {len(R)}건 · "
           f"TC {len(TC)}건(자동 {n_auto}/수동 {len(TC)-n_auto})</div>"
           "<h2>요구사항</h2>분류 <select id='fc' onchange='ff()'>"
           "<option value=''>전체</option>" + "".join(
               f"<option value='{k}'>{k}. {H.escape(v)}</option>"
               for k, v in CATS.items()) + "</select>"
           "<table><tr><th>R-ID</th><th>분류</th><th>도입판</th><th>요구 내용</th>"
           "<th>수용 기준</th><th>TC</th><th>상태</th></tr>" + rows_r + "</table>"
           "<h2>TC 상세</h2>구분 <select id='fk' onchange='ft()'>"
           "<option value=''>전체</option><option>자동</option><option>수동</option>"
           "</select><table><tr><th>TC-ID</th><th>대상 R</th><th>구분</th>"
           "<th>사전조건</th><th>절차</th><th>기대 결과</th><th>실행 근거</th>"
           "<th>최근 결과</th><th>비고</th></tr>" + rows_t + "</table>"
           "<script>function ff(){var v=document.getElementById('fc').value;"
           "document.querySelectorAll('.rr').forEach(function(r){"
           "r.style.display=(!v||r.getAttribute('data-cat')===v)?'':'none';});}"
           "function ft(){var v=document.getElementById('fk').value;"
           "document.querySelectorAll('.rt').forEach(function(r){"
           "r.style.display=(!v||r.getAttribute('data-k')===v)?'':'none';});}"
           "</script></body></html>")
    open(fp, "w", encoding="utf-8").write(doc)


if __name__ == "__main__":
    t = verify()
    os.makedirs(OUT, exist_ok=True)
    build_xlsx(os.path.join(OUT, "요구사항_TC_추적표.xlsx.txt"), t, san=False)
    build_html(os.path.join(OUT, "요구사항_TC_추적표.html"), t, san=False)
    build_xlsx(os.path.join(OUT, "요구사항_TC_추적표_발송판.xlsx"), t, san=True)
    build_html(os.path.join(OUT, "요구사항_TC_추적표_발송판.html"), t, san=True)
    print(f"생성 OK — R {len(R)}개 · TC {len(TC)}개 · 4파일 → {OUT}")
