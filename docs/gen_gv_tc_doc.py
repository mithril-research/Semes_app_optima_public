# -*- coding: utf-8 -*-
"""[골든 비교] 요구사항 전수 추적표 + TC 문서 생성기 (v158~v164).

원칙(기존 gen_tc_doc 동일): 근거 없는 PASS 금지 — 자동 TC 는 (감사파일, 태그)
①태그가 감사 소스에 실재 ②최신 통합 로그(gv_last_audit.log)에서 그 태그가
PASS 로 실측 — 둘 다 확인된 것만 '자동/PASS'. 수동 TC 는 절차만 기술.
"""
import datetime as dt
import html as H
import os
import re

SC = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SC, "gv_tc_doc")
LOG = os.path.join(SC, "gv_last_audit.log")
TODAY = dt.date.today().isoformat()

# ── R: (rid, 요구 출처, 요구 내용, 수용 기준) ──
R = [
 ("R-GV01", "1·1-1", "설정 기간의 모든 웨이퍼를 센서별 평균 → 골든", "기간·필터 반영 평균, 센서별 mean 산출"),
 ("R-GV02", "2·2-1", "골든 2종: 설비 대표 + DSV별(설비당 대표1+DSV수, 동적)", "설비 2종·DSV 3종 더미에서 골든 6개(대표2+DSV4)"),
 ("R-GV03", "3·3-1", "분석자료탭형 불러오기: 경로(;다중)+설비/DSV/레시피 체크박스·직접입력+월+기간+센서 선택", "필터 정확(직접입력 리스트 포함)·패싯 체크박스·기간 자동"),
 ("R-GV04", "4", "불량조치내역 CSV 로드(관용 파서 — 인코딩/형식 수용)", "건수 판독·경로 오류 명시"),
 ("R-GV05", "확정", "골든 산정 시 CSV 보고분(설비+DSV+일자) 제외", "제외 수 = 기대 대조 일치·화면 표시"),
 ("R-GV06", "5-1·확정", "골든(평균=스무딩) vs 실데이터(노이즈) 격차 고려 판정 — 미보고 불량 오염 방어 포함", "σ밴드 초과 면적 지표 + 자체 정화(그룹 상대 컷)로 미보고 형상 불량 전량 제외"),
 ("R-GV07", "5-2", "offset 무시, 스텝별 파형(형상)만 비교", "스텝 내 평균 차감 후 비교 — 레벨차만 있는 웨이퍼 OK"),
 ("R-GV08", "5-3·6-1", "룰베이스: 넓이 기반 판정(노이즈 강건)", "직선·봉우리 형상 불량 검출, 정상 오탐 <5%"),
 ("R-GV09", "5-2·6-5·6-6", "골든-실데이터 길이·offset 정합(스텝별 리샘플) + 스텝별 원 길이 차이를 카드에 표기", "가변 길이(3,500~3,900pt)에서 판정 정상·길이 배지(기준/실제/%)"),
 ("R-GV10", "5-4", "룰베이스 분석/VLM 분석 두 실행 모드(+하이브리드)", "모드 선택 UI·양 모드 실행"),
 ("R-GV11", "6·6-2", "VLM: 스텝별 골든+실데이터 오버레이 이미지 입력 → '차이 크다/안 크다' 이진 → NG 시 불량", "이미지 생성·이진 파싱·판정 기록"),
 ("R-GV12", "6-2", "룰베이스 불량 기준은 하이퍼파라미터(k·α·β)로 사용자 설정", "값 변경 즉시 재판정(재스캔 없음)·단조성"),
 ("R-GV13", "확정·7-2", "한 스텝 NG→센서 NG, 선택 센서 중 하나라도 NG→불량 웨이퍼", "압력 센서만 불량인 웨이퍼가 웨이퍼 NG 로 집계"),
 ("R-GV14", "6-4·6-5", "카드: 실측+현재 DSV 골든+설비 대표 골든(흐린 가이드), 전 센서", "3곡선 동시·대표 골든 회색 실선·σ밴드"),
 ("R-GV15", "7~7-3", "추세는 막대그래프: x=날짜(집계 일/주/월 선택), y=정상/불량 수 — 9개월 대응", "집계 전환·합 보존·적층 2계열·레전드"),
 ("R-GV16", "6-3", "분석 UX: 막대 클릭→기간 웨이퍼 목록(불량 우선)→클릭→전 센서 카드", "드릴다운 전 흐름"),
 ("R-GV17", "8", "HTML 내보내기(단일 파일: 요약·파라미터·막대·판정표·불량 전수 카드+정상 표본)", "다운로드 실검증·외부 참조 0건"),
 ("R-GV18", "9", "분석 도구 개선안 조사·제공 + 채택분 우선 구현", "개선안 12항 제시·채택 3항(근거 랭킹·판정 CSV·VLM 근거) 구현"),
 ("R-GV19", "파생", "스텝 구성이 다른 웨이퍼의 누락 스텝은 '판정불가' — NG 로 오분류 금지", "누락 스텝 NaN 집계·NG 목록에 미포함·배지 표기"),
 ("R-GV20", "파생", "판정 캐시: 일별 npz+서명, VLM 판정 캐시 — 재실행 즉시", "재실행 <30초·VLM 2회째 무호출"),
 ("R-GV21", "사용자 지적", "VLM 은 비전 모델만 — 텍스트 전용(gpt-oss류) 지정 시 자동 차단·전환", "가드 로그·실사용 모델이 비전 모델"),
 ("R-GV22", "상시", "폐쇄망: 산출 HTML 외부 참조 0", "내보내기 파일 http 참조 검사 0건"),
 ("R-GV23", "v161 지시", "골든 기간(정상 구간)과 분석 기간(판정 대상)은 별도 입력 — 분석 기간 변경이 골든에 영향 없음", "분석 기간 재실행 시 결과 한정·골든 요약 불변"),
 ("R-GV24", "v163 지시", "설비 대표 골든=여러 DSV 합산 평균·DSV별 골든=단일 DSV + DSV별 골든 기간 개별 입력(비우면 공통)", "dsv1 만 하루로 제한 시 그 골든만 축소·대표=DSV 합"),
 ("R-GV25", "v163 지시", "룰베이스 스텝별 민감도(α) 개별 설정 — 즉시 재판정·하이브리드 VLM 후보 선별에도 반영", "step5 둔감화 시 스텝5 NG 소멸(타 스텝 유지)·역방향 동작"),
 ("R-GV26", "v162 지적", "카드의 실측 선은 항상 표시 — 스텝 정합 실패 구간은 전역 리샘플로 메움(판정 경로 무영향)", "스텝 누락 웨이퍼 카드 실측 결측 0"),
 ("R-GV27", "v164 UX", "단계형 화면: ①데이터 ②골든 ③판정 ④결과 — 완료 배지·다음 단계 자동 열림·고급 설정 접기", "단계 진행 시 배지 갱신·다음 섹션 자동 표시"),
 ("R-GV28", "v164 UX", "골든 갤러리: 산정 직후 센서별 골든 카드(대표 mean±σ + DSV별 겹침) + 자체 정화 제외 목록 확인·복원 재산정", "갤러리 카드 렌더·복원 지정 웨이퍼가 제외 목록에서 빠짐"),
 ("R-GV29", "v164 UX", "결과 판정표: 정렬(판정/날짜/웨이퍼/그룹/최대 A)·검색·불량만 보기·VLM 근거 열·센서별 불량 장수 요약", "불량만/검색 시 행 축소·요약 표시"),
 ("R-GV30", "v164 UX", "카드 정리: 불량 센서 우선·정상 센서 접기·스텝별 A/α 수치·VLM 이 본 잔차 이미지 동봉", "카드 A 수치 표시·VLM PNG API 응답"),
 ("R-GV31", "v164 UX", "민감도 프리셋(보수/표준/민감) + 값 변경 시 즉시 '불량 N장' 피드백", "프리셋 선택 시 k/α 반영·피드백 문구 갱신"),
 ("R-GV32", "v164 UX", "분석 조건 저장/불러오기 + 재접속(새로고침) 시 목록·골든·판정표·조건 자동 복원", "저장→변경→불러오기 복원·새로고침 후 판정표 복원"),
 ("R-GV34", "v164 검수", "패싯 체크박스(설비/DSV/레시피) 선택이 골든 산정·판정 대상에 실제 반영(전체 체크=무필터)·라벨은 값 이름", "dsv1 만 체크 시 골든 대표+dsv1·판정 dsv1 만"),
 ("R-GV35", "v166 지시", "VLM 은 파형 '모양'을 말로 서술·비교해 판정(골든 모양→현재 모양→방향·형태·변화량·변곡 비교→결론) — 한국어 전용 네 줄 양식(판정/골든/현재/근거)", "더미 8/8 정답·영어 단어 0·네 줄 양식"),
 ("R-GV36", "v166 지시", "VLM 사용자 지시 편집: 공통 지시 + 스텝별 파형 특징·검사 방법을 직접 적어 전달, 조건 저장 포함, 시스템 지시 열람", "스텝 지시가 서술에 반영·캐시 키에 프롬프트 해시"),
 ("R-GV37", "v166 지시", "카드: 불량 스텝 구간 빨강 음영 하이라이트 + 센서마다 원본 스케일 비교·패턴(스텝별 offset 제거) 비교 두 그래프", "카드 제목 '원본 스케일'/'패턴 비교'·음영 rect"),
 ("R-GV38", "v166 지시", "판정 기준 골든 토글: 챔버(DSV)별 ↔ 설비 통합 — 선택이 판정·캐시·메인 그래프, 반대쪽은 회색 가이드", "토글 재실행 시 카드 '판정 기준: 설비 통합 골든'"),
 ("R-GV39", "v166 보강", "여러 폴더에 같은 파일명이 있어도 덮어쓰지 않음(폴더명으로 구분) + ①단계에 파일/태그/중복 진단 표시", "TC·UI 감사 통과(회귀 없음)"),
 ("R-GV33", "v164 UX", "데이터 시각화 탭 목록 공유 사용 + 불량 일자를 분석자료/패턴 인식 달력 ▲마크로 전달", "공유 가드(미로드 400)·마크 전달 함수"),
]

# ── TC: (tcid, rids, 유형, 방법, 감사파일, 태그) — 수동은 파일/태그 없음 ──
TC = [
 ("TC-GV01", "R-GV01,R-GV02", "자동", "설비2×DSV3 더미(122장, 정상 3,500~3,900pt 가변) → 골든 산정 → 대표2+DSV4 산출·소표본 배지", "audit_gv_tc.py", "TCG-03 PASS"),
 ("TC-GV02", "R-GV03", "자동", "전체/설비 직접입력(WUKA)/월 필터 각각 조회 → 장수·패싯·센서 원본 수(56) 대조", "audit_gv_tc.py", "TCG-01 PASS"),
 ("TC-GV03", "R-GV03", "자동", "실브라우저: 패싯 체크박스 렌더·센서 선택·기간 자동 채움", "audit_gv_ui.py", "UI2 PASS"),
 ("TC-GV04", "R-GV04", "자동", "CSV 2건 판독·오경로 400", "audit_gv_tc.py", "TCG-02 PASS"),
 ("TC-GV05", "R-GV05", "자동", "CSV 보고일 웨이퍼 수 = excluded_n 대조", "audit_gv_tc.py", "TCG-03 PASS"),
 ("TC-GV06", "R-GV06", "자동", "미보고 형상 불량(직선5+봉우리2)을 자체 정화가 전량 제외(경계 여분 ≤2)·제외 목록 json 기록", "audit_gv_tc.py", "TCG-04 PASS"),
 ("TC-GV07", "R-GV06,R-GV08", "자동", "정화 후 σ밴드 판정: 미탐 0(직선10)·오탐 0/104 (단일 센서 세트)", "audit_gv.py", "G3 PASS"),
 ("TC-GV08", "R-GV07,R-GV08,R-GV13", "자동", "2센서 세트: 직선(온도)·봉우리(압력) 각각 해당 센서만 NG → 웨이퍼 NG 합산, 오탐 0/110", "audit_gv_tc.py", "TCG-05 PASS"),
 ("TC-GV09", "R-GV09", "자동", "가변 길이(3,500~3,900pt) 정합 판정 + 스텝5 +40% 웨이퍼는 길이 지표만 NG(형상 오분류 0)", "audit_gv_tc.py", "TCG-05 PASS"),
 ("TC-GV10", "R-GV09,R-GV14", "자동", "카드: DSV 골든+대표 골든+σ밴드 동시, 길이 배지(기준→실제), 누락 스텝 배지 None", "audit_gv_tc.py", "TCG-09 PASS"),
 ("TC-GV11", "R-GV10,R-GV11", "자동", "하이브리드 실행 → 직선 불량 2/2 VLM NG·해당 센서와 스텝5 정확 지목(σ정규화 잔차 뷰·3표 다수결)", "audit_gv_vlm.py", "V1 PASS"),
 ("TC-GV12", "R-GV20", "자동", "VLM 캐시: 2회째 실행 즉시(재호출 없음)·캐시 파일 존재", "audit_gv_vlm.py", "V2 PASS"),
 ("TC-GV13", "R-GV21", "자동", "GV_VLM_MODEL=gpt-oss 지정 → 차단 로그 + 비전 모델 자동 전환 실측", "audit_gv_vlm.py", "V3 PASS"),
 ("TC-GV14", "R-GV12", "자동", "α 5.0→NG 0(즉시)·α 0.0001→NG 증가(단조)·재판정 <10초", "audit_gv.py", "G5 PASS"),
 ("TC-GV15", "R-GV12,R-GV20", "자동", "k/α 즉시 재판정 + 판정 재실행 캐시 적중(<30초)", "audit_gv_tc.py", "TCG-07 PASS"),
 ("TC-GV16", "R-GV15", "자동", "일/주/월 집계 전환 시 합·판정 수 보존 + 기간(d0~d1) 조회 한정", "audit_gv_tc.py", "TCG-08 PASS"),
 ("TC-GV17", "R-GV15,R-GV16", "자동", "실브라우저: 막대 렌더→막대 클릭 목록 한정→웨이퍼 클릭 카드·길이 배지", "audit_gv_ui.py", "UI5 PASS"),
 ("TC-GV18", "R-GV16,R-GV18", "자동", "판정 후 초과면적 랭킹 표시·판정 CSV 다운로드", "audit_gv_ui.py", "UI7 PASS"),
 ("TC-GV19", "R-GV17,R-GV22", "자동", "HTML 내보내기 실다운로드 — 요약·판정표·카드 포함·외부 참조 0건", "audit_gv_ui.py", "UI7 PASS"),
 ("TC-GV20", "R-GV19", "자동", "스텝6 누락 웨이퍼: 판정불가 집계(na>0)·누락 스텝 NG 오분류 0", "audit_gv_tc.py", "TCG-05 PASS"),
 ("TC-GV21", "R-GV02,R-GV13", "자동", "다설비: 전 웨이퍼가 자기 설비 골든과 매칭(gkey 100%)·타 설비 불량도 검출", "audit_gv_tc.py", "TCG-06 PASS"),
 ("TC-GV22", "R-GV11", "수동", "현장: 실데이터 VLM 판정 품질 — 대시보드의 VLM 근거 문장을 표본 대조(더미 실증은 TC-GV11 자동화: 형상 불량 8/8 정답·정상 OK)", "", ""),
 ("TC-GV23", "R-GV01,R-GV15", "수동", "현장: 9개월 실데이터 규모 — 골든 산정·판정 소요/막대 가독성 확인(막대 집계 '월' 권장)", "", ""),
 ("TC-GV24", "R-GV23", "자동", "실브라우저: 분석 기간만 변경 재실행 → 결과 그 기간 한정·골든 요약 불변", "audit_gv_ui.py", "UI9 PASS"),
 ("TC-GV25", "R-GV24", "자동", "per_dsv 로 dsv1 하루 한정 재산정 → dsv1 골든만 축소·타 DSV 유지·대표=DSV 합집합", "audit_gv_tc.py", "TCG-10 PASS"),
 ("TC-GV26", "R-GV25", "자동", "alpha_steps step5=9.9 → 직선 NG 0(봉우리 유지) / 공통 α 둔감+step5 만 0.05 → 직선 전량 검출", "audit_gv_tc.py", "TCG-11 PASS"),
 ("TC-GV27", "R-GV26", "자동", "스텝 누락 웨이퍼 카드의 실측 시리즈 결측 0(배지는 판정불가 유지)", "audit_gv_tc.py", "TCG-09 PASS"),
 ("TC-GV28", "R-GV27", "자동", "실브라우저: 목록 로드 후 ① 배지·② 섹션 자동 열림, 산정 후 ② 배지, 판정 후 ④ 배지·섹션 자동 열림", "audit_gv_ui.py", "UI4 PASS"),
 ("TC-GV29", "R-GV28", "자동", "산정 직후 갤러리 카드 렌더 + gv_golden 그룹 2·복원 재산정 시 제외 목록에서 제거", "audit_gv_tc.py", "TCG-12 PASS"),
 ("TC-GV30", "R-GV29", "자동", "판정표 불량만 체크 시 행 축소·검색 'linA' 시 해당 행만·센서별 불량 요약 표시", "audit_gv_ui.py", "UI11 PASS"),
 ("TC-GV31", "R-GV30", "자동", "gv_wafer 응답에 스텝별 A 수치·gv_vlm_png 가 PNG 반환", "audit_gv_tc.py", "TCG-12 PASS"),
 ("TC-GV32", "R-GV31", "자동", "프리셋 보수 선택 → k 3·α 0.1 반영 + 즉시 피드백 문구", "audit_gv_ui.py", "UI10 PASS"),
 ("TC-GV33", "R-GV32", "자동", "조건 저장→값 변경→불러오기 복원 · 새로고침 후 판정표·골든 배지·α 복원", "audit_gv_ui.py", "UI13 PASS"),
 ("TC-GV34", "R-GV33", "자동", "데이터 시각화 미로드 상태 from_viz → 400 가드", "audit_gv_tc.py", "TCG-12 PASS"),
 ("TC-GV36", "R-GV34", "자동", "facet dsv=[dsv1] 로 산정·판정 → 골든 집합 {대표,dsv1}·결과 전부 dsv1", "audit_gv_tc.py", "TCG-13 PASS"),
 ("TC-GV37", "R-GV34", "자동", "실브라우저: DSV별 골든 기간 라벨 'dsv1 골든 기간'·패싯 값이 이름", "audit_gv_ui.py", "UI2 PASS"),
 ("TC-GV38", "R-GV35,R-GV36", "자동", "하이브리드+스텝5 사용자 지시 → 직선 2/2 VLM NG·스텝5 지목(서술 네 줄·한국어)", "audit_gv_vlm.py", "V1 PASS"),
 ("TC-GV39", "R-GV37", "자동", "실브라우저: 카드에 '원본 스케일'·'패턴 비교' 두 그래프·판정 기준 라벨", "audit_gv_ui.py", "UI5 PASS"),
 ("TC-GV40", "R-GV38", "자동", "기준 골든 '설비 통합' 토글 재실행 → 카드 라벨 반영·불량 스텝 음영", "audit_gv_ui.py", "UI14 PASS"),
 ("TC-GV41", "R-GV35", "수동", "현장: VLM 서술(골든/현재/근거)이 실데이터 파형과 맞는지 표본 대조, 스텝 지시 문구 조정", "", ""),
 ("TC-GV35", "R-GV33", "수동", "현장: 분석자료 탭에서 '불량 일자를 달력 마크로' 후 ▲마크 표시 확인", "", ""),
]

BUGS = [
 ("B-GV01", "TC-GV02 작성 중 발견", "필터 직접입력이 목록 0장 — _parse_multi 가 JSON 배열을 str() 해 \"['wuka']\" 토큰이 됨(골든 비교 탭은 배열로 전송)", "배열 입력을 쉼표 결합 후 파싱하도록 수정 — 수정 후 TCG-01 PASS"),
 ("B-GV02", "TC-GV06 작성 중 발견", "다중 센서 선택 시 자체 정화 무력화 — 정화 지표(k=0)가 저진폭 센서의 노이즈 바닥(~0.18)에 지배돼 진짜 불량 신호(0.1)가 max 에서 묻힘", "정화 지표를 웨이퍼 자체 노이즈 σ(시간 차분 MAD·오염 무관) 기반 k=3 초과 면적으로 교체 — 분리도 복원"),
 ("B-GV03", "TC-GV06 작성 중 발견", "소수 오염(봉우리 2장)이 DSV 1차 골든 평균을 밀면 에코가 그 DSV 정상 전원에 얹혀 설비 전체 컷이 정상 26장을 과잉 제외(DSV 골든 소멸)", "컷을 DSV 그룹별 상대 기준으로 변경(에코는 그룹 중앙값도 함께 올려 상쇄)+그룹<10장은 설비 폴백 — 과잉 제외 1장 이내"),
 ("B-GV04", "VLM 실판정 실증 중 발견", "겹침 뷰에서 VLM 이 명백한 직선화·봉우리도 OK 오판 — 불량이 크면 밴드가 실낱이 돼 '확신 없음→OK', 밴드를 키우면 노이즈를 초과로 환각", "σ 정규화 잔차 뷰(밴드 ±3σ 고정 가시·±12σ 클립)로 교체 — 더미 실판정 8/8 정답(센서·스텝 지목 포함)"),
 ("B-GV05", "VLM 실판정 실증 중 발견", "비전 호출은 temperature 고정 불가(옵션 포함 시 일부 VL 모델 폭주 — 기존 실측)라 동일 이미지 판정이 흔들림", "후보 한정 3표 다수결(2표 일치 시 조기 종료)로 안정화 — 캐시에 최종 판정 저장"),
 ("B-GV06", "v164 화면 검수", "패싯 체크박스·DSV별 기간 라벨이 '0/1/2'로 표시(서버 배열을 객체 키로 읽음) + 체크 선택이 산정·판정에 전달되지 않는 장식(v158~)", "배열 그대로 사용 + facet 선택을 산정·판정 대상 필터로 연결 — TCG-13·UI2 로 고정"),
]


def _verify():
    log = open(LOG, encoding="utf-8").read() if os.path.isfile(LOG) else ""
    res = {}
    for tcid, _r, typ, _m, af, tag in TC:
        if typ != "자동":
            res[tcid] = ("수동", "절차 참조")
            continue
        src = os.path.join(SC, af)
        ok_src = os.path.isfile(src) and tag.split()[0] in open(src, encoding="utf-8").read()
        ok_log = tag in log
        res[tcid] = (("PASS" if (ok_src and ok_log) else "미확인"),
                     f"{af} · {tag}" + ("" if ok_log else " (최신 로그 무근거)"))
    return res


def make_xlsx(res):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = Workbook()
    hd = Font(bold=True, color="FFFFFF")
    fill = PatternFill("solid", fgColor="2456A6")
    wrap = Alignment(wrap_text=True, vertical="top")
    ws = wb.active; ws.title = "요구사항"
    ws.append(["R-ID", "요구 출처(사용자 항번)", "요구 내용", "수용 기준"])
    for c in ws[1]:
        c.font = hd; c.fill = fill
    for row in R:
        ws.append(list(row))
    for col, w in zip("ABCD", (10, 14, 62, 46)):
        ws.column_dimensions[col].width = w
    for r2 in ws.iter_rows(min_row=2):
        for c in r2:
            c.alignment = wrap
    ws2 = wb.create_sheet("TC")
    ws2.append(["TC-ID", "대상 R", "유형", "방법/절차", "근거(감사·태그)", "결과",
                f"검증일 {TODAY}"])
    for c in ws2[1]:
        c.font = hd; c.fill = fill
    for tcid, rids, typ, m, af, tag in TC:
        st, ev = res[tcid]
        ws2.append([tcid, rids, typ, m, ev if typ == "자동" else "현장 확인",
                    st, ""])
    for col, w in zip("ABCDEF", (10, 22, 6, 66, 30, 8)):
        ws2.column_dimensions[col].width = w
    for r2 in ws2.iter_rows(min_row=2):
        for c in r2:
            c.alignment = wrap
    ws3 = wb.create_sheet("발견 버그")
    ws3.append(["ID", "발견 경위", "증상/원인", "조치·재검 결과"])
    for c in ws3[1]:
        c.font = hd; c.fill = fill
    for row in BUGS:
        ws3.append(list(row))
    for col, w in zip("ABCD", (8, 18, 60, 48)):
        ws3.column_dimensions[col].width = w
    for r2 in ws3.iter_rows(min_row=2):
        for c in r2:
            c.alignment = wrap
    fp = os.path.join(OUT, "골든비교_요구사항_TC_추적표.xlsx.txt.txt")
    wb.save(fp)
    return fp


def make_html(res):
    def tbl(head, rows):
        h = "<table><tr>" + "".join(f"<th>{H.escape(x)}</th>" for x in head) + "</tr>"
        for r2 in rows:
            h += "<tr>" + "".join(f"<td>{H.escape(str(x))}</td>" for x in r2) + "</tr>"
        return h + "</table>"
    n_auto = sum(1 for t in TC if t[2] == "자동")
    n_pass = sum(1 for t in TC if res[t[0]][0] == "PASS")
    body = (f"<h1>[골든 비교] 요구사항 전수 추적표 + TC</h1>"
            f"<p>검증일 {TODAY} · 요구 {len(R)}건 · TC {len(TC)}건"
            f"(자동 {n_auto} — PASS {n_pass} · 수동 {len(TC) - n_auto}) · "
            f"발견 버그 {len(BUGS)}건(전부 조치·재검 통과)<br>"
            f"검증 데이터: 더미 122장 — 설비 2종×DSV 3종, 센서 수 모델 원본"
            f"(56) 유지, 정상 길이 3,500~3,900pt 랜덤(생성기 동봉)</p>"
            "<h2>1. 요구사항</h2>"
            + tbl(["R-ID", "출처", "요구 내용", "수용 기준"], R)
            + "<h2>2. 테스트 케이스</h2>"
            + tbl(["TC-ID", "대상 R", "유형", "방법", "근거", "결과"],
                  [(t[0], t[1], t[2], t[3],
                    res[t[0]][1] if t[2] == "자동" else "현장 확인",
                    res[t[0]][0]) for t in TC])
            + "<h2>3. TC 작성 중 발견·조치한 버그</h2>"
            + tbl(["ID", "발견 경위", "증상/원인", "조치·재검"], BUGS))
    doc = ("<!DOCTYPE html><html><head><meta charset='utf-8'>"
           "<title>골든 비교 TC</title><style>"
           "body{font-family:sans-serif;margin:20px;color:#2b3440;font-size:13px}"
           "table{border-collapse:collapse;margin:8px 0;width:100%}"
           "th,td{border:1px solid #dde3ea;padding:4px 8px;text-align:left;"
           "vertical-align:top}th{background:#2456a6;color:#fff}"
           "tr:nth-child(even){background:#f6f8fb}</style></head><body>"
           + body + "</body></html>")
    fp = os.path.join(OUT, "골든비교_요구사항_TC_추적표.html")
    open(fp, "w", encoding="utf-8").write(doc)
    return fp


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = _verify()
    bad = [t for t, (st, _e) in res.items() if st == "미확인"]
    x = make_xlsx(res)
    h = make_html(res)
    print(f"R {len(R)} · TC {len(TC)} · 미확인 {len(bad)} {bad[:5]}")
    print(x); print(h)
