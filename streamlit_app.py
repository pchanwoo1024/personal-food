import streamlit as st
import requests
from bs4 import BeautifulSoup
import glob
import re
from datetime import datetime, timedelta, time as dtime
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="개인 맞춤 영양 식단 추천 (대전고 급식)")
st.title("🍽️ AI 기반 개인 맞춤 영양 식단 추천 (대전고 3~5월 급식)")

# 사용자 입력
name      = st.text_input("이름")
sex       = st.selectbox("성별", ["M", "F"])
age       = st.slider("나이", 10, 80, 18)
height    = st.slider("키 (cm)", 140, 200, 170)
weight    = st.slider("몸무게 (kg)", 40, 120, 60)
activity  = st.slider("활동량 (1.0~5.0)", 1.0, 5.0, 3.0, 0.1)
wake_time = st.time_input("기상 시간", value=dtime(7,0))
sleep_time= st.time_input("취침 시간", value=dtime(22,0))

allergy_options = [
    "난류","우유","메밀","땅콩","대두","밀",
    "고등어","게","새우","돼지고기","복숭아",
    "토마토","아황산류","호두","닭고기","쇠고기",
    "오징어","조개류","잣"
]
allergies = st.multiselect("알레르기 (복수 선택)", allergy_options)

symptom_options = [
    "눈떨림","피로","두통","근육경련",
    "탈모","불면증","집중력저하","손발저림"
]
symptoms = st.multiselect("현재 증상", symptom_options)

# 메뉴 로드: HWP 파일 우선, 없다면 웹 파싱
@st.cache_data
def load_menu():
    dishes = set()
    # 1) HWP 파싱 (CP949/UTF-16LE 시도)
    for path in glob.glob("/mnt/data/*.hwp"):
        try:
            raw = open(path, 'rb').read()
            try:
                text = raw.decode('utf-16le', errors='ignore')
            except:
                try:
                    text = raw.decode('cp949', errors='ignore')
                except:
                    continue
            items = re.findall(r'[가-힣]{2,10}', text)
            for item in items:
                if item in ['급식','중식','조식','석식','메뉴','식단','학년도','월','식단표']:
                    continue
                dishes.add(item)
        except:
            continue
    if dishes:
        return sorted(dishes)
    # 2) 웹 파싱
    try:
        url = "https://djhs.djsch.kr/boardCnts/list.do?boardID=41832&m=020701&s=daejeon"
        session = requests.Session()
        session.headers.update({'User-Agent':'Mozilla/5.0'})
        res = session.get(url, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')
        for td in soup.select('table.boardList tbody tr td.title, table.tableList tbody tr td.title'):
            text = td.get_text(strip=True)
            if '중식' not in text:
                continue
            parts = text.split(']')
            menu_str = parts[-1] if len(parts) > 1 else text
            items = [itm.strip() for itm in menu_str.split(',') if itm.strip()]
            for item in items:
                clean = re.sub(r"\([^)]*\)", '', item).strip()
                if re.fullmatch(r'[가-힣 ]{2,15}', clean):
                    dishes.add(clean)
    except Exception as e:
        st.warning(f"웹에서 급식 메뉴를 불러오지 못했습니다: {e}")
    return sorted(dishes)

menu_names = load_menu()
if not menu_names:
    st.error("급식 메뉴를 불러오지 못했습니다. HWP 파일 또는 네트워크를 확인해주세요.")
    st.stop()
# 디버그: 로드된 메뉴
st.write(f"🔎 로드된 메뉴 ({len(menu_names)}개)", menu_names)

# 영양 정보 추정 함수
def est(name: str) -> dict:
    if '밥' in name or '죽' in name:
        kcal = 300
    elif any(x in name for x in ['국','찌개','탕']):
        kcal = 80
    elif any(x in name for x in ['볶음','조림','구이','스테이크']):
        kcal = 250
    elif any(x in name for x in ['전','만두','피자','파스타','면','떡볶이']):
        kcal = 200
    else:
        kcal = 180
    fat = kcal * 0.2
    prot = kcal * 0.15
    carb = kcal - fat - prot
    return {'name':name,'kcal':int(kcal),'carb':int(carb),'protein':int(prot),'fat':int(fat)}

menu_list = [est(n) for n in menu_names]

# 추천 실행
if st.button('식단 추천 실행'):
    # BMI·TDEE 계산
    bmi = weight / ((height/100)**2)
    target_bmi = 22.0
    target_weight = target_bmi * ((height/100)**2)
    weeks = abs((target_weight - weight) * 7700 / 500) / 7
    bmr = 10*weight + 6.25*height - 5*age + (5 if sex=='M' else -161)
    tdee = bmr * (1.2 + (activity-1)*0.15)

    # 알레르기 필터링
    filtered = [m for m in menu_list if not any(a in m['name'] for a in allergies)]
    if not filtered:
        st.warning('조건에 맞는 메뉴가 없습니다.')
        st.stop()

    # 조합 생성 및 학습
    combos = list(combinations(filtered, 3))
    X, y = [], []
    for combo in combos:
        k = sum(i['kcal'] for i in combo)
        c = sum(i['carb'] for i in combo)
        p = sum(i['protein'] for i in combo)
        f = sum(i['fat'] for i in combo)
        X.append([bmi, age, activity, k, c, p, f])
        total = c + p + f + 1e-6
        pr = p / total
        ideal_p = 0.2 + (activity-1)*0.05
        p_score = max(0, 1 - abs(pr - ideal_p))
        k_score = max(0, 1 - abs(k - tdee/3) / (tdee/3))
        y.append(1 if 0.6*k_score + 0.4*p_score >= 0.5 else 0)

    clf = None
    if len(set(y)) > 1:
        clf = Pipeline([('s',StandardScaler()),('m',RandomForestClassifier(n_estimators=100,random_state=42))])
        clf.fit(X, y)

    # 평가 및 추천
    recs = []
    for xi, combo in zip(X, combos):
        if clf:
            pr = clf.predict_proba([xi])[0]
            score = pr[1] if len(pr) > 1 else 0
        else:
            k = xi[3]; p = xi[5]; tot = xi[4] + p + xi[6] + 1e-6
            prt = p / tot; ideal_p = 0.2 + (activity-1)*0.05
            p_score = max(0, 1 - abs(prt - ideal_p))
            k_score = max(0, 1 - abs(k - tdee/3) / (tdee/3))
            score = 0.6*k_score + 0.4*p_score
        recs.append((combo, score))

    # 중복 없는 Top3 선택
    selected = []
    used = set()
    for combo, score in sorted(recs, key=lambda x:x[1], reverse=True):
        names = [i['name'] for i in combo]
        if any(n in used for n in names):
            continue
        selected.append((combo, score))
        used.update(names)
        if len(selected) == 3:
            break

    # 식사 시간: 문헌 기반 고정 시간
    b_earliest = (datetime.combine(datetime.today(), wake_time) + timedelta(minutes=30)).time()
    lunch_time = dtime(12,30)
    dinner_time= dtime(18,30)

    st.subheader(f"{name}님 맞춤 결과")
    st.write(f"- BMI: {bmi:.2f} | 목표 BMI: {target_bmi} | 목표 체중: {target_weight:.1f}kg")
    st.write(f"- TDEE: {tdee:.0f} kcal | 예상 소요: {weeks:.1f}주")
    st.markdown('### 🍽️ 하루 식단 추천')
    for (combo, score), meal_time in zip(selected, [b_earliest, lunch_time, dinner_time]):
        items = ' + '.join(i['name'] for i in combo)
        kc = sum(i['kcal'] for i in combo)
        st.write(f"{meal_time.strftime('%H:%M')} → **{items}** ({kc} kcal, 적합도 {score:.2f})")

    st.markdown('### ⏰ 증상별 영양소 일정')
    smap = {'눈떨림':[('10:00','마그네슘 300mg')],'피로':[('09:00','비타민 B2 1.4mg')]}    
    for s in symptoms:
        for tt, it in smap.get(s, []):
            st.write(f"{tt} → {it}")

    st.markdown('### ⏰ 연령별 권장 영양소')
    if age < 20:
        amap = [('08:00','칼슘 500mg'),('20:00','비타민 D 10µg')]
    elif age < 50:
        amap = [('09:00','비타민 D 10µg')]
    else:
        amap = [('08:00','칼슘 500mg'),('21:00','비타민 D 20µg')]
    for tt, it in amap:
        st.write(f"{tt} → {it}")
