import streamlit as st
import numpy as np
import random
import csv
import time
from io import StringIO

# --- 1. 상태 및 상수 초기화 ---

# 스도쿠 기본 정답 패턴 (9x9)
BASIC_SUDOKU = np.array([
    ['1','2','3','4','5','6','7','8','9'],
    ['4','5','6','7','8','9','1','2','3'],
    ['7','8','9','1','2','3','4','5','6'],
    ['2','3','1','8','9','7','5','6','4'],
    ['5','6','4','2','3','1','8','9','7'],
    ['8','9','7','5','6','4','2','3','1'],
    ['3','1','2','6','4','5','9','7','8'],
    ['6','4','5','9','7','8','3','1','2'],
    ['9','7','8','3','1','2','6','4','5']
])

def initialize_state():
    """Streamlit 세션 상태를 초기화합니다."""
    if 'is_playing' not in st.session_state:
        st.session_state.is_playing = False
    if 'board' not in st.session_state:
        st.session_state.board = [['' for _ in range(9)] for _ in range(9)]
    if 'solution' not in st.session_state:
        st.session_state.solution = [['' for _ in range(9)] for _ in range(9)]
    if 'disabled_cells' not in st.session_state:
        st.session_state.disabled_cells = [[False for _ in range(9)] for _ in range(9)]
    if 'start_time' not in st.session_state:
        st.session_state.start_time = 0
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    if 'message' not in st.session_state:
        st.session_state.message = "Shuffle 버튼을 눌러 게임을 시작하세요!"
    if 'highlight_incorrect' not in st.session_state:
        st.session_state.highlight_incorrect = False
    if 'probability' not in st.session_state:
        st.session_state.probability = 0.7

initialize_state()

# --- 2. 게임 로직 함수 ---

def generate_board(probability):
    """새로운 스도쿠 문제를 생성합니다."""
    
    # 1. 숫자 1-9를 무작위로 치환하여 정답 생성
    random19 = np.arange(1, 10)
    np.random.shuffle(random19)
    
    current_solution = BASIC_SUDOKU.copy()
    for i in range(9):
        for j in range(9):
            current_solution[i][j] = str(random19[int(current_solution[i][j])-1])

    # 2. 확률에 따라 빈칸(문제) 생성
    current_board = current_solution.copy()
    disabled = [[False for _ in range(9)] for _ in range(9)]
    
    for i in range(9):
        for j in range(9):
            if random.random() > probability:
                current_board[i][j] = ''
                disabled[i][j] = False # 입력 가능
            else:
                disabled[i][j] = True # 고정 숫자
            
            # UI 필드 값 초기화
            st.session_state[f'cell_{i}_{j}'] = current_board[i][j]

    # 3. 상태 업데이트 및 게임 시작
    st.session_state.solution = current_solution.tolist()
    st.session_state.board = current_board.tolist()
    st.session_state.disabled_cells = disabled
    st.session_state.is_playing = True
    st.session_state.start_time = time.time()
    st.session_state.error_count = 0
    st.session_state.message = f"게임 시작! 남은 기회: {3 - st.session_state.error_count}번"
    st.session_state.highlight_incorrect = False

def check_sudoku_rules(board):
    """스도쿠 규칙 (행/열/3x3 블록 중복)을 확인합니다."""
    
    board = np.array(board)
    
    # 빈칸 체크
    if '' in board:
        return False
        
    # 숫자 변환 및 행/열/블록 규칙 확인
    try:
        current_values = board.astype(int)
    except ValueError:
        return False

    # 행/열 체크
    for i in range(9):
        if len(set(current_values[i, :])) != 9 or len(set(current_values[:, i])) != 9:
            return False
            
    # 3x3 박스 체크
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            box = current_values[i:i+3, j:j+3].flatten()
            if len(set(box)) != 9:
                return False
                
    return True

def handle_finish_click():
    """Finish 버튼 클릭을 처리하고 정답을 확인합니다."""
    if not st.session_state.is_playing:
        st.session_state.message = "먼저 Shuffle 버튼을 눌러 게임을 시작하세요!"
        return

    is_filled = all(st.session_state.board[i][j] != '' for i in range(9) for j in range(9))
    is_correct = check_sudoku_rules(st.session_state.board)
    is_solution_match = all(st.session_state.board[i][j] == st.session_state.solution[i][j] 
                            for i in range(9) for j in range(9))

    if not is_filled:
        st.session_state.error_count += 1
        st.session_state.message = f"빈칸을 모두 채워주세요! 남은 기회: {3 - st.session_state.error_count}번"
        st.session_state.highlight_incorrect = False
    elif is_solution_match and is_correct: # 성공
        st.session_state.is_playing = False
        time_taken = time.time() - st.session_state.start_time
        st.session_state.time_taken = int(time_taken)
        st.session_state.message = "🎉 **!!! ~~~Congratulation~~~ !!!** 🎉"
        st.session_state.highlight_incorrect = False
        st.session_state.show_name_input = True
    else: # 실패
        st.session_state.error_count += 1
        st.session_state.highlight_incorrect = True
        
        if st.session_state.error_count >= 3:
            st.session_state.is_playing = False
            st.session_state.message = "**게임 오버:** 3번의 기회를 모두 소진했습니다."
            st.session_state.highlight_incorrect = True
        else:
            st.session_state.message = f"오류가 있습니다. 남은 기회: {3 - st.session_state.error_count}번"


def load_ranking():
    """메모리에서 순위 데이터를 읽습니다."""
    try:
        if 'ranking_data' not in st.session_state:
            st.session_state.ranking_data = StringIO("Name,Time\n")
            
        st.session_state.ranking_data.seek(0)
        reader = csv.reader(st.session_state.ranking_data)
        
        rankings = []
        for i, row in enumerate(reader):
            if i == 0 or not row or len(row) < 2: continue
            try:
                rankings.append((row[0], int(row[1])))
            except ValueError:
                continue

        rankings.sort(key=lambda x: x[1])
        return rankings
    except Exception:
        return []

def save_ranking(player_name, time_taken):
    """순위를 메모리 (StringIO)에 저장합니다."""
    st.session_state.ranking_data.seek(0, 2)
    writer = csv.writer(st.session_state.ranking_data)
    writer.writerow([player_name, time_taken])
    st.session_state.message = f"'{player_name}'님의 기록({time_to_string(time_taken)})이 등록되었습니다! ✨"
    st.session_state.show_name_input = False
    
def time_to_string(seconds):
    """초를 MM:SS 형식의 문자열로 변환합니다."""
    minutes = seconds // 60
    secs = seconds % 60
    return f"{minutes:02}:{secs:02}"

def update_cell_value(i, j):
    """셀 값이 변경될 때 상태를 업데이트합니다."""
    new_value = st.session_state[f'cell_{i}_{j}']
    
    # 입력 값 유효성 (1-9 또는 공백) 검사
    if new_value == '' or (new_value.isdigit() and 1 <= int(new_value) <= 9):
        # board 상태 업데이트
        st.session_state.board[i][j] = new_value
        st.session_state.highlight_incorrect = False # 하이라이트 해제
    else:
        # 유효하지 않은 입력 시 이전 값으로 되돌림
        st.session_state[f'cell_{i}_{j}'] = st.session_state.board[i][j]
        st.session_state.message = "Error: 1~9 사이의 숫자 또는 빈칸만 입력 가능합니다."


# --- 3. Streamlit UI 렌더링 ---

st.set_page_config(layout="wide")
st.title("🔢 Streamlit Sudoku Game")

# CSS 스타일 정의: 입력 필드, 3x3 경계선, 고정/오류 셀
st.markdown("""
<style>
.stTextInput > div > div > input {
    text-align: center;
    font-size: 1.25rem !important;
    padding: 0.5rem 0.2rem;
    height: 2.5rem;
    border-radius: 0.375rem;
}
.fixed-number > div > div > input {
    font-weight: bold;
    color: #1f2937 !important;
    background-color: #e5e7eb !important;
    cursor: default;
}
.sudoku-cell {
    border: 1px solid #d1d5db;
    padding: 1px;
}
.block-border-right {
    border-right: 3px solid #6b7280 !important;
}
.block-border-bottom {
    border-bottom: 3px solid #6b7280 !important;
}
.incorrect-cell > div > div > input {
    background-color: #fee2e2 !important;
    border: 2px solid #ef4444 !important;
    color: #ef4444 !important;
}
</style>
""", unsafe_allow_html=True)


# 상단 제어판 레이아웃
col_shuffle, col_prob, col_finish, col_timer = st.columns([1, 1, 1, 1])

with col_shuffle:
    # 게임 시작 버튼
    st.button("🔄 Shuffle", on_click=lambda: generate_board(st.session_state.probability), type="primary", use_container_width=True)

with col_prob:
    # 난이도 슬라이더
    st.session_state.probability = st.slider("빈칸 확률(1-p)", 0.1, 0.9, st.session_state.probability, 0.05)

with col_finish:
    # 정답 확인 버튼
    st.button("✅ Finish", on_click=handle_finish_click, type="secondary", use_container_width=True)

with col_timer:
    # 타이머 표시
    if st.session_state.is_playing and st.session_state.start_time != 0:
        elapsed_time = int(time.time() - st.session_state.start_time)
        timer_text = time_to_string(elapsed_time)
    elif 'time_taken' in st.session_state and not st.session_state.is_playing:
        timer_text = time_to_string(st.session_state.time_taken)
    else:
        timer_text = "00:00"
        
    st.markdown(f"""
        <div style="text-align: center; background-color: #f3f4f6; padding: 0.5rem; border-radius: 0.5rem;">
            <p style="font-size: 1.2rem; font-weight: bold; margin: 0;">⏱️ {timer_text}</p>
        </div>
    """, unsafe_allow_html=True)

# 메시지 출력
st.info(st.session_state.message)

# 랭킹 표시 (Sidebar)
with st.sidebar:
    st.subheader("🏆 Game Ranking")
    rankings = load_ranking()
    
    ranking_text = ""
    if rankings:
        for i, (name, time_taken) in enumerate(rankings[:5]):
            ranking_text += f"**{i+1}.** {name}: {time_to_string(time_taken)}\n"
    else:
        ranking_text = "기록이 없습니다."
        
    st.markdown(ranking_text)


# 스도쿠 보드 UI (9x9 그리드)
for i in range(9):
    cols = st.columns(9)
    for j in range(9):
        cell_key = f'cell_{i}_{j}'
        
        # 표시할 셀 값 가져오기
        cell_value_to_display = st.session_state.get(cell_key, st.session_state.board[i][j])
        
        is_disabled = st.session_state.disabled_cells[i][j]
        is_incorrect_highlight = st.session_state.highlight_incorrect and (cell_value_to_display != st.session_state.solution[i][j] and not is_disabled)
        
        # 3x3 블록 및 상태에 따른 CSS 클래스 설정
        cell_class = "sudoku-cell"
        if (j + 1) % 3 == 0 and j != 8:
            cell_class += " block-border-right"
        if (i + 1) % 3 == 0 and i != 8:
            cell_class += " block-border-bottom"
        if is_disabled:
            cell_class += " fixed-number"
        if is_incorrect_highlight:
             cell_class += " incorrect-cell"
        
        # Streamlit 텍스트 입력 위젯 렌더링
        with cols[j]:
            st.markdown(f'<div class="{cell_class}">', unsafe_allow_html=True)
            
            st.text_input(
                label="", 
                value=cell_value_to_display, 
                max_chars=1, 
                key=cell_key, 
                disabled=is_disabled or not st.session_state.is_playing, # 고정/게임 중 상태에 따라 비활성화
                label_visibility="collapsed",
                on_change=update_cell_value,
                args=(i, j)
            )
            st.markdown('</div>', unsafe_allow_html=True)


# 게임 완료 시 이름 입력 창 표시
if st.session_state.get('show_name_input', False):
    st.subheader("이름을 입력하여 순위를 등록하세요")
    player_name = st.text_input("사용자 이름", key="player_name_input")
    
    if st.button("순위 등록", use_container_width=True):
        if player_name:
            save_ranking(player_name, st.session_state.time_taken)
            del st.session_state.show_name_input
            st.rerun()
        else:
            st.warning("이름을 입력해주세요.")
