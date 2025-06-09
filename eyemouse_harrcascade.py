# Final Project: EyeMouse 구현
import pyautogui
import cv2
import time
import numpy as np

# 웹캠 지정
video = cv2.VideoCapture(0)

# harrcascade_eye.xml 지정 -> 눈 및 얼굴 RoI 인식용
# 얼굴 RoI 인식: 눈만 인식 대상으로 했을 때, 다른 물체(전구 등)도 눈으로 인식하는 부분 개선
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 마우스 속도 변수 지정
mouse_speed = 0.2  # 속도 더 감소
mouse_speed_step = 0.1

# 눈 영역 확대 비율
EYE_SCALE = 3

# 마우스 이동 제한 설정
MAX_MOVE_DISTANCE = 20  # 이동 거리 제한 더 감소

# 동공 추적 설정
MIN_CONTOUR_AREA = 50  # 최소 동공 크기
MOVEMENT_THRESHOLD = 10  # 움직임 감지 임계값 증가
last_pupil_pos = None  # 이전 동공 위치 저장
initial_position_set = False  # 초기 위치 설정 여부

# 사용할 눈 선택 (0: 왼쪽 눈, 1: 오른쪽 눈)
selected_eye = 0  # 기본값은 왼쪽 눈

# 눈 선택 UI 생성
def create_eye_selection_window():
    # 선택 창 생성
    selection_window = np.zeros((200, 400, 3), dtype=np.uint8)
    
    # 텍스트 추가
    cv2.putText(selection_window, "Select Eye for Mouse Control", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(selection_window, "1: Left Eye (Your Right)", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(selection_window, "2: Right Eye (Your Left)", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(selection_window, "Press 1 or 2 to select", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Eye Selection', selection_window)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            return 0  # 왼쪽 눈
        elif key == ord('2'):
            return 1  # 오른쪽 눈

# 눈 선택 UI 표시
selected_eye = create_eye_selection_window()
cv2.destroyWindow('Eye Selection')

while True:
    ret, frame = video.read()
    if not ret:
        # 프레임을 제대로 읽지 못했을 때 루프 종료
        print("frame not found")
        break
    
    # 해상도 조정: 해상도를 낮춰 프레임이 느려지지 않도록 설정
    frame = cv2.resize(frame, (640, 480))

    # 거울 모드로 변환
    frame = cv2.flip(frame, 1)

    # harrcascade: grayscale 기준으로 학습했기 때문에 gray로 변환
    # 해당 코드를 추가하지 않았을 때 얼굴 RoI 인식이 원활하지 않았음
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 RoI 인식 (이후 눈 RoI)
    # scalefactor: 이미지(눈) 크기 조정 비율, minneighbors: 값이 클수록 RoI 인식 정확도 높아짐
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)
    
    # 얼굴 인식 불가능 시: No face detected가 계속 떠있도록 설정됨
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        continue # faces 인식부터 다시 시도
        
    # 얼굴이 보이면 다음 단계(눈 ROI)로 진행
    else:
        for (x, y, w, h) in faces:
            if w < 100: # 얼굴 크기가 작게 잡히면, 크기 인식을 위한 명령어 출력 및 재시도 요구
                cv2.putText(frame, "Move closer!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue
            
            roi_gray = gray[y:y+h, x:x+w] # 얼굴 RoI 추출
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.3, minNeighbors = 4)
            
            # 양쪽 눈 모두 추적하도록 수정
            if len(eyes) > 0:
                # 눈들을 x 좌표 기준으로 정렬 (거울 모드에 맞게 정렬)
                sorted_eyes = sorted(eyes, key=lambda e: e[0], reverse=True)
                
                # 양쪽 눈에 대해 처리
                for i, (ex, ey, ew, eh) in enumerate(sorted_eyes):
                    # 입 관련 오탐 예방
                    if ey + eh > h * 0.6:
                        continue
                    
                    # 눈 RoI 추출 및 확대
                    eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                    if eye_img.size == 0:
                        continue
                    
                    # 눈 영역 확대
                    eye_resized = cv2.resize(eye_img, (0, 0), fx=EYE_SCALE, fy=EYE_SCALE)
                    
                    # 눈 영역 디버깅용 창 표시 (선택된 눈만)
                    if i == selected_eye:
                        cv2.imshow(f'Eye {i+1}', eye_resized)

                    # GaussianBlur 및 threshold 처리
                    eye_blur = cv2.GaussianBlur(eye_resized, (7, 7), 0)
                    thresh = cv2.adaptiveThreshold(
                        eye_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                        cv2.THRESH_BINARY_INV, 11, 2
                    )
                    
                    # contour 추출
                    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # contour 중 가장 큰 contour 설정 및 중심 좌표 계산
                    if contours:
                        # 면적이 충분히 큰 contour만 선택
                        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
                        if not valid_contours:
                            continue
                            
                        max_cnt = max(valid_contours, key=cv2.contourArea)
                        M = cv2.moments(max_cnt)
                        if M['m00'] != 0:
                            # 확대된 이미지에서의 중심 좌표 계산
                            cx = int(M['m10'] / M['m00']) // EYE_SCALE
                            cy = int(M['m01'] / M['m00']) // EYE_SCALE
                            
                            # 전체 화면에서 동공 중심의 좌표 설정
                            abs_x = x + ex + cx
                            abs_y = y + ey + cy
                            
                            # 선택된 눈에만 초록 원 그리기
                            if i == selected_eye:
                                cv2.circle(frame, (abs_x, abs_y), 5, (0, 255, 0), -1)
                                
                                # 눈 영역 내에서의 상대적 위치 계산
                                rel_x = (cx - (ex + ew // 2)) / (ew // 2)
                                rel_y = (cy - (ey + eh // 2)) / (eh // 2)
                                
                                # 현재 동공 위치 저장
                                current_pupil_pos = (rel_x, rel_y)
                                
                                # 초기 위치 설정
                                if not initial_position_set:
                                    screen_w, screen_h = pyautogui.size()
                                    pyautogui.moveTo(screen_w // 2, screen_h // 2)
                                    initial_position_set = True
                                    last_pupil_pos = current_pupil_pos
                                    continue
                                
                                # 이전 위치와 비교하여 움직임이 충분할 때만 마우스 이동
                                if last_pupil_pos is not None:
                                    dx = rel_x - last_pupil_pos[0]
                                    dy = rel_y - last_pupil_pos[1]
                                    
                                    # 움직임이 임계값을 넘을 때만 마우스 이동
                                    if abs(dx) > MOVEMENT_THRESHOLD/100 or abs(dy) > MOVEMENT_THRESHOLD/100:
                                        # 화면 크기에 맞게 변환
                                        screen_w, screen_h = pyautogui.size()
                                        move_x = dx * screen_w * mouse_speed
                                        move_y = dy * screen_h * mouse_speed
                                        
                                        # 이동 거리 제한
                                        move_x = max(-MAX_MOVE_DISTANCE, min(MAX_MOVE_DISTANCE, move_x))
                                        move_y = max(-MAX_MOVE_DISTANCE, min(MAX_MOVE_DISTANCE, move_y))
                                        
                                        # 부드러운 이동을 위해 moveRel 사용
                                        pyautogui.moveRel(move_x, move_y, duration=0.02)  # duration 증가
                                
                                # 현재 위치를 이전 위치로 저장
                                last_pupil_pos = current_pupil_pos

    # 웹캠 화면 출력
    cv2.imshow('Eye tracker', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # q: 종료, z: 마우스 속도 감소, x: 마우스 속도 증가, m: 화면 중앙으로 커서 이동
    if key == ord('q'):
        break
    elif key == ord('z'):
        mouse_speed = max(mouse_speed_step, mouse_speed - mouse_speed_step)
    elif key == ord('x'):
        mouse_speed += mouse_speed_step
    elif key == ord('m'):
        screen_w, screen_h = pyautogui.size()
        pyautogui.moveTo(screen_w // 2, screen_h // 2)


video.release()
cv2.destroyAllWindows()