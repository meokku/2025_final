# Final Project: EyeMouse 구현
import pyautogui
import cv2
import mediapipe as mp
import numpy as np

# 안구 마우스 문제점: 시선과 커서 이동 간의 불일치
# 이를 조정하기 위해 보정(calibrating) 시행, 관련 변수 설정
is_calibrating = False
calibration_complete = False
gaze_coords_x = []
gaze_coords_y = [] 
calib_box = None # 보정용 box 설정: 추후 calib_box의 범위가 모니터 전체 범위로 확대
selected_iris_index = None  # 주시안 선택 변수 (아직 지정되지 않음)

# 눈 깜빡임 감지용 상수
EAR_THRESHOLD = 0.23 # 사용자에 맞게 튜닝된 값
BLINK_CLICK_FRAMES = 5 # 의도적 클릭으로 판단할 프레임 수

# 커서 움직임이 너무 빠른 것을 방지하기 위해 smoothing factor 설정
SMOOTHING_FACTOR = 0.2

# 눈 깜빡임 카운터
blink_counter = 0

# 화면 크기 및 smoothing 좌표 초기화
screen_w, screen_h = pyautogui.size()
smoothed_mx, smoothed_my = screen_w / 2, screen_h / 2

# MediaPipe Face Mesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,                # 탐지 얼굴: 1개
    refine_landmarks=True,          # 눈, 입술 주변 랜드마크 정밀도 향상 (iris 추적에 필수)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 지정 및 해상도 설정
video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 추적할 눈의 랜드마크 인덱스
LEFT_IRIS_INDEX = 473
RIGHT_IRIS_INDEX = 468

# EAR 계산에 필요한 6개 랜드마크 인덱스 (P1, P2, P3, P4, P5, P6 순서)
EAR_LEFT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]
EAR_RIGHT_EYE_LANDMARKS = [33, 158, 159, 133, 144, 145]

# window_normal: 윈도우 크기 조정 가능한 방식
WINDOW_NAME = 'Eyemouse'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

# failsafe 해제: 커서 이동 시 화면 밖으로 나가는 경우 eyemouse가 멈추는 상황 방지
pyautogui.FAILSAFE = False

# 눈 영역 확대 함수
def zoom_eye_region(frame, eye_landmarks, zoom_factor=2.0):
    if eye_landmarks is None:
        return frame
    
    # 눈 영역의 중심점 계산
    center_x = int(eye_landmarks.x * frame.shape[1])
    center_y = int(eye_landmarks.y * frame.shape[0])
    
    # 확대할 영역의 크기 계산
    zoom_size = 100
    x1 = max(0, center_x - zoom_size)
    y1 = max(0, center_y - zoom_size)
    x2 = min(frame.shape[1], center_x + zoom_size)
    y2 = min(frame.shape[0], center_y + zoom_size)
    
    # 눈 영역 추출 및 확대
    eye_region = frame[y1:y2, x1:x2]
    if eye_region.size > 0:
        eye_region = cv2.resize(eye_region, None, fx=zoom_factor, fy=zoom_factor)
        
        # 확대된 영역에서 홍채 위치 계산
        zoomed_center_x = int((center_x - x1) * zoom_factor)
        zoomed_center_y = int((center_y - y1) * zoom_factor)
        
        # 확대된 영역에 홍채 위치 표시
        cv2.circle(eye_region, (zoomed_center_x, zoomed_center_y), 5, (0, 255, 0), -1)
        
        # 확대된 영역을 원본 프레임에 합성
        h, w = eye_region.shape[:2]
        frame[10:10+h, 10:10+w] = eye_region
        
        # 확대된 영역 표시
        cv2.rectangle(frame, (10, 10), (10+w, 10+h), (0, 255, 0), 2)
    
    return frame

# EAR 계산 함수 정의
def calculate_ear(eye_landmark_indices, face_landmarks):
    # eye_landmark_indices는 P1~P6에 해당하는 6개의 인덱스를 담은 리스트
    p1 = face_landmarks.landmark[eye_landmark_indices[0]]
    p2 = face_landmarks.landmark[eye_landmark_indices[1]]
    p3 = face_landmarks.landmark[eye_landmark_indices[2]]
    p4 = face_landmarks.landmark[eye_landmark_indices[3]]
    p5 = face_landmarks.landmark[eye_landmark_indices[4]]
    p6 = face_landmarks.landmark[eye_landmark_indices[5]]

    # 유클리드 거리 계산
    ver_dist1 = np.linalg.norm([p2.x - p6.x, p2.y - p6.y])
    ver_dist2 = np.linalg.norm([p3.x - p5.x, p3.y - p5.y])
    hor_dist = np.linalg.norm([p1.x - p4.x, p1.y - p4.y])

    # 수평 거리가 0일 때 발생하는 DivisionByZeroError 방지
    if hor_dist == 0: return 0.0
    ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)
    return ear

while True:
    ret, frame = video.read()
    if not ret:
        print("frame을 읽을 수 없습니다.")
        break
    
    # frame 좌우반전 시행 (사용자 화면과 ui 화면을 일치시키기 위해서)
    frame = cv2.flip(frame, 1)

    # MediaPipe 처리를 위해 BGR 이미지를 RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # FaceMesh 처리
    results = face_mesh.process(rgb_frame)
    
    # h, w, _: 현재 frame의 높이, 너비 및 채널 수
    h, w, _ = frame.shape
    
    # 얼굴 랜드마크가 감지된 경우
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        
        # 눈을 선택한 후, 보정 및 추후 작업 진행
        if selected_iris_index is not None: 
            # 선택된 눈 영역 확대
            iris_landmark = face_landmarks.landmark[selected_iris_index]
            frame = zoom_eye_region(frame, iris_landmark)

            ear_landmarks = EAR_LEFT_EYE_LANDMARKS if selected_iris_index == LEFT_IRIS_INDEX else EAR_RIGHT_EYE_LANDMARKS
            ear = calculate_ear(ear_landmarks, face_landmarks)
            cv2.putText(frame, f"EAR: {ear:.2f}", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter > BLINK_CLICK_FRAMES:
                    pyautogui.click()
                    print(f"Blink Click! (duration: {blink_counter} frames)")
                blink_counter = 0

            # 눈을 뜨고 있을 때만 커서 위치 업데이트 (클릭 시 커서 고정)
            if blink_counter == 0:
                # 특정 랜드마크(선택된 눈 홍채)의 좌표 추출
                iris_landmark = face_landmarks.landmark[selected_iris_index]
                
                # 랜드마크 좌표는 0~1 사이로 정규화되어 있으므로, 실제 픽셀 좌표로 변환
                iris_x = int(iris_landmark.x * w)
                iris_y = int(iris_landmark.y * h)
                
                if is_calibrating:
                    # 보정 중일 때, 현재 홍채 위치를 gaze_coords에 추가
                    gaze_coords_x.append(iris_x)
                    gaze_coords_y.append(iris_y)

                # 보정 완료 -> 실제 커서 위치 조정
                if calibration_complete and calib_box is not None:
                    # calib_box의 범위에서, 현재 홍채 위치 비율 계산
                    x_ratio = (iris_x - calib_box['min_x']) / (calib_box['max_x'] - calib_box['min_x'])
                    y_ratio = (iris_y - calib_box['min_y']) / (calib_box['max_y'] - calib_box['min_y'])

                    x_ratio = max(0.0, min(1.0, x_ratio))
                    y_ratio = max(0.0, min(1.0, y_ratio))

                    target_mx, target_my = screen_w * x_ratio, screen_h * y_ratio
                    
                    # smoothing을 적용하여 커서 부드럽게 이동
                    smoothed_mx = SMOOTHING_FACTOR * target_mx + (1.0 - SMOOTHING_FACTOR) * smoothed_mx
                    smoothed_my = SMOOTHING_FACTOR * target_my + (1.0 - SMOOTHING_FACTOR) * smoothed_my
                    
                    pyautogui.moveTo(smoothed_mx, smoothed_my)
                    
                    # calib_box 표시
                    cv2.rectangle(frame, (calib_box['min_x'], calib_box['min_y']), 
                                (calib_box['max_x'], calib_box['max_y']), (0, 255, 255), 1)
                
                cv2.circle(frame, (iris_x, iris_y), 5, (0, 255, 0), -1)
    else:
        cv2.putText(frame, "No face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


    # 홍채 인식 및 보정 상태에 따른 UI 설정
    if selected_iris_index is None:
        cv2.putText(frame, "Select Eye: L-key (Left) / R-key (Right)", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    elif is_calibrating:
        cv2.putText(frame, "Calibrating... Move head/eyes", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif not calibration_complete:
        cv2.putText(frame, "Press C to Calibrate", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    elif calibration_complete:
        cv2.putText(frame, "Calibrated. Blink to Click.", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow(WINDOW_NAME, frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # UI 관련 입력 로직
    # q: 종료
    if key == ord('q'):
        break
    
    # L/l: 화면 왼쪽 눈 선택
    elif key == ord('l') or key == ord('L'):
        selected_iris_index = RIGHT_IRIS_INDEX # 거울모드이므로 실제 오른쪽 눈
        print(f"Screen-Left (Actual Right) eye selected. Landmark: {selected_iris_index}")
        calibration_complete = False; is_calibrating = False; calib_box = None

    # R/r: 화면 오른쪽 눈 선택
    elif key == ord('r') or key == ord('R'):
        selected_iris_index = LEFT_IRIS_INDEX # 거울모드이므로 실제 왼쪽 눈
        print(f"Screen-Right (Actual Left) eye selected. Landmark: {selected_iris_index}")
        calibration_complete = False; is_calibrating = False; calib_box = None

    # C/c: 보정 시작/종료
    elif key == ord('c') or key == ord('C'):
        if selected_iris_index is not None: # 눈이 선택된 후에 동작
            if not is_calibrating:
                is_calibrating = True
                calibration_complete = False
                gaze_coords_x.clear(); gaze_coords_y.clear()
                print("Calibration started...")
            else:
                is_calibrating = False
                if gaze_coords_x and gaze_coords_y:
                    calib_box = {
                        'min_x': min(gaze_coords_x), 'max_x': max(gaze_coords_x),
                        'min_y': min(gaze_coords_y), 'max_y': max(gaze_coords_y)
                    }
                    
                    # padding 설정: 실제 보정 값보다 조금 더(10%) 쉽게 커서 이동
                    padding_x = int((calib_box['max_x'] - calib_box['min_x']) * 0.1)
                    padding_y = int((calib_box['max_y'] - calib_box['min_y']) * 0.1)
                    calib_box['min_x'] -= padding_x; calib_box['max_x'] += padding_x
                    calib_box['min_y'] -= padding_y; calib_box['max_y'] += padding_y

                    calibration_complete = True
                    print("Calibration complete!"); print("Calibrated Box:", calib_box)
                    
                    # 보정 완료 후 커서를 화면 중앙으로 이동
                    screen_w, screen_h = pyautogui.size()
                    pyautogui.moveTo(screen_w / 2, screen_h / 2)
                    smoothed_mx, smoothed_my = screen_w / 2, screen_h / 2
                else:
                    calibration_complete = False; calib_box = None
                    print("Calibration failed: No movement detected.")
    
    # m: 커서 중앙 이동
    elif key == ord('m'):
        screen_w, screen_h = pyautogui.size()
        pyautogui.moveTo(screen_w / 2, screen_h / 2)

video.release()
cv2.destroyAllWindows()