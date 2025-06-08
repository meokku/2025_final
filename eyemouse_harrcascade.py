# Final Project: EyeMouse 구현
import pyautogui
import cv2
import time

# 웹캠 지정
video = cv2.VideoCapture(0)

# harrcascade_eye.xml 지정 -> 눈 및 얼굴 RoI 인식용
# 얼굴 RoI 인식: 눈만 인식 대상으로 했을 때, 다른 물체(전구 등)도 눈으로 인식하는 부분 개선
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

# 마우스 속도 변수 지정: 처음 실행했을 때 마우스 속도가 너무 빨라서 설정
mouse_speed = 2
mouse_speed_step = 0.2

while True:
    ret, frame = video.read()
    if not ret:
        # 프레임을 제대로 읽지 못했을 때 루프 종료
        print("frame not found")
        break
    
    # 해상도 조정: 해상도를 낮춰 프레임이 느려지지 않도록 설정
    frame = cv2.resize(frame, (640, 480))

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
            
            # 마우스 커서 이동 시 안정성을 위해, 한쪽 눈만 추적하도록 설정
            if len(eyes) > 0:
                (ex, ey, ew, eh) = sorted(eyes, key=lambda e: e[0])[0] # 첫번째 눈: 오른쪽 눈만 인식하도록 설정 (웹캠과 실제 눈 위치는 반대)    
                # 얼굴 내에서 너무 아래 위치한 박스는 무시 -> 입 관련 오탐 예방 (딥러닝 기반 방식이 아니라서 이러한 로직 설정 필요)
                if ey + eh > h * 0.6:
                    continue
                
                # 눈 RoI 추출
                eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
                if eye_img.size == 0:
                    continue
            
                # 눈 영역 확대: 동공 부분 추출 및 contour 추적을 위한 전처리
                # SCALE: 2 -> x, y 측면에서 2배 확대
                SCALE = 2 
                eye_resized = cv2.resize(eye_img, (0, 0), fx=SCALE, fy=SCALE)

                # GaussianBlur: 노이즈 제거, adaptiveThreshold: 동공 부분을 강조(흰색으로 처리)
                # GaussinanBlur는 강의시간에 언급된 내용이라 써봄 (threshold도 배웠던 것 같긴 한데 정확히 찾을 필요 있음)
                eye_blur = cv2.GaussianBlur(eye_resized, (7, 7), 0)
                thresh = cv2.adaptiveThreshold(
                    eye_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                
                # 흰색 처리한 동공 부분에서 contour 추출
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                # contour 중 가장 큰 contour 설정 및 중심 좌표 계산
                if contours:
                    max_cnt = max(contours, key=cv2.contourArea)
                    M = cv2.moments(max_cnt)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00']) // SCALE
                        cy = int(M['m01'] / M['m00']) // SCALE
                        
                        # 전체 화면에서 동공 중심의 좌표 설정
                        abs_x = x + ex + cx
                        abs_y = y + ey + cy
                        
                        # 동공 구체화: 초록 원 그리기
                        cv2.circle(frame, (abs_x, abs_y), 5, (0, 255, 0), -1)

                        # 마우스 이동 로직 추가: pyautogui 기반 #
                        # 화면 해상도 조회: 따로 size함수 안쓰는 휴리스틱 방식은 커서 이동이 원활하지 않았음
                        screen_w, screen_h = pyautogui.size()
                        cam_w, cam_h = frame.shape[1], frame.shape[0] # cam_w, cam_h: 카메라 해상도 
                        
                        # 커서 이동 시 앞서 설정한 mouse_speed 적용
                        dx = (abs_x - cam_w / 2) * mouse_speed
                        dy = (abs_y - cam_h / 2) * mouse_speed
                        
                        # mx, my: screen 변수 기반, abs 값을 실제 화면에 매칭
                        mx = screen_w / 2 + dx
                        my = screen_h / 2 + dy
                        
                        # 커서 이동 값(mx, my)가 화면 범위 안에 존재하도록 설정
                        mx = max(0, min(mx, screen_w  - 1))
                        my = max(0, min(my, screen_h  - 1))
                        
                        # 커서 이동
                        pyautogui.moveTo(mx, my)
                        
                        # 눈 영역 contour 시각화
                        # eye_color = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2BGR)
                        # scaled_cnt = (max_cnt / SCALE).astype(int)
                        # cv2.drawContours(eye_color, [scaled_cnt], -1, (0, 255, 0), 1)


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
        pyautogui.moveTo(screen_w / 2, screen_h / 2)


video.release()
cv2.destroyAllWindows()