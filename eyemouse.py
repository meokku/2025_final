# Final Project: EyeMouse 구현
import pyautogui
import cv2
import time
import pyautogui

# 웹캠 지정
video = cv2.VideoCapture(0)

# harrcascade_eye.xml 지정 -> 눈 및 얼굴 RoI 인식용
# 얼굴 RoI 인식: 눈만 인식 대상으로 했을 때, 다른 물체(전구)도 눈으로 인식하는 부분 개선
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

while True:
    ret, frame = video.read()
    if not ret:
        # 프레임을 제대로 읽지 못했을 때 루프 종료
        print("frame not found")
        break

    # harrcascade: grayscale 기준으로 학습했기 때문
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 RoI 인식 (이후 눈 RoI)
    # scalefactor: 이미지(눈) 크기 조정 비율, minneighbors: 값이 클수록 RoI 인식 정확도 높아짐
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    # 얼굴 인식 불가능 시: No face detected가 계속 떠있도록 설정됨
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        continue # faces 인식부터 다시 시도
        
    # 얼굴이 보이면 다음 단계(눈 ROI)로 진행
    else:
        for (x, y, w, h) in faces:
            if w < 100: # 얼굴 크기가 작게 잡히면, 관련 메세지 출력하면서 다시 시도
                cv2.putText(frame, "Move closer!", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                continue
            
            roi_gray = gray[y:y+h, x:x+w] # 얼굴 RoI 추출
            
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=5)
            eyes = sorted(eyes, key=lambda e: e[0])[:2] # 눈 RoI 추출: 왼쪽 눈, 오른쪽 눈 순서로 정렬
            
            for i, (ex, ey, ew, eh) in enumerate(eyes):
                # 얼굴 내에서 너무 아래 위치한 박스는 무시 -> 입 관련 오탐 예방 (딥러닝 기반 방식이 아니라서 이러한 로직 설정 필요)
                if ey + eh > h * 0.6:
                    continue
                
                # 눈 크기 관련 로직 -> 동공 가끔 안잡히는게 이것 때문일 수도 있음
                # 이미 Face -> Eye로 추출했기 때문에 해당 로직까지는 필요 없을 수도 있음
                if ew < 10 or eh < 10 or ew > w * 0.7:
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
                        
                        # 전체 화면에서 동공 중심 좌표 설정
                        abs_x = x + ex + cx
                        abs_y = y + ey + cy
                        
                        # 동공 구체화: 초록 원 그리기
                        cv2.circle(frame, (abs_x, abs_y), 5, (0, 255, 0), -1)

                        # 마우스 이동 로직 추가: pyautogui 기반 #
                        # 화면 해상도 조회: 따로 size함수 안쓰는 휴리스틱 방식은 커서 이동에 문제가 있었음
                        screen_w, screen_h = pyautogui.size()
                        cam_w, cam_h = frame.shape[1], frame.shape[0]
                        
                        # mx, my: screen 변수 기반, abs 값을 실제 화면에 매칭
                        mx = int((abs_x / cam_w) * screen_w)
                        my = int((abs_y / cam_h) * screen_h)
                        
                        # 커서 이동 값(mx, my)가 화면 범위 안에 존재하도록 설정
                        mx = max(0, min(mx, screen_w  - 1))
                        my = max(0, min(my, screen_h  - 1))
                        
                        # 커서 이동
                        pyautogui.moveTo(mx, my)
                        
                        # 눈 영역 contour 시각화
                        eye_color = cv2.cvtColor(eye_img, cv2.COLOR_GRAY2BGR)
                        scaled_cnt = (max_cnt / SCALE).astype(int)
                        cv2.drawContours(eye_color, [scaled_cnt], -1, (0, 255, 0), 1)

                # 디버깅용 시각화: thresh와 eye roi, eye contour 각각 출력 가능
                # cv2.imshow(f'Thresholded Eye {i}', thresh)
                # cv2.imshow(f'Eye Region {i}', eye_img)
                # cv2.imshow(f'Eye Contour {i}', eye_color)


    # 전체 프레임 출력
    cv2.imshow('Eye Tracker', frame)

    # 적절한 프레임 설정 필요 (해당 수치 조정 or 프레임 최적화 방안 탐색)
    if cv2.waitKey(20) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()