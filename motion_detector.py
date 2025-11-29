#!/usr/bin/env python3
"""
움직임 감지 경고 시스템
어두운 환경에서도 작동하도록 최적화됨
"""

import cv2
import numpy as np
import threading
import time
import argparse
from datetime import datetime

# 소리를 내기 위한 라이브러리
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("경고: pygame이 설치되지 않았습니다. 시스템 beep 소리를 사용합니다.")

class MotionDetector:
    def __init__(self, sensitivity=5, min_area=100, alarm_duration=3.0, auto_resume=300):
        self.cap = None
        self.prev_frame = None
        self.motion_detected = False
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.running = False
        
        # 히스테리시스 설정
        self.motion_counter = 0
        self.motion_threshold_high = 3  # 경고 시작 임계값 (연속 3프레임)
        self.motion_threshold_low = 1   # 경고 해제 임계값
        self.alarm_active = False
        
        # 경고음 지속 시간 관리
        self.alarm_duration = alarm_duration  # 경고음 지속 시간 (초)
        self.alarm_end_time = 0
        self.alarm_thread = None
        self.alarm_thread_running = False
        
        # 경고 활성화/비활성화 설정
        self.alarm_enabled = True  # 경고 활성화 상태
        self.auto_resume_time = auto_resume  # 자동 재활성화 시간 (초)
        self.alarm_disabled_at = 0  # 경고 비활성화 시각
        
        # pygame 초기화 (경고음용)
        if PYGAME_AVAILABLE:
            pygame.mixer.init()
            self.create_alert_sound()
    
    def create_alert_sound(self):
        """경고음 생성"""
        if not PYGAME_AVAILABLE:
            return
        
        # 간단한 경고음 생성 (880Hz, 0.5초)
        sample_rate = 22050
        duration = 0.5
        frequency = 880
        
        n_samples = int(sample_rate * duration)
        buf = np.sin(2 * np.pi * frequency * np.linspace(0, duration, n_samples))
        buf = (buf * 32767).astype(np.int16)
        
        # 스테레오로 변환
        stereo_buf = np.column_stack((buf, buf))
        sound = pygame.sndarray.make_sound(stereo_buf)
        self.alert_sound = sound
    
    def alarm_loop(self):
        """경고음 반복 재생 쓰레드"""
        while self.alarm_thread_running and time.time() < self.alarm_end_time:
            if PYGAME_AVAILABLE:
                self.alert_sound.play()
            else:
                print('\a')  # beep
            
            time.sleep(1.0)  # 1초 간격 (0.5초 소리 + 0.5초 쉼)
        
        self.alarm_thread_running = False
    
    def toggle_alarm(self):
        """스페이스바로 경고 켜기/끄기"""
        self.alarm_enabled = not self.alarm_enabled
        
        if self.alarm_enabled:
            print(f"\n✅ [{datetime.now().strftime('%H:%M:%S')}] 경고 활성화됨")
            self.alarm_disabled_at = 0
        else:
            print(f"\n⏸️  [{datetime.now().strftime('%H:%M:%S')}] 경고 비활성화됨 ({self.auto_resume_time}초 후 자동 재활성화)")
            self.alarm_disabled_at = time.time()
            # 현재 울리는 경고음 중지
            self.alarm_thread_running = False
    
    def check_auto_resume(self):
        """자동 재활성화 확인"""
        if not self.alarm_enabled and self.alarm_disabled_at > 0:
            elapsed = time.time() - self.alarm_disabled_at
            if elapsed >= self.auto_resume_time:
                self.alarm_enabled = True
                self.alarm_disabled_at = 0
                print(f"\n🔔 [{datetime.now().strftime('%H:%M:%S')}] 경고 자동 재활성화됨")
    
    def trigger_alarm(self):
        """경고음 트리거 (움직임 감지시 호출)"""
        # 경고가 비활성화된 경우 무시
        if not self.alarm_enabled:
            return
        
        current_time = time.time()
        
        # 경고음 지속 시간 연장
        self.alarm_end_time = current_time + self.alarm_duration
        
        # 이미 울리고 있으면 시간만 연장하고 리턴
        if self.alarm_thread_running:
            return
        
        # 새로운 경고음 쓰레드 시작
        print(f"\n🚨 [{datetime.now().strftime('%H:%M:%S')}] 움직임 감지! 경고음 시작 🚨")
        self.alarm_thread_running = True
        self.alarm_thread = threading.Thread(target=self.alarm_loop, daemon=True)
        self.alarm_thread.start()
    
    def initialize_camera(self):
        """카메라 초기화"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            raise Exception("웹캠을 열 수 없습니다!")
        
        # 어두운 환경을 위한 카메라 설정
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # 노출 자동 조정
        self.cap.set(cv2.CAP_PROP_GAIN, 10)  # 게인 증가
        
        print("카메라 초기화 완료")
        print("몇 초간 대기하여 카메라가 안정화되도록 합니다...")
        time.sleep(2)
    
    def detect_motion(self, frame):
        """움직임 감지 (히스테리시스 적용)"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 노이즈 감소를 위한 블러
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # 첫 프레임 저장
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, frame
        
        # 프레임 차이 계산
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        
        # 임계값 적용
        thresh = cv2.threshold(frame_diff, self.sensitivity, 255, cv2.THRESH_BINARY)[1]
        
        # 노이즈 제거
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        
        # 움직임 분석
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            motion_detected = True
            
            # 움직임 영역 표시
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # 히스테리시스 로직
        if motion_detected:
            self.motion_counter += 1
            if self.motion_counter >= self.motion_threshold_high:
                self.alarm_active = True
                self.trigger_alarm()
        else:
            self.motion_counter = max(0, self.motion_counter - 1)
            if self.motion_counter <= self.motion_threshold_low:
                self.alarm_active = False
        
        # 현재 프레임을 이전 프레임으로 업데이트
        self.prev_frame = gray
        
        return self.alarm_active, frame
    
    def run(self):
        """메인 실행 루프"""
        try:
            self.initialize_camera()
            self.running = True
            
            print("\n" + "="*50)
            print("움직임 감지 시스템 시작")
            print("="*50)
            print("종료: 'q' | 경고 켜기/끄기: 스페이스바")
            print(f"민감도: {self.sensitivity} (낮을수록 민감)")
            print(f"최소 감지 영역: {self.min_area} 픽셀")
            print(f"경고음 지속 시간: {self.alarm_duration}초")
            print(f"자동 재활성화: {self.auto_resume_time}초")
            print(f"히스테리시스: {self.motion_threshold_high}프레임 이상 감지시 경고")
            print("="*50 + "\n")
            
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("프레임을 읽을 수 없습니다!")
                    break
                
                # 자동 재활성화 확인
                self.check_auto_resume()
                
                # 밝기 향상 (어두운 환경용)
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
                
                # 움직임 감지
                motion_detected, processed_frame = self.detect_motion(frame)
                
                if motion_detected:
                    # 화면에 경고 텍스트 표시
                    cv2.putText(processed_frame, "MOTION DETECTED!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 상태 표시 (경고음 울리는 중인지 표시)
                if not self.alarm_enabled:
                    status = "ALARM DISABLED"
                    color = (128, 128, 128)
                    # 남은 시간 표시
                    if self.alarm_disabled_at > 0:
                        remaining = int(self.auto_resume_time - (time.time() - self.alarm_disabled_at))
                        remaining = max(0, remaining)
                        status += f" ({remaining}s)"
                elif self.alarm_thread_running:
                    status = "ALARM ACTIVE!"
                    color = (0, 0, 255)
                elif motion_detected:
                    status = "Motion..."
                    color = (0, 165, 255)
                else:
                    status = "Monitoring..."
                    color = (0, 255, 0)
                cv2.putText(processed_frame, status, (10, processed_frame.shape[0] - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # 시간 표시
                cv2.putText(processed_frame, datetime.now().strftime('%H:%M:%S'), 
                          (processed_frame.shape[1] - 100, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 화면 표시
                cv2.imshow('Motion Detector', processed_frame)
                
                # 키 입력 처리
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # 스페이스바
                    self.toggle_alarm()
            
        except KeyboardInterrupt:
            print("\n사용자가 중단했습니다.")
        except Exception as e:
            print(f"\n오류 발생: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """정리 작업"""
        self.running = False
        self.alarm_thread_running = False
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
        print("\n시스템 종료됨")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='움직임 감지 경고 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python motion_detector.py                          # 기본 설정으로 실행
  python motion_detector.py -s 10 -m 200             # 민감도 10, 최소영역 200
  python motion_detector.py -d 5                     # 경고음 5초 지속
  python motion_detector.py -r 600                   # 자동 재활성화 10분
  python motion_detector.py -s 5 -m 100 -d 3 -r 300  # 모든 옵션 지정
  
조작:
  스페이스바: 경고 켜기/끄기 (끄면 설정 시간 후 자동으로 다시 켜짐)
  q: 종료
        """
    )
    
    parser.add_argument('-s', '--sensitivity', type=int, default=5,
                        help='민감도 (낮을수록 민감, 기본값: 5)')
    parser.add_argument('-m', '--min-area', type=int, default=100,
                        help='최소 감지 영역 (픽셀, 기본값: 100)')
    parser.add_argument('-d', '--duration', type=float, default=3.0,
                        help='경고음 지속 시간 (초, 기본값: 3.0)')
    parser.add_argument('-r', '--auto-resume', type=int, default=300,
                        help='경고 자동 재활성화 시간 (초, 기본값: 300 = 5분)')
    
    args = parser.parse_args()
    
    detector = MotionDetector(
        sensitivity=args.sensitivity,
        min_area=args.min_area,
        alarm_duration=args.duration,
        auto_resume=args.auto_resume
    )
    detector.run()