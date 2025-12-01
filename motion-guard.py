#!/usr/bin/env python3
"""
Motion Detection Alert System
Optimized for low-light environments
"""

import cv2
import numpy as np
import threading
import time
import argparse
import locale
from datetime import datetime

# Multi-language message table
MESSAGES = {
    'en': {
        'pygame_warning': "Warning: pygame not installed. Using system beep sound.",
        'camera_init': "Camera initialized",
        'camera_wait': "Waiting a few seconds for camera to stabilize...",
        'camera_error': "Cannot open webcam!",
        'system_start': "Motion Detection System Started",
        'controls': "Quit: 'q' | Toggle Alarm: Spacebar",
        'sensitivity': "Sensitivity",
        'lower_more_sensitive': "lower = more sensitive",
        'min_area': "Min Detection Area",
        'pixels': "pixels",
        'alarm_duration': "Alarm Duration",
        'seconds': "seconds",
        'auto_resume': "Auto Resume",
        'hysteresis': "Hysteresis",
        'frames_to_trigger': "frames or more to trigger",
        'motion_detected_alarm': "Motion detected! Alarm started",
        'alarm_enabled': "Alarm enabled",
        'alarm_disabled': "Alarm disabled",
        'auto_resume_after': "Auto resume after {0} seconds",
        'alarm_auto_resumed': "Alarm auto-resumed",
        'user_interrupted': "User interrupted.",
        'error_occurred': "Error occurred",
        'system_shutdown': "System shutdown",
        'frame_read_error': "Cannot read frame!",
    },
    'ko': {
        'pygame_warning': "경고: pygame이 설치되지 않았습니다. 시스템 beep 소리를 사용합니다.",
        'camera_init': "카메라 초기화 완료",
        'camera_wait': "몇 초간 대기하여 카메라가 안정화되도록 합니다...",
        'camera_error': "웹캠을 열 수 없습니다!",
        'system_start': "움직임 감지 시스템 시작",
        'controls': "종료: 'q' | 경고 켜기/끄기: 스페이스바",
        'sensitivity': "민감도",
        'lower_more_sensitive': "낮을수록 민감",
        'min_area': "최소 감지 영역",
        'pixels': "픽셀",
        'alarm_duration': "경고음 지속 시간",
        'seconds': "초",
        'auto_resume': "자동 재활성화",
        'hysteresis': "히스테리시스",
        'frames_to_trigger': "프레임 이상 감지시 경고",
        'motion_detected_alarm': "움직임 감지! 경고음 시작",
        'alarm_enabled': "경고 활성화됨",
        'alarm_disabled': "경고 비활성화됨",
        'auto_resume_after': "{0}초 후 자동 재활성화",
        'alarm_auto_resumed': "경고 자동 재활성화됨",
        'user_interrupted': "사용자가 중단했습니다.",
        'error_occurred': "오류 발생",
        'system_shutdown': "시스템 종료됨",
        'frame_read_error': "프레임을 읽을 수 없습니다!",
    }
}

def get_system_language():
    """Detect system language and return appropriate language code"""
    try:
        system_lang = locale.getdefaultlocale()[0]
        if system_lang:
            # Extract language code (e.g., 'ko_KR' -> 'ko')
            lang_code = system_lang.split('_')[0].lower()
            # Return if supported, otherwise default to 'en'
            return lang_code if lang_code in MESSAGES else 'en'
    except:
        pass
    return 'en'

# Try to import pygame
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

class MotionDetector:
    def __init__(self, sensitivity=5, min_area=100, alarm_duration=3.0, auto_resume=300, lang=None, camera_index=0):
        self.cap = None
        self.prev_frame = None
        self.motion_detected = False
        self.sensitivity = sensitivity
        self.min_area = min_area
        self.running = False
        
        self.camera_index = camera_index 
        
        # Language settings - use system language if not specified
        if lang is None:
            lang = get_system_language()
        self.lang = lang if lang in MESSAGES else 'en'
        self.msg = MESSAGES[self.lang]
        
        # Print pygame warning if not available
        if not PYGAME_AVAILABLE:
            print(self.msg['pygame_warning'])
        
        # Hysteresis settings
        self.motion_counter = 0
        self.motion_threshold_high = 3  # Threshold to start alarm (3 consecutive frames)
        self.motion_threshold_low = 1   # Threshold to stop alarm
        self.alarm_active = False
        
        # Alarm duration management
        self.alarm_duration = alarm_duration
        self.alarm_end_time = 0
        self.alarm_thread = None
        self.alarm_thread_running = False
        
        # Alarm enable/disable settings
        self.alarm_enabled = True
        self.auto_resume_time = auto_resume
        self.alarm_disabled_at = 0
        
        # Initialize pygame (for alarm sound)
        if PYGAME_AVAILABLE:
            pygame.mixer.init()
            self.create_alert_sound()
    
    def create_alert_sound(self):
        """Generate alert sound"""
        if not PYGAME_AVAILABLE:
            return
        
        # Generate simple alert tone (880Hz, 0.5 seconds)
        sample_rate = 22050
        duration = 0.5
        frequency = 880
        
        n_samples = int(sample_rate * duration)
        buf = np.sin(2 * np.pi * frequency * np.linspace(0, duration, n_samples))
        buf = (buf * 32767).astype(np.int16)
        
        # Convert to stereo
        stereo_buf = np.column_stack((buf, buf))
        sound = pygame.sndarray.make_sound(stereo_buf)
        self.alert_sound = sound
    
    def alarm_loop(self):
        """Alarm sound loop thread"""
        while self.alarm_thread_running and time.time() < self.alarm_end_time:
            if PYGAME_AVAILABLE:
                self.alert_sound.play()
            else:
                print('\a')  # System beep
            
            time.sleep(1.0)  # 1 second interval (0.5s sound + 0.5s pause)
        
        self.alarm_thread_running = False
    
    def toggle_alarm(self):
        """Toggle alarm on/off with spacebar"""
        self.alarm_enabled = not self.alarm_enabled
        
        if self.alarm_enabled:
            print(f"\n✅ [{datetime.now().strftime('%H:%M:%S')}] {self.msg['alarm_enabled']}")
            self.alarm_disabled_at = 0
        else:
            print(f"\n⏸️  [{datetime.now().strftime('%H:%M:%S')}] {self.msg['alarm_disabled']} ({self.msg['auto_resume_after'].format(self.auto_resume_time)})")
            self.alarm_disabled_at = time.time()
            # Stop currently playing alarm
            self.alarm_thread_running = False
    
    def check_auto_resume(self):
        """Check for auto-resume"""
        if not self.alarm_enabled and self.alarm_disabled_at > 0:
            elapsed = time.time() - self.alarm_disabled_at
            if elapsed >= self.auto_resume_time:
                self.alarm_enabled = True
                self.alarm_disabled_at = 0
                print(f"\n🔔 [{datetime.now().strftime('%H:%M:%S')}] {self.msg['alarm_auto_resumed']}")
    
    def trigger_alarm(self):
        """Trigger alarm (called when motion detected)"""
        # Ignore if alarm is disabled
        if not self.alarm_enabled:
            return
        
        current_time = time.time()
        
        # Extend alarm duration
        self.alarm_end_time = current_time + self.alarm_duration
        
        # If already playing, just extend time and return
        if self.alarm_thread_running:
            return
        
        # Start new alarm thread
        print(f"\n🚨 [{datetime.now().strftime('%H:%M:%S')}] {self.msg['motion_detected_alarm']} 🚨")
        self.alarm_thread_running = True
        self.alarm_thread = threading.Thread(target=self.alarm_loop, daemon=True)
        self.alarm_thread.start()
    
    def initialize_camera(self):
        """Initialize camera"""
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            raise Exception(f"{self.msg['camera_error']} (Index: {self.camera_index})")
        
        # Camera settings for low-light environments
        self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Auto-adjust exposure
        self.cap.set(cv2.CAP_PROP_GAIN, 10)      # Increase gain
        
        print(self.msg['camera_init'])
        print(self.msg['camera_wait'])
        time.sleep(2)
    
    def detect_motion(self, frame):
        """Detect motion (with hysteresis)"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Store first frame
        if self.prev_frame is None:
            self.prev_frame = gray
            return False, frame
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        
        # Apply threshold
        thresh = cv2.threshold(frame_diff, self.sensitivity, 255, cv2.THRESH_BINARY)[1]
        
        # Remove noise
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        
        # Analyze motion
        for contour in contours:
            if cv2.contourArea(contour) < self.min_area:
                continue
            
            motion_detected = True
            
            # Draw rectangle around motion area
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Hysteresis logic
        if motion_detected:
            self.motion_counter += 1
            if self.motion_counter >= self.motion_threshold_high:
                self.alarm_active = True
                self.trigger_alarm()
        else:
            self.motion_counter = max(0, self.motion_counter - 1)
            if self.motion_counter <= self.motion_threshold_low:
                self.alarm_active = False
        
        # Update previous frame
        self.prev_frame = gray
        
        return self.alarm_active, frame
    
    def run(self):
        """Main execution loop"""
        try:
            self.initialize_camera()
            self.running = True
            
            print("\n" + "="*50)
            print(self.msg['system_start'])
            print("="*50)
            print(self.msg['controls'])
            print(f"{self.msg['sensitivity']}: {self.sensitivity} ({self.msg['lower_more_sensitive']})")
            print(f"{self.msg['min_area']}: {self.min_area} {self.msg['pixels']}")
            print(f"{self.msg['alarm_duration']}: {self.alarm_duration}{self.msg['seconds']}")
            print(f"{self.msg['auto_resume']}: {self.auto_resume_time}{self.msg['seconds']}")
            print(f"{self.msg['hysteresis']}: {self.motion_threshold_high}{self.msg['frames_to_trigger']}")
            print(f"Camera Index / 카메라 인덱스: {self.camera_index}") 
            print("="*50 + "\n")
            
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print(self.msg['frame_read_error'])
                    break
                
                # Check auto-resume
                self.check_auto_resume()
                
                # Brightness enhancement (for low-light)
                frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)
                
                # Detect motion
                motion_detected, processed_frame = self.detect_motion(frame)
                
                if motion_detected:
                    # Display warning text on screen (always in English)
                    cv2.putText(processed_frame, "MOTION DETECTED!", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display status (always in English for OpenCV window)
                if not self.alarm_enabled:
                    status = "ALARM DISABLED"
                    color = (128, 128, 128)
                    # Show remaining time
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
                
                # Display time
                cv2.putText(processed_frame, datetime.now().strftime('%H:%M:%S'), 
                          (processed_frame.shape[1] - 100, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show window
                cv2.imshow('Motion Detector', processed_frame)
                
                # Handle key input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord(' '):  # Spacebar
                    self.toggle_alarm()
            
        except KeyboardInterrupt:
            print(f"\n{self.msg['user_interrupted']}")
        except Exception as e:
            print(f"\n{self.msg['error_occurred']}: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup operations"""
        self.running = False
        self.alarm_thread_running = False
        if self.alarm_thread and self.alarm_thread.is_alive():
            self.alarm_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()
        print(f"\n{self.msg['system_shutdown']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Motion Detection Alert System / 움직임 감지 경고 시스템',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples / 예제:
  python motion-guard.py                              # Default settings / 기본 설정 (Index 0 사용)
  python motion-guard.py -i 1                         # Use camera index 1 / 카메라 인덱스 1 사용
  python motion-guard.py --lang ko -i 2               # Korean language, camera index 2 / 한국어, 카메라 인덱스 2
  python motion-guard.py -s 10 -m 200                 # Sensitivity 10, min area 200
  python motion-guard.py -d 5                         # 5 second alarm duration
  python motion-guard.py -r 600                       # Auto-resume after 10 minutes
  python motion-guard.py -s 5 -m 100 -d 3 -r 300      # All options combined
  python motion-guard.py --lang ko -s 5 -m 100        # Korean with custom settings
  
Controls / 조작:
  Spacebar: Toggle alarm on/off (auto-resumes after set time)
  스페이스바: 경고 켜기/끄기 (설정 시간 후 자동으로 다시 켜짐)
  Q: Quit / 종료
        """
    )
    
    parser.add_argument('-s', '--sensitivity', type=int, default=5,
                        help='Sensitivity threshold (lower = more sensitive, default: 5) / 민감도 (낮을수록 민감, 기본값: 5)')
    parser.add_argument('-m', '--min-area', type=int, default=100,
                        help='Minimum motion area in pixels (default: 100) / 최소 감지 영역 픽셀 (기본값: 100)')
    parser.add_argument('-d', '--duration', type=float, default=3.0,
                        help='Alarm duration in seconds (default: 3.0) / 경고음 지속 시간 초 (기본값: 3.0)')
    parser.add_argument('-r', '--auto-resume', type=int, default=300,
                        help='Auto re-enable time in seconds (default: 300 = 5 min) / 자동 재활성화 시간 초 (기본값: 300 = 5분)')
    parser.add_argument('-i', '--camera-index', type=int, default=0,
                        help='Camera device index (0, 1, 2, ... default: 0) / 카메라 장치 인덱스 (0, 1, 2, ... 기본값: 0)')
    parser.add_argument('--lang', type=str, default=None,
                        help='Language (default: system language, fallback to en) / 언어 (기본값: 시스템 언어, 없으면 en)')
    
    args = parser.parse_args()
    
    detector = MotionDetector(
        sensitivity=args.sensitivity,
        min_area=args.min_area,
        alarm_duration=args.duration,
        auto_resume=args.auto_resume,
        lang=args.lang,
        camera_index=args.camera_index # 인스턴스 생성 시 전달
    )
    detector.run()