import cv2
from deepface import DeepFace
import time
from collections import Counter
import threading

data_lock = threading.Lock()
latest_frame = None
latest_analysis = {}
last_known_emotions = []
stop_thread = False

def analyze_face_emotions():
    global latest_frame, latest_analysis, last_known_emotions, data_lock, stop_thread
    
    print("Background analysis thread started.")
    
    while not stop_thread:
        with data_lock:
            if latest_frame is None:
                time.sleep(0.1) 
                continue
            frame_to_analyze = latest_frame.copy()

        try:
            analysis_list = DeepFace.analyze(
                frame_to_analyze, 
                actions=['emotion'], 
                enforce_detection=False, 
                detector_backend='opencv',
                silent=True
            )
            
            if analysis_list and isinstance(analysis_list, list) and analysis_list[0]:
                analysis = analysis_list[0]
                
                with data_lock:
                    latest_analysis = analysis 
                    last_known_emotions.append(analysis['dominant_emotion'])

        except Exception as e:
            with data_lock:
                latest_analysis = {}
        
        time.sleep(0.5) 
    
    print("Background analysis thread stopped.")


def get_mood_from_webcam():
    global latest_frame, latest_analysis, last_known_emotions, data_lock, stop_thread
    
    latest_frame = None
    latest_analysis = {}
    last_known_emotions = []
    stop_thread = False
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    window_name = 'Mood Analyzer - Look at the camera!'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    print("Analyzing your mood... The camera window will pop up.")
    print("Capturing mood for 30 seconds... Press 'q' to stop early.")

    analysis_thread = threading.Thread(target=analyze_face_emotions, daemon=True)
    analysis_thread.start()

    start_time = time.time()
    
    while (time.time() - start_time) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)

        with data_lock:
            latest_frame = frame.copy()
            current_analysis = latest_analysis.copy()

        if current_analysis:
            try:
                facial_area = current_analysis['region']
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                emotions = current_analysis.get('emotion', {})
                dominant_emotion = current_analysis.get('dominant_emotion', 'N/A')
                
                y_text_offset = y + h + 30
                cv2.putText(frame, "---EMOTIONS---", (x, y_text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                y_text_offset += 25

                for emotion, score in emotions.items():
                    text = f"{emotion.capitalize()}: {score:.1f}%"
                    color = (0, 255, 0) if emotion == dominant_emotion else (255, 255, 255)
                    cv2.putText(frame, text, (x, y_text_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    y_text_offset += 20
            except Exception:
                pass 
        
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User quit early.")
            break

    print("Stopping analysis thread...")
    stop_thread = True 
    cap.release()
    cv2.destroyAllWindows()
    analysis_thread.join(timeout=2) 
    print("Mood analysis complete.")

    if not last_known_emotions:
        print("Could not detect any mood. Defaulting to neutral.")
        return 'neutral'
    
    emotional_moods = [mood for mood in last_known_emotions if mood != 'neutral']
    
    if emotional_moods:
        mood_counts = Counter(emotional_moods)
        final_mood = mood_counts.most_common(1)[0][0]
    else:
        mood_counts = Counter(last_known_emotions)
        final_mood = mood_counts.most_common(1)[0][0]
        
    print(f"Final determined mood: {final_mood.capitalize()}")
    return final_mood

if __name__ == "__main__":
    get_mood_from_webcam()