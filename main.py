import cv2
import pygame
import time
import math
import pyttsx3
import threading
from ultralytics import YOLO

# ─────────────────────────── CONFIG ────────────────────────────────────────── #
CONFIG = {
    "base_model": "yolo11n.pt",       
    "custom_model": "models/best.pt", 
    "beep_path": "assets/beep.wav",
    "heartbeat_path": "assets/heartbeat.wav",
}

class SystemState:
    def __init__(self):
        self.mode = "SCANNING"
        self.handoff_triggered = False
        self.startup_frames = 0
        self.detected_street_hazards = False

state = SystemState()

# ─────────────────────────── SPEECH MODULE ────────────────────────────────── #
def speak(text):
    def run():
        engine = pyttsx3.init()
        engine.setProperty('rate', 165)
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()

# ─────────────────────────── MAIN ENGINE ──────────────────────────────────── #
def main():
    base_brain = YOLO(CONFIG["base_model"])   
    custom_brain = YOLO(CONFIG["custom_model"]) 
    
    pygame.mixer.init()
    beep = pygame.mixer.Sound(CONFIG["beep_path"])
    heartbeat = pygame.mixer.Sound(CONFIG["heartbeat_path"])
    
    cap = cv2.VideoCapture(0)
    last_beep, last_hb, prev_time = 0, 0, time.time()

    print("[VisionDock] Initializing Environment Scan...")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok: break
        now = time.time()
        h, w = frame.shape[:2]

        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'): # Manual Override Button
            state.mode = "HUD" if state.mode == "STREET" else "STREET"
            speak(state.mode)

        # --- 1. STARTUP AUTO-DETECTION (THE FIRST 30 FRAMES) ---
        if state.mode == "SCANNING":
            results = base_brain(frame, verbose=False, conf=0.4)
            for r in results:
                labels = [base_brain.names[int(c)] for c in r.boxes.cls]
                if any(x in labels for x in ["car", "motorcycle", "bus", "truck"]):
                    state.detected_street_hazards = True
            
            state.startup_frames += 1
            if state.startup_frames > 30: # After ~1 second of scanning
                if state.detected_street_hazards:
                    state.mode = "STREET"
                    speak("STREET")
                else:
                    state.mode = "HUD"
                    speak("HUD")
            continue # Wait for scan to finish

        # --- 2. VISION PROCESSING ---
        # Street Mode: High speed/Hazard detection | HUD Mode: Indoor/Goal detection
        active_model = base_brain if state.mode == "STREET" else custom_brain
        results = active_model.track(frame, persist=True, stream=True, verbose=False, conf=0.4)
        
        best_score, best_pan = 0, 0.5
        annotated = frame.copy()

        for r in results:
            annotated = r.plot()
            if not r.boxes: continue
            for i, box in enumerate(r.boxes):
                xc, _, bw, bh = box.xywh[0]
                nx = float(xc)/w
                area = float(bw*bh)/(h*w) # Use area as a proxy for distance
                
                # Logic: Higher area = Closer object = Higher Score
                score = min(100, area * 500) 
                if score > best_score:
                    best_score, best_pan = score, nx

        # --- 3. DYNAMIC AUDIO ENGINE (YOUR WORKFLOW) ---
        if best_score > 15: # Obstacle Detected
            # Freq Logic: Far (Low Area) = Slow Beep | Near (High Area) = Fast Beep
            # Interval scale: 0.6s (Far) down to 0.05s (Near)
            interval = max(0.05, 0.6 - (best_score / 150))
            
            if now - last_beep >= interval:
                # Volume increases as it gets nearer
                vol = min(1.0, 0.3 + (best_score / 100))
                # Spatial Panning: (1-nx) is Left volume, nx is Right volume
                beep.play().set_volume(vol * (1 - best_pan), vol * best_pan)
                last_beep = now
        else:
            # EVERYTHING CLEAR: Play Heartbeat
            if now - last_hb >= 1.5:
                heartbeat.play().set_volume(0.1, 0.1)
                last_hb = now

        # --- 4. HUD UI ---
        cv2.rectangle(annotated, (0,0), (w, 60), (20,20,20), -1)
        status_color = (0,255,0) if state.mode == "STREET" else (255,150,0)
        cv2.putText(annotated, f"SYSTEM: {state.mode} ACTIVE", (20, 40), 1, 1.5, status_color, 2)
        
        cv2.imshow("VisionDock - Team DHARMA", annotated)
        if key in (ord('q'), 27): break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()