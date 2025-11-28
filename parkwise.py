# parkwise.py

import os
import random
import logging
from datetime import datetime

import numpy as np
import cv2
import pandas as pd
import imageio.v2 as imageio
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_from_directory, render_template

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Optional Gemini setup
smart_parking_ai = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        smart_parking_ai = genai.GenerativeModel("models/gemini-2.5-flash")
        logging.info("Gemini configured.")
    except Exception as e:
        logging.warning("Gemini setup failed: %s", e)
        smart_parking_ai = None

# Global memory for last simulation
LATEST_PARKING_DATA = {}

# -------------------------
# Utility functions
# -------------------------
def create_parking_layout(cols, rows, slot_w=90, slot_h=180, padding=20):
    slots = []
    for r in range(rows):
        for c in range(cols):
            x1 = c * (slot_w + padding) + padding
            y1 = r * (slot_h + padding) + padding
            x2 = x1 + slot_w
            y2 = y1 + slot_h
            slots.append((x1, y1, x2, y2))
    return slots

def draw_parking_image(slots, occupancy, img_w=1200, img_h=1200, bg_color=(255,255,255)):
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * np.array(bg_color, dtype=np.uint8)
    for (x1,y1,x2,y2), occ in zip(slots, occupancy):
        border_color = (0,255,0) if occ == 0 else (0,0,255)
        cv2.rectangle(img, (x1, y1), (x2, y2), border_color, 3)
        if occ == 1:
            cv2.rectangle(img, (x1+10, y1+10), (x2-10, y2-10), (0,100,200), -1)
    return img

def detect_occupied_slots(slots, img, red_threshold=200, min_red_pixels=300):
    detected = []
    if img is None:
        return [0 for _ in slots]

    h, w, _ = img.shape
    for (x1,y1,x2,y2) in slots:
        x1c, y1c = max(0, int(x1)), max(0, int(y1))
        x2c, y2c = min(w, int(x2)), min(h, int(y2))

        if x2c <= x1c or y2c <= y1c:
            detected.append(0)
            continue

        slot_region = img[y1c:y2c, x1c:x2c]

        red_mask = (
            (slot_region[:,:,2] > red_threshold) &
            (slot_region[:,:,2] > slot_region[:,:,1]) &
            (slot_region[:,:,2] > slot_region[:,:,0])
        )
        detected.append(1 if int(red_mask.sum()) > min_red_pixels else 0)

    return detected

# -------------------------
# Base Agent & Agents
# -------------------------
class AgentBase:
    def __init__(self, name):
        self.name = name
    def setup(self, **kwargs): pass
    def run(self, *args, **kwargs): pass

class SimulationAgent(AgentBase):
    def __init__(self, name="SimulationAgent"):
        super().__init__(name)

    def setup(self, cols=8, rows=4, fill_prob=0.6, n_frames=10, out_folder="simulation_output"):
        self.cols = cols
        self.rows = rows
        self.fill_prob = fill_prob
        self.n_frames = n_frames
        self.out_folder = out_folder
        os.makedirs(out_folder, exist_ok=True)
        self.slots = create_parking_layout(cols, rows)

    def run(self):
        frames_meta = []

        for t in range(self.n_frames):
            occupancy = [1 if random.random() < self.fill_prob else 0 for _ in self.slots]
            img = draw_parking_image(self.slots, occupancy)

            path = os.path.join(self.out_folder, f"frame_{t:03d}.png")
            cv2.imwrite(path, img)

            frames_meta.append((path, occupancy))

        gif_path = os.path.join(self.out_folder, "simulation.gif")
        try:
            imageio.mimsave(gif_path, [imageio.imread(p) for p,_ in frames_meta], fps=2)
        except:
            pass

        return {"frames": frames_meta, "gif": gif_path, "slots": self.slots}

class VisionAgent(AgentBase):
    def __init__(self, name="VisionAgent"):
        super().__init__(name)

    def setup(self, slots):
        self.slots = slots

    def run(self, frame_path):
        img = cv2.imread(frame_path)
        detections = detect_occupied_slots(self.slots, img)
        return {"frame": frame_path, "detections": detections}

class ReportingAgent(AgentBase):
    def __init__(self, name="ReportingAgent"):
        super().__init__(name)

    def setup(self, slots, out_folder="simulation_output", annotated_folder="annotated_frames"):
        self.slots = slots
        self.out_folder = out_folder
        self.annotated_folder = annotated_folder
        os.makedirs(annotated_folder, exist_ok=True)

    def annotate(self, frame_path, detections):
        img = cv2.imread(frame_path)
        if img is None:
            return frame_path

        # Draw boxes
        for i, (x1, y1, x2, y2) in enumerate(self.slots):
            color = (0,255,0) if detections[i] == 0 else (0,0,255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

        outpath = os.path.join(self.annotated_folder, os.path.basename(frame_path))
        cv2.imwrite(outpath, img)
        return outpath

    def run(self, results):
        rows = []
        annotated = []

        for r in results:
            frame_path = r["frame"]
            detections = r["detections"]

            rows.append({
                "frame": os.path.basename(frame_path),
                "occupied": int(sum(detections)),
                "detections": detections
            })

            # FIX: annotate the frame and append to gif list
            annotated_frame = self.annotate(frame_path, detections)
            annotated.append(annotated_frame)

        # Save CSV
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.out_folder, "detections.csv")
        df.to_csv(csv_path, index=False)

        # FIX: generate animated GIF correctly
        gif_path = os.path.join(self.out_folder, "annotated.gif")
        try:
            imgs = [imageio.imread(p) for p in annotated]
            imageio.mimsave(gif_path, imgs, fps=2)
        except Exception as e:
            print("GIF generation error:", e)

        return {"csv": csv_path, "gif": gif_path, "df": df}


# -------------------------
# Full Pipeline
# -------------------------
def run_full_pipeline(params):
    sim = SimulationAgent(); sim.setup(**params)
    s = sim.run()

    vis = VisionAgent(); vis.setup(s["slots"])
    results = [vis.run(fp) for fp,_ in s["frames"]]

    rep = ReportingAgent(); rep.setup(slots=s["slots"])
    report = rep.run(results)

    return report

# Default params
params = {"cols": 8, "rows": 4, "fill_prob": 0.6, "n_frames": 10, "out_folder": "simulation_output"}

# -------------------------
# Local summary helper
# -------------------------
def summarize_frames(frames):
    vals = [int(f["occupied"]) for f in frames]
    total_frames = len(vals)

    return {
        "avg": sum(vals)/total_frames,
        "max": max(vals),
        "min": min(vals),
        "max_frame": frames[vals.index(max(vals))]["frame"],
        "min_frame": frames[vals.index(min(vals))]["frame"]
    }

# -------------------------
# AI handler
# -------------------------
def ask_ai(question):

    if not LATEST_PARKING_DATA:
        return "Run a simulation first."

    total_slots = LATEST_PARKING_DATA["total_slots"]
    frames = LATEST_PARKING_DATA["frames"]
    summary = summarize_frames(frames)

    avg_occupied = round(summary["avg"])
    avg_empty = total_slots - avg_occupied

    # Local fallback responder
    def local(q):
        q = q.lower()
        if "empty" in q:
            return f"There are on average {avg_empty} empty spaces."
        if "occupied" in q:
            return f"Average occupied: {avg_occupied}/{total_slots}."
        if "most" in q:
            return f"Peak occupancy: {summary['max']} in {summary['max_frame']}."
        if "least" in q:
            return f"Least occupancy: {summary['min']} in {summary['min_frame']}."
        return f"Summary: avg {avg_occupied} occupied, {avg_empty} empty."

    if smart_parking_ai is None:
        return local(question)

    prompt = f"""
    You are ParkWise, a smart parking analysis AI.

    Simulation summary:
    - Total slots: {total_slots}
    - Avg occupied: {avg_occupied}
    - Avg empty: {avg_empty}
    - Most in frame: {summary['max_frame']}
    - Least in frame: {summary['min_frame']}
    - Frame data: {frames}

    User question: {question}

    Answer clearly based on the simulation.
    """

    try:
        reply = smart_parking_ai.generate_content(prompt).text
        return reply
    except:
        return local(question)

# -------------------------
# Flask Setup
# -------------------------
app = Flask(__name__, template_folder="frontend")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run-simulation")
def run_sim():
    global LATEST_PARKING_DATA

    try:
        report = run_full_pipeline(params)

        df = report["df"]
        frames = [
    {
        "frame": row["frame"],
        "occupied": int(row["occupied"]),
        "detections": row["detections"]  # NEW
    }
    for _, row in df.iterrows()
]


        LATEST_PARKING_DATA = {
            "total_slots": params["cols"] * params["rows"],
            "frames": frames,
            "generated": datetime.utcnow().isoformat()+"Z"
        }

        return jsonify({
            "status":"success",
            "gif_url":"/simulation_output/annotated.gif",
            "csv_url":"/simulation_output/detections.csv",
            "occupancy_summary":frames
        })

    except Exception as e:
        logging.exception("Simulation failed:")
        return jsonify({"status":"error", "message":str(e)})

@app.route("/simulation_output/<path:filename>")
def files(filename):
    directory = os.path.join(os.getcwd(), "simulation_output")
    return send_from_directory(directory, filename)


@app.route("/parking-ai", methods=["POST"])
def parking_ai():
    question = request.json.get("question","")
    answer = ask_ai(question)
    return jsonify({"response":answer})

if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    app.run(host="0.0.0.0", port=PORT, debug=False)
