# 🚗 ParkWise – AI-Powered Smart Parking Analytics & Simulation System  
### *Kaggle Agents Intensive · Capstone Project 2025*  
**Track:** Agents for Good (Sustainability)

---

## 📘 Overview

ParkWise is an **AI-powered multi-agent system** that simulates, analyzes, and optimizes parking space usage.  
It demonstrates how intelligent agents can reduce congestion, fuel waste, and emissions by enabling smarter parking management.

The system includes:

- **Simulation Agent** – generates parking frames  
- **Vision Agent** – detects occupied vs. free slots  
- **Reporting Agent** – produces CSV logs, insights, and GIFs  
- **Gemini LLM Agent** – answers parking-related queries  
- **Interactive Dashboard** – heatmap, analytics, report history, AI chat  

ParkWise showcases how agents can support sustainability and smart-city planning.

---

## 🧠 Problem Statement

In urban areas, drivers often waste:

- **20–30% of travel time searching for parking**  
- Fuel due to idling and circling  
- Time spent navigating full parking lots  

This creates:

- Higher emissions  
- Traffic congestion  
- Poor land utilization  

Traditional parking systems lack real-time analytics, predictions, and automation tools.

---

## 💡 Solution Summary

ParkWise provides an automated, agent-driven solution:

### ✔ Multi-Agent Pipeline  
- SimulationAgent → creates synthetic frames  
- VisionAgent → detects slot occupancy  
- ReportingAgent → builds analytics, data logs, visuals  
- Gemini LLM → answers any natural-language parking queries  
- MemoryBank → persists internal state  

### ✔ Dashboard  
- Run a full simulation  
- View animated occupancy GIF  
- Heatmap with per-slot detections  
- AI insights based on simulations  
- Download CSV, PDF, GIF  
- Report table with timestamps  
- Clear-history functionality  
- Integrated AI chat modal  

---

## 🧩 Features

### 🟦 Multi-Agent Intelligence
- Parallel & sequential agent pipeline  
- LLM-powered reasoning  
- Memory-driven insight generation  

### 🟩 Dashboard Tools
- Parking heatmap  
- Simulation GIF preview  
- AI insights  
- Downloadable reports  
- Report history table  
- Built-in chat bubble & modal  

### 🟨 Utilities
- PDF generation  
- CSV data export  
- GIF export  
- Logging & status tracking  

---

## 🔧 Installation

Clone the repository:

```sh
git clone https://github.com/yourusername/ParkWise.git
cd ParkWise
Create a virtual environment:


python -m venv venv
venv\Scripts\activate  # Windows
Install dependencies:


pip install -r requirements.txt
🔐 Environment Variables
Create a .env file:

GEMINI_API_KEY=YOUR_GEMINI_KEY

⚠️ Do NOT commit .env to GitHub.

Add this to .gitignore:

.env

🚀 Running the Application
Start the server:

python parkwise.py

Open in browser:

http://127.0.0.1:8000


📁 Folder Structure

ParkWise/
│-- parkwise.py
│-- requirements.txt
│-- .env.example
│-- .gitignore
│-- /frontend
│     └── index.html
│-- /simulation_output   # generated after running
│-- README.md


🛠️ Technology Stack
Python, Flask

JavaScript, Bootstrap

OpenCV, ImageIO, Pandas

Gemini API (LLM)

html2pdf.js

📜 License
MIT License 

✨ Author
Dolla.Chaitra
Kaggle Agents Intensive — Capstone Project 2025