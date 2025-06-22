# Application_data_summary
# RAG-Based Interactive Dashboard with Chatbot

This project integrates a **Retrieval-Augmented Generation (RAG)** chatbot with a **Streamlit-powered interactive dashboard** to help visualize and query data related to On-the-Job Training (OJT) programs across institutions and industries. It uses FAISS for fast similarity search over embedded text data and OpenAI's GPT model for natural language understanding.

---

## 🚀 Features

- 🔍 **Question Answering Chatbot** (test version): Ask questions like _"How many companies?"_ or _"What jobs are available?"_
- 📈 **Five Tableau Dashboards** embedded for visual insights:
  - Regional Job Distribution
  - Industry-wise Company Breakdown
  - Monthly Application Trends
  - Institutional & Student Placement Overview
  - Geographic Distribution of Students
- 🧠 **Embedding-based Retrieval** using FAISS + `text-embedding-3-small` from OpenAI
- 🌐 **Live Query Interface** with instant response using GPT-4

---

## 🗂️ Dataset Used

The system loads data from three CSV files:
- `company_profile.csv` – Company descriptions and job postings
- `Job_Postings.csv` – Job titles and statuses
- `Student_Applications.csv` – Student-school-company mappings

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/rag-chatbot-dashboard.git
cd rag-chatbot-dashboard
pip install -r requirements.txt
