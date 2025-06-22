
from dotenv import load_dotenv
from openai import OpenAI
import faiss
import json
import numpy as np
import os
import pandas as pd
import streamlit as st
import urllib.parse

load_dotenv()  # 加载.env文件中的环境变量
openai_api_key = os.getenv("OPENAI_API_KEY")

base_path = r"./"

company_profiles = pd.read_csv(os.path.join(base_path, "company_profile.csv"))
job_posts = pd.read_csv(os.path.join(base_path, "Job_Postings.csv"))
Applications = pd.read_csv(os.path.join(base_path, "Student_Applications.csv"))

client = OpenAI()


def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


def load_data():
    if os.path.exists("texts.json"):
        texts = json.load(open("texts.json"))
    else:
        texts = []
        for _, row in company_profiles.iterrows():
            text = f"company_profiles: Company {row.get('Company', '')}, description {row.get('description', '')}, Industry {row.get('Industry', '')}, Location(country) {row.get('Location(country)', '')}, Location(State) {row.get('Location(State)', '')}, Job_post {row.get('Job_post', '')}"
            texts.append(text)

        for _, row in Applications.iterrows():
            text = f"Applications: Student {row.get('Student', '')},Status {row.get('Status','')}, school {row.get('Institution_name', '')}"
            texts.append(text)

        # for _, row in job_posts.iterrows():
        #     text = f"job_title: {row.get('job_title', '')}, Company: {row.get('Company', '')}, Job_Status: {row.get('Job_Status', '')}"
        #     texts.append(text)
        for _, row in job_posts.iterrows():
            text = f"Job_post: job_title {row.get('job_title', '')}, Company {row.get('Company', '')}, Job_Status {row.get('Job_Status', '')}"
            texts.append(text)

    if not os.path.exists("faiss_all_tables.index"):
        embeddings = [get_embedding(t) for t in texts]
        dim = len(embeddings[0])

        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embeddings).astype("float32"))

        faiss.write_index(index, "faiss_all_tables.index")
        with open("texts.json", "w") as f:
            json.dump(texts, f, indent=2)

    return texts


def answer_query(query, top_k=150):
    query_lower = query.strip().lower()

    # 关键词组
    SCHOOL_KEYWORDS = ["school", "institution", "college", "university", "academy","schools","institutions","colleges","universitys",'universities']
    COMPANY_KEYWORDS = ["company", "organization", "firm", "business", "employer","companies",'companys',"firms","organizations"]
    STUDENT_KEYWORDS = ["student", "applicant", "candidate", "learner", "individual","students","applicants","candidates","learners","individuals"]
    JOB_KEYWORDS = ["job", "position", "role", "title", "vacancy", "opening","jobs","positions","roles","titles"]

    def contains_any(keyword_list):
        return any(kw in query_lower for kw in keyword_list)

    # 只处理包含 "how many" 的问题
    if "how many" in query_lower:
        if contains_any(SCHOOL_KEYWORDS):
            unique_schools = sorted(set(Applications['Institution_name'].dropna()))
            return f"There are {len(unique_schools)} unique schools in the data."

        if contains_any(COMPANY_KEYWORDS):
            unique_companies = sorted(set(company_profiles['Company'].dropna()))
            return f"There are {len(unique_companies)} unique companies in the data."

        if contains_any(STUDENT_KEYWORDS):
            unique_students = sorted(set(Applications['Student'].dropna()))
            return f"There are {len(unique_students)} unique students in the dataset."

        if contains_any(JOB_KEYWORDS):
            unique_jobs = sorted(set(job_posts['job_title'].dropna()))
            return f"There are {len(unique_jobs)} distinct job titles in the dataset."

        return "Could you clarify what you're trying to count?"


    texts = load_data()

    q_vec = np.array(get_embedding(query), dtype=np.float32).reshape(1, -1)
    index = faiss.read_index("faiss_all_tables.index")
    D, I = index.search(q_vec, top_k)

    if len(I[0]) == 0:
        return "Sorry, I couldn't find any relevant information."

    context = [texts[idx] for idx in I[0] if idx < len(texts)]
    prompt = f"Answer the question based on the following:\n{context}\n\nQuestion: {query}"

    chat = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for HR and student-company job matching."},
            {"role": "user", "content": prompt}
        ]
    )
    return chat.choices[0].message.content



import streamlit as st
import streamlit.components.v1 as components

st.title("📊 OJT Connect Summary")
# Dashboard 1
st.subheader("📍 Dashboard 1")
st.markdown("""
<style>
  ul { margin-left: 20px; padding-left: 15px; }
  li { margin-bottom: 6px; }
</style>

<div style='font-size:16px; line-height:1.6'>
  <b>Summary</b><br>
  <ul>
    <li>This visual summarizes the OJT (On-the-Job Training) connection across Canada and the United States.</li>
    <li>Canada hosts 10 companies offering 15 job postings.</li>
    <li>The U.S. leads with 26 companies and 39 jobs.</li>
    <li>Regional highlights:
      <ul>
        <li><b>In Canada:</b>
          <ul>
            <li>Saskatchewan (SK) has the highest job count (5).</li>
          </ul>
        </li>
        <li><b>In the U.S.:</b>
          <ul>
            <li>New York (NY) and California (CA) each have 4 companies, with 8 and 6 job postings respectively.</li>
          </ul>
        </li>
      </ul>
    </li>
  </ul>
</div>
""", unsafe_allow_html=True)
components.html(
    """
    <div class='tableauPlaceholder' id='viz1' style='position: relative;'>
      <noscript><a href='#'><img alt='Summary of OJT Connection' src='https://public.tableau.com/static/images/oj/ojt_connect_123/Dashboard1/1_rss.png' style='border: none' /></a></noscript>
      <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='ojt_connect_123/Dashboard1' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/oj/ojt_connect_123/Dashboard1/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='en-US' />
      </object>
    </div>
    <script type='text/javascript'>
      var divElement = document.getElementById('viz1');
      var vizElement = divElement.getElementsByTagName('object')[0];
      vizElement.style.minWidth='1100px';
      vizElement.style.minWidth='100%';
      vizElement.style.height='1000px';
      var scriptElement = document.createElement('script');
      scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
      vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """,
    height=1050,
)


st.subheader("📍 Dashboard 2")
st.markdown("""
<style>
  ul { margin-left: 20px; padding-left: 15px; }
  li { margin-bottom: 6px; }
</style>

<div style='font-size:16px; line-height:1.6'>
  <b>Summary</b><br>
  <ul>
    <li>This dashboard highlights the distribution of companies and jobs across different industries within the OJT program.</li>
    <li>Among 36 companies:
      <ul>
        <li>IT Software industry leads with 10 companies.</li>
        <li>Followed by Construction (6), Engineering (5), Insurance (3), and Law (3).</li>
      </ul>
    </li>
    <li>Job distribution insights:
      <ul>
        <li>Software Developer is the most in-demand role (15 positions).</li>
        <li>Other key roles include Data Analyst (7), Legal Associate and Project Manager (6 each), and Engineer (5).</li>
      </ul>
    </li>
    <li>The visualization emphasizes the dominance of tech-related opportunities and suggests a market trend favoring software and data-oriented roles.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

components.html(
    """
    <div class='tableauPlaceholder' id='viz1749954646080' style='position: relative;'>
      <noscript><a href='#'><img alt='Dashboard 2' src='https://public.tableau.com/static/images/oj/ojt_connect_dashboard_2/Dashboard2/1_rss.png' style='border: none' /></a></noscript>
      <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='ojt_connect_dashboard_2/Dashboard2' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/oj/ojt_connect_dashboard_2/Dashboard2/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='en-US' />
      </object>
    </div>
    <script type='text/javascript'>
      var divElement = document.getElementById('viz1749954646080');
      var vizElement = divElement.getElementsByTagName('object')[0];
      vizElement.style.minWidth='1100px';
      vizElement.style.width='100%';
      vizElement.style.height='1200px';
      var scriptElement = document.createElement('script');
      scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
      vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """,
    height=1100,
)
st.subheader("📍 Dashboard 3")
st.markdown("""
<style>
  ul { margin-left: 20px; padding-left: 15px; }
  li { margin-bottom: 6px; }
</style>

<div style='font-size:16px; line-height:1.6'>
  <b>Summary</b><br>
  <ul>
    <li>This dashboard presents application trends over time and across industries for 2025.</li>
    <li>Application growth by month:
      <ul>
        <li>March: 7 applications</li>
        <li>April: 20 applications</li>
        <li>May: peak with 50 applications</li>
        <li>June: 22 applications in the first 10 days — likely to surpass April</li>
      </ul>
    </li>
    <li>Industry-wise application distribution:
      <ul>
        <li>IT Software leads with 33 applications</li>
        <li>Construction, Insurance, and Law each have 14</li>
        <li>Engineering follows with 10</li>
      </ul>
    </li>
    <li>These trends highlight sustained interest in tech and professional sectors among OJT applicants.</li>
  </ul>
</div>
""", unsafe_allow_html=True)
components.html(
    """
    <div class='tableauPlaceholder' id='viz1749955061230' style='position: relative;'>
      <noscript>
        <a href='#'>
          <img alt='Dashboard 3' src='https://public.tableau.com/static/images/oj/ojt_connect_dashboard_3/Dashboard3/1_rss.png' style='border: none' />
        </a>
      </noscript>
      <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='ojt_connect_dashboard_3/Dashboard3' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/oj/ojt_connect_dashboard_3/Dashboard3/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='en-US' />
        <param name='filter' value='publish=yes' />
      </object>
    </div>
    <script type='text/javascript'>
      var divElement = document.getElementById('viz1749955061230');
      var vizElement = divElement.getElementsByTagName('object')[0];
      vizElement.style.minWidth='1100px';
      vizElement.style.width='100%';
      vizElement.style.height='1200px';
      var scriptElement = document.createElement('script');
      scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
      vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """,
    height=1100,
)

st.subheader("📍 Dashboard 4")
st.markdown("""
<style>
  ul { margin-left: 20px; padding-left: 15px; }
  li { margin-bottom: 6px; }
</style>

<div style='font-size:16px; line-height:1.6'>
  <b>Summary</b><br>
  <ul>
    <li><b>Institutional Distribution by State:</b>
      <ul>
        <li>Pangasinan remains the dominant region with 6 institutions.</li>
        <li>La Union follows with 2, while Ilocos Norte and Batangas have 1 each.</li>
      </ul>
    </li>
    <li><b>Student Distribution by Province:</b>
      <ul>
        <li>Pangasinan contributes the most students overall: 16 to Canada and 43 to the US.</li>
        <li>Other provinces have significantly fewer students, with La Union contributing 1 to Canada and 13 to the US.</li>
      </ul>
    </li>
    <li><b>Program Participation by Industry:</b>
      <ul>
        <li>IT Software leads again with 33 students.</li>
        <li>Insurance and Law each have 14 students, followed by Construction (14), Engineering (10), and Health Care (6).</li>
      </ul>
    </li>
    <li><b>Top Institutions by Total Student Count:</b>
      <ul>
        <li>CICOSAT Colleges (13), Binalatongan Community College (13), and Luna Colleges (10) lead the list.</li>
        <li>Asbury College and Golden West Colleges follow closely with 14 and 10 students, respectively.</li>
      </ul>
    </li>
    <li><b>International Distribution (US vs. Canada):</b>
      <ul>
        <li>Most students are placed in the US, especially from institutions like Luna Colleges, Asbury College, and Malasiqui Agno Valley College.</li>
        <li>Binalatongan Community College and Polaris College show relatively balanced placement between Canada and the US.</li>
      </ul>
    </li>
  </ul>
</div>
""", unsafe_allow_html=True)
components.html(
    """
    <div class='tableauPlaceholder' id='viz1749957750342' style='position: relative;'>
      <noscript>
        <a href='#'>
          <img alt='Dashboard 4' src='https://public.tableau.com/static/images/oj/ojt_connect_dashboard_4/Dashboard4/1_rss.png' style='border: none' />
        </a>
      </noscript>
      <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='site_root' value='' />
        <param name='name' value='ojt_connect_dashboard_4/Dashboard4' />
        <param name='tabs' value='no' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/oj/ojt_connect_dashboard_4/Dashboard4/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='en-US' />
        <param name='filter' value='publish=yes' />
      </object>
    </div>
    <script type='text/javascript'>
      var divElement = document.getElementById('viz1749957750342');
      var vizElement = divElement.getElementsByTagName('object')[0];
      vizElement.style.minWidth='1000px';
      vizElement.style.width='100%';
      vizElement.style.height='1200px';
      var scriptElement = document.createElement('script');
      scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
      vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """,
    height=1100,
)
st.subheader("📍 Dashboard 5")
st.markdown("""
<style>
  ul { margin-left: 20px; padding-left: 15px; }
  li { margin-bottom: 6px; }
</style>

<div style='font-size:16px; line-height:1.6'>
  <b>Summary</b><br>
  <ul>
    <li><b>Student Distribution by State:</b>
      <ul>
        <li><b>Pangasinan</b> dominates with <b>53 students</b>, accounting for the majority of the dataset.</li>
        <li><b>La Union</b> follows with <b>14 students</b>.</li>
      </ul>
    </li>
    <li><b>Student Distribution by Institution:</b>
      <ul>
        <li><b>CICOSAT Colleges</b> ranks first with <b>12 students</b>.</li>
        <li><b>Luna Colleges</b> and <b>Binalatongan Community College</b> each contribute <b>10 students</b>.</li>
      </ul>
    </li>
    <li><b>Geographic Visualization Insights:</b>
      <ul>
        <li>The map clearly visualizes student density, with Pangasinan shaded darkest due to its high contribution.</li>
        <li>Key institution locations (e.g., CICOSAT, Luna Colleges) are marked with proportional circle sizes.</li>
      </ul>
    </li>
  </ul>
</div>
""", unsafe_allow_html=True)
components.html(
    """
    <div class='tableauPlaceholder' id='viz1750123860621' style='position: relative;'>
      <noscript>
        <a href='#'>
          <img alt='Dashboard 5' src='https://public.tableau.com/static/images/DS/DS989YQC9/1_rss.png' style='border: none' />
        </a>
      </noscript>
      <object class='tableauViz' style='display:none;'>
        <param name='host_url' value='https%3A%2F%2Fpublic.tableau.com%2F' />
        <param name='embed_code_version' value='3' />
        <param name='path' value='shared/DS989YQC9' />
        <param name='toolbar' value='yes' />
        <param name='static_image' value='https://public.tableau.com/static/images/DS/DS989YQC9/1.png' />
        <param name='animate_transition' value='yes' />
        <param name='display_static_image' value='yes' />
        <param name='display_spinner' value='yes' />
        <param name='display_overlay' value='yes' />
        <param name='display_count' value='yes' />
        <param name='language' value='en-US' />
        <param name='filter' value='publish=yes' />
      </object>
    </div>
    <script type='text/javascript'>
      var divElement = document.getElementById('viz1750123860621');
      var vizElement = divElement.getElementsByTagName('object')[0];
      vizElement.style.minWidth='1000px';
      vizElement.style.width='100%';
      vizElement.style.height='1200px';
      var scriptElement = document.createElement('script');
      scriptElement.src = 'https://public.tableau.com/javascripts/api/viz_v1.js';
      vizElement.parentNode.insertBefore(scriptElement, vizElement);
    </script>
    """,
    height=1100,
)

st.markdown("---")

params = st.query_params
query = urllib.parse.unquote(params["user_query"]) if "user_query" in params else None
answer = answer_query(query) if query else ""

# 浮动聊天窗口 + 回答展示
answer_html = answer.replace("\n", "<br>") if answer else ""

st.markdown(f"""
    <style>
    .chatbot-box {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        width: 480px;
        background-color: white;
        padding: 25px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        border-radius: 10px;
        z-index: 9999;
        font-family: Arial, sans-serif;
    }}
    .chatbot-box h4 {{
        margin-top: 0;
    }}
    .chatbot-answer {{
        margin-top: 10px;
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 8px;
        font-size: 16px;
        max-height: 400px;
        overflow-y: auto;
    }}
    </style>

    <div class="chatbot-box">
        <h4>💬 Ask a Question (test version)</h4>
        <form action="" method="get">
            <input type="text" name="user_query" placeholder="Ask something..." style="width: 100%; padding: 12px; margin-bottom: 10px; font-size: 18px;">
            <input type="submit" value="Submit" style="width: 100%; padding: 12px; font-size: 18px;">
        </form>
        <div class="chatbot-answer"><strong>Respond:</strong><br>{answer_html}</div>
        <p style="color: black; font-size: 13px; margin-top: 5px; margin-bottom: 5px;">
            💡We are working on the accurancy of the chatbot, so answer may not correct.
        </p>
    </div>
""", unsafe_allow_html=True)




