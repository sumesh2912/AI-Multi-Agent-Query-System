import streamlit as st
import requests
import json

# ---------------- CONFIG ----------------
API_URL = "http://127.0.0.1:8000/query"

st.set_page_config(
    page_title="AI Multi-Agent Search System",
    page_icon="🤖",
    layout="centered"
)

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center;'>🤖 AI Multi-Agent Query System</h1>
    <p style='text-align: center; color: grey;'>
    Natural language interface for Database & External Search
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ---------------- INPUT ----------------
st.subheader("Enter your query")

query = st.text_input(
    label="",
    placeholder="e.g. Find machine learning engineers in San Francisco",
)

submit = st.button("Run Query", use_container_width=True)

# ---------------- HELP ----------------
# with st.expander("💡 Example Queries"):
#     st.markdown("""
#     **Local DB**
#     - Show me all people in the database
#     - How many engineers are in Mumbai?

#     **External Search**
#     - Find machine learning engineers in San Francisco
#     - Search for AI researchers in Europe

#     **Hybrid**
#     - Find ML engineers in SF and add top 5 to database
#     """)

# ---------------- PROCESS ----------------
if submit:
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing your query..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": query},
                    timeout=120
                )

                if response.status_code != 200:
                    st.error(f"API Error: {response.status_code}")
                    st.text(response.text)
                else:
                    data = response.json()

                    # ---------------- RESULT HEADER ----------------
                    st.success("Query processed successfully")

                    col1, col2 = st.columns(2)
                    col1.metric("Intent", data.get("intent", "N/A"))
                    col2.metric(
                        "🤖 Agent",
                        data.get("response", {}).get("agent", "N/A")
                    )

                    st.divider()

                    # ---------------- RESPONSE DISPLAY ----------------
                    final_response = data.get("response")

                    if not final_response:
                        st.warning("No response received.")
                    else:
                        agent = final_response.get("agent")

                        # ----- LOCAL DB -----
                        if agent == "LOCAL_DB":
                            st.subheader("Database Result")
                            st.write(final_response.get("message"))

                        # ----- EXTERNAL SEARCH -----
                        elif agent == "EXTERNAL_SEARCH":
                            st.subheader("External Candidates")

                            results = final_response.get("results", [])
                            st.caption(f"Found {len(results)} candidates")

                            for i, r in enumerate(results, 1):
                                with st.container(border=True):
                                    st.markdown(f"### {i}. {r['name']}")
                                    st.markdown(f"**Role:** {r['role']}")
                                    st.markdown(f"**Location:** {r['location']}")
                                    st.markdown("**Source:** External")

                        # ----- HYBRID -----
                        elif agent == "HYBRID":
                            st.subheader("Hybrid Operation Summary")

                            summary = final_response.get("summary", {})
                            db_ops = summary.get("database_operation", {})

                            st.markdown("### External Search")
                            st.write(summary.get("external_search"))

                            st.markdown("### Database Operation")
                            st.json(db_ops)

                            st.markdown("### Inserted Candidates")
                            inserted = final_response.get("inserted_people", [])
                            if inserted:
                                for p in inserted:
                                    st.markdown(
                                        f"- **{p['name']}** | {p['role']} | {p['location']}"
                                    )
                            else:
                                st.info("No new candidates inserted.")

                        # ----- ERROR -----
                        else:
                            st.error("❌ Error from system")
                            st.json(final_response)

                    # ---------------- RAW JSON ----------------
                    with st.expander("View Raw JSON Response"):
                        st.json(data)

            except requests.exceptions.Timeout:
                st.error("Request timed out. Model may be loading.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# ---------------- FOOTER ----------------
st.divider()
# st.caption(
#     "Built wusing FastAPI, LangGraph & Ollama | Local AI System"
# )
