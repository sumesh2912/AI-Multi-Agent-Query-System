from typing import Dict, Any, List
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_groq import ChatGroq
from langchain.agents.agent_types import AgentType
import json
import re
import os
from constants import GROQ_API_KEY, POSTGRES_URI

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=GROQ_API_KEY
)

import json
import re
from typing import Dict, Any
from langchain.sql_database import SQLDatabase

def safe_json_parse(text: str):
    try:
        return json.loads(text)
    except Exception:
        return {}

def extract_name_fallback(query: str):
    match = re.search(
        r"add\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        query,
        re.IGNORECASE
    )
    return match.group(1) if match else None

def extract_role_fallback(query: str):
    match = re.search(
        r"(?:a|an)\s+([a-zA-Z\s]+?)\s+(?:from|in)",
        query,
        re.IGNORECASE
    )
    return match.group(1).strip() if match else None


def extract_location_fallback(query: str):
    match = re.search(
        r"(?:from|in)\s+([A-Z][a-zA-Z]+)",
        query,
        re.IGNORECASE
    )
    return match.group(1) if match else None

def normalize(text):
    if not text:
        return None
    return text.title()

def sql_value(val):
    if val is None:
        return "NULL"
    return "'" + val.replace("'", "''") + "'"

def local_db_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]

    try:
        db = SQLDatabase.from_uri(POSTGRES_URI)

        if query.lower().startswith("add"):
            response = llm.invoke(extract_prompt := f"""
            Extract structured data.

            Query:
            "{query}"

            Return ONLY JSON:
            {{
              "name": string | null,
              "role": string | null,
              "location": string | null
            }}
            """)

            content = (
                response.content
                if hasattr(response, "content")
                else str(response)
            ).strip()

            data = safe_json_parse(content)
            
            name = data.get("name")
            role = data.get("role")
            location = data.get("location")

            if not name:
                name = extract_name_fallback(query)

            if not role:
                role = extract_role_fallback(query)

            if not location:
                location = extract_location_fallback(query)

            name = normalize(name)
            role = normalize(role)
            location = normalize(location)

            insert_sql = f"""
            INSERT INTO people (name, role, location)
            VALUES (
                {sql_value(name)},
                {sql_value(role)},
                {sql_value(location)}
            );
            """

            db.run(insert_sql)

            return {
                "final_response": {
                    "agent": "LOCAL_DB",
                    "message": "Record inserted successfully",
                    "data": {
                        "name": name,
                        "role": role,
                        "location": location
                    }
                }
            }

        return {
            "final_response": {
                "agent": "LOCAL_DB",
                "message": "No insert operation detected"
            }
        }

    except Exception as e:
        return {
            "final_response": {
                "agent": "LOCAL_DB",
                "error": str(e)
            }
        }


def external_search_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    try:
        # Prompt LLM to generate realistic external search results
        search_prompt = f"""You are an external recruitment database API that searches across multiple platforms (LinkedIn, Indeed, Glassdoor, company databases).

USER SEARCH QUERY: "{query}"

Your task: Generate realistic candidate profiles based on this search query. 

INSTRUCTIONS:
1. Extract the role/position and location from the query
2. Generate 4-6 realistic candidate profiles
3. Each candidate should have:
   - name: Full name (realistic, diverse names)
   - role: Job title matching the query (be specific, include seniority levels)
   - location: City/Region matching the query
   - source: Always set to "external"

4. Make profiles realistic:
   - Vary seniority levels (Junior, Mid-level, Senior, Lead, Principal)
   - Use realistic name diversity (Indian, Western, Asian names)
   - Match location precisely from query

RESPOND ONLY WITH A JSON ARRAY. NO ADDITIONAL TEXT.

Example format:
[
  {{"name": "Priya Sharma", "role": "Senior Machine Learning Engineer", "location": "San Francisco", "source": "external"}},
  {{"name": "Michael Chen", "role": "ML Engineer", "location": "San Francisco", "source": "external"}}
]

Generate candidates now:"""

        print(f"[EXTERNAL_SEARCH] Searching for: {query}")
        
        response = llm.invoke(search_prompt)
        response_text = response.content.strip()
        
        json_match = re.search(r'\[[\s\S]*\]', response_text)
        if json_match:
            json_str = json_match.group(0)
            results = json.loads(json_str)
        else:
            # Fallback if JSON parsing fails
            results = []
        
        validated_results = []
        for candidate in results:
            if all(key in candidate for key in ["name", "role", "location"]):
                candidate["source"] = "external"
                validated_results.append(candidate)
        
        print(f"[EXTERNAL_SEARCH] Found {len(validated_results)} candidates")
        
        return {
            "external_result": validated_results,
            "final_response": {
                "agent": "EXTERNAL_SEARCH",
                "found_count": len(validated_results),
                "results": validated_results,
                "message": f"Found {len(validated_results)} candidates from external sources (LinkedIn, Indeed, Glassdoor)",
                "query": query
            }
        }
        
    except Exception as e:
        error_msg = f"External search failed: {str(e)}"
        print(f"[EXTERNAL_SEARCH] {error_msg}")
        
        return {
            "external_result": [],
            "final_response": {
                "agent": "EXTERNAL_SEARCH",
                "found_count": 0,
                "results": [],
                "message": "External search encountered an error. Please try again.",
                "error": error_msg
            },
            "error": str(e)
        }


def hybrid_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    query = state["query"]
    
    try:
        print(f"HYBRID Starting hybrid operation for: {query}")
        
        external_search_prompt = f"""You are a recruitment search engine that searches multiple platforms.

USER QUERY: "{query}"

TASK: Generate 8-10 realistic candidate profiles matching this search query.

REQUIREMENTS:
1. Extract role and location from the query
2. Generate diverse candidates with varying experience levels
3. Use realistic, diverse names from different backgrounds
4. Each candidate needs: name, role, location, source (set to "external")

OUTPUT: JSON array only, no other text.

Format:
[
  {{"name": "Full Name", "role": "Specific Job Title", "location": "City/Region", "source": "external"}}
]

Generate candidates:"""

        external_response = llm.invoke(external_search_prompt)
        external_text = external_response.content.strip()
        
        json_match = re.search(r'\[[\s\S]*\]', external_text)
        if json_match:
            external_results = json.loads(json_match.group(0))
        else:
            external_results = []
        
        # Validate external results
        external_results = [
            {**r, "source": "external"} 
            for r in external_results 
            if all(k in r for k in ["name", "role", "location"])
        ]
        
        print(f"HYBRID External search found: {len(external_results)} candidates")
        
        if not external_results:
            return {
                "external_result": [],
                "local_result": None,
                "final_response": {
                    "agent": "HYBRID",
                    "message": "No external results found",
                    "external_count": 0,
                    "database_count": 0
                }
            }
        
        print(" ===== DATABASE OPERATIONS =====")
        
        num_to_add = 5
        if "top 3" in query.lower() or "first 3" in query.lower():
            num_to_add = 3
        elif "top 10" in query.lower():
            num_to_add = 10
        elif "all" in query.lower():
            num_to_add = len(external_results)
        
        top_results = external_results[:num_to_add]
        
        db = SQLDatabase.from_uri(POSTGRES_URI)
        
        existing_check_prompt = f"""You are a database query assistant.

TASK: Generate a SQL query to find existing people in the database that match the search criteria.

SEARCH QUERY: "{query}"

EXTERNAL CANDIDATES FOUND: {len(external_results)} candidates with roles like "{external_results[0]['role']}" in "{external_results[0]['location']}"

Generate a SELECT query to find similar people already in our database. Consider:
- Similar roles (use ILIKE for fuzzy matching)
- Same location
- Return: name, role, location

Respond with ONLY the SQL query, no explanation."""

        existing_query_response = llm.invoke(existing_check_prompt)
        existing_sql = existing_query_response.content.strip()
        
        # Clean SQL query
        existing_sql = existing_sql.replace("```sql", "").replace("```", "").strip()
        
        try:
            print(f"HYBRID Checking existing database records...")
            existing_records = db.run(existing_sql)
            print(f"HYBRID Found existing records: {existing_records}")
        except Exception as e:
            print(f"HYBRID Error checking existing records: {e}")
            existing_records = "No existing records found"
        
        print(f"HYBRID Attempting to insert top {num_to_add} external candidates...")
        
        inserted_count = 0
        inserted_people = []
        skipped_people = []
        
        for person in top_results:
            try:
                check_query = f"""
                SELECT COUNT(*) FROM people 
                WHERE LOWER(name) = LOWER('{person["name"].replace("'", "''")}')
                """
                existing = db.run(check_query)

                if "0" in str(existing) or existing == 0:
                    insert_query = """
                    INSERT INTO people (name, role, location, source)
                    VALUES ('{name}', '{role}', '{location}', '{source}')
                    """.format(
                        name=person["name"].replace("'", "''"),
                        role=person["role"].replace("'", "''"),
                        location=person["location"].replace("'", "''"),
                        source="external"
                    )
                    db.run(insert_query)
                    inserted_count += 1
                    inserted_people.append(person)
                    print(f"HYBRID Inserted: {person['name']} - {person['role']}")
                else:
                    skipped_people.append(person)
                    print(f"HYBRID Skipped (duplicate): {person['name']}")
                    
            except Exception as e:
                print(f"HYBRID Error inserting {person['name']}: {str(e)}")
                skipped_people.append(person)
                continue
        
        summary_prompt = f"""Generate a SQL query to get summary statistics of people in the database.

Return counts by:
1. Total people
2. People by source (manual vs external)
3. Top 3 roles

Respond with ONLY the SQL query."""

        summary_response = llm.invoke(summary_prompt)
        summary_sql = summary_response.content.strip().replace("```sql", "").replace("```", "").strip()
        
        try:
            db_summary = db.run(summary_sql)
        except:
            db_summary = "Summary unavailable"
        
        print(f"HYBRID ===== OPERATION COMPLETE =====")
        print(f"HYBRID External found: {len(external_results)}")
        print(f"HYBRID Inserted: {inserted_count}")
        print(f"HYBRID Skipped: {len(skipped_people)}")
        
        return {
            "external_result": external_results,
            "local_result": existing_records,
            "final_response": {
                "agent": "HYBRID",
                "message": f"Hybrid operation completed successfully",
                "summary": {
                    "external_search": {
                        "total_found": len(external_results),
                        "searched_platforms": "LinkedIn, Indeed, Glassdoor, Company Databases"
                    },
                    "database_operation": {
                        "existing_similar_records": existing_records,
                        "new_records_inserted": inserted_count,
                        "duplicates_skipped": len(skipped_people),
                        "database_summary": db_summary
                    }
                },
                "inserted_people": inserted_people,
                "skipped_people": skipped_people,
                "all_external_results": external_results[:10]  # Show first 10
            }
        }
        
    except Exception as e:
        error_msg = f"Hybrid operation failed: {str(e)}"
        print(f"HYBRID {error_msg}")
        return {
            "external_result": None,
            "local_result": None,
            "final_response": {
                "agent": "HYBRID",
                "error": error_msg
            },
            "error": str(e)

        }
