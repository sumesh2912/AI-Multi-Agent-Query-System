# AI-Multi-Agent-Query-System
A natural language interface for database and external search operations using LangGraph, FastAPI, and Streamlit.

Overview
This system uses three specialized AI agents to handle different types of queries:

LOCAL_DB_AGENT - Database operations (SELECT, INSERT, UPDATE, DELETE)
EXTERNAL_SEARCH_AGENT - Search external recruitment platforms (LinkedIn, Indeed, Glassdoor)
HYBRID_AGENT - Combines external search with automatic database insertion

Architecture

FastAPI Backend - API server handling query requests
LangGraph - Orchestration layer for agent routing
Streamlit Frontend - User-friendly web interface
PostgreSQL - Database for storing people records
Groq LLM - AI model for intent classification and data extraction

Setup Instructions

Python 3.8+
PostgreSQL database
Groq API key

1. Clone the Repository

2. Create Virtual Environment
Create virtual environment
python -m venv .venv

Activate virtual environment
On Windows:
.venv\Scripts\activate

On macOS/Linux:
source .venv/bin/activate

3. Install Dependencies

pip install -r requirements.txt

4. Environment Configuration
Create a .env file in the project root
GROQ_API_KEY=your_groq_api_key_here
POSTGRES_URI=postgresql:

5. Database Setup
Create database
CREATE DATABASE people;

Steps for Running the Application

Step 1: Start the FastAPI Backend
FastAPI Command
uvicorn main:app --port 8000 --reload

Step 2: Start the Streamlit Frontend
Make sure virtual environment is activated
streamlit run app.py
