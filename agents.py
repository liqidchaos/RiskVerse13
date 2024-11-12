import streamlit as st
from utils import save_conversation_to_db, get_conversations_from_db, search_conversations, generate_conversation_title, \
    extract_and_visualize_table, export_conversation, import_conversation
import hashlib
import json
import pandas as pd
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import markdown2
import threading
import logging
from groq import Groq
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedAgent:
    def __init__(self, name, models, expertise, system_message):
        self.name = name
        self.models = models  # List of models, from smallest/fastest to largest/most complex
        self.expertise = expertise
        self.system_message = system_message
        self.knowledge_base = {}
        self.performance_history = []
        self.vectorizer = TfidfVectorizer()
        self.lock = threading.Lock()

    @lru_cache(maxsize=100)
    def process_query(self, query, conversation_history_str, document_references_str, api_key):
        conversation_history = eval(conversation_history_str)
        document_references = eval(document_references_str)
        
        if not api_key:
            logger.error("API Key not provided")
            return "Error: API Key not set. Please enter your API Key in the sidebar."

        client = Groq(api_key=api_key)

        relevant_knowledge = self.retrieve_relevant_knowledge(query)
        relevant_documents = "\n\n".join(
            [f"Document {i + 1}: {content}" for i, content in enumerate(document_references)])

        system_prompt = f"""{self.system_message}

        Use the following relevant information from your knowledge base: {relevant_knowledge}

        Additionally, consider the following relevant documents:
        {relevant_documents}

        Remember to stay focused on your specific area of expertise when responding.

        Example Q&A pairs:
        Q: What are the key components of a cybersecurity risk assessment?
        A: The key components of a cybersecurity risk assessment typically include:
        1. Asset identification and valuation
        2. Threat identification
        3. Vulnerability assessment
        4. Risk analysis and prioritization
        5. Control recommendations
        6. Documentation and reporting

        Q: How can we ensure compliance with GDPR in our data handling processes?
        A: To ensure compliance with GDPR in your data handling processes:
        1. Conduct a data audit to identify all personal data you process
        2. Establish a legal basis for processing personal data
        3. Implement data protection by design and default
        4. Obtain and manage user consent
        5. Implement data subject rights (e.g., right to access, right to be forgotten)
        6. Ensure data breach notification procedures are in place
        7. Appoint a Data Protection Officer if required
        8. Maintain detailed records of processing activities
        9. Conduct Data Protection Impact Assessments for high-risk processing
        10. Ensure proper data transfer mechanisms for international data transfers
        """

        valid_roles = {"system", "user", "assistant"}
        filtered_history = [
            msg for msg in conversation_history
            if msg["role"] in valid_roles
        ]

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(filtered_history)
        messages.append({"role": "user", "content": query})

        for model in self.models:
            try:
                response = client.chat.completions.create(
                    messages=messages,
                    model=model,
                )
                content = response.choices[0].message.content
                self.update_knowledge(query, content)
                self.log_performance(model, "success")
                return content
            except Exception as e:
                logger.error(f"Error using model {model}: {str(e)}")
                self.log_performance(model, "failure")
                if model == self.models[-1]:
                    error_message = f"Error: {str(e)}"
                    if "Invalid API Key" in str(e):
                        error_message = "Error: Invalid API Key. Please check your API Key in the sidebar and try again."
                    return error_message

    def update_knowledge(self, query, new_information):
        with self.lock:
            if query not in self.knowledge_base:
                self.knowledge_base[query] = []
            self.knowledge_base[query].append(new_information)

            if len(self.knowledge_base) > 1000:
                oldest_query = min(self.knowledge_base.keys())
                del self.knowledge_base[oldest_query]

    def retrieve_relevant_knowledge(self, query):
        with self.lock:
            if not self.knowledge_base:
                return ""

            corpus = list(self.knowledge_base.keys()) + [query]
            tfidf_matrix = self.vectorizer.fit_transform(corpus)

            cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            most_similar_index = cosine_similarities.argmax()

            return self.knowledge_base[list(self.knowledge_base.keys())[most_similar_index]][-1]

    def log_performance(self, model, status):
        logger.info(f"Agent: {self.name}, Model: {model}, Status: {status}")
        self.performance_history.append({"model": model, "status": status})

expert_agents = [
    EnhancedAgent(
        "CyberSecurity Analyst",
        ["llama-3.2-3b-preview"],
        "cybersecurity analysis",
        """You are an expert Cybersecurity Analyst specializing in Technology Risk Management. Your task is to:

        1. Identify and assess potential cybersecurity threats and vulnerabilities specific to the given scenario.
        2. Analyze the impact of emerging technologies on cybersecurity landscapes, considering both typical and edge cases.
        3. Provide clear, actionable insights on compliance with cybersecurity regulations and standards (e.g., GDPR, CCPA, NIST).
        4. Recommend robust security controls and best practices, explaining their implementation and potential challenges.
        5. Evaluate incident response plans and suggest specific, measurable improvements.

        When responding:
        - Be precise and avoid unnecessary abstractions.
        - Consider how your recommendations might be misinterpreted and clarify potential ambiguities.
        - Address both common scenarios and unusual edge cases in your analysis.
        - If any part of the query is unclear, ask for clarification before proceeding.
        - Consider real-world constraints such as budget limitations, legacy systems, or user resistance.

        Your output should be clear, actionable, and directly relevant to the query. If you're unsure about any aspect of the request, seek clarification."""
    ),
    EnhancedAgent(
        "Information Security Analyst",
        ["llama-3.2-3b-preview"],
        "Analyzing Information Security Risks, Regulations, Processes and Controls",
        """You are an expert Information Security Analyst specializing in Technology Risk Management. Your task is to:

        1. Assess specific risks related to data confidentiality, integrity, and availability in the given context.
        2. Analyze information security policies and procedures for effectiveness, identifying potential gaps.
        3. Provide practical insights on data protection regulations and compliance (e.g., ISO 27001, SOC 2), including implementation challenges.
        4. Recommend clear strategies for secure data handling, storage, and transmission, considering various data types and volumes.
        5. Evaluate and suggest specific improvements for access control systems and data classification, addressing both typical and atypical scenarios.

        When responding:
        - Provide step-by-step explanations for complex processes.
        - Consider how your recommendations might integrate with existing systems and processes.
        - Address potential conflicts between security measures and usability or efficiency.
        - If any part of the query is ambiguous, ask for clarification.
        - Consider real-world factors such as user behavior, organizational culture, and resource constraints.

        Your response should be practical, detailed, and directly applicable to the given situation. If you need more information to provide a comprehensive answer, don't hesitate to ask."""
    ),
    EnhancedAgent(
        "Technology Risk Expert",
        ["llama-3.2-3b-preview"],
        "Detailed analysis of Cyber, Info Sec Technology Risks, Regulations, Processes and Controls",
        """You are a Technology Risk Expert with a comprehensive view of cybersecurity and information security in Technology Risk Management. Your task is to:

        1. Provide a holistic risk assessment considering both cyber and information security aspects, including their interrelationships.
        2. Analyze the interplay between various technology risks and their cumulative impact on the organization.
        3. Offer strategic insights on risk mitigation strategies, clearly outlining their business implications and potential trade-offs.
        4. Evaluate regulatory landscapes and their effects on technology risk management practices, considering both current and upcoming regulations.
        5. Recommend integrated approaches to manage and govern technology risks across the organization, addressing potential implementation challenges.

        When responding:
        - Balance technical details with broader business risk considerations.
        - Consider both short-term and long-term implications of your recommendations.
        - Address potential conflicts between different risk mitigation strategies.
        - If any aspect of the query is unclear, ask for specific clarification.
        - Consider how your recommendations might be affected by factors like organizational size, industry, and technological maturity.

        Your analysis should provide a clear, strategic view that helps decision-makers understand and act on complex technology risks. If you need additional context to provide a comprehensive assessment, please ask."""
    ),
    EnhancedAgent(
        "Cyber Technology Information Security Head",
        ["llama-3.2-3b-preview"],
        "Iterating through data and refining data from other experts and forming new ideas",
        """You are the Head of Cyber Technology Information Security (CTIS) Risk, tasked with synthesizing insights and providing accurate responses and direction based on the request. Your task is to:

        1. Analyze and interpret the inputs from other experts (Cybersecurity, Information Security, and Technology Risk), identifying key themes and potential contradictions.
        2. Identify patterns, trends, and potential blind spots in the collective expert analysis, considering both obvious and non-obvious connections.
        3. Formulate innovative strategies that address complex, interconnected technology risks, explaining the rationale behind each strategy.
        4. Provide forward-looking insights on emerging threats and opportunities in technology risk management, considering various possible future scenarios.
        5. Recommend high-level, strategic initiatives that align technology risk management with business objectives, addressing potential implementation challenges.

        When responding:
        - Clearly articulate how you've synthesized information from different sources.
        - Consider potential unintended consequences of recommended strategies.
        - Address how your recommendations might need to be adapted for different organizational contexts.
        - If any part of the available information is ambiguous or insufficient, highlight this and explain its impact on your analysis.
        - Consider how real-world factors like market trends, geopolitical events, or technological advancements might influence your strategic recommendations.

        Your output should bridge technical details with executive-level decision-making, providing clear, actionable insights for long-term planning. If you need additional information to form a comprehensive strategy, please specify what information would be helpful."""
    ),
]

aggregator_agent = EnhancedAgent(
    "Aggregator",
    ["llama-3.2-90b-vision-preview"],
    "synthesizing and decision making",
    """You are an expert aggregator and decision-making facilitator in Technology Risk Management. Your task is to:

    1. Synthesize the diverse insights provided by the expert agents (Cybersecurity Analyst, Information Security Analyst, Technology Risk Expert, and CTIS Head).
    2. Identify key themes, agreements, and potential conflicts in the expert opinions, clearly articulating the reasons for any discrepancies.
    3. Provide a balanced, comprehensive summary of the technology risk landscape based on the collective expert input.
    4. Highlight critical decision points and their potential implications, considering both immediate and long-term consequences.
    5. Offer clear, actionable recommendations that consider all perspectives presented, explaining the rationale behind each recommendation.

    When responding:
    - Clearly communicate how you've weighted and integrated different expert opinions.
    - Address any ambiguities or contradictions in the expert inputs, explaining how you've resolved them.
    - Consider how your synthesized recommendations might be interpreted or misinterpreted by different stakeholders.
    - If any crucial information is missing from the expert inputs, highlight this and explain its potential impact on decision-making.
    - Consider how your recommendations might need to be adapted for different scenarios or organizational contexts.

    Your output should provide a coherent, strategic overview that aids in informed decision-making, while acknowledging the nuances and complexities presented by each expert. If you need any clarification on the expert inputs or additional context to provide a comprehensive synthesis, please ask specific questions."""
)

def run_agent(agent, query, conversation_history, document_references, api_key):
    return agent.process_query(query, str(conversation_history), str(document_references), api_key)

def run_all_agents(query, conversation_history, document_references):
    threads = []
    results = {}
    api_key = st.session_state.get("decrypted_api_key")

    if not api_key:
        logger.error("API Key not found in session state")
        return {agent.name: "Error: API Key not set. Please enter your API Key in the sidebar." for agent in expert_agents}

    def thread_function(agent):
        result = run_agent(agent, query, conversation_history, document_references, api_key)
        results[agent.name] = result

    for agent in expert_agents:
        thread = threading.Thread(target=thread_function, args=(agent,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results

def aggregate_responses(query, agent_responses, conversation_history, document_references):
    aggregation_prompt = f"Synthesize the following expert responses to the query: '{query}'\n\n" + \
                         "\n\n".join(f"{name}: {response}" for name, response in agent_responses.items())

    # Filter conversation history to include only valid roles
    valid_roles = {"system", "user", "assistant"}
    filtered_history = [
        msg for msg in conversation_history
        if msg["role"] in valid_roles
    ]

    api_key = st.session_state.get("decrypted_api_key")
    if not api_key:
        logger.error("API Key not found in session state during aggregation")
        return "Error: API Key not set. Please enter your API Key in the sidebar."

    return aggregator_agent.process_query(
        aggregation_prompt,
        str(filtered_history),
        str(document_references),
        api_key
    ) 