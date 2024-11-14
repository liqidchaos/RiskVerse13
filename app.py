import streamlit as st
import pandas as pd
import sqlite3
from groq import Groq
from dotenv import load_dotenv
import os
import tempfile
import io
import PyPDF2
import docx
import re
from io import StringIO
import json
import plotly.express as px
from datetime import datetime
import uuid
from agents import expert_agents, aggregator_agent, run_all_agents, aggregate_responses
from graph_rag import GraphRAG

# Load environment variables from .env file
load_dotenv()

st.set_page_config(layout="wide", page_title="Control Analysis")

def init_db():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    # Initialize pathways table
    try:
        c.execute("SELECT pathway, description FROM pathways LIMIT 1")
    except sqlite3.OperationalError:
        # Only drop and recreate if wrong schema
        c.execute('DROP TABLE IF EXISTS pathways')
        c.execute('''CREATE TABLE IF NOT EXISTS pathways
                     (id INTEGER PRIMARY KEY, 
                      pathway TEXT NOT NULL,
                      description TEXT)''')
        conn.commit()
    
    # Initialize guidance table
    try:
        c.execute("SELECT name, category, description, requirements FROM guidance LIMIT 1")
    except sqlite3.OperationalError:
        # Create guidance table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS guidance
                     (id INTEGER PRIMARY KEY,
                      name TEXT NOT NULL,
                      category TEXT,
                      description TEXT,
                      requirements TEXT)''')
        conn.commit()
    
    # Initialize issues table
    try:
        c.execute("SELECT title FROM issues LIMIT 1")
    except sqlite3.OperationalError:
        # Create issues table if it doesn't exist
        c.execute('''CREATE TABLE IF NOT EXISTS issues
                     (id INTEGER PRIMARY KEY,
                      title TEXT NOT NULL,
                      description TEXT,
                      category TEXT,
                      severity TEXT,
                      status TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
    
    # Initialize risks table
    try:
        c.execute("""
            CREATE TABLE IF NOT EXISTS risks (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                category TEXT,
                likelihood TEXT,
                impact TEXT,
                inherent_risk_score REAL,
                residual_risk_score REAL,
                status TEXT,
                owner TEXT,
                review_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Initialize controls table
        c.execute("""
            CREATE TABLE IF NOT EXISTS controls (
                id TEXT PRIMARY KEY,
                risk_id TEXT,
                name TEXT NOT NULL,
                description TEXT,
                type TEXT,
                effectiveness TEXT,
                status TEXT,
                owner TEXT,
                implementation_date DATE,
                last_review_date DATE,
                next_review_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (risk_id) REFERENCES risks (id)
            )
        """)
        
        # Initialize risk_assessments table
        c.execute("""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id TEXT PRIMARY KEY,
                risk_id TEXT,
                assessment_date DATE,
                assessor TEXT,
                likelihood TEXT,
                impact TEXT,
                risk_score REAL,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (risk_id) REFERENCES risks (id)
            )
        """)
        
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Database error: {e}")
    finally:
        conn.close()

def save_pathway(pathway, description):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    # Check if pathway already exists
    c.execute("SELECT id FROM pathways WHERE pathway = ?", (pathway,))
    exists = c.fetchone()
    
    if not exists:
        c.execute("INSERT INTO pathways (pathway, description) VALUES (?, ?)", 
                  (pathway, description))
        conn.commit()
    conn.close()

def get_pathways():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT pathway, description FROM pathways")
    pathways = c.fetchall()
    conn.close()
    return pathways

def delete_pathway(pathway):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("DELETE FROM pathways WHERE pathway = ?", (pathway,))
    conn.commit()
    conn.close()

def classify_issue(client, issue_description, pathways):
    # Extract valid pathways and their descriptions
    valid_pathways = [p[0] for p in pathways]
    pathway_descriptions = {p[0]: p[1] for p in pathways}
    
    # Create a focused prompt
    prompt = f"""Given the following issue, identify the pathway that best represents the primary means or threat vector creating the impact.

Issue Description: {issue_description}

Available Pathways:
{chr(10).join(f'- {p}: {pathway_descriptions[p]}' for p in valid_pathways)}

Instructions: Analyze the issue based solely on its description and choose ONLY the name of the most appropriate pathway from the list above. Do not provide any additional explanations—return only the pathway name.

Selected pathway:"""
    
    try:
        # Use Groq's API with llama model
        response = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": prompt
            }],
            model="llama-3.2-11b-vision-preview",  # Use the same model as other parts of the application
            temperature=0,
            max_tokens=30,
            top_p=1,
            stream=False
        )
        
        # Extract the content
        result = response.choices[0].message.content.strip()
        result = result.replace('"', '').replace("'", "").strip()
        
        # Direct match check
        if result in valid_pathways:
            return result
        
        # Case-insensitive matching
        result_lower = result.lower()
        for pathway in valid_pathways:
            if pathway.lower() == result_lower:
                return pathway
            elif pathway.lower() in result_lower:
                return pathway
        
        return "Uncategorized"
        
    except Exception as e:
        print(f"Classification error: {str(e)}")
        return "Classification Error"

def find_issue_column(df):
    # Common column names that might contain issue descriptions
    possible_names = [
        'issue', 'description', 'problem', 'incident', 'details', 'summary',
        'issue_description', 'issue description', 'incident_description',
        'incident description', 'problem_description', 'problem description'
    ]
    
    # Check for exact matches (case-insensitive)
    for col in df.columns:
        if col.lower() in possible_names:
            return col
            
    # Check for partial matches
    for col in df.columns:
        for name in possible_names:
            if name in col.lower():
                return col
    
    # If no match found, let user select
    return None

def read_file(file):
    """Read and preprocess uploaded file with smart CSV detection"""
    try:
        if file.name.endswith('.csv'):
            # Read raw content
            file.seek(0)
            content = file.read().decode('utf-8')
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            
            # Process header and data separately
            header = ['Issue ID', 'Issue Description', 'Affected System', 
                     'Severity Level', 'Reported Date', 'Status']
            data_rows = []
            
            # Skip header line
            for line in lines[1:]:
                # Split the line and handle quotes
                parts = line.split(',')
                
                # Create row with proper structure
                row_data = {
                    'Issue ID': parts[0] if parts[0].strip().isdigit() else '',
                    'Issue Description': parts[1] if len(parts) > 1 else '',
                    'Affected System': parts[2] if len(parts) > 2 else '',
                    'Severity Level': parts[3] if len(parts) > 3 else '',
                    'Reported Date': parts[4] if len(parts) > 4 else '',
                    'Status': parts[5] if len(parts) > 5 else ''
                }
                
                data_rows.append(row_data)
            
            # Create DataFrame with explicit column order
            df = pd.DataFrame(data_rows, columns=header)
            
            # Clean up data values
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.strip().str.strip('"').str.strip("'")
            
            # Display info about the loaded data
            st.write(f"Loaded file with {len(df)} rows and {len(df.columns)} columns")
            
            # Preview the data
            st.write("Preview of detected columns:")
            st.dataframe(df.head(2))
            
            return df
            
        else:  # Excel files
            df = pd.read_excel(file)
            return df
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.error("Detailed error information:")
        st.code(str(e.__class__) + ": " + str(e))
        # Print the first few lines of the file for debugging
        try:
            file.seek(0)
            st.code("File contents (first 5 lines):\n" + '\n'.join(file.read().decode('utf-8').splitlines()[:5]))
        except:
            pass
        return None

def process_file(file, api_key):
    client = Groq(api_key=api_key)
    
    # Get pathways from database
    pathways = get_pathways()
    if not pathways:
        st.error("No pathways found! Please add pathways in the Pathway Management page first.")
        return None
    
    valid_pathways = [p[0] for p in pathways]
    
    try:
        df = read_file(file)
        if df is None:
            return None
            
        # Find issue column
        issue_column = find_issue_column(df)
        if issue_column is None:
            st.warning("Could not automatically detect issue column")
            issue_column = st.selectbox("Select issue column:", df.columns)
        
        # Add classification columns if not already present
        if 'Risk_Pathway' not in df.columns:
            df['Risk_Pathway'] = ''
            df['User_Override'] = ''
            
            # Initial classification with progress bar
            progress_text = "Classifying issues..."
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_issues = len(df)
            for idx, row in df.iterrows():
                status_text.write(f"Processing issue {idx + 1} of {total_issues}")
                issue_text = str(row[issue_column])
                pathway = classify_issue(client, issue_text, pathways)
                df.at[idx, 'Risk_Pathway'] = pathway
                progress_bar.progress((idx + 1) / total_issues)
            
            progress_bar.empty()
            status_text.empty()
            st.success("Initial classification complete!")
        
        # Review interface
        st.subheader("Review Classifications")
        st.write("Review and suggest changes to classifications. Changes will be applied when you click 'Apply All Changes'.")
        
        # Display classification summary and pending changes
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Current Classification Summary")
            summary = df['Risk_Pathway'].value_counts()
            st.write(summary)
        
        with col2:
            pending_overrides = df[df['User_Override'] != '']
            st.subheader("Pending Changes")
            st.write(f"Changes suggested: {len(pending_overrides)}")
        
        # Pagination controls
        chunk_size = 10
        total_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size > 0 else 0)
        chunk_index = st.number_input("Page", min_value=1, max_value=total_chunks, value=1) - 1
        
        start_idx = chunk_index * chunk_size
        end_idx = min(start_idx + chunk_size, len(df))
        
        # Display issues for review
        for idx in range(start_idx, end_idx):
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.text_area("Issue", df.iloc[idx][issue_column], height=100, key=f"issue_{idx}", disabled=True)
                with col2:
                    st.write("Current Classification:")
                    st.write(df.iloc[idx]['Risk_Pathway'])
                with col3:
                    current_override = df.at[idx, 'User_Override'] or "Keep Current Classification"
                    override = st.selectbox(
                        "Suggest New Classification",
                        ["Keep Current Classification"] + valid_pathways,
                        index=["Keep Current Classification"] + valid_pathways.index(current_override) if current_override in valid_pathways else 0,
                        key=f"override_{idx}"
                    )
                    if override != "Keep Current Classification":
                        df.at[idx, 'User_Override'] = override
                    elif current_override != "Keep Current Classification":
                        df.at[idx, 'User_Override'] = ''
                
                st.markdown("---")
        
        # Show pending changes detail
        if not pending_overrides.empty:
            if st.checkbox("Show detailed changes"):
                st.write("Pending classification changes:")
                for _, row in pending_overrides.iterrows():
                    st.write(f"• '{row[issue_column][:100]}...' will change from '{row['Risk_Pathway']}' to '{row['User_Override']}'")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply All Changes"):
                if not pending_overrides.empty:
                    with st.spinner("Applying changes..."):
                        # Apply overrides
                        mask = df['User_Override'] != ''
                        df.loc[mask, 'Risk_Pathway'] = df.loc[mask, 'User_Override']
                        df['User_Override'] = ''  # Clear overrides after applying
                    st.success("All changes applied successfully!")
                    st.experimental_rerun()  # Refresh the page to show updated classifications
                else:
                    st.info("No changes to apply")
                    
        with col2:
            if st.button("Finalize Classifications"):
                final_df = df.drop('User_Override', axis=1)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                    final_df.to_excel(tmp.name, index=False)
                    with open(tmp.name, 'rb') as f:
                        st.download_button(
                            label="Download Final Classifications",
                            data=f,
                            file_name="final_classifications.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                return final_df
        
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def pathway_management_page():
    st.title("Pathway Management")
    
    # Add multiple pathways
    st.write("""Enter pathways and descriptions (one per line). Format:
    
    Pathway Name: Description text
    Another Pathway: Its description text
    """)
    new_pathways = st.text_area("Enter pathways:", height=300)
    
    if st.button("Add Pathways"):
        if new_pathways:
            # Split by newlines and process each line
            pathway_entries = [entry.strip() for entry in new_pathways.split('\n') if entry.strip()]
            
            added_count = 0
            for entry in pathway_entries:
                # Check if entry contains a colon (separator between name and description)
                if ':' in entry:
                    # Split at first colon to separate pathway name and description
                    parts = entry.split(':', 1)
                    
                    # Clean up pathway name and description
                    pathway = parts[0].strip()
                    # Remove any numbering at the start (e.g., "3." or "4.")
                    pathway = re.sub(r'^\d+\.\s*', '', pathway)
                    
                    description = parts[1].strip() if len(parts) > 1 else ''
                    
                    if pathway:  # Only save if pathway name exists
                        save_pathway(pathway, description)
                        added_count += 1
            
            if added_count > 0:
                st.success(f"Added {added_count} pathways")
            else:
                st.warning("No valid pathways found to add")
        else:
            st.error("Please enter at least one pathway")
    
    # Display existing pathways
    st.subheader("Existing Pathways")
    pathways = get_pathways()
    for pathway, description in pathways:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{pathway}**")
            if description:
                st.write(description)
        with col2:
            if st.button("Delete", key=f"delete_{pathway}"):
                delete_pathway(pathway)
                st.rerun()

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    doc = docx.Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_file(file):
    """Extract and structure control data from various file types"""
    try:
        file_type = file.name.split('.')[-1].lower()
        
        if file_type in ['xlsx', 'xls']:
            # Debug information
            st.write(f"Reading Excel file: {file.name}")
            
            # Read Excel file with explicit engine
            df = pd.read_excel(file, engine='openpyxl')
            
            # Debug: Show dataframe info
            st.write("DataFrame Info:")
            st.write(f"Columns: {df.columns.tolist()}")
            st.write(f"Shape: {df.shape}")
            
            # Show preview of data
            st.write("Preview of data:")
            st.dataframe(df.head())
            
            # Identify columns
            columns = identify_control_columns(df)
            if columns:
                return structure_control_data(df, columns)
            else:
                st.error("Could not identify required control columns.")
                return None
            
        elif file_type == 'csv':
            df = pd.read_csv(file)
            columns = identify_control_columns(df)
            if columns:
                return structure_control_data(df, columns)
            else:
                st.error("Could not identify required control columns.")
                return None
            
    except Exception as e:
        st.error(f"Error reading file {file.name}")
        st.error(f"Detailed error: {str(e)}")
        st.write("File type:", file_type)
        return None

def identify_control_columns(df):
    """Identify control-related columns in the dataframe"""
    # Common patterns for control-related columns
    column_patterns = {
        'control_id': [
            'control.?id', 'control.?number', 'id', 'control.?ref', 
            'reference', 'ctrl.?id', 'control', 'number', '#'
        ],
        'control_name': [
            'control.?name', 'name', 'title', 'control.?title',
            'control.?objective', 'objective'
        ],
        'control_description': [
            'control.?desc', 'description', 'requirement', 
            'control.?requirement', 'control.?text', 'details',
            'control.?statement', 'statement'
        ],
        'category': [
            'category', 'domain', 'family', 'type', 
            'classification', 'group', 'control.?family'
        ]
    }
    
    identified_columns = {}
    
    # First try exact matches (case-insensitive)
    for col_type, patterns in column_patterns.items():
        for col in df.columns:
            col_lower = col.lower().strip()
            if any(pattern.lower() == col_lower for pattern in patterns):
                identified_columns[col_type] = col
                break
    
    # If not found, try partial matches
    for col_type, patterns in column_patterns.items():
        if col_type not in identified_columns:
            for col in df.columns:
                col_lower = col.lower().strip()
                if any(re.search(pattern, col_lower, re.IGNORECASE) for pattern in patterns):
                    identified_columns[col_type] = col
                    break
    
    # If still missing critical columns, try to make best guess
    if 'control_description' not in identified_columns and len(df.columns) > 0:
        # Try to identify the column with the longest text
        text_lengths = df.astype(str).apply(lambda x: x.str.len().mean())
        longest_col = text_lengths.idxmax()
        identified_columns['control_description'] = longest_col
    
    if 'control_id' not in identified_columns and len(df.columns) > 0:
        # Try to identify the column with short, unique values
        for col in df.columns:
            if df[col].astype(str).str.len().mean() < 20 and df[col].nunique() == len(df):
                identified_columns['control_id'] = col
                break
    
    # Display identified columns for verification
    st.write("Identified columns:")
    for col_type, col_name in identified_columns.items():
        st.write(f"{col_type}: {col_name}")
    
    # Allow user to override column selection
    if st.checkbox("Manually select columns?"):
        identified_columns = {
            'control_id': st.selectbox(
                "Select Control ID column", 
                df.columns, 
                index=df.columns.get_loc(identified_columns.get('control_id', df.columns[0]))
            ),
            'control_description': st.selectbox(
                "Select Control Description column", 
                df.columns, 
                index=df.columns.get_loc(identified_columns.get('control_description', df.columns[0]))
            )
        }
        # Optional columns
        if len(df.columns) > 2:
            identified_columns['control_name'] = st.selectbox(
                "Select Control Name column (optional)", 
                ['None'] + df.columns.tolist(),
                index=0
            )
            identified_columns['category'] = st.selectbox(
                "Select Category column (optional)", 
                ['None'] + df.columns.tolist(),
                index=0
            )
    
    return identified_columns

def structure_control_data(df, columns):
    """Structure the control data based on identified columns"""
    structured_data = []
    
    for _, row in df.iterrows():
        control = {
            'control_id': str(row[columns['control_id']]).strip(),
            'description': str(row[columns['control_description']]).strip(),
        }
        
        # Add optional fields if they exist
        if 'control_name' in columns and columns['control_name'] != 'None':
            control['title'] = str(row[columns['control_name']]).strip()
        else:
            control['title'] = ''
            
        if 'category' in columns and columns['category'] != 'None':
            control['category'] = str(row[columns['category']]).strip()
        else:
            control['category'] = ''
        
        structured_data.append(control)
    
    # Debug: Show structured data
    st.write("Structured control data preview:")
    st.write(structured_data[:2])
    
    return structured_data

def parse_requirements(text):
    """Parse text to extract control requirements"""
    requirements = []
    
    # Common requirement ID patterns
    patterns = [
        r'(?:Control|Requirement|[A-Z]+)[-._ ]?\d+(?:\.\d+)*:?\s*(.*?)(?=(?:Control|Requirement|[A-Z]+)[-._ ]?\d+(?:\.\d+)*:|$)',
        r'(?:\d+(?:\.\d+)*)[-._ ]+(.+?)(?=\d+(?:\.\d+)*[-._ ]+|$)',
        r'(?:[A-Z]+[-._ ]?\d+(?:\.\d+)*)[-._ ]+(.+?)(?=[A-Z]+[-._ ]?\d+(?:\.\d+)*|$)'
    ]
    
    # Try each pattern
    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.DOTALL)
        for match in matches:
            req_id = match.group(0).split(':')[0].strip()
            req_text = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).split(':', 1)[1].strip()
            if req_text:
                requirements.append((req_id, req_text))
    
    return requirements

def process_uploaded_guidance(file):
    """Process uploaded guidance file and extract requirements"""
    content = extract_text_from_file(file)
    
    if content is None:
        return None
    
    if isinstance(content, pd.DataFrame):
        # Handle structured data (CSV/Excel)
        st.write("Preview of uploaded data:")
        st.dataframe(content.head())
        
        # Let user map columns
        cols = content.columns.tolist()
        req_id_col = st.selectbox("Select Requirement ID column", cols)
        req_text_col = st.selectbox("Select Requirement Text column", cols)
        
        requirements = []
        for _, row in content.iterrows():
            requirements.append((str(row[req_id_col]), str(row[req_text_col])))
            
    else:
        # Handle unstructured text
        st.write("Extracted text preview:")
        st.text_area("Preview", content[:500] + "...", height=200, disabled=True)
        
        # Parse requirements
        requirements = parse_requirements(content)
        
        # Let user verify and edit parsed requirements
        st.write("Parsed Requirements:")
        requirements_df = pd.DataFrame(requirements, columns=['ID', 'Requirement'])
        edited_df = st.data_editor(requirements_df)
        
        requirements = list(zip(edited_df['ID'], edited_df['Requirement']))
    
    return requirements

def save_guidance(name, category, description, requirements):
    """Save guidance to database with better error handling"""
    conn = None
    try:
        conn = sqlite3.connect('data.db')
        c = conn.cursor()
        
        # Check if guidance already exists
        c.execute("SELECT id FROM guidance WHERE name = ?", (name,))
        exists = c.fetchone()
        
        if exists:
            # Update existing guidance
            c.execute("""UPDATE guidance 
                        SET category = ?, description = ?, requirements = ?
                        WHERE name = ?""",
                     (category, description, requirements, name))
        else:
            # Insert new guidance
            c.execute("""INSERT INTO guidance 
                        (name, category, description, requirements)
                        VALUES (?, ?, ?, ?)""",
                     (name, category, description, requirements))
        
        conn.commit()
        return True
        
    except Exception as e:
        st.error(f"Error saving guidance: {str(e)}")
        return False
        
    finally:
        if conn:
            conn.close()

def get_guidance():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("SELECT name, category, description, requirements FROM guidance")
    guidance = c.fetchall()
    conn.close()
    return guidance

def delete_guidance(name):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute("DELETE FROM guidance WHERE name = ?", (name,))
    conn.commit()
    conn.close()

def guidance_management_page():
    st.title("Control Guidance Management")
    
    # Add new guidance section
    st.header("Add New Guidance")
    
    input_method = st.radio(
        "Choose input method:",
        ["Manual Entry", "File Upload", "Use Template"]
    )
    
    if input_method == "File Upload":
        uploaded_file = st.file_uploader(
            "Upload guidance document",
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            st.info("Processing file... This may take a moment.")
            
            try:
                requirements = process_uploaded_guidance(uploaded_file)
                
                if requirements:
                    guidance_name = st.text_input("Guidance Name")
                    category = st.selectbox(
                        "Category",
                        ["Financial Services", "Healthcare", "Technology", 
                         "Privacy", "Cybersecurity", "Other"]
                    )
                    description = st.text_area("Description")
                    
                    # Format requirements for storage
                    formatted_reqs = "\n".join([f"{req_id}: {req_text}" 
                                              for req_id, req_text in requirements])
                    
                    if st.button("Save Guidance"):
                        if guidance_name:
                            save_guidance(guidance_name, category, description, formatted_reqs)
                            st.success(f"Successfully added guidance: {guidance_name}")
                        else:
                            st.error("Please provide a name for the guidance")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    elif input_method == "Manual Entry":
        guidance_name = st.text_input("Guidance Name (e.g., FFIEC Information Security Booklet)")
        
        category = st.selectbox(
            "Category",
            ["Financial Services", "Healthcare", "Technology", "Privacy", "Cybersecurity", "Other"],
            index=0
        )
        
        description = st.text_area(
            "Description",
            placeholder="Brief description of the guidance document..."
        )
        
        st.write("Enter requirements (one per line, format: 'Requirement ID: Requirement text')")
        requirements = st.text_area(
            "Requirements",
            height=300,
            placeholder="IS.B.1: Financial institutions should implement a comprehensive information security program...\nIS.B.2: The security program should include risk assessment processes..."
        )
    
    else:  # Use Template
        st.subheader("Select a Template")
        template_choice = st.selectbox(
            "Choose a control framework template:",
            [
                "NIST CSF 2.0",
                "ISO 27001:2022",
                "CIS Controls v8",
                "FFIEC Information Security",
                "PCI DSS 4.0",
                "HIPAA Security Rule",
                "SOC 2",
                "Custom Template"
            ]
        )
        
        if template_choice != "Custom Template":
            # Load template data
            template_data = get_template_data(template_choice)
            
            if template_data:
                guidance_name = template_data["name"]
                category = template_data["category"]
                description = template_data["description"]
                requirements = template_data["requirements"]
                
                st.write("**Template Preview:**")
                st.write(f"Category: {category}")
                st.write("Description:")
                st.write(description)
                st.write("Sample Requirements:")
                st.write(requirements[:500] + "...")
                
                if st.checkbox("Customize Template"):
                    guidance_name = st.text_input("Guidance Name", value=guidance_name)
                    category = st.selectbox("Category", ["Financial Services", "Healthcare", "Technology", "Privacy", "Cybersecurity", "Other"], index=["Financial Services", "Healthcare", "Technology", "Privacy", "Cybersecurity", "Other"].index(category))
                    description = st.text_area("Description", value=description)
                    requirements = st.text_area("Requirements", value=requirements, height=300)
    
    # Save button (common for all methods)
    if st.button("Add Guidance"):
        if 'guidance_name' in locals() and 'requirements' in locals():
            if guidance_name and requirements:
                save_guidance(guidance_name, category, description, requirements)
                st.success(f"Added guidance: {guidance_name}")
            else:
                st.error("Please provide at least the guidance name and requirements")
    
    # Display existing guidance
    st.header("Existing Guidance")
    guidance_list = get_guidance()
    
    if not guidance_list:
        st.info("No guidance documents added yet. Add some above!")
    else:
        for name, category, description, reqs in guidance_list:
            with st.expander(f"{name} ({category})"):
                st.write("**Description:**")
                st.write(description)
                st.write("**Requirements:**")
                for req in reqs.split('\n'):
                    if req.strip():
                        st.write(f"- {req.strip()}")
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button("Delete", key=f"delete_{name}"):
                        delete_guidance(name)
                        st.rerun()
                with col2:
                    if st.button("Export", key=f"export_{name}"):
                        export_guidance(name, category, description, reqs)

def get_template_data(template_name):
    """Return template data for common control frameworks"""
    templates = {
        "NIST CSF 2.0": {
            "name": "NIST Cybersecurity Framework 2.0",
            "category": "Cybersecurity",
            "description": "The NIST Cybersecurity Framework (CSF) 2.0 provides a comprehensive set of guidelines for better managing and reducing cybersecurity risks.",
            "requirements": """ID.1: Identify critical functions, assets, and risks
ID.2: Establish asset management processes
PR.1: Implement access control measures
PR.2: Deploy protective technology
DE.1: Enable anomaly detection
DE.2: Continuous security monitoring
RS.1: Response planning implementation
RS.2: Incident management procedures
RC.1: Recovery planning execution
RC.2: Business continuity management"""
        },
        "ISO 27001:2022": {
            "name": "ISO/IEC 27001:2022",
            "category": "Technology",
            "description": "ISO/IEC 27001 is an international standard for information security management systems (ISMS).",
            "requirements": """A.5.1: Information security policies
A.5.2: Review of information security policies
A.6.1: Internal organization
A.6.2: Mobile devices and teleworking
A.7.1: Prior to employment
A.7.2: During employment
A.7.3: Termination and change of employment"""
        },
        # Add more templates as needed
    }
    
    return templates.get(template_name)

def export_guidance(name, category, description, requirements):
    """Export guidance to Excel file"""
    df = pd.DataFrame({
        'Requirement ID': [req.split(':')[0].strip() for req in requirements.split('\n') if ':' in req],
        'Requirement Text': [req.split(':')[1].strip() for req in requirements.split('\n') if ':' in req]
    })
    
    # Create Excel buffer
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Requirements')
    
    # Provide download button
    st.download_button(
        label="Download Excel file",
        data=buffer.getvalue(),
        file_name=f"{name}_requirements.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def extract_regulatory_content(file, client):
    """Extract specific regulatory content from documents with focus on risk management and controls"""
    try:
        # Extract text based on file type
        text = extract_text_from_file(file)
        
        if text is None or not text.strip():
            st.error(f"No text could be extracted from {file.name}")
            return []

        # Add debug information
        st.write(f"Extracted text length: {len(text)} characters")
        
        # Define specialized extraction prompt
        prompt = """Analyze this control document and extract all controls in a structured format.
        For each control, provide:
        - Control ID (if present)
        - Control Name/Title
        - Control Description
        - Requirements
        - Category (if present)
        
        Format each control as:
        CONTROL_ID: [ID]
        CATEGORY: [Category]
        CONTROL_OBJECTIVE: [Title/Objective]
        REQUIREMENTS: [Detailed requirements]
        
        Document content to analyze:
        {text}"""

        # Process document in manageable chunks
        chunk_size = 4000
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        extracted_items = []

        # Show progress
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        for idx, chunk in enumerate(text_chunks):
            progress_text.text(f"Processing chunk {idx + 1} of {len(text_chunks)}...")
            progress_bar.progress((idx + 1) / len(text_chunks))
            
            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """You are a control framework expert. 
                    Extract and structure control requirements from documents."""},
                    {"role": "user", "content": prompt.format(text=chunk)}
                ],
                model="llama-3.2-11b-vision-preview",
                temperature=0.1,
                max_tokens=1500
            )

            # Parse structured response
            result = response.choices[0].message.content.strip()
            items = parse_regulatory_response(result)
            extracted_items.extend(items)

        progress_text.empty()
        progress_bar.empty()

        if not extracted_items:
            st.warning(f"No structured controls found in {file.name}. Please check the file format.")
            
        return extracted_items

    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return []

def parse_regulatory_response(response_text):
    """Parse AI response into structured control and risk management items"""
    items = []
    current_item = {}
    
    for line in response_text.split('\n'):
        line = line.strip()
        if not line:
            if current_item:
                items.append(current_item)
                current_item = {}
            continue
            
        # Parse key fields
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            
            if key in ['control_id', 'category', 'control_objective', 'requirements',
                      'risk_considerations', 'evidence', 'measurement_criteria']:
                current_item[key] = value

    # Add last item if exists
    if current_item:
        items.append(current_item)
    
    return items

def display_analysis_results(gaps, target_requirements):
    """Display the gap analysis results in a structured format"""
    try:
        with st.container():
            st.markdown("## Gap Analysis Results")
            
            # Calculate metrics
            total_reqs = len(target_requirements)
            covered_reqs = sum(1 for gap in gaps if gap['coverage_status'].lower() == 'covered')
            partial_reqs = sum(1 for gap in gaps if gap['coverage_status'].lower() == 'partial')
            missing_reqs = sum(1 for gap in gaps if gap['coverage_status'].lower() == 'missing')
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Full Coverage", 
                    f"{(covered_reqs/total_reqs)*100:.1f}%",
                    help="Requirements fully met by existing controls"
                )
            with col2:
                st.metric(
                    "Partial Coverage", 
                    f"{(partial_reqs/total_reqs)*100:.1f}%",
                    help="Requirements partially met and needing improvements"
                )
            with col3:
                st.metric(
                    "Gaps", 
                    f"{(missing_reqs/total_reqs)*100:.1f}%",
                    help="Requirements with no coverage"
                )

            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Gaps", "Partial Coverage", "Full Coverage"])
            
            with tab1:
                st.markdown("### Overall Analysis")
                for gap in gaps:
                    with st.expander(f"Requirement {gap['requirement_id']}"):
                        st.markdown(f"**Status:** {gap['coverage_status']}")
                        st.markdown(f"**Requirement Text:** {gap['requirement_text']}")
                        if gap['matching_controls']:
                            st.markdown("**Matching Controls:**")
                            st.write(", ".join(gap['matching_controls']))
                        st.markdown("**Gap Description:**")
                        st.write(gap['gap_description'])
            
            with tab2:
                st.markdown("### Missing Controls")
                missing_gaps = [g for g in gaps if g['coverage_status'].lower() == 'missing']
                for gap in missing_gaps:
                    with st.expander(f"Requirement {gap['requirement_id']}"):
                        st.markdown(f"**Requirement:** {gap['requirement_text']}")
                        st.markdown("**Suggested New Controls:**")
                        st.write(gap['new_controls'])
                        st.markdown("**Implementation Notes:**")
                        st.write(gap['implementation_notes'])
            
            with tab3:
                st.markdown("### Partial Coverage - Improvement Needed")
                partial_gaps = [g for g in gaps if g['coverage_status'].lower() == 'partial']
                for gap in partial_gaps:
                    with st.expander(f"Requirement {gap['requirement_id']}"):
                        st.markdown(f"**Requirement:** {gap['requirement_text']}")
                        st.markdown("**Current Controls:**")
                        st.write(", ".join(gap['matching_controls']))
                        st.markdown("**📝 Suggested Improvements:**")
                        st.write(gap['control_improvements'])
                        st.markdown("**➕ Additional Controls Needed:**")
                        st.write(gap['new_controls'])
                        st.markdown("**🔧 Implementation Guide:**")
                        st.write(gap['implementation_notes'])
            
            with tab4:
                st.markdown("### Fully Covered Requirements")
                covered_gaps = [g for g in gaps if g['coverage_status'].lower() == 'covered']
                for gap in covered_gaps:
                    with st.expander(f"Requirement {gap['requirement_id']}"):
                        st.markdown(f"**Requirement:** {gap['requirement_text']}")
                        st.markdown("**Matching Controls:**")
                        st.write(", ".join(gap['matching_controls']))

            # Export functionality
            st.markdown("### Export Results")
            df = pd.DataFrame(gaps)
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Analysis",
                csv,
                "gap_analysis_results.csv",
                "text/csv",
                key='download-csv'
            )

    except Exception as e:
        st.error(f"Error displaying results: {str(e)}")
        st.write("Detailed error:")
        st.write(e)

def analyze_control_gaps(current_controls, target_requirements, client):
    """Analyze gaps between current controls and target requirements"""
    gaps = []
    
    try:
        progress_bar = st.progress(0)
        for idx, target_req in enumerate(target_requirements):
            analysis_prompt = f"""As a control framework expert, analyze this requirement against the existing controls and provide detailed suggestions:

REQUIREMENT TO ANALYZE:
ID: {target_req['ID']}
Text: {target_req['Requirement']}
CURRENT CONTROLS:
{json.dumps(current_controls, indent=2)}

Provide a detailed analysis including:
1. Whether the requirement is fully met, partially met, or not met by existing controls
2. Which specific existing controls (if any) partially or fully address this requirement
3. For partial matches:
   - Exactly what aspects are missing
   - Specific suggested updates to existing control language
4. For missing coverage:
   - Draft complete new control(s) with ID, objective, and requirements
5. Implementation guidance for any changes

Format your response exactly as follows:
COVERAGE_STATUS: [Covered/Partial/Missing]
MATCHING_CONTROLS: [List specific control IDs that match]
GAP_DESCRIPTION: [Detail what aspects are not covered]
CONTROL_IMPROVEMENTS: [If partial coverage, provide specific updated control text]
NEW_CONTROLS: [For gaps, provide complete new control statements]
IMPLEMENTATION_NOTES: [Specific guidance for implementing changes]

Be specific and provide implementation-ready control language."""

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """You are a control framework expert specializing in drafting precise control language. 
                    Provide detailed, implementation-ready control statements that align with common frameworks like NIST, ISO, and CIS."""},
                    {"role": "user", "content": analysis_prompt}
                ],
                model="llama-3.2-11b-vision-preview",
                temperature=0.1,
                max_tokens=2500
            )

            # Parse the response
            analysis = {
                'requirement_id': target_req['ID'],
                'requirement_text': target_req['Requirement'],
                'coverage_status': 'Unknown',
                'matching_controls': [],
                'gap_description': '',
                'control_improvements': '',
                'new_controls': '',
                'implementation_notes': ''
            }

            # Enhanced response parsing with better handling of multi-line content
            response_text = response.choices[0].message.content.strip()
            current_field = None
            current_content = []

            for line in response_text.split('\n'):
                line = line.strip()
                
                # Check for new field markers
                if line.startswith('COVERAGE_STATUS:'):
                    current_field = 'coverage_status'
                    analysis['coverage_status'] = line.split(':', 1)[1].strip()
                elif line.startswith('MATCHING_CONTROLS:'):
                    current_field = 'matching_controls'
                    controls = line.split(':', 1)[1].strip()
                    analysis['matching_controls'] = [c.strip() for c in controls.strip('[]').split(',') if c.strip()]
                elif line.startswith('GAP_DESCRIPTION:'):
                    current_field = 'gap_description'
                    current_content = [line.split(':', 1)[1].strip()]
                elif line.startswith('CONTROL_IMPROVEMENTS:'):
                    current_field = 'control_improvements'
                    current_content = [line.split(':', 1)[1].strip()]
                elif line.startswith('NEW_CONTROLS:'):
                    current_field = 'new_controls'
                    current_content = [line.split(':', 1)[1].strip()]
                elif line.startswith('IMPLEMENTATION_NOTES:'):
                    current_field = 'implementation_notes'
                    current_content = [line.split(':', 1)[1].strip()]
                elif line and current_field in ['gap_description', 'control_improvements', 'new_controls', 'implementation_notes']:
                    current_content.append(line)
                
                # Update the analysis dict with accumulated content
                if current_field in ['gap_description', 'control_improvements', 'new_controls', 'implementation_notes']:
                    analysis[current_field] = '\n'.join(current_content)

            gaps.append(analysis)
            progress_bar.progress((idx + 1) / len(target_requirements))

        return gaps

    except Exception as e:
        st.error(f"Error in gap analysis: {str(e)}")
        return []

def process_controls_file(uploaded_files, api_key):
    """Main function to process regulatory documents and perform gap analysis"""
    client = Groq(api_key=api_key)
    extracted_data = []
    
    with st.spinner("Analyzing regulatory documents..."):
        for file in uploaded_files:
            items = extract_text_from_file(file)
            if items:
                for item in items:
                    # Ensure all required fields are present
                    structured_item = {
                        'control_id': item.get('control_id', 'Unknown'),
                        'title': item.get('title', ''),
                        'description': item.get('description', ''),
                        'category': item.get('category', ''),
                        'source': file.name
                    }
                    extracted_data.append(structured_item)
                st.success(f"Extracted {len(items)} control items from {file.name}")
            else:
                st.warning(f"No relevant content found in {file.name}")

    if not extracted_data:
        st.error("No control or risk management data was extracted.")
        return None

    # Display extracted controls for review
    st.subheader("Review Extracted Controls")
    df = pd.DataFrame(extracted_data)
    
    # Allow editing of extracted controls
    edited_df = st.data_editor(
        df,
        use_container_width=True,
        height=400,
        column_config={
            "control_id": "Control ID",
            "title": "Control Name",
            "description": "Control Description",
            "category": "Category",
            "source": "Source File"
        }
    )

    if st.button("Run Gap Analysis"):
        # Continue with gap analysis...
        guidance_list = get_guidance()
        if not guidance_list:
            st.error("Please add some control guidance documents first!")
            return None
            
        # Parse guidance requirements
        requirements_text = guidance_list[0][3]
        requirements = []
        for line in requirements_text.split('\n'):
            if ':' in line:
                req_id, req_text = line.split(':', 1)
                requirements.append({
                    'ID': req_id.strip(),
                    'Requirement': req_text.strip()
                })
        
        # Perform gap analysis with structured control data
        gaps = analyze_control_gaps(edited_df.to_dict('records'), requirements, client)
        
        if gaps:
            display_analysis_results(gaps, requirements)
            return pd.DataFrame(gaps)
        else:
            st.error("No results generated from the analysis")
            return None

    return None

def visualization_page():
    st.title("Issue Analytics Dashboard")
    
    conn = get_db_connection()
    if not conn:
        st.error("Could not connect to database")
        return
        
    try:
        # Load all issues into a DataFrame
        df_issues = pd.read_sql_query("""
            SELECT title, description, category, severity, status, created_at 
            FROM issues
        """, conn)
        
        # Load pathways and guidance data
        df_pathways = pd.read_sql_query("SELECT * FROM pathways", conn)
        df_guidance = pd.read_sql_query("SELECT * FROM guidance", conn)
        
        if df_issues.empty:
            st.info("No issues found in the database")
            return
        
        # Convert created_at to datetime
        df_issues['created_at'] = pd.to_datetime(df_issues['created_at'])
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Issues", "Pathway Management", "Control Guidance"])
        
        with tab1:
            # Create layout
            col1, col2 = st.columns(2)
            
            with col1:
                # Issues by Status
                st.subheader("Issues by Status")
                status_counts = df_issues['status'].value_counts()
                st.plotly_chart(
                    px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title="Issue Distribution by Status"
                    ),
                    use_container_width=True
                )
                
            with col2:
                # Issues by Severity with custom colors
                st.subheader("Issues by Severity")
                severity_counts = df_issues['severity'].value_counts()
                fig_severity = px.bar(
                    x=severity_counts.index,
                    y=severity_counts.values,
                    title="Issues by Severity Level",
                    labels={'x': 'Severity', 'y': 'Count'}
                )
                
                # Update bar colors based on severity
                severity_colors = {
                    'Low': '#2ECC71',      # Green
                    'Medium': '#F1C40F',   # Yellow
                    'High': '#E67E22',     # Orange
                    'Critical': '#E74C3C'  # Red
                }
                fig_severity.update_traces(
                    marker_color=[severity_colors.get(sev, '#808080') for sev in severity_counts.index]
                )
                st.plotly_chart(fig_severity, use_container_width=True)
            
            # Issues over Time
            st.subheader("Issues Over Time")
            df_time = df_issues.set_index('created_at')
            daily_issues = df_time.resample('D').size()
            
            st.plotly_chart(
                px.line(
                    daily_issues,
                    title="Daily Issue Trends",
                    labels={'value': 'Number of Issues', 'created_at': 'Date'}
                ),
                use_container_width=True
            )
            
            # Category Distribution
            st.subheader("Category Analysis")
            col3, col4 = st.columns(2)
            
            with col3:
                category_counts = df_issues['category'].value_counts()
                st.plotly_chart(
                    px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Issues by Category"
                    ),
                    use_container_width=True
                )
                
            with col4:
                # Severity by Category with custom colors
                severity_by_category = pd.crosstab(df_issues['category'], df_issues['severity'])
                fig_severity_cat = px.bar(
                    severity_by_category,
                    title="Severity Distribution by Category",
                    barmode='group',
                    labels={'value': 'Count', 'category': 'Category'}
                )
                
                # Update colors for each severity level
                for i, severity in enumerate(severity_by_category.columns):
                    fig_severity_cat.data[i].marker.color = severity_colors.get(severity, '#808080')
                    
                st.plotly_chart(fig_severity_cat, use_container_width=True)
            
            # Status Timeline with severity colors
            st.subheader("Status Timeline")
            severity_timeline = df_issues.pivot_table(
                index='created_at',
                columns='severity',
                aggfunc='size',
                fill_value=0
            ).resample('W').sum()
            
            fig_timeline = px.area(
                severity_timeline,
                title="Severity Trends Over Time",
                labels={'value': 'Number of Issues', 'created_at': 'Date'}
            )
            
            # Update area colors
            for i, severity in enumerate(severity_timeline.columns):
                fig_timeline.data[i].fillcolor = severity_colors.get(severity, '#808080')
                fig_timeline.data[i].line.color = severity_colors.get(severity, '#808080')
                
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        with tab2:
            st.subheader("Pathway Management Overview")
            st.write(f"Total Pathways: {len(df_pathways)}")
            st.dataframe(df_pathways)
        
        with tab3:
            st.subheader("Control Guidance Overview")
            st.write(f"Total Guidance Documents: {len(df_guidance)}")
            st.dataframe(df_guidance)
        
    except Exception as e:
        st.error(f"Error generating visualizations: {str(e)}")
    
    finally:
        conn.close()

def pathway_management_visualization():
    st.title("Pathway Management Analytics")
    
    conn = get_db_connection()
    if not conn:
        st.error("Could not connect to database")
        return
    
    try:
        df_pathways = pd.read_sql_query("SELECT * FROM pathways", conn)
        
        if df_pathways.empty:
            st.info("No pathways found in the database")
            return
        
        # Display total pathways
        st.metric("Total Pathways", len(df_pathways))
        
        # Example visualization: Pathways by Description Length
        df_pathways['description_length'] = df_pathways['description'].apply(len)
        st.plotly_chart(
            px.histogram(
                df_pathways,
                x='description_length',
                nbins=20,
                title="Distribution of Pathway Description Lengths",
                labels={'description_length': 'Description Length'}
            ),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error generating pathway visualizations: {str(e)}")
    
    finally:
        conn.close()

def control_guidance_visualization():
    st.title("Control Guidance Analytics")
    
    conn = get_db_connection()
    if not conn:
        st.error("Could not connect to database")
        return
    
    try:
        df_guidance = pd.read_sql_query("SELECT * FROM guidance", conn)
        
        if df_guidance.empty:
            st.info("No guidance documents found in the database")
            return
        
        # Display total guidance documents
        st.metric("Total Guidance Documents", len(df_guidance))
        
        # Example visualization: Guidance by Category
        category_counts = df_guidance['category'].value_counts()
        st.plotly_chart(
            px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Guidance Distribution by Category"
            ),
            use_container_width=True
        )
        
    except Exception as e:
        st.error(f"Error generating guidance visualizations: {str(e)}")
    
    finally:
        conn.close()

def get_db_connection():
    """Establish a connection to the SQLite database."""
    try:
        conn = sqlite3.connect('data.db')
        return conn
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {str(e)}")
        return None

def save_risk(risk_data):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    try:
        if 'id' not in risk_data:
            risk_data['id'] = str(uuid.uuid4())
        
        c.execute("""
            INSERT OR REPLACE INTO risks (
                id, name, description, category, likelihood, impact,
                inherent_risk_score, residual_risk_score, status, owner,
                review_date, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            risk_data['id'],
            risk_data['name'],
            risk_data.get('description', ''),
            risk_data.get('category', ''),
            risk_data.get('likelihood', ''),
            risk_data.get('impact', ''),
            risk_data.get('inherent_risk_score', 0.0),
            risk_data.get('residual_risk_score', 0.0),
            risk_data.get('status', 'Open'),
            risk_data.get('owner', ''),
            risk_data.get('review_date', None),
            datetime.now()
        ))
        conn.commit()
        return risk_data['id']
    except sqlite3.Error as e:
        st.error(f"Error saving risk: {e}")
        return None
    finally:
        conn.close()

def get_risks():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    try:
        c.execute("SELECT * FROM risks ORDER BY created_at DESC")
        columns = [description[0] for description in c.description]
        risks = [dict(zip(columns, row)) for row in c.fetchall()]
        return risks
    except sqlite3.Error as e:
        st.error(f"Error fetching risks: {e}")
        return []
    finally:
        conn.close()

def delete_risk(risk_id):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    try:
        # Delete associated controls and assessments first
        c.execute("DELETE FROM controls WHERE risk_id = ?", (risk_id,))
        c.execute("DELETE FROM risk_assessments WHERE risk_id = ?", (risk_id,))
        c.execute("DELETE FROM risks WHERE id = ?", (risk_id,))
        conn.commit()
        return True
    except sqlite3.Error as e:
        st.error(f"Error deleting risk: {e}")
        return False
    finally:
        conn.close()

def risk_management_page():
    st.title("Risk Management")
    
    # Update only this part to get API key from .env
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key not found in .env file. Please make sure you have set GROQ_API_KEY in your .env file.")
        return
        
    # Initialize Groq client with env variable
    client = Groq(api_key=groq_api_key)
    
    # Sidebar for API Key
    if "decrypted_api_key" not in st.session_state:
        st.sidebar.text_input("Enter Groq API Key", type="password", key="api_key_input", 
                            on_change=lambda: st.session_state.update({"decrypted_api_key": st.session_state.api_key_input}))
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Document Analysis", "Risk Visualization"])
    
    with tab1:
        st.subheader("Risk Analysis")
        query = st.text_area("Enter your risk-related query:", height=100,
                            help="Ask about cybersecurity risks, compliance, or technology risk management.")
        
        if st.button("Analyze Risk"):
            if not query:
                st.warning("Please enter a query.")
                return
                
            with st.spinner("Analyzing with multiple expert agents..."):
                # Initialize conversation history if not exists
                if "conversation_history" not in st.session_state:
                    st.session_state.conversation_history = []
                
                # Get responses from all expert agents
                agent_responses = run_all_agents(
                    query, 
                    st.session_state.conversation_history,
                    []  # Empty document references for now
                )
                
                # Display individual expert responses in an expander
                with st.expander("View Individual Expert Analyses"):
                    for agent_name, response in agent_responses.items():
                        st.markdown(f"**{agent_name}**")
                        st.markdown(response)
                        st.divider()
                
                # Get aggregated response
                aggregated_response = aggregate_responses(
                    query,
                    agent_responses,
                    st.session_state.conversation_history,
                    []  # Empty document references for now
                )
                
                # Display aggregated response
                st.markdown("### Aggregated Analysis")
                st.markdown(aggregated_response)
                
                # Update conversation history
                st.session_state.conversation_history.extend([
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": aggregated_response}
                ])
    
    with tab2:
        st.subheader("Document Analysis")
        uploaded_files = st.file_uploader(
            "Upload documents for analysis (PDF, TXT, DOCX)", 
            accept_multiple_files=True,
            type=['pdf', 'txt', 'docx']
        )
        
        if uploaded_files:
            documents = []
            for file in uploaded_files:
                try:
                    # For simplicity, reading as text. You might want to add proper PDF/DOCX handling
                    content = file.read().decode('utf-8')
                    documents.append(content)
                except Exception as e:
                    st.error(f"Error reading file {file.name}: {str(e)}")
            
            if documents:
                # Initialize GraphRAG with uploaded documents
                rag = GraphRAG(documents)
                
                # Document query interface
                doc_query = st.text_input("Ask a question about your documents:")
                if doc_query:
                    with st.spinner("Searching documents..."):
                        response = rag.query(doc_query)
                        st.markdown(response)
    
    with tab3:
        st.subheader("Risk Visualization")
        if "conversation_history" in st.session_state and st.session_state.conversation_history:
            # Extract risk-related information from conversation history
            risk_docs = [
                msg["content"] 
                for msg in st.session_state.conversation_history 
                if msg["role"] == "assistant"
            ]
            
            if risk_docs:
                # Create graph visualization
                rag = GraphRAG(risk_docs)
                rag._add_edges()  # Add relationships between documents
                
                # Display the graph
                st.pyplot(rag.visualize_graph())
                
                # Add some metrics
                st.markdown("### Risk Analysis Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Risk Analyses", len(risk_docs))
                with col2:
                    st.metric("Connected Insights", len(list(rag.graph.edges())))
        else:
            st.info("No risk analysis history available for visualization. Start by analyzing some risks in the Risk Analysis tab.")

def issue_management_page():
    st.title("Issue Management")
    
    # Get API key from environment variable
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ API key not found in .env file. Please make sure you have set GROQ_API_KEY in your .env file.")
        return
        
    # Initialize Groq client
    client = Groq(api_key=groq_api_key)
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["View Issues", "Add Issues", "Search Issues"])
    
    # Tab 1: View Existing Issues
    with tab1:
        st.subheader("Existing Issues")
        
        # Read issues from database
        conn = sqlite3.connect('data.db')
        df_issues = pd.read_sql_query("""
            SELECT id, title, description, category, severity, status, created_at 
            FROM issues ORDER BY created_at DESC
        """, conn)
        conn.close()
        
        if not df_issues.empty:
            # Add filters
            col1, col2, col3 = st.columns(3)
            with col1:
                status_filter = st.multiselect(
                    "Filter by Status",
                    options=df_issues['status'].unique()
                )
            with col2:
                severity_filter = st.multiselect(
                    "Filter by Severity",
                    options=df_issues['severity'].unique()
                )
            with col3:
                category_filter = st.multiselect(
                    "Filter by Category",
                    options=df_issues['category'].unique()
                )
            
            # Apply filters
            filtered_df = df_issues.copy()
            if status_filter:
                filtered_df = filtered_df[filtered_df['status'].isin(status_filter)]
            if severity_filter:
                filtered_df = filtered_df[filtered_df['severity'].isin(severity_filter)]
            if category_filter:
                filtered_df = filtered_df[filtered_df['category'].isin(category_filter)]
            
            # Display filtered issues
            st.dataframe(
                filtered_df,
                hide_index=True,
                column_config={
                    "created_at": st.column_config.DatetimeColumn("Created At", format="D MMM YYYY, HH:mm"),
                }
            )
        else:
            st.info("No issues found in the database")
    
    # Tab 2: Add New Issues
    with tab2:
        st.subheader("Add New Issues")
        
        input_method = st.radio(
            "Choose input method:",
            ["Single Issue", "Bulk Upload"]
        )
        
        if input_method == "Single Issue":
            with st.form("new_issue_form"):
                title = st.text_input("Issue Title")
                description = st.text_area("Issue Description")
                category = st.selectbox(
                    "Category",
                    ["Technical", "Process", "Security", "Performance", "Other"]
                )
                severity = st.select_slider(
                    "Severity",
                    options=["Low", "Medium", "High", "Critical"]
                )
                status = st.selectbox(
                    "Status",
                    ["Open", "In Progress", "Under Review", "Resolved", "Closed"]
                )
                
                if st.form_submit_button("Add Issue"):
                    if title and description:
                        conn = sqlite3.connect('data.db')
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO issues (title, description, category, severity, status)
                            VALUES (?, ?, ?, ?, ?)
                        """, (title, description, category, severity, status))
                        conn.commit()
                        conn.close()
                        st.success("Issue added successfully!")
                    else:
                        st.error("Title and description are required")
        
        else:  # Bulk Upload
            uploaded_file = st.file_uploader(
                "Upload CSV or Excel file", 
                type=['csv', 'xlsx'],
                key="bulk_issue_upload"
            )
            
            if uploaded_file is not None:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    
                    st.write("Preview of uploaded data:")
                    st.dataframe(df.head())
                    
                    if st.button("Process and Add Issues"):
                        conn = sqlite3.connect('data.db')
                        df.to_sql('issues', conn, if_exists='append', index=False)
                        conn.close()
                        st.success(f"Successfully added {len(df)} issues to the database!")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    # Tab 3: Semantic Search
    with tab3:
        st.subheader("Search Issues")
        
        search_query = st.text_input("Enter your search query")
        
        if search_query:
            with st.spinner("Searching..."):
                # Get all issues from database
                conn = sqlite3.connect('data.db')
                df_issues = pd.read_sql_query("""
                    SELECT id, title, description, category, severity, status, created_at 
                    FROM issues
                """, conn)
                conn.close()
                
                # Prepare a simplified version of issues for the prompt
                simplified_issues = df_issues.apply(
                    lambda x: {
                        'id': x['id'],
                        'title': x['title'],
                        'description': x['description'][:200] if x['description'] else ''  # Limit description length
                    }, 
                    axis=1
                ).tolist()
                
                # Create search prompt
                search_prompt = f"""
                Search Query: {search_query}
                
                Find relevant issues from the list below. For each matching issue:
                1. Assign a relevance score (0-100)
                2. Briefly explain why it matches (max 50 words)
                
                Return JSON format:
                {{
                    "total_matches": number,
                    "matches": [
                        {{
                            "issue_id": id,
                            "relevance_score": number,
                            "explanation": "text"
                        }}
                    ]
                }}
                
                Issues:
                {json.dumps(simplified_issues[:20])}  # Limit to top 20 most recent issues
                """
                
                try:
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": search_prompt}],
                        model="llama-3.2-90b-vision-preview",
                        temperature=0.1,
                        max_tokens=1000
                    )
                    
                    # Parse the response
                    try:
                        results = json.loads(response.choices[0].message.content)
                        
                        # Display summary
                        st.metric("Matching Issues Found", results["total_matches"])
                        
                        # Sort matches by relevance score
                        results["matches"].sort(key=lambda x: x["relevance_score"], reverse=True)
                        
                        # Display each matching issue
                        for match in results["matches"]:
                            issue_id = match["issue_id"]
                            issue_data = df_issues[df_issues['id'] == issue_id].iloc[0]
                            
                            with st.expander(f"Issue: {issue_data['title']} (Relevance: {match['relevance_score']}%)"):
                                # Issue details
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown(f"**Category:** {issue_data['category']}")
                                with col2:
                                    st.markdown(f"**Severity:** {issue_data['severity']}")
                                with col3:
                                    st.markdown(f"**Status:** {issue_data['status']}")
                                
                                # Description and match explanation
                                st.markdown("**Description:**")
                                st.markdown(issue_data['description'])
                                
                                st.markdown("**Why this matches:**")
                                st.markdown(match['explanation'])
                                
                                # Created date
                                st.markdown(f"**Created:** {pd.to_datetime(issue_data['created_at']).strftime('%Y-%m-%d %H:%M')}")
                    
                    except json.JSONDecodeError:
                        st.error("Error parsing search results. Please try a different search query.")
                
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")

def main():
    # Initialize session state for navigation if it doesn't exist
    if 'page' not in st.session_state:
        st.session_state.page = "Pathway Analysis"
    
    st.sidebar.title("Navigation")
    
    # Risk Analysis Section with unique keys
    st.sidebar.header("Risk Analysis")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Pathway Analysis", key="sidebar_pathway_analysis_1"):
            st.session_state.page = "Pathway Analysis"
    with col2:
        if st.button("Pathway Management", key="sidebar_pathway_mgmt_1"):
            st.session_state.page = "Pathway Management"
        
    # Control Analysis Section with unique keys
    st.sidebar.header("Control Analysis")
    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button("Control Gap Analysis", key="sidebar_control_analysis_1"):
            st.session_state.page = "Control Gap Analysis"
    with col4:
        if st.button("Control Guidance", key="sidebar_control_guidance_1"):
            st.session_state.page = "Control Guidance"
    
    # Issue Management Section
    st.sidebar.header("Issue Management")
    if st.sidebar.button("Issue Management", key="sidebar_issue_mgmt_1"):
        st.session_state.page = "Issue Management"
    
    # Analytics Section
    st.sidebar.header("Analytics")
    if st.sidebar.button("Issue Analytics", key="sidebar_analytics"):
        st.session_state.page = "Issue Analytics"
    if st.sidebar.button("Pathway Management Analytics", key="sidebar_pathway_mgmt_analytics"):
        st.session_state.page = "Pathway Management Analytics"
    if st.sidebar.button("Control Guidance Analytics", key="sidebar_control_guidance_analytics"):
        st.session_state.page = "Control Guidance Analytics"
    
    # Add Risk Management button in sidebar
    st.sidebar.header("Risk Management")
    if st.sidebar.button("Risk Management", key="risk_mgmt"):
        st.session_state.page = "Risk Management"
    
    # Add to your page routing
    if st.session_state.page == "Risk Management":
        risk_management_page()
    
    # Initialize database
    init_db()
    
    # Page content based on selection
    if st.session_state.page == "Pathway Analysis":
        st.title("Technology Issue Classifier")
        st.write("Upload your file (CSV or Excel) containing technology issues:")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'], key="file_uploader_1")
        
        if uploaded_file is not None:
            try:
                processed_df = process_file(uploaded_file, os.getenv('GROQ_API_KEY'))
                if processed_df is not None:
                    # Create a temporary file to save the processed data
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                        processed_df.to_excel(tmp.name, index=False)
                        
                        # Provide download button
                        with open(tmp.name, 'rb') as f:
                            st.download_button(
                                label="Download Processed File",
                                data=f,
                                file_name="processed_issues.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    # Display preview of processed data
                    st.write("Preview of processed data:")
                    st.dataframe(processed_df)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    elif st.session_state.page == "Control Gap Analysis":
        st.title("Control Gap Analysis")
        
        st.markdown("### 1. Upload Current Controls")
        uploaded_files = st.file_uploader(
            "Upload current controls document",
            type=['pdf', 'xlsx', 'xls', 'csv', 'docx'],
            accept_multiple_files=True,
            key="control_uploader_1",
            help="Upload your existing control documentation"
        )
        
        if uploaded_files:
            try:
                processed_df = process_controls_file(uploaded_files, os.getenv('GROQ_API_KEY'))
                if processed_df is not None:
                    # Create Excel file for download
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
                        processed_df.to_excel(tmp.name, index=False)
                        
                        # Provide download button
                        with open(tmp.name, 'rb') as f:
                            st.download_button(
                                "📥 Download Analysis Results",
                                data=f,
                                file_name="control_gap_analysis.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                st.write("Detailed error:")
                st.code(str(e))
    
    elif st.session_state.page == "Control Guidance":
        guidance_management_page()  # Function to manage control guidance
    
    elif st.session_state.page == "Issue Management":
        issue_management_page()
    
    elif st.session_state.page == "Issue Analytics":
        visualization_page()
    
    elif st.session_state.page == "Pathway Management":
        pathway_management_page()  # Function to manage pathways
    
    elif st.session_state.page == "Pathway Management Analytics":
        pathway_management_visualization()
    
    elif st.session_state.page == "Control Guidance Analytics":
        control_guidance_visualization()

if __name__ == "__main__":
    main()