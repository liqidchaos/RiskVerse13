import streamlit as st
import pandas as pd
import sqlite3
from openai import OpenAI
from dotenv import load_dotenv
import os
import tempfile
import io
import PyPDF2
import docx
import re
from io import StringIO
import json

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client with error checking
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("No OpenAI API key found. Please set OPENAI_API_KEY in your .env file")

client = OpenAI(api_key=api_key)


# Set OpenAI API key

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

def classify_issue(issue_description, pathways):
    # Update prompt to include descriptions
    prompt = "Analyze the following technology issue and classify it into one of these specific risk pathways:\n\n"
    for pathway, description in pathways:
        prompt += f"{pathway}:\n{description}\n\n"

    prompt += f"Issue: {issue_description}\n\nSelect the most appropriate pathway from the list above. Response should be exactly one of the pathway names (without description):"

    try:
        response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}],
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=100)

        return response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
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
                    'Issue ID': parts[0] if len(parts) > 0 else '',
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
    pathways = get_pathways()

    if not pathways:
        st.error("Please add some pathways in the Pathway Management page first!")
        return None

    # Initialize session state for tracking revisions if not exists
    if 'current_df' not in st.session_state:
        st.session_state.current_df = None
        st.session_state.revision_count = 0

    # Initial processing
    if st.session_state.current_df is None:
        try:
            # Read and preprocess the file
            df = read_file(file)
            if df is None:
                return None

            # Display column preview
            st.write("Preview of detected columns:")
            st.dataframe(df.head(2))

            # Try to automatically detect issue column
            issue_column = find_issue_column(df)

            # If can't automatically detect, let user select
            if issue_column is None:
                st.warning("Could not automatically detect the issue description column. Please select it manually:")
                issue_column = st.selectbox("Select the column containing issue descriptions:", df.columns)
            else:
                st.success(f"Automatically detected issue description column: {issue_column}")
                # Allow user to override if needed
                if st.checkbox("Use a different column?"):
                    issue_column = st.selectbox("Select the column containing issue descriptions:", df.columns)

            # Add columns for pathways, suggestions, and rationale
            df['Risk_Pathway'] = ''
            df['Suggested_Pathway'] = ''
            df['Model_Rationale'] = ''

            # First pass: AI classification
            progress_bar = st.progress(0)
            for idx, row in df.iterrows():
                pathway = classify_issue(str(row[issue_column]), pathways)
                df.at[idx, 'Risk_Pathway'] = pathway
                progress_bar.progress((idx + 1) / len(df))

            st.session_state.current_df = df
            st.session_state.issue_column = issue_column

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            return None

    # Display current state
    st.write(f"Revision Round: {st.session_state.revision_count + 1}")

    # Create column configuration for data editor
    column_config = {
        'Suggested_Pathway': st.column_config.SelectboxColumn(
            'Suggested_Pathway',
            help='Select an alternative pathway if needed',
            width='medium',
            options=[p[0] for p in pathways],
            required=False
        ),
        'Model_Rationale': st.column_config.TextColumn(
            'Model_Rationale',
            help='AI explanation for classification',
            width='large',
        )
    }

    # Display editable dataframe
    edited_df = st.data_editor(
        st.session_state.current_df,
        disabled=["Risk_Pathway", "Model_Rationale", st.session_state.issue_column],
        column_config=column_config,
        hide_index=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Reprocess with Suggestions"):
            with st.spinner("Reprocessing with user feedback..."):
                try:
                    for idx, row in edited_df.iterrows():
                        suggestion = row['Suggested_Pathway']
                        if suggestion and suggestion.strip():
                            prompt = f"""Reconsider this classification:
                            Issue: {row[st.session_state.issue_column]}
                            Initial classification: {row['Risk_Pathway']}
                            User suggested: {suggestion}
                            
                            Provide your final classification and detailed rationale in this format:
                            Classification: [chosen pathway]
                            Rationale: [detailed explanation considering both the initial classification and user suggestion]
                            """

                            response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}],
                            model="gpt-4o-mini",
                            temperature=0.1,
                            max_tokens=300)

                            result = response.choices[0].message.content.strip()

                            if 'Classification:' in result and 'Rationale:' in result:
                                classification = result.split('Classification:')[1].split('Rationale:')[0].strip()
                                rationale = result.split('Rationale:')[1].strip()
                            else:
                                classification = result
                                rationale = "Response format was unexpected"

                            edited_df.at[idx, 'Risk_Pathway'] = classification
                            edited_df.at[idx, 'Model_Rationale'] = rationale

                    st.session_state.current_df = edited_df
                    st.session_state.revision_count += 1
                    st.rerun()

                except Exception as e:
                    st.error(f"Error during reprocessing: {str(e)}")

    with col2:
        if st.button("Finalize Classifications"):
            # Create final output
            final_df = st.session_state.current_df.copy()

            # Clean up the dataframe
            final_df = final_df.drop('Suggested_Pathway', axis=1)

            # Reset session state
            st.session_state.current_df = None
            st.session_state.revision_count = 0

            return final_df

    return None

def pathway_management_page():
    st.title("Pathway Management")

    # Add multiple pathways
    st.write("Enter pathways and descriptions (in the format 'Pathway Name:\\n\\nDescription: description text')")
    new_pathways = st.text_area("Enter pathways:", height=300)

    if st.button("Add Pathways"):
        if new_pathways:
            # Split by double newline to separate different pathways
            pathway_blocks = [block.strip() for block in new_pathways.split('\n\n') if block.strip()]

            for block in pathway_blocks:
                if ':' in block:
                    # Split the first line (pathway name) from description
                    lines = block.split('\n', 1)
                    pathway = lines[0].replace(':', '').strip()

                    # Extract description if it exists
                    description = ''
                    if len(lines) > 1 and 'Description:' in lines[1]:
                        description = lines[1].replace('Description:', '').strip()

                    save_pathway(pathway, description)

            st.success(f"Added {len(pathway_blocks)} pathways")
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
    """Extract text from various file formats"""
    file_extension = file.name.split('.')[-1].lower()

    try:
        if file_extension == 'pdf':
            return extract_text_from_pdf(file)
        elif file_extension == 'docx':
            return extract_text_from_docx(file)
        elif file_extension == 'txt':
            return file.getvalue().decode('utf-8')
        elif file_extension in ['csv', 'xlsx', 'xls']:
            if file_extension == 'csv':
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            return df
        else:
            # Instead of using textract, return an error for unsupported formats
            st.error(f"Unsupported file format: {file_extension}")
            return None
    except Exception as e:
        st.error(f"Error extracting text from file: {str(e)}")
        return None

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
                      'risk_considerations', 'expected_evidence', 'measurement_criteria']:
                current_item[key] = value

    # Add last item if exists
    if current_item:
        items.append(current_item)

    return items

def extract_regulatory_content(file):
    """Extract specific regulatory content from documents with focus on risk management and controls"""
    try:
        # Extract text based on file type
        text = extract_text_from_file(file)

        if text is None:
            st.error("Could not extract text from file")
            return []

        # Show preview of extracted text
        st.write("Preview of extracted text:")
        st.code(text[:500] + "..." if len(text) > 500 else text)

        # Define specialized extraction prompt
        prompt = """Analyze this regulatory document section and extract control requirements.
For each control or requirement found, provide the following structure:

Control ID: [unique identifier]
Category: [Risk/Governance/Compliance/Security]
Control Objective: [brief objective]
Requirements: [specific requirements]
Risk Considerations: [associated risks]
Expected Evidence: [required documentation]
Measurement Criteria: [success criteria]

Extract as many controls as you can find in the text, maintaining this exact structure."""

        # Process document in manageable chunks
        chunk_size = 2000  # Reduced chunk size for better processing
        if isinstance(text, str):
            text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        else:
            # Handle DataFrame case
            text_chunks = [text.to_string()]

        extracted_items = []
        progress_bar = st.progress(0)
        
        for idx, chunk in enumerate(text_chunks):
            try:
                response = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a control framework analyst. Extract and structure control requirements from regulatory documents."},
                        {"role": "user", "content": f"{prompt}\n\nDocument text:\n{chunk}"}
                    ],
                    model="gpt-4o",
                    temperature=0.1,
                    max_tokens=1500
                )

                result = response.choices[0].message.content.strip()
                items = parse_regulatory_response(result)
                extracted_items.extend(items)
                
                # Update progress
                progress_bar.progress((idx + 1) / len(text_chunks))
                
            except Exception as e:
                st.warning(f"Error processing chunk {idx + 1}: {str(e)}")
                continue

        if not extracted_items:
            st.error("No structured controls found in the document")
            st.write("Please ensure your document contains control requirements in a readable format")
            return []

        # Show preview of extracted items
        st.success(f"Successfully extracted {len(extracted_items)} controls")
        st.write("Preview of first extracted control:")
        st.write(extracted_items[0] if extracted_items else "No controls found")

        return extracted_items

    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return []

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
            requirements.append({
                'ID': str(row[req_id_col]), 
                'Requirement': str(row[req_text_col])
            })

    else:
        # Handle unstructured text
        st.write("Extracted text preview:")
        st.text_area("Preview", content[:500] + "...", height=200, disabled=True)

        # Extract requirements using AI model
        prompt = """Extract specific control requirements from this text. 
        Format each requirement as:
        ID: [unique identifier]
        Requirement: [specific control requirement text]
        
        Text to analyze:
        """ + content

        try:
            response = client.chat.completions.create(
                messages=[{
                    "role": "system",
                    "content": "You are a control framework expert. Extract specific, actionable control requirements from regulatory documents."
                }, {
                    "role": "user",
                    "content": prompt
                }],
                model="gpt-4o",
                temperature=0.1,
                max_tokens=1500
            )

            # Parse the response
            requirements = []
            current_req = {}
            
            for line in response.choices[0].message.content.split('\n'):
                line = line.strip()
                if line.startswith('ID:'):
                    if current_req:
                        requirements.append(current_req)
                        current_req = {}
                    req_id = line.split(':')[1].strip()
                    current_req['ID'] = req_id
                elif line.startswith('Requirement:'):
                    req_text = line.split(':')[1].strip()
                    current_req['Requirement'] = req_text

        except Exception as e:
            st.error(f"Error processing uploaded guidance: {str(e)}")
            return None

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

        requirements_json = json.dumps(requirements)

        if exists:
            # Update existing guidance
            c.execute("""UPDATE guidance 
                        SET category = ?, description = ?, requirements = ?
                        WHERE name = ?""",
                     (category, description, requirements_json, name))
        else:
            # Insert new guidance
            c.execute("""INSERT INTO guidance 
                        (name, category, description, requirements)
                        VALUES (?, ?, ?, ?)""",
                     (name, category, description, requirements_json))

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
    guidance_list = []
    for name, category, description, reqs in guidance:
        try:
            reqs_list = json.loads(reqs) if reqs else []
            guidance_list.append((name, category, description, reqs_list))
        except json.JSONDecodeError:
            st.warning(f"Invalid data format for guidance: {name}")
            # Use empty list for invalid requirements
            guidance_list.append((name, category, description, []))
    conn.close()
    return guidance_list

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

                    if st.button("Save Guidance"):
                        if guidance_name:
                            save_guidance(guidance_name, category, description, requirements)
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
        requirements_text = st.text_area(
            "Requirements",
            height=300,
            placeholder="IS.B.1: Financial institutions should implement a comprehensive information security program...\nIS.B.2: The security program should include risk assessment processes..."
        )

        requirements = []
        for line in requirements_text.strip().split('\n'):
            if ':' in line:
                req_id, req_text = line.split(':', 1)
                requirements.append({'ID': req_id.strip(), 'Requirement': req_text.strip()})

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
                requirements_text = template_data["requirements"]

                st.write("**Template Preview:**")
                st.write(f"Category: {category}")
                st.write("Description:")
                st.write(description)
                st.write("Sample Requirements:")
                st.write(requirements_text[:500] + "...")

                if st.checkbox("Customize Template"):
                    guidance_name = st.text_input("Guidance Name", value=guidance_name)
                    category = st.selectbox("Category", ["Financial Services", "Healthcare", "Technology", "Privacy", "Cybersecurity", "Other"], index=["Financial Services", "Healthcare", "Technology", "Privacy", "Cybersecurity", "Other"].index(category))
                    description = st.text_area("Description", value=description)
                    requirements_text = st.text_area("Requirements", value=requirements_text, height=300)

                requirements = []
                for line in requirements_text.strip().split('\n'):
                    if ':' in line:
                        req_id, req_text = line.split(':', 1)
                        requirements.append({'ID': req_id.strip(), 'Requirement': req_text.strip()})

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
                # Add delete button at the top of each guidance
                if st.button("🗑️ Delete Guidance", key=f"delete_{name}", type="secondary"):
                    if st.warning(f"Are you sure you want to delete '{name}'?"):
                        delete_guidance(name)
                        st.success(f"Deleted {name}")
                        st.rerun()
                
                st.write("**Description:**")
                st.write(description)
                st.write("**Requirements:**")
                for req in reqs:
                    st.write(f"- {req['ID']}: {req['Requirement']}")
                
                # Export button at the bottom
                if st.button("📥 Export", key=f"export_{name}"):
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
    df = pd.DataFrame(requirements)

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

def display_preview_controls(controls, title):
    """Display controls in a table format with full width"""
    if not controls:
        st.warning("No controls to display")
        return

    # Convert controls to DataFrame
    df = pd.DataFrame(controls)
    
    # Rename columns for better display
    column_mapping = {
        'control_id': 'Control ID',
        'category': 'Category',
        'control_objective': 'Objective',
        'requirements': 'Requirements',
        'risk_considerations': 'Risks',
        'expected_evidence': 'Evidence',
        'measurement_criteria': 'Measurement'
    }
    df = df.rename(columns=column_mapping)
    
    # Truncate long text fields
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(lambda x: str(x)[:200] + '...' if len(str(x)) > 200 else str(x))
    
    st.markdown(f"**{title}**")
    
    # Use container for full width
    with st.container():
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=400  # Increased height for better visibility
        )

def display_analysis_results(gaps, target_requirements):
    """Display the gap analysis results with metrics and suggestions"""
    try:
        # Calculate compliance metrics
        total_reqs = len(target_requirements)
        covered_reqs = sum(1 for gap in gaps if gap['coverage_status'].lower() == 'covered')
        partial_reqs = sum(1 for gap in gaps if gap['coverage_status'].lower() == 'partial')
        missing_reqs = sum(1 for gap in gaps if gap['coverage_status'].lower() == 'missing')

        # Display compliance scores
        st.markdown("## Compliance Scores")
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

        # Display detailed analysis
        st.markdown("## Detailed Analysis")
        tabs = st.tabs(["All Requirements", "Gaps", "Partial Coverage", "Full Coverage"])
        
        with tabs[0]:  # All Requirements
            st.dataframe(pd.DataFrame(gaps), use_container_width=True)
        
        with tabs[1]:  # Gaps
            missing = [g for g in gaps if g['coverage_status'].lower() == 'missing']
            if missing:
                st.dataframe(pd.DataFrame(missing), use_container_width=True)
            else:
                st.info("No gaps found")
        
        with tabs[2]:  # Partial Coverage
            partial = [g for g in gaps if g['coverage_status'].lower() == 'partial']
            if partial:
                st.markdown("### Controls Needing Improvement")
                for gap in partial:
                    with st.expander(f"🔍 {gap['requirement_id']}: {gap['requirement_text'][:100]}..."):
                        st.markdown("**Current Controls:**")
                        st.write(", ".join(gap['matching_controls']))
                        
                        st.markdown("**📝 Suggested Control Improvements:**")
                        st.write(gap['control_improvements'])
                        
                        st.markdown("**➕ New Controls Needed:**")
                        st.write(gap['new_controls'])
                        
                        st.markdown("**🔧 Implementation Notes:**")
                        st.write(gap['implementation_notes'])
            else:
                st.info("No partially covered requirements found")
        
        with tabs[3]:  # Full Coverage
            covered = [g for g in gaps if g['coverage_status'].lower() == 'covered']
            if covered:
                st.dataframe(pd.DataFrame(covered), use_container_width=True)
            else:
                st.info("No fully covered requirements found")

        # Export option
        st.markdown("## Export Results")
        st.download_button(
            "📥 Download Analysis Report",
            data=pd.DataFrame(gaps).to_csv(index=False),
            file_name="gap_analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

    except Exception as e:
        st.error(f"Error displaying analysis results: {str(e)}")

def control_gap_analysis_page():
    st.title("Control Gap Analysis")
    
    # Add some spacing after title
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("### 1. Upload Current Controls")
    current_file = st.file_uploader(
        "Upload current controls document",
        type=['pdf', 'xlsx', 'xls', 'csv', 'docx'],
        help="Upload your existing control documentation"
    )
    
    if current_file:
        st.info("Processing current controls...")
        current_controls = extract_regulatory_content(current_file)
        if current_controls:
            st.success(f"Successfully extracted {len(current_controls)} controls")
            with st.expander("📋 Preview Current Controls", expanded=True):
                display_preview_controls(current_controls[:5], "Sample of Extracted Controls")
        else:
            st.error("No controls could be extracted from the document")

    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Framework Selection Section
    st.markdown("### 2. Select Target Framework")
    guidance_list = get_guidance()
    if not guidance_list:
        st.error("Please add some control guidance documents first!")
        return
    
    framework_names = [name for name, _, _, _ in guidance_list]
    selected_framework = st.selectbox(
        "Select target framework",
        framework_names,
        help="Choose the framework to compare against"
    )
    
    if selected_framework:
        target_framework = next((framework for framework in guidance_list if framework[0] == selected_framework), None)
        if target_framework:
            st.success(f"Selected framework has {len(target_framework[3])} requirements")
            with st.expander("📋 Preview Framework Requirements", expanded=True):
                display_preview_controls(target_framework[3][:5], "Sample of Framework Requirements")

    # Add spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # Analysis Section
    if current_file and selected_framework:
        st.markdown("### 3. Perform Analysis")
        
        # Use full width for the button container
        if st.button("🔍 Run Gap Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing gaps..."):
                try:
                    current_controls = extract_regulatory_content(current_file)
                    target_framework = next((framework for framework in guidance_list if framework[0] == selected_framework), None)
                    
                    if not current_controls:
                        st.error("No controls could be extracted from the current controls document")
                        return
                    
                    if not target_framework or not target_framework[3]:
                        st.error("No requirements found in the selected framework")
                        return
                    
                    # Preview section using full width container
                    with st.container():
                        st.markdown("### Preview of extracted text:")
                        st.dataframe(
                            pd.DataFrame(current_controls),
                            use_container_width=True,
                            height=300
                        )
                    
                    st.success(f"Successfully extracted {len(current_controls)} controls")
                    
                    # Preview first control in a more readable format
                    with st.container():
                        st.markdown("### Preview of first extracted control:")
                        control = current_controls[0]
                        
                        # Create a DataFrame for better display
                        control_df = pd.DataFrame({
                            'Field': ['Control ID', 'Category', 'Objective', 'Requirements'],
                            'Value': [
                                control.get('control_id', 'N/A'),
                                control.get('category', 'N/A'),
                                control.get('control_objective', 'N/A'),
                                control.get('requirements', 'N/A')
                            ]
                        })
                        st.dataframe(
                            control_df,
                            use_container_width=True,
                            hide_index=True,
                            height=200
                        )
                    
                    # Perform gap analysis
                    gaps = analyze_control_gaps(current_controls, target_framework[3])
                    
                    if gaps:
                        st.markdown("<br>", unsafe_allow_html=True)
                        display_gap_analysis(gaps, filter_status="all")
                    else:
                        st.error("No results generated from the analysis")
                
                except Exception as e:
                    st.error(f"Error during analysis: {str(e)}")
                    st.write("Detailed error:")
                    st.code(str(e))

def analyze_control_gaps(current_controls, target_requirements):
    """Analyze gaps between current controls and target requirements"""
    gaps = []
    
    try:
        progress_bar = st.progress(0)
        for idx, target_req in enumerate(target_requirements):
            analysis_prompt = f"""Analyze this control requirement against existing controls:

Target Requirement: {target_req['Requirement']}

Current Controls:
{json.dumps(current_controls, indent=2)}

Provide a detailed analysis in this exact format:
COVERAGE_STATUS: [Covered/Partial/Missing]
MATCHING_CONTROLS: [List control IDs that match]
GAP_DESCRIPTION: [Describe specific gaps]
CONTROL_IMPROVEMENTS: [For partial coverage, suggest specific wording changes to existing controls]
NEW_CONTROLS: [Suggest new controls needed]
IMPLEMENTATION_NOTES: [Provide implementation guidance]"""

            response = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a control framework expert. Provide detailed gap analysis and specific control improvements."},
                    {"role": "user", "content": analysis_prompt}
                ],
                model="gpt-4o",  # Using GPT-4 for better analysis
                temperature=0.1,
                max_tokens=1000
            )

            # Parse the response
            analysis = {
                'requirement_id': target_req.get('ID', 'N/A'),
                'requirement_text': target_req.get('Requirement', ''),
                'coverage_status': 'Unknown',
                'matching_controls': [],
                'gap_description': '',
                'control_improvements': '',
                'new_controls': '',
                'implementation_notes': ''
            }

            # Parse response line by line
            for line in response.choices[0].message.content.split('\n'):
                if line.startswith('COVERAGE_STATUS:'):
                    analysis['coverage_status'] = line.split(':', 1)[1].strip()
                elif line.startswith('MATCHING_CONTROLS:'):
                    controls = line.split(':', 1)[1].strip()
                    analysis['matching_controls'] = [c.strip() for c in controls.strip('[]').split(',') if c.strip()]
                elif line.startswith('GAP_DESCRIPTION:'):
                    analysis['gap_description'] = line.split(':', 1)[1].strip()
                elif line.startswith('CONTROL_IMPROVEMENTS:'):
                    analysis['control_improvements'] = line.split(':', 1)[1].strip()
                elif line.startswith('NEW_CONTROLS:'):
                    analysis['new_controls'] = line.split(':', 1)[1].strip()
                elif line.startswith('IMPLEMENTATION_NOTES:'):
                    analysis['implementation_notes'] = line.split(':', 1)[1].strip()

            gaps.append(analysis)
            progress_bar.progress((idx + 1) / len(target_requirements))

    except Exception as e:
        st.error(f"Error in gap analysis: {str(e)}")
        return []

    return gaps

def display_gap_analysis(gaps, filter_status="all"):
    """Display gap analysis results with enhanced improvement suggestions"""
    filtered_gaps = [g for g in gaps if filter_status == "all" or g['coverage_status'] == filter_status]
    
    if not filtered_gaps:
        st.info(f"No {filter_status.lower()} requirements found.")
        return

    for gap in filtered_gaps:
        with st.expander(f"**{gap['requirement_id']}**: {gap['requirement_text'][:100]}..."):
            status_color = {
                'Covered': 'green',
                'Partial': 'orange',
                'Missing': 'red',
                'Not Applicable': 'grey'
            }.get(gap['coverage_status'], 'black')
            
            st.markdown(f"**Status:** :{status_color}[{gap['coverage_status']}]")
            
            if gap['matching_controls']:
                st.write("**Matching Controls:**")
                for control in gap['matching_controls']:
                    st.write(f"- {control}")
            
            if gap['gap_description']:
                st.write("**Gap Description:**")
                st.write(gap['gap_description'])
            
            # Show improvement suggestions for partial coverage
            if gap['coverage_status'] == 'Partial':
                if gap['control_improvements']:
                    st.write("**📝 Suggested Control Improvements:**")
                    st.write(gap['control_improvements'])
                
                if gap['new_controls']:
                    st.write("**➕ Recommended New Controls:**")
                    st.write(gap['new_controls'])
                
                if gap['implementation_notes']:
                    st.write("**🔧 Implementation Considerations:**")
                    st.write(gap['implementation_notes'])

# Update main() function to use the new page
def main():
    # Initialize session state for navigation if it doesn't exist
    if 'page' not in st.session_state:
        st.session_state.page = "Pathway Analysis"

    st.sidebar.title("Navigation")

    # Risk Analysis Section
    st.sidebar.header("Risk Analysis")
    if st.sidebar.button("Pathway Analysis", key="pathway_analysis"):
        st.session_state.page = "Pathway Analysis"
    if st.sidebar.button("Pathway Management", key="pathway_management"):
        st.session_state.page = "Pathway Management"

    # Control Analysis Section
    st.sidebar.header("Control Analysis")
    if st.sidebar.button("Control Gap Analysis", key="control_analysis"):
        st.session_state.page = "Control Gap Analysis"
    if st.sidebar.button("Control Guidance", key="control_guidance"):
        st.session_state.page = "Control Guidance"

    # Initialize database
    init_db()

    if st.session_state.page == "Pathway Analysis":
        st.title("Technology Issue Classifier")
        st.write("Upload your file (CSV or Excel) containing technology issues:")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])

        if uploaded_file is not None:
            try:
                processed_df = process_file(uploaded_file, None)
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
        control_gap_analysis_page()

    elif st.session_state.page == "Control Guidance":
        guidance_management_page()

    else:
        pathway_management_page()

if __name__ == "__main__":
    main()
