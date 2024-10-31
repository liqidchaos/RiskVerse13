import streamlit as st
import pandas as pd
import sqlite3
from groq import Groq
from dotenv import load_dotenv
import os
import tempfile

# Load environment variables from .env file
load_dotenv()

def init_db():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    
    # Check if table exists with correct schema
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
    # Update prompt to include descriptions
    prompt = "Analyze the following technology issue and classify it into one of these specific risk pathways:\n\n"
    for pathway, description in pathways:
        prompt += f"{pathway}:\n{description}\n\n"
    
    prompt += f"Issue: {issue_description}\n\nSelect the most appropriate pathway from the list above. Response should be exactly one of the pathway names (without description):"
    
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.2-11b-vision-preview",
            temperature=0.1,
            max_tokens=100
        )
        
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
            with st.progress(0):
                for idx, row in df.iterrows():
                    pathway = classify_issue(client, str(row[issue_column]), pathways)
                    df.at[idx, 'Risk_Pathway'] = pathway
                    st.progress((idx + 1) / len(df))
            
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
                            
                            response = client.chat.completions.create(
                                messages=[{"role": "user", "content": prompt}],
                                model="llama-3.2-11b-vision-preview",
                                temperature=0.1,
                                max_tokens=300
                            )
                            
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

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["File Processing", "Pathway Management"])
    
    # Initialize database
    init_db()
    
    if page == "File Processing":
        st.title("Technology Issue Classifier")
        st.write("Upload your file (CSV or Excel) containing technology issues:")
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
        
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
    
    else:
        pathway_management_page()

if __name__ == "__main__":
    main() 