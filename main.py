import pandas as pd
import streamlit as st 

from utils import *

backgroundPattern = """

<style>
[data-testid="stAppViewContainer"] {
background-color: #FFFFFF;
opacity: 1;
background-image: radial-gradient(#D1D1D1 0.75px, #FFFFFF 0.75px);
background-size: 15px 15px;
}
</style>
"""
st.markdown(backgroundPattern, unsafe_allow_html=True)

st.write("""
# RESUME SCREENING TOOL
""")


tab1, tab2 = st.tabs(['Rank', 'Classify'])

with tab1:
    st.header('Upload Job Description')
    
    # Option to paste or type the job description
    job_description_option = st.radio("Select how to provide the job description:", ("Paste or type the job description", "Upload a file"))
    if job_description_option == "Paste or type the job description":
        job_description_text = st.text_area("Paste or type the job description here:", key='job_description_text')
        uploadedJobDescriptionRnk = BytesIO(job_description_text.encode('utf-8'))  # Convert text to BytesIO object
    else:
        uploadedJobDescriptionRnk = st.file_uploader('Upload Job Description', type=['txt', 'xlsx', 'pdf', 'doc', 'docx'], key='upload-jd-rnk')

    # Upload Resumes
    uploadedResumeRnk = st.file_uploader('Upload Resumes', type=['xlsx', 'pdf', 'doc', 'txt', 'docx'], key='upload-resume-rnk')

    if all([uploadedJobDescriptionRnk, uploadedResumeRnk]):
        isButtonDisabledRnk = False
    else:
        st.session_state.processRank = False
        isButtonDisabledRnk = True


    if 'processRank' not in st.session_state:
        st.session_state.processRank = False

    st.button('Match Resumes', on_click = clickRank, disabled = isButtonDisabledRnk, key = 'process-rnk')

    if st.session_state.processRank:
        st.divider()
        st.header('Output')
        jobDescriptionRnk = uploadedJobDescriptionRnk.read().decode('utf-8')
        resumeRnk = pd.read_excel(uploadedResumeRnk)

        if 'Resume' in resumeRnk.columns:
            resumeRnk = rankResumes(jobDescriptionRnk, resumeRnk)
            with st.expander('View Job Description'):
                st.write(jobDescriptionRnk)
            currentRnk = filterDataframeRnk(resumeRnk)
            st.dataframe(currentRnk, use_container_width = True, hide_index = True)
            xlsxRnk = convertDfToXlsx(currentRnk)
            st.download_button(label='Save Current Output as XLSX', data = xlsxRnk, file_name = 'Resumes_ranked.xlsx')
        else:
            st.error("""
            #### Oops! Something went wrong.
            Check whether You have Uploaded in the given Format Only:)
            """)
with tab2:
    st.header('Classify Here...')
    uploadedResumeClf = st.file_uploader('Upload Resumes', type = ['xlsx', 'pdf', 'doc', 'txt', 'docx'], key = 'upload-resume-clf')

    if uploadedResumeClf is not None:
        isButtonDisabledClf = False
    else:
        st.session_state.processClf = False 
        isButtonDisabledClf = True

    if 'processClf' not in st.session_state:
        st.session_state.processClf = False

    st.button('Start Processing', on_click = clickClassify, disabled = isButtonDisabledClf, key = 'process-clf')

    if st.session_state.processClf:
        st.divider()
        st.header('Output')
        resumeClf = pd.read_excel(uploadedResumeClf)
        
        if 'Resume' in resumeClf.columns:
            resumeClf = classifyResumes(resumeClf)
            with st.expander('View Bar Chart'):
                barChart = createBarChart(resumeClf)
                st.altair_chart(barChart, use_container_width = True)
            currentClf = filterDataframeClf(resumeClf)
            st.dataframe(currentClf, use_container_width = True, hide_index = True)
            xlsxClf = convertDfToXlsx(currentClf)
            st.download_button(label='Save Current Output as XLSX', data = xlsxClf, file_name = 'Resumes_categorized.xlsx')
        else:
            st.error("""
            #### Oops! Something went wrong.
            Check whether You have Uploaded in the given Format Only:)
            """)
