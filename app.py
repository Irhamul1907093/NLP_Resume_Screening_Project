import streamlit as st
import pickle
import re
import nltk
from io import BytesIO
from PyPDF2 import PdfReader

# Load pre-trained models
clf = pickle.load(open('clf.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidf.pkl', 'rb'))


# Function to clean resume text
def cleanResume(txt):
    cleanText = re.sub(r'http\S+', ' ', txt)
    cleanText = re.sub(r'\b(RT|cc)\b', ' ', cleanText)
    cleanText = re.sub(r'#\S+', ' ', cleanText)
    cleanText = re.sub(r'@\S+', ' ', cleanText)
    cleanText = re.sub(r'[{}]'.format(re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText).strip()
    return cleanText


# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(BytesIO(pdf_file))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Main function for Streamlit app
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader("Upload CV", type=['txt', 'pdf'])

    if upload_file is not None:
        try:
            if upload_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(upload_file.read())
            elif upload_file.type == "text/plain":
                resume_bytes = upload_file.read()
                resume_text = resume_bytes.decode('utf-8')
            else:
                st.error("Unsupported file format.")
                return
        except Exception as e:
            st.error(f"Error processing file: {e}")
            return

        # Clean and process resume text
        cleaned_resume = cleanResume(resume_text)
        input_features = tfidf_vectorizer.transform([cleaned_resume])
        prediction_id = clf.predict(input_features)[0]
        #st.write(prediction_id)

        category_mapping = {
            6: "Data Science",
            12: "HR",
            0: "Advocate",
            1: "Arts",
            24: "Web designing",
            16: "Mechanical Engineering",
            22: "Sales",
            14: "Health and Fitness",
            5: "Civil Engineering",
            15: "Java Developer",
            4: "Business Analyst",
            21: "SAP Developer",
            2: "Automation Testing",
            11: "Electrical Engineering",
            18: "Operations Manager",
            20: "Python Developer",
            8: "DevOps Engineering",
            17: "Network Security Engineer",
            19: "PMO",
            7: "Database",
            13: "Hadoop",
            10: "ETL Developer",
            9: "DotNet Developer",
            3: "Blockchain",
            23: "Testing"
        }
        category_name = category_mapping.get(prediction_id, "Unknown")

        st.write("Predicted Category : ",category_name)

if __name__ == "__main__":
    main()