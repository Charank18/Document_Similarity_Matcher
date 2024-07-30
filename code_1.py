import os
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocMatcher:
    def __init__(self):
        self.database = {}
        self.vectorizer = TfidfVectorizer()

    def extract_text(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                pdf = PdfReader(file)
                text = ''
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    def add_to_database(self, invoice_id, pdf_path):
        text = self.extract_text(pdf_path)
        if text:
            self.database[invoice_id] = text

    def preprocess_database(self):
        texts = list(self.database.values())
        if texts:
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            print("No documents")

    def find_most_similar(self, input_pdf_path):
        input_text = self.extract_text(input_pdf_path)
        if not input_text:
            return None, 0

        input_vector = self.vectorizer.transform([input_text])
        
        similarities = cosine_similarity(input_vector, self.tfidf_matrix)[0]
        most_similar_index = np.argmax(similarities)
        most_similar_invoice_id = list(self.database.keys())[most_similar_index]
        similarity_score = similarities[most_similar_index]
        
        return most_similar_invoice_id, similarity_score

matcher = DocMatcher()

def add_all_invoices_to_database(matcher, directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            filepath = os.path.join(directory_path, filename)
            matcher.add_to_database(filename, filepath)
    print("All invoices have been added.")

directory_path = r"C:\Users\chara\Desktop\matching document project\train"
add_all_invoices_to_database(matcher, directory_path)

matcher.preprocess_database()

most_similar_id, similarity_score = matcher.find_most_similar(r"C:\Users\chara\Desktop\matching document project\test\invoice_77098.pdf")

if most_similar_id:
    print(f"Most similar invoice: {most_similar_id}")
    print(f"Similarity score: {similarity_score:.2f}")
else:
    print("Error processing input invoice.")
