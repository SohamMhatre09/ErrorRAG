import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import numpy as np
import json
import re
import google.generativeai as genai
from flask import Flask, request, jsonify

app = Flask(__name__)

class ErrorClassifier:
    def __init__(self, csv_file):
        """Initialize the ErrorClassifier with a CSV file containing error templates"""
        try:
            # Read the CSV file
            self.df = pd.read_csv(csv_file)
            print(f"Loaded CSV file with {len(self.df)} rows")
            
            # Clean and validate the EventTemplate column
            self.df['EventTemplate'] = self.df['EventTemplate'].astype(str)
            self.df = self.df[self.df['EventTemplate'].str.strip().str.len() > 0]
            print(f"After cleaning, {len(self.df)} valid templates remain")
            
            # Initialize vectorizer
            self.vectorizer = TfidfVectorizer(
                strip_accents='unicode',
                lowercase=True,
                analyzer='word',
                stop_words='english',
                min_df=1
            )
            
            # Fit the vectorizer first
            print("Fitting vectorizer...")
            self.vectorizer = self.vectorizer.fit(self.df['EventTemplate'])
            print(f"Vectorizer fitted successfully. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
            
            # Then transform the data
            print("Transforming templates...")
            self.tfidf_matrix = self.vectorizer.transform(self.df['EventTemplate'])
            print(f"Templates transformed successfully. Matrix shape: {self.tfidf_matrix.shape}")
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            raise
    
    def preprocess_text(self, text):
        """Clean and preprocess input text"""
        try:
            if not isinstance(text, str):
                text = str(text)
            text = text.strip()
            if not text:
                raise ValueError("Empty input text")
            return text.replace('<*>', '.*')
        except Exception as e:
            print(f"Error in text preprocessing: {str(e)}")
            raise
    
    def fuzzy_match_score(self, str1, str2):
        """Calculate fuzzy matching score between two strings"""
        try:
            return fuzz.ratio(str1, str2) / 100.0
        except Exception as e:
            print(f"Error in fuzzy matching: {str(e)}")
            return 0.0
    
    def semantic_similarity(self, query):
        """Calculate semantic similarity using TF-IDF"""
        try:
            query_vector = self.vectorizer.transform([query])
            return cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        except Exception as e:
            print(f"Error in semantic similarity calculation: {str(e)}")
            return np.zeros(len(self.df))
    
    def combined_similarity(self, query, word_weight=0.8, semantic_weight=0.2):
        """Calculate combined similarity scores"""
        try:
            query = self.preprocess_text(query)
            fuzzy_scores = np.array([
                self.fuzzy_match_score(query, event) 
                for event in self.df['EventTemplate']
            ])
            semantic_scores = self.semantic_similarity(query)
            
            if len(fuzzy_scores) != len(semantic_scores):
                raise ValueError("Score length mismatch")
                
            return word_weight * fuzzy_scores + semantic_weight * semantic_scores
        except Exception as e:
            print(f"Error in combined similarity calculation: {str(e)}")
            raise
    
    def find_top_matches(self, query, n=1):
        """Find top N matching templates"""
        try:
            combined_scores = self.combined_similarity(query)
            if len(combined_scores) == 0:
                raise ValueError("No valid scores calculated")
                
            top_indices = combined_scores.argsort()[-n:][::-1]
            return self.df.iloc[top_indices]
        except Exception as e:
            print(f"Error finding top matches: {str(e)}")
            raise

class ErrorAnalysisSystem:
    def __init__(self, csv_file, api_key):
        """Initialize the Error Analysis System"""
        try:
            self.classifier = ErrorClassifier(csv_file)
            genai.configure(api_key=api_key)
            self.genai_model = genai.GenerativeModel('gemini-1.5-pro')
            print("ErrorAnalysisSystem initialized successfully")
        except Exception as e:
            print(f"Error initializing ErrorAnalysisSystem: {str(e)}")
            raise

    def generate_ai_response(self, error_message, model_output):
        """Generate AI response using Gemini"""
        prompt = f"""
        # Error Analysis Prompt Template

        You are an AI assistant specialized in analyzing error messages and providing detailed insights. 
        Given an error message and its classification, provide a comprehensive analysis in JSON format.

        ## Input Error Message: "{error_message}"

        ## Model Output:
        {model_output}

        Provide a detailed analysis in JSON format with the following fields:
        - analysis (core issue description)
        - classification (error type)
        - severity (error level)
        - likelyCause (probable causes)
        - suggestedSolution (array of solutions)
        - tips (array of preventive measures)
        - actionableRecommendations (array of specific actions)
        """

        try:
            result = self.genai_model.generate_content(prompt)
            cleaned_result = result.text.strip()
            json_match = re.search(r'\{.*\}', cleaned_result, re.DOTALL)
            
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError as e:
                    print(f'Failed to parse AI response as JSON: {e}')
                    return {"error": "Invalid JSON format from AI"}
            else:
                print('No JSON object found in AI response')
                return {"error": "No JSON object found in AI response"}
                
        except Exception as e:
            print(f"Error generating AI response: {str(e)}")
            return {"error": f"AI generation error: {str(e)}"}

    def process_error(self, error_message):
        """Process an error message and return analysis"""
        try:
            top_matches = self.classifier.find_top_matches(error_message)
            model_output = top_matches[['Level', 'Component', 'EventTemplate', 'type']].to_dict(orient='records')
            ai_response = self.generate_ai_response(error_message, json.dumps(model_output))
            
            return {
                "errorMessage": error_message,
                "modelClassification": model_output,
                "aiAnalysis": ai_response
            }
        except Exception as e:
            print(f"Error processing error message: {str(e)}")
            return {"error": f"Processing error: {str(e)}"}

# Initialize the system
csv_file = './combined_error.csv'  # Update this path to your CSV file
api_key = "AIzaSyCi_rpYtGy-ms-Io7_2fz0CpjUhCIoBFlE"    # Replace with your actual API key

try:
    system = ErrorAnalysisSystem(csv_file, api_key)
    print("System initialization complete")
except Exception as e:
    print(f"Failed to initialize system: {str(e)}")
    exit(1)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

@app.route('/analyze', methods=['POST'])
def analyze_error():
    """Endpoint to analyze error messages"""
    try:
        data = request.json
        if not data or 'error_message' not in data:
            return jsonify({"error": "No error_message provided"}), 400
        
        error_message = data['error_message']
        if not error_message or not error_message.strip():
            return jsonify({"error": "Empty error message"}), 400
            
        result = system.process_error(error_message)
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in analyze endpoint: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)