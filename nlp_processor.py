# nlp_processor.py - NLP Symptom Detection with Warning System
import re

class NLPSymptomProcessor:
    def __init__(self):
        # Symptom keywords mapping
        self.symptom_keywords = {
            'Fever': ['fever', 'high temperature', 'hot', 'chills', 'sweating', 'high temp', 'temperature', 'warm body'],
            'Cough': ['cough', 'coughing', 'dry cough', 'wet cough', 'phlegm', 'constant cough', 'hacking cough'],
            'Fatigue': ['tired', 'fatigue', 'exhausted', 'low energy', 'weak', 'sleepy', 'no energy', 'lethargic', 'drained', 'worn out'],
            'Difficulty_Breathing': ['breathing', 'shortness of breath', 'breathless', 'difficulty breathing', 'chest tightness', 'wheezing', "can't breathe", 'out of breath', 'gasping'],
            'Headache': ['headache', 'head pain', 'migraine', 'head hurting', 'pain in head', 'throbbing head'],
            'SoreThroat': ['sore throat', 'throat pain', 'scratchy throat', 'painful throat', 'itchy throat'],
            'BodyAche': ['body ache', 'muscle pain', 'body pain', 'joint pain', 'muscle sore', 'body hurting', 'bodyache', 'aches', 'muscle ache'],
            'RunnyNose': ['runny nose', 'stuffy nose', 'congestion', 'nasal', 'blocked nose', 'sniffles', 'running nose']
        }
        
        # Common symptom variations for suggestions
        self.symptom_suggestions = {
            'tooth pain': 'Headache (tooth pain may indicate sinus issue)',
            'stomach pain': 'Nausea (not in dataset, consider consulting doctor)',
            'nausea': 'Nausea (not in dataset - please see doctor)',
            'vomiting': 'Vomiting (not in dataset - please see doctor)',
            'dizziness': 'Dizziness (not in dataset - please see doctor)',
            'chest pain': 'Difficulty Breathing (chest pain can be serious - seek medical help)',
            'back pain': 'Body Ache (back pain as body ache)',
            'sneezing': 'Runny Nose (sneezing often accompanies allergies)'
        }
    
    def extract_symptoms(self, text):
        """Extract known symptoms from user text"""
        text = text.lower()
        detected_symptoms = []
        
        for symptom, keywords in self.symptom_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    detected_symptoms.append(symptom)
                    break
        
        # Remove duplicates while preserving order
        unique_symptoms = []
        for s in detected_symptoms:
            if s not in unique_symptoms:
                unique_symptoms.append(s)
        
        return unique_symptoms
    
    def find_unknown_symptoms(self, text):
        """Find words/phrases that might be symptoms but aren't in our dataset"""
        text = text.lower()
        
        # Common medical terms to check for
        medical_terms = [
            'pain', 'ache', 'hurt', 'sore', 'swelling', 'nausea', 'vomiting',
            'dizzy', 'dizziness', 'rash', 'itching', 'numbness', 'bleeding',
            'weakness', 'stomach', 'chest', 'back', 'neck', 'shoulder', 'knee',
            'tooth', 'ear', 'eye', 'throat', 'nose', 'mouth', 'skin'
        ]
        
        unknown = []
        words = re.findall(r'\b\w+\b', text)
        
        for i, word in enumerate(words):
            if word in medical_terms:
                # Look for phrases (e.g., "tooth pain", "stomach ache")
                phrase = word
                if i + 1 < len(words) and words[i + 1] in ['pain', 'ache', 'hurt', 'sore']:
                    phrase = word + ' ' + words[i + 1]
                elif i > 0 and words[i - 1] in ['tooth', 'stomach', 'chest', 'back', 'neck']:
                    phrase = words[i - 1] + ' ' + word
                
                # Check if this phrase is not already a known symptom
                is_known = False
                for symptom, keywords in self.symptom_keywords.items():
                    if phrase in keywords or word in keywords:
                        is_known = True
                        break
                
                if not is_known and phrase not in unknown:
                    unknown.append(phrase)
        
        return list(set(unknown))
    
    def get_suggestion(self, unknown_symptom):
        """Get suggestion for unknown symptom"""
        for key, suggestion in self.symptom_suggestions.items():
            if key in unknown_symptom.lower():
                return suggestion
        
        # Check for partial matches
        unknown_lower = unknown_symptom.lower()
        if 'pain' in unknown_lower:
            return "Try selecting from: Headache, Body Ache, or consult a doctor"
        elif 'stomach' in unknown_lower or 'nausea' in unknown_lower:
            return "Stomach issues aren't in our dataset. Please select from available symptoms or consult a doctor"
        elif 'chest' in unknown_lower:
            return "⚠️ Chest pain can be serious. Please select 'Difficulty Breathing' or seek medical attention"
        else:
            return f"'{unknown_symptom}' is not in our symptom list. Available: Fever, Cough, Fatigue, Difficulty Breathing, Headache, Sore Throat, Body Ache, Runny Nose"
    
    def process_message(self, text):
        """Process user message and return symptoms with warnings"""
        detected = self.extract_symptoms(text)
        unknown = self.find_unknown_symptoms(text)
        
        warnings = []
        suggestions = []
        
        for uk in unknown:
            suggestion = self.get_suggestion(uk)
            warnings.append(f"⚠️ '{uk}' not recognized")
            suggestions.append(f"💡 {suggestion}")
        
        return {
            'detected_symptoms': detected,
            'unknown_symptoms': unknown,
            'warnings': warnings,
            'suggestions': suggestions,
            'has_known': len(detected) > 0,
            'has_unknown': len(unknown) > 0
        }

# Create global instance
nlp_processor = NLPSymptomProcessor()