# chatbot.py - Complete Working Chatbot
import re
import random
from datetime import datetime

class MedicalChatbot:
    def __init__(self):
        self.session_data = {}
        
        # Symptom keywords for extraction
        self.symptom_keywords = {
            'fever': ['fever', 'high temperature', 'hot', 'chills', 'sweating', 'high temp', 'temperature'],
            'cough': ['cough', 'coughing', 'dry cough', 'wet cough', 'phlegm'],
            'fatigue': ['tired', 'fatigue', 'exhausted', 'low energy', 'weak', 'sleepy', 'no energy', 'lethargic', 'drained'],
            'difficulty_breathing': ['breathing', 'shortness of breath', 'breathless', 'difficulty breathing', 'chest tightness', 'wheezing', "can't breathe", 'out of breath'],
            'headache': ['headache', 'head pain', 'migraine', 'head hurting', 'pain in head', 'headache'],
            'sore_throat': ['sore throat', 'throat pain', 'scratchy throat', 'painful throat', 'sore throat'],
            'body_ache': ['body ache', 'muscle pain', 'body pain', 'joint pain', 'muscle sore', 'body hurting', 'bodyache', 'aches'],
            'nausea': ['nausea', 'vomiting', 'feel sick', 'stomach upset', 'queasy', 'throwing up'],
            'runny_nose': ['runny nose', 'stuffy nose', 'congestion', 'nasal', 'blocked nose', 'stuffy']
        }
        
        # Responses for different intents
        self.responses = {
            'greeting': [
                "Hello! 👋 I'm Cura.\n\nHow are you feeling today?",
                "Hi there! 🏥 Tell me what's bothering you.",
                "Welcome! 💬 I'm here to help. What symptoms are you experiencing?"
            ],
            
            'feeling_bad': [
                "I'm sorry you're not feeling well. 😔\n\nCan you tell me more about your symptoms? (e.g., fever, cough, headache)",
                "That's unfortunate. 😟 Let me help you.\n\nWhat specific symptoms are you having?",
                "I understand. 🤝 Please describe what you're feeling - fever, cough, headache, etc."
            ],
            
            'feeling_good': [
                "That's great to hear! 😊\n\nIs there anything health-related I can help you with today?",
                "Awesome! 🎉 Let me know if you need any health advice."
            ],
            
            'symptoms_found': [
                "✅ I understand. You're experiencing: {symptoms}\n\n💡 Here's what helps:\n{advice}\n\n🔍 Would you like me to analyze these symptoms? (yes/no)",
                "📋 Got it! Symptoms detected: {symptoms}\n\n🏥 Quick advice:\n{advice}\n\n🎯 Should I prepare a full health assessment? (yes/no)"
            ],
            
            'no_symptoms': [
                "I couldn't identify specific symptoms. 🤔\n\nPlease tell me what you're feeling. For example:\n• 'I have fever and cough'\n• 'Headache and tiredness'\n• 'Sore throat with body ache'",
                "Help me understand better. 💭\n\nTry saying something like: 'I have fever and headache' or 'I'm feeling very tired with a cough'"
            ],
            
            'ask_more': [
                "Any other symptoms? (yes/no)",
                "Is there anything else bothering you?",
                "Do you have any other symptoms like fever, cough, or fatigue?"
            ],
            
            'more_symptoms': [
                "Okay! Please tell me what other symptoms you have.",
                "Sure! What else are you feeling?",
                "Go ahead, describe your other symptoms."
            ],
            
            'ready_to_analyze': [
                "Perfect! 🎯 I have all your symptoms.\n\n📋 Click the 'Apply to Form' button below, then click 'Analyze Health Risk' for your complete report!",
                "Great! ✅ Your symptoms are ready for analysis.\n\n👉 Use the 'Apply to Form' button to transfer them to the assessment form."
            ],
            
            'decline_analyze': [
                "No problem! 😊\n\nI'm still here if you need health advice or want to describe more symptoms.",
                "Okay! Let me know if you change your mind or need any help."
            ],
            
            'advice_for_symptom': [
                "💊 For {symptom}:\n\n{advice}\n\nAnything else I can help with?",
                "Here's what helps with {symptom}:\n\n{advice}\n\nNeed advice for anything else?"
            ],
            
            'help': [
            "💡 I'm Cura, here's how I can help:\n\n• Tell me symptoms: 'I have fever and cough'\n• Ask for advice: 'What helps with headache?'\n• Say 'analyze' for full assessment\n• Say 'help' for this menu\n• Say 'bye' to end\n\nWhat would you like to do?",
            "📋 Cura Commands:\n\n1. Describe symptoms: 'fever, cough, tired'\n2. Ask: 'what to do for headache?'\n3. Say 'analyze' for prediction\n4. Say 'clear' to reset\n\nTry it out!"
            ],
            
            'farewell': [
                "Take care! 🏥\n\n⚠️ Remember: I'm an AI assistant. Please consult a real doctor for serious concerns.\n\nGet well soon! 💪",
                "Goodbye! 👋 Stay healthy!\n\nCome back anytime you need health advice."
            ],
            
            'clear': [
                "Chat history cleared! 🗑️\n\nLet's start over. How are you feeling today?",
                "Reset complete! ✅\n\nTell me about your symptoms."
            ],
            
            'fallback': [
                "I'm not sure I understood. 🤔\n\nTry:\n• Telling me your symptoms (fever, cough, etc.)\n• Saying 'help' for options\n• Asking 'what helps with fever?'\n• Saying 'bye' to exit",
                
                "Can you rephrase that? 💭\n\nExample: 'I have fever and headache' or 'What should I do for cough?'"
            ]
        }
        
        # Detailed advice for each symptom
        self.advice_map = {
            'fever': "🌡️ Monitor temperature every 4-6 hours\n💧 Drink 8-10 glasses of water daily\n🛁 Take lukewarm baths\n🛌 Get adequate rest\n💊 Use fever reducers if >101°F",
            'cough': "🍯 Drink warm honey ginger tea\n❄️ Avoid cold foods and drinks\n💨 Use a humidifier\n🧂 Gargle with salt water\n😷 Wear mask to prevent spread",
            'fatigue': "😴 Get 7-8 hours of quality sleep\n🥬 Eat iron-rich foods (spinach, legumes)\n☕ Take short breaks throughout the day\n🚶 Light exercise like walking\n💆 Practice stress reduction",
            'difficulty_breathing': "🚨 SEEK MEDICAL ATTENTION if severe\n💺 Sit upright to ease breathing\n💨 Use prescribed inhaler if available\n🧘 Practice deep breathing\n📞 Call emergency if worsens",
            'headache': "💆 Rest in dark quiet room\n💧 Stay hydrated\n❄️ Apply cold or warm compress\n📱 Limit screen time\n🧘 Gentle neck stretches",
            'sore_throat': "🧂 Gargle with warm salt water\n🍵 Drink warm tea with honey\n🤫 Rest your voice\n🍬 Use throat lozenges\n❄️ Avoid cold foods",
            'body_ache': "🛀 Rest and relax\n🔥 Apply warm compress\n🧘 Gentle stretching exercises\n💧 Stay hydrated\n💊 OTC pain relief if needed",
            'nausea': "🍚 Eat bland foods (rice, bananas, toast)\n💧 Small sips of water\n🍵 Ginger tea helps\n👃 Avoid strong smells\n🛌 Rest after eating",
            'runny_nose': "💨 Use saline nasal spray\n💧 Stay hydrated\n💨 Use a humidifier\n🛌 Get plenty of rest\n🚿 Warm shower helps"
        }
    
    def extract_symptoms(self, text):
        """Extract symptoms from user text"""
        text = text.lower()
        detected = []
        
        for symptom, keywords in self.symptom_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    detected.append(symptom)
                    break
        
        # Remove duplicates while preserving order
        unique = []
        for s in detected:
            if s not in unique:
                unique.append(s)
        return unique
    
    def get_advice(self, symptoms):
        """Get combined advice for multiple symptoms"""
        if not symptoms:
            return "👨‍⚕️ Monitor your symptoms, rest well, stay hydrated, and consult a doctor if needed."
        
        advice_list = []
        for symptom in symptoms[:3]:  # Limit to 3 symptoms
            if symptom in self.advice_map:
                advice_list.append(self.advice_map[symptom])
        
        return "\n\n".join(advice_list)
    
    def get_single_advice(self, symptom):
        """Get advice for a single symptom"""
        return self.advice_map.get(symptom, "👨‍⚕️ Rest, stay hydrated, and consult a doctor if symptoms persist.")
    
    def process_message(self, user_message, session_id):
        """Main method to process user messages"""
        
        # Initialize session
        if session_id not in self.session_data:
            self.session_data[session_id] = {
                'history': [],
                'detected_symptoms': [],
                'awaiting_response': False,
                'last_question': None,
                'message_count': 0
            }
        
        session = self.session_data[session_id]
        user_message = user_message.lower().strip()
        session['history'].append(user_message)
        session['message_count'] += 1
        
        # ========== INTENT DETECTION ==========
        
        # 1. Greeting detection
        greetings = ['hi', 'hello', 'hey', 'namaste', 'good morning', 'good afternoon', 'good evening', 'hola']
        if any(g in user_message for g in greetings) and session['message_count'] <= 2:
            return random.choice(self.responses['greeting'])
        
        # 2. Farewell detection
        farewells = ['bye', 'goodbye', 'exit', 'quit', 'see you', 'take care', 'cya']
        if any(f in user_message for f in farewells):
            return random.choice(self.responses['farewell'])
        
        # 3. Help detection
        if user_message == 'help' or 'help' in user_message.split():
            return random.choice(self.responses['help'])
        
        # 4. Clear detection
        if user_message == 'clear' or 'reset' in user_message or 'start over' in user_message:
            return random.choice(self.responses['clear'])
        
        # 5. Feeling detection (not well, sick, etc.)
        feeling_bad = ['not well', 'not good', 'sick', 'unwell', 'feeling bad', 'feeling sick', 
                       'not feeling well', 'ill', 'under the weather', 'terrible', 'awful']
        if any(f in user_message for f in feeling_bad):
            return random.choice(self.responses['feeling_bad'])
        
        # 6. Feeling good detection
        feeling_good = ['good', 'great', 'fine', 'well', 'excellent', 'perfect', 'awesome', 
                        'feeling good', 'feeling great', 'doing well']
        if any(f in user_message for f in feeling_good) and 'not' not in user_message:
            return random.choice(self.responses['feeling_good'])
        
        # 7. Yes/No response handling
        if user_message in ['yes', 'yeah', 'yep', 'sure', 'ok', 'okay', 'of course']:
            if session.get('awaiting_response'):
                session['awaiting_response'] = False
                return random.choice(self.responses['ready_to_analyze'])
            else:
                return "Okay! 😊 What would you like to do? (Say 'help' for options)"
        
        if user_message in ['no', 'nope', 'not really', 'no thanks', 'nah']:
            if session.get('awaiting_response'):
                session['awaiting_response'] = False
                return random.choice(self.responses['decline_analyze'])
            else:
                return "Alright! 😊 Let me know if you need anything else."
        
        # 8. Analyze request
        analyze_words = ['analyze', 'assessment', 'predict', 'check', 'evaluate', 'diagnose']
        if any(a in user_message for a in analyze_words):
            if session['detected_symptoms']:
                return random.choice(self.responses['ready_to_analyze'])
            else:
                return "I don't have any symptoms to analyze yet. 😕\n\nPlease tell me what symptoms you're experiencing first."
        
        # 9. Advice question (e.g., "what helps with fever?")
        advice_triggers = ['what helps', 'what to do', 'how to treat', 'remedy for', 'cure for', 'medicine for']
        if any(t in user_message for t in advice_triggers):
            for symptom in self.symptom_keywords.keys():
                if symptom in user_message:
                    advice = self.get_single_advice(symptom)
                    return random.choice(self.responses['advice_for_symptom']).format(
                        symptom=symptom.capitalize(),
                        advice=advice
                    )
        
        # 10. Extract symptoms from message
        detected_symptoms = self.extract_symptoms(user_message)
        
        if detected_symptoms:
            # Merge with existing symptoms
            all_symptoms = list(set(session['detected_symptoms'] + detected_symptoms))
            session['detected_symptoms'] = all_symptoms
            session['awaiting_response'] = True
            
            symptoms_str = ', '.join(all_symptoms)
            advice = self.get_advice(all_symptoms)
            
            return random.choice(self.responses['symptoms_found']).format(
                symptoms=symptoms_str,
                advice=advice
            )
        
        # 11. Check for "more symptoms" response
        if session['detected_symptoms'] and any(w in user_message for w in ['more', 'anything else', 'additional']):
            return random.choice(self.responses['more_symptoms'])
        
        # 12. Fallback for unrecognized messages
        return random.choice(self.responses['fallback'])
    
    def get_detected_symptoms(self, session_id):
        """Return detected symptoms for a session"""
        if session_id in self.session_data:
            return self.session_data[session_id]['detected_symptoms']
        return []
    
    def clear_session(self, session_id):
        """Clear session data"""
        if session_id in self.session_data:
            del self.session_data[session_id]
            return True
        return False

# Create global chatbot instance
chatbot = MedicalChatbot()