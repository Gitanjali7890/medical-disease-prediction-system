// static/js/chatbot.js - Complete Working Chatbot

class MedicalChatbot {
    constructor() {
        this.sessionId = this.getSessionId();
        this.isOpen = false;
        this.detectedSymptoms = [];
        this.init();
    }
    
    getSessionId() {
        let sessionId = localStorage.getItem('chatbot_session_id');
        if (!sessionId) {
            sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('chatbot_session_id', sessionId);
        }
        return sessionId;
    }
    
    init() {
        this.createChatbotUI();
        this.attachEventListeners();
        this.addWelcomeMessage();
    }
    
    createChatbotUI() {
        // Check if chatbot already exists
        if (document.querySelector('.chatbot-container')) return;
        
        const chatbotHTML = `
            <div class="chatbot-container">
                <button class="chatbot-toggle" id="chatbotToggle">
                    💬
                </button>
                <div class="chatbot-window" id="chatbotWindow">
                    <div class="chatbot-header">
                        <h4>
                            <span>🩺</span> Cura - Your Health Assistant
                        </h4>
                        <button class="chatbot-close" id="chatbotClose">×</button>
                    </div>
                    <div class="chatbot-messages" id="chatbotMessages">
                        <!-- Messages will appear here -->
                    </div>
                    <div class="chatbot-actions">
                        <button class="chatbot-action-btn" id="clearChatBtn">
                            🗑️ Clear Chat
                        </button>
                        <button class="chatbot-action-btn primary" id="applySymptomsBtn">
                            📋 Apply to Form
                        </button>
                    </div>
                    <div class="chatbot-input-area">
                        <input type="text" class="chatbot-input" id="chatbotInput" 
                               placeholder="Describe your symptoms... (e.g., 'I have fever')">
                        <button class="chatbot-send" id="chatbotSend">
                            ➤
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.insertAdjacentHTML('beforeend', chatbotHTML);
    }
    
    attachEventListeners() {
        const toggleBtn = document.getElementById('chatbotToggle');
        const closeBtn = document.getElementById('chatbotClose');
        const sendBtn = document.getElementById('chatbotSend');
        const input = document.getElementById('chatbotInput');
        const clearBtn = document.getElementById('clearChatBtn');
        const applyBtn = document.getElementById('applySymptomsBtn');
        
        if (toggleBtn) toggleBtn.addEventListener('click', () => this.toggleWindow());
        if (closeBtn) closeBtn.addEventListener('click', () => this.toggleWindow());
        if (sendBtn) sendBtn.addEventListener('click', () => this.sendMessage());
        if (input) input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.sendMessage();
        });
        if (clearBtn) clearBtn.addEventListener('click', () => this.clearChat());
        if (applyBtn) applyBtn.addEventListener('click', () => this.applySymptomsToForm());
    }
    
    toggleWindow() {
        const window = document.getElementById('chatbotWindow');
        this.isOpen = !this.isOpen;
        if (window) window.classList.toggle('open', this.isOpen);
        
        if (this.isOpen) {
            const input = document.getElementById('chatbotInput');
            if (input) input.focus();
        }
    }
    
    addWelcomeMessage() {
        setTimeout(() => {
            this.addMessage(
                "Hello! 👋 I'm Cura, your personal health assistant.\n\n💡 Tell me about your symptoms.\n\nFor example:\n• 'I have fever and cough'\n• 'Headache and body ache'\n• 'Feeling tired'",
                'bot'
            );
        }, 500);
    }
    
    addMessage(text, sender) {
        const messagesContainer = document.getElementById('chatbotMessages');
        if (!messagesContainer) return;
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}`;
        messageDiv.innerHTML = `<div class="message-content">${this.formatMessage(text)}</div>`;
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    formatMessage(text) {
        return text.replace(/\n/g, '<br>');
    }
    
    showTypingIndicator() {
        const messagesContainer = document.getElementById('chatbotMessages');
        if (!messagesContainer) return;
        
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot';
        typingDiv.id = 'typingIndicator';
        typingDiv.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        messagesContainer.appendChild(typingDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }
    
    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) {
            indicator.remove();
        }
    }
    
    async sendMessage() {
        const input = document.getElementById('chatbotInput');
        const message = input.value.trim();
        
        if (!message) return;
        
        this.addMessage(message, 'user');
        input.value = '';
        this.showTypingIndicator();
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId
                })
            });
            
            const data = await response.json();
            this.hideTypingIndicator();
            
            if (data.success) {
                this.addMessage(data.response, 'bot');
                if (data.detected_symptoms && data.detected_symptoms.length > 0) {
                    this.detectedSymptoms = data.detected_symptoms;
                }
            } else {
                this.addMessage("Sorry, I'm having trouble connecting. Please try again.", 'bot');
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage("Network error. Please check if the server is running.", 'bot');
        }
    }
    
    async clearChat() {
        try {
            await fetch('/api/chat/clear', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ session_id: this.sessionId })
            });
            
            const messagesContainer = document.getElementById('chatbotMessages');
            if (messagesContainer) {
                messagesContainer.innerHTML = '';
            }
            
            this.sessionId = this.getSessionId();
            this.detectedSymptoms = [];
            this.addWelcomeMessage();
            
        } catch (error) {
            console.error('Error clearing chat:', error);
        }
    }
    
    async applySymptomsToForm() {
        if (!this.detectedSymptoms || this.detectedSymptoms.length === 0) {
            this.addMessage("❌ No symptoms detected yet.\n\nPlease describe your symptoms first. Example: 'I have fever and headache'", 'bot');
            return;
        }
        
        this.addMessage(`📋 Applying detected symptoms to form: ${this.detectedSymptoms.join(', ')}`, 'bot');
        
        try {
            const response = await fetch('/api/apply_symptoms', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ symptoms: this.detectedSymptoms })
            });
            
            const data = await response.json();
            
            if (data.success) {
                for (const [field, value] of Object.entries(data.form_data)) {
                    const checkbox = document.getElementById(field);
                    if (checkbox) {
                        checkbox.checked = true;
                    }
                }
                this.addMessage("✅ Symptoms applied to the assessment form!\n\n👉 Click 'Analyze Health Risk' to get your prediction.", 'bot');
            } else {
                this.addMessage("❌ Could not apply symptoms. Please try again.", 'bot');
            }
        } catch (error) {
            console.error('Error applying symptoms:', error);
            this.addMessage("❌ Error applying symptoms.", 'bot');
        }
    }
}

// Initialize chatbot when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.chatbot = new MedicalChatbot();
});