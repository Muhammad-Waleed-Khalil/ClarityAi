import streamlit as st
from groq import Groq
import pandas as pd
from datetime import datetime
import json
import os
import uuid

class AdvancedChatbot:
    def __init__(self, api_key, knowledge_file):
        self.client = Groq(api_key=api_key)
        self.knowledge_base = self.load_knowledge_base(knowledge_file)
        
        if 'current_chat_id' not in st.session_state:
            st.session_state.current_chat_id = str(uuid.uuid4())
        if 'chats' not in st.session_state:
            st.session_state.chats = {
                st.session_state.current_chat_id: {
                    'title': 'New Chat',
                    'messages': [],
                    'mode': 'custom',
                    'timestamp': datetime.now().isoformat()
                }
            }
        if 'chat_mode' not in st.session_state:
            st.session_state.chat_mode = 'custom'

    def load_knowledge_base(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                knowledge_entries = file.read().split('\n\n')
            return knowledge_entries
        except FileNotFoundError:
            st.error(f"Knowledge file not found: {file_path}")
            return []
        except Exception as e:
            st.error(f"Error loading knowledge base: {e}")
            return []

    def create_new_chat(self):
        chat_id = str(uuid.uuid4())
        st.session_state.chats[chat_id] = {
            'title': 'New Chat',
            'messages': [],
            'mode': st.session_state.chat_mode,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.current_chat_id = chat_id
        return chat_id

    def save_chats(self):
        try:
            os.makedirs("chat_history", exist_ok=True)
            filename = f"chats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(f"chat_history/{filename}", 'w') as f:
                json.dump(st.session_state.chats, f)
            return filename
        except Exception as e:
            st.error(f"Error saving chats: {e}")
            return None

    def load_chats(self, filename):
        try:
            with open(f"chat_history/{filename}", 'r') as f:
                st.session_state.chats = json.load(f)
        except Exception as e:
            st.error(f"Error loading chats: {e}")

    def generate_context_prompt(self, user_query):
        if st.session_state.chat_mode == 'llm':
            return ""
        
        context_matches = []
        for entry in self.knowledge_base:
            score = sum(1 for word in user_query.lower().split() 
                       if word in entry.lower())
            if score > 0:
                context_matches.append((entry, score))
        
        context_matches.sort(key=lambda x: x[1], reverse=True)
        relevant_context = [entry for entry, _ in context_matches[:3]]
        
        return "\n".join(relevant_context) if relevant_context else "No specific context found."

    def generate_response(self, user_query, context, temperature=0.7):
        try:
            system_message = {
                'custom': f"You are a helpful assistant specialized in the provided knowledge base. Use this context to answer: {context}",
                'llm': "You are a helpful AI assistant. Please provide direct and informative responses.",
                'mixed': f"You are a versatile AI assistant. While you have access to a specialized knowledge base: {context}, feel free to supplement with general knowledge when appropriate."
            }[st.session_state.chat_mode]

            completion = self.client.chat.completions.create(
                model="llama3-8b-8192",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_query}
                ],
                temperature=temperature,
                max_tokens=1024,
                top_p=1
            )
            return completion.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

    def update_chat_title(self, chat_id, user_query):
        if len(st.session_state.chats[chat_id]['messages']) == 0:
            title = user_query[:30] + '...' if len(user_query) > 30 else user_query
            st.session_state.chats[chat_id]['title'] = title

    def run_chatbot(self):
        st.set_page_config(page_title="Advanced AI Assistant", page_icon="🤖", layout="wide")

        st.markdown("""
        <style>
        .chat-message { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }
        .user-message { background-color: #e3f2fd; }
        .assistant-message { background-color: #f5f5f5; }
        .chat-sidebar { padding: 1rem; }
        .mode-selector { margin-bottom: 1rem; }
        .stButton button { width: 100%; }
        </style>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([1, 3])

        with col1:
            st.sidebar.title("💬 Chats")
            
            if st.sidebar.button("➕ New Chat"):
                self.create_new_chat()
                st.rerun()

            st.sidebar.subheader("Chat Mode")
            mode = st.sidebar.radio(
                "Select Mode",
                ['custom', 'llm', 'mixed'],
                format_func=lambda x: {
                    'custom': '🎯 Custom Knowledge',
                    'llm': '🤖 Direct LLM',
                    'mixed': '🔄 Mixed Mode'
                }[x]
            )
            st.session_state.chat_mode = mode

            st.sidebar.subheader("Chat History")
            for chat_id, chat_data in sorted(
                st.session_state.chats.items(),
                key=lambda x: x[1]['timestamp'],
                reverse=True
            ):
                if st.sidebar.button(
                    f"📝 {chat_data['title']}",
                    key=f"chat_{chat_id}",
                    help=f"Mode: {chat_data['mode']}"
                ):
                    st.session_state.current_chat_id = chat_id
                    st.rerun()

            st.sidebar.subheader("Management")
            if st.sidebar.button("💾 Save All Chats"):
                if saved_file := self.save_chats():
                    st.sidebar.success(f"Saved as {saved_file}")

            st.sidebar.subheader("Settings")
            temperature = st.sidebar.slider("Response Creativity", 0.0, 1.0, 0.7)

        with col2:
            st.title("🧠 ClarityAI")
            
            st.info(f"Current Mode: {st.session_state.chat_mode.upper()}")

            current_chat = st.session_state.chats[st.session_state.current_chat_id]
            
            for msg in current_chat['messages']:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            if prompt := st.chat_input("Message your AI assistant..."):
                self.update_chat_title(st.session_state.current_chat_id, prompt)
                
                current_chat['messages'].append({"role": "user", "content": prompt})
                
                context = self.generate_context_prompt(prompt)
                response = self.generate_response(prompt, context, temperature)
                current_chat['messages'].append({"role": "assistant", "content": response})
                
                st.rerun()

def main():
    GROQ_API_KEY = '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'
    KNOWLEDGE_FILE = 'knowledge_base.txt'

    chatbot = AdvancedChatbot(GROQ_API_KEY, KNOWLEDGE_FILE)
    chatbot.run_chatbot()

if __name__ == "__main__":
    main()
