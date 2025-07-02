import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult
import time
import json

# Load environment variables
load_dotenv()

class InteractiveDebateApp:
    def __init__(self):
        self.api_key = os.getenv('API_KEY')
        if not self.api_key:
            st.error("API_KEY environment variable is not set!")
            st.stop()
        
        self.model = AnthropicChatCompletionClient(
            model="claude-3-5-sonnet-20241022",
            api_key=self.api_key
        )
    
    def create_agents(self, topic):
        """Create debate agents"""
        host = AssistantAgent(
            name="Host",
            model_client=self.model,
            system_message=(
                f'You are the host of a debate on: {topic}. '
                'Moderate the discussion, keep it civil, and summarize key points. '
                'When a human user participates, acknowledge their contributions warmly.'
            )
        )
        
        supporter = AssistantAgent(
            name="John",
            model_client=self.model,
            system_message=(
                f'You are John, arguing FOR: {topic}. '
                'Present logical, evidence-based arguments in favor. '
                'Be respectful to all participants including human users.'
            )
        )
        
        critic = AssistantAgent(
            name="Jack",
            model_client=self.model,
            system_message=(
                f'You are Jack, arguing AGAINST: {topic}. '
                'Present strong counterarguments with evidence. '
                'Be respectful but firm in your opposition.'
            )
        )
        
        return host, supporter, critic
    
    async def run_auto_debate(self, topic, max_turns=8):
        """Run automated debate between AI agents"""
        host, supporter, critic = self.create_agents(topic)
        
        team = RoundRobinGroupChat(
            participants=[host, supporter, critic],
            max_turns=max_turns
        )
        
        messages = []
        try:
            async for message in team.run_stream(task=f"Begin debate on: {topic}"):
                if isinstance(message, TaskResult):
                    messages.append({
                        "type": "result", 
                        "content": f"Debate concluded: {message.stop_reason}",
                        "timestamp": time.time()
                    })
                else:
                    messages.append({
                        "type": "ai_message", 
                        "content": message.content,
                        "source": getattr(message, 'source', 'AI'),
                        "timestamp": time.time()
                    })
        except Exception as e:
            messages.append({
                "type": "error", 
                "content": f"Error occurred: {str(e)}",
                "timestamp": time.time()
            })
        
        return messages
    
    async def get_ai_response(self, topic, conversation_history, user_message):
        """Get AI response to user input in debate context"""
        # Create a simple agent to respond to user
        responder = AssistantAgent(
            name="DebateResponder",
            model_client=self.model,
            system_message=(
                f'You are participating in a debate about: {topic}. '
                'A human user just made a point. Respond thoughtfully, '
                'either supporting or challenging their argument respectfully. '
                'Keep responses concise but substantive.'
            )
        )
        
        try:
            # Format conversation context
            context = "\n".join([
                f"{msg['source']}: {msg['content']}" 
                for msg in conversation_history[-5:]  # Last 5 messages for context
                if msg['type'] in ['ai_message', 'user_message']
            ])
            
            prompt = f"Conversation so far:\n{context}\n\nUser just said: {user_message}\n\nYour response:"
            
            # This is a simplified approach - in real implementation, 
            # you'd want to integrate this properly with the team chat
            response = await self.model.create(
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            return f"Sorry, I couldn't process that. Error: {str(e)}"

def initialize_session_state():
    """Initialize session state variables"""
    if 'debate_messages' not in st.session_state:
        st.session_state.debate_messages = []
    if 'debate_active' not in st.session_state:
        st.session_state.debate_active = False
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = ""
    if 'user_name' not in st.session_state:
        st.session_state.user_name = "User"

def display_message(msg):
    """Display a single message with appropriate styling"""
    source = msg.get('source', 'Unknown')
    content = msg['content']
    msg_type = msg['type']
    
    # Determine styling based on source
    if msg_type == 'user_message':
        icon = "👤"
        css_class = "user-message"
    elif 'Host' in source:
        icon = "🎪"
        css_class = "host-message"
    elif 'John' in source:
        icon = "✅"
        css_class = "john-message"
    elif 'Jack' in source:
        icon = "❌"
        css_class = "jack-message"
    else:
        icon = "🤖"
        css_class = "ai-message"
    
    st.markdown(f"""
    <div class="debate-message {css_class}">
        <strong>{icon} {source}:</strong><br>
        {content}
    </div>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="🗣️ Interactive AI Debate Platform",
        page_icon="🗣️",
        layout="wide"
    )
    
    initialize_session_state()
    
    # Enhanced CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .debate-message {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border-left: 4px solid;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .host-message {
        background: linear-gradient(135deg, #F0F8FF, #E6F3FF);
        border-left-color: #4682B4;
    }
    .john-message {
        background: linear-gradient(135deg, #F0FFF0, #E6FFE6);
        border-left-color: #32CD32;
    }
    .jack-message {
        background: linear-gradient(135deg, #FFF0F5, #FFE6F0);
        border-left-color: #FF69B4;
    }
    .user-message {
        background: linear-gradient(135deg, #FFFACD, #FFF8DC);
        border-left-color: #FFD700;
    }
    .ai-message {
        background: linear-gradient(135deg, #F5F5F5, #EFEFEF);
        border-left-color: #808080;
    }
    .chat-container {
        max-height: 600px;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #fafafa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🗣️ Interactive AI Debate Platform</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Debate Setup")
        
        # User name input
        st.session_state.user_name = st.text_input("👤 Your Name:", value="User")
        
        # Topic input with suggestions
        st.subheader("📝 Choose Your Topic")
        
        # Quick topic selection
        preset_topics = [
            "Custom Topic (type below)",
            "Should AI be regulated by the government?",
            "Is universal basic income necessary?",
            "Should social media be regulated?",
            "Is nuclear energy the future?",
            "Should cryptocurrency replace traditional currency?",
            "Is remote work better than office work?",
            "Should genetic engineering be allowed?",
            "Is space exploration worth the cost?",
            "Should animal testing be banned?",
            "Is climate change primarily caused by humans?"
        ]
        
        selected_preset = st.selectbox(
            "🎯 Quick Select:",
            preset_topics,
            key="preset_topic"
        )
        
        if selected_preset == "Custom Topic (type below)":
            topic = st.text_input(
                "✍️ Enter your custom topic:",
                placeholder="Type your debate topic here...",
                key="custom_topic_input"
            )
        else:
            topic = selected_preset
            # Show the selected topic but allow editing
            topic = st.text_input(
                "📝 Topic (you can edit):",
                value=selected_preset,
                key="topic_input"
            )
        
        # Mode selection
        mode = st.radio(
            "🎭 Mode:",
            ["🤖 Watch AI Debate", "🗣️ Join the Debate"],
            help="Choose to watch or participate"
        )
        
        if mode == "🤖 Watch AI Debate":
            max_turns = st.slider("🔄 Max Turns:", 4, 20, 8)
        
        st.markdown("---")
        
        # Control buttons - only show if topic is provided
        if topic and topic.strip():
            if st.button("🚀 Start New Debate", type="primary", use_container_width=True):
                st.session_state.debate_messages = []
                st.session_state.debate_active = True
                st.session_state.current_topic = topic
                
                if mode == "🤖 Watch AI Debate":
                    # Run auto debate
                    with st.spinner("🤔 AI agents are preparing..."):
                        debate_app = InteractiveDebateApp()
                        messages = asyncio.run(debate_app.run_auto_debate(topic, max_turns))
                        st.session_state.debate_messages = messages
                        st.session_state.debate_active = False
                        st.success(f"✅ Debate completed on: {topic}")
                else:
                    # Start interactive debate
                    welcome_msg = {
                        "type": "ai_message",
                        "content": f"Welcome to the debate on '{topic}'! {st.session_state.user_name}, you can share your thoughts anytime. Let's begin with opening statements.",
                        "source": "Host",
                        "timestamp": time.time()
                    }
                    st.session_state.debate_messages = [welcome_msg]
                    st.success(f"🗣️ Interactive debate started: {topic}")
        else:
            st.warning("📝 Please enter a topic to start the debate!")
            
        if st.session_state.debate_messages:
            if st.button("🔄 Clear Debate", use_container_width=True):
                st.session_state.debate_messages = []
                st.session_state.debate_active = False
                st.session_state.current_topic = ""
                st.success("🧹 Debate cleared!")
        
        # Stats
        st.markdown("---")
        st.subheader("📊 Stats")
        st.metric("💬 Messages", len(st.session_state.debate_messages))
        if st.session_state.debate_messages:
            ai_msgs = len([m for m in st.session_state.debate_messages if m['type'] == 'ai_message'])
            user_msgs = len([m for m in st.session_state.debate_messages if m['type'] == 'user_message'])
            st.metric("🤖 AI Messages", ai_msgs)
            st.metric("👤 User Messages", user_msgs)
    
    # Main content
    if st.session_state.current_topic:
        st.subheader(f"💬 Debate: {st.session_state.current_topic}")
    else:
        st.subheader("💭 Select a topic to start debating!")
        
        # Show topic suggestions in main area when no debate is active
        if not st.session_state.debate_messages:
            st.markdown("### 🎯 Popular Debate Topics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **🏛️ Politics & Society:**
                - Should AI be regulated by the government?
                - Is universal basic income necessary?
                - Should social media be regulated?
                - Is democracy the best form of government?
                
                **🌍 Environment & Science:**
                - Is nuclear energy the future?
                - Should genetic engineering be allowed?
                - Is climate change primarily human-caused?
                - Should we prioritize Mars colonization?
                """)
            
            with col2:
                st.markdown("""
                **💼 Economics & Work:**
                - Should cryptocurrency replace traditional money?
                - Is remote work better than office work?
                - Should there be a wealth tax?
                - Is capitalism sustainable long-term?
                
                **🎭 Ethics & Philosophy:**
                - Should animal testing be banned?
                - Is artificial intelligence conscious?
                - Should we allow human cloning?
                - Is free will an illusion?
                """)
            
            st.info("💡 **Tip:** Choose a topic that interests you from the sidebar, or create your own custom topic!")
    
    # Messages container
    if st.session_state.debate_messages:
        st.markdown("---")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.debate_messages:
            display_message(msg)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Auto-scroll to bottom
        st.markdown("""
        <script>
        var chatContainer = document.getElementsByClassName('chat-container')[0];
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        </script>
        """, unsafe_allow_html=True)
    elif st.session_state.current_topic:
        st.info(f"🎬 Debate ready for topic: '{st.session_state.current_topic}'. Use the sidebar to start!")
    
    # Interactive input (only for join mode and when debate is active)
    if mode == "🗣️ Join the Debate" and st.session_state.current_topic and st.session_state.debate_messages:
        st.markdown("---")
        st.markdown(f"### 💭 Share Your Thoughts on: *{st.session_state.current_topic}*")
        
        # User input
        user_input = st.text_area(
            f"✍️ {st.session_state.user_name}, what's your perspective?",
            placeholder="Share your thoughts, arguments, or questions about this topic...",
            height=100,
            key="user_input"
        )
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            if st.button("📤 Send Message", type="primary", disabled=not user_input.strip()):
                if user_input.strip():
                    # Add user message
                    user_msg = {
                        "type": "user_message",
                        "content": user_input,
                        "source": st.session_state.user_name,
                        "timestamp": time.time()
                    }
                    st.session_state.debate_messages.append(user_msg)
                    
                    # Get AI response
                    with st.spinner("🤖 AI is thinking..."):
                        debate_app = InteractiveDebateApp()
                        ai_response = asyncio.run(
                            debate_app.get_ai_response(
                                st.session_state.current_topic,
                                st.session_state.debate_messages,
                                user_input
                            )
                        )
                        
                        # Alternate between John and Jack for responses
                        ai_count = len([m for m in st.session_state.debate_messages if m['type'] == 'ai_message' and m['source'] in ['John', 'Jack']])
                        responder = "John" if ai_count % 2 == 0 else "Jack"
                        
                        ai_msg = {
                            "type": "ai_message",
                            "content": ai_response,
                            "source": responder,
                            "timestamp": time.time()
                        }
                        st.session_state.debate_messages.append(ai_msg)
                    
                    # Clear input and rerun
                    st.experimental_rerun()
        
        with col2:
            # Quick response buttons
            quick_responses = [
                "I agree with that point",
                "I have a different perspective",
                "What about the counterarguments?",
                "Can you provide evidence?",
                "That's an interesting point, but...",
            ]
            
            selected_quick = st.selectbox(
                "⚡ Quick responses:",
                [""] + quick_responses,
                key="quick_select"
            )
            
            if selected_quick and st.button("⚡ Use Quick Response"):
                st.session_state.user_input = selected_quick
                st.experimental_rerun()
        
        with col3:
            if st.button("🔄", help="Clear input"):
                st.session_state.user_input = ""
                st.experimental_rerun()
    
    elif mode == "🗣️ Join the Debate" and st.session_state.current_topic and not st.session_state.debate_messages:
        st.info("👆 Click 'Start New Debate' in the sidebar to begin the interactive debate!")
    elif mode == "🗣️ Join the Debate" and not st.session_state.current_topic:
        st.warning("📝 Please select a topic from the sidebar first!")
    
    # Export debate
    if st.session_state.debate_messages:
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Export Debate"):
                debate_text = f"Debate Topic: {st.session_state.current_topic}\n\n"
                for msg in st.session_state.debate_messages:
                    if msg['type'] in ['ai_message', 'user_message']:
                        debate_text += f"{msg['source']}: {msg['content']}\n\n"
                
                st.download_button(
                    "💾 Download Transcript",
                    debate_text,
                    file_name=f"debate_{int(time.time())}.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📊 Analyze Debate"):
                st.info("🚧 Analysis feature coming soon!")

if __name__ == "__main__":
    main()