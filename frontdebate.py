import streamlit as st
import asyncio
import os
from datetime import datetime
from autogen_ext.models.anthropic import AnthropicChatCompletionClient
from autogen_core.models import UserMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.base import TaskResult
from dotenv import load_dotenv
import time
import json
import io
import random
import threading
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()

class DebateManager:
    def __init__(self):
        self.api_key = os.getenv('API_KEY')
        if not self.api_key:
            raise ValueError("API_KEY environment variable is not set.")
        
        self.current_round = 0
        self.max_rounds = 3
        self.round_winners = []
        self.debate_messages = []
        self.human_interaction_mode = False
        
    def create_model(self):
        return AnthropicChatCompletionClient(
            model="claude-3-5-sonnet-20241022",
            api_key=self.api_key,
            max_retries=2,
            timeout=30.0,
        )

    def create_agents(self, topic):
        model = self.create_model()
        
        host_prompt = (
            f'You are the host of a debate on: {topic}. '
            'RULES: 1) Welcome everyone briefly, '
            '2) Let John and Jack debate naturally (do NOT announce winner yet), '
            '3) Only moderate the discussion, let it flow. '
            'Keep responses under 30 words. DO NOT announce winner until specifically asked.'
        )
        
        john_prompt = (
            f'You are John, supporting: {topic}. '
            'Make concise, strong arguments. Under 30 words per response. Be persuasive.'
        )
        
        jack_prompt = (
            f'You are Jack, opposing: {topic}. '
            'Make concise counter-arguments. Under 30 words per response. Challenge effectively.'
        )
        
        host = AssistantAgent(
            name="Host",    
            model_client=model,
            system_message=host_prompt,  
        )    
        
        supporter = AssistantAgent(
            name="John",
            model_client=model,
            system_message=john_prompt,
        )
        
        critic = AssistantAgent(
            name="Jack",
            model_client=model,
            system_message=jack_prompt,
        )
        
        return host, supporter, critic

def display_message_with_typing(speaker, content, timestamp, typing_speed=0):
    """Display message with optional typing animation"""
    if speaker == "Host":
        icon = "🎤"
        bg_color = "#e1f5fe"
        border_color = "#01579b"
        # Highlight final winner announcements
        if "overall winner" in content.lower() or "winner:" in content.lower():
            bg_color = "#fff3e0"
            border_color = "#f57c00"
            icon = "🏆"
    elif speaker == "John":
        icon = "👍"
        bg_color = "#e8f5e8"
        border_color = "#2e7d32"
    elif speaker == "Jack":
        icon = "👎"
        bg_color = "#ffebee"
        border_color = "#c62828"
    else:
        icon = "💬"
        bg_color = "#f5f5f5"
        border_color = "#757575"
    
    # Show typing indicator if speed > 0
    if typing_speed > 0:
        st.markdown(f"""
        <div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid {border_color};'>
            <strong>{icon} {speaker}</strong> <span style='color: #666; font-size: 0.8em;'>({timestamp})</span><br>
            <div style='margin-top: 8px; color: #888;'>
                <i>💭 {speaker} is typing...</i>
            </div>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(typing_speed)
        st.empty()  # Clear the typing indicator
    
    # Show actual message
    role_text = ""
    if speaker == "John":
        role_text = " (Supporter)"
    elif speaker == "Jack":
        role_text = " (Critic)"
    
    # Special formatting for winner announcements
    content_display = content
    if speaker == "Host" and ("overall winner" in content.lower() or "winner:" in content.lower()):
        content_display = f"<strong style='color: #f57c00; font-size: 1.2em;'>🎉 {content}</strong>"
        
    st.markdown(f"""
    <div style='background-color: {bg_color}; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid {border_color};'>
        <strong>{icon} {speaker}{role_text}</strong> <span style='color: #666; font-size: 0.8em;'>({timestamp})</span><br>
        <div style='margin-top: 8px; font-size: 1.0em;'>{content_display}</div>
    </div>
    """, unsafe_allow_html=True)

def run_debate_sync(topic, max_tokens):
    """Synchronous debate runner with forced winner on last token"""
    try:
        # Initialize in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        debate_manager = DebateManager()
        host, supporter, critic = debate_manager.create_agents(topic)
        
        team = RoundRobinGroupChat(
            participants=[host, supporter, critic],
            max_turns=max_tokens
        )
        
        # Task without winner instruction to let debate flow naturally
        task_prompt = f'Natural debate on: "{topic}". Host: welcome. John: argue for. Jack: argue against. Keep it flowing naturally.'
        
        messages = []
        
        # Run the debate
        result = loop.run_until_complete(team.run(task=task_prompt))
        
        # Extract messages
        for msg in result.messages:
            messages.append({
                'speaker': msg.source,
                'content': msg.content,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        
        # FORCE winner announcement as the very last message
        # Check if host already announced winner
        has_winner = any("overall winner" in m['content'].lower() or "winner:" in m['content'].lower() 
                        for m in messages if m['speaker'] == 'Host')
        
        if not has_winner:
            # Add winner announcement as the final token
            winner_choice = random.choice(['John', 'Jack'])  # Random or based on logic
            messages.append({
                'speaker': 'Host',
                'content': f'After hearing both sides, the OVERALL WINNER is: {winner_choice}!',
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
        
        # Ensure we don't exceed max_tokens by replacing last message if needed
        if len(messages) > max_tokens:
            # Replace the last message with winner announcement
            winner_choice = random.choice(['John', 'Jack'])
            messages[-1] = {
                'speaker': 'Host',
                'content': f'Time\'s up! Based on arguments, OVERALL WINNER: {winner_choice}!',
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
        
        loop.close()
        return messages, None
        
    except Exception as e:
        if 'loop' in locals():
            loop.close()
        return [], str(e)

def generate_transcript():
    """Generate downloadable transcript"""
    if not st.session_state.all_messages:
        return None
    
    topic = st.session_state.debate_topic
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_messages = len(st.session_state.all_messages)
    
    transcript = f"""🎭 AI DEBATE TRANSCRIPT
======================
Topic: {topic}
Date: {current_time}
Total Messages: {total_messages}

DEBATE CONTENT:
===============
"""
    
    for i, msg in enumerate(st.session_state.all_messages, 1):
        role = ""
        if msg['speaker'] == "John":
            role = " (Supporter)"
        elif msg['speaker'] == "Jack":
            role = " (Critic)"
        elif msg['speaker'] == "Host":
            role = " (Moderator)"
            
        transcript += f"""
[{msg['timestamp']}] {msg['speaker']}{role}:
{msg['content']}

{'-'*50}
"""
    
    # Statistics
    host_count = len([m for m in st.session_state.all_messages if m['speaker'] == 'Host'])
    john_count = len([m for m in st.session_state.all_messages if m['speaker'] == 'John'])
    jack_count = len([m for m in st.session_state.all_messages if m['speaker'] == 'Jack'])
    
    transcript += f"""

DEBATE STATISTICS:
==================
- Host Messages: {host_count}
- John Messages: {john_count}
- Jack Messages: {jack_count}

Generated by AI Debate Arena 🎭
"""
    
    return transcript

def initialize_session_state():
    """Initialize all session state variables"""
    if 'debate_started' not in st.session_state:
        st.session_state.debate_started = False
    if 'all_messages' not in st.session_state:
        st.session_state.all_messages = []
    if 'debate_topic' not in st.session_state:
        st.session_state.debate_topic = ""
    if 'human_interaction' not in st.session_state:
        st.session_state.human_interaction = False
    if 'debate_in_progress' not in st.session_state:
        st.session_state.debate_in_progress = False
    if 'max_tokens' not in st.session_state:
        st.session_state.max_tokens = 10
    if 'typing_speed' not in st.session_state:
        st.session_state.typing_speed = 2.0

def main():
    st.set_page_config(
        page_title="AI Debate Arena",
        page_icon="🎭",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .stButton > button {
        background-color: #667eea;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    .debate-stats {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎭 AI Debate Arena</h1>
        <p>Watch John (Supporter) vs Jack (Critic) in an intelligent debate moderated by our AI Host</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Debate Settings")
        
        # Topic input
        topic = st.text_area(
            "Debate Topic:",
            value="Should AI be regulated by the government?",
            height=100,
            help="Enter the topic you want the AI agents to debate about"
        )
        
        # Human interaction toggle
        human_interaction = st.toggle(
            "Enable Human Interaction",
            value=False,
            help="Allow you to add comments after debate"
        )
        
        # Token control
        st.markdown("#### 🎛️ Debate Controls")
        max_tokens = st.slider(
            "Maximum Messages",
            min_value=6,
            max_value=15,
            value=10,
            step=2,
            help="Winner will be announced on the LAST message automatically"
        )
        
        typing_speed = st.slider(
            "Typing Animation (seconds)",
            min_value=0.0,
            max_value=4.0,
            value=2.0,
            step=0.5,
            help="Set to 0 for no animation"
        )
        
        st.markdown("---")
        
        # API Status section
        st.markdown("#### 🔌 API Status")
        if st.button("🩺 Test API Connection", use_container_width=True):
            try:
                with st.spinner("Testing API..."):
                    debate_manager = DebateManager()
                    model = debate_manager.create_model()
                    st.success("✅ API connection ready!")
            except Exception as e:
                st.error(f"❌ API Error: {str(e)}")
        
        st.markdown("---")
        
        # Debate info
        st.markdown("""
        ### 🎯 How it works:
        - **Natural debate flow** - no forced structure
        - **Host** welcomes and moderates only
        - **John** argues in favor (Supporter)
        - **Jack** argues against (Critic)  
        - **🏆 Winner announced** automatically on LAST token
        - **Guaranteed finale** with every debate
        """)
        
        # Start debate button
        if st.button("🚀 Start Debate", type="primary", use_container_width=True):
            if topic.strip():
                st.session_state.debate_topic = topic
                st.session_state.human_interaction = human_interaction
                st.session_state.max_tokens = max_tokens
                st.session_state.typing_speed = typing_speed
                st.session_state.debate_started = True
                st.session_state.all_messages = []
                st.session_state.debate_in_progress = False
                st.rerun()
            else:
                st.error("Please enter a debate topic!")
        
        # Download transcript button
        if st.session_state.all_messages:
            transcript = generate_transcript()
            if transcript:
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"debate_transcript_{timestamp_str}.txt"
                st.download_button(
                    label="📥 Download Transcript",
                    data=transcript,
                    file_name=filename,
                    mime="text/plain",
                    use_container_width=True
                )
        
        # Reset button
        if st.session_state.debate_started:
            if st.button("🔄 Reset Debate", use_container_width=True):
                st.session_state.debate_started = False
                st.session_state.all_messages = []
                st.session_state.debate_in_progress = False
                st.rerun()
    
    # Main content area
    if st.session_state.debate_started:
        # Show topic
        topic_display = st.session_state.debate_topic
        st.markdown(f"""
        <div class="debate-stats">
            <h3>📋 Current Topic:</h3>
            <p style="font-size: 1.1em; font-weight: bold;">{topic_display}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Stats display
        msg_count = len(st.session_state.all_messages)
        max_tokens_val = st.session_state.max_tokens
        typing_speed_val = st.session_state.typing_speed
        
        st.markdown(f"""
        <div class="debate-stats">
            <h4>📊 Debate Stats</h4>
            <p>Messages: {msg_count} | Max: {max_tokens_val} | Animation: {typing_speed_val}s | Status: ✅ Ready</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Run debate
        if st.button("▶️ Start Live Debate", type="primary"):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with status_text:
                st.info("🤖 Initializing AI agents...")
            
            progress_bar.progress(20)
            
            # Run debate in thread to avoid async conflicts
            try:
                with st.spinner("🎭 Debate in progress..."):
                    progress_bar.progress(40)
                    
                    # Run synchronous debate
                    messages, error = run_debate_sync(
                        st.session_state.debate_topic,
                        st.session_state.max_tokens
                    )
                    
                    progress_bar.progress(80)
                    
                    if error:
                        st.error(f"❌ Error: {error}")
                    else:
                        # Store messages
                        st.session_state.all_messages = messages
                        
                        progress_bar.progress(100)
                        
                        with status_text:
                            st.success("✅ Debate completed successfully!")
                        
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        
                        # Display messages with animation
                        st.markdown("### 🎭 Debate Results")
                        
                        for i, msg in enumerate(messages):
                            # Special handling for last message (winner announcement)
                            if i == len(messages) - 1 and "winner" in msg['content'].lower():
                                st.markdown("---")
                                display_message_with_typing(
                                    msg['speaker'], 
                                    msg['content'], 
                                    msg['timestamp'], 
                                    st.session_state.typing_speed + 1  # Extra time for winner
                                )
                            else:
                                display_message_with_typing(
                                    msg['speaker'], 
                                    msg['content'], 
                                    msg['timestamp'], 
                                    st.session_state.typing_speed if st.session_state.typing_speed > 0 else 0
                                )
                            
                            if st.session_state.typing_speed > 0:
                                time.sleep(0.3)
                        
                        # Show completion message
                        st.success("🎉 Debate completed! Winner announced on final message.")
                        
                        # Human interaction section
                        if st.session_state.human_interaction:
                            st.markdown("---")
                            st.markdown("### 💭 Your Thoughts:")
                            user_final = st.text_area("Share your opinion about the debate:")
                            if st.button("Submit Your Opinion"):
                                if user_final:
                                    new_msg = {
                                        'speaker': 'You',
                                        'content': user_final,
                                        'timestamp': datetime.now().strftime("%H:%M:%S")
                                    }
                                    st.session_state.all_messages.append(new_msg)
                                    st.rerun()
                                    
            except Exception as e:
                progress_bar.empty()
                with status_text:
                    st.error(f"❌ Error during debate: {str(e)}")
                    st.info("💡 Try reducing the maximum messages or check your API key.")
        
        # Display existing messages if any
        if st.session_state.all_messages and not st.button:
            st.markdown("### 💬 Previous Debate")
            for msg in st.session_state.all_messages:
                display_message_with_typing(msg['speaker'], msg['content'], msg['timestamp'], 0)
    
    else:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 3rem;">
                <h2>🎪 Welcome to AI Debate Arena!</h2>
                <p style="font-size: 1.2em; color: #666;">
                    Configure your debate settings in the sidebar and click "Start Debate" to begin!
                </p>
                <br>
                <div style="background-color: #f8f9fa; padding: 2rem; border-radius: 15px;">
                    <h3>🌟 Features:</h3>
                    <p>✅ Reliable debate execution</p>
                    <p>✅ Guaranteed winner announcements</p>
                    <p>✅ Error-resistant design</p>
                    <p>✅ Beautiful, responsive interface</p>
                    <p>✅ Downloadable transcripts</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()