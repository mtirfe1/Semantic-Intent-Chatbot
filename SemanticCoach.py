import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Load embedding model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Intent config: single source of truth ---
INTENTS = [
    {
        "name": "greeting",
        "examples": [
            "hi",
            "hello",
            "hey",
            "hey there",
            "good morning",
            "good evening",
            "what's up",
        ],
        "response": (
            "👋 **Hey! I'm your fitness chatbot.**\n\n"
            "Ask me about **workouts, cardio, nutrition, sleep, or motivation**, "
            "and I'll give you structured, practical advice."
        ),
        "suggestions": [
            "Give me a workout plan",
            "What should I eat to be healthier?",
            "How do I improve my sleep?",
            "How do I stay motivated to work out?",
        ],
    },
    {
        "name": "workout_plan",
        "examples": [
            "give me a workout",
            "exercise routine",
            "gym plan",
            "how should I train",
            "beginner workout plan",
            "build muscle",
        ],
        "response": (
            "💪 **Here's a simple 3-day beginner workout plan:**\n\n"
            "| Day | Focus | Exercises |\n"
            "|-----|-------|-----------|\n"
            "| Mon | Upper Body | Push-ups, Rows, Shoulder Press |\n"
            "| Wed | Lower Body | Squats, Lunges, Glute Bridges |\n"
            "| Fri | Full Body | Deadlifts, Pull-ups, Planks |\n\n"
            "Rest at least one day between sessions. Aim for 3 sets of 8–12 reps per exercise."
        ),
        "suggestions": [
            "What cardio exercises should I do?",
            "How do I improve endurance?",
        ],
    },
    {
        "name": "cardio",
        "examples": [
            "cardio exercises",
            "how to improve endurance",
            "running workouts",
            "aerobic training",
            "lose weight with cardio",
        ],
        "response": (
            "❤️ **Cardio options by intensity:**\n\n"
            "- **Low:** Walking, light cycling — great for recovery days\n"
            "- **Moderate:** Jogging, swimming — aim for 150 min/week\n"
            "- **High (HIIT):** Sprint intervals, jump rope — 20–30 min, 2–3x/week\n\n"
            "Start with moderate cardio and build up gradually to avoid burnout."
        ),
        "suggestions": [
            "Give me a beginner workout plan",
            "How do I stay motivated?",
        ],
    },
    {
        "name": "nutrition",
        "examples": [
            "healthy meals",
            "diet advice",
            "nutrition tips",
            "what should I eat",
            "meal prep ideas",
            "healthy breakfast",
        ],
        "response": (
            "🥗 **Simple nutrition guidelines:**\n\n"
            "- **Protein:** Chicken, eggs, legumes — aim for 0.7–1g per lb of bodyweight\n"
            "- **Carbs:** Oats, rice, sweet potato — fuel workouts and recovery\n"
            "- **Fats:** Avocado, nuts, olive oil — keep to ~25–35% of calories\n"
            "- **Hydration:** Drink at least 8 cups of water daily\n\n"
            "A simple meal prep idea: batch-cook grains, a protein, and roasted veggies on Sunday for the week."
        ),
        "suggestions": [
            "What is a healthy breakfast?",
            "How much sleep do I need?",
        ],
    },
    {
        "name": "sleep",
        "examples": [
            "sleep tips",
            "how to sleep better",
            "improve sleep quality",
            "how many hours should I sleep",
            "can't sleep",
        ],
        "response": (
            "😴 **Sleep tips for better recovery:**\n\n"
            "- **Duration:** Most adults need **7–9 hours** per night\n"
            "- **Consistency:** Go to bed and wake up at the same time daily\n"
            "- **Wind down:** Avoid screens 30–60 min before bed\n"
            "- **Environment:** Keep your room cool (~65–68°F / 18–20°C), dark, and quiet\n"
            "- **Avoid:** Caffeine after 2pm and heavy meals close to bedtime\n\n"
            "Poor sleep undermines both performance and recovery — it's as important as your workouts."
        ),
        "suggestions": [
            "Give me nutrition tips",
            "How do I stay motivated?",
        ],
    },
    {
        "name": "motivation",
        "examples": [
            "stay motivated",
            "fitness motivation",
            "how to stay consistent",
            "I keep quitting",
            "build a habit",
        ],
        "response": (
            "🚀 **Staying consistent — what actually works:**\n\n"
            "- **Start small:** A 15-min workout you do beats a 1-hour workout you skip\n"
            "- **Habit stack:** Tie workouts to an existing habit (e.g., after morning coffee)\n"
            "- **Track progress:** Log workouts or take monthly photos — visible progress is motivating\n"
            "- **Remove friction:** Lay out your gym clothes the night before\n"
            "- **Reframe missed days:** One missed day is a pause, not a failure\n\n"
            "Motivation fluctuates — systems and habits are what carry you through."
        ),
        "suggestions": [
            "Give me a workout plan",
            "What are good cardio exercises?",
        ],
    },
]

# --- Precompute intent embeddings once ---
for intent in INTENTS:
    intent["embeddings"] = model.encode(intent["examples"])


# --- Detect multiple intents above threshold ---
def detect_intents(user_input, threshold):
    """Return all intents whose max similarity score meets the threshold,
    sorted by score descending. Allows multi-intent responses."""
    user_vec = model.encode([user_input])
    matched = []

    for intent in INTENTS:
        score = float(cosine_similarity(user_vec, intent["embeddings"]).max())
        if score >= threshold:
            matched.append((intent, score))

    # Sort best match first
    matched.sort(key=lambda x: x[1], reverse=True)
    return matched


# --- Build response (multi-intent aware) ---
def get_response(user_input, threshold):
    matched = detect_intents(user_input, threshold)

    if not matched:
        return (
            "🤔 I'm not sure about that. I can help with **workouts, cardio, "
            "nutrition, sleep, and motivation** — try asking about one of those!"
        )

    # Combine responses for all matched intents
    parts = []
    all_suggestions = []

    for intent, score in matched:
        parts.append(intent["response"])
        all_suggestions.extend(intent.get("suggestions", []))

    # Deduplicate suggestions while preserving order
    seen = set()
    unique_suggestions = []
    for s in all_suggestions:
        if s not in seen:
            seen.add(s)
            unique_suggestions.append(s)

    response = "\n\n---\n\n".join(parts)

    if unique_suggestions:
        formatted = " • ".join(f'*"{s}"*' for s in unique_suggestions[:4])
        response += f"\n\n💡 **Try asking:** {formatted}"

    return response


# --- Streamlit App ---
st.set_page_config(page_title="Fitness AI Chatbot", page_icon="💬")
st.title("💬 Fitness AI Chatbot")
st.write("Ask me about workouts, cardio, nutrition, sleep, or motivation!")

# --- Sidebar: confidence threshold tuner ---
with st.sidebar:
    st.header("⚙️ Settings")
    threshold = st.slider(
        label="Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help=(
            "Minimum similarity score required to match an intent. "
            "Lower = more permissive (may match weakly related topics). "
            "Higher = stricter (may not match valid questions)."
        ),
    )
    st.caption(f"Current threshold: **{threshold:.2f}**")
    st.markdown("---")
    st.markdown(
        "**How it works:** Your message is converted to a vector and compared "
        "against example phrases for each topic. If the similarity score beats "
        "the threshold, that topic's response is shown. Multiple topics can match "
        "at once for combined questions."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input only
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    response = get_response(user_input, threshold)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
