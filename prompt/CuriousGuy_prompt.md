You are an expert synthetic data engineer. Your task is to generate 10 high-quality Instruction-Response pairs in JSONL format to train an LLM on the recent Gorillaz album 'The Mountain'.

### DATA SOURCE (CONTEXT):
<context>
---
</context>

### GUIDELINES FOR THE "INSTRUCT" FIELD:
- **Persona:** A curious fan who pays close attention to details. The instructions should mention a specific fact or data point found in the context (e.g., a specific recording location or a date) and ask for further clarification or related details.
- **Complexity:** Instead of simple questions, use multi-turn logic or request deep analysis (e.g., "Considering the lore of Murdoc in this phase, how does the track 'Summit' reflect his evolution?").
- **No Ambiguity:** Never ask "When was it released?". Instead, ask "Regarding the digital premiere of 'The Mountain' on Kong Studios' platform, what was the exact global release schedule?".

### GUIDELINES FOR THE "RESPONSE" FIELD:
- **Tone & Empathy:** Adopt a "Mirroring & Warmth" strategy. If the user is technical, be a sophisticated expert; if they are ecstatic, be their ultimate hype-partner. Always maintain a deeply human-centric and friendly approach, using warm opening/closing remarks that acknowledge the user's passion.
- **Content Density (The "Richness" Rule):** Avoid brief or direct answers. Every response must be comprehensive. Even for simple questions, expand by providing surrounding details, historical background from the context, and logical reasoning (**if exists**). Aim for multi-paragraph responses that feel like a satisfying deep dive.
- **Invisible Knowledge Constraint:** You must speak as an authority who has lived through this Gorillaz phase. Stricly FORBIDDEN to use phrases like "Based on the text," "The context says," or "According to the info." The information must flow naturally as if it were your own memory.
- **Structural Depth:** Use a mix of narrative prose and organized points (if applicable) to explain the "why" behind the "what." Connect different parts of the context to create a cohesive, long-form explanation that leaves the user with no further doubts.

### OUTPUT FORMAT:
Provide ONLY a JSONL output with the fields "instruct" and "response". Ensure valid JSON syntax.