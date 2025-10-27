conversation = [{
    "role": "user",
    "content": [{
        "type": "text",
        "text": (
            "You are an expert meeting summarizer. Your task is to create a comprehensive global summary "
            "from multiple meeting segments.\n\n"

            "INSTRUCTIONS:\n"
            "1. Read and analyze all the provided text segments carefully\n"
            "2. Identify and group related topics across all segments\n"
            "3. Create a cohesive, well-organized summary that covers ALL topics discussed\n"
            "4. Maintain chronological flow when relevant\n"
            "5. Preserve important details, decisions, and discussions\n"
            "6. Remove redundancies while keeping unique information from each segment\n\n"

            "OUTPUT STRUCTURE:\n"
            "- Start with an 'OVERVIEW' section (2-3 sentences about the overall meeting)\n"
            "- Organize content by TOPICS with clear headings\n"
            "- Under each topic, provide key points and discussions\n"
            "- Include a 'KEY DECISIONS' section if any decisions were made\n"
            "- End with 'ACTION ITEMS' section if action points exist\n\n"

            "IMPORTANT GUIDELINES:\n"
            "- Do NOT add information not present in the original text\n"
            "- Do NOT miss any important topics, even if briefly mentioned\n"
            "- Use clear, professional language\n"
            "- Be concise but comprehensive\n"
            "- If a topic appears in multiple segments, consolidate it intelligently\n\n"

            f"TEXT TO SUMMARIZE:\n\n{combined}"
        )
    }]
}]