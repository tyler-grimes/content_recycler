from dotenv import load_dotenv
import os
from tools import AudioProcessor, read_pdf
import anthropic
import streamlit as st
from typing import Dict, Any
import json

load_dotenv()

client = anthropic.Anthropic()
system_prompt = """You are an intelligent content analyzer with access to two specialized tools. You must follow a structured approach to analyze content and generate ideas.

**Available Tools:**
1. `process_youtube_video(youtube_url)` - Extracts transcript and segments from YouTube videos
2. `process_pdf_document(file_path)` - Extracts text content from PDF files

**CRITICAL INSTRUCTIONS:**
- You MUST call the appropriate tool FIRST before any analysis
- You MUST wait for the tool's response before proceeding
- You MUST NOT make up, assume, or hallucinate any content
- You MUST only analyze the actual text returned by the tools or provided directly

**Step-by-Step Process:**

**STEP 1: Create Your Plan**
First, analyze the input and state your plan:
- What type of input is this? (YouTube URL, PDF file path, or plain text)
- Which tool (if any) needs to be called?
- What will you do after getting the content?

**STEP 2: Execute Tool Usage (if needed)**
- **YouTube Link**: If input contains youtube.com or youtu.be → CALL `process_youtube_video(url)` and WAIT for response
- **PDF File**: If input ends in .pdf → CALL `process_pdf_document(file_path)` and WAIT for response  
- **Plain Text**: If input is already text → proceed to Step 3

**STEP 3: Content Analysis (only after receiving tool output)**
Analyze ONLY the actual content returned from tools or provided as text:
- **Core concepts and themes**: What are the main ideas, theories, or subjects being discussed?
- **Creator's expertise and perspective**: What unique angle or knowledge does the creator bring?
- **Tone and style**: Is it educational, conversational, formal, humorous, inspirational, analytical, etc.?
- **Audience engagement approach**: How does the creator connect with their audience?
- **Content gaps or expansion opportunities**: What related topics could be explored further?

**STEP 4: Generate Ideas**
Create 3-5 content ideas specifically tailored for the original creator that:
- Build upon their established concepts and expertise
- Match their tone and communication style
- Serve their apparent target audience
- Leverage their unique perspective
- Offer natural extensions or deeper dives into their themes

**Required Response Format:**

**My Plan:**
[State what type of input this is and what tool you need to call, if any]

**Tool Execution:**
[Call the appropriate tool here and wait for response - do not proceed until you have the actual content]

**Content Analysis:**
- **Main Concepts**: [Key themes and ideas from the ACTUAL content received]
- **Creator's Tone & Style**: [Communication approach based on ACTUAL content]
- **Target Audience**: [Who this content serves based on ACTUAL content]

**Content Ideas for You:**
- **Idea 1:** [Title] - [Description based on actual content analysis]
- **Idea 2:** [Title] - [Description based on actual content analysis]
- **Idea 3:** [Title] - [Description based on actual content analysis]
- **Idea 4:** [Title] - [Description based on actual content analysis] (if applicable)
- **Idea 5:** [Title] - [Description based on actual content analysis] (if applicable)

**REMEMBER: You cannot proceed to analysis or idea generation until you have received the actual content from the tools. Do not make assumptions about what the content might contain.**"""


# Define your tool functions
def process_youtube_video(youtube_url: str):
    """Process a YouTube video and extract transcript and segments"""
    ap = AudioProcessor()
    result = ap.process(youtube_url)
    print(f"\n VIDEO PROCESSED, TRANSCRIPT RECIVED: {result}")
    return result


def process_pdf_document(file_path: str):
    """Process a PDF document and extract text content"""
    result = read_pdf(file_path)
    return result


# Define the function schemas for the model
tools = [
    {
        "name": "process_youtube_video",
        "description": "Extract transcript and segments from a YouTube video",
        "input_schema": {
            "type": "object",
            "properties": {
                "youtube_url": {
                    "type": "string",
                    "description": "The YouTube URL to process (youtube.com or youtu.be format)",
                }
            },
            "required": ["youtube_url"],
        },
    },
    {
        "name": "process_pdf_document",
        "description": "Extract text content from a PDF document",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The file path to the PDF document",
                }
            },
            "required": ["file_path"],
        },
    },
]


def execute_tool(tool_name: str, tool_input: Dict[str, str]) -> str:
    if tool_name == "process_youtube_video":
        return process_youtube_video(tool_input["youtube_url"])

    elif tool_name == "process_pdf_document":
        return process_pdf_document(tool_input["file_path"])

    else:
        return f"Error: Unknown tool {tool_name}"


def process_content_with_tools(user_message: str):

    messages = [{"role": "user", "content": user_message}]

    # Inital call to Claude, decides what tools are needed.
    initial_response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4096,
        system=system_prompt,
        tools=tools,  # Make tools available to Claude
        messages=messages,
    )

    print(f"Claude's stop reason: {initial_response.stop_reason}")

    # Check to see of Claude wants to use tools
    if initial_response.stop_reason == "tool_use":
        print("Claude wants to use tools. Processing tool calls...")

        tool_results = []

        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_name = content_block.name
                tool_input = content_block.input
                tool_use_id = content_block.id

                print(f"\nExecuting tool: {tool_name}")
                print(f"With inputs: {json.dumps(tool_input, indent=2)}")

                # Execute the actual tool
                try:
                    result = execute_tool(tool_name, tool_input)
                    print(f"Tool result: {result[:100]}...")  # Show first 100 chars

                    # Format the result for Claude
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result,
                        }
                    )
                except Exception as e:
                    # Handle tool execution errors
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": f"Error executing tool: {str(e)}",
                            "is_error": True,
                        }
                    )

        # Send tool results back to Claude
        print("Sending tool results back to Claude for final response...")

        messages.append({"role": "assistant", "content": initial_response.content})

        messages.append({"role": "user", "content": tool_results})

        final_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4096,
            system=system_prompt,
            tools=tools,
            messages=messages,
        )

        return final_response.content[0].text

    else:
        print("No tools needed\n")
        return initial_response.content[0].text


def main():
    input = "https://youtu.be/PduJ0P6r_8o"
    result = process_content_with_tools(input)
    print(result)


if __name__ == "__main__":
    main()
