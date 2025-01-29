from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from io import StringIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from typing import Optional
from tempfile import NamedTemporaryFile
import anthropic
import google.generativeai as genai

load_dotenv()
open_api_key = os.getenv("OPENAI_API_KEY")
claude_api_key = os.getenv("CLAUDE_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

openaiClient = OpenAI(api_key=open_api_key)
claudeClient = anthropic.Anthropic(api_key=claude_api_key)
genai.configure(api_key=gemini_api_key)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:3001",
    "https://ezydata.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommandRequest(BaseModel):
    csv_data: str
    instruction: str
    model: Optional[str] = "openai"

def get_openai_response(prompt: str) -> str:
    try:
        response = openaiClient.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail="OpenAI API request failed")

def get_claude_response(prompt: str) -> str:
    try:
        response = claudeClient.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text.strip()
    except Exception as e:
        logging.error(f"Claude API error: {e}")
        raise HTTPException(status_code=500, detail="Claude API request failed")

def get_gemini_response(prompt: str) -> str:
    print('ENTROU GEMINI')
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        raise HTTPException(status_code=500, detail="Gemini API request failed")

@app.post("/process-command/")
async def process_command(request: CommandRequest):
    try:
        df = pd.read_csv(StringIO(request.csv_data))
        column_names_str = ", ".join(df.columns.tolist())

        prompt = (f"You are an expert in pandas. Convert the following instruction into a pandas command. "
                  f"The column names in the dataset are: {column_names_str}.\n\n"
                  f"Instruction: {request.instruction}")

        if request.model == "claude":
            generated_command = get_claude_response(prompt)
        elif request.model == "gemini":
            generated_command = get_gemini_response(prompt)
        else:
            generated_command = get_openai_response(prompt)

        match = re.search(r"```python\n(.*?)```", generated_command, re.S)
        if match:
            generated_command = match.group(1).strip()

        local_vars = {"df": df}
        exec(f"result = {generated_command}", {}, local_vars)
        result_df = local_vars.get("result")

        if not isinstance(result_df, pd.DataFrame):
            raise HTTPException(status_code=400, detail="Generated command did not produce a valid DataFrame.")

        return {"type": "table", "data": result_df.to_json(orient="split")}
    except Exception as e:
        logging.error(f"Error processing command: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to execute command: {str(e)}")

@app.post("/generate-chart/")
async def generate_chart(request: CommandRequest):
    try:
        df = pd.read_csv(StringIO(request.csv_data))
        column_names_str = ", ".join(df.columns.tolist())

        prompt = (f"You are an expert in Python data visualization. Generate a valid matplotlib/seaborn command "
                  f"to create a graph based on the instruction below. The column names are: {column_names_str}.\n\n"
                  f"Instruction: {request.instruction}")

        if request.model == "claude":
            generated_response = get_claude_response(prompt)
        elif request.model == "gemini":
            generated_response = get_gemini_response(prompt)
        else:
            generated_response = get_openai_response(prompt)

        match = re.search(r"```(?:python)?\n(.*?)```", generated_response, re.S)
        if match:
            graph_command = match.group(1).strip()
        else:
            raise HTTPException(status_code=400, detail="No valid code block found in response")

        if not any(x in graph_command for x in ['plt.show()', 'plt.savefig', 'sns.']):
            raise HTTPException(status_code=400, detail="Generated code doesn't contain valid plotting commands")

        local_vars = {"df": df, "plt": plt, "sns": sns}
        exec(graph_command, {}, local_vars)
        
        with NamedTemporaryFile(suffix=".png") as tmp_file:
            plt.savefig(tmp_file.name, bbox_inches='tight')
            plt.clf()
            tmp_file.seek(0)
            graph_data = base64.b64encode(tmp_file.read()).decode("utf-8")

        return {"type": "graph", "graph": graph_data}
    except Exception as e:
        logging.error(f"Chart error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Chart generation failed: {str(e)}")
