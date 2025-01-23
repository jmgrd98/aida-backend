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

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:3001",
    "https://aida-data.vercel.app",
    "https://ezydata.vercel.app"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.info(f"CORS origins allowed: {origins}")

class CommandRequest(BaseModel):
    csv_data: str
    instruction: str

@app.post("/process-command/")
async def process_command(request: CommandRequest):
    try:
        df = pd.read_csv(StringIO(request.csv_data))
        column_names = df.columns.tolist()
        column_names_str = ", ".join(column_names)
        
        system_message = f"You are a helpful assistant that converts instructions into pandas commands or generates graphs using matplotlib/seaborn. The column names in the dataset are: {column_names_str}."
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Analyze this dataset and generate either a pandas command or a graph based on the following instruction. Give me only the code and nothing else. {request.instruction}"}
            ],
        )
        
        # Extract the generated command
        generated_response = completion.choices[0].message.content.strip()
        
        # Check if the response contains graph instructions
        if "matplotlib" in generated_response or "seaborn" in generated_response:
            match = re.search(r"```python\n(.*?)```", generated_response, re.S)
            if match:
                graph_command = match.group(1).strip()
            else:
                raise HTTPException(status_code=400, detail="No valid Python graph command found.")
            
            # Execute the graph command
            local_vars = {"df": df, "plt": plt, "sns": sns}
            exec(graph_command, {}, local_vars)
            
            # Save the plot to a temporary file
            with NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                plt.savefig(tmp_file.name, format="png")
                plt.close()
                tmp_file.seek(0)
                graph_data = base64.b64encode(tmp_file.read()).decode("utf-8")
            
            return {"type": "graph", "graph": graph_data}

        else:
            # Handle table operations (Pandas commands)
            match = re.search(r"```python\n(.*?)```", generated_response, re.S)
            if match:
                pandas_command = match.group(1).strip()
            else:
                raise HTTPException(status_code=400, detail="No valid Python command found.")
            
            # Execute the Pandas command
            local_vars = {"df": df}
            exec(f"result = {pandas_command}", {}, local_vars)
            
            result_df = local_vars.get("result")
            if result_df is None or not isinstance(result_df, pd.DataFrame):
                raise HTTPException(status_code=400, detail="Generated command did not produce a valid DataFrame.")
            
            return {"type": "table", "data": result_df.to_json(orient="split")}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to execute command: {str(e)}")
