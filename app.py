from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from io import StringIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import re

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommandRequest(BaseModel):
    csv_data: str
    instruction: str


from openai import OpenAI
from dotenv import load_dotenv
import os
import pandas as pd
from io import StringIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommandRequest(BaseModel):
    csv_data: str
    instruction: str


@app.post("/process-command/")
async def process_command(request: CommandRequest):
    try:
        # Load the CSV data into a DataFrame
        df = pd.read_csv(StringIO(request.csv_data))

        # Extract column names
        column_names = df.columns.tolist()
        column_names_str = ", ".join(column_names)
        print('COLUMN NAMES', column_names_str)
        # Create the dynamic system message with the column names
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that converts instructions into pandas commands. The column names in the dataset are: {column_names_str}."
                },
                {
                    "role": "user",
                    "content": f"Converta a seguinte instrução em português para um comando pandas. Me dê somente o comando pandas e mais nada.\n\nInstrução: {request.instruction}\n\nComando pandas:"
                }
            ],
        )

        generated_command = completion.choices[0].message.content.strip()
        match = re.search(r"```python\n(.*?)```", generated_command, re.S)
        if match:
            python_command = match.group(1).strip()
        else:
            raise HTTPException(status_code=400, detail="No valid Python command found in the generated response.")
        print('GENERATED COMMAND', python_command)
        # Execute the generated command safely
        try:
            compile(python_command, '<string>', 'exec')
        except SyntaxError as e:
            raise HTTPException(status_code=400, detail=f"Invalid syntax: {e}")

        local_vars = {"df": df}
        exec(f"result = {python_command}", {}, local_vars)

        result_df = local_vars.get("result")
        if result_df is None or not isinstance(result_df, pd.DataFrame):
            raise HTTPException(
                status_code=400, detail="Generated command did not produce a valid DataFrame."
            )

        return {"data": result_df.to_json(orient="split")}

    except Exception as e:
        print("Error processing request:", e)
        raise HTTPException(status_code=400, detail=f"Failed to execute command: {str(e)}")
