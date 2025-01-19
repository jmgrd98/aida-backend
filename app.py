import openai
from dotenv import load_dotenv
import os
import pandas as pd
from io import StringIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

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
        print("Received CSV Data:", request.csv_data)
        print("Received Instruction:", request.instruction)

        # Use the new method for OpenAI's API
        openai_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Use the appropriate model
            messages=[{
                "role": "system", 
                "content": "You are a helpful assistant that converts instructions into pandas commands."
            }, {
                "role": "user", 
                "content": f"Converta a seguinte instrução em português para um comando pandas:\n\nInstrução: {request.instruction}\n\nComando pandas:"
            }],
            max_tokens=150,
            temperature=0
        )

        generated_command = openai_response['choices'][0]['message']['content'].strip()
        print("Generated Command:", generated_command)

        df = pd.read_csv(StringIO(request.csv_data))
        local_vars = {"df": df}

        exec(generated_command, {}, local_vars)

        updated_df = local_vars["df"]
        return {"data": updated_df.to_json(orient="split")}
    except Exception as e:
        print("Error processing request:", e)
        raise HTTPException(status_code=400, detail=f"Failed to execute command: {str(e)}")