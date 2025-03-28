1. Tested with the annual reports of the following 3 companies:
   RIL (https://www.ril.com/ar2023-24/index.html)  
   TCS (https://www.tcs.com/content/dam/tcs/investor-relations/financial-statements/2023-24/ar/annual-report-2023-2024.pdf)  
   ONGC (https://ongcindia.com/web/eng/investors/annual-reports)

Please find the screenshots of the tests performed in the file App_Screenprints.pdf

To run the app locally, you need to have Python installed on your machine.
1. Install dependencies
Install chromadb langchain langchain-community pdfplumber PyPDFLoader flask pydantic using the command
pip install <dependency-name>
```
pip install flask
pip install pandas
pip install pdfplumber
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install langchain-core
pip install chromadb
```

2. Unzip the uploaded Python project genai-financial-analysis.zip.
3. Run the app (on a command prompt run the following command from the project folder)
```
python app.py
```


3. Access the application at http://localhost:5000/