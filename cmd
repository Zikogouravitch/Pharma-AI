
Invoke-WebRequest -Uri "https://github.com/tesseract-ocr/tessdata/raw/main/fra.traineddata" -OutFile "C:\Program Files\Tesseract-OCR\tessdata\fra.traineddata"
py -m uvicorn main:app --reload
pip install python-multipart