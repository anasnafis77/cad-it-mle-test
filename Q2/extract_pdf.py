import PyPDF2
import re
import pandas as pd

pdf_path = 'Q2/Q2.pdf'
# creating a pdf file object
pdfFileObject = open(pdf_path, 'rb')

pdfReader = PyPDF2.PdfReader(pdfFileObject)

desc_list = []
cause_list = []
page_list = []
for i in range(0, len(pdfReader.pages)):
    # creating a page object
    pageObj = pdfReader.pages[i]
    # extracting text from page
    target = []
    text = pageObj.extract_text().lower()
    
    desc_pat = '(?<=description)(.*)(?=\n)'
    prc_pat = '(?<=\d\.\s)(.*)(?=\.)'
    page_pat = '(?<=\n)\d*(?=[a-zA-Z])'
    desc = re.findall(desc_pat, text)[0].strip(' ')
    cause = '. '.join(re.findall(prc_pat, text))
    page = re.findall(page_pat, text)[0]
    desc_list.append(desc.replace(',', ' '))
    cause_list.append(cause.replace(',', ' '))
    page_list.append(page)

output = pd.DataFrame({'Description': desc_list, 'Possible Root Cause': cause_list, 'Page': page_list})
output.to_csv('Q2/extract_result.csv', index=False)